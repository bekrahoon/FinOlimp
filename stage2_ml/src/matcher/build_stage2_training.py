"""Create run-local silver training tables for Stage 2 from this run's inputs.

The pair targets reproduce Stage 1's mechanical grouping/matching decisions;
status targets mirror Stage 1 row admission. These are training labels, not
independent ground truth. Source labels are prepared separately from the
completed Stage 1 discrepancy report after a bootstrap Stage 2 pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
FINOLIMP = HERE.parents[2]
S1 = FINOLIMP / "stage1_rules"
sys.path.insert(0, str(S1 / "src"))
sys.path.insert(0, str(HERE))

import parse as R  # noqa: E402
import evaluate_s3 as Eval  # noqa: E402
from s3_row_features import (STATUS_FEATURES, act_status_features,
                             fit_status_model)  # noqa: E402
from payment_pairing import match_payments_with_pairs  # noqa: E402
from reconcile_s3 import _parse_raw_dates  # noqa: E402
from temporal_split import (date_masks, pair_masks, select_cutoff,
                             select_month_cutoff)  # noqa: E402

SEED = 2026
MAX_NEGATIVES_PER_KIND = 3000
DEFAULT_VALIDATION_FRACTION = 0.2


def _status_validation_metrics(model, features: pd.DataFrame,
                               labels: pd.Series) -> dict:
    if not len(labels):
        return {"n": 0, "accuracy": None, "keep_precision": None,
                "keep_recall": None}
    classes = list(model.classes_)
    probabilities = model.predict_proba(features.values)
    keep_probability = (probabilities[:, classes.index(1)] if 1 in classes
                        else np.zeros(len(labels), dtype=float))
    predicted = keep_probability >= 0.5
    actual = labels.to_numpy(dtype=bool)
    return {
        "n": len(labels),
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "keep_precision": round(float(precision_score(
            actual, predicted, zero_division=0)), 6),
        "keep_recall": round(float(recall_score(actual, predicted, zero_division=0)), 6),
        "threshold": 0.5,
        "target_provenance": "Stage 1 row admission; silver, not independent truth",
    }


def _pair_counts(table: pd.DataFrame) -> dict:
    return {
        "rows": len(table),
        "positive_rows": int(table["label_silver"].eq(1).sum()),
        "negative_rows": int(table["label_silver"].eq(0).sum()),
    }


def _snapshot_pair(kind: str, label: int, ai: int, ei: int,
                   acts: pd.DataFrame, etm: pd.DataFrame) -> dict:
    a = acts.loc[ai]
    e = etm.loc[ei]
    return {
        "kind": kind,
        "label_silver": int(label),
        "a_idx": int(ai),
        "e_idx": int(ei),
        "a_amount": float(a["sum_1c"]),
        "a_date": str(a["date_norm"]),
        "a_tickets": "|".join(sorted(a["ticket_set"])),
        "a_text": str(a.get("doc", "") or ""),
        "a_agent": str(a.get("folder", "") or ""),
        "e_amount": float(e["sum_etm"]),
        "e_date": str(e["date_norm"]),
        "e_tickets": "|".join(sorted(e["ticket_set"])),
        "e_text": str(e.get("comment", "") or ""),
        "e_agent": str(e.get("agent", "") or ""),
        "e_kind": str(e.get("kind", "") or ""),
    }


def _build_pair_table(acts: pd.DataFrame, etm: pd.DataFrame,
                      cfg: dict) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add(kind: str, label: int, ai: int, ei: int) -> bool:
        pair = (int(ai), int(ei))
        if pair in seen:
            return False
        seen.add(pair)
        rows.append(_snapshot_pair(kind, label, pair[0], pair[1], acts, etm))
        return True

    agents = sorted(set(acts["agent_norm"]) | set(etm["agent_norm"]))
    component_count = 0
    mismatch_components = 0
    for agent in agents:
        aa = acts[(acts["agent_norm"] == agent) & acts["ticket_cell"].notna()]
        ee = etm[(etm["agent_norm"] == agent) & etm["tickets"].notna()]
        for act_ids, etm_ids in R.ticket_components(aa, ee):
            if not act_ids or not etm_ids:
                continue
            component_count += 1
            sum_a = round(float(acts.loc[act_ids, "sum_1c"].sum()), 2)
            sum_e = round(float(etm.loc[etm_ids, "sum_etm"].sum()), 2)
            if abs(sum_a - sum_e) < cfg["tolerance"]:
                for ai in act_ids:
                    for ei in etm_ids:
                        add("ticket_clean", 1, int(ai), int(ei))
            elif len(act_ids) == 1 and len(etm_ids) == 1:
                # reconcile_s3 treats a shared-tail edge as component membership,
                # even when the amounts disagree.
                mismatch_components += 1
                add("neg_same_ticket_diff_sum", 0, int(act_ids[0]), int(etm_ids[0]))

    payment_matches = match_payments_with_pairs(acts, etm, cfg)
    for ai, ei, kind in payment_matches:
        add(str(kind), 1, int(ai), int(ei))
    matched = {(int(ai), int(ei)) for ai, ei, _ in payment_matches}

    pay_acts = acts[acts["ticket_cell"].isna()]
    pay_etm = etm[etm["tickets"].isna()]
    negative_counts: dict[str, int] = defaultdict(int)

    # Same-agent, same-amount but unmatched payments are useful hard negatives.
    for agent in agents:
        ga = pay_acts[pay_acts["agent_norm"] == agent]
        ge = pay_etm[pay_etm["agent_norm"] == agent]
        for ai, ar in ga.iterrows():
            candidates = ge[(ge["sum_etm"] - float(ar["sum_1c"])).abs() < cfg["tolerance"]]
            for ei in candidates.index:
                pair = (int(ai), int(ei))
                if pair not in matched and negative_counts["neg_same_amt"] < MAX_NEGATIVES_PER_KIND:
                    negative_counts["neg_same_amt"] += add("neg_same_amt", 0, *pair)
                if negative_counts["neg_same_amt"] >= MAX_NEGATIVES_PER_KIND:
                    break
            if negative_counts["neg_same_amt"] >= MAX_NEGATIVES_PER_KIND:
                break
        if negative_counts["neg_same_amt"] >= MAX_NEGATIVES_PER_KIND:
            break

    # Near-amount payment negatives; keep the sample bounded and deterministic.
    for agent in agents:
        ga = pay_acts[pay_acts["agent_norm"] == agent]
        ge = pay_etm[pay_etm["agent_norm"] == agent]
        if ga.empty or ge.empty:
            continue
        sampled = ga.sample(min(len(ga), 40), random_state=SEED)
        for ai, ar in sampled.iterrows():
            amount = float(ar["sum_1c"])
            relative = (ge["sum_etm"] - amount).abs() / max(abs(amount), 1.0)
            for ei in ge.index[(relative >= 0.01) & (relative < 0.05)][:3]:
                pair = (int(ai), int(ei))
                if pair not in matched and negative_counts["neg_near_amt"] < MAX_NEGATIVES_PER_KIND:
                    negative_counts["neg_near_amt"] += add("neg_near_amt", 0, *pair)
            if negative_counts["neg_near_amt"] >= MAX_NEGATIVES_PER_KIND:
                break
        if negative_counts["neg_near_amt"] >= MAX_NEGATIVES_PER_KIND:
            break

    # Cross-agent ticket-tail lookalikes are negative links for the ticket model.
    ticket_index: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for ai, row in acts[acts["ticket_cell"].notna()].iterrows():
        for tail in row["ticket_set"]:
            ticket_index[tail].add(("a", int(ai)))
    for ei, row in etm[etm["tickets"].notna()].iterrows():
        for tail in row["ticket_set"]:
            ticket_index[tail].add(("e", int(ei)))
    cross_ticket_pairs: set[tuple[int, int]] = set()
    for tail in sorted(ticket_index):
        members = ticket_index[tail]
        act_ids = sorted(idx for side, idx in members if side == "a")
        etm_ids = sorted(idx for side, idx in members if side == "e")
        for ai in act_ids:
            for ei in etm_ids:
                if acts.loc[ai, "agent_norm"] != etm.loc[ei, "agent_norm"]:
                    cross_ticket_pairs.add((ai, ei))
                    if len(cross_ticket_pairs) >= MAX_NEGATIVES_PER_KIND:
                        break
            if len(cross_ticket_pairs) >= MAX_NEGATIVES_PER_KIND:
                break
        if len(cross_ticket_pairs) >= MAX_NEGATIVES_PER_KIND:
            break
    for ai, ei in sorted(cross_ticket_pairs):
        if add("neg_cross_agent_ticket", 0, ai, ei):
            negative_counts["neg_cross_agent_ticket"] += 1

    # Cross-agent negatives across the population, with a finite retry budget.
    rng = np.random.default_rng(SEED)
    all_a = acts.index.to_numpy()
    all_e = etm.index.to_numpy()
    target = min(MAX_NEGATIVES_PER_KIND, max(len(rows) // 2, 100))
    attempts = 0
    max_attempts = target * 50
    while (negative_counts["neg_random"] < target and len(all_a) and len(all_e)
           and len(agents) > 1 and attempts < max_attempts):
        attempts += 1
        ai = int(rng.choice(all_a))
        ei = int(rng.choice(all_e))
        if acts.loc[ai, "agent_norm"] == etm.loc[ei, "agent_norm"]:
            continue
        if add("neg_random", 0, ai, ei):
            negative_counts["neg_random"] += 1

    table = pd.DataFrame(rows, columns=[
        "kind", "label_silver", "a_idx", "e_idx", "a_amount", "a_date",
        "a_tickets", "a_text", "a_agent", "e_amount", "e_date",
        "e_tickets", "e_text", "e_agent", "e_kind",
    ])
    report = {
        "pair_rows": len(table),
        "positive_rows": int((table["label_silver"] == 1).sum()) if len(table) else 0,
        "negative_rows": int((table["label_silver"] == 0).sum()) if len(table) else 0,
        "ticket_components": component_count,
        "ticket_sum_mismatch_components": mismatch_components,
        "payment_positive_pairs": len(payment_matches),
        "negative_rows_by_kind": dict(negative_counts),
        "label_provenance": "current-input Stage 1 grouping/matching silver targets",
    }
    return table, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--stage1-reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float,
                        default=DEFAULT_VALIDATION_FRACTION,
                        help="Fraction of latest distinct operation dates reserved for validation.")
    parser.add_argument("--validation-months", type=int, default=None,
                        help="Reserve the last N full calendar months for validation "
                             "(month-aligned cutoff on the 1st); overrides --validation-fraction.")
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    reports_dir = args.stage1_reports.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    discrepancy_path = reports_dir / "discrepancies.csv"
    if not discrepancy_path.is_file():
        raise FileNotFoundError(f"Stage 1 report not found: {discrepancy_path}")

    config_path = input_dir / "reconcile.yaml"
    cfg = R.load_config(config_path if config_path.is_file() else None)
    qlog = R.QualityLog()
    acts, etm, _, _ = R.load_data(
        input_dir / "acts.csv", input_dir / "etm.csv",
        input_dir / "registry.csv", cfg, qlog)
    raw_acts = R.strip_string_cells(R.read_csv_smart(input_dir / "acts.csv", cfg)[0])
    raw_etm = R.strip_string_cells(R.read_csv_smart(input_dir / "etm.csv", cfg)[0])
    raw_act_dates = _parse_raw_dates(raw_acts["date"])
    raw_etm_dates = _parse_raw_dates(raw_etm["date"])
    all_dates = pd.concat([raw_act_dates, raw_etm_dates], ignore_index=True)
    if args.validation_months is not None:
        cutoff = select_month_cutoff(all_dates, args.validation_months)
        split_method = ("calendar-month split: last "
                        f"{int(args.validation_months)} full months are validation")
    else:
        cutoff = select_cutoff(all_dates, args.validation_fraction)
        split_method = "chronological split by unique calendar operation dates"
    engine_to_raw, _ = Eval._raw_row_maps(raw_acts, raw_etm, acts, etm)
    retained_raw_ids = set(engine_to_raw.values())

    status_features = act_status_features(raw_acts)
    status_labels = np.array(
        [int(idx in retained_raw_ids) for idx in raw_acts.index], dtype=int)
    status_table = status_features.copy()
    status_table.insert(0, "raw_idx", raw_acts.index.astype(int))
    status_table["operation_date"] = raw_act_dates.dt.strftime("%Y-%m-%d")
    status_table["label_keep"] = status_labels

    pair_table, pair_report = _build_pair_table(acts, etm, cfg)
    status_train_mask, status_validation_mask, status_unknown_mask = date_masks(
        status_table["operation_date"], cutoff)
    pair_train_mask, pair_validation_mask, pair_excluded_mask = pair_masks(
        pair_table, cutoff)
    status_train = status_table.loc[status_train_mask].copy()
    status_validation = status_table.loc[status_validation_mask].copy()
    pair_train = pair_table.loc[pair_train_mask].copy()
    pair_validation = pair_table.loc[pair_validation_mask].copy()

    if status_train.empty:
        raise ValueError("Chronological status training fold is empty")
    status_model = fit_status_model(
        status_train[STATUS_FEATURES].values, status_train["label_keep"].values)
    status_validation_metrics = _status_validation_metrics(
        status_model, status_validation[STATUS_FEATURES],
        status_validation["label_keep"])

    output_dir.mkdir(parents=True, exist_ok=True)
    status_train.to_csv(output_dir / "act_status_train.csv", index=False)
    status_validation.to_csv(output_dir / "act_status_validation.csv", index=False)
    pair_train.to_csv(output_dir / "pair_train.csv", index=False)
    pair_validation.to_csv(output_dir / "pair_validation.csv", index=False)

    status_report = {
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "validation_fraction_requested": args.validation_fraction,
        "validation_months_requested": args.validation_months,
        "n_rows": len(status_table),
        "n_keep": int(status_labels.sum()),
        "n_drop": int((status_labels == 0).sum()),
        "n_train": len(status_train),
        "n_validation": len(status_validation),
        "n_unknown_date_excluded": int(status_unknown_mask.sum()),
        "features": STATUS_FEATURES,
        "label_provenance": "current-input Stage 1 row admission; silver target",
        "validation_vs_silver": status_validation_metrics,
    }
    (output_dir / "act_status_train_report.json").write_text(
        json.dumps(status_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pair_report["temporal_split"] = {
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "train": _pair_counts(pair_train),
        "validation": _pair_counts(pair_validation),
        "cross_period_or_unknown_date_excluded": int(pair_excluded_mask.sum()),
        "validation_vs_silver_only": True,
    }
    (output_dir / "tickets_pairs_report.json").write_text(
        json.dumps(pair_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporal_split_report = {
        "method": split_method,
        "validation_fraction_requested": args.validation_fraction,
        "validation_months_requested": args.validation_months,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "train_dates": f"before {cutoff:%Y-%m-%d}",
        "validation_dates": f"on/after {cutoff:%Y-%m-%d}",
        "date_population": "valid dates from current acts.csv and etm.csv",
        "unique_dates": int(pd.concat([raw_act_dates, raw_etm_dates])
                             .dropna().dt.normalize().nunique()),
        "status_rows": {
            "train": len(status_train), "validation": len(status_validation),
            "unknown_date_excluded": int(status_unknown_mask.sum()),
        },
        "pair_rows": {
            "train": len(pair_train), "validation": len(pair_validation),
            "cross_period_or_unknown_date_excluded": int(pair_excluded_mask.sum()),
        },
        "label_quality": "Stage 1-derived silver targets; validation is not independent truth",
        "feature_boundary": (
            "status group context uses only prior operation dates; pair rows crossing the "
            "cutoff are excluded from both folds."),
    }
    combined = {
        "training_scope": "per-run; chronological train/validation from current input and Stage 1 results",
        "label_quality": "silver; derived from Stage 1 rules, not independent truth",
        "independent_evaluation": False,
        "temporal_validation": True,
        "temporal_split": temporal_split_report,
        "status": status_report,
        "pairs": pair_report,
        "stage1_discrepancy_rows": int(len(pd.read_csv(discrepancy_path))),
        "artifacts": ["act_status_train.csv", "act_status_validation.csv",
                      "pair_train.csv", "pair_validation.csv"],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
