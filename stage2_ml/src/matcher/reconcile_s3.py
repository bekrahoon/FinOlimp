"""Stage 2: raw-input ML reconciliation, with no Stage 1 rules at inference.

Inference reads raw acts/ETM CSVs, an explicit run-local temporal split, and pure
Stage 2 features. Candidate generation is deliberately permissive:
ticket rows block on an exact shared 10-digit tail; payment rows block on
relative amount <=5% and date gap <=180d, with exact document-token pairs added
as extra candidates. These are compute-saving candidate gates; ML scores and
acceptance decisions are downstream.

Output: run-local ML discrepancies plus candidate/match/cluster artifacts.
Reference comparisons are isolated in evaluate_s3.py.
"""
from __future__ import annotations

import ast
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
FINOLIMP = HERE.parents[2]
S1IN = FINOLIMP / "stage1_rules" / "input"
S1SRC = FINOLIMP / "stage1_rules" / "src"
sys.path.insert(0, str(S1SRC))
import io_clean as C  # noqa: E402 (shared mechanical plumbing)
from s2_features import pay_features, ticket_features, union_find  # noqa: E402
from s3_row_features import (STATUS_FEATURES, act_status_features,
                             fit_status_model)  # noqa: E402
from source_features_s3 import SOURCE_FEATURES, build_source_features  # noqa: E402
from source_s3_model import SOURCE_CLASSES, fit_source_model  # noqa: E402
from source_s3_model import _metrics as source_metrics  # noqa: E402
from temporal_split import pair_masks  # noqa: E402

SEED = 7
THRESHOLD = 0.5
PAY_RELATIVE_CANDIDATE_MAX = 0.05
PAY_DATE_CANDIDATE_DAYS = 180
MONEY_SEPARATORS = [" ", "\u00a0", "'", "’"]
DOC_REF = re.compile(r"ЦБ-С\d+")


def _purity_check() -> None:
    """Guard this inference entry point against Stage 1 rule imports/calls."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    banned_modules = {"parse", "reconcile2", "pairs"}
    banned_names = {"load_data", "ticket_components", "canonical_labels",
                    "apply_aliases", "norm_agent", "resolve_registry_party",
                    "match_payments", "match_payments_with_pairs",
                    "source_for_comment"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names
                         if alias.name.split(".")[0] in banned_modules)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in banned_modules:
                found.append(node.module)
            found.extend(alias.name for alias in node.names
                         if alias.name in banned_names)
        elif isinstance(node, ast.Name) and node.id in banned_names:
            found.append(node.id)
    assert not found, f"Stage 1 rule dependency in inference: {sorted(set(found))}"


def _read_csv(path: Path) -> pd.DataFrame:
    read_cfg = {"input": {"encodings": ["utf-8-sig", "utf-8", "cp1251"]}}
    frame, _ = C.read_csv_smart(path, read_cfg)
    return C.strip_string_cells(frame)


def _money(series: pd.Series) -> pd.Series:
    return series.map(lambda v: C.parse_money_value(v, MONEY_SEPARATORS)).astype(float)


def _date_string(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S")


def _parse_raw_dates(series: pd.Series) -> pd.Series:
    """Parse known raw date formats in priority order; ISO must not be day-first."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.notna()
    for date_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                        "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
                        "%d/%m/%Y", "%Y/%m/%d"):
        if not remaining.any():
            break
        value = pd.to_datetime(series[remaining], format=date_format, errors="coerce")
        hit = value.notna()
        parsed.loc[value[hit].index] = value[hit]
        remaining = remaining & parsed.isna()
    if remaining.any():
        value = pd.to_datetime(series[remaining], errors="coerce", dayfirst=True)
        parsed.loc[value[value.notna()].index] = value[value.notna()]
    return parsed


def load_raw_inputs(acts_path: Path, etm_path: Path
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source rows mechanically; preserve their original CSV row indices."""
    acts = _read_csv(acts_path)
    etm = _read_csv(etm_path)
    required_a = {"folder", "date", "doc", "ticket_cell", "debet", "credit"}
    required_e = {"agent", "date", "txn_id", "kind", "tickets", "amount_kgs",
                  "comment"}
    if missing := required_a - set(acts.columns):
        raise ValueError(f"acts.csv missing raw columns: {sorted(missing)}")
    if missing := required_e - set(etm.columns):
        raise ValueError(f"etm.csv missing raw columns: {sorted(missing)}")
    acts = acts.copy()
    etm = etm.copy()
    acts["raw_idx"] = acts.index.astype(int)
    etm["raw_idx"] = etm.index.astype(int)
    acts["amount"] = (_money(acts["debet"]).fillna(0)
                       - _money(acts["credit"]).fillna(0)).round(2)
    raw_etm_amount = _money(etm["amount_kgs"])
    if raw_etm_amount.isna().any():
        raise ValueError("Stage 2 raw inference requires populated ETM amount_kgs")
    # ETM's raw credit-positive sign is inverted to the model's 1C sign.
    etm["amount"] = -raw_etm_amount.round(2)
    acts["date_norm"] = _parse_raw_dates(acts["date"])
    etm["date_norm"] = _parse_raw_dates(etm["date"])
    acts["tickets_norm"] = acts["ticket_cell"].map(
        lambda value: "|".join(sorted(C.ticket_numbers(value, None))))
    etm["tickets_norm"] = etm["tickets"].map(
        lambda value: "|".join(sorted(C.ticket_numbers(value, None))))
    return acts, etm


def _pair_frame(acts: pd.DataFrame, etm: pd.DataFrame,
                pairs: list[tuple[int, int]]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["a_idx", "e_idx", "a_tickets", "e_tickets",
                                     "a_amount", "e_amount", "a_date", "e_date",
                                     "a_agent", "e_agent", "a_text", "e_text", "e_kind"])
    a_idx = [int(a) for a, _ in pairs]
    e_idx = [int(e) for _, e in pairs]
    arows = acts.loc[a_idx].reset_index(drop=True)
    erows = etm.loc[e_idx].reset_index(drop=True)
    return pd.DataFrame({
        "a_idx": a_idx,
        "e_idx": e_idx,
        "a_tickets": arows["tickets_norm"].fillna(""),
        "e_tickets": erows["tickets_norm"].fillna(""),
        "a_amount": arows["amount"].to_numpy(),
        "e_amount": erows["amount"].to_numpy(),
        "a_date": _date_string(arows["date_norm"]),
        "e_date": _date_string(erows["date_norm"]),
        "a_agent": arows["folder"].fillna("").astype(str),
        "e_agent": erows["agent"].fillna("").astype(str),
        "a_text": arows["doc"].fillna("").astype(str),
        "e_text": erows["comment"].fillna("").astype(str),
        "e_kind": erows["kind"].fillna("").astype(str),
    })


def ticket_candidate_pairs(acts: pd.DataFrame, etm: pd.DataFrame
                           ) -> tuple[list[tuple[int, int]], list[int]]:
    by_tail_a: dict[str, list[int]] = defaultdict(list)
    by_tail_e: dict[str, list[int]] = defaultdict(list)
    for idx, value in acts.loc[acts["ticket_cell"].notna(), "ticket_cell"].items():
        for tail in C.ticket_numbers(value, None):
            by_tail_a[tail].append(int(idx))
    for idx, value in etm.loc[etm["tickets"].notna(), "tickets"].items():
        for tail in C.ticket_numbers(value, None):
            by_tail_e[tail].append(int(idx))
    pairs: set[tuple[int, int]] = set()
    overlap_count: dict[tuple[int, int], int] = defaultdict(int)
    for tail in sorted(set(by_tail_a).intersection(by_tail_e)):
        for a_idx in by_tail_a[tail]:
            for e_idx in by_tail_e[tail]:
                key = (a_idx, e_idx)
                pairs.add(key)
                overlap_count[key] += 1
    ordered = sorted(pairs)
    return ordered, [overlap_count[pair] for pair in ordered]


def same_side_ticket_candidate_pairs(rows: pd.DataFrame, side: str
                                     ) -> tuple[list[tuple[int, int]], list[int]]:
    by_tail: dict[str, list[int]] = defaultdict(list)
    for idx, value in rows["tickets_norm"].items():
        for tail in str(value or "").split("|"):
            if tail:
                by_tail[tail].append(int(idx))
    overlap_count: dict[tuple[int, int], int] = defaultdict(int)
    for ids in by_tail.values():
        unique_ids = sorted(set(ids))
        for left_pos, left in enumerate(unique_ids):
            for right in unique_ids[left_pos + 1:]:
                overlap_count[(left, right)] += 1
    ordered = sorted(overlap_count)
    return ordered, [overlap_count[pair] for pair in ordered]


def same_side_pair_frame(rows: pd.DataFrame, pairs: list[tuple[int, int]],
                         side: str) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["side", "left_idx", "right_idx", "a_tickets",
                                     "e_tickets", "a_amount", "e_amount", "a_date",
                                     "e_date", "a_agent", "e_agent", "a_text",
                                     "e_text", "e_kind"])
    left_ids, right_ids = zip(*pairs)
    left, right = rows.loc[list(left_ids)].reset_index(drop=True), \
        rows.loc[list(right_ids)].reset_index(drop=True)
    agent_col = "folder" if side == "A" else "agent"
    text_col = "doc" if side == "A" else "comment"
    return pd.DataFrame({
        "side": side, "left_idx": left_ids, "right_idx": right_ids,
        "a_tickets": left["tickets_norm"].fillna(""),
        "e_tickets": right["tickets_norm"].fillna(""),
        "a_amount": left["amount"].to_numpy(),
        "e_amount": right["amount"].to_numpy(),
        "a_date": _date_string(left["date_norm"]),
        "e_date": _date_string(right["date_norm"]),
        "a_agent": left[agent_col].fillna("").astype(str),
        "e_agent": right[agent_col].fillna("").astype(str),
        "a_text": left[text_col].fillna("").astype(str),
        "e_text": right[text_col].fillna("").astype(str),
        "e_kind": right["kind"].fillna("").astype(str) if side == "E" else "",
    })


def payment_candidate_pairs(acts: pd.DataFrame, etm: pd.DataFrame
                            ) -> tuple[list[tuple[int, int]], dict[str, int]]:
    pay_a = acts.loc[acts["ticket_cell"].isna()]
    pay_e = etm.loc[etm["tickets"].isna()]
    candidate_reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
    e_amount = pay_e["amount"].to_numpy(dtype=float)
    a_amount = pay_a["amount"].to_numpy(dtype=float)
    e_dates = pay_e["date_norm"].to_numpy(dtype="datetime64[D]")
    e_ids = pay_e.index.to_numpy(dtype=int)
    a_ids = pay_a.index.to_numpy(dtype=int)
    e_amount = np.nan_to_num(e_amount, nan=0.0)
    chunk_size = 128
    amount_date_pairs = 0
    for start in range(0, len(pay_a), chunk_size):
        stop = min(start + chunk_size, len(pay_a))
        amounts = a_amount[start:stop, None]
        denom = np.maximum(np.maximum(np.abs(amounts), np.abs(e_amount[None, :])), 1.0)
        rel = np.abs(amounts - e_amount[None, :]) / denom
        a_dates = pay_a.iloc[start:stop]["date_norm"].to_numpy(dtype="datetime64[D]")
        with np.errstate(invalid="ignore"):
            gap = np.abs(a_dates[:, None] - e_dates[None, :]).astype("timedelta64[D]")
            gap_days = gap.astype(float)
        keep = (rel <= PAY_RELATIVE_CANDIDATE_MAX) \
            & (gap_days <= PAY_DATE_CANDIDATE_DAYS)
        rows, cols = np.nonzero(keep)
        amount_date_pairs += len(rows)
        for row_pos, col_pos in zip(rows, cols):
            candidate_reasons[(int(a_ids[start + row_pos]), int(e_ids[col_pos]))].add(
                "amount_date")

    # Exact document references are another candidate block. The model, not
    # the reference token, still decides whether to accept the pair.
    refs_a: dict[str, list[int]] = defaultdict(list)
    refs_e: dict[str, list[int]] = defaultdict(list)
    for idx, text in pay_a["doc"].items():
        for ref in set(DOC_REF.findall(str(text))):
            refs_a[ref].append(int(idx))
    for idx, text in pay_e["comment"].items():
        for ref in set(DOC_REF.findall(str(text))):
            refs_e[ref].append(int(idx))
    doc_pairs = 0
    for ref in sorted(set(refs_a).intersection(refs_e)):
        for a_idx in refs_a[ref]:
            for e_idx in refs_e[ref]:
                candidate_reasons[(a_idx, e_idx)].add("doc_ref")
                doc_pairs += 1
    pairs = sorted(candidate_reasons)
    reason_counts = {
        "amount_date_candidates": amount_date_pairs,
        "doc_ref_pair_occurrences": doc_pairs,
        "candidate_union": len(pairs),
        "doc_ref_only_candidates": sum(
            reasons == {"doc_ref"} for reasons in candidate_reasons.values()),
    }
    return pairs, reason_counts


def _positive_probability(model, features: pd.DataFrame | np.ndarray,
                          positive_label=1) -> np.ndarray:
    """Return P(positive_label), including constant one-class fallback models."""
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if positive_label not in classes:
        return np.zeros(len(probabilities), dtype=float)
    return probabilities[:, classes.index(positive_label)]


def _fit_pair_model(training: pd.DataFrame, validation: pd.DataFrame,
                    ticket: bool) -> tuple[object, dict]:
    def kind_rows(frame: pd.DataFrame) -> pd.DataFrame:
        has_ticket = frame["e_tickets"].fillna("").astype(str).str.strip().ne("")
        return frame.loc[has_ticket if ticket else ~has_ticket].reset_index(drop=True)

    train_frame, validation_frame = kind_rows(training), kind_rows(validation)
    train_features = (ticket_features(train_frame) if ticket
                      else pay_features(train_frame))
    validation_features = (ticket_features(validation_frame) if ticket
                           else pay_features(validation_frame))
    # pandas 3 may expose read-only arrays; relabel component edges on a copy.
    y_train = train_frame["label_silver"].astype(int).to_numpy(copy=True)
    y_validation = validation_frame["label_silver"].astype(int).to_numpy(copy=True)

    def relabel_component_edges(frame, features, targets):
        if not ticket or frame.empty:
            return 0
        component_edge = frame["kind"].astype(str).eq(
            "neg_same_ticket_diff_sum").to_numpy()
        if component_edge.any() and not features.loc[
                component_edge, "tail_any"].eq(1).all():
            raise AssertionError("sum-discrepancy relabel contains a non-shared ticket pair")
        targets[component_edge] = 1
        return int(component_edge.sum())

    train_relabels = relabel_component_edges(
        train_frame, train_features, y_train)
    validation_relabels = relabel_component_edges(
        validation_frame, validation_features, y_validation)
    if not len(y_train):
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(np.zeros((1, train_features.shape[1]), dtype=float), np.array([0]))
    elif len(np.unique(y_train)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_train[0]))
        model.fit(train_features.values, y_train)
    else:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               max_leaf_nodes=15, random_state=SEED)
        model.fit(train_features.values, y_train)

    validation_report = None
    if len(y_validation):
        probabilities = _positive_probability(model, validation_features.values)
        predicted = probabilities >= THRESHOLD
        actual = y_validation.astype(bool)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        validation_report = {
            "n": len(y_validation),
            "threshold": THRESHOLD,
            "precision_vs_silver_target": round(tp / max(tp + fp, 1), 6),
            "recall_vs_silver_target": round(tp / max(tp + fn, 1), 6),
            "false_positives": fp,
            "false_negatives": fn,
            "target_provenance": "Stage 1-derived silver labels; not independent truth",
        }
    return model, {
        "n_train": len(y_train),
        "n_validation": len(y_validation),
        "n_train_positive": int(y_train.sum()),
        "n_validation_positive": int(y_validation.sum()),
        "component_membership_relabels_train": train_relabels,
        "component_membership_relabels_validation": validation_relabels,
        "fit_scope": "earlier-date train fold only",
        "empty_train_fallback": "constant-negative" if not len(y_train) else None,
        "validation_at_threshold": validation_report,
    }


def _silver_candidate_coverage(validation: pd.DataFrame) -> dict:
    """Measure candidate gates on chronological validation silver positives."""
    hold = validation.reset_index(drop=True)
    pos = hold["label_silver"].astype(int).eq(1)
    ticket = hold[hold["e_tickets"].notna()].copy()
    ticket_pos = ticket["label_silver"].astype(int).eq(1)
    shared_tail = [bool(C.ticket_numbers(a, None) & C.ticket_numbers(e, None))
                   for a, e in zip(ticket["a_tickets"], ticket["e_tickets"])]
    ticket_covered = int((ticket_pos & pd.Series(shared_tail, index=ticket.index)).sum())

    pay = hold[hold["e_tickets"].isna()].copy()
    pay_pos = pay["label_silver"].astype(int).eq(1)
    relative = ((pay["a_amount"] - pay["e_amount"]).abs()
                / np.maximum(np.maximum(pay["a_amount"].abs(),
                                        pay["e_amount"].abs()), 1.0))
    gap = (pd.to_datetime(pay["a_date"], errors="coerce")
           - pd.to_datetime(pay["e_date"], errors="coerce")).abs().dt.days
    pay_covered = int((pay_pos & (relative <= PAY_RELATIVE_CANDIDATE_MAX)
                       & (gap <= PAY_DATE_CANDIDATE_DAYS)).sum())
    return {
        "validation_silver_positive_pairs": int(pos.sum()),
        "ticket_positive_pairs": int(ticket_pos.sum()),
        "ticket_exact_tail_gate_covered": ticket_covered,
        "ticket_exact_tail_gate_recall_silver": round(
            ticket_covered / max(int(ticket_pos.sum()), 1), 6),
        "payment_positive_pairs": int(pay_pos.sum()),
        "payment_amount_date_gate_covered": pay_covered,
        "payment_amount_date_gate_recall_silver": round(
            pay_covered / max(int(pay_pos.sum()), 1), 6),
        "note": "Temporal validation coverage vs silver labels, not true-match recall.",
    }


def _ticket_clusters(acts: pd.DataFrame, etm: pd.DataFrame,
                     candidate_frame: pd.DataFrame,
                     same_side_frame: pd.DataFrame
                     ) -> tuple[pd.DataFrame, list[dict]]:
    act_ids = acts.index[acts["ticket_cell"].notna()].astype(int).tolist()
    etm_ids = etm.index[etm["tickets"].notna()].astype(int).tolist()
    nodes = [("A", idx) for idx in act_ids] + [("E", idx) for idx in etm_ids]
    node_no = {node: i for i, node in enumerate(nodes)}
    accepted = candidate_frame[candidate_frame["accepted"]]
    edges = [(node_no[("A", int(r.a_idx))], node_no[("E", int(r.e_idx))])
             for r in accepted.itertuples(index=False)]
    for row in same_side_frame[same_side_frame["accepted"]].itertuples(index=False):
        side = str(row.side)
        edges.append((node_no[(side, int(row.left_idx))],
                      node_no[(side, int(row.right_idx))]))
    labels = union_find(len(nodes), edges)
    groups: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for node, label in zip(nodes, labels):
        groups[label].append(node)
    ordered_groups = sorted(groups.values(), key=lambda group: min(group))
    cluster_rows, components = [], []
    for number, members in enumerate(ordered_groups, 1):
        cluster_id = f"T{number:06d}"
        a_members = sorted(idx for side, idx in members if side == "A")
        e_members = sorted(idx for side, idx in members if side == "E")
        cluster_rows.extend({"side": side, "raw_idx": idx, "cluster_id": cluster_id}
                            for side, idx in sorted(members))
        components.append({"cluster_id": cluster_id, "a_indices": a_members,
                           "e_indices": e_members})
    return pd.DataFrame(cluster_rows), components


def _ticket_discrepancies(acts: pd.DataFrame, etm: pd.DataFrame,
                          components: list[dict]) -> list[dict]:
    rows = []
    for component in components:
        ai, ei = component["a_indices"], component["e_indices"]
        arows, erows = acts.loc[ai], etm.loc[ei]
        sum_a = round(float(arows["amount"].sum()), 2)
        sum_e = round(float(erows["amount"].sum()), 2)
        diff = round(sum_a - sum_e, 2)
        if abs(diff) < 0.01:
            continue
        ticket_set = set()
        for value in arows["ticket_cell"]:
            ticket_set.update(C.ticket_numbers(value, None))
        for value in erows["tickets"]:
            ticket_set.update(C.ticket_numbers(value, None))
        dates = pd.concat([arows["date_norm"], erows["date_norm"]]).dropna()
        if len(arows):
            agents = arows["folder"].dropna().astype(str)
        else:
            agents = erows["agent"].dropna().astype(str)
        agent = agents.mode().iloc[0] if len(agents) and len(agents.mode()) else ""
        rows.append({
            "row_type": "ticket", "component_id": component["cluster_id"],
            "agent": agent,
            "date": dates.min().date().isoformat() if len(dates) else "",
            "identifier": "|".join(sorted(ticket_set)),
            "sum_1c": sum_a, "sum_etm": sum_e, "diff": diff,
            "comment": "ML ticket component; review source documents",
            "source": "ML_UNCLASSIFIED",
            "a_indices": json.dumps(ai), "e_indices": json.dumps(ei),
        })
    return rows


def _payment_matches(pay_candidates: pd.DataFrame
                     ) -> tuple[pd.DataFrame, set[int], set[int]]:
    used_a: set[int] = set()
    used_e: set[int] = set()
    matches = []
    order = np.argsort(-pay_candidates["probability"].to_numpy(), kind="stable")
    for position in order:
        row = pay_candidates.iloc[int(position)]
        a_idx, e_idx = int(row["a_idx"]), int(row["e_idx"])
        if row["probability"] < THRESHOLD or a_idx in used_a or e_idx in used_e:
            continue
        used_a.add(a_idx)
        used_e.add(e_idx)
        matches.append({"a_idx": a_idx, "e_idx": e_idx,
                        "probability": float(row["probability"])})
    return pd.DataFrame(matches, columns=["a_idx", "e_idx", "probability"]), used_a, used_e


def _payment_discrepancies(acts: pd.DataFrame, etm: pd.DataFrame,
                           used_a: set[int], used_e: set[int]) -> list[dict]:
    rows = []
    pay_a = acts.loc[acts["ticket_cell"].isna()]
    pay_e = etm.loc[etm["tickets"].isna()]
    for idx, row in pay_a.iterrows():
        if int(idx) in used_a:
            continue
        refs = DOC_REF.findall(str(row["doc"]))
        amount = float(row["amount"])
        rows.append({
            "row_type": "payment", "component_id": f"PA{int(idx)}",
            "agent": str(row["folder"] or ""),
            "date": row["date_norm"].date().isoformat()
            if pd.notna(row["date_norm"]) else "",
            "identifier": refs[0] if refs else str(row["doc"]),
            "sum_1c": amount, "sum_etm": 0.0, "diff": amount,
            "comment": "ML payment unmatched on 1C side",
            "source": "ML_UNCLASSIFIED",
            "a_indices": json.dumps([int(idx)]), "e_indices": "[]",
        })
    for idx, row in pay_e.iterrows():
        if int(idx) in used_e:
            continue
        amount = float(row["amount"])
        rows.append({
            "row_type": "payment", "component_id": f"PE{int(idx)}",
            "agent": str(row["agent"] or ""),
            "date": row["date_norm"].date().isoformat()
            if pd.notna(row["date_norm"]) else "",
            "identifier": str(row["txn_id"]),
            "sum_1c": 0.0, "sum_etm": amount, "diff": round(-amount, 2),
            "comment": "ML payment unmatched on ETM side",
            "source": "ML_UNCLASSIFIED",
            "a_indices": "[]", "e_indices": json.dumps([int(idx)]),
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    _purity_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, default=S1IN / "acts.csv")
    parser.add_argument("--etm", type=Path, default=S1IN / "etm.csv")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--output-prefix", default="stage2")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write inference artifacts.")
    parser.add_argument("--training-dir", type=Path, required=True,
                        help="Run-local directory created by build_stage2_training.py.")
    parser.add_argument("--skip-source", action="store_true",
                        help="bootstrap discrepancy rows before source labels are prepared")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_prefix):
        raise ValueError("--output-prefix may contain only letters, digits, _ and -")

    def output(suffix: str) -> Path:
        return args.output_dir / f"{args.output_prefix}_{suffix}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.training_dir = args.training_dir.expanduser().resolve()
    acts, etm = load_raw_inputs(args.acts, args.etm)
    registry_path = args.registry
    if registry_path is None:
        sibling_registry = args.acts.parent / "registry.csv"
        if sibling_registry.exists():
            registry_path = sibling_registry
    raw_registry = _read_csv(registry_path) if registry_path is not None else None
    raw_counts = {
        "acts_rows": len(acts), "etm_rows": len(etm),
        "ticket_acts": int(acts["ticket_cell"].notna().sum()),
        "ticket_etm": int(etm["tickets"].notna().sum()),
        "payment_acts": int(acts["ticket_cell"].isna().sum()),
        "payment_etm": int(etm["tickets"].isna().sum()),
        "raw_status_rows": int(acts.get(
            "act_status", pd.Series(index=acts.index, dtype=object)).notna().sum()),
    }
    training_report_path = args.training_dir / "training_report.json"
    status_train_path = args.training_dir / "act_status_train.csv"
    status_validation_path = args.training_dir / "act_status_validation.csv"
    pair_train_path = args.training_dir / "pair_train.csv"
    pair_validation_path = args.training_dir / "pair_validation.csv"
    source_train_path = args.training_dir / "source_s3_train.csv"
    source_validation_path = args.training_dir / "source_s3_validation.csv"
    if not training_report_path.is_file():
        raise FileNotFoundError(
            f"Missing run-local training manifest in {args.training_dir}; "
            "run build_stage2_training.py first (frozen training inputs are unsupported).")
    required_paths = [status_train_path, status_validation_path,
                      pair_train_path, pair_validation_path]
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing run-local temporal split files in {args.training_dir}: "
            f"{', '.join(missing_paths)}")
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    temporal_split = training_report.get("temporal_split", {})
    cutoff = temporal_split.get("cutoff")
    if not cutoff:
        raise ValueError("training_report.json has no chronological validation cutoff")
    status_train = pd.read_csv(status_train_path)
    status_validation = pd.read_csv(status_validation_path)
    status_model = fit_status_model(
        status_train[STATUS_FEATURES].values, status_train["label_keep"].values)
    status_x = act_status_features(acts)
    status_probability_keep = _positive_probability(status_model, status_x.values)
    status_keep = status_probability_keep >= 0.5
    pd.DataFrame({
        "raw_idx": acts["raw_idx"].astype(int),
        "probability_keep": status_probability_keep,
        "included": status_keep,
    }).to_csv(output("act_status_predictions.csv"), index=False)
    acts = acts.loc[status_keep].copy()
    pair_train = pd.read_csv(pair_train_path)
    pair_validation = pd.read_csv(pair_validation_path)

    # Tickets: blocking by exact shared tail; model decides every candidate edge.
    ticket_model, ticket_train_info = _fit_pair_model(
        pair_train, pair_validation, ticket=True)
    ticket_pairs, overlap_n = ticket_candidate_pairs(acts, etm)
    ticket_frame = _pair_frame(acts, etm, ticket_pairs)
    if len(ticket_frame):
        ticket_x = ticket_features(ticket_frame)
        ticket_frame["probability"] = _positive_probability(ticket_model, ticket_x.values)
        ticket_frame["overlapping_tail_count"] = overlap_n
        ticket_frame["accepted"] = ticket_frame["probability"] >= THRESHOLD
    else:
        ticket_frame["probability"] = pd.Series(dtype=float)
        ticket_frame["overlapping_tail_count"] = pd.Series(dtype=int)
        ticket_frame["accepted"] = pd.Series(dtype=bool)
    same_a_pairs, overlap_a = same_side_ticket_candidate_pairs(acts, "A")
    same_e_pairs, overlap_e = same_side_ticket_candidate_pairs(etm, "E")
    same_side_frame = pd.concat([
        same_side_pair_frame(acts, same_a_pairs, "A").assign(
            overlapping_tail_count=overlap_a),
        same_side_pair_frame(etm, same_e_pairs, "E").assign(
            overlapping_tail_count=overlap_e),
    ], ignore_index=True)
    if len(same_side_frame):
        same_x = ticket_features(same_side_frame)
        same_side_frame["probability"] = _positive_probability(ticket_model, same_x.values)
        same_side_frame["accepted"] = same_side_frame["probability"] >= THRESHOLD
    else:
        same_side_frame["probability"] = pd.Series(dtype=float)
        same_side_frame["accepted"] = pd.Series(dtype=bool)
    ticket_clusters, ticket_groups = _ticket_clusters(
        acts, etm, ticket_frame, same_side_frame)
    ticket_discrepancies = _ticket_discrepancies(acts, etm, ticket_groups)

    # Payments: globally block on amount/date plus doc-ref candidates; no agent
    # alias, canonical-agent map, or Stage 1 greedy assignment is used.
    pay_model, pay_train_info = _fit_pair_model(
        pair_train, pair_validation, ticket=False)
    pay_pairs, pay_candidate_info = payment_candidate_pairs(acts, etm)
    pay_frame = _pair_frame(acts, etm, pay_pairs)
    if len(pay_frame):
        pay_x = pay_features(pay_frame)
        pay_frame["probability"] = _positive_probability(pay_model, pay_x.values)
    else:
        pay_frame["probability"] = pd.Series(dtype=float)
    matches, used_a, used_e = _payment_matches(pay_frame)
    payment_discrepancies = _payment_discrepancies(acts, etm, used_a, used_e)

    discrepancies = pd.DataFrame(
        ticket_discrepancies + payment_discrepancies,
        columns=["row_type", "component_id", "agent", "date", "identifier",
                 "sum_1c", "sum_etm", "diff", "comment", "source",
                 "a_indices", "e_indices"])
    source_train = pd.read_csv(source_train_path) if source_train_path.is_file() else None
    source_validation = (pd.read_csv(source_validation_path)
                        if source_validation_path.is_file() else None)
    source_model, source_train_mask, source_prob = None, None, None
    if args.skip_source:
        discrepancies["source"] = "ML_UNCLASSIFIED"
    else:
        if source_train is None or source_train.empty:
            raise ValueError("Source temporal train fold is empty or missing")
        if source_validation is None:
            raise FileNotFoundError(
                f"Missing source validation table: {source_validation_path}")
        # source_s3_train.csv has already been restricted to dates before cutoff.
        source_model = fit_source_model(source_train)
        source_x = build_source_features(discrepancies, acts, etm, raw_registry)
        source_x = source_x[SOURCE_FEATURES].fillna(-1)
        source_prob = (source_model.predict_proba(source_x.values) if len(source_x)
                       else np.empty((0, len(source_model.classes_)), dtype=float))
        source_pred = source_model.classes_[source_prob.argmax(axis=1)]
        discrepancies["source"] = source_pred
        discrepancies["source_confidence"] = source_prob.max(axis=1)
        for class_idx, cls in enumerate(source_model.classes_):
            discrepancies[f"p_source_{cls}"] = source_prob[:, class_idx]
    source_validation_report = None
    if source_model is not None and source_validation is not None:
        if len(source_validation):
            validation_x = source_validation[SOURCE_FEATURES].fillna(-1).values
            validation_y = source_validation["label_source"].astype(str).to_numpy()
            validation_pred = source_model.predict(validation_x)
            source_validation_report = source_metrics(validation_y, validation_pred)
        else:
            source_validation_report = {"n": 0, "accuracy": None, "macro_f1": None}
    ticket_frame[["a_idx", "e_idx", "probability", "overlapping_tail_count",
                  "accepted"]].to_csv(output("ticket_candidates.csv"), index=False)
    same_side_frame[["side", "left_idx", "right_idx", "probability",
                     "overlapping_tail_count", "accepted"]].to_csv(
        output("ticket_same_side_candidates.csv"), index=False)
    ticket_clusters.to_csv(output("ticket_clusters.csv"), index=False)
    pay_frame[["a_idx", "e_idx", "probability"]].to_csv(
        output("payment_candidates.csv"), index=False)
    matches.to_csv(output("payment_matches.csv"), index=False)
    discrepancies.to_csv(output("discrepancies_ml.csv"), index=False)

    report = {
        "seed": SEED, "cutoff": cutoff, "threshold": THRESHOLD,
        "temporal_split": temporal_split,
        "inference_boundary": {
            "raw_inputs": [str(args.acts), str(args.etm)]
            + ([str(registry_path)] if registry_path is not None else []),
            "training_dir": str(args.training_dir),
            "training_files": ["pair_train.csv", "act_status_train.csv"]
            + (["source_s3_train.csv"] if source_train is not None else []),
            "training_provenance": (
                "run-local Stage 1-derived silver labels, earlier-date train fold"),
            "independent_evaluation": False,
            "temporal_validation": True,
            "validation_rows_excluded_from_fit": True,
            "independent_truth": False,
            "rules_imported": False,
            "source_classification": "bootstrap skipped"
            if args.skip_source else "four-class ML; silver Stage 1 targets",
        },
        "raw_population": {
            **raw_counts,
            "ml_status_included_acts": len(acts),
            "ml_status_dropped_acts": len(status_keep) - len(acts),
            "ml_status_ticket_acts": int(acts["ticket_cell"].notna().sum()),
            "ml_status_payment_acts": int(acts["ticket_cell"].isna().sum()),
        },
        "status_model": {
            "classifier": type(status_model).__name__,
            "training_rows": len(status_train),
            "training_kept": int(status_train["label_keep"].sum()),
            "training_dropped": int((status_train["label_keep"] == 0).sum()),
            "validation_rows": len(status_validation),
            "validation_vs_silver": training_report.get("status", {}).get(
                "validation_vs_silver"),
            "predicted_keep": int(status_keep.sum()),
            "predicted_drop": int((~status_keep).sum()),
            "threshold": 0.5,
            "temporal_validation_cutoff": cutoff,
        },
        "ticket_model": {
            **ticket_train_info,
            "classifier": type(ticket_model).__name__,
            "candidate_gate_silver_validation": _silver_candidate_coverage(
                pair_validation),
            "candidate_coverage_population": "chronological validation fold",
            "candidate_generation": "exact shared 10-digit ticket tail; all agents",
            "candidate_pairs": len(ticket_frame),
            "accepted_edges": int(ticket_frame["accepted"].sum()),
            "same_side_candidate_pairs": len(same_side_frame),
            "same_side_accepted_edges": int(same_side_frame["accepted"].sum()),
            "components": len(ticket_groups),
            "discrepancy_rows": len(ticket_discrepancies),
        },
        "payment_model": {
            **pay_train_info,
            "classifier": type(pay_model).__name__,
            "candidate_gate_silver_validation": _silver_candidate_coverage(
                pair_validation),
            "candidate_coverage_population": "chronological validation fold",
            "candidate_generation": {
                "relative_amount_le_5pct_and_gap_le_180d_or_exact_doc_ref": True,
                **pay_candidate_info,
            },
            "candidate_pairs": len(pay_frame),
            "accepted_matches": len(matches),
            "unmatched_acts": int(acts["ticket_cell"].isna().sum() - len(used_a)),
            "unmatched_etm": int(etm["tickets"].isna().sum() - len(used_e)),
            "discrepancy_rows": len(payment_discrepancies),
        },
        "source_model": ({"skipped_for_bootstrap": True}
        if args.skip_source else {
            "classifier": type(source_model).__name__,
            "classes": list(source_model.classes_),
            "training_rows": len(source_train),
            "fit_rows": len(source_train),
            "validation_rows": len(source_validation) if source_validation is not None else 0,
            "validation_vs_silver": source_validation_report,
            "fit_scope": "earlier-date train fold only",
            "target_source_counts": source_train["label_source"].value_counts().to_dict(),
            "validation_source_counts": (
                source_validation["label_source"].value_counts().to_dict()
                if source_validation is not None else {}),
            "predicted_source_counts": discrepancies["source"].value_counts().to_dict(),
            "mean_max_probability": round(float(source_prob.max(axis=1).mean()), 6)
            if len(source_prob) else None,
            "probability_calibration": "Uncalibrated model scores, not calibrated confidence",
            "target_provenance": "Stage 1 source labels; silver, not true source",
        }),
        "output": {
            "discrepancy_rows": len(discrepancies),
            "ticket_rows": len(ticket_discrepancies),
            "payment_rows": len(payment_discrepancies),
            "source_values": sorted(discrepancies["source"].dropna().unique().tolist()),
            "source_confidence_column": None if args.skip_source else "source_confidence",
        },
    }
    with open(output("inference_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"discrepancies -> {output('discrepancies_ml.csv')}")


if __name__ == "__main__":
    main()
