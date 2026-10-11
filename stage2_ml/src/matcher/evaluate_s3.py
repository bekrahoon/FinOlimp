"""Offline fidelity harness for reconcile_s3.py.

Stage 1 modules/data are read here only as reference labels. This file is not
imported by or called from the Stage 2 inference runner.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
FINOLIMP = HERE.parents[2]
S1 = FINOLIMP / "stage1_rules"
S1SRC = S1 / "src"
sys.path.insert(0, str(S1SRC))
sys.path.insert(0, str(HERE))


def _row_key(row: pd.Series, agent_key: str) -> tuple:
    return (str(row["row_type"]), agent_key, str(row["date"]),
            str(row["identifier"]), round(float(row["sum_1c"]), 2),
            round(float(row["sum_etm"]), 2))


def _engine_ticket_components(acts, etm, R2):
    labels: dict[tuple[str, int], str] = {}
    components: dict[str, set[tuple[str, int]]] = {}
    direct_pairs: set[tuple[int, int]] = set()
    same_side_pairs: set[tuple[str, int, int]] = set()
    agents = sorted(set(acts["agent_norm"]) | set(etm["agent_norm"]))
    for agent in agents:
        aa = acts[(acts["agent_norm"] == agent) & acts["ticket_cell"].notna()]
        ee = etm[(etm["agent_norm"] == agent) & etm["tickets"].notna()]
        for ci, (a_ids, e_ids) in enumerate(R2.ticket_components(aa, ee)):
            cid = f"{agent}#{ci}"
            members = {*(('A', int(i)) for i in a_ids),
                       *(('E', int(i)) for i in e_ids)}
            components[cid] = members
            for side, idx in members:
                labels[(side, idx)] = cid
            for a_idx in a_ids:
                for e_idx in e_ids:
                    if acts.loc[a_idx, "ticket_set"] & etm.loc[e_idx, "ticket_set"]:
                        direct_pairs.add((int(a_idx), int(e_idx)))
            for side_ids, frame, side in ((a_ids, acts, "A"), (e_ids, etm, "E")):
                for pos, left in enumerate(side_ids):
                    for right in side_ids[pos + 1:]:
                        if frame.loc[left, "ticket_set"] & frame.loc[right, "ticket_set"]:
                            same_side_pairs.add((side, int(left), int(right)))
    return labels, components, direct_pairs, same_side_pairs


def _raw_row_maps(acts_raw: pd.DataFrame, etm_raw: pd.DataFrame,
                  acts_engine: pd.DataFrame, etm_engine: pd.DataFrame
                  ) -> tuple[dict[int, int], dict[int, int]]:
    """Map parser row indices back to original CSV row numbers for evaluation."""
    a_cols = [col for col in acts_raw.columns if col in acts_engine.columns]
    numeric_cols = {"debet", "credit", "saldo_start", "saldo_end"}

    def value_key(value, numeric=False):
        if pd.isna(value):
            return None
        if numeric:
            return round(float(value), 2)
        return str(value).strip()

    def key(row, columns):
        return tuple(value_key(row[col], col in numeric_cols)
                     for col in columns)

    raw_queues: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in acts_raw.iterrows():
        raw_queues[key(row, a_cols)].append(int(idx))
    used: dict[tuple, int] = defaultdict(int)
    a_engine_to_raw = {}
    for idx, row in acts_engine.iterrows():
        row_key = key(row, a_cols)
        pos = used[row_key]
        if row_key not in raw_queues or pos >= len(raw_queues[row_key]):
            raise AssertionError(f"could not map Stage 1 acts row {idx} to raw CSV")
        a_engine_to_raw[int(idx)] = raw_queues[row_key][pos]
        used[row_key] += 1

    e_raw_by_txn = {str(row["txn_id"]): int(idx)
                    for idx, row in etm_raw.iterrows()}
    e_engine_to_raw = {}
    for idx, row in etm_engine.iterrows():
        txn = str(row["txn_id"])
        if txn not in e_raw_by_txn:
            raise AssertionError(f"could not map Stage 1 ETM row {idx} to raw CSV")
        e_engine_to_raw[int(idx)] = e_raw_by_txn[txn]
    return a_engine_to_raw, e_engine_to_raw


def _norm_for_stage2_row(row: pd.Series, raw_a_to_norm: dict[int, str],
                         raw_e_to_norm: dict[int, str]) -> str:
    indices_a = json.loads(row["a_indices"] or "[]")
    indices_e = json.loads(row["e_indices"] or "[]")
    norms = set()
    norms.update(raw_a_to_norm[i] for i in indices_a if i in raw_a_to_norm)
    norms.update(raw_e_to_norm[i] for i in indices_e if i in raw_e_to_norm)
    if len(norms) == 1:
        return next(iter(norms))
    return "MULTI:" + "|".join(sorted(norms))


def _component_metrics(node_labels: dict[tuple[str, int], str],
                      stage2_labels: dict[tuple[str, int], str],
                      ref_nodes: set[tuple[str, int]]) -> dict:
    common = sorted(ref_nodes.intersection(stage2_labels))
    ari = adjusted_rand_score([node_labels[n] for n in common],
                              [stage2_labels[n] for n in common]) \
        if len(common) > 1 else 1.0
    ml_sets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    ref_sets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for node in common:
        ml_sets[stage2_labels[node]].add(node)
        ref_sets[node_labels[node]].add(node)
    ml_components = {frozenset(nodes) for nodes in ml_sets.values()}
    exact = sum(frozenset(nodes) in ml_components for nodes in ref_sets.values())
    return {"ari": round(float(ari), 6), "exact": exact,
            "reference_components": len(ref_sets), "nodes_compared": len(common)}


def _ticket_threshold_sweep(candidate_df: pd.DataFrame, same_side_df: pd.DataFrame,
                            nodes: list[tuple[str, int]],
                            node_labels: dict[tuple[str, int], str],
                            ref_nodes: set[tuple[str, int]],
                            engine_direct_pairs: set[tuple[int, int]],
                            engine_same_side_pairs: set[tuple[str, int, int]]) -> list[dict]:
    from s2_features import union_find

    node_no = {node: i for i, node in enumerate(nodes)}
    out = []
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        chosen = candidate_df[candidate_df["probability"] >= threshold]
        edges = [(node_no[("A", int(r.a_idx))], node_no[("E", int(r.e_idx))])
                 for r in chosen.itertuples(index=False)]
        same_chosen = same_side_df[same_side_df["probability"] >= threshold]
        edges.extend((node_no[(str(r.side), int(r.left_idx))],
                      node_no[(str(r.side), int(r.right_idx))])
                     for r in same_chosen.itertuples(index=False))
        chosen_pairs = set(zip(chosen["a_idx"].astype(int),
                               chosen["e_idx"].astype(int)))
        chosen_same_pairs = set(zip(same_chosen["side"].astype(str),
                                    same_chosen["left_idx"].astype(int),
                                    same_chosen["right_idx"].astype(int)))
        labels = union_find(len(nodes), edges)
        stage2_labels = {node: str(label) for node, label in zip(nodes, labels)}
        metrics = _component_metrics(node_labels, stage2_labels, ref_nodes)
        out.append({"threshold": threshold, "accepted_edges": len(chosen),
                    "engine_direct_edges_accepted": len(chosen_pairs & engine_direct_pairs),
                    "non_reference_edges_accepted": len(chosen_pairs - engine_direct_pairs),
                    "same_side_candidate_edges": len(same_chosen),
                    "engine_same_side_edges_accepted": len(
                        chosen_same_pairs & engine_same_side_pairs),
                    "non_reference_same_side_edges_accepted": len(
                        chosen_same_pairs - engine_same_side_pairs),
                    **metrics})
    return out


def _attach_evaluation_scope(report: dict, inference_report: dict) -> dict:
    """Record the current run-local temporal-validation caveat."""
    boundary = inference_report.get("inference_boundary", {})
    if not boundary.get("temporal_validation"):
        raise ValueError("Expected a run-local temporal-validation inference report")
    report["evaluation_scope"] = {
        "independent_evaluation": False,
        "temporal_validation": True,
        "fit_scope": "earlier-date train; latest-date validation",
        "interpretation": (
            "Temporal holdout is excluded from model fitting, but targets come from Stage 1 "
            "silver labels; this is out-of-time fidelity, not independent human truth. "
            "Whole-dataset metrics also include in-sample rows."),
    }
    discrepancy_report = report["discrepancies"]
    discrepancy_report["source_agreement_temporal_validation_vs_silver"] = \
        discrepancy_report["source_agreement_temporal_holdout"]
    discrepancy_report["source_temporal_validation_rows"] = \
        discrepancy_report["source_temporal_holdout_rows"]
    discrepancy_report["source_agreement_temporal_holdout"] = None
    discrepancy_report["source_temporal_holdout_rows"] = None
    discrepancy_report["source_note"] = (
        "The date slice is excluded from model fitting and measures out-of-time fidelity "
        "to Stage 1 silver source labels; it is not independent human truth.")
    report["model_validation_vs_silver"] = {
        "status_rows": inference_report.get("status_model", {}).get(
            "validation_vs_silver"),
        "ticket_pairs": inference_report.get("ticket_model", {}).get(
            "validation_at_threshold"),
        "payment_pairs": inference_report.get("payment_model", {}).get(
            "validation_at_threshold"),
        "source_rows": inference_report.get("source_model", {}).get(
            "validation_vs_silver"),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Raw Stage 1 input directory.")
    parser.add_argument("--stage1-reports", type=Path, required=True,
                        help="Stage 1 report directory used as the fidelity reference.")
    parser.add_argument("--stage2-dir", type=Path, required=True,
                        help="Directory containing Stage 2 artifacts.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write the evaluation report (default: --stage2-dir).")
    parser.add_argument("--output-prefix", default="stage2",
                        help="Run-local artifact prefix to evaluate (default: stage2).")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_prefix):
        parser.error("--output-prefix may contain only letters, digits, _ and -")
    input_dir = args.input_dir.expanduser().resolve()
    stage1_reports = args.stage1_reports.expanduser().resolve()
    stage2_dir = args.stage2_dir.expanduser().resolve()
    output_dir = (args.output_dir or stage2_dir).expanduser().resolve()
    inference_report_path = stage2_dir / f"{args.output_prefix}_inference_report.json"
    if not inference_report_path.is_file():
        raise FileNotFoundError(
            f"Missing run-local inference report: {inference_report_path}")
    inference_report = json.loads(inference_report_path.read_text(encoding="utf-8"))
    validation_cutoff = inference_report.get("cutoff")
    if not validation_cutoff:
        raise ValueError("Run-local inference report has no temporal cutoff")
    source_validation_indices = None
    training_dir = inference_report.get("inference_boundary", {}).get("training_dir")
    validation_path = (Path(training_dir) / "source_s3_validation.csv"
                       if training_dir else None)
    if validation_path is not None and validation_path.is_file():
        validation_rows = pd.read_csv(validation_path)
        if "stage2_output_idx" in validation_rows:
            source_validation_indices = set(
                validation_rows["stage2_output_idx"].astype(int).tolist())

    import parse as R  # reference-only
    import reconcile2 as R2  # reference-only
    import payment_pairing as PairBuild  # reference-only replay for engine payment pairs

    config_path = input_dir / "reconcile.yaml"
    cfg = R.load_config(config_path if config_path.is_file() else None)
    qlog = R.QualityLog()
    acts_raw = R.strip_string_cells(R.read_csv_smart(input_dir / "acts.csv", cfg)[0])
    etm_raw = R.strip_string_cells(R.read_csv_smart(input_dir / "etm.csv", cfg)[0])
    acts, etm, reg, _ = R.load_data(
        input_dir / "acts.csv", input_dir / "etm.csv", input_dir / "registry.csv", cfg, qlog)
    a_engine_to_raw, e_engine_to_raw = _raw_row_maps(
        acts_raw, etm_raw, acts, etm)
    raw_a_to_norm = {raw_idx: str(acts.loc[engine_idx, "agent_norm"])
                     for engine_idx, raw_idx in a_engine_to_raw.items()}
    raw_e_to_norm = {raw_idx: str(etm.loc[engine_idx, "agent_norm"])
                     for engine_idx, raw_idx in e_engine_to_raw.items()}

    # Component-level ticket fidelity (reference rows only; raw extra rows are
    # disclosed separately, not silently filtered from the Stage 2 output).
    (engine_of_engine_idx, engine_components_engine_idx, engine_direct_engine_idx,
     engine_same_side_engine_idx) = _engine_ticket_components(acts, etm, R2)
    engine_of = {(side, a_engine_to_raw[idx] if side == "A" else e_engine_to_raw[idx]): comp
                 for (side, idx), comp in engine_of_engine_idx.items()}
    engine_components = {comp: {
        (side, a_engine_to_raw[idx] if side == "A" else e_engine_to_raw[idx])
        for side, idx in members}
        for comp, members in engine_components_engine_idx.items()}
    engine_direct = {(a_engine_to_raw[a], e_engine_to_raw[e])
                     for a, e in engine_direct_engine_idx}
    engine_same_side = {
        (side,
         a_engine_to_raw[left] if side == "A" else e_engine_to_raw[left],
         a_engine_to_raw[right] if side == "A" else e_engine_to_raw[right])
        for side, left, right in engine_same_side_engine_idx}
    stage2_clusters = pd.read_csv(
        stage2_dir / f"{args.output_prefix}_ticket_clusters.csv")
    stage2_of = {(str(r.side), int(r.raw_idx)): str(r.cluster_id)
                 for r in stage2_clusters.itertuples(index=False)}
    ref_nodes = set(engine_of)
    common_nodes = sorted(ref_nodes.intersection(stage2_of))
    projected_ml: dict[str, set[tuple[str, int]]] = defaultdict(set)
    projected_engine: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for node in common_nodes:
        projected_ml[stage2_of[node]].add(node)
        projected_engine[engine_of[node]].add(node)
    base_component_metrics = _component_metrics(engine_of, stage2_of, ref_nodes)

    ticket_candidates = pd.read_csv(
        stage2_dir / f"{args.output_prefix}_ticket_candidates.csv")
    ticket_candidates["accepted"] = ticket_candidates["accepted"].astype(bool)
    candidate_pairs = set(zip(ticket_candidates["a_idx"].astype(int),
                              ticket_candidates["e_idx"].astype(int)))
    accepted_pairs = set(zip(
        ticket_candidates.loc[ticket_candidates["accepted"], "a_idx"].astype(int),
        ticket_candidates.loc[ticket_candidates["accepted"], "e_idx"].astype(int)))
    same_side_candidates_df = pd.read_csv(
        stage2_dir / f"{args.output_prefix}_ticket_same_side_candidates.csv")
    same_side_candidates = set(zip(
        same_side_candidates_df["side"].astype(str),
        same_side_candidates_df["left_idx"].astype(int),
        same_side_candidates_df["right_idx"].astype(int)))
    same_side_accepted = set(zip(
        same_side_candidates_df.loc[same_side_candidates_df["accepted"], "side"].astype(str),
        same_side_candidates_df.loc[same_side_candidates_df["accepted"], "left_idx"].astype(int),
        same_side_candidates_df.loc[same_side_candidates_df["accepted"], "right_idx"].astype(int)))
    engine_direct_covered = engine_direct & candidate_pairs
    engine_direct_accepted = engine_direct & accepted_pairs
    raw_nodes = set(stage2_of)
    non_reference_ticket_candidates = candidate_pairs - engine_direct
    filtered_act_candidate_pairs = {
        (a_idx, e_idx) for a_idx, e_idx in non_reference_ticket_candidates
        if a_idx not in raw_a_to_norm
    }
    accepted_non_reference = accepted_pairs - engine_direct
    ticket_sweep_nodes = ([('A', int(i)) for i in stage2_clusters.loc[
        stage2_clusters["side"] == "A", "raw_idx"].astype(int).sort_values()]
        + [('E', int(i)) for i in stage2_clusters.loc[
            stage2_clusters["side"] == "E", "raw_idx"].astype(int).sort_values()])
    ticket_sweep = _ticket_threshold_sweep(
        ticket_candidates, same_side_candidates_df, ticket_sweep_nodes, engine_of,
        ref_nodes, engine_direct, engine_same_side)
    direct_scores = ticket_candidates.set_index(["a_idx", "e_idx"])["probability"]
    direct_edge_scores = [float(direct_scores.loc[pair]) for pair in engine_direct
                          if pair in direct_scores.index]

    # Payment pair fidelity against the offline engine replay.
    engine_payment_pairs = {
        (a_engine_to_raw[int(a)], e_engine_to_raw[int(e)])
        for a, e, _ in PairBuild.match_payments_with_pairs(acts, etm, cfg)
    }
    stage2_payment_matches_df = pd.read_csv(
        stage2_dir / f"{args.output_prefix}_payment_matches.csv")
    stage2_payment_pairs = set(zip(
        stage2_payment_matches_df["a_idx"].astype(int),
        stage2_payment_matches_df["e_idx"].astype(int)))
    payment_candidates_df = pd.read_csv(
        stage2_dir / f"{args.output_prefix}_payment_candidates.csv")
    payment_candidate_pairs = set(zip(
        payment_candidates_df["a_idx"].astype(int),
        payment_candidates_df["e_idx"].astype(int)))
    engine_payment_covered = engine_payment_pairs & payment_candidate_pairs
    stage2_pay_a = set(acts_raw.index[acts_raw["ticket_cell"].isna()].astype(int))
    stage2_pay_e = set(etm_raw.index[etm_raw["tickets"].isna()].astype(int))
    engine_pay_a = {a_engine_to_raw[int(i)] for i in acts.index[acts["ticket_cell"].isna()]}
    engine_pay_e = {e_engine_to_raw[int(i)] for i in etm.index[etm["tickets"].isna()]}
    engine_matched_a = {a for a, _ in engine_payment_pairs}
    engine_matched_e = {e for _, e in engine_payment_pairs}
    stage2_matched_a = {a for a, _ in stage2_payment_pairs}
    stage2_matched_e = {e for _, e in stage2_payment_pairs}
    engine_unmatched_a, engine_unmatched_e = engine_pay_a - engine_matched_a, \
        engine_pay_e - engine_matched_e
    stage2_unmatched_a, stage2_unmatched_e = stage2_pay_a - stage2_matched_a, \
        stage2_pay_e - stage2_matched_e

    # Final discrepancy fidelity. Use raw row indices to normalize the Stage 2
    # raw names with the reference labels only inside this evaluation harness.
    engine_rows = pd.read_csv(stage1_reports / "discrepancies.csv")
    engine_rows["row_type"] = engine_rows["comment"].map(
        lambda v: "payment" if str(v).startswith("Платеж") else "ticket")
    labels = R.canonical_labels(acts, etm, cfg)
    norm_by_label = {label: norm for norm, label in labels.items()}
    engine_keys = Counter()
    for _, row in engine_rows.iterrows():
        norm = norm_by_label.get(str(row["agent"]), str(row["agent"]))
        engine_keys[_row_key(row, norm)] += 1
    stage2_rows = pd.read_csv(
        stage2_dir / f"{args.output_prefix}_discrepancies_ml.csv")
    stage2_keys = Counter()
    for stage2_idx, row in stage2_rows.iterrows():
        stage2_keys[_row_key(row, _norm_for_stage2_row(
            row, raw_a_to_norm, raw_e_to_norm))] += 1
    exact_row_overlap = sum((engine_keys & stage2_keys).values())
    row_precision = exact_row_overlap / max(len(stage2_rows), 1)
    row_recall = exact_row_overlap / max(len(engine_rows), 1)
    row_overlap_by_type = {}
    for row_type in ("ticket", "payment"):
        engine_type = Counter({key: count for key, count in engine_keys.items()
                               if key[0] == row_type})
        stage2_type = Counter({key: count for key, count in stage2_keys.items()
                               if key[0] == row_type})
        overlap = sum((engine_type & stage2_type).values())
        row_overlap_by_type[row_type] = {
            "stage1_rows": sum(engine_type.values()),
            "stage2_rows": sum(stage2_type.values()),
            "exact_overlap": overlap,
        }

    source_classes = engine_rows["source"].value_counts().to_dict()
    source_queues: dict[tuple, list[str]] = defaultdict(list)
    for _, row in engine_rows.iterrows():
        norm = norm_by_label.get(str(row["agent"]), str(row["agent"]))
        source_queues[_row_key(row, norm)].append(str(row["source"]))
    source_confusion: Counter[tuple[str, str]] = Counter()
    n_source_compared = source_correct = 0
    n_temporal_compared = temporal_correct = 0
    for _, row in stage2_rows.iterrows():
        key = _row_key(row, _norm_for_stage2_row(
            row, raw_a_to_norm, raw_e_to_norm))
        labels_for_row = source_queues.get(key, [])
        if not labels_for_row:
            continue
        reference_source = labels_for_row.pop(0)
        predicted_source = str(row["source"])
        source_confusion[(reference_source, predicted_source)] += 1
        n_source_compared += 1
        source_correct += reference_source == predicted_source
        is_validation_row = (
            int(stage2_idx) in source_validation_indices
            if source_validation_indices is not None else
            pd.to_datetime(row["date"], errors="coerce")
            >= pd.Timestamp(validation_cutoff))
        if is_validation_row:
            n_temporal_compared += 1
            temporal_correct += reference_source == predicted_source
    report = {
        "reference": "Stage 1 outputs used only in this offline evaluator",
        "tickets": {
            "engine_reference_nodes": len(ref_nodes),
            "stage2_raw_ticket_nodes": len(raw_nodes),
            "reference_nodes_covered": len(common_nodes),
            "stage2_only_ticket_nodes": len(raw_nodes - ref_nodes),
            "engine_components": len(engine_components),
            "exact_components_on_common_nodes": (
                f"{base_component_metrics['exact']}/"
                f"{base_component_metrics['reference_components']}"),
            "ari_on_common_nodes": base_component_metrics["ari"],
            "engine_direct_shared_tail_pairs": len(engine_direct),
            "direct_pairs_in_stage2_candidate_set": len(engine_direct_covered),
            "direct_pairs_accepted_by_ml": len(engine_direct_accepted),
            "non_reference_candidate_pairs": len(non_reference_ticket_candidates),
            "non_reference_pairs_with_stage1_filtered_act": len(
                filtered_act_candidate_pairs),
            "accepted_non_reference_candidate_pairs": len(accepted_non_reference),
            "candidate_coverage": round(len(engine_direct_covered)
                                         / max(len(engine_direct), 1), 6),
            "accepted_edge_recall_vs_engine_direct_pairs": round(
                len(engine_direct_accepted) / max(len(engine_direct), 1), 6),
            "engine_same_side_pairs": len(engine_same_side),
            "same_side_pairs_in_stage2_candidate_set": len(
                engine_same_side & same_side_candidates),
            "same_side_pairs_accepted_by_ml": len(
                engine_same_side & same_side_accepted),
            "same_side_candidate_pairs": len(same_side_candidates_df),
            "engine_direct_edge_score_quantiles": {
                str(q): round(float(pd.Series(direct_edge_scores).quantile(q)), 6)
                for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
            "threshold_sweep": ticket_sweep,
        },
        "payments": {
            "stage2_only_raw_act_payment_rows": len(stage2_pay_a - engine_pay_a),
            "stage2_only_raw_etm_payment_rows": len(stage2_pay_e - engine_pay_e),
            "engine_matches": len(engine_payment_pairs),
            "stage2_matches": len(stage2_payment_pairs),
            "same_pair_matches": len(engine_payment_pairs & stage2_payment_pairs),
            "pair_precision_vs_engine": round(
                len(engine_payment_pairs & stage2_payment_pairs)
                / max(len(stage2_payment_pairs), 1), 6),
            "pair_recall_vs_engine": round(
                len(engine_payment_pairs & stage2_payment_pairs)
                / max(len(engine_payment_pairs), 1), 6),
            "engine_pairs_in_stage2_candidate_set": len(engine_payment_covered),
            "candidate_coverage_vs_engine": round(
                len(engine_payment_covered) / max(len(engine_payment_pairs), 1), 6),
            "engine_unmatched_acts": len(engine_unmatched_a),
            "stage2_unmatched_acts_all_raw": len(stage2_unmatched_a),
            "stage2_unmatched_acts_engine_population": len(
                stage2_unmatched_a & engine_pay_a),
            "same_unmatched_acts": len(engine_unmatched_a & stage2_unmatched_a),
            "engine_unmatched_etm": len(engine_unmatched_e),
            "stage2_unmatched_etm_all_raw": len(stage2_unmatched_e),
            "stage2_unmatched_etm_engine_population": len(
                stage2_unmatched_e & engine_pay_e),
            "same_unmatched_etm": len(engine_unmatched_e & stage2_unmatched_e),
        },
        "discrepancies": {
            "stage1_rows": len(engine_rows),
            "stage2_rows": len(stage2_rows),
            "exact_row_overlap": exact_row_overlap,
            "exact_overlap_by_type": row_overlap_by_type,
            "precision_vs_engine": round(row_precision, 6),
            "recall_vs_engine": round(row_recall, 6),
            "stage1_abs_diff_total": round(float(engine_rows["diff"].abs().sum()), 2),
            "stage2_abs_diff_total": round(float(stage2_rows["diff"].abs().sum()), 2),
            "stage1_source_counts": source_classes,
            "source_rows_compared": n_source_compared,
            "source_agreement_on_exact_rows": round(
                source_correct / max(n_source_compared, 1), 6),
            "source_temporal_holdout_rows": n_temporal_compared,
            "source_agreement_temporal_holdout": round(
                temporal_correct / max(n_temporal_compared, 1), 6),
            "source_confusion_reference_vs_ml": {
                f"{ref}->{pred}": count
                for (ref, pred), count in source_confusion.items()},
            "source_note": (
                "Fidelity to Stage 1 source classes; the temporal validation is out-of-time "
                "but still compared with silver labels."),
        },
    }
    report = _attach_evaluation_scope(report, inference_report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.output_prefix}_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"evaluation -> {report_path}")


if __name__ == "__main__":
    main()
