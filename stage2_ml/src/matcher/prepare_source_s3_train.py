"""Prepare source labels from a Stage 1 reference and Stage 2 bootstrap rows.

The source target is rule-derived (silver). The resulting feature table uses
only raw-safe operation/registry features; Stage 1 comments, source prefixes,
agent aliases and linked-party decisions are not features. Targets are split
using the current run's temporal cutoff.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FINOLIMP = HERE.parents[2]
S1 = FINOLIMP / "stage1_rules"
S1SRC, S1IN, S1REP = S1 / "src", S1 / "input", S1 / "reports"
sys.path.insert(0, str(S1SRC))
sys.path.insert(0, str(HERE))

import parse as R  # noqa: E402 (offline silver-label source)
import evaluate_s3 as Eval  # noqa: E402 (reference identity mapping)
from reconcile_s3 import _read_csv, load_raw_inputs  # noqa: E402
from source_features_s3 import (SOURCE_CATEGORICAL_FEATURES, SOURCE_FEATURES,
                                 build_source_features)  # noqa: E402
from temporal_split import date_masks  # noqa: E402

SOURCE_CLASSES = {"бот", "1С", "агент", "выяснить"}


def _date_key(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    return date.strftime("%Y-%m-%d") if pd.notna(date) else ""


def _component_operation_date(row: pd.Series, acts: pd.DataFrame,
                              etm: pd.DataFrame) -> str:
    """Use the latest member date so a cross-cutoff component cannot enter train."""
    def indices(value) -> list[int]:
        if pd.isna(value) or not str(value).strip():
            return []
        return json.loads(str(value))

    a_ids = indices(row.get("a_indices", "[]"))
    e_ids = indices(row.get("e_indices", "[]"))
    dates = []
    if a_ids:
        dates.extend(acts.loc[a_ids, "date_norm"].dropna().tolist())
    if e_ids:
        dates.extend(etm.loc[e_ids, "date_norm"].dropna().tolist())
    latest = max(dates) if dates else pd.to_datetime(row.get("date", ""), errors="coerce")
    return latest.strftime("%Y-%m-%d") if pd.notna(latest) else ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=S1IN)
    parser.add_argument("--stage1-reports", type=Path, default=S1REP)
    parser.add_argument("--stage2-discrepancies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    reports_dir = args.stage1_reports.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    stage2_discrepancies = args.stage2_discrepancies.expanduser().resolve()
    training_report_path = output_dir / "training_report.json"
    if not training_report_path.is_file():
        raise FileNotFoundError(
            f"Run-local temporal split manifest not found: {training_report_path}")
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    cutoff = training_report.get("temporal_split", {}).get("cutoff")
    if not cutoff:
        raise ValueError("training_report.json has no chronological validation cutoff")

    config_path = input_dir / "reconcile.yaml"
    cfg = R.load_config(config_path if config_path.is_file() else None)
    qlog = R.QualityLog()
    acts_ref, etm_ref, _, _ = R.load_data(
        input_dir / "acts.csv", input_dir / "etm.csv", input_dir / "registry.csv", cfg, qlog)
    acts_csv = R.strip_string_cells(R.read_csv_smart(input_dir / "acts.csv", cfg)[0])
    etm_csv = R.strip_string_cells(R.read_csv_smart(input_dir / "etm.csv", cfg)[0])
    a_engine_to_raw, e_engine_to_raw = Eval._raw_row_maps(
        acts_csv, etm_csv, acts_ref, etm_ref)
    raw_a_to_norm = {raw_idx: str(acts_ref.loc[engine_idx, "agent_norm"])
                     for engine_idx, raw_idx in a_engine_to_raw.items()}
    raw_e_to_norm = {raw_idx: str(etm_ref.loc[engine_idx, "agent_norm"])
                     for engine_idx, raw_idx in e_engine_to_raw.items()}
    norm_by_label = {label: norm for norm, label in
                     R.canonical_labels(acts_ref, etm_ref, cfg).items()}

    reference = pd.read_csv(reports_dir / "discrepancies.csv")
    target_by_key: dict[tuple, list[str]] = defaultdict(list)
    for _, row in reference.iterrows():
        cls = str(row["source"])
        if cls not in SOURCE_CLASSES:
            continue
        norm = norm_by_label.get(str(row["agent"]), str(row["agent"]))
        key = (norm, _date_key(row["date"]), str(row["identifier"]),
               round(float(row["sum_1c"]), 2), round(float(row["sum_etm"]), 2))
        target_by_key[key].append(cls)

    s2_rows = pd.read_csv(stage2_discrepancies)
    s2_norms = []
    s2_keys = []
    for _, row in s2_rows.iterrows():
        a_ids = json.loads(row["a_indices"] or "[]")
        e_ids = json.loads(row["e_indices"] or "[]")
        norms = {raw_a_to_norm[i] for i in a_ids if i in raw_a_to_norm}
        norms.update(raw_e_to_norm[i] for i in e_ids if i in raw_e_to_norm)
        norm = next(iter(norms)) if len(norms) == 1 else "MULTI:" + "|".join(sorted(norms))
        s2_norms.append(norm)
        s2_keys.append((norm, _date_key(row["date"]), str(row["identifier"]),
                         round(float(row["sum_1c"]), 2),
                         round(float(row["sum_etm"]), 2)))

    raw_acts, raw_etm = load_raw_inputs(input_dir / "acts.csv", input_dir / "etm.csv")
    raw_reg = _read_csv(input_dir / "registry.csv")
    feat = build_source_features(s2_rows, raw_acts, raw_etm, raw_reg)
    feat["stage2_output_idx"] = s2_rows.index.astype(int)
    feat["a_indices_eval_only"] = s2_rows["a_indices"].astype(str)
    feat["e_indices_eval_only"] = s2_rows["e_indices"].astype(str)
    labels = []
    for key in s2_keys:
        vals = target_by_key.get(key, [])
        labels.append(vals.pop(0) if vals else None)
    feat["label_source"] = labels
    feat["agent_norm_eval_only"] = s2_norms
    feat["operation_date"] = [
        _component_operation_date(row, raw_acts, raw_etm)
        for _, row in s2_rows.iterrows()
    ]
    feat["identifier_eval_only"] = s2_rows["identifier"].astype(str)

    matched = feat["label_source"].notna()
    all_labeled = feat.loc[matched].copy()
    train_mask, validation_mask, unknown_mask = date_masks(
        all_labeled["operation_date"], cutoff)
    train = all_labeled.loc[train_mask].copy()
    validation = all_labeled.loc[validation_mask].copy()
    counts = train["label_source"].value_counts().to_dict()
    report = {
        "stage2_output_rows": len(s2_rows),
        "matched_to_stage1_source_label": len(all_labeled),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "unknown_date_excluded_rows": int(unknown_mask.sum()),
        "cutoff": cutoff,
        "split_date_basis": "latest operation date among a_indices/e_indices",
        "unmatched_stage2_rows": int((~matched).sum()),
        "source_class_counts": counts,
        "validation_source_class_counts": validation["label_source"].value_counts().to_dict(),
        "source_feature_count": len(SOURCE_FEATURES),
        "features": SOURCE_FEATURES,
        "raw_categorical_feature_count": len(SOURCE_CATEGORICAL_FEATURES),
        "raw_categorical_features": SOURCE_CATEGORICAL_FEATURES,
        "categorical_feature_provenance": (
            "Raw operation/registry values summarized within raw ticket-linked "
            "candidate rows; no Stage 1 alias or party-link decisions."),
        "label_provenance": "Stage 1 discrepancies.source; training target only",
        "independent_evaluation": False,
        "excluded_from_features": ["Stage 1 source/comment/prefix", "aliases",
                                    "Stage 1 registry links", "agent_norm_eval_only",
                                    "identifier_eval_only", "stage2_output_idx",
                                    "a_indices_eval_only", "e_indices_eval_only"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    all_labeled.to_csv(output_dir / "source_s3_all.csv", index=False)
    train.to_csv(output_dir / "source_s3_train.csv", index=False)
    validation.to_csv(output_dir / "source_s3_validation.csv", index=False)
    (output_dir / "source_s3_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
