"""Match bundle (Шаг 8): one command from upload folder to all reports.

Reads acts.csv / etm.csv / registry.csv (+ reconcile.yaml) from --input
(default: ../input), writes to --out (default: ../reports): discrepancies.csv
(+source, Шаг 4), reconciliation_summary.csv, quality_report.csv,
review_candidates.csv (non-authoritative audit queue), agent_risk.csv +
employee_risk.csv (Шаг 7, C+C), agent_actions.csv (Шаг 6), label_map.csv
(raw name -> canonical label per file).
Deterministic: same inputs -> byte-identical outputs (no timestamps).
Run (cwd = stage1_rules/src/): python3 match.py [--input ../input --out ../reports]
Requires: Python 3, pandas, pyyaml. (Etap 1 — tolko rules: ni odnogo ML-importa;
scikit-learn nuzhen tolko etapu 2.)
"""
import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description="Sverka 1C<->ETM: papka vygruzok -> otchety.")
    ap.add_argument("--input", type=Path, default=HERE.parent / "input")
    ap.add_argument("--out", type=Path, default=HERE.parent / "reports")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    src = args.input
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = args.config or (src / "reconcile.yaml")

    import reconcile2 as R  # noqa: E402

    # 1. engine (same code as reconcile2.py main, outputs redirected)
    sys.argv = ["reconcile2.py", "--acts", str(src / "acts.csv"),
                "--etm", str(src / "etm.csv"),
                "--registry", str(src / "registry.csv"),
                "--output", str(out / "discrepancies.csv"),
                 "--summary", str(out / "reconciliation_summary.csv"),
                 "--quality", str(out / "quality_report.csv"),
                 "--review-candidates", str(out / "review_candidates.csv"),
                 "--config", str(cfg)]
    runpy.run_module("reconcile2", run_name="__main__")

    # 2. risk features + rankings + actions (risk scripts, paths via env override).
    # MATCH_SRC = raw inputs (acts/etm/registry); MATCH_OUT = reports
    # (discrepancies/summary in, features/ranks/actions out).
    import os  # noqa: E402
    os.environ["MATCH_OUT"] = str(out)
    os.environ["MATCH_SRC"] = str(src)
    runpy.run_path(str(HERE / "build_features.py"), run_name="__main__")
    runpy.run_path(str(HERE / "rank.py"), run_name="__main__")
    runpy.run_path(str(HERE / "build_actions.py"), run_name="__main__")

    # 3. label map (raw -> canonical per file)
    import parse as P  # noqa: E402
    import pandas as pd  # noqa: E402
    config = P.load_config(cfg)
    qlog = P.QualityLog()
    acts, etm, reg, _ = P.load_data(src / "acts.csv", src / "etm.csv",
                                    src / "registry.csv", config, qlog)
    labels = P.canonical_labels(acts, etm, config)
    rows = []
    for raw, norm in sorted(set(zip(acts["folder"].astype(str),
                                       acts["agent_norm"])), key=lambda t: t[1]):
        rows.append({"file": "acts.csv", "raw": raw, "norm": norm,
                     "label": labels.get(norm, norm)})
    for raw, norm in sorted(set(zip(etm["agent"].astype(str),
                                        etm["agent_norm"])), key=lambda t: t[1]):
        rows.append({"file": "etm.csv", "raw": raw, "norm": norm,
                     "label": labels.get(norm, norm)})
    for raw, norm in sorted(set(zip(reg["party"].astype(str),
                                      reg["agent_norm"])), key=lambda t: t[1]):
        rows.append({"file": "registry.csv", "raw": raw, "norm": norm,
                     "label": labels.get(norm, norm)})
    pd.DataFrame(rows).to_csv(out / "label_map.csv", index=False)
    print(f"WROTE bundle -> {out}: discrepancies, summary, quality, review_candidates, "
          f"agent_risk, employee_risk, agent_actions, label_map")


if __name__ == "__main__":
    main()
