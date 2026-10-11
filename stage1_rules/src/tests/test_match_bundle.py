"""Testy bandla match.py (shag 8): odna komanda -> vse otchety, bayt-v-bayt.

Progonyaet match.py vo vremennuyu papku i sravnivaet md5 s zafiksirovannymi
(golden dvizhka + zakommitchennye artefakty risk_model). Lyuboy dreyf — krasnyy.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

SS = Path(__file__).resolve().parent.parent
INPUT = SS.parent / "input"

GOLDEN = {
    "discrepancies.csv": "b2bdf894cc78d1a8d38d4ae64683d81c",
    "reconciliation_summary.csv": "9af63dbc4e5e3f93f959199046a9c51d",
    "features.csv": "a55219bb9b3a7e8be378493122c636af",
    "agent_risk.csv": "4357359059a4b29fd7a881281fc0e857",
    "employee_risk.csv": "f12c67ed8b955b41f45c2479eec4a2af",
    "agent_actions.csv": "feee01871e36cbeee46743f78023e4c5",
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_match_bundle_reproducible(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SS / "match.py"),
         "--input", str(INPUT), "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=str(SS))
    assert r.returncode == 0, r.stderr[-2000:]
    for name, digest in GOLDEN.items():
        assert (tmp_path / name).exists(), name
        assert md5(tmp_path / name) == digest, name
    assert (tmp_path / "label_map.csv").exists()
    assert (tmp_path / "quality_report.csv").exists()
    assert (tmp_path / "agent_actions.csv").exists()
    review = pd.read_csv(tmp_path / "review_candidates.csv")
    assert set(review["issue"]) <= {
        "missing_act_block", "registry_void_without_etm_void", "possible_fx_difference",
    }
    assert review["issue"].value_counts().to_dict() == {
        "registry_void_without_etm_void": 48,
        "missing_act_block": 19,
        "possible_fx_difference": 20,
    }

    discrepancies = pd.read_csv(tmp_path / "discrepancies.csv")
    summary = pd.read_csv(tmp_path / "reconciliation_summary.csv").set_index("agent")
    actions = pd.read_csv(tmp_path / "agent_actions.csv").set_index("agent")
    unresolved = (discrepancies.loc[discrepancies["source"] == "выяснить"]
                  .groupby("agent")["diff"].sum())
    expected_remaining_gap = (
        summary["opening_gap"] + summary["interperiod_1c_gap"]
        + unresolved.reindex(summary.index, fill_value=0.0)
    )
    remaining_gap = (
        summary["ending_gap"] + actions["move_etm_by"] + actions["fix_1c_by"]
    )
    assert (remaining_gap - expected_remaining_gap).abs().max() < 0.011
