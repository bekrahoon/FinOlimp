from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from build_actions import build_actions  # noqa: E402


def test_actions_are_balance_deltas_and_leave_only_unresolved_gap():
    discrepancies = pd.DataFrame([
        {"agent": "Agent A", "source": "бот", "diff": 120.0},
        {"agent": "Agent A", "source": "агент", "diff": -30.0},
        {"agent": "Agent A", "source": "1С", "diff": 50.0},
        {"agent": "Agent A", "source": "выяснить", "diff": -10.0},
        {"agent": "Agent A", "source": "нет ошибки", "diff": 5.0},
    ])
    summary = pd.DataFrame([{
        "agent": "Agent A",
        "operation_diff": 135.0,
        "residual": 0.0,
        "opening_gap": 40.0,
        "interperiod_1c_gap": 0.0,
        "ending_gap": 175.0,
    }])

    action = build_actions(discrepancies, summary).iloc[0]

    # diff>0 means ETM's deposit balance must be reduced to match 1C's larger debt.
    assert action["move_etm_by"] == pytest.approx(-90.0)
    assert action["fix_1c_by"] == pytest.approx(-50.0)
    assert action["clarify_money"] == pytest.approx(10.0)

    remaining_gap_after_actions = (
        summary.loc[0, "ending_gap"] + action["move_etm_by"] + action["fix_1c_by"]
    )
    expected_remaining_gap = (
        summary.loc[0, "opening_gap"] + summary.loc[0, "interperiod_1c_gap"]
        + discrepancies.loc[discrepancies["source"].isin(["выяснить", "нет ошибки"]), "diff"].sum()
    )
    assert remaining_gap_after_actions == pytest.approx(expected_remaining_gap)


@pytest.mark.parametrize(
    "source,diff,expected_move,expected_fix",
    [
        ("бот", 100.0, -100.0, 0.0),
        ("агент", -100.0, 100.0, 0.0),
        ("1С", 100.0, 0.0, -100.0),
    ],
)
def test_action_sign_matches_balance_adjustment(source, diff, expected_move, expected_fix):
    discrepancies = pd.DataFrame([{"agent": "Agent A", "source": source, "diff": diff}])
    summary = pd.DataFrame([{"agent": "Agent A", "operation_diff": diff, "residual": 0.0}])

    action = build_actions(discrepancies, summary).iloc[0]

    assert action["move_etm_by"] == pytest.approx(expected_move)
    assert action["fix_1c_by"] == pytest.approx(expected_fix)
