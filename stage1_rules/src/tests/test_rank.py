from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from rank import build_employee_risk  # noqa: E402


def test_employee_rates_use_all_registry_rows_as_denominator():
    evidence = pd.DataFrame([
        {"employee": "a.test", "suspect": 1, "witness": 0,
         "abs_diff": 100.0, "agent": "Agent A"},
        {"employee": "a.test", "suspect": 0, "witness": 1,
         "abs_diff": 50.0, "agent": "Agent B"},
        {"employee": "b.test", "suspect": 0, "witness": 1,
         "abs_diff": 20.0, "agent": "Agent A"},
        {"employee": "c.test", "suspect": 1, "witness": 0,
         "abs_diff": 30.0, "agent": "Agent C"},
    ])
    registry = pd.DataFrame({
        "employee": ["a.test", "a.test", "b.test", "", None],
    })

    risk = build_employee_risk(evidence, registry).set_index("employee")

    assert risk.loc["a.test", "total_rows"] == 2
    assert risk.loc["a.test", "suspect_rows"] == 1
    assert risk.loc["a.test", "suspect_rate"] == 50.0
    assert risk.loc["b.test", "total_rows"] == 1
    assert risk.loc["b.test", "suspect_rate"] == 0.0
    assert risk.loc["c.test", "total_rows"] == 0
    assert pd.isna(risk.loc["c.test", "suspect_rate"])
