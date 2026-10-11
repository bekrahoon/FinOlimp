from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SS))

import reconcile2 as R  # noqa: E402
from review_candidates import build_review_candidates  # noqa: E402


TICKET = "1234567890"


def test_missing_act_month_is_a_separate_review_candidate():
    acts = pd.DataFrame([
        {"agent_norm": "agent a", "period_start_norm": pd.Timestamp("2026-01-01"),
         "saldo_start": 0.0, "saldo_end": 100.0},
        {"agent_norm": "agent a", "period_start_norm": pd.Timestamp("2026-03-01"),
         "saldo_start": 200.0, "saldo_end": 300.0},
    ])
    etm = pd.DataFrame([{
        "agent_norm": "agent a", "date_norm": pd.Timestamp("2026-02-12"),
        "sum_etm": 100.0, "txn_id": "etm-feb-1",
    }])

    candidates = build_review_candidates(
        acts, etm, pd.DataFrame(), R.load_config(None), {"agent a": "Agent A"})

    row = candidates.loc[candidates["issue"] == "missing_act_block"].iloc[0]
    assert row["agent"] == "Agent A"
    assert row["identifier"] == "2026-02"
    assert row["amount_delta"] == 0.0
    assert "ETM net matches" in row["evidence"]
    assert "etm-feb-1" in row["source_rows"]


def test_registry_void_without_etm_void_is_review_not_an_attribution():
    etm = pd.DataFrame([{
        "agent_norm": "agent a", "kind": "выкуп", "ticket_set": {TICKET},
        "txn_id": "sale-1",
    }])
    registry = pd.DataFrame([{
        "agent_norm": "agent a", "kind": "войд", "ticket_set": {TICKET},
        "date_norm": pd.Timestamp("2026-02-03"),
    }])

    candidates = build_review_candidates(
        pd.DataFrame(), etm, registry, R.load_config(None), {"agent a": "Agent A"})

    assert candidates["issue"].tolist() == ["registry_void_without_etm_void"]
    row = candidates.iloc[0]
    assert row["agent"] == "Agent A"
    assert row["identifier"] == TICKET
    assert pd.isna(row["amount_delta"])
    assert "sale-1" in row["evidence"]
    assert "Check cancellation" in row["evidence"]


def test_fx_date_shift_requires_ticket_link_currency_and_matching_rate_ratio():
    acts = pd.DataFrame([{
        "agent_norm": "agent a", "ticket_cell": TICKET, "ticket_set": {TICKET},
        "doc": "Реализация ticket", "date_norm": pd.Timestamp("2026-03-01"),
        "sum_1c": 850.0,
    }])
    etm = pd.DataFrame([{
        "agent_norm": "agent a", "tickets": TICKET, "ticket_set": {TICKET},
        "kind": "выкуп", "date_norm": pd.Timestamp("2026-03-02"), "sum_etm": 860.0,
        "currency": "USD", "txn_id": "sale-usd-1",
    }])
    registry = pd.DataFrame({
        "date_norm": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")],
        "rate_usd": [85.0, 86.0],
        "rate_eur": [float("nan")] * 2,
        "rate_rub": [float("nan")] * 2,
        "rate_kzt": [float("nan")] * 2,
    })

    candidates = build_review_candidates(
        acts, etm, registry, R.load_config(None), {"agent a": "Agent A"})

    assert candidates["issue"].tolist() == ["possible_fx_difference"]
    row = candidates.iloc[0]
    assert row["agent"] == "Agent A"
    assert row["amount_delta"] == -10.0
    assert "currency USD" in row["evidence"]
    assert "Candidate only" in row["evidence"]
    assert "sale-usd-1" in row["source_rows"]


def test_fx_candidate_is_suppressed_without_a_tight_rate_ratio_match():
    acts = pd.DataFrame([{
        "agent_norm": "agent a", "ticket_cell": TICKET, "ticket_set": {TICKET},
        "doc": "Реализация ticket", "date_norm": pd.Timestamp("2026-03-01"),
        "sum_1c": 800.0,
    }])
    etm = pd.DataFrame([{
        "agent_norm": "agent a", "tickets": TICKET, "ticket_set": {TICKET},
        "kind": "выкуп", "date_norm": pd.Timestamp("2026-03-02"), "sum_etm": 860.0,
        "currency": "USD", "txn_id": "sale-usd-1",
    }])
    registry = pd.DataFrame({
        "date_norm": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")],
        "rate_usd": [85.0, 86.0],
        "rate_eur": [float("nan")] * 2,
        "rate_rub": [float("nan")] * 2,
        "rate_kzt": [float("nan")] * 2,
    })

    candidates = build_review_candidates(
        acts, etm, registry, R.load_config(None), {"agent a": "Agent A"})

    assert candidates.empty


def test_fx_candidate_supports_etm_date_before_1c():
    acts = pd.DataFrame([{
        "agent_norm": "agent a", "ticket_cell": TICKET, "ticket_set": {TICKET},
        "doc": "Реализация ticket", "date_norm": pd.Timestamp("2026-03-02"),
        "sum_1c": 860.0,
    }])
    etm = pd.DataFrame([{
        "agent_norm": "agent a", "tickets": TICKET, "ticket_set": {TICKET},
        "kind": "выкуп", "date_norm": pd.Timestamp("2026-03-01"), "sum_etm": 850.0,
        "currency": "USD", "txn_id": "sale-usd-early",
    }])
    registry = pd.DataFrame({
        "date_norm": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")],
        "rate_usd": [85.0, 86.0],
        "rate_eur": [float("nan")] * 2,
        "rate_rub": [float("nan")] * 2,
        "rate_kzt": [float("nan")] * 2,
    })

    candidates = build_review_candidates(
        acts, etm, registry, R.load_config(None), {"agent a": "Agent A"})

    assert candidates["issue"].tolist() == ["possible_fx_difference"]
    assert "1 day before 1C" in candidates.iloc[0]["evidence"]
