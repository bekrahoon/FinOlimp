from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MATCHER = Path(__file__).resolve().parents[1] / "matcher"
sys.path.insert(0, str(MATCHER))

from build_stage2_training import _build_pair_table  # noqa: E402
from prepare_source_s3_train import _component_operation_date  # noqa: E402
import reconcile_s3 as S3  # noqa: E402
import evaluate_s3 as Eval  # noqa: E402
from s3_row_features import act_status_features  # noqa: E402
from source_features_s3 import SOURCE_FEATURES, build_source_features  # noqa: E402
from source_s3_model import fit_source_model  # noqa: E402
from temporal_split import (date_masks, pair_masks, select_cutoff,
                             select_month_cutoff)  # noqa: E402


def test_pair_train_is_built_from_current_rows_without_external_train_files():
    date = pd.Timestamp("2026-03-01")
    acts = pd.DataFrame([
        {"agent_norm": "A", "ticket_cell": "1111111111",
         "ticket_set": {"1111111111"}, "sum_1c": 100.0,
         "date_norm": date, "folder": "Agent A", "doc": "ticket A"},
        {"agent_norm": "B", "ticket_cell": "2222222222",
         "ticket_set": {"2222222222"}, "sum_1c": 150.0,
         "date_norm": date, "folder": "Agent B", "doc": "ticket B"},
        {"agent_norm": "B", "ticket_cell": "1111111111",
         "ticket_set": {"1111111111"}, "sum_1c": 100.0,
         "date_norm": date, "folder": "Agent B", "doc": "cross-agent ticket"},
        {"agent_norm": "A", "ticket_cell": float("nan"),
         "ticket_set": set(), "sum_1c": 200.0,
         "date_norm": date, "folder": "Agent A", "doc": "pay A"},
        {"agent_norm": "B", "ticket_cell": float("nan"),
         "ticket_set": set(), "sum_1c": 300.0,
         "date_norm": date, "folder": "Agent B", "doc": "pay B"},
    ], index=[0, 1, 2, 3, 4])
    etm = pd.DataFrame([
        {"agent_norm": "A", "tickets": "1111111111",
         "ticket_set": {"1111111111"}, "sum_etm": 100.0,
         "date_norm": date, "agent": "Agent A", "comment": "ticket A", "kind": "выкуп"},
        {"agent_norm": "B", "tickets": "2222222222",
         "ticket_set": {"2222222222"}, "sum_etm": 155.0,
         "date_norm": date, "agent": "Agent B", "comment": "ticket B", "kind": "выкуп"},
        {"agent_norm": "A", "tickets": float("nan"),
         "ticket_set": set(), "sum_etm": 200.0,
         "date_norm": date, "agent": "Agent A", "comment": "pay A", "kind": "оплата"},
        {"agent_norm": "B", "tickets": float("nan"),
         "ticket_set": set(), "sum_etm": 305.0,
         "date_norm": date, "agent": "Agent B", "comment": "pay B", "kind": "оплата"},
    ], index=[10, 11, 12, 13])
    cfg = {
        "tolerance": 0.01,
        "payments": {
            "prefer_doc_match": False,
            "date_window_days": 7,
            "doc_pattern": r"ЦБ-С\d+",
        },
    }

    train, report = _build_pair_table(acts, etm, cfg)

    assert ((train["kind"] == "ticket_clean") & (train["label_silver"] == 1)).any()
    assert ((train["kind"] == "neg_same_ticket_diff_sum")
            & (train["label_silver"] == 0)).any()
    assert ((train["kind"] == "neg_cross_agent_ticket")
            & (train["label_silver"] == 0)).any()
    assert ((train["kind"] == "pay_window") & (train["label_silver"] == 1)).any()
    assert report["label_provenance"].startswith("current-input Stage 1")
    assert len(train) == len(train[["a_idx", "e_idx"]].drop_duplicates())


def test_pair_model_fits_early_fold_and_scores_later_fold():
    train = pd.DataFrame([
        {"kind": "neg_same_ticket_diff_sum", "label_silver": 0,
         "a_tickets": "1111111111", "e_tickets": "1111111111",
         "a_amount": 100.0, "e_amount": 125.0,
         "a_date": "2026-01-01", "e_date": "2026-01-01",
         "a_agent": "A", "e_agent": "A", "a_text": "ticket", "e_text": "ticket",
         "e_kind": "выкуп"},
        {"kind": "neg_cross_agent_ticket", "label_silver": 0,
         "a_tickets": "2222222222", "e_tickets": "2222222222",
         "a_amount": 100.0, "e_amount": 100.0,
         "a_date": "2026-01-02", "e_date": "2026-01-02",
         "a_agent": "A", "e_agent": "B", "a_text": "ticket", "e_text": "ticket",
         "e_kind": "выкуп"},
    ])
    validation = pd.DataFrame([
        {"kind": "neg_same_ticket_diff_sum", "label_silver": 0,
         "a_tickets": "3333333333", "e_tickets": "3333333333",
         "a_amount": 100.0, "e_amount": 125.0,
         "a_date": "2026-03-01", "e_date": "2026-03-01",
         "a_agent": "A", "e_agent": "A", "a_text": "ticket", "e_text": "ticket",
         "e_kind": "выкуп"},
    ])

    _, report = S3._fit_pair_model(train, validation, ticket=True)

    assert report["component_membership_relabels_train"] == 1
    assert report["component_membership_relabels_validation"] == 1
    assert report["n_train"] == 2
    assert report["n_validation"] == 1
    assert report["validation_at_threshold"]["n"] == 1
    assert int(validation.loc[0, "label_silver"]) == 0


def test_empty_pair_train_uses_explicit_constant_negative_fallback():
    validation = pd.DataFrame([{
        "kind": "ticket_clean", "label_silver": 1,
        "a_tickets": "1111111111", "e_tickets": "1111111111",
        "a_amount": 100.0, "e_amount": 100.0,
        "a_date": "2026-03-01", "e_date": "2026-03-01",
        "a_agent": "A", "e_agent": "A", "a_text": "ticket", "e_text": "ticket",
        "e_kind": "выкуп",
    }])

    model, report = S3._fit_pair_model(validation.iloc[0:0], validation, ticket=True)

    assert list(model.classes_) == [0]
    assert report["n_train"] == 0
    assert report["empty_train_fallback"] == "constant-negative"


def test_temporal_cutoff_and_pair_split_exclude_cross_boundary_edges():
    cutoff = select_cutoff(pd.date_range("2026-01-01", periods=5), 0.4)
    pairs = pd.DataFrame([
        {"a_date": "2026-01-01", "e_date": "2026-01-02"},
        {"a_date": "2026-01-04", "e_date": "2026-01-05"},
        {"a_date": "2026-01-03", "e_date": "2026-01-04"},
    ])

    train, validation, excluded = pair_masks(pairs, cutoff)

    assert cutoff == pd.Timestamp("2026-01-04")
    assert train.tolist() == [True, False, False]
    assert validation.tolist() == [False, True, False]
    assert excluded.tolist() == [False, False, True]


def test_month_cutoff_is_first_day_of_earliest_validation_month():
    dates = pd.to_datetime(["2026-01-15", "2026-06-10", "2026-07-31"])

    assert select_month_cutoff(dates, 2) == pd.Timestamp("2026-06-01")
    assert select_month_cutoff(dates, 1) == pd.Timestamp("2026-07-01")

    train, validation, unknown = date_masks(dates, "2026-06-01")

    assert train.tolist() == [True, False, False]
    assert validation.tolist() == [False, True, True]
    assert unknown.tolist() == [False, False, False]


def test_source_component_with_future_member_is_assigned_to_validation():
    acts = pd.DataFrame({"date_norm": [pd.Timestamp("2026-01-10")]}, index=[0])
    etm = pd.DataFrame({"date_norm": [pd.Timestamp("2026-03-01")]}, index=[1])
    row = pd.Series({"a_indices": "[0]", "e_indices": "[1]",
                     "date": "2026-01-10"})

    assert _component_operation_date(row, acts, etm) == "2026-03-01"


def test_status_group_context_does_not_include_future_dates():
    acts = pd.DataFrame([
        {"folder": "A", "period_start": "P", "date": "2026-01-01",
         "act_status": "черновик"},
        {"folder": "A", "period_start": "P", "date": "2026-01-02",
         "act_status": "переиздан"},
        {"folder": "A", "period_start": "P", "date": "2026-01-03",
         "act_status": ""},
    ])

    features = act_status_features(acts)

    assert features["group_row_count"].tolist() == [0.0, 1.0, 2.0]
    assert features["group_draft_count"].tolist() == [0.0, 1.0, 1.0]
    assert features["group_reissue_count"].tolist() == [0.0, 0.0, 1.0]
    assert features["group_has_reissue"].tolist() == [0.0, 0.0, 1.0]


def test_source_registry_features_ignore_records_after_discrepancy_date():
    discrepancies = pd.DataFrame([{
        "row_type": "ticket", "date": "2026-01-05", "sum_1c": 100.0,
        "sum_etm": 90.0, "diff": 10.0, "identifier": "1111111111",
        "a_indices": "[0]", "e_indices": "[0]",
    }])
    acts = pd.DataFrame([{
        "date_norm": pd.Timestamp("2026-01-05"), "ticket_cell": "1111111111",
        "folder": "Agent A", "doc": "doc",
    }])
    etm = pd.DataFrame([{
        "date_norm": pd.Timestamp("2026-01-05"), "tickets": "1111111111",
        "agent": "Agent A", "comment": "comment", "kind": "выкуп",
    }])
    registry = pd.DataFrame([
        {"date": "2026-01-01", "tickets": "1111111111", "party": "old",
         "employee": "", "pnr": "", "route": "", "kind": "продажа",
         "airline": "", "pay_cell": ""},
        {"date": "2026-01-10", "tickets": "1111111111", "party": "future",
         "employee": "", "pnr": "", "route": "", "kind": "продажа",
         "airline": "", "pay_cell": ""},
    ])

    features = build_source_features(discrepancies, acts, etm, registry)

    assert features.loc[0, "n_registry_rows"] == 1
    assert features.loc[0, "registry_party_category"] == "old"


def test_evaluator_marks_temporal_silver_validation_without_claiming_truth():
    report = {"discrepancies": {
        "source_agreement_temporal_holdout": 1.0,
        "source_temporal_holdout_rows": 610,
    }}

    result = Eval._attach_evaluation_scope(report, {
        "inference_boundary": {
            "independent_evaluation": False,
            "temporal_validation": True,
        }
    })

    assert result["evaluation_scope"]["independent_evaluation"] is False
    assert result["evaluation_scope"]["temporal_validation"] is True
    assert result["discrepancies"]["source_agreement_temporal_validation_vs_silver"] == 1.0
    assert result["discrepancies"]["source_temporal_validation_rows"] == 610
    assert result["discrepancies"]["source_agreement_temporal_holdout"] is None
    assert result["discrepancies"]["source_temporal_holdout_rows"] is None


def test_source_model_fits_only_the_explicit_run_local_training_table():
    train = pd.DataFrame({feature: [0.0, 1.0] for feature in SOURCE_FEATURES})
    train["label_source"] = ["1С", "бот"]

    model = fit_source_model(train)

    assert set(model.classes_) == {"1С", "бот"}
