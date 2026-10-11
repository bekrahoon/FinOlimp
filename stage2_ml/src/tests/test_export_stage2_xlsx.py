from __future__ import annotations

import json
import sys
from pathlib import Path

from openpyxl import load_workbook

REPORTING = Path(__file__).resolve().parents[1] / "reporting"
sys.path.insert(0, str(REPORTING.parent))

from reporting.export_stage2_xlsx import (  # noqa: E402
    REPORT_NAME,
    _validation_rows,
    export_stage2_workbook,
)


def test_exports_russian_stage2_model_report_with_validation_details(tmp_path):
    stage2 = tmp_path / "stage2"
    training_dir = stage2 / "train"
    training_dir.mkdir(parents=True)
    inference = {
        "cutoff": "2026-06-19",
        "temporal_split": {
            "cutoff": "2026-06-19",
            "validation_fraction_requested": 0.2,
            "unique_dates": 212,
            "status_rows": {"train": 14253, "validation": 4800},
            "pair_rows": {"train": 23468, "validation": 6717,
                          "cross_period_or_unknown_date_excluded": 2276},
        },
        "status_model": {
            "classifier": "HistGradientBoostingClassifier",
            "validation_vs_silver": {"n": 4800, "accuracy": 1.0,
                                      "keep_precision": 0.99, "keep_recall": 0.98,
                                      "threshold": 0.5},
        },
        "ticket_model": {
            "classifier": "HistGradientBoostingClassifier",
            "validation_at_threshold": {
                "n": 6204, "threshold": 0.5, "precision_vs_silver_target": 0.999,
                "recall_vs_silver_target": 1.0, "false_positives": 2,
                "false_negatives": 0,
            },
            "candidate_gate_silver_validation": {
                "ticket_positive_pairs": 5928,
                "ticket_exact_tail_gate_covered": 5924,
                "ticket_exact_tail_gate_recall_silver": 0.999325,
                "payment_positive_pairs": 421,
                "payment_amount_date_gate_covered": 421,
                "payment_amount_date_gate_recall_silver": 1.0,
            },
        },
        "payment_model": {
            "classifier": "DummyClassifier",
            "validation_at_threshold": {
                "n": 513, "threshold": 0.5, "precision_vs_silver_target": 1.0,
                "recall_vs_silver_target": 1.0, "false_positives": 0,
                "false_negatives": 0,
            },
        },
        "source_model": {
            "classifier": "HistGradientBoostingClassifier",
            "classes": ["1С", "агент", "бот", "выяснить"],
            "training_rows": 1770,
            "validation_rows": 392,
            "validation_vs_silver": {
                "n": 392, "accuracy": 0.897959, "macro_f1": 0.696746,
                "per_class": {
                    "1С": {"support": 126, "precision": 0.961538,
                          "recall": 0.992063, "f1": 0.976562},
                    "агент": {"support": 2, "precision": 0.071429,
                              "recall": 0.5, "f1": 0.125},
                    "бот": {"support": 38, "precision": 0.649123,
                            "recall": 0.973684, "f1": 0.778947},
                    "выяснить": {"support": 226, "precision": 0.989529,
                                 "recall": 0.836283, "f1": 0.906475},
                },
                "confusion": [[125, 0, 1, 0], [0, 1, 0, 1],
                              [0, 0, 37, 1], [5, 13, 19, 189]],
            },
        },
        "output": {"discrepancy_rows": 2169},
    }
    training = {
        "status": {"n_train": 14253, "n_validation": 4800},
        "pairs": {"temporal_split": {
            "train": {"rows": 23468}, "validation": {"rows": 6717},
            "cross_period_or_unknown_date_excluded": 2276,
        }},
    }
    (stage2 / "stage2_inference_report.json").write_text(
        json.dumps(inference, ensure_ascii=False), encoding="utf-8")
    (training_dir / "training_report.json").write_text(
        json.dumps(training, ensure_ascii=False), encoding="utf-8")

    output = export_stage2_workbook(stage2, "=1+1")

    assert output.name == REPORT_NAME
    workbook = load_workbook(output, data_only=False)
    assert workbook.sheetnames == [
        "Сводка", "Валидация", "Покрытие кандидатов", "По классам источника",
        "Матрица ошибок", "Пояснения",
    ]
    assert workbook["Сводка"]["B2"].value == "=1+1"
    assert workbook["Сводка"]["B2"].data_type == "s"
    assert workbook["Валидация"]["A3"].value == "Пары билетов"
    assert workbook["Валидация"]["E3"].value == 0.999
    assert workbook["Валидация"]["E3"].number_format == "0.00%"
    assert workbook["По классам источника"]["B3"].value == 2
    assert workbook["Матрица ошибок"]["E5"].value == 189
    assert any("не независимую точность" in str(row[1].value)
               for row in workbook["Пояснения"].iter_rows(min_row=1))


def test_validation_sheet_remains_rectangular_when_source_model_is_skipped():
    rows = _validation_rows({
        "status_model": {},
        "ticket_model": {},
        "payment_model": {},
        "source_model": {"skipped_for_bootstrap": True},
    })

    assert len(rows) == 4
    assert all(len(row) == 10 for row in rows)
    assert "не обучалась" in rows[-1][-1]
