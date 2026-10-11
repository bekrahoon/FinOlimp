"""Purity granitsy etapov (sm. stage2_ml/docs/EVAL_PROTOCOL.md).

Etap 1 — tolko rules: v stage1_rules/src net ML-importov.
io_clean — tolko plumbing: ni odnogo importa iz parse/reconcile2.
Stage 2 inference purity separately checked by test_reconcile_s3.py.
"""
from __future__ import annotations

import re
from pathlib import Path

SS = Path(__file__).resolve().parent.parent
S2SRC = SS.parent.parent / "stage2_ml" / "src"

ML_IMPORTS = ("sklearn", "torch", "transformers", "catboost",
              "lightgbm", "xgboost", "tensorflow", "keras")
RULES_MODULES = ("reconcile2", "pairs", "source_for_comment", "ticket_components",
                 "match_payments_with_pairs", "resolve_registry_party",
                 "canonical_labels", "apply_aliases", "norm_agent")


def _imported_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_.]+)", line)
        if m:
            names.add(m.group(1).split(".")[0])
    return names


def _uses_name(path: Path, name: str) -> bool:
    return re.search(r"\b" + re.escape(name) + r"\b",
                     path.read_text(encoding="utf-8")) is not None


def test_stage1_has_no_ml_imports():
    """Etap 1 — tolko rules."""
    offenders = []
    for path in sorted(SS.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        for mod in ML_IMPORTS:
            if mod in _imported_names(path):
                offenders.append(f"{path.name}: {mod}")
    assert not offenders, offenders


def test_io_clean_imports_no_rules():
    """Plumbing ne znaet pro rules (ni importom, ni ispolzovaniem)."""
    path = SS / "io_clean.py"
    assert "parse" not in _imported_names(path)
    assert "reconcile2" not in _imported_names(path)
    offenders = [n for n in RULES_MODULES if _uses_name(path, n)]
    assert not offenders, offenders


def test_parse_still_reexports_plumbing():
    """Split ne slomal sovmestimost: from parse import X == from io_clean import X."""
    import sys
    sys.path.insert(0, str(SS))
    import parse as P
    import io_clean as C
    for name in ("QualityLog", "read_csv_smart", "strip_string_cells",
                 "parse_money_value", "parse_money_series", "strip_timezone",
                 "parse_date_series", "ticket_numbers", "TICKETS_DEFAULT"):
        assert getattr(P, name) is getattr(C, name), name
