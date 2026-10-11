"""Stage 2 entry-point boundary checks that need no saved run artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

MATCHER = Path(__file__).resolve().parents[1] / "matcher"
sys.path.insert(0, str(MATCHER))

import reconcile_s3 as S3  # noqa: E402


def test_s3_inference_entry_points_pass_rules_boundary_checks():
    S3._purity_check()
