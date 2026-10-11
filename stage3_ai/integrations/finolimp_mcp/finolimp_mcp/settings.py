from pathlib import Path


STAGE3_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = (
    STAGE3_ROOT.parent
    if (STAGE3_ROOT.parent / "stage1_rules" / "src").is_dir()
    else STAGE3_ROOT
)
STAGE1_SRC = PROJECT_ROOT / "stage1_rules" / "src"
CURRENT_INPUT = PROJECT_ROOT / "stage1_rules" / "input"
RUNTIME_ROOT = STAGE3_ROOT / ".finolimp" / "runtime"

INPUT_FILES = ("acts.csv", "etm.csv", "registry.csv")
OPTIONAL_INPUT_FILES = ("reconcile.yaml",)
REPORT_FILES = (
    "discrepancies.csv",
    "reconciliation_summary.csv",
    "quality_report.csv",
    "review_candidates.csv",
    "agent_actions.csv",
    "agent_risk.csv",
    "employee_risk.csv",
    "features.csv",
    "label_map.csv",
)
REPORT_WORKBOOK = "finolimp_reports.xlsx"
REPORT_README = "README.xlsx"
