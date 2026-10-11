#!/usr/bin/env python3
"""Run Stage 1 and build/train Stage 2 in isolated per-run folders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "stage1_rules" / "input"
RUNS_ROOT = ROOT / ".finolimp" / "manual_runs"
STAGE1_SCRIPT = ROOT / "stage1_rules" / "src" / "match.py"
STAGE2_SCRIPT = ROOT / "stage2_ml" / "src" / "matcher" / "reconcile_s3.py"
STAGE2_SRC = ROOT / "stage2_ml" / "src"
STAGE2_TRAIN_SCRIPT = ROOT / "stage2_ml" / "src" / "matcher" / "build_stage2_training.py"
STAGE2_SOURCE_TRAIN_SCRIPT = ROOT / "stage2_ml" / "src" / "matcher" / "prepare_source_s3_train.py"
RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


def _venv_python() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def _run_in_project_venv() -> int | None:
    """Re-launch with the installed project venv, or return an actionable error."""
    venv_python = _venv_python()
    if not venv_python.is_file():
        print(
            "Не найдено виртуальное окружение проекта. Сначала запустите "
            "install_windows.ps1 или install_ubuntu.sh.",
            file=sys.stderr,
        )
        return 2
    if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        return None
    completed = subprocess.run(
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)


def _command(label: str, command: list[str]) -> None:
    print(f"\n{label}", flush=True)
    previous_umask = os.umask(0o077) if os.name == "posix" else None
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        if previous_umask is not None:
            os.umask(previous_umask)


def _freeze_completed_run(run_dir: Path) -> None:
    """Make completed output owner-readable and read-only on POSIX systems."""
    if os.name != "posix":
        return
    paths = [run_dir, *run_dir.rglob("*")]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Неожиданная символическая ссылка в папке результатов.")
    for path in paths:
        if path.is_file():
            path.chmod(0o400)
    for path in sorted((p for p in paths if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o500)


def _localize_stage1_codes(reports_dir: Path) -> None:
    """Translate only check/issue values using the same dictionary as the MCP reports."""
    integration_root = ROOT / "stage3_ai" / "integrations" / "finolimp_mcp"
    if str(integration_root) not in sys.path:
        sys.path.insert(0, str(integration_root))
    from finolimp_mcp.service import _localize_report_code_values  # noqa: PLC0415

    _localize_report_code_values(reports_dir)


def _export_stage1_workbooks(reports_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Build the accountant-facing Stage 1 workbook and field dictionary."""
    integration_root = ROOT / "stage3_ai" / "integrations" / "finolimp_mcp"
    if str(integration_root) not in sys.path:
        sys.path.insert(0, str(integration_root))
    from finolimp_mcp.excel_reports import (  # noqa: PLC0415
        export_report_readme,
        export_reports_workbook,
    )
    from finolimp_mcp.settings import (  # noqa: PLC0415
        REPORT_FILES,
        REPORT_README,
        REPORT_WORKBOOK,
    )

    artifacts: dict[str, str] = {}
    warnings: list[str] = []
    for label, filename, exporter in (
        ("workbook", REPORT_WORKBOOK, export_reports_workbook),
        ("field_dictionary", REPORT_README, export_report_readme),
    ):
        output_path = reports_dir / filename
        try:
            exporter(reports_dir, REPORT_FILES, output_path)
            artifacts[label] = str(output_path)
        except Exception as exc:
            warnings.append(f"{filename}: {type(exc).__name__}")
    return artifacts, warnings


def _export_stage2_workbook(stage2_dir: Path, run_id: str) -> Path:
    """Build the separate Russian-language workbook for one Stage 2 run."""
    if str(STAGE2_SRC) not in sys.path:
        sys.path.insert(0, str(STAGE2_SRC))
    from reporting.export_stage2_xlsx import export_stage2_workbook  # noqa: PLC0415

    return export_stage2_workbook(stage2_dir, run_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 формирует правила сверки; Stage 2 строит per-run train из входов "
            "и Stage 1 и обучает ML-второе мнение. Результаты пишутся в новую папку."
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help="Папка с acts.csv, etm.csv, registry.csv и reconcile.yaml (по умолчанию stage1_rules/input).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Необязательный уникальный ID папок результатов; по умолчанию создаётся автоматически.",
    )
    parser.add_argument(
        "--stage1-reports", type=Path, default=None,
        help="Папка с отчётами завершённого Stage 1; обязательна для --stage2-only.",
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=0.2,
        help="Доля последних дат для temporal validation Stage 2 (по умолчанию 20%%).",
    )
    parser.add_argument(
        "--validation-months", type=int, default=None,
        help="Последние N полных календарных месяцев — validation Stage 2 "
             "(ровная граница на 1-е число; перекрывает --validation-fraction).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage1-only", action="store_true", help="Запустить только правила Stage 1.")
    group.add_argument(
        "--stage2-only", action="store_true",
        help="Stage 2 по входам и готовому отчёту Stage 1 (укажите --stage1-reports).",
    )
    args = parser.parse_args()

    if args.validation_months is not None:
        if args.validation_months < 1:
            parser.error("--validation-months должно быть не меньше 1.")
    elif not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction должна быть между 0 и 1.")
    if args.stage2_only and args.stage1_reports is None:
        parser.error("Для --stage2-only укажите --stage1-reports с отчётами Stage 1.")

    if not (ROOT / ".venv").is_dir() or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        reexec_result = _run_in_project_venv()
        if reexec_result is not None:
            return reexec_result

    input_dir = args.input_dir.expanduser().resolve()
    required_inputs = ("acts.csv", "etm.csv", "registry.csv")
    missing_inputs = [name for name in required_inputs if not (input_dir / name).is_file()]
    if missing_inputs:
        parser.error(f"В папке {input_dir} отсутствуют файлы: {', '.join(missing_inputs)}")

    run_id = args.run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if not RUN_ID_RE.fullmatch(run_id):
        parser.error("--run-id допускает только латинские буквы, цифры, дефис и подчёркивание (до 80 знаков).")
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        parser.error(f"Папка запуска уже существует и не будет перезаписана: {run_dir}")

    run_stage1 = not args.stage2_only
    run_stage2 = not args.stage1_only
    stage1_reference_dir = (
        args.stage1_reports.expanduser().resolve() if args.stage1_reports else None
    )
    if stage1_reference_dir is not None:
        missing_stage1_reports = [
            name for name in ("discrepancies.csv",)
            if not (stage1_reference_dir / name).is_file()
        ]
        if missing_stage1_reports:
            parser.error(
                f"В {stage1_reference_dir} отсутствуют отчёты Stage 1: "
                + ", ".join(missing_stage1_reports)
            )

    files_to_hash = [input_dir / name for name in required_inputs]
    config_path = input_dir / "reconcile.yaml"
    if config_path.is_file():
        files_to_hash.append(config_path)
    state_root = ROOT / ".finolimp"
    if state_root.is_symlink() or RUNS_ROOT.is_symlink():
        parser.error("Папки состояния не могут быть символическими ссылками.")
    if state_root.exists() and not state_root.is_dir():
        parser.error(f"Путь состояния не является папкой: {state_root}")
    state_root.mkdir(mode=0o700, exist_ok=True)
    RUNS_ROOT.mkdir(mode=0o700, exist_ok=True)
    if os.name == "posix":
        RUNS_ROOT.chmod(0o700)
    run_dir.mkdir(mode=0o700)
    if os.name == "posix":
        run_dir.chmod(0o700)
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "input_sha256": {path.name: _sha256(path) for path in files_to_hash},
        "stage1_reports": str(stage1_dir) if run_stage1 else None,
        "stage1_reference_reports": (
            str(stage1_reference_dir) if not run_stage1 and stage1_reference_dir else None
        ),
        "stage2_reports": str(stage2_dir) if run_stage2 else None,
        "stage2_training_data": "generated in this run from inputs and Stage 1 results",
        "stage2_temporal_validation_fraction": (
            None if args.validation_months is not None else args.validation_fraction
        ),
        "stage2_temporal_validation_months": args.validation_months,
        "stage2_note": (
            "Stage 2 fits on silver targets derived from this run's Stage 1 results; "
            "the latest dates are kept out of fit for temporal validation. Targets are "
            "still silver, not independent truth, and Stage 2 does not replace Stage 1 actions."
        ),
    }
    _write_manifest(manifest_path, manifest)

    try:
        if run_stage1:
            _command(
                "Запускаю детерминированную сверку Stage 1...",
                [
                    sys.executable, str(STAGE1_SCRIPT),
                    "--input", str(input_dir), "--out", str(stage1_dir),
                ],
            )
            try:
                workbook_artifacts, workbook_warnings = _export_stage1_workbooks(stage1_dir)
            except Exception as exc:
                workbook_artifacts = {}
                workbook_warnings = [f"Excel-отчёты: {type(exc).__name__}"]
            manifest["stage1_presentation_files"] = workbook_artifacts
            if workbook_warnings:
                manifest["presentation_warnings"] = workbook_warnings
                print(
                    "Предупреждение: не все Excel-книги Stage 1 созданы: "
                    + "; ".join(workbook_warnings),
                    file=sys.stderr,
                )
            try:
                _localize_stage1_codes(stage1_dir)
                manifest["localized_csv_values"] = [
                    "quality_report.csv:check", "review_candidates.csv:issue",
                ]
            except Exception as exc:
                manifest["csv_localization_warning"] = (
                    f"{type(exc).__name__}; непереведённые значения кодов сохранены."
                )
                print(
                    "Предупреждение: не все значения check/issue удалось перевести; "
                    "исходные коды оставлены без изменения.",
                    file=sys.stderr,
                )
        if run_stage2:
            stage1_reports_dir = stage1_dir if run_stage1 else stage1_reference_dir
            if stage1_reports_dir is None:
                raise ValueError("Не указаны отчёты Stage 1 для построения train Stage 2")
            training_dir = stage2_dir / "train"
            bootstrap_dir = stage2_dir / "bootstrap"
            _command(
                "Stage 2: формирую run-local train из входов и результатов Stage 1...",
                [
                    sys.executable, str(STAGE2_TRAIN_SCRIPT),
                    "--input-dir", str(input_dir),
                    "--stage1-reports", str(stage1_reports_dir),
                    "--output-dir", str(training_dir),
                ] + (
                    ["--validation-months", str(args.validation_months)]
                    if args.validation_months is not None
                    else ["--validation-fraction", str(args.validation_fraction)]
                ),
            )
            _command(
                "Stage 2: обучаю match-модели и строю промежуточные ML-компоненты...",
                [
                    sys.executable, str(STAGE2_SCRIPT),
                    "--acts", str(input_dir / "acts.csv"),
                    "--etm", str(input_dir / "etm.csv"),
                    "--registry", str(input_dir / "registry.csv"),
                    "--output-prefix", "bootstrap",
                    "--output-dir", str(bootstrap_dir),
                    "--training-dir", str(training_dir),
                    "--skip-source",
                ],
            )
            _command(
                "Stage 2: готовлю source-train по меткам отчёта Stage 1...",
                [
                    sys.executable, str(STAGE2_SOURCE_TRAIN_SCRIPT),
                    "--input-dir", str(input_dir),
                    "--stage1-reports", str(stage1_reports_dir),
                    "--stage2-discrepancies",
                    str(bootstrap_dir / "bootstrap_discrepancies_ml.csv"),
                    "--output-dir", str(training_dir),
                ],
            )
            source_report = json.loads(
                (training_dir / "source_s3_train_report.json").read_text(encoding="utf-8")
            )
            skip_source = not bool(source_report.get("train_rows", 0))
            if skip_source:
                manifest["stage2_source_model"] = "skipped: no Stage 1 source labels in train fold"
                print(
                    "Предупреждение: source-модель Stage 2 пропущена — "
                    "в ранней temporal train-части нет сопоставленных меток Stage 1.",
                    file=sys.stderr,
                )
            manifest["stage2_status"] = "running"
            _command(
                "Stage 2: обучаю модели на run-local train и выполняю ML-сверку...",
                [
                    sys.executable, str(STAGE2_SCRIPT),
                    "--acts", str(input_dir / "acts.csv"),
                    "--etm", str(input_dir / "etm.csv"),
                    "--registry", str(input_dir / "registry.csv"),
                    "--output-prefix", "stage2",
                    "--output-dir", str(stage2_dir),
                    "--training-dir", str(training_dir),
                ] + (["--skip-source"] if skip_source else []),
            )
            inference_report = json.loads(
                (stage2_dir / "stage2_inference_report.json").read_text(encoding="utf-8")
            )
            try:
                stage2_workbook = _export_stage2_workbook(stage2_dir, run_id)
                manifest["stage2_presentation_files"] = {
                    "model_report": str(stage2_workbook),
                }
            except Exception as exc:
                manifest["stage2_presentation_warnings"] = [
                    f"{type(exc).__name__}: не удалось создать Excel-отчёт Stage 2",
                ]
                print(
                    "Предупреждение: не удалось создать русскоязычный Excel-отчёт Stage 2 "
                    f"({type(exc).__name__}).",
                    file=sys.stderr,
                )
            manifest["stage2_temporal_split"] = inference_report.get("temporal_split")
            manifest["stage2_validation"] = {
                "status_rows": inference_report.get("status_model", {}).get(
                    "validation_vs_silver"),
                "ticket_pairs": inference_report.get("ticket_model", {}).get(
                    "validation_at_threshold"),
                "payment_pairs": inference_report.get("payment_model", {}).get(
                    "validation_at_threshold"),
                "source_rows": inference_report.get("source_model", {}).get(
                    "validation_vs_silver"),
            }
            manifest["stage2_status"] = "completed"
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        _freeze_completed_run(run_dir)
    except (KeyboardInterrupt, Exception) as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise

    print(f"\nГотово. ID запуска: {run_id}")
    print(f"Папка результатов: {run_dir}")
    if run_stage1:
        print(f"Отчёты Stage 1: {stage1_dir}")
    if run_stage2:
        print(f"Отчёты Stage 2 (второе мнение): {stage2_dir}")
        stage2_workbook = manifest.get("stage2_presentation_files", {}).get("model_report")
        if stage2_workbook:
            print(f"Excel-отчёт моделей Stage 2: {stage2_workbook}")
    print(f"Манифест: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
