"""Small, bounded application service behind the FinOlimp MCP tools.

Model-visible inputs are IDs and filters, never filesystem paths or commands.
Each dataset is a snapshot and every reconciliation writes to a new run folder.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .excel_reports import export_report_readme, export_reports_workbook
from .report_dictionary import (
    QUALITY_CHECK_MEANINGS,
    QUALITY_CHECK_TITLES,
    REVIEW_ISSUE_MEANINGS,
    REVIEW_ISSUE_TITLES,
    quality_check_meaning,
    quality_check_title,
    review_issue_meaning,
    review_issue_title,
    review_issue_title,
)
from .settings import (
    CURRENT_INPUT,
    INPUT_FILES,
    OPTIONAL_INPUT_FILES,
    PROJECT_ROOT,
    REPORT_FILES,
    REPORT_README,
    REPORT_WORKBOOK,
    RUNTIME_ROOT,
    STAGE1_SRC,
)

DATASET_ID_RE = re.compile(r"^ds-[0-9a-f]{32}$")
RUN_ID_RE = re.compile(r"^run-[0-9a-f]{32}$")
CASE_ID_RE = re.compile(r"^case-[0-9a-f]{32}$")
DISCREPANCY_ID_RE = re.compile(r"^dsp-[0-9]{1,9}$")
MAX_PAGE_SIZE = 100
PIPELINE_TIMEOUT_SECONDS = 15 * 60
SOURCE_LABELS = {"бот", "1С", "агент", "выяснить"}
QUALITY_MEANINGS = QUALITY_CHECK_MEANINGS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_private_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Runtime-каталог не может быть символической ссылкой.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)


def _freeze_tree(path: Path) -> None:
    """Make a completed snapshot/run read-only for ordinary accidental edits."""
    if os.name != "posix":
        return
    paths = [path, *path.rglob("*")]
    for item in paths:
        if item.is_symlink():
            raise ValueError("Неожиданная символическая ссылка в runtime-данных.")
        item.chmod(0o500 if item.is_dir() else 0o400)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _ensure_private_dir(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if os.name == "posix":
        temp.chmod(0o600)
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Не удалось прочитать запись запуска или набора данных.") from exc
    if not isinstance(value, dict):
        raise ValueError("Повреждена запись запуска или набора данных.")
    return value


def _validate_id(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"Некорректный идентификатор {label}.")
    return value


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _dataset_dir(dataset_id: str, runtime_root: Path = RUNTIME_ROOT) -> Path:
    dataset_id = _validate_id(dataset_id, DATASET_ID_RE, "набора данных")
    root = runtime_root / "datasets"
    directory = root / dataset_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Набор данных не найден. Сначала зарегистрируйте входные файлы.")
    if directory.resolve().parent != root.resolve():
        raise ValueError("Небезопасное расположение набора данных.")
    return directory


def _run_dir(run_id: str, runtime_root: Path = RUNTIME_ROOT) -> Path:
    run_id = _validate_id(run_id, RUN_ID_RE, "запуска")
    root = runtime_root / "runs"
    directory = root / run_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Запуск не найден.")
    if directory.resolve().parent != root.resolve():
        raise ValueError("Небезопасное расположение запуска.")
    return directory


def _load_dataset(dataset_id: str, runtime_root: Path = RUNTIME_ROOT) -> tuple[Path, dict[str, Any]]:
    directory = _dataset_dir(dataset_id, runtime_root)
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError("Идентификатор набора данных не совпадает с записью.")
    return directory, manifest


def register_current_dataset(
    input_root: Path = CURRENT_INPUT,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    """Snapshot the fixed, configured input directory. No caller-supplied paths."""
    source_root = input_root.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    selected = list(INPUT_FILES)
    for name in selected:
        source = source_root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Не найден обязательный файл выгрузки: {name}.")
    for name in OPTIONAL_INPUT_FILES:
        source = source_root / name
        if source.exists() and (source.is_symlink() or not source.is_file()):
            raise ValueError(f"Файл настроек {name} имеет небезопасный формат.")
        if source.is_file():
            selected.append(name)

    datasets_root = runtime_root / "datasets"
    _ensure_private_dir(runtime_root)
    _ensure_private_dir(datasets_root)
    dataset_id = f"ds-{uuid.uuid4().hex}"
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=datasets_root))
    final = datasets_root / dataset_id
    try:
        for name in selected:
            source = source_root / name
            destination = staging / name
            shutil.copyfile(source, destination)
            if os.name == "posix":
                destination.chmod(0o600)
            digest, size = _sha256(destination)
            files[name] = {"sha256": digest, "size_bytes": size}
        _write_json(staging / "manifest.json", {
            "dataset_id": dataset_id,
            "created_at": _now(),
            "files": files,
        })
        os.replace(staging, final)
        _freeze_tree(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "dataset_id": dataset_id,
        "created_at": _read_json(final / "manifest.json")["created_at"],
        "files": files,
        "message": "Создан неизменяемый snapshot выгрузок для этой сверки.",
    }


def validate_dataset(dataset_id: str, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    """Run Stage 1's input parser without producing reconciliation reports."""
    directory, manifest = _load_dataset(dataset_id, runtime_root)
    missing = [name for name in INPUT_FILES if name not in manifest.get("files", {})]
    if missing:
        return {"dataset_id": dataset_id, "ready": False, "problems": [
            f"В snapshot отсутствуют файлы: {', '.join(missing)}"
        ]}
    recorded_files = manifest.get("files", {})
    allowed_files = {*INPUT_FILES, *OPTIONAL_INPUT_FILES}
    unknown_files = sorted(set(recorded_files) - allowed_files)
    if unknown_files:
        return {"dataset_id": dataset_id, "ready": False,
                "problems": ["В записи snapshot есть неизвестные файлы."]}
    for name, recorded in recorded_files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return {"dataset_id": dataset_id, "ready": False,
                    "problems": [f"Файл snapshot недоступен: {name}."]}
        digest, size = _sha256(path)
        if not isinstance(recorded, dict) or digest != recorded.get("sha256") \
                or size != recorded.get("size_bytes"):
            return {"dataset_id": dataset_id, "ready": False,
                    "problems": [f"Snapshot изменился после регистрации: {name}. Создайте новый snapshot."]}

    stage_src = str(STAGE1_SRC)
    if stage_src not in sys.path:
        sys.path.insert(0, stage_src)
    import parse as stage1_parse  # noqa: PLC0415

    config_path = directory / "reconcile.yaml"
    cfg = stage1_parse.load_config(config_path if config_path.exists() else None)
    qlog = stage1_parse.QualityLog()
    acts, etm, registry, ignored_drafts = stage1_parse.load_data(
        directory / "acts.csv",
        directory / "etm.csv",
        directory / "registry.csv",
        cfg,
        qlog,
    )

    def date_range(frame: Any) -> dict[str, str] | None:
        if "date_norm" not in frame.columns:
            return None
        values = frame["date_norm"].dropna()
        if values.empty:
            return None
        return {"from": values.min().date().isoformat(), "to": values.max().date().isoformat()}

    return {
        "dataset_id": dataset_id,
        "ready": True,
        "row_counts": {
            "acts": int(len(acts)),
            "etm": int(len(etm)),
            "registry": int(len(registry)),
            "ignored_draft_acts": int(sum(ignored_drafts.values())),
        },
        "date_ranges": {
            "acts": date_range(acts),
            "etm": date_range(etm),
            "registry": date_range(registry),
        },
        "quality_observations": qlog.summary(),
        "message": "Структура и формат выгрузок приняты движком Stage 1. Это ещё не результат сверки.",
    }


def _pipeline_fingerprint() -> str:
    files = (
        "match.py", "reconcile2.py", "parse.py", "io_clean.py", "review_candidates.py",
        "build_features.py", "rank.py", "build_actions.py",
    )
    digest = hashlib.sha256()
    for name in files:
        path = STAGE1_SRC / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def run_reconciliation(
    dataset_id: str,
    runtime_root: Path = RUNTIME_ROOT,
    timeout_seconds: int = PIPELINE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run Stage 1 and create Russian accountant-facing reports."""
    dataset_dir, dataset_manifest = _load_dataset(dataset_id, runtime_root)
    validation = validate_dataset(dataset_id, runtime_root)
    if not validation.get("ready"):
        return {"status": "blocked", "dataset_id": dataset_id,
                "problems": validation.get("problems", [])}

    runs_root = runtime_root / "runs"
    _ensure_private_dir(runtime_root)
    _ensure_private_dir(runs_root)
    run_id = f"run-{uuid.uuid4().hex}"
    run_dir = runs_root / run_id
    reports_dir = run_dir / "reports"
    _ensure_private_dir(run_dir)
    _ensure_private_dir(reports_dir)
    started_at = _now()
    fingerprint = _pipeline_fingerprint()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_files": dataset_manifest.get("files", {}),
        "pipeline_fingerprint": fingerprint,
        "started_at": started_at,
        "status": "running",
        "reports": [],
    }
    _write_json(run_dir / "manifest.json", manifest)

    command = [
        sys.executable,
        str(STAGE1_SRC / "match.py"),
        "--input", str(dataset_dir),
        "--out", str(reports_dir),
    ]
    environment = os.environ.copy()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(STAGE1_SRC),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            output, _ = await process.communicate()
            manifest.update({
                "status": "failed",
                "finished_at": _now(),
                "error": f"Превышено время выполнения ({timeout_seconds} секунд).",
            })
            (run_dir / "pipeline.log").write_bytes(output[-200_000:])
            if os.name == "posix":
                (run_dir / "pipeline.log").chmod(0o600)
            _write_json(run_dir / "manifest.json", manifest)
            _freeze_tree(run_dir)
            return {"run_id": run_id, "dataset_id": dataset_id, "status": "failed",
                    "message": manifest["error"]}
        except asyncio.CancelledError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            manifest.update({"status": "cancelled", "finished_at": _now()})
            _write_json(run_dir / "manifest.json", manifest)
            _freeze_tree(run_dir)
            raise

        (run_dir / "pipeline.log").write_bytes(output[-200_000:])
        if os.name == "posix":
            (run_dir / "pipeline.log").chmod(0o600)
        if process.returncode != 0:
            manifest.update({
                "status": "failed",
                "finished_at": _now(),
                "error": "Движок не смог завершить сверку. Проверьте выгрузки или передайте запуск специалисту.",
            })
            _write_json(run_dir / "manifest.json", manifest)
            _freeze_tree(run_dir)
            return {"run_id": run_id, "dataset_id": dataset_id, "status": "failed",
                    "message": manifest["error"]}

        workbook_path = reports_dir / REPORT_WORKBOOK
        readme_path = reports_dir / REPORT_README
        presentation_warnings = []
        try:
            export_reports_workbook(reports_dir, REPORT_FILES, workbook_path)
        except Exception as exc:
            # XLSX is a convenience view; a presentation failure must not erase
            # or misclassify the completed deterministic CSV result.
            presentation_warnings.append(
                f"{REPORT_WORKBOOK} не создана ({type(exc).__name__})."
            )
        try:
            export_report_readme(reports_dir, REPORT_FILES, readme_path)
        except Exception as exc:
            presentation_warnings.append(
                f"{REPORT_README} не создана ({type(exc).__name__})."
            )
        try:
            _localize_report_code_values(reports_dir)
        except Exception as exc:
            presentation_warnings.append(
                "Не удалось перевести значения кодов в CSV на русский "
                f"({type(exc).__name__}); исходные коды сохранены."
            )
        presentation_warning = " ".join(presentation_warnings) or None

        available_reports = [
            name for name in REPORT_FILES if (reports_dir / name).is_file()
        ]
        available_reports.extend(
            name for name, path in (
                (REPORT_WORKBOOK, workbook_path),
                (REPORT_README, readme_path),
            ) if path.is_file()
        )
        manifest.update({
            "status": "completed",
            "finished_at": _now(),
            "reports": available_reports,
        })
        if presentation_warning:
            manifest["presentation_warning"] = presentation_warning
        _write_json(run_dir / "manifest.json", manifest)
        summary = get_run_summary(run_id, runtime_root)
        _freeze_tree(run_dir)
        return {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "status": "completed",
            "pipeline_fingerprint": fingerprint,
            "summary": summary,
            "available_reports": available_reports,
            "report_workbook": REPORT_WORKBOOK if workbook_path.is_file() else None,
            "report_readme": REPORT_README if readme_path.is_file() else None,
            "presentation_warning": presentation_warning,
            "message": "Сверка завершена. Кандидаты на ручную проверку не являются готовыми бухгалтерскими выводами.",
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        manifest.update({"status": "failed", "finished_at": _now(),
                         "error": f"Ошибка запуска движка: {type(exc).__name__}."})
        _write_json(run_dir / "manifest.json", manifest)
        _freeze_tree(run_dir)
        return {"run_id": run_id, "dataset_id": dataset_id, "status": "failed",
                "message": manifest["error"]}


def _completed_run(run_id: str, runtime_root: Path = RUNTIME_ROOT) -> tuple[Path, dict[str, Any]]:
    directory = _run_dir(run_id, runtime_root)
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("run_id") != run_id:
        raise ValueError("Идентификатор запуска не совпадает с записью.")
    if manifest.get("status") != "completed":
        raise ValueError("Этот запуск не завершился успешно.")
    return directory, manifest


def _restore_report_code(filename: str, column: str, value: str) -> str:
    if filename == "quality_report.csv" and column == "check":
        meanings_to_codes = {meaning: code for code, meaning in QUALITY_CHECK_MEANINGS.items()}
        titles_to_codes = {title: code for code, title in QUALITY_CHECK_TITLES.items()}
        if value in meanings_to_codes:
            return meanings_to_codes[value]
        if value in titles_to_codes:
            return titles_to_codes[value]
        dynamic_prefix = "В поле ETM «"
        dynamic_suffix = "» обнаружено значение, не предусмотренное правилами сверки. Проверьте исходную запись."
        if value.startswith(dynamic_prefix) and value.endswith(dynamic_suffix):
            column_name = value[len(dynamic_prefix):-len(dynamic_suffix)]
            return f"unknown_etm_{column_name}"
    elif filename == "review_candidates.csv" and column == "issue":
        meanings_to_codes = {meaning: code for code, meaning in REVIEW_ISSUE_MEANINGS.items()}
        titles_to_codes = {title: code for code, title in REVIEW_ISSUE_TITLES.items()}
        if value in meanings_to_codes:
            return meanings_to_codes[value]
        if value in titles_to_codes:
            return titles_to_codes[value]
    return value


def _localize_report_code_values(reports_dir: Path) -> None:
    """Translate only quality/check and review/issue cell values, never headers."""
    targets = (
        ("quality_report.csv", "check", QUALITY_CHECK_MEANINGS, quality_check_meaning),
        ("review_candidates.csv", "issue", REVIEW_ISSUE_MEANINGS, review_issue_meaning),
    )
    staged: list[tuple[Path, Path]] = []
    try:
        for filename, column, meanings, explain in targets:
            path = reports_dir / filename
            if not path.is_file():
                continue
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=reports_dir,
            )
            temporary = Path(temporary_name)
            staged.append((temporary, path))
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as source, \
                        os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as target:
                    reader = csv.DictReader(source)
                    fieldnames = reader.fieldnames
                    if not fieldnames or column not in fieldnames:
                        raise ValueError(f"В отчёте {filename} отсутствует поле {column}.")
                    writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
                    writer.writeheader()
                    for row in reader:
                        code = _restore_report_code(filename, column, row.get(column, ""))
                        if code not in meanings \
                                and not (filename == "quality_report.csv" and code.startswith("unknown_etm_")):
                            raise ValueError(f"Для значения {code!r} в {filename} нет русского пояснения.")
                        row[column] = explain(code)
                        writer.writerow(row)
            except Exception:
                # fdopen owns the descriptor after entering its context; close it
                # here only when opening failed before ownership transferred.
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise

        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"В отчёте запуска отсутствует файл {path.name}.")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for column in ("check", "issue"):
            if column in row:
                row[column] = _restore_report_code(path.name, column, row[column])
    return rows


def _report(run_dir: Path, name: str) -> Path:
    if name not in REPORT_FILES:
        raise ValueError("Запрошен неизвестный тип отчёта.")
    path = run_dir / "reports" / name
    if path.resolve().parent != (run_dir / "reports").resolve():
        raise ValueError("Небезопасное расположение отчёта.")
    return path


def list_runs(limit: int = 20, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit должен быть от 1 до {MAX_PAGE_SIZE}.")
    root = runtime_root / "runs"
    results = []
    if root.exists():
        for path in root.iterdir():
            if not RUN_ID_RE.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            try:
                manifest = _read_json(path / "manifest.json")
            except ValueError:
                continue
            results.append({key: manifest.get(key) for key in
                            ("run_id", "dataset_id", "status", "started_at", "finished_at")})
    results.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    results = results[:limit]
    return {"runs": results, "count": len(results)}


def get_run_summary(run_id: str, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    run_dir, manifest = _completed_run(run_id, runtime_root)
    discrepancies = _read_rows(_report(run_dir, "discrepancies.csv"))
    summary = _read_rows(_report(run_dir, "reconciliation_summary.csv"))
    candidates = _read_rows(_report(run_dir, "review_candidates.csv"))
    quality = _read_rows(_report(run_dir, "quality_report.csv"))
    actions = _read_rows(_report(run_dir, "agent_actions.csv"))

    by_source: dict[str, int] = {}
    for row in discrepancies:
        key = row.get("source", "") or "не указано"
        by_source[key] = by_source.get(key, 0) + 1
    residual_agents = []
    for row in summary:
        try:
            residual = float(row.get("residual", "nan"))
        except (TypeError, ValueError):
            residual = float("nan")
        if residual == residual and abs(residual) > 0.01:
            residual_agents.append({"agent": row.get("agent", ""), "residual": row.get("residual", "")})
    candidate_types: dict[str, int] = {}
    for row in candidates:
        key = row.get("issue", "не указано")
        candidate_types[key] = candidate_types.get(key, 0) + 1
    quality_counts: dict[str, int] = {}
    for row in quality:
        try:
            count = int(float(row.get("count", "0")))
        except (TypeError, ValueError):
            count = 0
        key = row.get("check", "не указано")
        quality_counts[key] = quality_counts.get(key, 0) + count
    quality_observations = [
        {
            "check": quality_check_title(check),
            "check_code": check,
            "count": count,
            "meaning": quality_check_meaning(check),
        }
        for check, count in sorted(quality_counts.items())
    ]

    def sum_action(field: str) -> float:
        total = 0.0
        for row in actions:
            try:
                value = float(row.get(field, "0"))
            except (TypeError, ValueError):
                continue
            if value == value:
                total += value
        return round(total, 2)

    action_totals = {
        "agents_with_actions": len(actions),
        "move_etm_by": sum_action("move_etm_by"),
        "fix_1c_by": sum_action("fix_1c_by"),
        "clarify_money": sum_action("clarify_money"),
        "clarify_rows": sum_action("clarify_n"),
        "notice": "Это расчётные поручения Stage 1, а не автоматически проведённые операции.",
    }

    return {
        "run_id": run_id,
        "dataset_id": manifest.get("dataset_id"),
        "pipeline_fingerprint": manifest.get("pipeline_fingerprint"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "discrepancy_count": len(discrepancies),
        "discrepancies_by_source": by_source,
        "proposed_actions": action_totals,
        "agent_count": len(summary),
        "agents_with_nonzero_residual": residual_agents,
        "manual_review_candidate_count": len(candidates),
        "manual_review_candidates_by_type": candidate_types,
        "manual_review_candidate_types": [
            {
                "reason": review_issue_title(issue),
                "count": count,
                "meaning": review_issue_meaning(issue),
            }
            for issue, count in sorted(candidate_types.items())
        ],
        "quality_observations": quality_observations,
        "available_reports": manifest.get("reports", []),
        "report_workbook": (
            REPORT_WORKBOOK if REPORT_WORKBOOK in manifest.get("reports", []) else None
        ),
        "report_readme": (
            REPORT_README if REPORT_README in manifest.get("reports", []) else None
        ),
        "presentation_warning": manifest.get("presentation_warning"),
    }


def list_quality_findings(
    run_id: str,
    check: str | None = None,
    limit: int = 25,
    offset: int = 0,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    if not 1 <= limit <= MAX_PAGE_SIZE or not 0 <= offset <= 100_000:
        raise ValueError(f"Используйте limit 1–{MAX_PAGE_SIZE} и offset 0–100000.")
    run_dir, _ = _completed_run(run_id, runtime_root)
    rows = _read_rows(_report(run_dir, "quality_report.csv"))
    selected = []
    for index, row in enumerate(rows, start=1):
        if check and row.get("check") != check:
            continue
        selected.append({
            "finding_id": f"quality-{index:06d}",
            **row,
            "check_title": quality_check_title(row.get("check", "")),
            "meaning": quality_check_meaning(row.get("check", "")),
        })
    return {
        "run_id": run_id,
        "total": len(selected),
        "offset": offset,
        "limit": limit,
        "findings": selected[offset:offset + limit],
        "notice": "Это диагностические наблюдения, не бухгалтерские ошибки и не проводки.",
    }


def list_discrepancies(
    run_id: str,
    source: str | None = None,
    agent: str | None = None,
    search: str | None = None,
    limit: int = 25,
    offset: int = 0,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    if not 1 <= limit <= MAX_PAGE_SIZE or not 0 <= offset <= 100_000:
        raise ValueError(f"Используйте limit 1–{MAX_PAGE_SIZE} и offset 0–100000.")
    if source is not None and source not in SOURCE_LABELS:
        raise ValueError("Неизвестная категория source.")
    if search is not None and len(search) > 120:
        raise ValueError("Поисковый текст не должен превышать 120 символов.")
    if agent is not None and len(agent) > 120:
        raise ValueError("Название субагента не должно превышать 120 символов.")
    run_dir, _ = _completed_run(run_id, runtime_root)
    all_rows = _read_rows(_report(run_dir, "discrepancies.csv"))
    query = (search or "").casefold()
    selected = []
    for index, row in enumerate(all_rows, start=1):
        if source and row.get("source") != source:
            continue
        if agent and agent.casefold() not in (row.get("agent") or "").casefold():
            continue
        if query and not any(query in str(row.get(field, "")).casefold()
                             for field in ("agent", "identifier", "comment", "date")):
            continue
        selected.append({"discrepancy_id": f"dsp-{index:06d}", **row})
    return {
        "run_id": run_id,
        "total": len(selected),
        "offset": offset,
        "limit": limit,
        "discrepancies": selected[offset:offset + limit],
    }


def get_discrepancy_detail(
    run_id: str,
    discrepancy_id: str,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    if not isinstance(discrepancy_id, str) or not DISCREPANCY_ID_RE.fullmatch(discrepancy_id):
        raise ValueError("Некорректный ID расхождения.")
    run_dir, _ = _completed_run(run_id, runtime_root)
    rows = _read_rows(_report(run_dir, "discrepancies.csv"))
    index = int(discrepancy_id[4:])
    if index < 1 or index > len(rows):
        raise ValueError("Расхождение не найдено в этом запуске.")
    row = rows[index - 1]
    explanations = {
        "бот": "Правила Stage 1 предполагают, что нужно проверить загрузку ETM/бота.",
        "1С": "Правила Stage 1 предполагают, что операцию нужно сверить с учётом 1С.",
        "агент": "Правила Stage 1 предполагают, что нужно уточнить запись у субагента.",
        "выяснить": "Правила Stage 1 не позволяют однозначно определить сторону ошибки; нужна первичка.",
    }
    return {
        "run_id": run_id,
        "discrepancy_id": discrepancy_id,
        "agent": row.get("agent", ""),
        "date": row.get("date", ""),
        "identifier": row.get("identifier", ""),
        "sum_1c": row.get("sum_1c", ""),
        "sum_etm": row.get("sum_etm", ""),
        "diff": row.get("diff", ""),
        "source": row.get("source", ""),
        "report_comment": row.get("comment", ""),
        "interpretation": explanations.get(row.get("source", ""), "Проверьте первичные документы."),
        "limitation": "Это классификация правил Stage 1; подтверждение требует проверки первичных документов.",
        "source_rows_available": False,
    }


def list_review_candidates(
    run_id: str,
    issue: str | None = None,
    agent: str | None = None,
    limit: int = 25,
    offset: int = 0,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    if not 1 <= limit <= MAX_PAGE_SIZE or not 0 <= offset <= 100_000:
        raise ValueError(f"Используйте limit 1–{MAX_PAGE_SIZE} и offset 0–100000.")
    if agent is not None and len(agent) > 120:
        raise ValueError("Название субагента не должно превышать 120 символов.")
    run_dir, _ = _completed_run(run_id, runtime_root)
    rows = _read_rows(_report(run_dir, "review_candidates.csv"))
    selected = []
    for index, row in enumerate(rows, start=1):
        if issue and row.get("issue") != issue:
            continue
        if agent and agent.casefold() not in (row.get("agent") or "").casefold():
            continue
        selected.append({
            "candidate_id": f"cand-{index:06d}",
            **row,
            "issue_title": review_issue_title(row.get("issue", "")),
            "meaning": review_issue_meaning(row.get("issue", "")),
        })
    return {"run_id": run_id, "total": len(selected), "offset": offset, "limit": limit,
            "candidates": selected[offset:offset + limit],
            "notice": "Кандидаты требуют ручной проверки и не являются готовыми атрибуциями.",
            }


def submit_case_for_build(
    description: str,
    expected_behavior: str,
    run_id: str | None = None,
    discrepancy_id: str | None = None,
    supporting_notes: str = "",
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    description = description.strip() if isinstance(description, str) else ""
    expected_behavior = expected_behavior.strip() if isinstance(expected_behavior, str) else ""
    supporting_notes = supporting_notes.strip() if isinstance(supporting_notes, str) else ""
    if not description or len(description) > 5000:
        raise ValueError("Описание случая обязательно и не должно превышать 5000 символов.")
    if not expected_behavior or len(expected_behavior) > 3000:
        raise ValueError("Ожидаемое поведение обязательно и не должно превышать 3000 символов.")
    if len(supporting_notes) > 5000:
        raise ValueError("Дополнительные заметки не должны превышать 5000 символов.")

    linked_discrepancy = None
    if discrepancy_id:
        if not run_id:
            raise ValueError("Для связи с расхождением требуется ID запуска.")
        linked_discrepancy = get_discrepancy_detail(run_id, discrepancy_id, runtime_root)
    elif run_id:
        _completed_run(run_id, runtime_root)

    cases_root = runtime_root / "cases"
    _ensure_private_dir(runtime_root)
    _ensure_private_dir(cases_root)
    case_id = f"case-{uuid.uuid4().hex}"
    case = {
        "case_id": case_id,
        "status": "submitted",
        "created_at": _now(),
        "description": description,
        "expected_behavior": expected_behavior,
        "supporting_notes": supporting_notes,
        "run_id": run_id,
        "discrepancy": linked_discrepancy,
        "notice": "Кейс передан в Build на рассмотрение; рабочий пайплайн не изменён.",
    }
    _write_json(cases_root / f"{case_id}.json", case)
    return {"case_id": case_id, "status": "submitted", "created_at": case["created_at"],
            "message": case["notice"]}


def list_build_cases(limit: int = 50, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit должен быть от 1 до {MAX_PAGE_SIZE}.")
    cases_root = runtime_root / "cases"
    results = []
    if cases_root.exists():
        for path in cases_root.glob("case-*.json"):
            if path.is_symlink() or not CASE_ID_RE.fullmatch(path.stem):
                continue
            try:
                item = _read_json(path)
            except ValueError:
                continue
            results.append({key: item.get(key) for key in
                            ("case_id", "status", "created_at", "description", "expected_behavior", "run_id")})
    results.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    results = results[:limit]
    return {"cases": results, "count": len(results)}


def get_build_case(case_id: str, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any]:
    case_id = _validate_id(case_id, CASE_ID_RE, "кейса")
    cases_root = runtime_root / "cases"
    path = cases_root / f"{case_id}.json"
    if path.is_symlink() or not path.is_file() or path.resolve().parent != cases_root.resolve():
        raise ValueError("Кейс для Build не найден.")
    case = _read_json(path)
    if case.get("case_id") != case_id:
        raise ValueError("Идентификатор кейса не совпадает с записью.")
    return case
