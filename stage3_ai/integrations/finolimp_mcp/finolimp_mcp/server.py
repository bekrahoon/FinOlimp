"""MCP stdio entrypoint for OpenCode."""

from mcp.server.fastmcp import FastMCP

from . import service

mcp = FastMCP("FinOlimp reconciliation")


@mcp.tool()
def register_current_dataset() -> dict:
    """Создать неизменяемый snapshot текущих acts.csv, etm.csv и registry.csv."""
    return service.register_current_dataset()


@mcp.tool()
def validate_dataset(dataset_id: str) -> dict:
    """Проверить структуру и формат зарегистрированного snapshot без сверки."""
    return service.validate_dataset(dataset_id)


@mcp.tool()
async def run_reconciliation(dataset_id: str) -> dict:
    """Запустить Stage 1 и подготовить XLSX-отчёты и словарь колонок этого запуска."""
    return await service.run_reconciliation(dataset_id)


@mcp.tool()
def list_runs(limit: int = 20) -> dict:
    """Показать последние запуски сверки."""
    return service.list_runs(limit)


@mcp.tool()
def get_run_summary(run_id: str) -> dict:
    """Показать итог сверки, residual, очередь ручной проверки и замечания качества."""
    return service.get_run_summary(run_id)


@mcp.tool()
def list_discrepancies(
    run_id: str,
    source: str | None = None,
    agent: str | None = None,
    search: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict:
    """Найти расхождения в отчёте. source: бот, 1С, агент или выяснить."""
    return service.list_discrepancies(run_id, source, agent, search, limit, offset)


@mcp.tool()
def get_discrepancy_detail(run_id: str, discrepancy_id: str) -> dict:
    """Открыть одну строку Stage 1 с интерпретацией и ограничениями вывода."""
    return service.get_discrepancy_detail(run_id, discrepancy_id)


@mcp.tool()
def list_review_candidates(
    run_id: str,
    issue: str | None = None,
    agent: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict:
    """Показать отдельные, неатрибутированные кандидаты на ручную проверку."""
    return service.list_review_candidates(run_id, issue, agent, limit, offset)


@mcp.tool()
def list_quality_findings(
    run_id: str,
    check: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict:
    """Показать технические замечания Stage 1 с кратким объяснением."""
    return service.list_quality_findings(run_id, check, limit, offset)


@mcp.tool()
def submit_case_for_build(
    description: str,
    expected_behavior: str,
    run_id: str | None = None,
    discrepancy_id: str | None = None,
    supporting_notes: str = "",
) -> dict:
    """Передать неподдержанный случай на рассмотрение Build; пайплайн не меняется."""
    return service.submit_case_for_build(
        description, expected_behavior, run_id, discrepancy_id, supporting_notes,
    )


@mcp.tool()
def list_build_cases(limit: int = 50) -> dict:
    """Показать кейсы, переданные на рассмотрение Build."""
    return service.list_build_cases(limit)


@mcp.tool()
def get_build_case(case_id: str) -> dict:
    """Получить описание кейса и ожидаемое поведение для работы в Build."""
    return service.get_build_case(case_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
