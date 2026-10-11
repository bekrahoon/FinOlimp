# FinOlimp: сверка 1С ↔ ETM (два этапа)

Инструмент разбит на два этапа с чёткой границей (файлы этапа 1 — входы этапа 2).

## Установка на чистый компьютер и запуск

Из корня полной копии проекта:

- **Windows (PowerShell):** `powershell -ExecutionPolicy Bypass -File .\install_windows.ps1`
- **Ubuntu:** `bash install_ubuntu.sh` (потребуется `sudo` для системных утилит).

Установщик ставит Python 3.12, создаёт `.venv`, устанавливает зависимости Stage 1/2,
подготавливает отдельное заблокированное окружение MCP и устанавливает OpenCode CLI.
После установки из корня проекта запускайте:

```text
Windows: .\.venv\Scripts\python.exe .\run_stage_1_2.py
         .\.venv\Scripts\python.exe .\run_ai.py
Ubuntu:  .venv/bin/python run_stage_1_2.py
         .venv/bin/python run_ai.py
```

`run_stage_1_2.py` создаёт отдельную папку `.finolimp/manual_runs/<run-id>/` с
отчётами Stage 1, отдельным Stage 2 и манифестом; существующие результаты не
перезаписываются. Stage 2 строит run-local `train/` из входов и результата Stage 1,
обучает модели, затем выполняет ML-сверку. В `stage1/` создаются
`finolimp_reports.xlsx` и `README.xlsx`; ML-файлы остаются в соседней папке `stage2/`
и в книгу Stage 1 не подмешиваются. Stage 2 дополнительно создаёт отдельную
русскоязычную книгу `finolimp_stage2_model_report.xlsx` с temporal validation,
показателями по классам и ограничениями интерпретации метрик. `--stage1-only` запускает только правила;
`--stage2-only --stage1-reports <папка>` строит Stage 2 из входов и ранее созданного
отчёта Stage 1. ML-результат не заменяет поручения Stage 1. В per-run CSV технические
заголовки сохраняются, а значения `check` и `issue` заменяются русскими пояснениями.
Train-метки Stage 2 являются производными от текущего Stage 1 («silver»), поэтому
это самообучение/второе мнение, а не независимая проверка качества правил.
На Ubuntu завершённая папка запуска становится доступной только владельцу и только
для чтения; в Windows доступ определяется ACL папки проекта.
Для AI запустите `run_ai.py` той же командой Python: он откроет OpenCode в
`stage3_ai`, но сверка начнётся только по вашему запросу.
При первом использовании OpenCode подключите провайдера через `/connect`.
Зависимости основного запуска перечислены в `requirements-runner.txt`.

## Этап 1 — правила, Шаги 1–6 (`stage1_rules/`)
- `input/` — входные выгрузки: `acts.csv`, `etm.csv`, `registry.csv` + `reconcile.yaml`.
- `src/` — движок и скрипты: `parse.py`, `reconcile2.py`, `match.py`,
  `build_features.py`, `rank.py`, `build_actions.py`, `sample_adjudication.py`,
  `tests/` (`pytest src/tests/ -q` из `stage1_rules/`).
- `reports/` создаётся при ручном запуске `match.py`; готовые отчёты в копию
  проекта и переносимый ZIP не включаются. `run_stage_1_2.py` сохраняет их отдельно
  в каталоге конкретного запуска.
- `docs/` — `README.md` (методика движка), `features.md`, `ORPHANS.md`.
- Запуск бухгалтеру — см. `RUNBOOK.md`.

## Этап 2 — run-local ML-сверка (`stage2_ml/`)
- Для каждого запуска создаются `stage2/train/` (производные train-таблицы),
  `stage2/bootstrap/` (первичный проход для сопоставления строк) и ML-отчёты.
  Старые результаты/обучающие наборы не нужны и не перезаписываются.
- Таблицы Stage 2 делятся хронологически: по умолчанию последние 20% уникальных
  дат входных операций — validation, более ранние даты — train. Пары, пересекающие
  границу, не попадают ни в одну выборку; контекст статусов использует только прошлые
  даты. Долю можно менять через `--validation-fraction`.
- `src/matcher/build_stage2_training.py` собирает pair/status train из текущих
  входов и правил группировки Stage 1; `prepare_source_s3_train.py` добавляет
  source-метки из `stage1/discrepancies.csv` и делит их тем же cutoff. Классификаторы
  обучаются только на ранней части; validation-метрики — fidelity к silver-меткам
  Stage 1, не независимая человеческая оценка.
- `src/matcher/` содержит код построения run-local таблиц, temporal split,
  обучения и inference. Legacy-эксперименты, frozen benchmark и adjudication-зависимый
  код удалены. Основной runner создаёт данные только внутри нового запуска.
- Для аудита Stage 2 vs Stage 1 используйте `src/matcher/evaluate_s3.py` с
  `--input-dir`, `--stage1-reports`, `--stage2-dir` и `--output-prefix`; при
  run-local запуске он разделяет общую fidelity и out-of-time validation по silver-меткам.
- `docs/EVAL.md` описывает текущую temporal validation и её ограничения; старые
  результаты экспериментов и их производные файлы не входят в проектную копию.

## Правило границы
Этап 2 никогда не правит файлы этапа 1 руками — только читает. Этап 1 не знает
про этап 2 (ноль импортов в ту сторону).

## OpenCode для бухгалтера (первый срез)

FinOlimp-агент `Reconcile`, его бизнес-скилы и локальный Python MCP подключаются
только при запуске OpenCode из `stage3_ai/`. Из корня проекта FinOlimp эти
специализированные настройки не загружаются; доступны обычные `Build` и `Plan`.
Скрипт `run_ai.py` запускает OpenCode с правильной рабочей папкой.
В `stage3_ai/` Reconcile по умолчанию использует прямые ограниченные MCP-инструменты
(без Code Mode), чтобы проверять/запускать полный Stage 1 bundle, читать результаты
и передавать неподдержанные кейсы в Build.

Нужен `uv` и Python 3.11+. Runtime snapshots и отчёты хранятся в
`stage3_ai/.finolimp/runtime/` (не в Git). Подробности и ограничения: см.
`stage3_ai/integrations/finolimp_mcp/README.md`. Загрузка файлов через чат, запуск отдельных
шагов, рабочая копия Build и одобрение/публикация версии пока не реализованы.
Каждый новый завершённый запуск также содержит `reports/finolimp_reports.xlsx`:
все CSV-отчёты именно этого запуска разложены по отдельным листам с русскими
названиями и заголовками колонок; к кодам проверок добавлены русские пояснения.
Это отдельная книга от статического Stage 2 Excel-отчёта. Заголовки CSV сохраняются;
в значениях `check` и `issue` машинные коды заменяются русскими пояснениями.
Рядом генерируется `reports/README.xlsx` — словарь с русским названием и описанием
каждого поля, также с отдельным листом на каждый отчёт.

## Motion graphics (Remotion)

Изолированный Remotion-проект находится в `motion-graphics/` и не входит в
Python-зависимости сверки. Откройте его в OpenCode из этой папки либо запустите
предпросмотр командами `cd motion-graphics && npm run dev`. В проект добавлены
официальные Remotion Agent Skills, доступные OpenCode через `.agents/skills/`, и
локальный навык `.opencode/skills/remotion-motion-graphics/`. Для проверки типов
и стиля используйте `cd motion-graphics && npm run lint`.
