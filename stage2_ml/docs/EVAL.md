# Stage 2: temporal validation

## Workflow

`run_stage_1_2.py` builds the training tables for each run from the current input
files and that run's Stage 1 output. Tables, models, inference results, and metrics
are stored under `.finolimp/manual_runs/<run-id>/stage2/`; there are no checked-in
or frozen Stage 2 training tables.

By default, the last 20% of unique operation dates are held out for validation.
The cutoff is calculated from the current inputs and can be changed with
`--validation-fraction`. Alternatively, `--validation-months N` reserves the
last N full calendar months for validation with a round month-aligned cutoff
on the 1st (e.g. `--validation-months 2` on January–July data validates on
June–July with cutoff 2026-06-01).

- Earlier dates are used for fitting; validation rows are not included in model fit.
- Candidate pairs that cross the cutoff or have unknown dates are excluded.
- Status aggregates use earlier dates only. Source features do not use future
  registry rows or end-of-export-dependent fields.
- A source component spanning the cutoff is assigned to validation using its latest
  participant date.

## How to interpret the metrics

Stage 2 targets are derived from Stage 1 decisions (silver labels). Validation
measures out-of-time fidelity to those labels; it is not an independent human
assessment of matching accuracy or of Stage 1. Overall metrics may also include
the training period, so use the validation metrics when assessing temporal
generalization.

## Russian Excel report

Each successful Stage 2 run also creates
`stage2/finolimp_stage2_model_report.xlsx`, separate from the accountant-facing
Stage 1 workbook. It includes a run summary, temporal-validation metrics, candidate
gate coverage, per-class source metrics, a source confusion matrix, and notes on
silver-label and small-class limitations. The workbook is generated from that run's
JSON reports and does not modify the CSV artifacts.

Legacy benchmark code and its reports, frozen train data, adjudication samples, and
saved predictions have been removed. No real-input temporal validation was run as
part of this cleanup; its numerical metrics remain unknown until a new run is
explicitly requested.
