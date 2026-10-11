# Stage 2 evaluation boundaries

## Data and fit boundary

- Each run builds explicit train and validation tables from its current input files
  and Stage 1 output. `stage2_ml/train/` is not an input to the runner.
- The chronological cutoff is selected from unique operation dates. Models fit only
  on earlier dates; validation dates are held out.
- Cross-cutoff or undated candidate pairs are excluded. Status aggregates use past
  dates only; source features must not use future registry rows or features that
  depend on the end of the export.
- Stage 1-derived labels are silver targets. Validation measures temporal fidelity
  to Stage 1, not independent human accuracy or true recall.

## Stage boundary

Stage 1 remains the accounting reference and is not changed by Stage 2. Stage 2
may use Stage 1 outputs to construct silver labels during training, but inference
results are written separately in each run and do not replace Stage 1 decisions.

Legacy benchmark modules and saved reports have been removed. Evaluate the current
run only from its own training manifest and inference report; its targets remain
Stage 1-derived silver labels, not independent human truth.
