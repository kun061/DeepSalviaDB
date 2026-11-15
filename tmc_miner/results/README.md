# TMC-Miner validation results

This directory contains only small validation summaries suitable for Git:

- `experiment_metadata.json`: validation design, stages, sample-pairing limitations, models, and DEG counts.
- `summary_by_model.csv`: aggregate metrics for four models across metabolite-stage combinations.

Per-combination metrics, per-fold predictions, and per-gene stability files can be regenerated with `analysis/cross_validate.py`, but are excluded from the release repository by default.

All four models have negative median R² in the current summaries. Interpret this result alongside the use of six-stage means, unpaired samples, and non-nested DEG filtering.
