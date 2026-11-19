---
name: dsh-kronos
description: K-line scenario forecasting for DeepSeek Harness — seeded stationary-block-bootstrap fans (P10/P50/P90, P-above-current) from real history; deterministic, zero-dependency; the statistical baseline that neural foundation models must beat.
---

Use this skill when asked "接下来可能怎么走", for scenario fans, or for
forecast-model comparisons. v0 is the statistical baseline — say so whenever
quoting it.

## Run

```bash
KIT=<this-directory>
python3 $KIT/scripts/scenario.py --data <klines.json> --horizon 30 --paths 1000 --seed 42 --save
```

K-lines JSON: {"SYMBOL": [[date, open, close, high, low, volume], ...]}.

## Discipline

- Always attach the disclosure: statistical baseline, not a learned model;
  scenarios ≠ predictions.
- Same seed → same fan; report the seed with any number.
- The fan is unconditional (no covariates) — don't present it as conditional
  forecasting. Not investment advice.
