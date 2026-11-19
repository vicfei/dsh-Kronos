---
name: dsh-kronos
description: K-line scenario forecasting for DeepSeek Harness — seeded reproducible scenario fans (stationary block bootstrap, zero-dependency), disclosure and reproducibility envelope on every payload, rolling OOS self-evaluation, and an honest head-to-head layer where any neural backend must beat the bootstrap baseline.
---

Use this skill when asked about future scenarios / ranges / risk for a symbol
("what could BTC look like in 30 days", "downside range", "scenario fan").
The engine is deterministic statistics — you orchestrate, it computes.
Requires python3 on PATH (stdlib only).

## Run

```bash
dsh-kronos --data <klines.json> --horizon 30 --paths 1000 --seed 42 --save
dsh-kronos --data klines.json --lang zh ...                       # Chinese fallback
python3 scripts/evaluate.py --data <klines.json> --horizon 30     # self-evaluation harness
```

Input: klines JSON `{symbol: [[date, open, close, high, low, volume], ...]}`
(family format). Output: quantile fan (P10-P90 terminal quantiles,
P_above_current) + SVG card; every payload carries the reproducibility
envelope (method/seed/paths/block/window) and the disclosure (statistical
baseline, not a learned model).

## Discipline

- The disclosure never detaches from output; the seed is reported with the
  numbers — reproducibility is part of the product.
- Scenarios are not predictions; never present a fan as advice.
- The bootstrap baseline stays forever as the comparison anchor; any neural
  backend must significantly beat it to count (rolling OOS, both sides
  reported).
