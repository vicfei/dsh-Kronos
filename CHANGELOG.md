# Changelog

## 0.3.0 (2026-11-19) — first release: hardening + comparison flagship + depth

The V1–V3 line ships as one release (registry jumps placeholder → 0.3.0).

### Hardening (V1)
- English-first with `--lang zh` fallback; disclosure is structural — every
  artifact carries a reproducibility envelope (method/seed/paths/block_len/
  window/conditioning) and source metadata (date range, coverage gaps >10d).
- Honest pre-flight: insufficient history / non-positive closes / unstable
  path counts refused with reasons, never run on degenerate inputs.
- **Self-evaluation harness** (`evaluate.py`): rolling OOS scoring of the fan
  itself — quantile coverage vs nominal, empirical CRPS (O(n log n)
  decomposition, verified against analytic N(0,1)), mean width.

### The comparison flagship (V2)
- Backend registry: bootstrap (the anchor), gaussian (parametric i.i.d.
  contrast), kronos (optional neural backend over upstream HF weights; lazy
  torch import with honest unavailability).
- `compare.py`: same anchors/realized/seed policy for every backend; verdict
  distinguishes BETTER from significant (>5% relative CRPS edge, a declared
  threshold); **neural only counts when it significantly beats bootstrap**.

### Depth (V3)
- Joint multi-symbol scenarios: shared block draws preserve realized
  co-movement; simulated terminal correlations reported.
- Vol-regime conditioning with the pool disclosed; honest unavailable below
  20 blocks.
- Optimizer objective interface (`--objective-out`) for family consumption
  (distribution metrics, not point predictions).

### Packaging
- npm bin + DSH bundle + skill; 31 hermetic checks + CI.

## 0.0.1
Name placeholder.
