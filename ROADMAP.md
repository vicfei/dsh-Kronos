# ROADMAP — dsh-Kronos

North star: the family's time-series scenario seat, riding the 47.9k★ Kronos
gravity (the family's second-largest name), 100–300★ in 30 days. Naming on all
surfaces: GitHub = dsh-Kronos (mirror CamelCase) / npm = dsh-kronos.

## Positioning discipline (this repo's own)

Upstream [Kronos](https://github.com/shiyu-coder/Kronos) is an AAAI 2026
neural foundation model (VQ tokenizer + autoregressive Transformer over
OHLCV). We are **not** the model-training seat — we are the **lightweight
scenario seat**: zero-dependency, seed-deterministic, audited statistics, with
the neural model as an *optional* backend that must justify itself against the
bootstrap baseline.

**Competitive read (2026-11-19 deep-dive, issue #1)**: the incumbent's
engineering side has been dormant for 7 months (last push 2026-04-13, 298 open
issues untouched); its own README admits the demo is raw signals, not pure
alpha, and offers **no systematic baseline comparison**. That gap — "does the
neural model actually beat honest statistics, measured properly?" — is our
differentiated hook. The window is open: upstream is not evolving.

## Milestones

### M0 ✅ (2026-11-17)
Statistical baseline scenario engine (stationary block bootstrap,
seed-deterministic, zero-dependency; P10–P90 fan + P_above_current + SVG
scenario chart). Real smoke: BTC 365d, 30-step fan (P10/50/90 = 53.1k/62.6k/
73.2k, P_above = 38.9%), disclosure rides with the output.

### V1 · Hardening + release train → npm 0.1.0 (next session)
- [ ] V1.1 EN-first: engine output via i18n table (en default, `--lang zh`
      fallback); README pure EN + `README.zh-CN.md` translation
- [ ] V1.2 disclosure systematization: every artifact carries
      {method: statistical-baseline-not-a-model, seed, paths, block_len,
      window, source metadata}; unavailable states (insufficient history)
      render with reasons — never silent
- [ ] V1.3 source metadata rides: data source/date-range/bars/coverage with
      every scenario (the calibers-ride DNA, data edition)
- [ ] V1.4 **self-evaluation harness**: rolling out-of-sample evaluation of
      the baseline itself — fit window → generate fan → score realized path
      (quantile coverage, CRPS, fan width); deterministic, zero LLM. This is
      the anchor the neural backend will be judged against in V2
- [ ] V1.5 release train: root package (bin shim, dsh.bundle + skill, hermetic
      tests + CI, `scripts/*.py` glob); owner publish; awesome-dsh submission
      (family queue #16227/#16447/#16751/#16769 precedent)

### V2 · The comparison flagship → npm 0.2.0
- [ ] V2.1 `--via kronos` optional backend: lazy torch/transformers import
      with an honest unavailability message when absent (zero-dep default
      intact); HF weights via NeoQuasar (HF_ENDPOINT mirror support for CN
      networks); sample_count aligned to our paths; seed recorded
- [ ] V2.2 **对拍基准 (the flagship)**: same window, same horizon — bootstrap
      vs Kronos on CRPS / quantile coverage / interval width, OOS protocol,
      both sides reported, no cherry-picking. **Neural must significantly beat
      bootstrap to count** (factor-authentication lineage)
- [ ] V2.3 the comparison verdict itself is an artifact (SCENARIO-REPORT
      seed): replayable, diffable, honest about what the comparison does and
      does not show

### V3 · Depth (0.3.x)
- [ ] V3.1 multi-asset joint scenarios: shared index-block resampling that
      preserves cross-asset correlation
- [ ] V3.2 family integration: jesse optimization targets become scenario
      distribution metrics (P_above, fan width), not point values
- [ ] V3.3 conditional scenarios: block pools filtered by regime (vol
      similarity), with the conditioning disclosed

### V4 · Ecosystem (0.4.x)
- [ ] SCENARIO-REPORT spec v0.1 (the family's sixth interchange format):
      versioned, stdlib validator, CI enforcement
- [ ] searchable scenario history via the WeKnora KB bridge (family pattern,
      zero bespoke storage)
- [ ] monthly incumbent digest: watch for upstream revival (dormancy is an
      input, not a guarantee)

## Star strategy

1. Name = traffic: `kronos` full name + family cross-links; topics carry
   dsh/dsh-plugin (auto-curated lists).
2. Hook: **"does the 48k★ neural model beat honest bootstrap? OOS 对拍
   inside"** — a question nobody upstream answers, asked with reproducible
   numbers.
3. The SVG fan card is post-ready; the comparison report doubles as hook #2.
4. awesome-dsh submission after npm 0.1.0; same-day family combo discipline.

## Discipline (non-negotiable)

- Disclosure (statistical baseline / not a forecast) never detaches from output.
- The seed is reported with the numbers — reproducibility is part of the product.
- Once the neural backend lands, the bootstrap baseline stays forever as the
  comparison anchor. No investment advice.
