#!/usr/bin/env python3
"""Scenario backends — one interface, honest differences (V2).

A backend turns (bars, horizon, n_paths, seed) into a list of terminal closes
(length n_paths). Backends never hide what they are; the comparison report
prints each backend's own description beside its numbers.

- bootstrap: stationary block bootstrap on log-returns (the default; the
  anchor everything else must beat)
- gaussian: parametric random walk with fitted μ,σ — the distributional
  assumption contrast (i.i.d. normal, no vol clustering)
- kronos: optional neural backend over the upstream HF weights
  (NeoQuasar/Kronos-*). Lazy torch/transformers import; without them it
  reports unavailability honestly instead of pretending. Written to the
  upstream-documented KronosPredictor API; exercised only on hosts with
  torch installed — if upstream's API drifts, the error surfaces at run time
  and belongs in an issue.
"""
import json, math, random, sys


def _logrets(bars):
    closes = [b[2] for b in bars]
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))], closes[-1]


def bootstrap(bars, horizon, n_paths, seed, block_len=5):
    rs, last = _logrets(bars)
    rng = random.Random(seed)
    n = len(rs)
    terminals = []
    for _ in range(n_paths):
        level, steps = last, 0
        while steps < horizon:
            start = rng.randrange(n)
            for k in range(block_len):
                if steps >= horizon:
                    break
                level *= math.exp(rs[(start + k) % n])
                steps += 1
        terminals.append(level)
    return terminals


def gaussian(bars, horizon, n_paths, seed):
    rs, last = _logrets(bars)
    n = len(rs)
    mu = sum(rs) / n
    var = sum((r - mu) ** 2 for r in rs) / (n - 1)
    sd = math.sqrt(var)
    rng = random.Random(seed)
    return [last * math.exp(sum(rng.gauss(mu, sd) for _ in range(horizon))) for _ in range(n_paths)]


class BackendUnavailable(Exception):
    """raised when an optional backend's dependencies are absent — honest, with the fix"""


def kronos(bars, horizon, n_paths, seed, model_name="NeoQuasar/Kronos-small",
           tokenizer_name="NeoQuasar/Kronos-Tokenizer-base"):
    try:
        import torch  # noqa: F401
        from model import Kronos, KronosTokenizer, KronosPredictor  # upstream repo layout
    except ImportError as e:
        raise BackendUnavailable(
            "backend 'kronos' requires torch + transformers and the upstream model package "
            "(github.com/shiyu-coder/Kronos checked out on PYTHONPATH); HF weights download at "
            "first use (set HF_ENDPOINT=https://hf-mirror.com on CN networks). "
            f"Import error was: {e}")
    import datetime as dt
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    df = {"open": [b[1] for b in bars], "high": [b[3] for b in bars],
          "low": [b[4] for b in bars], "close": [b[2] for b in bars],
          "volume": [b[5] for b in bars], "amount": [0.0 for _ in bars]}
    day = dt.timedelta(days=1)
    t0 = dt.datetime(2020, 1, 1)
    x_ts = [t0 + day * i for i in range(len(bars))]
    y_ts = [t0 + day * (len(bars) + i) for i in range(horizon)]
    torch.manual_seed(seed)
    terminals = []
    for _ in range(n_paths):  # one path per call: we need the path distribution, not the mean
        pred = predictor.predict(df=df, x_timestamp=x_ts, y_timestamp=y_ts,
                                 pred_len=horizon, T=1.0, top_p=0.9, sample_count=1)
        terminals.append(float(pred["close"].iloc[-1]))
    return terminals


REGISTRY = {
    "bootstrap": {"fn": lambda b, h, n, s: bootstrap(b, h, n, s), "kind": "statistical",
                  "desc": "stationary block bootstrap on log-returns (the anchor)"},
    "gaussian": {"fn": lambda b, h, n, s: gaussian(b, h, n, s), "kind": "statistical",
                 "desc": "parametric i.i.d. normal walk (fitted mu, sigma; no vol clustering)"},
    "kronos": {"fn": kronos, "kind": "neural",
               "desc": "upstream Kronos HF weights, autoregressive sampling (optional; torch required)"},
}


def get(name):
    if name not in REGISTRY:
        sys.exit(f"unknown backend {name!r}; available: {', '.join(REGISTRY)}")
    return REGISTRY[name]
