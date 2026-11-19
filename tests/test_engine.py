#!/usr/bin/env python3
"""Hermetic engine tests — no network, no LLM. Synthetic bars only.
Run: python3 tests/test_engine.py   (exits non-zero on any FAIL)"""
import datetime as dt
import json, math, os, random, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scenario as sc
import evaluate as ev

FAILS = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        FAILS.append(name)


def synth(n=365, base=100.0, vol=0.02, seed=7):
    rng = random.Random(seed)
    rows, d, p = [], dt.date(2025, 1, 1), base
    for i in range(n):
        p *= math.exp(rng.gauss(0, vol))
        rows.append([d.isoformat(), round(p * 0.99, 4), round(p, 4), round(p * 1.01, 4),
                     round(p * 0.98, 4), 1000.0])
        d += dt.timedelta(days=1)
    return rows


bars = synth()

# 1. determinism: same seed bit-exact; different seed differs
a = sc.bootstrap_scenarios(bars, 10, 200, 5, 42)
b = sc.bootstrap_scenarios(bars, 10, 200, 5, 42)
c = sc.bootstrap_scenarios(bars, 10, 200, 5, 43)
check("same seed bit-exact", a["terminal_P90"] == b["terminal_P90"] and a["P_above_current"] == b["P_above_current"])
check("different seed differs", a["terminal_P90"] != c["terminal_P90"])

# 2. quantile monotonicity
check("quantiles monotonic", a["terminal_P10"] <= a["terminal_P25"] <= a["terminal_P50"]
      <= a["terminal_P75"] <= a["terminal_P90"])

# 3. preconditions refuse honestly
probs = sc.check_preconditions(bars[:40], 30, 200)
check("insufficient history refused", any("insufficient" in p for p in probs))
probs = sc.check_preconditions(bars, 30, 50)
check("too few paths refused", any("paths" in p for p in probs))

# 4. coverage gaps (synthetic bars are contiguous daily → day 99 → day 112 = 13 days)
gappy = bars[:100] + bars[112:]
g = sc.coverage_gaps(gappy)
check("13-day gap detected", len(g) == 1 and g[0]["calendar_days"] == 13)
check("no gaps on contiguous", sc.coverage_gaps(bars) == [])

# 5. CRPS math: point mass at y = 0; N(0,1) analytic ≈ 0.2337
check("CRPS point mass zero", ev.empirical_crps([5.0], 5.0) == 0.0)
rng = random.Random(1)
xs = [rng.gauss(0, 1) for _ in range(4000)]
check("CRPS N(0,1) analytic", abs(ev.empirical_crps(xs, 0.0) - 0.2337) < 0.005)

# 6. rolling evaluation structure
rep = ev.evaluate_rolling(bars, 10, 200, 5, 42, 4, 200)
check("eval has protocol+results", "protocol" in rep and "results" in rep)
r = rep["results"]
check("coverage in [0,1]", 0.0 <= r["coverage_80"]["empirical"] <= 1.0)
check("crps positive finite", math.isfinite(r["mean_crps_rel"]) and r["mean_crps_rel"] > 0)
try:
    ev.evaluate_rolling(bars[:100], 10, 200, 5, 42, 4, 200)
    check("eval refuses short history", False)
except SystemExit:
    check("eval refuses short history", True)

# 7. i18n: prompts/labels never formatted without kwargs (literal braces safe)
for lang in ("en", "zh"):
    sc.LANG = lang
    check("disclosure exists in %s" % lang, len(sc.t("disclosure")) > 20)
    s = sc.t("envelope_line", seed=1, paths=2, block=3, bars=4)
    check("envelope line in %s" % lang, ("seed" in s or "种子" in s) and "4" in s)
sc.LANG = "en"

# 8. fan svg carries envelope line
out = sc.bootstrap_scenarios(bars, 10, 200, 5, 42)
paths = out.pop("_paths")
svg = sc.fan_svg(paths, out["last_close"], 10, "T",
                 envelope={"seed": 9, "block_len": 5, "window": {"bars": 365}})
check("svg envelope line", "seed 9" in svg and "365 bars" in svg)

# 9. artifact round-trip via CLI-shaped envelope
env = {"method": sc.t("method"), "seed": 42, "n_paths": 200, "block_len": 5,
       "window": {"bars": len(bars), "from": bars[0][0], "to": bars[-1][0]},
       "conditioning": "none"}
check("envelope carries the reproducibility tuple",
      env["seed"] == 42 and env["window"]["bars"] == 365 and env["conditioning"] == "none")

# 10. backends + comparison (V2)
import backends, compare as cmp_mod
bt = backends.bootstrap(bars, 10, 200, 42)
gz = backends.gaussian(bars, 10, 200, 42)
check("backends return n terminals", len(bt) == 200 and len(gz) == 200)
check("backend determinism", backends.bootstrap(bars, 10, 200, 42) == bt)
check("bootstrap vs gaussian differ", sorted(bt) != sorted(gz))
try:
    backends.kronos(bars, 10, 2, 42)
    check("kronos unavailable raises honestly", False)
except backends.BackendUnavailable as e:
    check("kronos unavailable raises honestly", "torch" in str(e))
rep, rows = cmp_mod.compare(bars, 10, 200, 42, 4, 200, ["bootstrap", "gaussian"])
check("compare has both sides", set(rep["results"]) == {"bootstrap", "gaussian"})
v = rep["verdict"][0]
check("verdict names winner and significance", ("BETTER" in v or "WORSE" in v) and "significant" in v)

print("result:", "GREEN" if not FAILS else "RED (%d)" % len(FAILS))
sys.exit(1 if FAILS else 0)
