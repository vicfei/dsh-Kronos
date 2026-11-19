#!/usr/bin/env python3
"""Self-evaluation harness — rolling out-of-sample scoring of the baseline (V1.4)

The anchor for V2's neural-vs-bootstrap verdict: before judging any backend,
quantify how good the bootstrap fan itself is, honestly.

Protocol: for each anchor t in the tail (spaced `horizon` apart, so scored
windows don't overlap): fit on bars[t-W, t), generate the fan for `horizon`
steps, score the realized terminal against the fan — quantile coverage
(empirical vs nominal 80%/50%), empirical CRPS (O(n log n) decomposition),
and mean relative fan width. Deterministic: seed per anchor = seed + t,
disclosed in the report.

Usage:
  python3 scripts/evaluate.py --data /tmp/btc_365.json --horizon 30 --anchors 10
"""
import argparse, bisect, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario import bootstrap_scenarios, coverage_gaps, MIN_BARS


def empirical_crps(samples, y):
    """CRPS of an empirical distribution vs observation y:
    CRPS = E|X−y| − ½E|X−X′|; the pairwise term via sorted coefficients, O(n log n)."""
    xs = sorted(samples)
    n = len(xs)
    i = bisect.bisect_left(xs, y)
    term1 = (sum(xs[i:]) - (n - i) * y + i * y - sum(xs[:i])) / n
    term2 = 2 * sum((2 * k - n + 1) * x for k, x in enumerate(xs)) / (n * n)
    return term1 - 0.5 * term2


def evaluate_rolling(bars, horizon, n_paths, block_len, seed, n_anchors, fit_window):
    if len(bars) < fit_window + horizon:
        sys.exit(f"refused honestly — need >= {fit_window + horizon} bars for rolling "
                 f"evaluation (fit {fit_window} + horizon {horizon}), got {len(bars)}")
    anchors = []
    t = len(bars) - horizon
    while len(anchors) < n_anchors and t - fit_window >= 0:
        anchors.append(t)
        t -= horizon
    anchors.reverse()
    rows = []
    for t in anchors:
        hist = bars[t - fit_window:t]
        realized = bars[t + horizon - 1][2]
        out = bootstrap_scenarios(hist, horizon, n_paths, block_len, seed + t)
        terminals = sorted(p[-1] for p in out["_paths"])
        rows.append({
            "anchor_date": str(hist[-1][0])[:10],
            "covered_80": out["terminal_P10"] <= realized <= out["terminal_P90"],
            "covered_50": out["terminal_P25"] <= realized <= out["terminal_P75"],
            "crps_rel": empirical_crps(terminals, realized) / out["last_close"],
            "fan_width_80_rel": (out["terminal_P90"] - out["terminal_P10"]) / out["last_close"],
        })
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    return {
        "kind": "self-evaluation",
        "protocol": {"anchors": n, "horizon": horizon, "fit_window": fit_window,
                     "paths_per_anchor": n_paths, "block_len": block_len,
                     "seed_policy": "seed + anchor_index (disclosed)",
                     "spacing": "anchors spaced `horizon` apart — scored windows do not overlap"},
        "results": {
            "coverage_80": {"nominal": 0.80, "empirical": mean("covered_80")},
            "coverage_50": {"nominal": 0.50, "empirical": mean("covered_50")},
            "mean_crps_rel": mean("crps_rel"),
            "mean_fan_width_80_rel": mean("fan_width_80_rel"),
        },
        "anchor_rows": rows,
        "disclosure": "rolling out-of-sample scores of the bootstrap fan itself; "
                      "coverage compares empirical hit rates to nominal levels — a fan can "
                      "be calibrated yet uninformative (width) or sharp yet miscalibrated; "
                      "read both. CRPS is relative to the anchor close (scale-free). "
                      "This is the anchor any neural backend must beat (V2).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--block_len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchors", type=int, default=10)
    ap.add_argument("--fit-window", type=int, default=250)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.data))
    symbol, bars = next(iter(data.items()))
    rep = evaluate_rolling(bars, args.horizon, args.paths, args.block_len,
                           args.seed, args.anchors, args.fit_window)
    rep["symbol"] = symbol
    rep["source"] = {"origin": "user-provided klines JSON", "bars": len(bars),
                     "coverage_gaps": coverage_gaps(bars)}
    r = rep["results"]
    print(f"## self-evaluation · {symbol} · {rep['protocol']['anchors']} anchors × "
          f"{args.horizon} steps (fit {args.fit_window}, {args.paths} paths)\n")
    print(f"- coverage P10–P90: empirical {r['coverage_80']['empirical']:.0%} vs nominal 80%")
    print(f"- coverage P25–P75: empirical {r['coverage_50']['empirical']:.0%} vs nominal 50%")
    print(f"- mean CRPS (rel): {r['mean_crps_rel']:.4%}")
    print(f"- mean fan width P10–P90 (rel): {r['mean_fan_width_80_rel']:.1%}\n")
    print("| anchor | cov80 | cov50 | crps_rel | width |")
    print("|---|---|---|---|---|")
    for row in rep["anchor_rows"]:
        print(f"| {row['anchor_date']} | {row['covered_80']} | {row['covered_50']} "
              f"| {row['crps_rel']:.4%} | {row['fan_width_80_rel']:.1%} |")
    print(f"\n{rep['disclosure']}")
    if args.save:
        os.makedirs("scenarios", exist_ok=True)
        path = f"scenarios/{symbol.replace('/', '_')}-eval-h{args.horizon}.json"
        json.dump(rep, open(path, "w"), ensure_ascii=False, indent=1, default=str)
        print(f"→ {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
