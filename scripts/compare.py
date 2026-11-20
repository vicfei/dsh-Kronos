#!/usr/bin/env python3
"""Head-to-head comparison — the flagship (V2): does anything beat honest bootstrap?

Protocol: identical rolling out-of-sample anchors as the self-evaluation
harness (evaluate.py). Every backend faces the SAME anchor windows, the SAME
realized terminals, the SAME seed policy (seed + anchor index). Metrics:
quantile coverage (empirical vs nominal 80/50), mean relative CRPS, mean
relative fan width. Verdict discipline: all sides always reported, no
cherry-picking; a neural backend only "counts" when it significantly beats
bootstrap on CRPS — and the report says which side won, not which side we
wanted to win.

Usage:
  python3 scripts/compare.py --data /tmp/btc_365.json --via bootstrap,gaussian
  python3 scripts/compare.py --data klines.json --via bootstrap,kronos   # torch host
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends
from evaluate import empirical_crps
from scenario import coverage_gaps


def compare(bars, horizon, n_paths, seed, n_anchors, fit_window, via):
    if len(bars) < fit_window + horizon:
        sys.exit(f"refused honestly — need >= {fit_window + horizon} bars, got {len(bars)}")
    anchors = []
    t = len(bars) - horizon
    while len(anchors) < n_anchors and t - fit_window >= 0:
        anchors.append(t)
        t -= horizon
    anchors.reverse()
    per_backend = {name: {"cov80": [], "cov50": [], "crps": [], "width": []} for name in via}
    realized_rows = []
    for t in anchors:
        hist = bars[t - fit_window:t]
        realized = bars[t + horizon - 1][2]
        realized_rows.append({"anchor_date": str(hist[-1][0])[:10], "realized": realized})
        for name in via:
            be = backends.get(name)
            terminals = sorted(be["fn"](hist, horizon, n_paths, seed + t))
            n = len(terminals)
            q = lambda p: terminals[min(int(p * n), n - 1)]
            p10, p25, p75, p90 = q(0.10), q(0.25), q(0.75), q(0.90)
            per_backend[name]["cov80"].append(p10 <= realized <= p90)
            per_backend[name]["cov50"].append(p25 <= realized <= p75)
            per_backend[name]["crps"].append(empirical_crps(terminals, realized) / hist[-1][2])
            per_backend[name]["width"].append((p90 - p10) / hist[-1][2])
    results = {}
    for name, m in per_backend.items():
        n = len(m["cov80"])
        results[name] = {
            "kind": backends.REGISTRY[name]["kind"],
            "desc": backends.REGISTRY[name]["desc"],
            "coverage_80": sum(m["cov80"]) / n, "coverage_50": sum(m["cov50"]) / n,
            "mean_crps_rel": sum(m["crps"]) / n, "mean_width_rel": sum(m["width"]) / n,
        }
    verdict_lines = []
    base = results.get("bootstrap")
    for name, r in results.items():
        if name == "bootstrap" or not base:
            continue
        delta = base["mean_crps_rel"] - r["mean_crps_rel"]  # >0 means backend beats bootstrap
        better = delta > 0
        significant = better and delta > 0.05 * base["mean_crps_rel"]  # >5% relative CRPS edge
        role = "neural" if r["kind"] == "neural" else "baseline variant"
        verdict_lines.append(
            f"- {name} ({role}): CRPS {'BETTER' if better else 'WORSE'} than bootstrap by "
            f"{abs(delta):.4%} rel ({'significant' if significant else 'NOT significant'} at the 5% edge); "
            + (f"{'COUNTS' if significant else 'DOES NOT COUNT'} — neural must significantly beat bootstrap"
               if r["kind"] == "neural" else "reported for contrast, bootstrap stays the anchor"))
    report = {
        "kind": "comparison", "horizon": horizon, "anchors": len(anchors),
        "fit_window": fit_window, "paths_per_anchor": n_paths,
        "seed_policy": "seed + anchor_index — identical draws offered to every backend",
        "protocol": "same anchor windows, same realized terminals, same metrics; all sides reported",
        "results": results, "verdict": verdict_lines or ["bootstrap only — nothing to compare"],
        "disclosure": "rolling OOS on a single symbol/window family; small anchor counts are "
                      "reported as-is and generalize to nothing; CRPS is scale-free (relative "
                      "to the anchor close); 'significant' here means a >5% relative CRPS edge, "
                      "a declared threshold — not a statistical test.",
    }
    return report, realized_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--via", default="bootstrap,gaussian",
                    help="comma-separated backends: bootstrap,gaussian,kronos")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--fit-window", type=int, default=250)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.data))
    symbol, bars = next(iter(data.items()))
    via = [v.strip() for v in args.via.split(",") if v.strip()]
    unavailable = []
    runnable = []
    for name in via:
        try:
            if name == "kronos":
                import torch  # noqa: F401
            runnable.append(name)
        except ImportError:
            unavailable.append(name)
    if unavailable:
        print(f"⚠ backends unavailable on this host (reported honestly, skipped): "
              f"{', '.join(unavailable)}", file=sys.stderr)
    if "bootstrap" not in runnable:
        runnable.insert(0, "bootstrap")  # the anchor is never optional
    report, rows = compare(bars, args.horizon, args.paths, args.seed,
                           args.anchors, args.fit_window, runnable)
    report["symbol"] = symbol
    report["source"] = {"origin": "user-provided klines JSON", "bars": len(bars),
                        "coverage_gaps": coverage_gaps(bars)}
    print(f"## comparison · {symbol} · {report['anchors']} anchors × {args.horizon} steps "
          f"· backends: {', '.join(runnable)}\n")
    print("| backend | kind | cov80 | cov50 | CRPS rel | width |")
    print("|---|---|---|---|---|---|")
    for name, r in report["results"].items():
        print(f"| {name} | {r['kind']} | {r['coverage_80']:.0%} | {r['coverage_50']:.0%} "
              f"| {r['mean_crps_rel']:.4%} | {r['mean_width_rel']:.1%} |")
    print("\n## verdict")
    for line in report["verdict"]:
        print(line)
    print(f"\n{report['disclosure']}")
    print("\n★ github.com/vicfei/dsh-Kronos")
    if args.save:
        os.makedirs("scenarios", exist_ok=True)
        path = f"scenarios/{symbol.replace('/', '_')}-compare-h{args.horizon}.json"
        json.dump(report, open(path, "w"), ensure_ascii=False, indent=1, default=str)
        print(f"→ {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
