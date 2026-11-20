#!/usr/bin/env python3
"""K-line scenario engine (Kronos concept seat: statistical baseline, zero-dependency)

Upstream Kronos (47.9k★, MIT) = "a foundation model for the language of
financial markets": an autoregressive neural model over K-line sequences that
generates future OHLCV. This plugin provides the concept seat's
**statistical baseline**: stationary block bootstrap scenario generation —
resampling blocks of real historical log-returns, approximately preserving
volatility clustering, producing a deterministic, reproducible quantile
scenario fan.

Modes: single symbol (default) · joint multi-symbol (--joint: shared block
draws preserve realized co-movement) · vol-regime conditioning (--condition
vol: block pool restricted to a band around current volatility, disclosed).

Honest disclosure (never detaches): this is a **statistical baseline, not a
learned foundation model**; the neural backend (upstream Kronos, HF weights)
is optional (--via kronos on compare.py) and must significantly beat this
baseline to count (the family's 对拍 lineage).

Usage:
  python3 scripts/scenario.py --data /tmp/btc_365.json --horizon 30 --paths 1000 --seed 42 --save
  python3 scripts/scenario.py --data multi.json --joint          # shared-draw joint fan
  python3 scripts/scenario.py --data btc.json --condition vol    # regime-conditioned pool
"""
import argparse, datetime as dt, json, math, random, sys

# English is the first language (family rule); Chinese is the --lang zh fallback.
LANG = "en"
STR = {
    "en": {
        "disclosure": "Statistical baseline (not a learned foundation model): block resampling "
                      "approximates volatility clustering but learns no conditional structure; "
                      "the neural backend (upstream Kronos, HF weights) is optional and must "
                      "significantly beat this baseline to count",
        "fan_title": "{symbol} · {horizon}-step scenario fan",
        "card_footer": "red dashed = current {last:.2f} · {horizon} steps · bootstrap scenario fan",
        "envelope_line": "seed {seed} · {paths} paths · block {block} · window {bars} bars — reproduce with the same seed",
        "saved": "→ {path}",
        "objective_written": "→ objective {path}",
        "method": "stationary block bootstrap on log-returns",
    },
    "zh": {
        "disclosure": "统计基线（非学习式基础模型）：块重采样近似波动聚集，不学习条件结构；"
                      "神经后端（上游 Kronos，HF 权重）为可选，接入后须显著赢过本基线才算数",
        "fan_title": "{symbol} · {horizon} 步情景扇",
        "card_footer": "红虚线 = 现价 {last:.2f} · {horizon} 步 · bootstrap 情景扇",
        "envelope_line": "种子 {seed} · {paths} 条路径 · 块长 {block} · 窗口 {bars} 根——同种子可复现",
        "saved": "→ {path}",
        "objective_written": "→ 目标文件 {path}",
        "method": "对数收益率的平稳块 bootstrap",
    },
}


def t(key, **kw):
    return STR[LANG].get(key, key).format(**kw) if kw else STR[LANG].get(key, key)


MIN_BARS = 60
MIN_PATHS = 100


def check_preconditions(bars, horizon, n_paths):
    """honest pre-flight: state the reason, never run on degenerate inputs"""
    problems = []
    if len(bars) < MIN_BARS:
        problems.append(f"insufficient history: {len(bars)} bars, need >= {MIN_BARS}")
    if any(not math.isfinite(b[2]) or b[2] <= 0 for b in bars):
        problems.append("non-finite or non-positive close in input bars")
    if horizon < 1:
        problems.append(f"horizon must be >= 1, got {horizon}")
    if n_paths < MIN_PATHS:
        problems.append(f"paths must be >= {MIN_PATHS} for stable quantiles, got {n_paths}")
    return problems


def coverage_gaps(bars, threshold=10):
    """calendar-day jumps above threshold, reported as possible data holes"""
    gaps = []
    for i in range(1, len(bars)):
        try:
            a = dt.date.fromisoformat(str(bars[i - 1][0])[:10])
            b = dt.date.fromisoformat(str(bars[i][0])[:10])
        except ValueError:
            continue
        if (b - a).days > threshold:
            gaps.append({"from": str(bars[i - 1][0])[:10], "to": str(bars[i][0])[:10],
                         "calendar_days": (b - a).days})
    return gaps


def _rolling_vol(rs, win=20):
    vols = [None] * len(rs)
    for i in range(len(rs)):
        lo = max(0, i - win + 1)
        seg = rs[lo:i + 1]
        if len(seg) >= 5:
            m = sum(seg) / len(seg)
            vols[i] = math.sqrt(sum((r - m) ** 2 for r in seg) / (len(seg) - 1))
    return vols


def allowed_starts(rs, band=(0.5, 2.0), win=20, min_pool=20):
    """vol-regime conditioning: block starts whose trailing vol lies within
    [band]× of the current trailing vol; honest unavailable when the pool is small."""
    vols = _rolling_vol(rs, win)
    cur = vols[-1]
    if cur is None or cur <= 0:
        return None, "current trailing vol unavailable (window too short or zero)"
    pool = [i for i, v in enumerate(vols) if v is not None and band[0] * cur <= v <= band[1] * cur]
    if len(pool) < min_pool:
        return None, (f"conditioned pool too small: {len(pool)} blocks (< {min_pool}) within "
                      f"[{band[0]}×,{band[1]}×] of current vol — widen the band or the window")
    return pool, f"vol-regime band [{band[0]}×,{band[1]}×] of current {win}d vol; pool {len(pool)} of {len(rs)} blocks"


def bootstrap_scenarios(bars, horizon, n_paths, block_len, seed, condition=None):
    closes = [b[2] for b in bars]
    logrets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    rng = random.Random(seed)
    n = len(logrets)
    last = closes[-1]
    cond_desc = "none"
    if condition == "vol":
        pool, cond_desc = allowed_starts(logrets)
        if pool is None:
            sys.exit(f"refused honestly — conditioning unavailable: {cond_desc}")
        pick = lambda: pool[rng.randrange(len(pool))]
    else:
        pick = lambda: rng.randrange(n)
    terminals, paths = [], []
    for _ in range(n_paths):
        path = [last]
        while len(path) <= horizon:
            start = pick()
            for k in range(block_len):
                if len(path) > horizon:
                    break
                path.append(path[-1] * math.exp(logrets[(start + k) % n]))
        paths.append(path)
        terminals.append(path[-1])
    terminals.sort()
    def q(p): return terminals[min(int(p * n_paths), n_paths - 1)]
    out = {"last_close": last, "horizon": horizon, "paths": n_paths,
           "block_len": block_len, "seed": seed,
           "terminal_P10": q(0.10), "terminal_P25": q(0.25), "terminal_P50": q(0.50),
           "terminal_P75": q(0.75), "terminal_P90": q(0.90),
           "P_above_current": sum(1 for x in terminals if x > last) / n_paths,
           "hist_span": f"{bars[0][0]} → {bars[-1][0]}",
           "_paths": paths}
    if condition:
        out["_conditioning"] = cond_desc
    return out


def joint_scenarios(symbols_bars, horizon, n_paths, block_len, seed):
    """shared-index block resampling across symbols (V3.1): ONE sequence of
    block-start draws drives every symbol's own logret series, so realized
    co-movement of the same historical dates is preserved in the scenarios.
    Alignment assumption (disclosed): equal-length series, position-aligned."""
    series, lasts = {}, {}
    for sym, bars in symbols_bars.items():
        closes = [b[2] for b in bars]
        series[sym] = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        lasts[sym] = closes[-1]
    lens = {len(v) for v in series.values()}
    if len(lens) != 1:
        sys.exit(f"refused honestly — joint mode requires equal-length series, got {sorted(lens)}")
    n = next(iter(lens))
    rng = random.Random(seed)
    terminals = {s: [] for s in series}
    for _ in range(n_paths):
        levels = dict(lasts)
        steps = 0
        while steps < horizon:
            start = rng.randrange(n)  # ONE draw, shared across symbols
            for k in range(block_len):
                if steps >= horizon:
                    break
                for s, rs in series.items():
                    levels[s] *= math.exp(rs[(start + k) % n])
                steps += 1
        for s in series:
            terminals[s].append(levels[s])
    def pearson(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        return num / (da * db) if da and db else None
    out = {"kind": "joint", "horizon": horizon, "paths": n_paths,
           "block_len": block_len, "seed": seed, "symbols": sorted(series),
           "alignment": "position-aligned equal-length series (same dates assumed)",
           "per_symbol": {}, "sim_terminal_corr": {}, "note":
           "shared block-start draws preserve realized co-movement; correlation is of "
           "simulated terminals across symbols, not a forecast of dependence"}
    syms = sorted(series)
    for s in syms:
        ts = sorted(terminals[s])
        q = lambda p: ts[min(int(p * n_paths), n_paths - 1)]
        out["per_symbol"][s] = {"last_close": lasts[s],
                                "terminal_P10": q(0.10), "terminal_P50": q(0.50), "terminal_P90": q(0.90),
                                "P_above_current": sum(1 for x in terminals[s] if x > lasts[s]) / n_paths}
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            out["sim_terminal_corr"][f"{a}|{b}"] = pearson(terminals[a], terminals[b])
    return out


def fan_svg(paths, last, horizon, title, w=720, h=240, envelope=None):
    import statistics
    steps = list(range(horizon + 1))
    lo = min(min(p) for p in paths) * 0.995
    hi = max(max(p) for p in paths) * 1.005
    X = lambda i: 40 + i * (w - 60) / horizon
    Y = lambda v: 20 + (hi - v) * (h - 60) / (hi - lo + 1e-9)
    bands = []
    for band in (0.10, 0.25):
        pts = []
        for i in steps:
            xs = sorted(p[i] for p in paths)
            pts.append((xs[int(band * len(xs))], xs[int((1 - band) * len(xs))]))
        up = " ".join(f"{X(i):.1f},{Y(u):.1f}" for i, (l_, u) in enumerate(pts))
        dn = " ".join(f"{X(i):.1f},{Y(l_):.1f}" for i, (l_, u) in reversed(list(enumerate(pts))))
        bands.append(f'<polygon points="{up} {dn}" fill="#2456c4" opacity="{0.10 if band == 0.10 else 0.18}"/>')
    med = " ".join(f"{X(i):.1f},{Y(statistics.median([p[i] for p in paths])):.1f}" for i in steps)
    footer = t("card_footer", last=last, horizon=horizon)
    env_line = t("envelope_line", seed=envelope["seed"], paths=len(paths),
                 block=envelope["block_len"], bars=envelope["window"]["bars"]) if envelope else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h+16}" font-family="monospace">'
            f'<text x="10" y="14" font-size="12">{title} · median with P25–P75 / P10–P90 fans</text>'
            + "".join(bands) + f'<polyline points="{med}" fill="none" stroke="#1a3a8a" stroke-width="1.6"/>'
            + f'<line x1="40" y1="{Y(last):.1f}" x2="{w-20}" y2="{Y(last):.1f}" stroke="#c23531" stroke-dasharray="4,3"/>'
            + f'<text x="10" y="{h-6}" font-size="10" fill="#567">{footer}</text>'
            + f'<text x="10" y="{h+10}" font-size="10" fill="#567">{env_line}</text></svg>')


def _write_objective(path, block, symbol, horizon):
    obj = {"symbol": symbol, "horizon": horizon,
           "P_above_current": block["P_above_current"],
           "terminal_P10": block.get("terminal_P10"), "terminal_P50": block.get("terminal_P50"),
           "terminal_P90": block.get("terminal_P90"),
           "fan_width_80_rel": ((block.get("terminal_P90") or 0) - (block.get("terminal_P10") or 0))
                               / block.get("last_close", 1),
           "note": "optimizer objective interface (V3.2): optimize distribution metrics "
                   "(P_above / width), not point predictions; family dsh-jesse consumption"}
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=1)
    print(t("objective_written", path=path), file=sys.stderr)


def main():
    global LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--block_len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--condition", choices=("none", "vol"), default="none",
                    help="condition the block pool (vol = trailing-vol regime band; disclosed)")
    ap.add_argument("--joint", action="store_true",
                    help="joint multi-symbol scenarios: shared block draws preserve co-movement")
    ap.add_argument("--objective-out", metavar="FILE",
                    help="write an optimizer-objective JSON (P_above, quantiles, width)")
    ap.add_argument("--lang", choices=("en", "zh"), default="en",
                    help="output language (default en; zh is the fallback)")
    args = ap.parse_args()
    LANG = args.lang
    data = json.load(open(args.data))
    symbol, bars = next(iter(data.items()))
    problems = check_preconditions(bars, args.horizon, args.paths)
    if problems:
        sys.exit("refused honestly — " + "; ".join(problems))

    if args.joint or len(data) > 1:
        joint = joint_scenarios(data, args.horizon, args.paths, args.block_len, args.seed)
        joint["meta"] = {"method": t("method"), "seed": args.seed, "n_paths": args.paths,
                         "block_len": args.block_len,
                         "window": {"bars": len(bars), "from": str(bars[0][0])[:10],
                                    "to": str(bars[-1][0])[:10]},
                         "conditioning": "none (joint draws shared across symbols)",
                         "generated_at": dt.datetime.now().isoformat(timespec="seconds")}
        joint["disclosure"] = t("disclosure")
        print(json.dumps(joint, ensure_ascii=False, indent=1, default=str))
        print("★ github.com/vicfei/dsh-Kronos", file=sys.stderr)
        if args.objective_out:
            first = sorted(joint["per_symbol"])[0]
            _write_objective(args.objective_out, joint["per_symbol"][first], first, args.horizon)
        if args.save:
            import os
            os.makedirs("scenarios", exist_ok=True)
            path = f"scenarios/joint-h{args.horizon}-s{args.seed}.json"
            json.dump(joint, open(path, "w"), ensure_ascii=False, indent=1, default=str)
            print(t("saved", path=path), file=sys.stderr)
        return

    out = bootstrap_scenarios(bars, args.horizon, args.paths, args.block_len, args.seed,
                              condition=None if args.condition == "none" else args.condition)
    cond_desc = out.pop("_conditioning", None)
    paths = out.pop("_paths")
    gaps = coverage_gaps(bars)
    out.update({"symbol": symbol,
                "disclosure": t("disclosure"),
                "meta": {"method": t("method"), "seed": args.seed, "n_paths": args.paths,
                         "block_len": args.block_len,
                         "window": {"bars": len(bars), "from": str(bars[0][0])[:10],
                                    "to": str(bars[-1][0])[:10]},
                         "conditioning": cond_desc or "none",
                         "generated_at": dt.datetime.now().isoformat(timespec="seconds")},
                "source": {"origin": "user-provided klines JSON", "symbol": symbol,
                           "bars": len(bars),
                           "date_range": [str(bars[0][0])[:10], str(bars[-1][0])[:10]],
                           "coverage_gaps": gaps}})
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print("★ github.com/vicfei/dsh-Kronos", file=sys.stderr)
    if args.objective_out:
        _write_objective(args.objective_out, out, symbol, args.horizon)
    if args.save:
        import os
        os.makedirs("scenarios", exist_ok=True)
        title = t("fan_title", symbol=symbol, horizon=args.horizon)
        html = ("<!doctype html><meta charset=utf-8><body style='font-family:monospace;max-width:760px;margin:20px auto'>"
                f"<h3>{title}</h3>"
                + fan_svg(paths, out["last_close"], args.horizon, symbol, envelope=out["meta"])
                + "<pre>" + json.dumps({k: v for k, v in out.items() if k != "disclosure"}, ensure_ascii=False, indent=1) + "</pre>"
                + f"<p style='color:#666;font-size:12px'>{out['disclosure']}</p></body>")
        path = f"scenarios/{symbol.replace('/', '_')}-h{args.horizon}-s{args.seed}.html"
        open(path, "w").write(html)
        print(t("saved", path=path), file=sys.stderr)


if __name__ == "__main__":
    main()
