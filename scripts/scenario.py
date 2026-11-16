#!/usr/bin/env python3
"""K 线情景引擎（Kronos 概念位 M0：统计基线，零依赖）

上游 Kronos（47.9k★，MIT）=「金融市场的语言基础模型」：K 线序列的神经
自回归模型，48K 上下文，生成未来 OHLCV。本插件 v0 提供其概念位的
**统计基线**：stationary block bootstrap 情景生成——从真实历史收益率块
重采样，保留波动聚集的近似结构，产出确定性可复现的分位数情景扇。

诚实披露（不可剥离）：这是**统计基线而非学习式基础模型**；上游神经模型
权重（HF）接入为 M1（--via kronos），届时本基线恰好充当对照锚——
神经模型的增量价值须显著超越 bootstrap 才算数（家族对拍血统）。

用法：
  python3 scripts/scenario.py --data /tmp/btc_365.json --horizon 30 --paths 1000 --seed 42 --save
"""
import argparse, datetime as dt, json, math, random


def bootstrap_scenarios(bars, horizon, n_paths, block_len, seed):
    closes = [b[2] for b in bars]
    logrets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    rng = random.Random(seed)
    n = len(logrets)
    last = closes[-1]
    terminals, paths = [], []
    for _ in range(n_paths):
        path, pos = [last], 0
        while len(path) <= horizon:
            start = rng.randrange(n)
            for k in range(block_len):
                if len(path) > horizon:
                    break
                path.append(path[-1] * math.exp(logrets[(start + k) % n]))
        paths.append(path)
        terminals.append(path[-1])
    terminals.sort()
    def q(p): return terminals[min(int(p * n_paths), n_paths - 1)]
    return {"last_close": last, "horizon": horizon, "paths": n_paths,
            "block_len": block_len, "seed": seed,
            "terminal_P10": q(0.10), "terminal_P25": q(0.25), "terminal_P50": q(0.50),
            "terminal_P75": q(0.75), "terminal_P90": q(0.90),
            "P_above_current": sum(1 for t in terminals if t > last) / n_paths,
            "hist_span": f"{bars[0][0]} → {bars[-1][0]}",
            "_paths": paths}


def fan_svg(paths, last, horizon, title, w=720, h=240):
    import statistics
    steps = list(range(horizon + 1))
    def quantiles_at(i):
        xs = sorted(p[i] for p in paths)
        n = len(xs)
        return xs[int(0.10 * n)], xs[int(0.50 * n)], xs[int(0.90 * n)]
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
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="monospace">'
            f'<text x="10" y="14" font-size="12">{title} · median with P25–P75 / P10–P90 fans</text>'
            + "".join(bands) + f'<polyline points="{med}" fill="none" stroke="#1a3a8a" stroke-width="1.6"/>'
            f'<line x1="40" y1="{Y(last):.1f}" x2="{w-20}" y2="{Y(last):.1f}" stroke="#c23531" stroke-dasharray="4,3"/>'
            f'<text x="10" y="{h-6}" font-size="10" fill="#567">red dashed = current {last:.2f} · {horizon} steps · bootstrap scenario fan</text></svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--block_len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.data))
    symbol, bars = next(iter(data.items()))
    out = bootstrap_scenarios(bars, args.horizon, args.paths, args.block_len, args.seed)
    paths = out.pop("_paths")
    out.update({"symbol": symbol, "method": "stationary block bootstrap on log-returns",
                "disclosure": "统计基线（非学习式基础模型）：块重采样近似波动聚集，不学习条件结构；"
                              "神经模型（上游 Kronos，HF 权重）接入为 M1，届时本基线为对照锚",
                "fetched_at": dt.datetime.now().isoformat(timespec="seconds")})
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if args.save:
        import os
        os.makedirs("scenarios", exist_ok=True)
        html = ("<!doctype html><meta charset=utf-8><body style='font-family:monospace;max-width:760px;margin:20px auto'>"
                f"<h3>{symbol} · {args.horizon}-step scenario fan</h3>"
                + fan_svg(paths, out["last_close"], args.horizon, symbol)
                + "<pre>" + json.dumps({k: v for k, v in out.items() if k != "disclosure"}, ensure_ascii=False, indent=1) + "</pre>"
                + f"<p style='color:#666;font-size:12px'>{out['disclosure']}</p></body>")
        path = f"scenarios/{symbol.replace('/', '_')}-h{args.horizon}-s{args.seed}.html"
        open(path, "w").write(html)
        print(f"→ {path}", file=sys.stderr)


if __name__ == "__main__":
    import sys
    main()
