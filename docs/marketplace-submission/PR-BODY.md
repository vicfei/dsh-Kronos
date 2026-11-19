# PR body（提交 awesome-dsh-plugin 用——owner 发布 npm 0.3.0 后执行）

**Adds vicfei/dsh-Kronos — K-line scenario forecasting, the baseline any neural FM must beat (tools)**

One file added: `data/plugins/vicfei__dsh-Kronos--npm.yml`.

- Install: `dsh plugin add dsh-kronos` (npm 0.3.0; dsh.bundle + skill + CLI
  spawning the stdlib-only Python engine)
- What it does: seeded, reproducible scenario fans from real history
  (stationary block bootstrap); the reproducibility envelope rides on every
  payload; a rolling OOS self-evaluation harness scores the fan itself
  (coverage / CRPS); and an honest head-to-head layer where any backend —
  including the optional upstream Kronos HF weights (`--via kronos`) —
  **must significantly beat bootstrap to count**. Joint multi-symbol
  scenarios preserve realized co-movement; vol-regime conditioning is
  disclosed with its pool.
- Real working code: 31 hermetic checks + CI green; CRPS verified against
  the analytic N(0,1) value; live smokes on real BTC data throughout.
- Concept slot of shiyu-coder/Kronos (MIT upstream, no upstream code);
  independent native re-implementation. Not investment advice. Family:
  dsh-TradingAgents · dsh-Vibe-Trading · dsh-jesse …

PR 执行命令（owner 在 npm 0.3.0 发布后跑）：

```bash
cd /tmp && git clone --depth 1 git@ssh.github.com:vicfei/awesome-dsh-plugin.git && cd awesome-dsh-plugin
git checkout -b add-dsh-kronos
cp ~/dsh-kronos/docs/marketplace-submission/vicfei__dsh-Kronos--npm.yml data/plugins/
git add data/plugins/vicfei__dsh-Kronos--npm.yml
git commit -m "Add vicfei/dsh-Kronos (K-line scenario forecasting)"
git push -f origin add-dsh-kronos
# 开 PR：base awesome-dsh-plugin/awesome-dsh-plugin ← vicfei:add-dsh-kronos，body 用上文
```
