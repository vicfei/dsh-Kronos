# dsh-Kronos — DeepSeek Harness 的 K 线情景预测

**从真实历史生成可复现的未来 K 线情景扇——任何神经基础模型都必须先打赢的
统计基线。零依赖、按种子确定性、披露随载荷。**

[English](README.md) | [简体中文](README.zh-CN.md)

[![DeepSeek Harness](https://img.shields.io/badge/DSH-plugin-4E6DD4)](https://github.com/deepseek-ai) [![License: MIT](https://img.shields.io/badge/license-MIT-3da639)](LICENSE) [![Kronos MIT](https://img.shields.io/badge/upstream-Kronos%20MIT-4f46e5)](https://github.com/shiyu-coder/Kronos)

> [Kronos](https://github.com/shiyu-coder/Kronos)（47.9k★，"金融市场的语言
> 基础模型"）的概念位，DSH 原生。**v0 交付统计基线（平稳块 bootstrap）而非
> 神经模型**：诚实披露随每个载荷输出；真模型接入后（`--via kronos`，HF 权重）
> 本基线即对照锚——神经须赢过 bootstrap 才算数。无隶属。
> 家族：[dsh-TradingAgents](https://github.com/vicfei/dsh-TradingAgents) ·
> [dsh-Vibe-Trading](https://github.com/vicfei/dsh-Vibe-Trading) ·
> [dsh-jesse](https://github.com/vicfei/dsh-jesse) ·
> [dsh-quantstats](https://github.com/vicfei/dsh-quantstats) 等。

## 为什么（Kronos 是神经模型——为什么先发 bootstrap？）

| | Kronos（神经 FM） | **dsh-Kronos v0（基线）** |
|---|---|---|
| 依赖 | torch + HF 权重（GB 级） | **零**（纯标准库） |
| 确定性 | 采样种子 | **种子化 bootstrap——位级可复现** |
| 运行处 | Python/GPU 实验室 | **DSH 会话内**——agent 循环中直接拿情景扇 |
| 角色 | 学习式条件结构 | **任何学习模型都必须打赢的锚**——权重/网络不在时的诚实兜底 |

## 安装

```bash
npm i -g dsh-kronos             # 或：dsh plugin add dsh-kronos
dsh-kronos --data <klines.json> --horizon 30 --paths 1000 --seed 42 --save
```

需要 PATH 上有 `python3`（纯标准库，无 pip 安装）。源码方式：clone 后直接
`python3 scripts/scenario.py`。

## 快速开始

```bash
git clone https://github.com/vicfei/dsh-Kronos && cd dsh-kronos
python3 scripts/scenario.py --data <klines.json> --horizon 30 --paths 1000 --seed 42 --save
```

真实冒烟（binance BTC/USDT 365 天，2026-11-17，seed 42）：30 步扇，终点
P10/P50/P90 = **53.1k / 62.6k / 73.2k**，P(高于现价) = **38.9%**——市场自身
的历史给下行的定价比上行更宽。同种子位级复现。

输出英文优先；`--lang zh` 切换用户可见标签与披露文本。

## 状态与路线图

v0.3.0——V1 硬化、V2 对拍主轴、V3 深度全部交付；31 项 hermetic 测试 + CI 绿。

**盒内**：带可复现信封的情景扇；自评 harness（滚动 OOS 覆盖率/CRPS）；
对拍层（compare.py——bootstrap vs gaussian vs 可选 `--via kronos`，神经须显
著赢基线才算数）；保联动联合情景；波动 regime 条件化（池披露）；家族优化
器目标接口。见 [ROADMAP](ROADMAP.md)——V4：SCENARIO-REPORT 格式。

## 许可与免责

MIT；MIT 上游的概念位——零上游代码。**情景不是预测；非投资建议。**
