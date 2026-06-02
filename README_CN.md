<div align="center">

# PRIME：超越黑箱
### 一个面向可解释选股的能量统一框架

**P**otential **R**obust **I**ntegrated **M**acro **E**nergy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![KDD 2026](https://img.shields.io/badge/KDD-2026-7b3fe4.svg)](https://kdd.org/)

KDD 2026 投稿配套代码 ｜ [English](README.md) · [中文说明](README_CN.md)

</div>

---

## 简介

从大规模市场数据中进行量化选股是获取超额收益的核心，但**尾部风险**始终是难题：
极端事件会触发非平稳的市场状态切换，使学到的模式失效、扭曲组合逻辑。多数深度学习
方法是黑箱，既无法刻画尾部事件**如何**重塑市场动态，也缺乏可解释的应对工具。

**PRIME** 从动量视角出发，把收益生成与尾部风控统一到**能量模型**与**博弈论**框架中：

- **语义编码器**将市场力量分解为*多头动量*、*空头阻力*、*摩擦耗散*，把个股估值表达为
  多空博弈中的一个**能量状态**；
- **宏观感知调制**通过可学习系数 (α, β, γ) 在牛熊切换中重塑能量景观，使打分几何自适应
  尾部驱动的非平稳性；
- **聚合**采用**势博弈（potential game）**，其纳什均衡性质保证在分布漂移下排序稳定；
- **风险卫士（Risk Guardian）**闭环检测异常能量尖峰作为崩盘信号，并在组合构建中过滤。

在 **S&P 500** 与 **CSI 500** 基准上的实验（含大量消融）表明，PRIME 相较 SOTA 基线约有
**10% 的提升**，并具备收敛性与排序一致性的理论保证。

> **可解释性**：把黑箱预测转化为势函数后，能量最小化让投资者能把每次选股归因到
> *强动量 / 弱阻力 / 低摩擦*。

---

## 核心贡献

1. **能量特征编码范式** —— 把黑箱预测转化为可解释的能量分解，实现有原则的选股。
2. **宏观状态调制机制** —— 用能量语义调制，使选股逻辑动态适应宏观经济状态（牛/中性/熊）。
3. **严格的理论保证** —— Gibbs 分布的最大熵推导、成对学习可解性、排序一致性收敛、
   有界能量设计，以及 *O(1/T)* 的优化收敛速率。

---

## 方法概览：四阶段主链路

架构在所有市场间保持不变，只切换市场 preset（预设窗口、成本、特征语义）。

| 阶段 | 名称 | 作用 |
|:----:|------|------|
| **1** | Sanity Check | 校验特征/标签，预热编码器。 |
| **2** | Game EBM 排序 | 在势博弈目标下学习能量分解与成对排序。 |
| **3** | Risk Guardian 风控 | 挖掘崩盘困难样本，针对模型盲点训练 GBDT 崩盘过滤器。 |
| **4** | Integrator + 回测 | 融合排序与卫士构建组合，进行考虑成本的回测。 |

**特征分解（44 维，4 个语义组）：**

```
多头 Bull  (16) ─ 动量 / RSI / 资金流入 / 成长 / 筹码支撑 …  → E_bull   (动量)
空头 Bear  (10) ─ 估值排名 / 乖离 / 套牢比 / 阻力 …          → E_bear   (阻力)
摩擦 Frict ( 9) ─ 换手 / 波动 / 振幅 / 成本离散 …            → E_friction
宏观 Macro ( 9) ─ CPI / PPI / PMI / 利率 / 市场波动·流动性 … → α, β, γ 调制
```

同一套 44 维结构通过语义字段重映射复用到 **ETF**（如*资金流入→份额流入*、
*筹码支撑→份额增长*），`GameEnergyModel` 无需任何架构改动即可切换资产类别。

---

## 目录结构

```text
prime_kdd_2026/
├── README.md / README_CN.md      # 文档（英文 / 中文）
├── LICENSE                       # MIT
├── requirements.txt
├── CITATION.cff
├── prime/                        # ── 源代码 ───────────────────────────────
│   ├── main.py                   # 统一 CLI 入口
│   ├── config.py                 # Config / Feature / Training / Backtest 配置
│   ├── market_profiles.py        # 市场预设（CSI500 / SP500 / Nikkei / STOXX）
│   ├── data_loader.py            # 数据管线 + market-aware mock 生成器
│   ├── dataset.py                # StockDataset / PairwiseDataset / 数据加载器
│   ├── trainer.py                # 四阶段训练编排
│   ├── backtest.py               # 含成本的组合回测
│   ├── experiment_runner.py      # 主实验 / 消融 / 鲁棒性
│   ├── experiment_catalog.py     # 实验注册表
│   ├── integrity_check.py        # 一条命令端到端 smoke 检查
│   ├── visualization.py          # 全部论文图表与诊断图
│   ├── models/                   # energy_model · risk_guardian · losses
│   └── baselines/                # 10 个基线（LSTM … TIME-LLM、DeepTrader）
├── data/
│   ├── README.md                 # 数据格式、获取方式与 mock 数据政策
│   └── example/                  # 小型合成示例面板（演示 schema）
├── figures/                      # 精选论文图表（含 supplementary/）
└── scripts/                      # 一键复现脚本
```

> **数据政策**：本仓库**不包含任何原始金融数据**，仅含代码、配置、小型**合成示例数据**、
> 论文图表与复现脚本。真实数据缺失时，PRIME 会**自动生成 market-aware 的 mock 数据**，
> 保证所有命令开箱即跑。详见 [`data/README.md`](data/README.md)。

---

## 安装

```bash
git clone https://github.com/XXXiaoNick/prime_kdd_2026.git
cd prime_kdd_2026

# （推荐）创建隔离环境
python -m venv .venv && source .venv/bin/activate    # 或 conda create -n prime python=3.10

pip install -r requirements.txt
```

`torch` 是唯一较重的依赖；GPU 非必需但推荐用于完整训练。`lightgbm` 为可选项 ——
未安装时 Risk Guardian 会自动回退到 scikit-learn 分类器（如需完全复现论文请安装）。

---

## 快速开始

所有命令在仓库根目录运行，代码包位于 `prime/`。

```bash
# 0) 端到端快速自检（使用自动生成的 mock 数据）
python prime/integrity_check.py --quick --skip_baselines

# 1) 在 CSI 500 股票上训练 + 回测 PRIME（无数据则自动生成 mock）
python prime/main.py --mode train --market_profile csi500 --asset_type stock --fast

# 2) S&P 500
python prime/main.py --mode train --market_profile sp500  --asset_type stock --fast

# 3) ETF 模式
python prime/main.py --mode train --market_profile csi500 --asset_type etf   --fast

# 4) 回测已保存的检查点
python prime/main.py --mode backtest --checkpoint outputs/checkpoints/<run_name>
```

**使用自己的数据**（可选）：把面板放到
`data/<market>/panel/panel_data_complete.parquet`（或 `.csv`），或用 `--data_root /abs/path`
/ 环境变量 `PRIME_DATA_ROOT` 指定。schema 见 [`data/README.md`](data/README.md)
（`data/example/` 中的示例面板即为现成模板）。

---

## 复现论文

```bash
bash scripts/run_main_experiments.sh   # 主实验：市场套件 / ETF 泛化 / 跨地域 / 案例研究
bash scripts/run_ablation.sh           # 消融：module / feature_grouping / aggregation / guardian
bash scripts/run_robustness.sh         # 鲁棒性：noise / topk / lr / epochs / 滚动窗口 …
bash scripts/run_baselines.sh          # 基线：LSTM … TIME-LLM、AlphaStock、DeepTrader
```

查看各注册表：

```bash
python prime/main.py --list_main_experiments
python prime/main.py --list_ablations
python prime/main.py --list_robustness
python prime/main.py --list_baselines
```

结果（指标、净值曲线、持仓、检查点、图表）写入 `outputs/` 与 `outputs_baselines/`
（已在 `.gitignore` 中默认忽略）。

---

## 图表

论文图表在 [`figures/`](figures/)（PNG + 矢量 PDF），图号映射见
[`figures/README.md`](figures/README.md)：Fig.1 理论能量景观、Fig.3 博弈相空间与卫士判别、
Fig.4 经验能量动态、Fig.5 状态相关能量景观、Fig.6 状态相关能量相关性、Fig.7 计算效率分析。
消融/鲁棒性/学习曲线图在 [`figures/supplementary/`](figures/supplementary/)。

---

## 市场与基线

**市场**（`prime/market_profiles.py` 预设）：`csi500`、`sp500`、`nikkei225`、`stoxx600`，
各有独立的交易窗口、成本与 mock 风格。**资产类型**：`stock`、`etf`。

**基线**（`prime/baselines/`）：Market Index、LSTM、ALSTM、GRU、Transformer、PatchTST、
GPT4TS、TIME-LLM、AlphaStock、DeepTrader。

---

## 引用

```bibtex
@inproceedings{prime2026,
  title     = {Beyond Black Boxes: An Energy-Based Unified Framework for Interpretable Stock Selection},
  author    = {Anonymous},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2026}
}
```

---

## 许可证

基于 [MIT 许可证](LICENSE) 发布。代码仅供研究与教育用途，**不构成任何投资建议**。
