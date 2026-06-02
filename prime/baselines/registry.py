"""
baseline 注册表。

负责维护 baseline 名称、论文来源、默认超参数与输入组织方式，
相当于 baseline 子系统的静态配置中心。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    display_name: str
    family: str
    paper_title: str
    paper_url: str
    code_url: Optional[str]
    summary: str
    seq_len: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    patch_len: int = 8
    stride: int = 4
    prompt_length: int = 8
    freeze_backbone: bool = False
    hf_backbone: Optional[str] = None
    cross_sectional: bool = False
    fidelity: str = "faithful_adaptation"
    notes: List[str] = field(default_factory=list)


BASELINE_SPECS: Dict[str, BaselineSpec] = {
    "market_index": BaselineSpec(
        name="market_index",
        display_name="Market Index",
        family="Benchmark",
        paper_title="Benchmark market index buy-and-hold baseline",
        paper_url="",
        code_url=None,
        summary="Benchmark buy-and-hold baseline using the market index when available, otherwise an equal-weight universe proxy.",
        seq_len=1,
        epochs=0,
        batch_size=1,
        learning_rate=0.0,
        fidelity="benchmark_proxy",
        notes=[
            "Uses the real benchmark index series when available in the local data root.",
            "Falls back to an equal-weight daily universe return proxy when benchmark data is unavailable.",
        ],
    ),
    "lstm": BaselineSpec(
        name="lstm",
        display_name="LSTM",
        family="RNN",
        paper_title="Supervised Sequence Labelling with Recurrent Neural Networks",
        paper_url="https://doi.org/10.1007/978-3-642-24797-2_4",
        code_url=None,
        summary="Vanilla recurrent baseline using a temporal LSTM encoder over per-asset factor sequences.",
        seq_len=20,
        epochs=12,
        batch_size=512,
        learning_rate=1e-3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        notes=[
            "Uses the original LSTM backbone in a unified stock-ranking setting.",
            "Trained on the project's factor panel and evaluated with the same backtest pipeline as PRIME.",
        ],
    ),
    "alstm": BaselineSpec(
        name="alstm",
        display_name="ALSTM",
        family="RNN+Attention",
        paper_title="Enhancing Stock Movement Prediction with Adversarial Training",
        paper_url="https://www.ijcai.org/proceedings/2019/0810.pdf",
        code_url="https://github.com/ClementPerroud/Adv-ALSTM",
        summary="Attentive LSTM backbone from Adv-ALSTM, implemented here without adversarial perturbation for the ALSTM baseline.",
        seq_len=20,
        epochs=12,
        batch_size=384,
        learning_rate=1e-3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.15,
        notes=[
            "Implements the attentive LSTM encoder used as the non-adversarial backbone in Adv-ALSTM.",
            "Adversarial perturbation is intentionally omitted because the paper baseline name is ALSTM, not Adv-ALSTM.",
        ],
    ),
    "gru": BaselineSpec(
        name="gru",
        display_name="GRU",
        family="RNN",
        paper_title="Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling",
        paper_url="https://arxiv.org/abs/1412.3555",
        code_url=None,
        summary="GRU sequence encoder over rolling factor windows.",
        seq_len=20,
        epochs=12,
        batch_size=512,
        learning_rate=1e-3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ),
    "transformer": BaselineSpec(
        name="transformer",
        display_name="Transformer",
        family="Transformer",
        paper_title="Attention Is All You Need",
        paper_url="https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf",
        code_url=None,
        summary="Vanilla Transformer encoder applied to rolling factor sequences.",
        seq_len=32,
        epochs=12,
        batch_size=256,
        learning_rate=8e-4,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        notes=[
            "This is the generic Transformer baseline rather than a stock-specific enhanced transformer.",
            "It shares the same data splits and evaluation pipeline with PRIME for fair comparison.",
        ],
    ),
    "patchtst": BaselineSpec(
        name="patchtst",
        display_name="PatchTST",
        family="Patch Transformer",
        paper_title="A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
        paper_url="https://arxiv.org/abs/2211.14730",
        code_url="https://github.com/yuqinie98/PatchTST",
        summary="Patch-based channel-independent Transformer adapted to financial factor sequences.",
        seq_len=64,
        epochs=12,
        batch_size=192,
        learning_rate=6e-4,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        patch_len=8,
        stride=4,
        dropout=0.1,
        notes=[
            "Retains patch segmentation and channel-independence as the core ideas from PatchTST.",
            "Forecast head is adapted to cross-sectional stock scoring on the project's panel data.",
        ],
    ),
    "gpt4ts": BaselineSpec(
        name="gpt4ts",
        display_name="GPT4TS",
        family="Pretrained LM",
        paper_title="One Fits All: Power General Time Series Analysis by Pretrained LM",
        paper_url="https://arxiv.org/abs/2302.11939",
        code_url="https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All",
        summary="Pretrained GPT-style backbone with learned time-series adapters, following the GPT4TS/FPT recipe.",
        seq_len=64,
        epochs=8,
        batch_size=96,
        learning_rate=5e-4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        patch_len=8,
        stride=4,
        hf_backbone="sshleifer/tiny-gpt2",
        freeze_backbone=True,
        fidelity="lightweight_backbone_adaptation",
        notes=[
            "Defaults to a tiny GPT-2 backbone for practical reproducibility inside this project.",
            "The CLI can switch to a larger Hugging Face GPT-2 family checkpoint for closer paper-style runs.",
        ],
    ),
    "time_llm": BaselineSpec(
        name="time_llm",
        display_name="TIME-LLM",
        family="LLM Reprogramming",
        paper_title="Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
        paper_url="https://arxiv.org/abs/2310.01728",
        code_url="https://github.com/KimMeen/Time-LLM",
        summary="Frozen language model with prompt-as-prefix and input reprogramming for time-series tokens.",
        seq_len=64,
        epochs=8,
        batch_size=96,
        learning_rate=5e-4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        patch_len=8,
        stride=4,
        prompt_length=8,
        hf_backbone="sshleifer/tiny-gpt2",
        freeze_backbone=True,
        fidelity="lightweight_backbone_adaptation",
        notes=[
            "Retains the core Time-LLM design ideas: frozen LM, reprogramming layer, and prompt prefix.",
            "Uses a lightweight GPT-2 family checkpoint by default so it can run in the existing project pipeline.",
        ],
    ),
    "alphastock": BaselineSpec(
        name="alphastock",
        display_name="AlphaStock",
        family="Cross-Asset Attention",
        paper_title="AlphaStock: A Buying-Winners-and-Selling-Losers Investment Strategy using Interpretable Deep Reinforcement Attention Networks",
        paper_url="https://arxiv.org/abs/1908.02646",
        code_url=None,
        summary="Cross-asset attention network with a Sharpe-oriented portfolio objective on each trading date.",
        seq_len=20,
        epochs=10,
        batch_size=1,
        learning_rate=8e-4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        cross_sectional=True,
        fidelity="portfolio_objective_adaptation",
        notes=[
            "Implements the paper's core cross-asset attention and portfolio-style optimization ideas.",
            "The full RL environment is replaced by a differentiable portfolio utility objective to fit the project's shared training and backtest stack.",
        ],
    ),
    "deeptrader": BaselineSpec(
        name="deeptrader",
        display_name="DeepTrader",
        family="RL+Graph",
        paper_title="DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding",
        paper_url="https://doi.org/10.1609/aaai.v35i1.16144",
        code_url="https://github.com/CMACH508/DeepTrader",
        summary="Graph-aware portfolio policy with market-condition embedding and risk-return balancing.",
        seq_len=20,
        epochs=10,
        batch_size=1,
        learning_rate=7e-4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        cross_sectional=True,
        fidelity="portfolio_policy_adaptation",
        notes=[
            "Preserves the two-part idea of asset scoring plus market-condition embedding.",
            "Portfolio policy is trained with a differentiable long-only utility inside the shared backtest stack.",
        ],
    ),
    "deeppocket": BaselineSpec(
        name="deeppocket",
        display_name="DeepPocket",
        family="Graph RL",
        paper_title="Deep Graph Convolutional Reinforcement Learning for Financial Portfolio Management -- DeepPocket",
        paper_url="https://arxiv.org/abs/2105.08664",
        code_url=None,
        summary="Graph-convolutional portfolio policy with compressed asset embeddings and actor-critic style heads.",
        seq_len=20,
        epochs=10,
        batch_size=1,
        learning_rate=7e-4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        cross_sectional=True,
        fidelity="portfolio_policy_adaptation",
        notes=[
            "Retains graph-aware asset interaction and actor-critic style scoring.",
            "Uses differentiable portfolio supervision instead of an online trading simulator loop.",
        ],
    ),
    "alphagat": BaselineSpec(
        name="alphagat",
        display_name="AlphaGAT",
        family="Two-Stage Graph RL",
        paper_title="AlphaGAT: A Two-Stage Learning Approach for Adaptive Portfolio Selection",
        paper_url="https://www.ijcai.org/proceedings/2025/834",
        code_url=None,
        summary="Two-stage alpha-factor mining and graph-attention portfolio selection.",
        seq_len=24,
        epochs=10,
        batch_size=1,
        learning_rate=7e-4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        cross_sectional=True,
        fidelity="two_stage_adaptation",
        notes=[
            "Keeps the paper's two-stage structure: factor extraction followed by graph-based portfolio policy.",
            "Implements a lightweight stage-I temporal factor miner and stage-II graph attention policy within the project pipeline.",
        ],
    ),
    "stockformer": BaselineSpec(
        name="stockformer",
        display_name="StockFormer",
        family="GNN+Transformer",
        paper_title="Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks",
        paper_url="https://github.com/Eric991005/Multitask-Stockformer",
        code_url="https://github.com/Eric991005/Multitask-Stockformer",
        summary="Dual-frequency temporal encoder with stock-wise attention and multi-task regression/classification heads.",
        seq_len=32,
        epochs=10,
        batch_size=1,
        learning_rate=7e-4,
        hidden_dim=160,
        num_layers=3,
        num_heads=4,
        cross_sectional=True,
        fidelity="factor_panel_adaptation",
        notes=[
            "Implements wavelet-style dual-frequency decomposition, stock-wise attention, and multi-task heads.",
            "Graph structure is approximated from industry relations and cross-sectional context available in the project's panel data.",
        ],
    ),
}


def available_baselines() -> List[str]:
    return list(BASELINE_SPECS.keys())


def get_baseline_spec(name: str) -> BaselineSpec:
    key = name.lower()
    if key not in BASELINE_SPECS:
        raise KeyError(f"Unknown baseline: {name}")
    return BASELINE_SPECS[key]


def baseline_registry_lines() -> List[str]:
    lines: List[str] = []
    for spec in BASELINE_SPECS.values():
        lines.append(
            f"{spec.name}: {spec.display_name} | {spec.family} | {spec.paper_title}"
        )
    return lines
