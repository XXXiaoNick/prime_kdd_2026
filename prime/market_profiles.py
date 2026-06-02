"""
市场 preset 与跨市场字段约定。

用于描述不同市场在数据根目录、交易成本、mock 风格与默认超参数上的差异，
让主链路可以在不改训练架构的前提下切换市场。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data root resolution (public release)
# ---------------------------------------------------------------------------
# Raw financial data is intentionally NOT shipped with this repository.
# The data root is resolved in the following order:
#   1. The ``PRIME_DATA_ROOT`` environment variable, if it is set.
#   2. Otherwise ``<repo_root>/data`` (this file lives in ``<repo_root>/prime/``).
#
# When the resolved panel is missing or does not cover the configured train /
# valid / test windows, ``data_loader`` automatically generates market-aware
# mock data under ``<data_root>/<market>/panel/`` so that every command runs
# out-of-the-box without any proprietary data. See ``data/README.md``.
BASE_DATA_ROOT = Path(
    os.environ.get("PRIME_DATA_ROOT", Path(__file__).resolve().parent.parent / "data")
)


@dataclass(frozen=True)
class MarketPreset:
    key: str
    label: str
    benchmark: str
    equity_root: Path
    etf_root: Path
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    crash_threshold: float
    drawdown_threshold: float
    etf_crash_threshold: float
    etf_drawdown_threshold: float
    batch_size: int
    learning_rate: float
    pairs_per_day: int
    head_sampling_ratio: float
    min_return_diff: float
    physics_weight: float
    vol_neutral_weight: float
    top_k_stock: int
    top_k_etf: int
    stock_costs: Dict[str, float]
    etf_costs: Dict[str, float]
    mock_equity_count: int
    mock_etf_count: int
    sector_weights: Dict[str, float]
    mock_style: Dict[str, float]
    notes: List[str] = field(default_factory=list)


MARKET_PRESETS: Dict[str, MarketPreset] = {
    "csi500": MarketPreset(
        key="csi500",
        label="CSI 500",
        benchmark="CSI 500",
        equity_root=BASE_DATA_ROOT / "china",
        etf_root=BASE_DATA_ROOT / "china_etf",
        train_start="2023-07-03",
        train_end="2024-06-28",
        valid_start="2024-07-01",
        valid_end="2024-09-30",
        test_start="2024-10-01",
        test_end="2024-12-31",
        crash_threshold=-0.10,
        drawdown_threshold=0.15,
        etf_crash_threshold=-0.07,
        etf_drawdown_threshold=0.10,
        batch_size=1024,
        learning_rate=1e-4,
        pairs_per_day=160,
        head_sampling_ratio=0.55,
        min_return_diff=0.015,
        physics_weight=0.12,
        vol_neutral_weight=0.24,
        top_k_stock=50,
        top_k_etf=20,
        stock_costs={
            "commission": 0.0010,
            "slippage": 0.0010,
            "stamp_tax": 0.0010,
            "single_limit": 0.10,
            "sector_limit": 0.30,
            "stop_loss": -0.10,
        },
        etf_costs={
            "commission": 0.0003,
            "slippage": 0.0005,
            "stamp_tax": 0.0,
            "single_limit": 0.15,
            "sector_limit": 0.40,
            "stop_loss": -0.08,
        },
        mock_equity_count=500,
        mock_etf_count=200,
        sector_weights={
            "Industrials": 0.18,
            "Information Technology": 0.16,
            "Health Care": 0.10,
            "Consumer Discretionary": 0.10,
            "Financials": 0.08,
            "Materials": 0.09,
            "Consumer Staples": 0.07,
            "Utilities": 0.06,
            "Energy": 0.05,
            "Communication Services": 0.05,
            "Real Estate": 0.03,
            "Other": 0.03,
        },
        mock_style={
            "daily_vol": 0.017,
            "sector_vol": 0.012,
            "idio_scale": 0.020,
            "turnover_scale": 1.40,
            "premium_noise": 0.0025,
            "tracking_error": 0.0040,
            "top_weight_scale": 0.8,
        },
        notes=[
            "Mid-cap and domestically driven with relatively balanced sector exposure.",
            "Higher turnover, policy sensitivity, and cross-sectional dispersion than mega-cap benchmarks.",
        ],
    ),
    "sp500": MarketPreset(
        key="sp500",
        label="S&P 500",
        benchmark="S&P 500",
        equity_root=BASE_DATA_ROOT / "us",
        etf_root=BASE_DATA_ROOT / "us_etf",
        train_start="2023-07-03",
        train_end="2024-06-28",
        valid_start="2024-07-01",
        valid_end="2024-09-30",
        test_start="2024-10-01",
        test_end="2024-12-31",
        crash_threshold=-0.08,
        drawdown_threshold=0.12,
        etf_crash_threshold=-0.06,
        etf_drawdown_threshold=0.08,
        batch_size=1536,
        learning_rate=8e-5,
        pairs_per_day=120,
        head_sampling_ratio=0.50,
        min_return_diff=0.010,
        physics_weight=0.10,
        vol_neutral_weight=0.18,
        top_k_stock=40,
        top_k_etf=20,
        stock_costs={
            "commission": 0.0005,
            "slippage": 0.0005,
            "stamp_tax": 0.0,
            "single_limit": 0.10,
            "sector_limit": 0.35,
            "stop_loss": -0.09,
        },
        etf_costs={
            "commission": 0.0002,
            "slippage": 0.0003,
            "stamp_tax": 0.0,
            "single_limit": 0.15,
            "sector_limit": 0.45,
            "stop_loss": -0.06,
        },
        mock_equity_count=503,
        mock_etf_count=200,
        sector_weights={
            "Information Technology": 0.31,
            "Financials": 0.13,
            "Communication Services": 0.09,
            "Consumer Discretionary": 0.10,
            "Health Care": 0.11,
            "Industrials": 0.09,
            "Consumer Staples": 0.06,
            "Energy": 0.04,
            "Utilities": 0.03,
            "Materials": 0.02,
            "Real Estate": 0.02,
        },
        mock_style={
            "daily_vol": 0.011,
            "sector_vol": 0.008,
            "idio_scale": 0.012,
            "turnover_scale": 0.95,
            "premium_noise": 0.0012,
            "tracking_error": 0.0020,
            "top_weight_scale": 1.4,
        },
        notes=[
            "Float-adjusted market-cap benchmark with strong mega-cap and technology concentration.",
            "Deep liquidity and lower frictions, but higher benchmark concentration risk.",
        ],
    ),
    "nikkei225": MarketPreset(
        key="nikkei225",
        label="Nikkei 225",
        benchmark="Nikkei 225",
        equity_root=BASE_DATA_ROOT / "japan",
        etf_root=BASE_DATA_ROOT / "japan_etf",
        train_start="2023-07-03",
        train_end="2024-06-28",
        valid_start="2024-07-01",
        valid_end="2024-09-30",
        test_start="2024-10-01",
        test_end="2024-12-31",
        crash_threshold=-0.09,
        drawdown_threshold=0.13,
        etf_crash_threshold=-0.06,
        etf_drawdown_threshold=0.09,
        batch_size=768,
        learning_rate=1.2e-4,
        pairs_per_day=140,
        head_sampling_ratio=0.58,
        min_return_diff=0.012,
        physics_weight=0.11,
        vol_neutral_weight=0.20,
        top_k_stock=25,
        top_k_etf=15,
        stock_costs={
            "commission": 0.0008,
            "slippage": 0.0008,
            "stamp_tax": 0.0,
            "single_limit": 0.12,
            "sector_limit": 0.35,
            "stop_loss": -0.09,
        },
        etf_costs={
            "commission": 0.0003,
            "slippage": 0.0004,
            "stamp_tax": 0.0,
            "single_limit": 0.15,
            "sector_limit": 0.40,
            "stop_loss": -0.06,
        },
        mock_equity_count=225,
        mock_etf_count=200,
        sector_weights={
            "Technology": 0.53,
            "Consumer Goods": 0.25,
            "Materials": 0.10,
            "Capital Goods/Others": 0.07,
            "Financials": 0.03,
            "Transportation and Utilities": 0.02,
        },
        mock_style={
            "daily_vol": 0.013,
            "sector_vol": 0.010,
            "idio_scale": 0.016,
            "turnover_scale": 1.05,
            "premium_noise": 0.0018,
            "tracking_error": 0.0028,
            "top_weight_scale": 1.6,
        },
        notes=[
            "Price-weighted benchmark with very high technology and consumer goods weight concentration.",
            "Export and FX sensitivity are more important than in domestic broad-market benchmarks.",
        ],
    ),
    "stoxx600": MarketPreset(
        key="stoxx600",
        label="STOXX Europe 600",
        benchmark="STOXX Europe 600",
        equity_root=BASE_DATA_ROOT / "europe",
        etf_root=BASE_DATA_ROOT / "europe_etf",
        train_start="2023-07-03",
        train_end="2024-06-28",
        valid_start="2024-07-01",
        valid_end="2024-09-30",
        test_start="2024-10-01",
        test_end="2024-12-31",
        crash_threshold=-0.085,
        drawdown_threshold=0.12,
        etf_crash_threshold=-0.055,
        etf_drawdown_threshold=0.08,
        batch_size=1280,
        learning_rate=9e-5,
        pairs_per_day=150,
        head_sampling_ratio=0.52,
        min_return_diff=0.009,
        physics_weight=0.09,
        vol_neutral_weight=0.22,
        top_k_stock=60,
        top_k_etf=25,
        stock_costs={
            "commission": 0.0007,
            "slippage": 0.0007,
            "stamp_tax": 0.0,
            "single_limit": 0.08,
            "sector_limit": 0.30,
            "stop_loss": -0.085,
        },
        etf_costs={
            "commission": 0.00025,
            "slippage": 0.0004,
            "stamp_tax": 0.0,
            "single_limit": 0.12,
            "sector_limit": 0.35,
            "stop_loss": -0.055,
        },
        mock_equity_count=600,
        mock_etf_count=200,
        sector_weights={
            "Financials": 0.16,
            "Industrials": 0.15,
            "Health Care": 0.13,
            "Consumer Discretionary": 0.10,
            "Consumer Staples": 0.09,
            "Information Technology": 0.08,
            "Energy": 0.06,
            "Materials": 0.06,
            "Utilities": 0.05,
            "Communication Services": 0.05,
            "Real Estate": 0.03,
            "Other": 0.04,
        },
        mock_style={
            "daily_vol": 0.010,
            "sector_vol": 0.008,
            "idio_scale": 0.011,
            "turnover_scale": 0.85,
            "premium_noise": 0.0015,
            "tracking_error": 0.0025,
            "top_weight_scale": 0.7,
        },
        notes=[
            "Broad European benchmark spanning large, mid, and small caps across 17 countries.",
            "Lower single-name concentration but higher multi-country macro and FX heterogeneity.",
        ],
    ),
}


def get_market_preset(key: str) -> MarketPreset:
    normalized = (key or "csi500").lower()
    if normalized not in MARKET_PRESETS:
        raise KeyError(f"未知市场配置: {key}")
    return MARKET_PRESETS[normalized]


def available_market_profiles() -> List[str]:
    return sorted(MARKET_PRESETS.keys())


def apply_market_profile(config, market_key: str, asset_type: str = "stock"):
    preset = get_market_preset(market_key)
    mode = "etf" if asset_type == "etf" else "stock"

    config.data.market_profile = preset.key
    config.data.market_label = preset.label
    config.data.mode = mode

    config.data.data_root = preset.equity_root
    config.data.panel_path = preset.equity_root / "panel" / "panel_data_complete.csv"
    config.data.stock_dir = preset.equity_root / "stock"
    config.data.valuation_dir = preset.equity_root / "valuation"
    config.data.financial_dir = preset.equity_root / "financial"
    config.data.capital_flow_dir = preset.equity_root / "capital_flow"
    config.data.chip_dir = preset.equity_root / "chip"
    config.data.margin_dir = preset.equity_root / "margin"
    config.data.northbound_dir = preset.equity_root / "northbound"
    config.data.macro_dir = preset.equity_root / "macro"
    config.data.index_dir = preset.equity_root / "index"
    config.data.industry_dir = preset.equity_root / "industry"
    config.data.market_dir = preset.equity_root / "market"
    config.data.meta_dir = preset.equity_root / "meta"
    config.data.etf_data_root = preset.etf_root
    config.data.etf_panel_path = preset.etf_root / "panel" / "etf_panel_complete.csv"
    config.data.etf_daily_dir = preset.etf_root / "etf"
    config.data.etf_nav_dir = preset.etf_root / "nav"
    config.data.etf_share_dir = preset.etf_root / "share"
    config.data.etf_benchmark_dir = preset.etf_root / "benchmark"
    config.data.etf_macro_dir = preset.etf_root / "macro"
    config.data.etf_index_dir = preset.etf_root / "index"
    config.data.etf_meta_dir = preset.etf_root / "meta"

    config.data.train_start = preset.train_start
    config.data.train_end = preset.train_end
    config.data.valid_start = preset.valid_start
    config.data.valid_end = preset.valid_end
    config.data.test_start = preset.test_start
    config.data.test_end = preset.test_end

    config.data.crash_threshold = preset.crash_threshold
    config.data.drawdown_threshold = preset.drawdown_threshold
    config.data.etf_crash_threshold = preset.etf_crash_threshold
    config.data.etf_drawdown_threshold = preset.etf_drawdown_threshold

    config.training.batch_size = preset.batch_size
    config.training.learning_rate = preset.learning_rate
    config.training.pairs_per_day = preset.pairs_per_day
    config.training.head_sampling_ratio = preset.head_sampling_ratio
    config.training.min_return_diff = preset.min_return_diff
    config.training.physics_weight = preset.physics_weight
    config.training.vol_neutral_weight = preset.vol_neutral_weight

    stock_costs = preset.stock_costs
    etf_costs = preset.etf_costs
    config.backtest.commission_rate = stock_costs["commission"]
    config.backtest.slippage = stock_costs["slippage"]
    config.backtest.stamp_tax = stock_costs["stamp_tax"]
    config.backtest.single_stock_limit = stock_costs["single_limit"]
    config.backtest.industry_limit = stock_costs["sector_limit"]
    config.backtest.stop_loss_threshold = stock_costs["stop_loss"]

    config.backtest.etf_commission_rate = etf_costs["commission"]
    config.backtest.etf_slippage = etf_costs["slippage"]
    config.backtest.etf_stamp_tax = etf_costs["stamp_tax"]
    config.backtest.etf_single_limit = etf_costs["single_limit"]
    config.backtest.etf_sector_limit = etf_costs["sector_limit"]
    config.backtest.etf_stop_loss_threshold = etf_costs["stop_loss"]

    config.backtest.top_k = preset.top_k_etf if mode == "etf" else preset.top_k_stock
    config.experiment_name = f"macro_game_ebm_{preset.key}_{mode}"
    return config


def _groupwise_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    if "ts_code" not in df.columns or "close" not in df.columns:
        return pd.Series(50.0, index=df.index)

    delta = df.groupby("ts_code")["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.groupby(df["ts_code"]).transform(
        lambda x: x.rolling(period, min_periods=max(3, period // 2)).mean()
    )
    avg_loss = loss.groupby(df["ts_code"]).transform(
        lambda x: x.rolling(period, min_periods=max(3, period // 2)).mean()
    )
    rs = avg_gain / (avg_loss + 1e-8)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def harmonize_panel_schema(df: pd.DataFrame, config) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "symbol": "ts_code",
        "code": "ts_code",
        "stock_code": "ts_code",
        "date": "trade_date",
        "datetime": "trade_date",
        "vol": "volume",
        "amount": "turnover",
        "pct_change": "pct_chg",
        "profit_margin": "profit_ratio",
        "debt_to_equity": "debt_ratio",
        "avg_cost": "asr",
        "sector": "industry",
        "fed_rate": "lpr_1y",
        "policy_rate": "lpr_1y",
        "benchmark_ret_20d": "csi500_ret_20d",
        "sp500_ret_20d": "csi500_ret_20d",
        "nikkei225_ret_20d": "csi500_ret_20d",
        "stoxx600_ret_20d": "csi500_ret_20d",
    }
    applicable = {
        src: dst for src, dst in rename_map.items() if src in df.columns and dst not in df.columns
    }
    if applicable:
        df = df.rename(columns=applicable)

    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])

    if "ts_code" not in df.columns:
        df["ts_code"] = "UNKNOWN"
    df["ts_code"] = df["ts_code"].astype(str)

    if "turnover" in df.columns and "amount" not in df.columns:
        df["amount"] = df["turnover"]
    if "amount" in df.columns and "turnover" not in df.columns:
        df["turnover"] = df["amount"]

    if "volume" not in df.columns and "turnover" in df.columns and "close" in df.columns:
        df["volume"] = df["turnover"] / df["close"].replace(0, np.nan)

    if {"ts_code", "trade_date"}.issubset(df.columns):
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if "close" in df.columns and "pre_close" not in df.columns:
        df["pre_close"] = df.groupby("ts_code")["close"].shift(1)
    if {"close", "pre_close"}.issubset(df.columns) and "change" not in df.columns:
        df["change"] = df["close"] - df["pre_close"]
    if {"close", "pre_close"}.issubset(df.columns) and "pct_chg" not in df.columns:
        df["pct_chg"] = (df["close"] / df["pre_close"].replace(0, np.nan) - 1.0) * 100.0

    if "adj_factor" not in df.columns:
        df["adj_factor"] = 1.0
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"] * df["adj_factor"]

    if "industry" not in df.columns:
        df["industry"] = "Unknown"

    if "turnover_rate" not in df.columns:
        if "float_share" in df.columns and "volume" in df.columns:
            df["turnover_rate"] = df["volume"] / df["float_share"].replace(0, np.nan) * 100.0
        else:
            df["turnover_rate"] = 1.0

    if "volume_ratio" not in df.columns and "volume" in df.columns:
        vol_ma20 = df.groupby("ts_code")["volume"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        df["volume_ratio"] = df["volume"] / (vol_ma20 + 1e-8)

    rolling_low_60 = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(60, min_periods=10).min()
    ) if "close" in df.columns else pd.Series(0.0, index=df.index)
    rolling_high_60 = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(60, min_periods=10).max()
    ) if "close" in df.columns else pd.Series(1.0, index=df.index)
    price_span = (rolling_high_60 - rolling_low_60).replace(0, np.nan)

    if "winner_rate" not in df.columns and "close" in df.columns:
        df["winner_rate"] = ((df["close"] - rolling_low_60) / (price_span + 1e-8)).clip(0.0, 1.0)
    if "trapped_ratio" not in df.columns and "close" in df.columns:
        df["trapped_ratio"] = ((rolling_high_60 - df["close"]) / (price_span + 1e-8)).clip(0.0, 1.0)
    if "chip_support" not in df.columns and "winner_rate" in df.columns:
        df["chip_support"] = (1.0 - df["winner_rate"]).abs()
    if "dist_to_resistance" not in df.columns and "close" in df.columns:
        df["dist_to_resistance"] = (rolling_high_60 / df["close"].replace(0, np.nan) - 1.0) * 100.0

    if "asr" not in df.columns and "close" in df.columns:
        df["asr"] = df.groupby("ts_code")["close"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
    if "avg_cost_deviation" not in df.columns and {"close", "asr"}.issubset(df.columns):
        df["avg_cost_deviation"] = (df["close"] - df["asr"]) / (df["asr"] + 1e-8) * 100.0
    if "cost_pressure" not in df.columns and {"close", "asr"}.issubset(df.columns):
        df["cost_pressure"] = (df["close"] < df["asr"]).astype(float)

    if "chip_concentration_change" not in df.columns:
        base_col = "turnover_rate" if "turnover_rate" in df.columns else "volume_ratio"
        df["chip_concentration_change"] = df.groupby("ts_code")[base_col].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )

    if "profit_ratio" not in df.columns:
        if "gross_margin" in df.columns:
            df["profit_ratio"] = df["gross_margin"]
        else:
            df["profit_ratio"] = 0.0

    if "debt_ratio" not in df.columns:
        if "debt_asset_ratio" in df.columns:
            df["debt_ratio"] = df["debt_asset_ratio"]
        else:
            df["debt_ratio"] = 0.0

    if "roe_growth" not in df.columns:
        if "roe" in df.columns:
            df["roe_growth"] = df.groupby("ts_code")["roe"].transform(
                lambda x: x.diff().fillna(0.0)
            )
        elif "netprofit_yoy" in df.columns:
            df["roe_growth"] = df["netprofit_yoy"]
        else:
            df["roe_growth"] = 0.0

    for neutral_col in ("north_net_flow", "main_net_inflow", "goodwill_risk", "revenue_growth"):
        if neutral_col not in df.columns:
            df[neutral_col] = 0.0

    if "rsi_6" not in df.columns and "close" in df.columns:
        df["rsi_6"] = _groupwise_rsi(df, 6)
    if "rsi_14" not in df.columns and "close" in df.columns:
        df["rsi_14"] = _groupwise_rsi(df, 14)

    if "pe_rank" not in df.columns:
        pe_col = "pe_ttm" if "pe_ttm" in df.columns else "pe" if "pe" in df.columns else None
        df["pe_rank"] = (
            df.groupby("trade_date")[pe_col].rank(pct=True) if pe_col is not None else 0.5
        )
    if "pb_rank" not in df.columns:
        pb_col = "pb" if "pb" in df.columns else "index_pb" if "index_pb" in df.columns else None
        df["pb_rank"] = (
            df.groupby("trade_date")[pb_col].rank(pct=True) if pb_col is not None else 0.5
        )

    market_daily = None
    if {"trade_date", "pct_chg"}.issubset(df.columns):
        market_daily = (
            df.groupby("trade_date", as_index=False)
            .agg(
                market_ret=("pct_chg", lambda x: x.mean() / 100.0),
                market_turnover=("amount", "sum") if "amount" in df.columns else ("pct_chg", "count"),
            )
            .sort_values("trade_date")
        )
        market_daily["csi500_ret_20d"] = market_daily["market_ret"].rolling(20, min_periods=5).sum() * 100.0
        market_daily["market_volatility"] = (
            market_daily["market_ret"].rolling(20, min_periods=5).std() * np.sqrt(252) * 100.0
        )
        market_daily["market_liquidity_change"] = (
            market_daily["market_turnover"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
        )
        market_daily = market_daily.drop(columns=["market_ret"])
        df = df.merge(market_daily, on="trade_date", how="left", suffixes=("", "_derived"))
        for col in ("csi500_ret_20d", "market_turnover", "market_volatility", "market_liquidity_change"):
            derived = f"{col}_derived"
            if derived in df.columns:
                if col not in df.columns:
                    df[col] = df[derived]
                else:
                    df[col] = df[col].fillna(df[derived])
                df = df.drop(columns=[derived])

    if "lpr_1y" not in df.columns:
        df["lpr_1y"] = 0.0
    if "cpi_yoy" not in df.columns:
        df["cpi_yoy"] = 0.0
    if "ppi_yoy" not in df.columns:
        df["ppi_yoy"] = 0.0
    if "pmi" not in df.columns:
        df["pmi"] = 50.0
    if "real_rate_proxy" not in df.columns:
        df["real_rate_proxy"] = df["lpr_1y"] - df["cpi_yoy"]

    if config.mode == "etf":
        if "fd_share" not in df.columns:
            df["fd_share"] = 1e8
        if "unit_nav" not in df.columns:
            df["unit_nav"] = df["close"]
        if "accum_nav" not in df.columns:
            df["accum_nav"] = df["unit_nav"]
        if "adj_nav" not in df.columns:
            df["adj_nav"] = df["unit_nav"]
        if "index_close" not in df.columns:
            df["index_close"] = df["close"]
        if "index_pe" not in df.columns:
            df["index_pe"] = 18.0
        if "index_pb" not in df.columns:
            df["index_pb"] = 2.0
        if "premium_abs" not in df.columns:
            df["premium_abs"] = 0.0
        if "premium_positive_freq" not in df.columns:
            df["premium_positive_freq"] = 0.5
        if "fund_flow_daily" not in df.columns:
            df["fund_flow_daily"] = 0.0
        if "fund_flow_5d" not in df.columns:
            df["fund_flow_5d"] = 0.0
        if "fund_flow_20d" not in df.columns:
            df["fund_flow_20d"] = 0.0
        if "share_growth_20d" not in df.columns:
            df["share_growth_20d"] = 0.0
        if "share_outflow_rate" not in df.columns:
            df["share_outflow_rate"] = 0.0
        if "share_change_volatility" not in df.columns:
            df["share_change_volatility"] = 0.0
        if "nav_growth_rate" not in df.columns:
            df["nav_growth_rate"] = 0.0
        if "nav_max_drawdown_60d" not in df.columns:
            df["nav_max_drawdown_60d"] = 0.0
        if "excess_return_20d" not in df.columns:
            df["excess_return_20d"] = 0.0
        if "tracking_error_20d" not in df.columns:
            df["tracking_error_20d"] = 0.0
        if "alpha_60d" not in df.columns:
            df["alpha_60d"] = 0.0
        if "weighted_avg_cost" not in df.columns:
            df["weighted_avg_cost"] = df["close"]
        if "index_pe_rank" not in df.columns:
            df["index_pe_rank"] = df.groupby("trade_date")["index_pe"].rank(pct=True)
        if "index_pb_rank" not in df.columns:
            df["index_pb_rank"] = df.groupby("trade_date")["index_pb"].rank(pct=True)

    return df


def has_required_date_coverage(df: pd.DataFrame, config) -> bool:
    if df is None or df.empty or "trade_date" not in df.columns:
        return False

    dates = pd.to_datetime(df["trade_date"])
    unique_dates = pd.Series(dates).drop_duplicates()
    windows = [
        ((unique_dates >= pd.to_datetime(config.data.train_start)) & (unique_dates <= pd.to_datetime(config.data.train_end))).sum(),
        ((unique_dates >= pd.to_datetime(config.data.valid_start)) & (unique_dates <= pd.to_datetime(config.data.valid_end))).sum(),
        ((unique_dates >= pd.to_datetime(config.data.test_start)) & (unique_dates <= pd.to_datetime(config.data.test_end))).sum(),
    ]
    min_window = 10 if config.mode == "etf" else 20
    return all(count >= min_window for count in windows)
