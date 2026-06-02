"""
论文实验注册表。

这个文件只描述“有哪些实验、每个实验有哪些变体、它们属于哪一类”，
不直接负责训练与回测执行，因此是整个论文实验体系的静态清单。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


ExperimentVariant = Dict[str, Any]
ExperimentSpec = Dict[str, Any]
ExperimentCatalog = Dict[str, Dict[str, ExperimentSpec]]


EXPERIMENT_CATALOG: ExperimentCatalog = {
    "main": {
        "market_suite": {
            "title": "Table M1: Main Market Suite",
            "kind": "prime_variant_grid",
            "variants": {
                "csi500_stock": {
                    "description": "CSI 500 stock benchmark",
                    "market_profile": "csi500",
                    "asset_type": "stock",
                    "overrides": {},
                },
                "sp500_stock": {
                    "description": "S&P 500 stock benchmark",
                    "market_profile": "sp500",
                    "asset_type": "stock",
                    "overrides": {},
                },
                "nikkei225_stock": {
                    "description": "Nikkei 225 stock benchmark",
                    "market_profile": "nikkei225",
                    "asset_type": "stock",
                    "overrides": {},
                },
                "stoxx600_stock": {
                    "description": "STOXX 600 stock benchmark",
                    "market_profile": "stoxx600",
                    "asset_type": "stock",
                    "overrides": {},
                },
                "csi500_etf": {
                    "description": "CSI 500 equity ETF benchmark",
                    "market_profile": "csi500",
                    "asset_type": "etf",
                    "overrides": {},
                },
            },
        },
        "etf_generalization": {
            "title": "Table M2: ETF Cross-Asset Generalization",
            "kind": "etf_generalization",
            "variants": {
                "default": {
                    "description": "A-share ETF experiment (~200 equity ETFs, 2023H2-2024)",
                    "market_profile": "csi500",
                    "asset_type": "etf",
                    "overrides": {},
                },
            },
        },
        "cross_geography": {
            "title": "Table M3: Cross-Geography Generalization",
            "kind": "cross_geography",
            "variants": {
                "nikkei225": {
                    "description": "Cross-geography generalization on Nikkei 225",
                    "market_profile": "nikkei225",
                    "asset_type": "stock",
                    "overrides": {},
                },
                "stoxx600": {
                    "description": "Cross-geography generalization on STOXX 600",
                    "market_profile": "stoxx600",
                    "asset_type": "stock",
                    "overrides": {},
                },
            },
        },
        "case_study": {
            "title": "Table M4: Failure-Aware Case Study",
            "kind": "case_study",
            "variants": {
                "default": {
                    "description": "Failure-aware interpretability case study",
                    "overrides": {},
                },
            },
        },
    },
    "ablation": {
        "module": {
            "title": "Table A1: Module Ablation",
            "kind": "prime_variant_grid",
            "variants": {
                "prime_default": {
                    "description": "Full PRIME model",
                    "overrides": {},
                },
                "no_macro_modulation": {
                    "description": "Remove macro modulation",
                    "overrides": {
                        "model.enable_macro_modulation": False,
                        "model.fixed_alpha": 1.0,
                        "model.fixed_beta": 1.0,
                        "model.fixed_gamma": 1.0,
                    },
                },
                "no_guardian": {
                    "description": "Remove Guardian",
                    "overrides": {
                        "training.enable_guardian": False,
                    },
                },
                "no_bull": {
                    "description": "Remove bull energy",
                    "overrides": {
                        "model.enable_bull": False,
                    },
                },
                "no_bear": {
                    "description": "Remove bear energy",
                    "overrides": {
                        "model.enable_bear": False,
                    },
                },
                "no_friction": {
                    "description": "Remove friction energy",
                    "overrides": {
                        "model.enable_friction": False,
                    },
                },
                "no_macro_timing": {
                    "description": "Remove macro timing",
                    "overrides": {
                        "model.enable_macro_timing": False,
                        "backtest.enable_macro_timing": False,
                    },
                },
                "bull_only": {
                    "description": "Bull only",
                    "overrides": {
                        "model.enable_bull": True,
                        "model.enable_bear": False,
                        "model.enable_friction": False,
                    },
                },
                "bear_only": {
                    "description": "Bear only",
                    "overrides": {
                        "model.enable_bull": False,
                        "model.enable_bear": True,
                        "model.enable_friction": False,
                    },
                },
            },
        },
        "feature_grouping": {
            "title": "Table A2: Feature Grouping Sensitivity",
            "kind": "prime_variant_grid",
            "variants": {
                "default": {
                    "description": "Expert grouping (paper default)",
                    "overrides": {},
                },
                "kmeans": {
                    "description": "Automatic K-means grouping",
                    "feature_action": "kmeans_grouping",
                    "overrides": {},
                },
                "random_shuffle": {
                    "description": "Randomly shuffled grouping",
                    "feature_action": "random_grouping",
                    "overrides": {},
                },
                "inverted": {
                    "description": "Invert bull and bear groups",
                    "feature_action": "swap_bull_bear",
                    "overrides": {},
                },
                "single_group": {
                    "description": "Single-group encoder without decomposition",
                    "feature_action": "single_group",
                    "overrides": {},
                },
            },
        },
        "aggregation": {
            "title": "Table A3: Aggregation Comparison",
            "kind": "prime_variant_grid",
            "variants": {
                "default": {
                    "description": "Potential-game aggregation (PRIME default)",
                    "overrides": {
                        "model.aggregation_mode": "potential_game",
                        "model.enable_macro_modulation": True,
                    },
                },
                "learnable_scalars": {
                    "description": "Learnable scalar aggregation",
                    "overrides": {
                        "model.aggregation_mode": "learnable_scalars",
                        "model.enable_macro_modulation": False,
                    },
                },
                "attention": {
                    "description": "Attention aggregation",
                    "overrides": {
                        "model.aggregation_mode": "attention",
                        "model.enable_macro_modulation": True,
                    },
                },
                "mlp": {
                    "description": "MLP nonlinear aggregation",
                    "overrides": {
                        "model.aggregation_mode": "mlp",
                        "model.enable_macro_modulation": True,
                    },
                },
                "fixed_equal": {
                    "description": "Fixed equal-weight aggregation",
                    "overrides": {
                        "model.aggregation_mode": "fixed_equal",
                        "model.enable_macro_modulation": False,
                    },
                },
            },
        },
        "guardian": {
            "title": "Table A4: Guardian Comparison",
            "kind": "prime_variant_grid",
            "variants": {
                "default": {
                    "description": "LightGBM Guardian (PRIME default)",
                    "overrides": {
                        "training.enable_guardian": True,
                        "model.guardian_type": "lightgbm",
                        "training.crash_aux_weight": 0.0,
                    },
                },
                "neural_mlp": {
                    "description": "Neural MLP crash predictor",
                    "overrides": {
                        "training.enable_guardian": True,
                        "model.guardian_type": "mlp",
                        "training.crash_aux_weight": 0.0,
                    },
                },
                "diff_penalty": {
                    "description": "Differentiable crash BCE penalty",
                    "overrides": {
                        "training.enable_guardian": False,
                        "training.crash_aux_weight": 0.25,
                    },
                },
                "no_guardian": {
                    "description": "No Guardian",
                    "overrides": {
                        "training.enable_guardian": False,
                        "training.crash_aux_weight": 0.0,
                    },
                },
            },
        },
    },
    "robustness": {
        "noise": {
            "title": "Table R1: Input Noise Sensitivity",
            "kind": "prime_variant_grid",
            "variants": {
                "noise_0": {
                    "description": "Noise 0%",
                    "overrides": {
                        "model.input_noise_std": 0.0,
                    },
                },
                "noise_1": {
                    "description": "Noise 1%",
                    "overrides": {
                        "model.input_noise_std": 0.01,
                    },
                },
                "noise_3": {
                    "description": "Noise 3%",
                    "overrides": {
                        "model.input_noise_std": 0.03,
                    },
                },
                "noise_5": {
                    "description": "Noise 5%",
                    "overrides": {
                        "model.input_noise_std": 0.05,
                    },
                },
                "noise_10": {
                    "description": "Noise 10%",
                    "overrides": {
                        "model.input_noise_std": 0.10,
                    },
                },
            },
        },
        "topk": {
            "title": "Table R2: Top-K Sensitivity",
            "kind": "prime_variant_grid",
            "variants": {
                "topk_20": {
                    "description": "Top-K = 20",
                    "overrides": {
                        "backtest.top_k": 20,
                    },
                },
                "topk_30": {
                    "description": "Top-K = 30",
                    "overrides": {
                        "backtest.top_k": 30,
                    },
                },
                "topk_50": {
                    "description": "Top-K = 50",
                    "overrides": {
                        "backtest.top_k": 50,
                    },
                },
                "topk_100": {
                    "description": "Top-K = 100",
                    "overrides": {
                        "backtest.top_k": 100,
                    },
                },
            },
        },
        "lr": {
            "title": "Table R3: Learning-Rate Sensitivity",
            "kind": "prime_variant_grid",
            "variants": {
                "lr_1e4": {
                    "description": "Learning rate = 1e-4",
                    "overrides": {
                        "training.learning_rate": 1e-4,
                    },
                },
                "lr_3e4": {
                    "description": "Learning rate = 3e-4",
                    "overrides": {
                        "training.learning_rate": 3e-4,
                    },
                },
                "lr_1e3": {
                    "description": "Learning rate = 1e-3",
                    "overrides": {
                        "training.learning_rate": 1e-3,
                    },
                },
            },
        },
        "epochs": {
            "title": "Table R4: Training-Epoch Sensitivity",
            "kind": "prime_variant_grid",
            "variants": {
                "epochs_20": {
                    "description": "Stage2 epochs = 20",
                    "overrides": {
                        "training.stage2_epochs": 20,
                    },
                },
                "epochs_30": {
                    "description": "Stage2 epochs = 30",
                    "overrides": {
                        "training.stage2_epochs": 30,
                    },
                },
                "epochs_50": {
                    "description": "Stage2 epochs = 50",
                    "overrides": {
                        "training.stage2_epochs": 50,
                    },
                },
            },
        },
        "crash_label": {
            "title": "Table R5: Crash Label Sensitivity",
            "kind": "prime_variant_grid",
            "variants": {
                "default": {
                    "description": "Absolute return crash label: ret<-10% OR dd>15%",
                    "overrides": {
                        "data.crash_threshold": -0.10,
                        "data.drawdown_threshold": 0.15,
                        "data.crash_relative_to_benchmark": False,
                    },
                },
                "strict": {
                    "description": "Strict crash label: ret<-5% OR dd>10%",
                    "overrides": {
                        "data.crash_threshold": -0.05,
                        "data.drawdown_threshold": 0.10,
                        "data.crash_relative_to_benchmark": False,
                    },
                },
                "lenient": {
                    "description": "Lenient crash label: ret<-15% OR dd>20%",
                    "overrides": {
                        "data.crash_threshold": -0.15,
                        "data.drawdown_threshold": 0.20,
                        "data.crash_relative_to_benchmark": False,
                    },
                },
                "return_only": {
                    "description": "Return-only crash label: ret<-10%",
                    "overrides": {
                        "data.crash_threshold": -0.10,
                        "data.drawdown_threshold": 999.0,
                        "data.crash_relative_to_benchmark": False,
                    },
                },
                "drawdown_only": {
                    "description": "Drawdown-only crash label: dd>15%",
                    "overrides": {
                        "data.crash_threshold": -999.0,
                        "data.drawdown_threshold": 0.15,
                        "data.crash_relative_to_benchmark": False,
                    },
                },
                "market_relative": {
                    "description": "Market-relative crash label: (ret-benchmark)<-10% OR dd>15%",
                    "overrides": {
                        "data.crash_threshold": -0.10,
                        "data.drawdown_threshold": 0.15,
                        "data.crash_relative_to_benchmark": True,
                    },
                },
            },
        },
        "rolling_window": {
            "title": "Table R6: Rolling-Window Evaluation",
            "kind": "prime_variant_grid",
            "variants": {
                "year2020": {
                    "description": "Rolling window test year 2020",
                    "overrides": {
                        "data.train_start": "2014-01-01",
                        "data.train_end": "2018-12-31",
                        "data.valid_start": "2019-01-01",
                        "data.valid_end": "2019-12-31",
                        "data.test_start": "2020-01-01",
                        "data.test_end": "2020-12-31",
                    },
                },
                "year2021": {
                    "description": "Rolling window test year 2021",
                    "overrides": {
                        "data.train_start": "2014-01-01",
                        "data.train_end": "2019-12-31",
                        "data.valid_start": "2020-01-01",
                        "data.valid_end": "2020-12-31",
                        "data.test_start": "2021-01-01",
                        "data.test_end": "2021-12-31",
                    },
                },
                "year2022": {
                    "description": "Rolling window test year 2022",
                    "overrides": {
                        "data.train_start": "2015-01-01",
                        "data.train_end": "2020-12-31",
                        "data.valid_start": "2021-01-01",
                        "data.valid_end": "2021-12-31",
                        "data.test_start": "2022-01-01",
                        "data.test_end": "2022-12-31",
                    },
                },
                "year2023": {
                    "description": "Rolling window test year 2023",
                    "overrides": {
                        "data.train_start": "2016-01-01",
                        "data.train_end": "2021-12-31",
                        "data.valid_start": "2022-01-01",
                        "data.valid_end": "2022-12-31",
                        "data.test_start": "2023-01-01",
                        "data.test_end": "2023-12-31",
                    },
                },
            },
        },
        "macro_degradation": {
            "title": "Table R7: Macro Signal Degradation",
            "kind": "prime_variant_grid",
            "variants": {
                "default": {
                    "description": "Full macro signals",
                    "overrides": {
                        "data.macro_lag": 0,
                        "model.enable_macro_modulation": True,
                    },
                },
                "delay_1m": {
                    "description": "Publication delay (+1 month)",
                    "overrides": {
                        "data.macro_lag": 1,
                        "model.enable_macro_modulation": True,
                    },
                },
                "noise_sigma_0_5": {
                    "description": "Macro noise corruption (sigma=0.5)",
                    "data_action": "macro_noise_0_5",
                    "overrides": {
                        "data.macro_lag": 0,
                        "model.enable_macro_modulation": True,
                    },
                },
                "missing_50": {
                    "description": "Macro random missing (50%)",
                    "data_action": "macro_missing_50",
                    "overrides": {
                        "data.macro_lag": 0,
                        "model.enable_macro_modulation": True,
                    },
                },
                "no_macro": {
                    "description": "Full macro removal",
                    "data_action": "macro_zero_out",
                    "overrides": {
                        "data.macro_lag": 0,
                        "model.enable_macro_modulation": False,
                    },
                },
            },
        },
        "energy_bounds": {
            "title": "Table R8: Energy Bound Design",
            "kind": "prime_variant_grid",
            "variants": {
                "default": {
                    "description": "Fixed bounds [0.1, 0.9] / sigma_target=0.15",
                    "overrides": {
                        "model.energy_bound_mode": "fixed",
                        "model.energy_target_std": 0.15,
                        "model.energy_bound_low": 0.10,
                        "model.energy_bound_high": 0.90,
                    },
                },
                "narrowed": {
                    "description": "Narrowed bounds [0.2, 0.8] / sigma_target=0.08",
                    "overrides": {
                        "model.energy_bound_mode": "fixed",
                        "model.energy_target_std": 0.08,
                        "model.energy_bound_low": 0.20,
                        "model.energy_bound_high": 0.80,
                    },
                },
                "widened": {
                    "description": "Widened bounds [0.05, 0.95] / sigma_target=0.25",
                    "overrides": {
                        "model.energy_bound_mode": "fixed",
                        "model.energy_target_std": 0.25,
                        "model.energy_bound_low": 0.05,
                        "model.energy_bound_high": 0.95,
                    },
                },
                "vix_adaptive": {
                    "description": "Volatility-adaptive energy bounds",
                    "overrides": {
                        "model.energy_bound_mode": "adaptive_vol",
                        "model.energy_target_std": 0.15,
                        "model.energy_target_std_min": 0.08,
                        "model.energy_target_std_max": 0.25,
                        "model.energy_bound_low": 0.09,
                        "model.energy_bound_high": 0.23,
                    },
                },
            },
        },
        "stress_period": {
            "title": "Table R9: Dual-Market Stress Periods",
            "kind": "stress_period",
            "variants": {
                "csi_2021q1": {
                    "description": "CSI 500 stress period 2021 Q1",
                    "market_profile": "csi500",
                    "asset_type": "stock",
                    "baseline_name": "alphagat",
                    "overrides": {
                        "data.train_start": "2014-01-01",
                        "data.train_end": "2019-12-31",
                        "data.valid_start": "2020-01-01",
                        "data.valid_end": "2020-12-31",
                        "data.test_start": "2021-01-01",
                        "data.test_end": "2021-03-31",
                    },
                },
                "csi_2022q2": {
                    "description": "CSI 500 stress period 2022 Q2",
                    "market_profile": "csi500",
                    "asset_type": "stock",
                    "baseline_name": "alphagat",
                    "overrides": {
                        "data.train_start": "2015-01-01",
                        "data.train_end": "2020-12-31",
                        "data.valid_start": "2021-01-01",
                        "data.valid_end": "2021-12-31",
                        "data.test_start": "2022-04-01",
                        "data.test_end": "2022-06-30",
                    },
                },
                "csi_2022": {
                    "description": "CSI 500 stress period 2022 full year",
                    "market_profile": "csi500",
                    "asset_type": "stock",
                    "baseline_name": "alphagat",
                    "overrides": {
                        "data.train_start": "2015-01-01",
                        "data.train_end": "2020-12-31",
                        "data.valid_start": "2021-01-01",
                        "data.valid_end": "2021-12-31",
                        "data.test_start": "2022-01-01",
                        "data.test_end": "2022-12-31",
                    },
                },
                "sp_2020q1": {
                    "description": "S&P 500 stress period 2020 Q1",
                    "market_profile": "sp500",
                    "asset_type": "stock",
                    "baseline_name": "stockformer",
                    "overrides": {
                        "data.train_start": "2014-01-01",
                        "data.train_end": "2018-12-31",
                        "data.valid_start": "2019-01-01",
                        "data.valid_end": "2019-12-31",
                        "data.test_start": "2020-01-01",
                        "data.test_end": "2020-03-31",
                    },
                },
                "sp_2022": {
                    "description": "S&P 500 stress period 2022 full year",
                    "market_profile": "sp500",
                    "asset_type": "stock",
                    "baseline_name": "stockformer",
                    "overrides": {
                        "data.train_start": "2015-01-01",
                        "data.train_end": "2020-12-31",
                        "data.valid_start": "2021-01-01",
                        "data.valid_end": "2021-12-31",
                        "data.test_start": "2022-01-01",
                        "data.test_end": "2022-12-31",
                    },
                },
            },
        },
    },
}


def _normalize_category(category: str) -> str:
    key = (category or "").strip().lower()
    if key not in EXPERIMENT_CATALOG:
        available = ", ".join(sorted(EXPERIMENT_CATALOG))
        raise KeyError(f"未知实验类别: {category}，可选: {available}")
    return key


def _normalize_experiment(category: str, experiment_name: str) -> str:
    category = _normalize_category(category)
    key = (experiment_name or "").strip().lower()
    if key not in EXPERIMENT_CATALOG[category]:
        available = ", ".join(sorted(EXPERIMENT_CATALOG[category]))
        raise KeyError(f"未知实验: {category}/{experiment_name}，可选: {available}")
    return key


def _normalize_variant(category: str, experiment_name: str, variant_name: Optional[str]) -> str:
    category = _normalize_category(category)
    experiment_name = _normalize_experiment(category, experiment_name)
    key = (variant_name or "default").strip().lower()
    variants = EXPERIMENT_CATALOG[category][experiment_name]["variants"]
    if key not in variants:
        available = ", ".join(sorted(variants))
        raise KeyError(
            f"未知实验变体: {category}/{experiment_name}/{variant_name}，可选: {available}"
        )
    return key


def _set_nested_attr(config, dotted_key: str, value: Any):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _apply_runtime_data_root_override(config, market_profile: Optional[str], asset_type: Optional[str]):
    """
    将实验运行时的临时数据根目录映射回配置。

    这主要用于完整性检查等隔离环境，避免实验 runner 在切市场时重新指回真实数据目录。
    """
    overrides = getattr(config, "_runtime_data_root_overrides", None)
    if not overrides or not market_profile:
        return

    resolved_asset = (asset_type or getattr(config.data, "mode", "stock") or "stock").lower()
    root = overrides.get((market_profile, resolved_asset))
    if root is None:
        root = overrides.get(f"{market_profile}:{resolved_asset}")
    if root is None:
        return

    root = Path(root)
    panel_dir = root / "panel"
    if resolved_asset == "etf":
        config.data.etf_data_root = root
        config.data.etf_panel_path = panel_dir / "etf_panel_complete.csv"
        config.data.etf_daily_dir = root / "etf"
        config.data.etf_nav_dir = root / "nav"
        config.data.etf_share_dir = root / "share"
        config.data.etf_benchmark_dir = root / "benchmark"
        config.data.etf_macro_dir = root / "macro"
        config.data.etf_index_dir = root / "index"
        config.data.etf_meta_dir = root / "meta"
    else:
        config.data.data_root = root
        config.data.panel_path = panel_dir / "panel_data_complete.csv"
        config.data.stock_dir = root / "stock"
        config.data.valuation_dir = root / "valuation"
        config.data.financial_dir = root / "financial"
        config.data.capital_flow_dir = root / "capital_flow"
        config.data.chip_dir = root / "chip"
        config.data.margin_dir = root / "margin"
        config.data.northbound_dir = root / "northbound"
        config.data.macro_dir = root / "macro"
        config.data.index_dir = root / "index"
        config.data.industry_dir = root / "industry"
        config.data.market_dir = root / "market"
        config.data.meta_dir = root / "meta"


def available_categories() -> List[str]:
    return sorted(EXPERIMENT_CATALOG.keys())


def available_experiments(category: str) -> List[str]:
    category = _normalize_category(category)
    return sorted(EXPERIMENT_CATALOG[category].keys())


def available_variants(category: str, experiment_name: str) -> List[str]:
    category = _normalize_category(category)
    experiment_name = _normalize_experiment(category, experiment_name)
    return sorted(EXPERIMENT_CATALOG[category][experiment_name]["variants"].keys())


def get_experiment_spec(category: str, experiment_name: str) -> ExperimentSpec:
    category = _normalize_category(category)
    experiment_name = _normalize_experiment(category, experiment_name)
    return deepcopy(EXPERIMENT_CATALOG[category][experiment_name])


def get_variant_spec(
    category: str,
    experiment_name: str,
    variant_name: Optional[str] = None,
) -> ExperimentVariant:
    category = _normalize_category(category)
    experiment_name = _normalize_experiment(category, experiment_name)
    variant_name = _normalize_variant(category, experiment_name, variant_name)
    variant = deepcopy(EXPERIMENT_CATALOG[category][experiment_name]["variants"][variant_name])
    variant["category"] = category
    variant["experiment_name"] = experiment_name
    variant["variant_name"] = variant_name
    return variant


def apply_experiment_variant(
    config,
    category: str,
    experiment_name: str,
    variant_name: Optional[str] = None,
) -> ExperimentVariant:
    variant = get_variant_spec(category, experiment_name, variant_name)
    market_profile = variant.get("market_profile")
    asset_type = variant.get("asset_type")
    if market_profile:
        config.apply_market_profile(market_profile, asset_type or config.mode)
        _apply_runtime_data_root_override(config, market_profile, asset_type or config.mode)
    for dotted_key, value in variant.get("overrides", {}).items():
        _set_nested_attr(config, dotted_key, value)
    base_name = str(config.experiment_name).split("__exp_")[0]
    config.experiment_name = (
        f"{base_name}__exp_{variant['category']}_{variant['experiment_name']}_{variant['variant_name']}"
    )
    return variant


def list_experiment_lines(category: Optional[str] = None) -> List[str]:
    categories = [_normalize_category(category)] if category else available_categories()
    lines: List[str] = []
    for current_category in categories:
        lines.append(f"[{current_category}]")
        for exp_name in available_experiments(current_category):
            spec = EXPERIMENT_CATALOG[current_category][exp_name]
            lines.append(f"  {exp_name}: {spec['title']}")
            for variant_name, variant in sorted(spec["variants"].items()):
                lines.append(f"    - {variant_name}: {variant['description']}")
    return lines
