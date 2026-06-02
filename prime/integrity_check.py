#!/usr/bin/env python3
"""
PRIME 论文实验体系完整性检查。

目标：
1. 静态编译与注册表检查
2. 真实数据源状态检查（含 mock 污染提示）
3. 预处理缓存 / embedding 加速链路检查
4. PRIME 主训练链路 smoke
5. main / ablation / robustness / baselines 统一纳入同一套回归检查
6. 输出、表格、回测图表、训练图表文件存在性检查

说明：
- 为避免污染真实数据目录，完整性 smoke 默认全部使用隔离的临时数据根目录。
- 真实数据目录只做“只读检查”，不会在 integrity_check 中触发自动 mock 写回。
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from baselines import BaselineRunConfig, BaselineRunner, available_baselines
from config import Config
from dataset import create_training_loaders
from experiment_runner import (
    AblationConfig,
    AblationRunner,
    MainExperimentConfig,
    MainExperimentRunner,
    RobustnessConfig,
    RobustnessRunner,
)
from experiment_catalog import (
    available_categories,
    available_experiments,
    available_variants,
)
from main import export_training_visualizations, run_backtest, run_data_pipeline
from market_profiles import available_market_profiles
from data_loader import generate_market_mock_panel
from trainer import CurriculumTrainer
from utils import set_seed


PROJECT_ROOT = Path(__file__).parent
# Optional local backup/archive root. Not shipped with the public repository;
# the integrity report simply records these as absent when the paths do not
# exist, so the check never hard-fails on a fresh clone.
BACKUP_ROOT = Path(os.environ.get("PRIME_BACKUP_ROOT", PROJECT_ROOT.parent / "backup"))
ACTIVE_REBUTTAL_DIR = PROJECT_ROOT / "rebuttal"
ARCHIVED_REBUTTAL_DIR = BACKUP_ROOT / "rebuttal_archive" / "rebuttal"

PIPELINE_COMBOS: List[Tuple[str, str]] = [
    ("csi500", "stock"),
    ("sp500", "stock"),
    ("nikkei225", "stock"),
    ("stoxx600", "stock"),
    ("csi500", "etf"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="PRIME 项目 / 论文实验体系完整性检查")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "integrity_check"),
        help="完整性检查输出目录",
    )
    parser.add_argument("--fast", action="store_true", help="使用快速数据流水线")
    parser.add_argument("--no_cache", action="store_true", help="禁用数据缓存")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速 smoke：仍覆盖全部实验类别，但使用更小隔离数据与更轻训练设置",
    )
    parser.add_argument("--skip_experiments", action="store_true", help="跳过 main/ablation/robustness")
    parser.add_argument("--skip_baselines", action="store_true", help="跳过 baseline smoke")
    return parser.parse_args()


def _capture_section(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        payload = fn(*args, **kwargs)
        return {
            "name": name,
            "status": "ok",
            "duration_sec": round(time.perf_counter() - start, 3),
            "payload": payload,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "duration_sec": round(time.perf_counter() - start, 3),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _compile_targets() -> List[Path]:
    return [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "config.py",
        PROJECT_ROOT / "data_loader.py",
        PROJECT_ROOT / "dataset.py",
        PROJECT_ROOT / "experiment_catalog.py",
        PROJECT_ROOT / "experiment_utils.py",
        PROJECT_ROOT / "experiment_runner.py",
        PROJECT_ROOT / "integrity_check.py",
        PROJECT_ROOT / "trainer.py",
        PROJECT_ROOT / "backtest.py",
        PROJECT_ROOT / "visualization.py",
        PROJECT_ROOT / "models" / "energy_model.py",
        PROJECT_ROOT / "models" / "losses.py",
        PROJECT_ROOT / "models" / "risk_guardian.py",
        PROJECT_ROOT / "baselines" / "registry.py",
        PROJECT_ROOT / "baselines" / "datasets.py",
        PROJECT_ROOT / "baselines" / "models.py",
        PROJECT_ROOT / "baselines" / "runner.py",
    ]


def run_static_compile_check() -> Dict[str, Any]:
    checked = []
    for path in _compile_targets():
        py_compile.compile(str(path), doraise=True)
        checked.append(str(path))
    return {
        "checked_files": checked,
        "count": len(checked),
    }


def run_registry_check() -> Dict[str, Any]:
    return {
        "categories": available_categories(),
        "main_experiments": available_experiments("main"),
        "ablations": available_experiments("ablation"),
        "robustness": available_experiments("robustness"),
        "baselines": available_baselines(),
        "variant_counts": {
            category: {
                experiment: len(available_variants(category, experiment))
                for experiment in available_experiments(category)
            }
            for category in ("main", "ablation", "robustness")
        },
        "active_rebuttal_exists": ACTIVE_REBUTTAL_DIR.exists(),
        "archived_rebuttal_exists": ARCHIVED_REBUTTAL_DIR.exists(),
    }


def _mock_meta_path(path: Path) -> Path:
    return path.parent / f"{path.stem}.mock_meta.json"


def _inspect_panel_path(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info

    info["size_mb"] = round(path.stat().st_size / (1024 * 1024), 3)
    info["mock_meta_exists"] = _mock_meta_path(path).exists()
    info["is_mock_generated_dir"] = "mock_generated" in path.parts

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=[c for c in ["trade_date", "ts_code"] if True])
    else:
        frame = pd.read_csv(
            path,
            usecols=lambda c: c in {"trade_date", "ts_code"},
            parse_dates=["trade_date"],
            low_memory=False,
        )
    info["rows"] = int(len(frame))
    info["assets"] = int(frame["ts_code"].nunique()) if "ts_code" in frame.columns else 0
    if "trade_date" in frame.columns and not frame.empty:
        dates = pd.to_datetime(frame["trade_date"])
        info["date_min"] = str(dates.min().date())
        info["date_max"] = str(dates.max().date())
    return info


def inspect_project_data_sources() -> Dict[str, Any]:
    records = []
    warnings = []
    for market, asset in PIPELINE_COMBOS:
        cfg = Config.market_mode(market, asset)
        panel_path = cfg.data.active_panel_path
        info = _inspect_panel_path(panel_path)
        info.update(
            {
                "market_profile": market,
                "asset_type": asset,
                "expected_root": str(cfg.data.active_data_root),
            }
        )
        if info.get("mock_meta_exists") or info.get("is_mock_generated_dir"):
            warnings.append(
                f"{market}/{asset} 当前活动 panel 看起来是 mock 数据: {panel_path}"
            )
        records.append(info)
    return {
        "sources": records,
        "warnings": warnings,
    }


def _set_data_root(config: Config, root: Path):
    root = Path(root)
    panel_dir = root / "panel"
    if config.data.mode == "etf":
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


def _write_isolated_panel(
    root: Path,
    market: str,
    asset: str,
    start: str,
    end: str,
    max_assets: int,
    seed: int,
) -> Dict[str, Any]:
    config = Config.market_mode(market, asset)
    config.seed = seed
    config.data.train_start = start
    config.data.train_end = "2021-12-31"
    config.data.valid_start = "2022-01-01"
    config.data.valid_end = "2022-12-31"
    config.data.test_start = "2023-01-01"
    config.data.test_end = end
    config.data.auto_mock_if_missing = False
    _set_data_root(config, root)

    df = generate_market_mock_panel(config)
    if max_assets > 0:
        keep_codes = sorted(df["ts_code"].unique())[:max_assets]
        df = df[df["ts_code"].isin(keep_codes)].copy()
    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    panel_dir = root / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    if asset == "etf":
        csv_path = panel_dir / "etf_panel_complete.csv"
        parquet_path = panel_dir / "etf_panel_complete.parquet"
    else:
        csv_path = panel_dir / "panel_data_complete.csv"
        parquet_path = panel_dir / "panel_data_complete.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    meta_path = panel_dir / f"{csv_path.stem}.mock_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "market_profile": market,
                "asset_type": asset,
                "rows": int(len(df)),
                "assets": int(df["ts_code"].nunique()),
                "date_min": str(pd.to_datetime(df["trade_date"]).min().date()),
                "date_max": str(pd.to_datetime(df["trade_date"]).max().date()),
                "integrity_check_generated": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "root": str(root),
        "panel_path": str(csv_path),
        "rows": int(len(df)),
        "assets": int(df["ts_code"].nunique()),
    }


def prepare_isolated_data_roots(base_dir: Path, quick: bool) -> Dict[str, Any]:
    start = "2018-01-01" if quick else "2014-01-01"
    end = "2024-12-31"
    stock_assets = 8 if quick else 24
    etf_assets = 6 if quick else 16
    roots: Dict[Tuple[str, str], Path] = {}
    manifests: Dict[str, Any] = {}

    for market, asset in PIPELINE_COMBOS:
        root = base_dir / f"{market}_{asset}"
        root.mkdir(parents=True, exist_ok=True)
        manifest = _write_isolated_panel(
            root=root,
            market=market,
            asset=asset,
            start=start,
            end=end,
            max_assets=etf_assets if asset == "etf" else stock_assets,
            seed=42,
        )
        manifests[f"{market}:{asset}"] = manifest
        roots[(market, asset)] = root

    return {
        "roots": roots,
        "runtime_override_map": {f"{market}:{asset}": str(root) for (market, asset), root in roots.items()},
        "manifests": manifests,
    }


def benchmark_cache_speed(output_root: Path, fast: bool) -> Dict[str, Any]:
    bench_root = output_root / "_isolated_data" / "cache_benchmark_csi500_stock"
    manifest = _write_isolated_panel(
        root=bench_root,
        market="csi500",
        asset="stock",
        start="2018-01-01",
        end="2024-12-31",
        max_assets=24 if fast else 16,
        seed=123,
    )
    processed_cache_dir = bench_root / "panel" / "processed_cache"
    if processed_cache_dir.exists():
        for path in processed_cache_dir.iterdir():
            path.unlink()

    config = Config.market_mode("csi500", "stock")
    config.seed = 123
    config.data.train_start = "2019-01-01"
    config.data.train_end = "2022-12-31"
    config.data.valid_start = "2023-01-01"
    config.data.valid_end = "2023-12-31"
    config.data.test_start = "2024-01-01"
    config.data.test_end = "2024-12-31"
    config.data.auto_mock_if_missing = False
    _set_data_root(config, bench_root)

    cold_start = time.perf_counter()
    splits_cold, full_df_cold, _ = run_data_pipeline(config, fast=fast, no_cache=False)
    cold_sec = time.perf_counter() - cold_start

    warm_start = time.perf_counter()
    splits_warm, full_df_warm, _ = run_data_pipeline(config, fast=fast, no_cache=False)
    warm_sec = time.perf_counter() - warm_start

    cache_files = sorted(str(p.name) for p in processed_cache_dir.glob("*"))
    return {
        "manifest": manifest,
        "rows": int(len(full_df_cold)),
        "train_rows": int(len(splits_cold.train)),
        "valid_rows": int(len(splits_cold.valid)),
        "test_rows": int(len(splits_cold.test)),
        "cold_build_sec": round(cold_sec, 3),
        "warm_cache_sec": round(warm_sec, 3),
        "speedup_x": round(cold_sec / max(warm_sec, 1e-6), 2),
        "cache_file_count": len(cache_files),
        "cache_files": cache_files,
        "warm_rows_match": len(full_df_cold) == len(full_df_warm),
        "warm_split_match": (
            len(splits_cold.train) == len(splits_warm.train)
            and len(splits_cold.valid) == len(splits_warm.valid)
            and len(splits_cold.test) == len(splits_warm.test)
        ),
    }


def run_pipeline_checks(
    roots: Dict[Tuple[str, str], Path],
    fast: bool,
    no_cache: bool,
) -> List[Dict[str, Any]]:
    rows = []
    for market, asset in PIPELINE_COMBOS:
        config = Config.market_mode(market, asset)
        config.data.auto_mock_if_missing = False
        _set_data_root(config, roots[(market, asset)])
        started = time.perf_counter()
        splits, full_df, feature_cols = run_data_pipeline(config, fast=fast, no_cache=no_cache)
        dates = pd.to_datetime(full_df["trade_date"])
        rows.append(
            {
                "market_profile": market,
                "asset_type": asset,
                "source_path": str(config.data.active_panel_path),
                "rows": int(len(full_df)),
                "cols": int(full_df.shape[1]),
                "assets": int(full_df["ts_code"].nunique()),
                "date_min": str(dates.min().date()),
                "date_max": str(dates.max().date()),
                "train_rows": int(len(splits.train)),
                "valid_rows": int(len(splits.valid)),
                "test_rows": int(len(splits.test)),
                "feature_dims": {key: len(value) for key, value in feature_cols.items()},
                "duration_sec": round(time.perf_counter() - started, 3),
                "has_daily_return": "daily_return" in full_df.columns,
                "has_crash_label": "crash_label" in full_df.columns,
                "has_macro_embed": any(col.startswith("macro_embed_") for col in full_df.columns),
            }
        )
    return rows


def _tighten_training_config(config: Config):
    config.device = "cpu"
    config.seed = 42
    config.training.stage1_epochs = 1
    config.training.stage2_epochs = 1
    config.training.stage3_epochs = 1
    config.training.batch_size = min(config.training.batch_size, 128)
    config.training.num_workers = 0
    config.training.persistent_workers = False
    config.training.use_amp = False
    config.training.patience = 1
    config.training.pairs_per_day = min(getattr(config.training, "pairs_per_day", 64), 32)
    config.training.head_sampling_ratio = min(getattr(config.training, "head_sampling_ratio", 0.5), 0.5)
    config.training.min_return_diff = max(getattr(config.training, "min_return_diff", 0.01), 0.01)
    config.backtest.top_k = min(config.backtest.top_k, 8 if config.mode == "stock" else 5)


def run_prime_model_flow(
    roots: Dict[Tuple[str, str], Path],
    output_root: Path,
    fast: bool,
    no_cache: bool,
) -> Dict[str, Any]:
    config = Config.market_mode("csi500", "stock")
    config.data.auto_mock_if_missing = False
    _set_data_root(config, roots[("csi500", "stock")])
    _tighten_training_config(config)
    set_seed(config.seed)

    splits, _, feature_cols = run_data_pipeline(config, fast=fast, no_cache=no_cache)
    dataset_bundle, train_loader, valid_loader, pairwise_loader = create_training_loaders(
        splits.train,
        splits.valid,
        config,
    )
    feature_dims = dataset_bundle.get_feature_dims()
    all_feature_cols = (
        feature_cols["bull"]
        + feature_cols["bear"]
        + feature_cols["friction"]
        + feature_cols["macro"]
    )

    trainer = CurriculumTrainer(config)
    integrator = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        pairwise_loader=pairwise_loader,
        train_df=splits.train,
        valid_df=splits.valid,
        feature_dims=feature_dims,
        feature_cols=all_feature_cols,
        feature_groups=feature_cols,
        skip_stage1=False,
        use_ic_loss=True,
        enable_landscape=False,
        enable_correlation=False,
    )

    run_dir = output_root / "prime_model_flow"
    checkpoint_dir = run_dir / "checkpoint"
    trainer.save(checkpoint_dir)
    backtest_result = run_backtest(config, splits.test, integrator, feature_cols, str(checkpoint_dir))
    export_training_visualizations(checkpoint_dir, config)

    expected = {
        "game_ebm": checkpoint_dir / "game_ebm.pt",
        "config": checkpoint_dir / "config.json",
        "guardian_dir": checkpoint_dir / "guardian",
        "backtest_metrics": checkpoint_dir / "backtest" / "metrics.json",
        "backtest_holdings": checkpoint_dir / "backtest" / "holdings.csv",
        "backtest_trades": checkpoint_dir / "backtest" / "trades.csv",
        "backtest_nav": checkpoint_dir / "backtest" / "nav.csv",
        "backtest_nav_fig": checkpoint_dir / "backtest" / "nav.png",
        "viz_history": checkpoint_dir / "viz_data" / "training_history.json",
        "viz_phase_space": checkpoint_dir / "viz_data" / "game_phase_space.csv",
        "viz_guardian": checkpoint_dir / "viz_data" / "guardian_distribution.json",
    }
    figure_dir = checkpoint_dir / "figures"
    figure_files = sorted(str(p.name) for p in figure_dir.glob("*")) if figure_dir.exists() else []

    return {
        "feature_dims": feature_dims,
        "train_rows": int(len(splits.train)),
        "valid_rows": int(len(splits.valid)),
        "test_rows": int(len(splits.test)),
        "train_batches": len(train_loader),
        "valid_batches": len(valid_loader),
        "pairwise_batches": len(pairwise_loader),
        "annual_return": float(backtest_result.metrics.get("annual_return", 0.0)),
        "sharpe_ratio": float(backtest_result.metrics.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(backtest_result.metrics.get("max_drawdown", 0.0)),
        "expected_files": {key: str(path) for key, path in expected.items()},
        "file_exists": {key: path.exists() for key, path in expected.items()},
        "figure_dir": str(figure_dir),
        "figure_files": figure_files,
    }


def _prepare_experiment_base_config(
    roots: Dict[Tuple[str, str], Path],
    runtime_override_map: Dict[str, str],
) -> Config:
    config = Config.market_mode("csi500", "stock")
    config.data.auto_mock_if_missing = False
    _set_data_root(config, roots[("csi500", "stock")])
    _tighten_training_config(config)
    config._runtime_data_root_overrides = runtime_override_map
    return config


def _summarize_category_outputs(
    output_dir: Path,
    category: str,
    category_payload: Dict[str, Any],
) -> Dict[str, Any]:
    results_dir = output_dir / "results" / category
    tables_dir = output_dir / "tables" / category
    artifacts_dir = output_dir / "artifacts" / category

    result_files = sorted(str(p.relative_to(output_dir)) for p in results_dir.glob("*")) if results_dir.exists() else []
    table_files = sorted(str(p.relative_to(output_dir)) for p in tables_dir.glob("*")) if tables_dir.exists() else []
    sample_backtest = next(artifacts_dir.rglob("backtest/metrics.json"), None) if artifacts_dir.exists() else None
    sample_nav = next(artifacts_dir.rglob("backtest/nav.png"), None) if artifacts_dir.exists() else None

    return {
        "experiment_names": list(category_payload.keys()),
        "result_file_count": len(result_files),
        "table_file_count": len(table_files),
        "artifact_dir_exists": artifacts_dir.exists(),
        "sample_backtest_metrics": str(sample_backtest.relative_to(output_dir)) if sample_backtest else None,
        "sample_backtest_nav": str(sample_nav.relative_to(output_dir)) if sample_nav else None,
        "result_files": result_files[:20],
        "table_files": table_files[:20],
    }


def run_categorized_experiment_checks(
    roots: Dict[Tuple[str, str], Path],
    runtime_override_map: Dict[str, str],
    output_root: Path,
    fast: bool,
    no_cache: bool,
) -> Dict[str, Any]:
    base_config = _prepare_experiment_base_config(roots, runtime_override_map)

    main_output = output_root / "main_experiments"
    ablation_output = output_root / "ablation"
    robustness_output = output_root / "robustness"

    main_runner = MainExperimentRunner(
        deepcopy(base_config),
        MainExperimentConfig(
            main_exp_type="all",
            output_dir=str(main_output),
            repeats=1,
            fast=fast,
            no_cache=no_cache,
            quick=True,
        ),
        device="cpu",
    )
    main_payload = main_runner.run(seed_base=42)

    ablation_runner = AblationRunner(
        deepcopy(base_config),
        AblationConfig(
            ablation_type="all",
            output_dir=str(ablation_output),
            repeats=1,
            fast=fast,
            no_cache=no_cache,
            quick=True,
        ),
        device="cpu",
    )
    ablation_payload = ablation_runner.run(seed_base=42)

    robustness_runner = RobustnessRunner(
        deepcopy(base_config),
        RobustnessConfig(
            robustness_type="all",
            output_dir=str(robustness_output),
            repeats=1,
            fast=fast,
            no_cache=no_cache,
            quick=True,
        ),
        device="cpu",
    )
    robustness_payload = robustness_runner.run(seed_base=42)

    return {
        "main": _summarize_category_outputs(main_output, "main", main_payload),
        "ablation": _summarize_category_outputs(ablation_output, "ablation", ablation_payload),
        "robustness": _summarize_category_outputs(robustness_output, "robustness", robustness_payload),
    }


def run_baseline_checks(
    roots: Dict[Tuple[str, str], Path],
    output_root: Path,
) -> Dict[str, Any]:
    config = Config.market_mode("csi500", "stock")
    config.data.auto_mock_if_missing = False
    _set_data_root(config, roots[("csi500", "stock")])
    _tighten_training_config(config)
    config.backtest.top_k = min(config.backtest.top_k, 6)

    splits, _, feature_cols = run_data_pipeline(config, fast=True, no_cache=False)
    baseline_output = output_root / "baselines"

    runner = BaselineRunner(
        config,
        BaselineRunConfig(
            baseline_name="all",
            output_dir=str(baseline_output),
            seq_len_override=12,
            epochs_override=2,
            batch_size_override=64,
            quick=True,
        ),
        device="cpu",
    )
    summaries = runner.run(splits, feature_cols)

    sample_navs = {}
    for summary in summaries:
        baseline_name = Path(summary["output_dir"]).name
        nav_path = Path(summary["output_dir"]) / "backtest" / "nav.png"
        metrics_path = Path(summary["output_dir"]) / "metrics.json"
        sample_navs[baseline_name] = {
            "metrics_exists": metrics_path.exists(),
            "nav_exists": nav_path.exists(),
        }

    return {
        "baseline_count": len(summaries),
        "baseline_names": [summary["baseline"] for summary in summaries],
        "summary_csv_exists": (baseline_output / "baseline_summary.csv").exists(),
        "sample_output_files": sample_navs,
    }


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    isolated = prepare_isolated_data_roots(output_root / "_isolated_data", quick=args.quick)
    roots = isolated["roots"]
    runtime_override_map = isolated["runtime_override_map"]

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "fast": args.fast,
        "no_cache": args.no_cache,
        "quick": args.quick,
        "isolated_data_manifests": isolated["manifests"],
        "sections": {},
    }

    report["sections"]["static_compile"] = _capture_section(
        "static_compile",
        run_static_compile_check,
    )
    report["sections"]["registry"] = _capture_section(
        "registry",
        run_registry_check,
    )
    report["sections"]["project_data_sources"] = _capture_section(
        "project_data_sources",
        inspect_project_data_sources,
    )
    report["sections"]["cache_benchmark"] = _capture_section(
        "cache_benchmark",
        benchmark_cache_speed,
        output_root,
        args.fast,
    )
    report["sections"]["pipelines"] = _capture_section(
        "pipelines",
        run_pipeline_checks,
        roots,
        args.fast,
        args.no_cache,
    )
    report["sections"]["prime_model_flow"] = _capture_section(
        "prime_model_flow",
        run_prime_model_flow,
        roots,
        output_root,
        args.fast,
        args.no_cache,
    )

    if not args.skip_experiments:
        report["sections"]["categorized_experiments"] = _capture_section(
            "categorized_experiments",
            run_categorized_experiment_checks,
            roots,
            runtime_override_map,
            output_root,
            args.fast,
            args.no_cache,
        )

    if not args.skip_baselines:
        report["sections"]["baselines"] = _capture_section(
            "baselines",
            run_baseline_checks,
            roots,
            output_root,
        )

    errors = [
        name
        for name, section in report["sections"].items()
        if section.get("status") != "ok"
    ]
    warnings = []
    source_section = report["sections"].get("project_data_sources", {})
    if source_section.get("status") == "ok":
        warnings.extend(source_section["payload"].get("warnings", []))
    report["summary"] = {
        "section_count": len(report["sections"]),
        "error_sections": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }

    report_path = output_root / "integrity_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"✓ 完整性检查报告已保存: {report_path}")
    if errors:
        print(f"✗ 存在失败分段: {', '.join(errors)}")
    else:
        print("✓ 所有完整性检查分段已通过")
    if warnings:
        print("! 检测到需要关注的数据源告警:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
