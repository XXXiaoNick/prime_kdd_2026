"""
统一实验执行器。

负责把主实验、消融实验与鲁棒性实验映射到同一条执行链：
数据加载 -> PRIME / baseline 训练 -> 回测 -> 表格与结果文件落盘。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import Backtester, PerformanceMetrics, save_backtest_result
from baselines import BaselineRunConfig, BaselineRunner
from config import Config
from dataset import create_training_loaders
from experiment_catalog import (
    apply_experiment_variant,
    available_experiments,
    available_variants,
    get_experiment_spec,
    get_variant_spec,
)
from experiment_utils import (
    apply_macro_data_action,
    build_feature_groups,
    compute_guardian_stats,
    daily_kendall_tau,
    daily_rank_ic,
    load_experiment_pipeline,
    score_with_integrator,
)
from trainer import CurriculumTrainer, Stage4Integrator
from utils import set_seed
from visualization import render_experiment_table


def _to_python(value):
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clone_config(config: Config) -> Config:
    return deepcopy(config)


def _maybe_quick_config(config: Config, quick: bool):
    if not quick:
        return
    config.training.stage1_epochs = min(config.training.stage1_epochs, 2)
    config.training.stage2_epochs = min(config.training.stage2_epochs, 4)
    config.training.stage3_epochs = min(config.training.stage3_epochs, 2)
    config.training.patience = min(config.training.patience, 1)
    config.training.batch_size = min(config.training.batch_size, 256)
    config.backtest.top_k = min(config.backtest.top_k, 12)


def _benchmark_annual_return(backtest_result) -> float:
    bench = backtest_result.benchmark_returns.dropna()
    if bench.empty:
        return 0.0
    return float(PerformanceMetrics.calculate_all(bench, bench)["annual_return"])


def _compute_equal_weight_metrics(test_df: pd.DataFrame, benchmark_returns: pd.Series):
    daily = test_df.groupby("trade_date")["daily_return"].mean()
    aligned_bench = benchmark_returns.reindex(daily.index).fillna(0.0)
    return PerformanceMetrics.calculate_all(daily, aligned_bench)


@dataclass
class ExperimentRunConfig:
    output_root: str
    fast: bool = False
    no_cache: bool = False
    quick: bool = False
    repeats: int = 1


class CategorizedExperimentRunner:
    def __init__(self, config: Config, run_config: ExperimentRunConfig, device: str = "cpu"):
        self.base_config = config
        self.run_config = run_config
        self.device = device
        self.output_root = Path(run_config.output_root)
        self.results_dir = self.output_root / "results"
        self.tables_dir = self.output_root / "tables"
        self.artifacts_dir = self.output_root / "artifacts"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        category: str,
        experiment_name: Optional[str] = None,
        seed_base: int = 42,
    ) -> Dict[str, object]:
        experiment_names = (
            [experiment_name]
            if experiment_name and experiment_name != "all"
            else available_experiments(category)
        )
        outputs: Dict[str, object] = {}
        for current_experiment in experiment_names:
            spec = get_experiment_spec(category, current_experiment)
            kind = spec["kind"]
            if kind == "case_study":
                payload = self._run_case_study(category, current_experiment, seed_base)
            elif kind == "etf_generalization":
                payload = self._run_etf_generalization(category, current_experiment, seed_base)
            elif kind == "cross_geography":
                payload = self._run_cross_geography(category, current_experiment, seed_base)
            elif kind == "stress_period":
                payload = self._run_stress_periods(category, current_experiment, seed_base)
            else:
                payload = self._run_prime_variant_grid(category, current_experiment, seed_base)
            outputs[current_experiment] = payload
            self._save_experiment(category, current_experiment, payload)
        return outputs

    def _seed_list(self, seed_base: int) -> List[int]:
        repeats = max(1, int(self.run_config.repeats))
        return [seed_base + idx for idx in range(repeats)]

    def _artifact_dir(self, category: str, experiment_name: str, variant_name: str, seed: int) -> Path:
        path = self.artifacts_dir / category / experiment_name / variant_name / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _prepare_experiment_context(
        self,
        category: str,
        experiment_name: str,
        variant_name: str,
        seed: int,
    ):
        config = _clone_config(self.base_config)
        variant = apply_experiment_variant(config, category, experiment_name, variant_name)
        _maybe_quick_config(config, self.run_config.quick)
        set_seed(seed)

        splits, _, feature_cols = load_experiment_pipeline(
            config,
            fast=self.run_config.fast,
            no_cache=self.run_config.no_cache,
        )
        feature_cols = build_feature_groups(
            feature_cols,
            splits.train,
            action=variant.get("feature_action"),
            seed=seed,
        )
        splits = apply_macro_data_action(
            splits,
            feature_cols["macro"],
            action=variant.get("data_action"),
            seed=seed,
        )

        config.feature.bull_features = list(feature_cols["bull"])
        config.feature.bear_features = list(feature_cols["bear"])
        config.feature.friction_features = list(feature_cols["friction"])
        config.feature.macro_features = list(feature_cols["macro"])
        return config, variant, splits, feature_cols

    def _train_prime(
        self,
        config: Config,
        splits,
        feature_cols: Dict[str, List[str]],
    ) -> Stage4Integrator:
        dataset_bundle, train_loader, valid_loader, pairwise_loader = create_training_loaders(
            splits.train,
            splits.valid,
            config,
        )
        trainer = CurriculumTrainer(config)
        return trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            pairwise_loader=pairwise_loader,
            train_df=splits.train,
            valid_df=splits.valid,
            feature_dims=dataset_bundle.get_feature_dims(),
            feature_cols=(
                feature_cols["bull"]
                + feature_cols["bear"]
                + feature_cols["friction"]
                + feature_cols["macro"]
            ),
            feature_groups=feature_cols,
            skip_stage1=True,
            use_ic_loss=True,
            enable_landscape=False,
            enable_correlation=False,
        )

    def _run_prime_variant_grid(
        self,
        category: str,
        experiment_name: str,
        seed_base: int,
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        seeds = self._seed_list(seed_base)
        for variant_name in available_variants(category, experiment_name):
            current_seeds = list(seeds)
            if category == "ablation" and experiment_name == "feature_grouping" and variant_name == "random_shuffle":
                current_seeds = current_seeds[: max(3, len(current_seeds))]
            for seed in current_seeds:
                results.append(self._run_prime_once(category, experiment_name, variant_name, seed))
        return results

    def _run_prime_once(
        self,
        category: str,
        experiment_name: str,
        variant_name: str,
        seed: int,
    ) -> Dict[str, object]:
        config, variant, splits, feature_cols = self._prepare_experiment_context(
            category,
            experiment_name,
            variant_name,
            seed,
        )
        integrator = self._train_prime(config, splits, feature_cols)

        scored_valid = score_with_integrator(splits.valid, integrator, feature_cols)
        scored_test = score_with_integrator(splits.test, integrator, feature_cols)
        guardian_cols = (
            feature_cols["bull"]
            + feature_cols["bear"]
            + feature_cols["friction"]
            + feature_cols["macro"]
        )
        guardian_recall, guardian_fpr = compute_guardian_stats(
            integrator.guardian,
            splits.valid,
            guardian_cols,
        )

        backtester = Backtester(config)
        backtest_result = backtester.run(
            scored_test,
            signal_col="ebm_score",
            return_col="daily_return" if "daily_return" in scored_test.columns else "fwd_return",
            date_col="trade_date",
            code_col="ts_code",
            top_k=config.backtest.top_k,
        )
        benchmark_annual = _benchmark_annual_return(backtest_result)

        artifact_dir = self._artifact_dir(category, experiment_name, variant_name, seed)
        save_backtest_result(backtest_result, artifact_dir / "backtest")

        sigma_target = None
        if experiment_name == "energy_bounds":
            sigma_target = f"[{config.model.energy_bound_low:.2f}, {config.model.energy_bound_high:.2f}]"

        record = {
            "category": category,
            "experiment_name": experiment_name,
            "variant_name": variant_name,
            "description": variant["description"],
            "seed": seed,
            "market_profile": config.data.market_profile,
            "asset_type": config.mode,
            "annual_return": backtest_result.metrics.get("annual_return", 0.0),
            "sharpe_ratio": backtest_result.metrics.get("sharpe_ratio", 0.0),
            "calmar_ratio": backtest_result.metrics.get("calmar_ratio", 0.0),
            "max_drawdown": backtest_result.metrics.get("max_drawdown", 0.0),
            "test_ic": daily_rank_ic(scored_test),
            "rank_ic": daily_rank_ic(scored_test),
            "kendall_tau": daily_kendall_tau(scored_test),
            "guardian_recall": guardian_recall,
            "guardian_fpr": guardian_fpr,
            "crash_ratio": float(splits.train["crash_label"].mean()) if "crash_label" in splits.train.columns else 0.0,
            "benchmark_annual_return": benchmark_annual,
            "excess_alpha": backtest_result.metrics.get("annual_return", 0.0) - benchmark_annual,
            "sigma_target": sigma_target,
            "alpha_mean": float(scored_test["alpha_market"].mean()) if "alpha_market" in scored_test.columns else None,
            "beta_mean": float(scored_test["beta_risk"].mean()) if "beta_risk" in scored_test.columns else None,
            "gamma_mean": float(scored_test["gamma_heat"].mean()) if "gamma_heat" in scored_test.columns else None,
            "artifact_dir": str(artifact_dir),
        }
        with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(_to_python(record), handle, ensure_ascii=False, indent=2)
        return record

    def _run_case_study(
        self,
        category: str,
        experiment_name: str,
        seed_base: int,
    ) -> Dict[str, object]:
        seed = self._seed_list(seed_base)[0]
        config, _, splits, feature_cols = self._prepare_experiment_context(
            category,
            experiment_name,
            "default",
            seed,
        )
        integrator = self._train_prime(config, splits, feature_cols)
        scored_test = score_with_integrator(splits.test, integrator, feature_cols)
        guardian_cols = [
            col
            for col in (
                feature_cols["bull"]
                + feature_cols["bear"]
                + feature_cols["friction"]
                + feature_cols["macro"]
            )
            if col in scored_test.columns
        ]

        chosen_date = None
        chosen_cases = []
        for date, group in scored_test.groupby("trade_date"):
            top_n = min(len(group), config.backtest.top_k * 2)
            candidates = group.nlargest(top_n, "ebm_score").copy()
            if getattr(integrator.guardian, "_is_dummy", False):
                crash_proba = np.zeros(len(candidates))
            else:
                X_guard = np.nan_to_num(
                    candidates[guardian_cols].fillna(0).values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                crash_proba = integrator.guardian.predict_proba(X_guard)
            candidates["crash_proba"] = crash_proba
            candidates["selected"] = candidates["crash_proba"] < integrator.guardian.threshold

            correct_selected = candidates[
                (candidates["selected"]) & (candidates["crash_label"] == 0) & (candidates["fwd_return"] > 0)
            ]
            miss_selected = candidates[
                (candidates["selected"]) & (candidates["crash_label"] == 1)
            ]
            correct_filtered = candidates[
                (~candidates["selected"]) & (candidates["crash_label"] == 1)
            ]
            miss_filtered = candidates[
                (~candidates["selected"]) & (candidates["crash_label"] == 0) & (candidates["fwd_return"] > 0)
            ]
            if all(len(frame) > 0 for frame in (correct_selected, miss_selected, correct_filtered, miss_filtered)):
                chosen_date = date
                chosen_cases = [
                    ("Correctly selected", correct_selected.sort_values("fwd_return", ascending=False).iloc[0]),
                    ("Miss-selected", miss_selected.sort_values("fwd_return").iloc[0]),
                    ("Correctly filtered", correct_filtered.sort_values("max_dd_10d", ascending=False).iloc[0]),
                    ("Miss-filtered", miss_filtered.sort_values("crash_proba", ascending=False).iloc[0]),
                ]
                break

        cases = []
        for case_type, row in chosen_cases:
            cases.append(
                {
                    "case_type": case_type,
                    "trade_date": pd.Timestamp(chosen_date).strftime("%Y-%m-%d") if chosen_date is not None else "-",
                    "ts_code": row["ts_code"],
                    "E_bull": float(row["E_bull"]),
                    "E_bear": float(row["E_bear"]),
                    "E_heat": float(row["E_heat"]),
                    "ebm_score": float(row["ebm_score"]),
                    "crash_proba": float(row["crash_proba"]),
                    "fwd_return": float(row.get("fwd_return", row.get("fwd_return_5d", 0.0))),
                    "max_dd_10d": float(row.get("max_dd_10d", 0.0)),
                }
            )

        directions = (
            integrator.game_ebm.get_feature_directions()
            if hasattr(integrator.game_ebm, "get_feature_directions")
            else {}
        )
        feature_attr = []
        if chosen_cases and directions:
            reference_row = chosen_cases[0][1]
            for group_name, cols in [
                ("bull", feature_cols["bull"]),
                ("bear", feature_cols["bear"]),
                ("friction", feature_cols["friction"]),
            ]:
                weights = directions.get(group_name, {})
                direction = weights.get("direction")
                importance = weights.get("importance")
                if direction is None or importance is None or not cols:
                    continue
                contrib = np.abs(
                    reference_row[cols].fillna(0.0).values[: len(direction.cpu().numpy())]
                    * direction.cpu().numpy()
                    * importance.cpu().numpy()
                )
                idx = int(np.argmax(contrib))
                feature_attr.append(
                    {
                        "component": group_name,
                        "feature": cols[idx],
                        "direction": float(direction.cpu().numpy()[idx]),
                        "importance": float(importance.cpu().numpy()[idx]),
                        "feature_value": float(reference_row[cols[idx]]),
                    }
                )

        payload = {
            "date": pd.Timestamp(chosen_date).strftime("%Y-%m-%d") if chosen_date is not None else None,
            "cases": cases,
            "feature_attribution": feature_attr,
        }
        artifact_dir = self._artifact_dir(category, experiment_name, "default", seed)
        with open(artifact_dir / "case_study.json", "w", encoding="utf-8") as handle:
            json.dump(_to_python(payload), handle, ensure_ascii=False, indent=2)
        return payload

    def _run_etf_generalization(
        self,
        category: str,
        experiment_name: str,
        seed_base: int,
    ) -> List[Dict[str, object]]:
        seed = self._seed_list(seed_base)[0]
        prime_record = self._run_prime_once(category, experiment_name, "default", seed)

        config, _, splits, _ = self._prepare_experiment_context(
            category,
            experiment_name,
            "default",
            seed,
        )
        benchmark_returns = (
            splits.test.groupby("trade_date")["benchmark_return"].first()
            if "benchmark_return" in splits.test.columns
            else pd.Series(dtype=float)
        )
        eq_metrics = _compute_equal_weight_metrics(splits.test, benchmark_returns)
        index_metrics = (
            PerformanceMetrics.calculate_all(benchmark_returns, benchmark_returns)
            if not benchmark_returns.empty
            else {}
        )

        return [
            {
                "model": config.data.market_label + " Index",
                "annual_return": index_metrics.get("annual_return", 0.0),
                "sharpe_ratio": index_metrics.get("sharpe_ratio"),
                "rank_ic": None,
                "excess_alpha": 0.0,
            },
            {
                "model": "Equal-weight ETF",
                "annual_return": eq_metrics.get("annual_return", 0.0),
                "sharpe_ratio": eq_metrics.get("sharpe_ratio"),
                "rank_ic": None,
                "excess_alpha": eq_metrics.get("annual_return", 0.0) - index_metrics.get("annual_return", 0.0),
            },
            {
                "model": "PRIME",
                "annual_return": prime_record["annual_return"],
                "sharpe_ratio": prime_record["sharpe_ratio"],
                "rank_ic": prime_record["rank_ic"],
                "excess_alpha": prime_record["annual_return"] - index_metrics.get("annual_return", 0.0),
            },
        ]

    def _run_cross_geography(
        self,
        category: str,
        experiment_name: str,
        seed_base: int,
    ) -> List[Dict[str, object]]:
        results = []
        seeds = self._seed_list(seed_base)
        for variant_name in available_variants(category, experiment_name):
            variant = get_variant_spec(category, experiment_name, variant_name)
            prime_records = [
                self._run_prime_once(category, experiment_name, variant_name, seed)
                for seed in seeds
            ]
            prime_df = pd.DataFrame(prime_records)
            results.append(
                {
                    "market_label": variant["market_profile"],
                    "model": "PRIME",
                    "annual_return": float(prime_df["annual_return"].mean()),
                    "annual_return_std": float(prime_df["annual_return"].std(ddof=0)),
                    "sharpe_ratio": float(prime_df["sharpe_ratio"].mean()),
                    "sharpe_ratio_std": float(prime_df["sharpe_ratio"].std(ddof=0)),
                    "max_drawdown": float(prime_df["max_drawdown"].mean()),
                    "max_drawdown_std": float(prime_df["max_drawdown"].std(ddof=0)),
                }
            )

            for baseline_name in ["alphagat", "stockformer", "gpt4ts", "deeptrader"]:
                baseline_rows = []
                for seed in seeds[:1 if self.run_config.quick else len(seeds)]:
                    config, _, splits, feature_cols = self._prepare_experiment_context(
                        category,
                        experiment_name,
                        variant_name,
                        seed,
                    )
                    output_dir = str(self._artifact_dir(category, experiment_name, f"{variant_name}_{baseline_name}", seed))
                    runner = BaselineRunner(
                        config,
                        BaselineRunConfig(
                            baseline_name=baseline_name,
                            output_dir=output_dir,
                            quick=self.run_config.quick,
                        ),
                        self.device,
                    )
                    baseline_rows.append(runner.run(splits, feature_cols)[0])
                baseline_df = pd.DataFrame(baseline_rows)
                results.append(
                    {
                        "market_label": variant["market_profile"],
                        "model": baseline_name,
                        "annual_return": float(baseline_df["annual_return"].mean()),
                        "annual_return_std": float(baseline_df["annual_return"].std(ddof=0)),
                        "sharpe_ratio": float(baseline_df["sharpe_ratio"].mean()),
                        "sharpe_ratio_std": float(baseline_df["sharpe_ratio"].std(ddof=0)),
                        "max_drawdown": float(baseline_df["max_drawdown"].mean()),
                        "max_drawdown_std": float(baseline_df["max_drawdown"].std(ddof=0)),
                    }
                )
        return results

    def _run_stress_periods(
        self,
        category: str,
        experiment_name: str,
        seed_base: int,
    ) -> List[Dict[str, object]]:
        rows = []
        seed = self._seed_list(seed_base)[0]
        for variant_name in available_variants(category, experiment_name):
            variant = get_variant_spec(category, experiment_name, variant_name)
            prime = self._run_prime_once(category, experiment_name, variant_name, seed)

            config, _, splits, feature_cols = self._prepare_experiment_context(
                category,
                experiment_name,
                variant_name,
                seed,
            )
            output_dir = str(self._artifact_dir(category, experiment_name, f"{variant_name}_{variant['baseline_name']}", seed))
            baseline_runner = BaselineRunner(
                config,
                BaselineRunConfig(
                    baseline_name=variant["baseline_name"],
                    output_dir=output_dir,
                    quick=self.run_config.quick,
                ),
                self.device,
            )
            baseline_summary = baseline_runner.run(splits, feature_cols)[0]

            benchmark_returns = (
                splits.test.groupby("trade_date")["benchmark_return"].first()
                if "benchmark_return" in splits.test.columns
                else pd.Series(dtype=float)
            )
            if benchmark_returns.empty:
                market_index_return = 0.0
            else:
                market_index_return = PerformanceMetrics.calculate_all(
                    benchmark_returns,
                    benchmark_returns,
                ).get("total_return", 0.0)

            rows.append(
                {
                    "market_label": config.data.market_label,
                    "stress_label": variant["description"],
                    "market_index_return": market_index_return,
                    "baseline_return": baseline_summary["annual_return"],
                    "prime_return": prime["annual_return"],
                    "prime_minus_baseline": prime["annual_return"] - baseline_summary["annual_return"],
                }
            )
        return rows

    def _save_experiment(self, category: str, experiment_name: str, payload):
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        category_results_dir = self.results_dir / category
        category_tables_dir = self.tables_dir / category
        category_results_dir.mkdir(parents=True, exist_ok=True)
        category_tables_dir.mkdir(parents=True, exist_ok=True)

        json_path = category_results_dir / f"{experiment_name}_{stamp}.json"
        csv_path = category_results_dir / f"{experiment_name}_{stamp}.csv"
        table_path = category_tables_dir / f"{experiment_name}.md"

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(_to_python(payload), handle, ensure_ascii=False, indent=2)

        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            pd.DataFrame(payload).to_csv(csv_path, index=False)

        table_path.write_text(
            render_experiment_table(category, experiment_name, payload),
            encoding="utf-8",
        )
        print(f"✓ {category}/{experiment_name} 结果已保存: {json_path}")
        print(f"✓ {category}/{experiment_name} 表格已保存: {table_path}")


@dataclass
class MainExperimentConfig:
    main_exp_type: str = "all"
    output_dir: str = "outputs/main_experiments"
    repeats: int = 1
    fast: bool = False
    no_cache: bool = False
    quick: bool = False


class MainExperimentRunner:
    def __init__(
        self,
        base_config: Config,
        experiment_config: MainExperimentConfig,
        device: str = "cpu",
    ):
        self.base_config = base_config
        self.experiment_config = experiment_config
        self.device = device
        self.runner = CategorizedExperimentRunner(
            base_config,
            ExperimentRunConfig(
                output_root=str(Path(experiment_config.output_dir)),
                fast=experiment_config.fast,
                no_cache=experiment_config.no_cache,
                quick=experiment_config.quick,
                repeats=experiment_config.repeats,
            ),
            device=device,
        )

    def run(self, seed_base: int = 42) -> Dict[str, object]:
        return self.runner.run(
            "main",
            None if self.experiment_config.main_exp_type == "all" else self.experiment_config.main_exp_type,
            seed_base=seed_base,
        )


@dataclass
class AblationConfig:
    ablation_type: str = "all"
    output_dir: str = "outputs/ablation"
    repeats: int = 3
    fast: bool = False
    no_cache: bool = False
    quick: bool = False


class AblationRunner:
    def __init__(
        self,
        base_config: Config,
        ablation_config: AblationConfig,
        device: str = "cpu",
    ):
        self.base_config = base_config
        self.ablation_config = ablation_config
        self.device = device
        self.runner = CategorizedExperimentRunner(
            base_config,
            ExperimentRunConfig(
                output_root=str(Path(ablation_config.output_dir)),
                fast=ablation_config.fast,
                no_cache=ablation_config.no_cache,
                quick=ablation_config.quick,
                repeats=ablation_config.repeats,
            ),
            device=device,
        )

    def load_data(self, *args, **kwargs):
        return None

    def run(self, seed_base: int = 42) -> Dict[str, object]:
        return self.runner.run(
            "ablation",
            None if self.ablation_config.ablation_type == "all" else self.ablation_config.ablation_type,
            seed_base=seed_base,
        )

    def run_all(self, seed_base: int = 42) -> Dict[str, object]:
        return self.run(seed_base=seed_base)


@dataclass
class RobustnessConfig:
    robustness_type: str = "all"
    output_dir: str = "outputs/robustness"
    repeats: int = 1
    fast: bool = False
    no_cache: bool = False
    quick: bool = False


class RobustnessRunner:
    def __init__(
        self,
        base_config: Config,
        robustness_config: RobustnessConfig,
        device: str = "cpu",
    ):
        self.base_config = base_config
        self.robustness_config = robustness_config
        self.device = device
        self.runner = CategorizedExperimentRunner(
            base_config,
            ExperimentRunConfig(
                output_root=str(Path(robustness_config.output_dir)),
                fast=robustness_config.fast,
                no_cache=robustness_config.no_cache,
                quick=robustness_config.quick,
                repeats=robustness_config.repeats,
            ),
            device=device,
        )

    def load_data(self, *args, **kwargs):
        return None

    def run(self, seed_base: int = 42) -> Dict[str, object]:
        return self.runner.run(
            "robustness",
            None if self.robustness_config.robustness_type == "all" else self.robustness_config.robustness_type,
            seed_base=seed_base,
        )

    def run_all(self, seed_base: int = 42) -> Dict[str, object]:
        return self.run(seed_base=seed_base)
