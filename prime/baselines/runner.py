"""
baseline 统一运行器。

它把 baseline 的训练、预测、回测与结果落盘收口到一条链上，
从而让 baseline 与 PRIME 主实验尽量共享一致的评估口径。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backtest import BacktestResult, Backtester, PerformanceMetrics, save_backtest_result
from .datasets import (
    CrossSectionSequenceDataset,
    SequencePanelDataset,
    build_history_frame,
    identity_collate,
)
from .models import build_baseline_model
from .registry import available_baselines, get_baseline_spec


@dataclass
class BaselineRunConfig:
    baseline_name: str = "all"
    output_dir: str = "outputs_baselines"
    epochs_override: Optional[int] = None
    seq_len_override: Optional[int] = None
    hf_backbone_override: Optional[str] = None
    batch_size_override: Optional[int] = None
    quick: bool = False


def _to_python(value):
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def mean_daily_spearman(pred_df: pd.DataFrame) -> float:
    if pred_df.empty:
        return 0.0
    scores: List[float] = []
    for _, group in pred_df.groupby("trade_date"):
        if len(group) < 5:
            continue
        if group["prediction"].nunique() <= 1 or group["label"].nunique() <= 1:
            continue
        corr = group["prediction"].corr(group["label"], method="spearman")
        if pd.notna(corr):
            scores.append(float(corr))
    return float(np.mean(scores)) if scores else 0.0


def prediction_mse(pred_df: pd.DataFrame) -> float:
    if pred_df.empty:
        return 0.0
    return float(np.mean((pred_df["prediction"] - pred_df["label"]) ** 2))


def differentiable_ic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.std(unbiased=False) * target.std(unbiased=False) + 1e-6
    return (pred * target).mean() / denom


class BaselineRunner:
    def __init__(self, config, run_config: BaselineRunConfig, device: str = "cpu"):
        self.config = config
        self.run_config = run_config
        self.device = torch.device(device)

    def list_baselines(self) -> List[str]:
        return available_baselines()

    def run(
        self,
        splits,
        feature_cols: Dict[str, List[str]],
    ) -> List[Dict]:
        history_df = build_history_frame(splits)
        output_root = Path(self.run_config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        baseline_names = (
            available_baselines()
            if self.run_config.baseline_name in {"all", "*"}
            else [self.run_config.baseline_name]
        )
        summaries: List[Dict] = []

        for baseline_name in baseline_names:
            summary = self._run_single(
                history_df=history_df,
                train_df=splits.train,
                valid_df=splits.valid,
                test_df=splits.test,
                feature_cols=feature_cols,
                baseline_name=baseline_name,
                output_root=output_root,
            )
            summaries.append(summary)

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(output_root / "baseline_summary.csv", index=False)
        return summaries

    def _run_single(
        self,
        history_df: pd.DataFrame,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Dict[str, List[str]],
        baseline_name: str,
        output_root: Path,
    ) -> Dict:
        spec = get_baseline_spec(baseline_name)
        spec = self._apply_overrides(spec)
        if spec.name == "market_index":
            return self._run_market_index(spec, test_df, output_root)
        all_features = (
            list(feature_cols["bull"])
            + list(feature_cols["bear"])
            + list(feature_cols["friction"])
            + list(feature_cols["macro"])
        )
        baseline_dir = output_root / spec.name
        baseline_dir.mkdir(parents=True, exist_ok=True)

        model, train_loader, valid_loader, test_loader = self._prepare_model_and_loaders(
            spec,
            history_df,
            train_df,
            valid_df,
            test_df,
            all_features,
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )

        best_state = None
        best_metric = -np.inf
        patience = 0
        max_patience = 3 if not self.run_config.quick else 1
        train_history: List[Dict] = []

        for epoch in range(spec.epochs):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, spec)
            valid_pred = self._predict(model, valid_loader, spec)
            valid_ic = mean_daily_spearman(valid_pred)
            valid_mse = prediction_mse(valid_pred)
            train_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "valid_ic": valid_ic,
                    "valid_mse": valid_mse,
                }
            )
            print(
                f"    [{spec.display_name}] epoch {epoch + 1}/{spec.epochs} "
                f"loss={train_loss:.5f} valid_ic={valid_ic:.4f} valid_mse={valid_mse:.6f}"
            )
            if valid_ic > best_metric:
                best_metric = valid_ic
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        valid_pred = self._predict(model, valid_loader, spec)
        test_pred = self._predict(model, test_loader, spec)
        test_df_pred = test_df.merge(
            test_pred[["trade_date", "ts_code", "prediction"]],
            on=["trade_date", "ts_code"],
            how="left",
        )
        test_df_pred["prediction"] = test_df_pred["prediction"].fillna(0.0)

        backtester = Backtester(self.config)
        backtest_result = backtester.run(
            test_df_pred,
            signal_col="prediction",
            return_col="daily_return" if "daily_return" in test_df_pred.columns else "fwd_return",
            date_col="trade_date",
            code_col="ts_code",
            top_k=self.config.backtest.top_k,
        )
        save_backtest_result(backtest_result, baseline_dir / "backtest")

        metrics = {
            "valid_ic": mean_daily_spearman(valid_pred),
            "valid_mse": prediction_mse(valid_pred),
            "test_ic": mean_daily_spearman(test_pred),
            "test_mse": prediction_mse(test_pred),
            **{f"backtest_{k}": float(v) for k, v in backtest_result.metrics.items()},
        }

        torch.save(model.state_dict(), baseline_dir / "model.pt")
        pd.DataFrame(train_history).to_json(baseline_dir / "training_history.json", orient="records", indent=2)
        valid_pred.to_parquet(baseline_dir / "valid_predictions.parquet", index=False)
        test_pred.to_parquet(baseline_dir / "test_predictions.parquet", index=False)
        backtest_result.holdings.to_csv(baseline_dir / "holdings.csv", index=False)
        backtest_result.trades.to_csv(baseline_dir / "trades.csv", index=False)
        with open(baseline_dir / "metrics.json", "w") as f:
            json.dump(_to_python(metrics), f, indent=2)
        with open(baseline_dir / "paper_meta.json", "w") as f:
            json.dump(_to_python(asdict(spec)), f, indent=2)

        summary = {
            "baseline": spec.display_name,
            "family": spec.family,
            "fidelity": spec.fidelity,
            "valid_ic": metrics["valid_ic"],
            "test_ic": metrics["test_ic"],
            "annual_return": metrics.get("backtest_annual_return", 0.0),
            "sharpe_ratio": metrics.get("backtest_sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("backtest_max_drawdown", 0.0),
            "paper_url": spec.paper_url,
            "code_url": spec.code_url or "",
            "output_dir": str(baseline_dir),
        }
        return summary

    def _run_market_index(self, spec, test_df: pd.DataFrame, output_root: Path) -> Dict:
        baseline_dir = output_root / spec.name
        baseline_dir.mkdir(parents=True, exist_ok=True)

        bench_ret = self._resolve_benchmark_returns(test_df)
        bench_ret = bench_ret.sort_index()
        metrics = PerformanceMetrics.calculate_all(bench_ret, bench_ret)
        metrics["info_ratio"] = None
        metrics["excess_annual"] = 0.0
        metrics["benchmark_proxy"] = self._benchmark_source_name(test_df)

        benchmark_df = pd.DataFrame(
            {
                "trade_date": bench_ret.index,
                "benchmark_return": bench_ret.values,
            }
        )
        benchmark_df.to_csv(baseline_dir / "benchmark_returns.csv", index=False)
        result = BacktestResult(
            portfolio_returns=bench_ret,
            benchmark_returns=bench_ret,
            excess_returns=bench_ret * 0.0,
            portfolio_nav=(1.0 + bench_ret).cumprod(),
            benchmark_nav=(1.0 + bench_ret).cumprod(),
            metrics=metrics,
            holdings=pd.DataFrame(),
            trades=pd.DataFrame(),
            energy_attribution=None,
        )
        save_backtest_result(result, baseline_dir / "backtest")
        with open(baseline_dir / "metrics.json", "w") as f:
            json.dump(_to_python(metrics), f, indent=2)
        with open(baseline_dir / "paper_meta.json", "w") as f:
            json.dump(_to_python(asdict(spec)), f, indent=2)

        return {
            "baseline": spec.display_name,
            "family": spec.family,
            "fidelity": spec.fidelity,
            "valid_ic": 0.0,
            "test_ic": 0.0,
            "annual_return": float(metrics.get("annual_return", 0.0)),
            "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "paper_url": spec.paper_url,
            "code_url": spec.code_url or "",
            "output_dir": str(baseline_dir),
        }

    def _apply_overrides(self, spec):
        data = asdict(spec)
        if self.run_config.epochs_override is not None:
            data["epochs"] = int(self.run_config.epochs_override)
        if self.run_config.seq_len_override is not None:
            data["seq_len"] = int(self.run_config.seq_len_override)
        if self.run_config.batch_size_override is not None:
            data["batch_size"] = int(self.run_config.batch_size_override)
        if self.run_config.hf_backbone_override is not None:
            data["hf_backbone"] = self.run_config.hf_backbone_override
        if self.run_config.quick:
            data["epochs"] = min(3, data["epochs"])
        return type(spec)(**data)

    def _prepare_model_and_loaders(
        self,
        spec,
        history_df: pd.DataFrame,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ):
        if spec.cross_sectional:
            train_dataset = CrossSectionSequenceDataset(history_df, train_df, feature_cols, spec.seq_len)
            valid_dataset = CrossSectionSequenceDataset(history_df, valid_df, feature_cols, spec.seq_len)
            test_dataset = CrossSectionSequenceDataset(history_df, test_df, feature_cols, spec.seq_len)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=identity_collate)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)
        else:
            train_dataset = SequencePanelDataset(history_df, train_df, feature_cols, spec.seq_len)
            valid_dataset = SequencePanelDataset(history_df, valid_df, feature_cols, spec.seq_len)
            test_dataset = SequencePanelDataset(history_df, test_df, feature_cols, spec.seq_len)
            train_loader = DataLoader(train_dataset, batch_size=spec.batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(valid_dataset, batch_size=spec.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=spec.batch_size, shuffle=False, num_workers=0)

        model = build_baseline_model(spec, input_dim=len(feature_cols)).to(self.device)
        train_count = len(train_dataset)
        valid_count = len(valid_dataset)
        test_count = len(test_dataset)
        if train_count == 0 or valid_count == 0 or test_count == 0:
            raise ValueError(
                f"{spec.display_name} 数据集为空: train={train_count}, valid={valid_count}, test={test_count}. "
                f"可尝试减小 --baseline_seq_len 或检查当前市场窗口。"
            )
        print(
            f"\n>>> Baseline {spec.display_name}: train={train_count} valid={valid_count} test={test_count} "
            f"seq_len={spec.seq_len}"
        )
        print(f"    paper: {spec.paper_title}")
        print(f"    fidelity: {spec.fidelity}")
        return model, train_loader, valid_loader, test_loader

    def _benchmark_source_name(self, test_df: pd.DataFrame) -> str:
        for col in ["benchmark_return", "market_return", "market_ret", "index_return"]:
            if col in test_df.columns:
                return col
        return "equal_weight_proxy"

    def _resolve_benchmark_returns(self, test_df: pd.DataFrame) -> pd.Series:
        date_col = "trade_date"
        for col in ["benchmark_return", "market_return", "market_ret", "index_return"]:
            if col in test_df.columns:
                series = (
                    test_df[[date_col, col]]
                    .dropna()
                    .drop_duplicates(subset=[date_col])
                    .sort_values(date_col)
                    .set_index(date_col)[col]
                    .astype(float)
                )
                if series.abs().max() > 1:
                    series = series / 100.0
                return series

        return_col = "daily_return" if "daily_return" in test_df.columns else "fwd_return"
        proxy = (
            test_df[[date_col, return_col]]
            .dropna()
            .groupby(date_col)[return_col]
            .mean()
            .sort_index()
            .astype(float)
        )
        return proxy

    def _train_one_epoch(self, model, loader, optimizer, spec) -> float:
        model.train()
        total_loss = 0.0
        n_steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = self._compute_loss(model, batch, spec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_steps += 1
        return total_loss / max(n_steps, 1)

    def _compute_loss(self, model, batch, spec) -> torch.Tensor:
        if spec.cross_sectional:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            industry_id = batch.get("industry_id")
            if industry_id is not None:
                industry_id = industry_id.to(self.device)
            pred, aux = model(x, industry_id=industry_id)
            if spec.name == "alphastock":
                weights = torch.softmax(pred, dim=0)
                portfolio_return = torch.sum(weights * y)
                concentration = torch.sum(weights.pow(2))
                ic_term = differentiable_ic(pred, y)
                return -portfolio_return + 0.02 * concentration - 0.05 * ic_term
            if spec.name == "deeptrader":
                weights = torch.softmax(pred, dim=0)
                portfolio_return = torch.sum(weights * y)
                downside = torch.relu(-portfolio_return)
                concentration = torch.sum(weights.pow(2))
                value_target = y.mean().detach()
                value_loss = F.mse_loss(aux["value"].reshape(()), value_target.reshape(()))
                ic_term = differentiable_ic(pred, y)
                return -portfolio_return + 0.30 * downside + 0.02 * concentration + 0.25 * value_loss - 0.03 * ic_term
            if spec.name == "deeppocket":
                weights = torch.softmax(pred, dim=0)
                portfolio_return = torch.sum(weights * y)
                concentration = torch.sum(weights.pow(2))
                value_target = y.mean().detach()
                value_loss = F.mse_loss(aux["value"].reshape(()), value_target.reshape(()))
                smoothness = aux.get("smoothness", torch.tensor(0.0, device=self.device))
                ic_term = differentiable_ic(pred, y)
                return -portfolio_return + 0.02 * concentration + 0.20 * value_loss + 0.05 * smoothness - 0.03 * ic_term
            if spec.name == "alphagat":
                weights = torch.softmax(pred, dim=0)
                portfolio_return = torch.sum(weights * y)
                concentration = torch.sum(weights.pow(2))
                ic_term = differentiable_ic(pred, y)
                return -portfolio_return + 0.02 * concentration - 0.03 * ic_term
            if spec.name == "stockformer":
                mse = F.mse_loss(pred, y)
                direction_logit = aux.get("direction_logit")
                direction_target = batch["direction"].to(self.device)
                cls = (
                    F.binary_cross_entropy_with_logits(direction_logit, direction_target)
                    if direction_logit is not None
                    else torch.tensor(0.0, device=self.device)
                )
                ic_term = differentiable_ic(pred, y)
                return mse + 0.3 * cls - 0.05 * ic_term
            mse = F.mse_loss(pred, y)
            ic_term = differentiable_ic(pred, y)
            return mse - 0.05 * ic_term

        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        pred, _ = model(x)
        mse = F.mse_loss(pred, y)
        ic_term = differentiable_ic(pred, y)
        return mse - 0.05 * ic_term

    @torch.no_grad()
    def _predict(self, model, loader, spec) -> pd.DataFrame:
        model.eval()
        rows: List[Dict] = []
        for batch in loader:
            if spec.cross_sectional:
                x = batch["x"].to(self.device)
                industry_id = batch.get("industry_id")
                if industry_id is not None:
                    industry_id = industry_id.to(self.device)
                pred, _ = model(x, industry_id=industry_id)
                pred_np = pred.detach().cpu().numpy()
                y_np = batch["y"].detach().cpu().numpy()
                date_value = int(batch["date_value"].detach().cpu().item())
                trade_date = pd.to_datetime(date_value)
                for code, p, y in zip(batch["codes"], pred_np, y_np):
                    rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": code,
                            "prediction": float(p),
                            "label": float(y),
                        }
                    )
                continue

            x = batch["x"].to(self.device)
            pred, _ = model(x)
            pred_np = pred.detach().cpu().numpy()
            y_np = batch["y"].detach().cpu().numpy()
            date_values = batch["date_value"].detach().cpu().numpy()
            codes = list(batch["code"])
            for code, date_value, p, y in zip(codes, date_values, pred_np, y_np):
                rows.append(
                    {
                        "trade_date": pd.to_datetime(int(date_value)),
                        "ts_code": code,
                        "prediction": float(p),
                        "label": float(y),
                    }
                )
        pred_df = pd.DataFrame(rows)
        if not pred_df.empty:
            pred_df = pred_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        return pred_df
