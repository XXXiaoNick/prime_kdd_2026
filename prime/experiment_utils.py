"""
实验运行辅助函数。

这里放的是 experiment_runner 会反复用到、但不适合塞进 runner 主体的
小工具，例如特征分组扰动、宏观信号退化和排名指标计算。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans

from data_loader import AdaptiveFeatureConfig, create_data_pipeline
from dataset import prepare_feature_arrays, resolve_model_input_dims, slice_feature_tensors


def load_experiment_pipeline(config, fast: bool = True, no_cache: bool = False):
    splits, full_df, _ = create_data_pipeline(
        config,
        fast=fast,
        use_gpu=True,
        use_cache=not no_cache,
        allow_source_fallback=fast,
    )

    adapter = AdaptiveFeatureConfig()
    data_config = adapter.create_config_from_data(full_df, config)
    feature_cols = data_config["feature_groups"]
    macro_embed_cols = [col for col in full_df.columns if col.startswith("macro_embed_")]
    if macro_embed_cols:
        feature_cols["macro"] = macro_embed_cols

    cleaner = data_config["cleaner"]
    all_features = feature_cols["bull"] + feature_cols["bear"] + feature_cols["friction"] + feature_cols["macro"]
    splits.train = cleaner.transform(splits.train, all_features)
    splits.valid = cleaner.transform(splits.valid, all_features)
    splits.test = cleaner.transform(splits.test, all_features)
    return splits, full_df, feature_cols


def build_feature_groups(
    feature_cols: Dict[str, List[str]],
    train_df: pd.DataFrame,
    action: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, List[str]]:
    feature_cols = {k: list(v) for k, v in feature_cols.items()}
    if not action:
        return feature_cols

    if action == "swap_bull_bear":
        feature_cols["bull"], feature_cols["bear"] = feature_cols["bear"], feature_cols["bull"]
        return feature_cols

    if action == "single_group":
        merged = list(feature_cols["bull"] + feature_cols["bear"] + feature_cols["friction"])
        feature_cols["bull"] = merged
        feature_cols["bear"] = []
        feature_cols["friction"] = []
        return feature_cols

    if action == "random_grouping":
        rng = np.random.default_rng(seed)
        merged = list(feature_cols["bull"] + feature_cols["bear"] + feature_cols["friction"])
        rng.shuffle(merged)
        n_bull = len(feature_cols["bull"])
        n_bear = len(feature_cols["bear"])
        feature_cols["bull"] = merged[:n_bull]
        feature_cols["bear"] = merged[n_bull:n_bull + n_bear]
        feature_cols["friction"] = merged[n_bull + n_bear:]
        return feature_cols

    if action == "kmeans_grouping":
        merged = list(feature_cols["bull"] + feature_cols["bear"] + feature_cols["friction"])
        stats = []
        for col in merged:
            values = train_df[col].fillna(0.0)
            corr = float(pd.Series(values).corr(train_df["fwd_return"])) if "fwd_return" in train_df.columns else 0.0
            stats.append([corr if pd.notna(corr) else 0.0, float(values.std()), float(values.abs().mean())])
        if len(stats) >= 3:
            labels = KMeans(n_clusters=3, random_state=seed, n_init=10).fit_predict(np.asarray(stats))
            cluster_score = {}
            for cluster_id in range(3):
                idx = np.where(labels == cluster_id)[0]
                cluster_score[cluster_id] = float(np.mean([stats[i][0] for i in idx])) if len(idx) else 0.0
            ordered = sorted(cluster_score, key=lambda cid: cluster_score[cid], reverse=True)
            mapping = {ordered[0]: "bull", ordered[1]: "friction", ordered[2]: "bear"}
            new_groups = {"bull": [], "bear": [], "friction": [], "macro": list(feature_cols["macro"])}
            for col, label in zip(merged, labels):
                new_groups[mapping[int(label)]].append(col)
            return new_groups
        return feature_cols

    return feature_cols


def apply_macro_data_action(splits, macro_cols: List[str], action: Optional[str], seed: int = 42):
    if not action or not macro_cols:
        return splits

    rng = np.random.default_rng(seed)
    for attr in ("train", "valid", "test"):
        frame = getattr(splits, attr).copy()
        if action == "macro_noise_0_5":
            std = frame[macro_cols].std().replace(0, 1.0).values
            noise = rng.normal(0.0, 0.5, size=(len(frame), len(macro_cols))) * std
            frame.loc[:, macro_cols] = frame[macro_cols].values + noise
        elif action == "macro_missing_50":
            mask = rng.uniform(0.0, 1.0, size=(len(frame), len(macro_cols))) < 0.5
            values = frame[macro_cols].values.copy()
            values[mask] = 0.0
            frame.loc[:, macro_cols] = values
        elif action == "macro_zero_out":
            frame.loc[:, macro_cols] = 0.0
        setattr(splits, attr, frame)
    return splits


def score_with_integrator(df: pd.DataFrame, integrator, feature_cols: Dict[str, List[str]]) -> pd.DataFrame:
    df = df.copy()
    integrator.game_ebm.eval()
    expected_dims = resolve_model_input_dims(integrator.game_ebm, feature_cols)
    feature_arrays, _ = prepare_feature_arrays(df, feature_cols, expected_dims)

    rows = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            x_bull, x_bear, x_friction, x_macro = slice_feature_tensors(
                feature_arrays,
                start,
                end,
                integrator.device,
            )
            energy, components = integrator.game_ebm(
                x_bull,
                x_bear,
                x_friction,
                x_macro,
                return_components=True,
            )
            chunk = df.iloc[start:end].copy()
            chunk["ebm_score"] = (-energy.squeeze()).cpu().numpy()
            chunk["E_bull"] = components.get("E_bull", torch.zeros_like(energy)).squeeze().cpu().numpy()
            chunk["E_bear"] = components.get("E_bear", torch.zeros_like(energy)).squeeze().cpu().numpy()
            chunk["E_heat"] = components.get(
                "E_heat",
                components.get("E_friction", torch.zeros_like(energy)),
            ).squeeze().cpu().numpy()
            chunk["alpha_market"] = components.get("alpha_market", torch.ones_like(energy)).squeeze().cpu().numpy()
            chunk["beta_risk"] = components.get("beta_risk", torch.ones_like(energy)).squeeze().cpu().numpy()
            chunk["gamma_heat"] = components.get("gamma_heat", torch.zeros_like(energy)).squeeze().cpu().numpy()
            rows.append(chunk)
    return pd.concat(rows, ignore_index=True) if rows else df


def daily_rank_ic(pred_df: pd.DataFrame, score_col: str = "ebm_score", label_col: str = "fwd_return") -> float:
    values = []
    for _, group in pred_df.groupby("trade_date"):
        if len(group) < 5 or group[score_col].nunique() <= 1 or group[label_col].nunique() <= 1:
            continue
        corr, _ = spearmanr(group[score_col], group[label_col])
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def daily_kendall_tau(pred_df: pd.DataFrame, score_col: str = "ebm_score", label_col: str = "fwd_return") -> float:
    values = []
    for _, group in pred_df.groupby("trade_date"):
        if len(group) < 5 or group[score_col].nunique() <= 1 or group[label_col].nunique() <= 1:
            continue
        tau, _ = kendalltau(group[score_col], group[label_col])
        if pd.notna(tau):
            values.append(float(tau))
    return float(np.mean(values)) if values else 0.0


def compute_guardian_stats(guardian, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[float, float]:
    if df.empty or not feature_cols:
        return 0.0, 0.0
    if getattr(guardian, "_is_dummy", False):
        return 0.0, 0.0

    cols = [col for col in feature_cols if col in df.columns]
    if not cols or "crash_label" not in df.columns:
        return 0.0, 0.0

    X = np.nan_to_num(df[cols].fillna(0).values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["crash_label"].values.astype(int)
    pred = guardian.predict(X)
    positives = max(1, int((y == 1).sum()))
    negatives = max(1, int((y == 0).sum()))
    recall = float(((pred == 1) & (y == 1)).sum() / positives)
    fpr = float(((pred == 1) & (y == 0)).sum() / negatives)
    return recall, fpr
