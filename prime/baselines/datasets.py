"""
baseline 专用数据集构造模块。

职责：
1. 把主项目的 panel / split 数据转换成 baseline 所需的样本格式
2. 统一序列长度、标签窗口和横截面组织方式
3. 尽量复用主项目已有的预处理结果
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def identity_collate(batch):
    return batch[0]


@dataclass
class HistoryBlock:
    features: np.ndarray
    labels: np.ndarray
    crash_labels: np.ndarray
    directions: np.ndarray
    dates: np.ndarray
    industries: List[str]
    date_to_pos: Dict[int, int]


def build_history_frame(splits) -> pd.DataFrame:
    history_df = pd.concat([splits.train, splits.valid, splits.test], axis=0, ignore_index=True)
    history_df = history_df.drop_duplicates(subset=["trade_date", "ts_code"]).copy()
    history_df["trade_date"] = pd.to_datetime(history_df["trade_date"])
    history_df = history_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return history_df


def _ensure_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = 0.0
    return df


def _normalize_date_values(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series).astype("int64").to_numpy()


class PanelHistoryIndex:
    def __init__(
        self,
        history_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "fwd_return",
        crash_label_col: str = "crash_label",
        date_col: str = "trade_date",
        code_col: str = "ts_code",
        industry_col: str = "industry",
    ):
        history_df = history_df.copy()
        history_df[date_col] = pd.to_datetime(history_df[date_col])
        history_df = _ensure_feature_matrix(history_df, feature_cols)
        history_df = history_df.sort_values([code_col, date_col]).reset_index(drop=True)

        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.crash_label_col = crash_label_col
        self.date_col = date_col
        self.code_col = code_col
        self.industry_col = industry_col

        self.blocks: Dict[str, HistoryBlock] = {}
        self.industry_to_id: Dict[str, int] = {"Unknown": 0}

        if industry_col in history_df.columns:
            industries = (
                history_df[industry_col]
                .fillna("Unknown")
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            ordered = ["Unknown"] + [name for name in industries if name != "Unknown"]
            self.industry_to_id = {name: idx for idx, name in enumerate(ordered)}

        for code, group in history_df.groupby(code_col, sort=False):
            group = group.sort_values(date_col).reset_index(drop=True)
            date_values = _normalize_date_values(group[date_col])
            self.blocks[str(code)] = HistoryBlock(
                features=np.nan_to_num(group[self.feature_cols].to_numpy(dtype=np.float32)),
                labels=np.nan_to_num(group[label_col].to_numpy(dtype=np.float32))
                if label_col in group.columns
                else np.zeros(len(group), dtype=np.float32),
                crash_labels=np.nan_to_num(group[crash_label_col].to_numpy(dtype=np.float32))
                if crash_label_col in group.columns
                else np.zeros(len(group), dtype=np.float32),
                directions=(np.nan_to_num(group[label_col].to_numpy(dtype=np.float32)) > 0).astype(np.float32)
                if label_col in group.columns
                else np.zeros(len(group), dtype=np.float32),
                dates=date_values,
                industries=group[industry_col].fillna("Unknown").astype(str).tolist()
                if industry_col in group.columns
                else ["Unknown"] * len(group),
                date_to_pos={int(date): idx for idx, date in enumerate(date_values)},
            )


class SequencePanelDataset(Dataset):
    def __init__(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int,
        label_col: str = "fwd_return",
        crash_label_col: str = "crash_label",
        date_col: str = "trade_date",
        code_col: str = "ts_code",
    ):
        self.seq_len = int(seq_len)
        self.date_col = date_col
        self.code_col = code_col
        self.history = PanelHistoryIndex(
            history_df,
            feature_cols=feature_cols,
            label_col=label_col,
            crash_label_col=crash_label_col,
            date_col=date_col,
            code_col=code_col,
        )

        target_df = target_df.copy()
        target_df[date_col] = pd.to_datetime(target_df[date_col])
        target_df = target_df.sort_values([date_col, code_col]).reset_index(drop=True)

        self.samples: List[Tuple[str, int]] = []
        for row in target_df[[date_col, code_col]].itertuples(index=False):
            date_value = int(pd.Timestamp(row[0]).value)
            code = str(row[1])
            block = self.history.blocks.get(code)
            if block is None:
                continue
            pos = block.date_to_pos.get(date_value)
            if pos is None or pos + 1 < self.seq_len:
                continue
            self.samples.append((code, pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        code, end_pos = self.samples[idx]
        block = self.history.blocks[code]
        start_pos = end_pos - self.seq_len + 1
        x = block.features[start_pos : end_pos + 1]
        item = {
            "x": torch.from_numpy(x),
            "y": torch.tensor(block.labels[end_pos], dtype=torch.float32),
            "crash_label": torch.tensor(block.crash_labels[end_pos], dtype=torch.float32),
            "direction": torch.tensor(block.directions[end_pos], dtype=torch.float32),
            "date_value": torch.tensor(block.dates[end_pos], dtype=torch.int64),
            "code": code,
        }
        return item


class CrossSectionSequenceDataset(Dataset):
    def __init__(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int,
        label_col: str = "fwd_return",
        crash_label_col: str = "crash_label",
        date_col: str = "trade_date",
        code_col: str = "ts_code",
        industry_col: str = "industry",
        min_assets: int = 8,
    ):
        self.seq_len = int(seq_len)
        self.date_col = date_col
        self.code_col = code_col
        self.industry_col = industry_col
        self.min_assets = int(min_assets)
        self.history = PanelHistoryIndex(
            history_df,
            feature_cols=feature_cols,
            label_col=label_col,
            crash_label_col=crash_label_col,
            date_col=date_col,
            code_col=code_col,
            industry_col=industry_col,
        )

        target_df = target_df.copy()
        target_df[date_col] = pd.to_datetime(target_df[date_col])
        if industry_col in target_df.columns:
            target_df[industry_col] = target_df[industry_col].fillna("Unknown").astype(str)
        else:
            target_df[industry_col] = "Unknown"
        target_df = target_df.sort_values([date_col, code_col]).reset_index(drop=True)

        self.samples: List[Tuple[int, List[Tuple[str, int]]]] = []
        for date_value, day_df in target_df.groupby(date_col, sort=True):
            sample_entries: List[Tuple[str, int]] = []
            for row in day_df[[code_col]].itertuples(index=False):
                code = str(row[0])
                block = self.history.blocks.get(code)
                if block is None:
                    continue
                pos = block.date_to_pos.get(int(pd.Timestamp(date_value).value))
                if pos is None or pos + 1 < self.seq_len:
                    continue
                sample_entries.append((code, pos))
            if len(sample_entries) >= self.min_assets:
                self.samples.append((int(pd.Timestamp(date_value).value), sample_entries))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        date_value, entries = self.samples[idx]
        xs: List[np.ndarray] = []
        ys: List[float] = []
        crash_labels: List[float] = []
        directions: List[float] = []
        codes: List[str] = []
        industry_ids: List[int] = []

        for code, end_pos in entries:
            block = self.history.blocks[code]
            start_pos = end_pos - self.seq_len + 1
            xs.append(block.features[start_pos : end_pos + 1])
            ys.append(float(block.labels[end_pos]))
            crash_labels.append(float(block.crash_labels[end_pos]))
            directions.append(float(block.directions[end_pos]))
            codes.append(code)
            industry_name = block.industries[end_pos] if end_pos < len(block.industries) else "Unknown"
            industry_ids.append(self.history.industry_to_id.get(industry_name, 0))

        return {
            "x": torch.tensor(np.stack(xs, axis=0), dtype=torch.float32),
            "y": torch.tensor(ys, dtype=torch.float32),
            "crash_label": torch.tensor(crash_labels, dtype=torch.float32),
            "direction": torch.tensor(directions, dtype=torch.float32),
            "industry_id": torch.tensor(industry_ids, dtype=torch.long),
            "date_value": torch.tensor(date_value, dtype=torch.int64),
            "codes": codes,
        }
