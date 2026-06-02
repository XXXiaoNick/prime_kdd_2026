"""
================================================================================
PyTorch 数据集与模型输入模块
================================================================================

当前文件统一承接两类职责：
1. 训练/验证/排序学习用的 Dataset 与 DataLoader
2. 推理阶段共用的特征过滤、维度对齐、张量切片工具

这样可以把“DataFrame -> tensor”相关逻辑集中到一个地方，避免 main / trainer
里重复维护同一套输入准备代码。
================================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class TrainingDataBundle:
    """共享底层数据张量的数据集集合。"""

    train_dataset: 'StockDataset'
    valid_dataset: 'StockDataset'
    pairwise_dataset: 'PairwiseDataset'

    def get_feature_dims(self) -> Dict[str, int]:
        return self.train_dataset.get_feature_dims()


class StockDataset(Dataset):
    """
    股票数据集 (修复版)
    
    支持两种宏观数据模式：
    1. embedding模式（推荐）：使用预计算的macro_embed_*列
    2. sequence模式：实时构建滑动窗口（用于GRU）
    
    特点：
    - 自动处理缺失特征（使用零填充）
    - 自动处理NaN值
    - 支持zscore和原始特征
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        bull_features: List[str],
        bear_features: List[str],
        friction_features: List[str],
        macro_features: List[str],
        label_col: str = 'fwd_return',
        crash_label_col: str = 'crash_label',
        use_zscore: bool = True,
        macro_mode: str = 'embedding',
        macro_seq_len: int = 12,
        date_col: str = 'trade_date'
    ):
        self.df = df.reset_index(drop=True)
        self.use_zscore = use_zscore
        self.macro_mode = macro_mode
        self.macro_seq_len = macro_seq_len
        self.date_col = date_col
        
        suffix = '_zscore' if use_zscore else ''
        
        # 【关键】获取可用特征，不存在的特征会被跳过
        self.bull_cols = self._get_feature_cols(bull_features, suffix)
        self.bear_cols = self._get_feature_cols(bear_features, suffix)
        self.friction_cols = self._get_feature_cols(friction_features, suffix)
        
        if macro_mode == 'embedding':
            self.macro_cols = self._get_macro_embedding_cols()
        else:
            self.macro_cols = self._get_feature_cols(macro_features, '')
            self._prepare_macro_sequence_index()
        
        self.label_col = label_col
        self.crash_label_col = crash_label_col if crash_label_col in df.columns else None
        
        # 打印特征信息
        self._print_feature_info()
        
        self._prepare_arrays()
        
    def _get_feature_cols(self, features: List[str], suffix: str) -> List[str]:
        """获取可用的特征列，自动处理缺失特征"""
        cols = []
        available = set(self.df.columns)
        
        for f in features:
            # 优先检查zscore版本
            if f'{f}{suffix}' in available:
                cols.append(f'{f}{suffix}')
            # 然后检查原始版本
            elif f in available:
                cols.append(f)
            # 特征不存在则跳过（不报错）
        
        return cols
    
    def _print_feature_info(self):
        """打印特征信息（仅首次创建时）"""
        if not hasattr(StockDataset, '_info_printed'):
            StockDataset._info_printed = True
            total = len(self.bull_cols) + len(self.bear_cols) + len(self.friction_cols) + len(self.macro_cols)
            print(f"    Dataset特征: Bull={len(self.bull_cols)}, Bear={len(self.bear_cols)}, "
                  f"Friction={len(self.friction_cols)}, Macro={len(self.macro_cols)}, 总计={total}")
    
    def _get_macro_embedding_cols(self) -> List[str]:
        embed_cols = [c for c in self.df.columns if c.startswith('macro_embed_')]
        if embed_cols:
            embed_cols = sorted(embed_cols, key=lambda x: int(x.split('_')[-1]))
        return embed_cols
    
    def _prepare_macro_sequence_index(self):
        """准备宏观序列索引（仅sequence模式）"""
        if self.date_col not in self.df.columns or not self.macro_cols:
            return
        
        self.date_to_idx = {}
        macro_by_date = self.df.groupby(self.date_col)[self.macro_cols].mean()
        self.macro_by_date = macro_by_date.values.astype(np.float32)
        self.date_list = list(macro_by_date.index)
        
        for i, date in enumerate(self.date_list):
            self.date_to_idx[date] = i
    
    def _prepare_arrays(self):
        """预提取数据为numpy数组并预转换为tensor（避免__getitem__中重复创建）"""
        self.bull_data = self.df[self.bull_cols].values.astype(np.float32) if self.bull_cols else np.zeros((len(self.df), 1), dtype=np.float32)
        self.bear_data = self.df[self.bear_cols].values.astype(np.float32) if self.bear_cols else np.zeros((len(self.df), 1), dtype=np.float32)
        self.friction_data = self.df[self.friction_cols].values.astype(np.float32) if self.friction_cols else np.zeros((len(self.df), 1), dtype=np.float32)
        self.macro_data = self.df[self.macro_cols].values.astype(np.float32) if self.macro_cols else np.zeros((len(self.df), 1), dtype=np.float32)

        self.labels = self.df[self.label_col].values.astype(np.float32) if self.label_col in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
        self.crash_labels = self.df[self.crash_label_col].values.astype(np.float32) if self.crash_label_col else np.zeros(len(self.df), dtype=np.float32)

        # 处理NaN
        self.bull_data = np.nan_to_num(self.bull_data, nan=0.0)
        self.bear_data = np.nan_to_num(self.bear_data, nan=0.0)
        self.friction_data = np.nan_to_num(self.friction_data, nan=0.0)
        self.macro_data = np.nan_to_num(self.macro_data, nan=0.0)
        self.labels = np.nan_to_num(self.labels, nan=0.0)

        # 预转换为tensor，避免__getitem__中重复创建tensor的开销
        self._bull_tensor = torch.from_numpy(self.bull_data)
        self._bear_tensor = torch.from_numpy(self.bear_data)
        self._friction_tensor = torch.from_numpy(self.friction_data)
        self._macro_tensor = torch.from_numpy(self.macro_data)
        self._labels_tensor = torch.from_numpy(self.labels)
        self._crash_labels_tensor = torch.from_numpy(self.crash_labels)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 使用预转换的tensor，通过索引获取子张量（零拷贝，极快）
        bull = self._bull_tensor[idx]
        bear = self._bear_tensor[idx]
        friction = self._friction_tensor[idx]

        result = {
            'bull': bull,
            'bear': bear,
            'friction': friction,
            'label': self._labels_tensor[idx],
            'crash_label': self._crash_labels_tensor[idx],
        }

        if self.macro_mode == 'embedding':
            macro = self._macro_tensor[idx]
            result['macro'] = macro
        else:
            macro = self._get_macro_sequence(idx)
            result['macro'] = macro

        # 拼接所有特征（用于残差EBM）
        if self.macro_mode == 'embedding':
            result['all_features'] = torch.cat([bull, bear, friction, macro], dim=-1)
        else:
            macro_flat = macro[-1] if len(macro.shape) > 1 else macro
            result['all_features'] = torch.cat([bull, bear, friction, macro_flat], dim=-1)

        return result
    
    def _get_macro_sequence(self, idx: int) -> torch.Tensor:
        """获取宏观序列"""
        if not hasattr(self, 'date_to_idx') or not self.macro_cols:
            return torch.zeros(self.macro_seq_len, 1)
        
        date = self.df.iloc[idx][self.date_col]
        if date not in self.date_to_idx:
            return torch.zeros(self.macro_seq_len, len(self.macro_cols))
        
        date_idx = self.date_to_idx[date]
        start_idx = max(0, date_idx - self.macro_seq_len + 1)
        seq = self.macro_by_date[start_idx:date_idx + 1]
        
        if len(seq) < self.macro_seq_len:
            padding = np.zeros((self.macro_seq_len - len(seq), len(self.macro_cols)), dtype=np.float32)
            seq = np.vstack([padding, seq])
        
        return torch.from_numpy(seq)
    
    def get_feature_dims(self) -> Dict[str, int]:
        return {
            'bull': len(self.bull_cols) if self.bull_cols else 1,
            'bear': len(self.bear_cols) if self.bear_cols else 1,
            'friction': len(self.friction_cols) if self.friction_cols else 1,
            'macro': len(self.macro_cols) if self.macro_cols else 1,
        }


class PairwiseDataset(Dataset):
    """成对数据集 - 用于排序学习"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        bull_features: List[str],
        bear_features: List[str],
        friction_features: List[str],
        macro_features: List[str],
        label_col: str = 'fwd_return',
        date_col: str = 'trade_date',
        pairs_per_day: int = 100,
        use_zscore: bool = True,
        macro_mode: str = 'embedding',
        head_sampling_ratio: float = 0.5,
        min_return_diff: float = 0.01,
        seed: int = 42,
        base_dataset: Optional[StockDataset] = None,
    ):
        # 【关键修复】重置索引，确保索引从0开始连续
        self.df = df.copy().reset_index(drop=True)
        self.date_col = date_col
        self.label_col = label_col
        self.pairs_per_day = pairs_per_day
        self.head_sampling_ratio = float(np.clip(head_sampling_ratio, 0.05, 1.0))
        self.min_return_diff = max(0.0, float(min_return_diff))
        self.rng = np.random.default_rng(seed)

        self.base_dataset = base_dataset or StockDataset(
            self.df,
            bull_features,
            bear_features,
            friction_features,
            macro_features,
            label_col,
            use_zscore=use_zscore,
            macro_mode=macro_mode,
        )
        
        self.pairs = self._generate_pairs()
    
    def _generate_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        
        if self.date_col not in self.df.columns:
            n = len(self.df)
            indices = np.arange(n)
            for _ in range(self.pairs_per_day * 100):
                i, j = np.random.choice(indices, 2, replace=False)
                if self.base_dataset.labels[i] != self.base_dataset.labels[j]:
                    pairs.append((i, j))
            return pairs
        
        for date, group in self.df.groupby(self.date_col):
            indices = group.index.tolist()
            labels = self.base_dataset.labels[indices]
            
            if len(indices) < 2:
                continue

            sorted_positions = np.argsort(labels)
            tail_size = max(1, int(len(indices) * self.head_sampling_ratio))

            loser_pos = sorted_positions[:tail_size]
            winner_pos = sorted_positions[-tail_size:]

            winners = [indices[pos] for pos in winner_pos]
            losers = [indices[pos] for pos in loser_pos]
            
            if not winners or not losers:
                continue

            winner_returns = self.base_dataset.labels[winners]
            loser_returns = self.base_dataset.labels[losers]
            informative_winners = [
                idx for idx, ret in zip(winners, winner_returns)
                if (ret - loser_returns.min()) >= self.min_return_diff
            ]
            informative_losers = [
                idx for idx, ret in zip(losers, loser_returns)
                if (winner_returns.max() - ret) >= self.min_return_diff
            ]

            if informative_winners and informative_losers:
                winners = informative_winners
                losers = informative_losers

            n_pairs = min(self.pairs_per_day, len(winners) * len(losers))
            for _ in range(n_pairs):
                winner_idx = int(self.rng.choice(winners))
                loser_idx = int(self.rng.choice(losers))
                if self.base_dataset.labels[winner_idx] <= self.base_dataset.labels[loser_idx]:
                    continue
                pairs.append((winner_idx, loser_idx))

        if not pairs and len(self.df) >= 2:
            labels = self.base_dataset.labels
            sorted_idx = np.argsort(labels)
            low_idx = int(sorted_idx[0])
            high_idx = int(sorted_idx[-1])
            if labels[high_idx] > labels[low_idx]:
                pairs.append((high_idx, low_idx))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        winner_idx, loser_idx = self.pairs[idx]
        winner = self.base_dataset[winner_idx]
        loser = self.base_dataset[loser_idx]
        
        return {
            'winner_bull': winner['bull'],
            'winner_bear': winner['bear'],
            'winner_friction': winner['friction'],
            'winner_macro': winner['macro'],
            'winner_label': winner['label'],
            'loser_bull': loser['bull'],
            'loser_bear': loser['bear'],
            'loser_friction': loser['friction'],
            'loser_macro': loser['macro'],
            'loser_label': loser['label'],
        }
    
    def get_feature_dims(self) -> Dict[str, int]:
        return self.base_dataset.get_feature_dims()


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config,
    mode: str = 'energy',
    macro_mode: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    feature_cfg = config.feature
    training_cfg = config.training
    macro_mode = macro_mode or getattr(config.model, 'macro_mode', 'embedding')
    
    if mode == 'simple':
        train_dataset = StockDataset(
            train_df, feature_cfg.bull_features, feature_cfg.bear_features,
            feature_cfg.friction_features, feature_cfg.macro_features, macro_mode=macro_mode
        )
        valid_dataset = StockDataset(
            valid_df, feature_cfg.bull_features, feature_cfg.bear_features,
            feature_cfg.friction_features, feature_cfg.macro_features, macro_mode=macro_mode
        )
    
    elif mode == 'energy':
        train_dataset = PairwiseDataset(
            train_df, feature_cfg.bull_features, feature_cfg.bear_features,
            feature_cfg.friction_features, feature_cfg.macro_features,
            macro_mode=macro_mode,
            pairs_per_day=getattr(training_cfg, 'pairs_per_day', 100),
            head_sampling_ratio=getattr(training_cfg, 'head_sampling_ratio', 0.5),
            min_return_diff=getattr(training_cfg, 'min_return_diff', 0.01),
            seed=getattr(config, 'seed', 42),
        )
        valid_dataset = StockDataset(
            valid_df, feature_cfg.bull_features, feature_cfg.bear_features,
            feature_cfg.friction_features, feature_cfg.macro_features, macro_mode=macro_mode
        )
    
    else:
        train_dataset = StockDataset(
            train_df, feature_cfg.bull_features, feature_cfg.bear_features,
            feature_cfg.friction_features, feature_cfg.macro_features,
            label_col='crash_label', macro_mode=macro_mode
        )
        valid_dataset = StockDataset(
            valid_df, feature_cfg.bull_features, feature_cfg.bear_features,
            feature_cfg.friction_features, feature_cfg.macro_features,
            label_col='crash_label', macro_mode=macro_mode
        )
    
    train_loader = _build_loader(
        train_dataset,
        training_cfg,
        shuffle=True,
        drop_last=True,
        loader_name='train',
    )
    valid_loader = _build_loader(
        valid_dataset,
        training_cfg,
        shuffle=False,
        drop_last=False,
        loader_name='valid',
    )
    
    return train_loader, valid_loader


def create_training_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config,
    macro_mode: Optional[str] = None,
) -> Tuple[TrainingDataBundle, DataLoader, DataLoader, DataLoader]:
    """
    创建共享底层数据的训练 DataLoader。

    与分别调用 `create_dataloaders(..., mode='simple')` 和
    `create_dataloaders(..., mode='energy')` 相比，可避免 train dataset
    被重复构建一次。
    """
    feature_cfg = config.feature
    training_cfg = config.training
    macro_mode = macro_mode or getattr(config.model, 'macro_mode', 'embedding')

    train_dataset = StockDataset(
        train_df,
        feature_cfg.bull_features,
        feature_cfg.bear_features,
        feature_cfg.friction_features,
        feature_cfg.macro_features,
        macro_mode=macro_mode,
    )
    valid_dataset = StockDataset(
        valid_df,
        feature_cfg.bull_features,
        feature_cfg.bear_features,
        feature_cfg.friction_features,
        feature_cfg.macro_features,
        macro_mode=macro_mode,
    )
    pairwise_dataset = PairwiseDataset(
        train_df,
        feature_cfg.bull_features,
        feature_cfg.bear_features,
        feature_cfg.friction_features,
        feature_cfg.macro_features,
        macro_mode=macro_mode,
        pairs_per_day=getattr(training_cfg, 'pairs_per_day', 100),
        head_sampling_ratio=getattr(training_cfg, 'head_sampling_ratio', 0.5),
        min_return_diff=getattr(training_cfg, 'min_return_diff', 0.01),
        seed=getattr(config, 'seed', 42),
        base_dataset=train_dataset,
    )

    bundle = TrainingDataBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        pairwise_dataset=pairwise_dataset,
    )

    train_loader = _build_loader(
        train_dataset,
        training_cfg,
        shuffle=True,
        drop_last=True,
        loader_name='train',
    )
    valid_loader = _build_loader(
        valid_dataset,
        training_cfg,
        shuffle=False,
        drop_last=False,
        loader_name='valid',
    )
    pairwise_loader = _build_loader(
        pairwise_dataset,
        training_cfg,
        shuffle=True,
        drop_last=True,
        loader_name='pairwise',
    )

    return bundle, train_loader, valid_loader, pairwise_loader


def _build_loader(
    dataset: Dataset,
    training_cfg,
    shuffle: bool,
    drop_last: bool,
    loader_name: str,
) -> DataLoader:
    """统一构建 DataLoader，减少重复参数拼装。"""
    if len(dataset) == 0:
        raise ValueError(
            f"{loader_name} dataset is empty. 请检查时间切分、筛选条件或输入数据范围。"
        )

    num_workers = getattr(training_cfg, 'num_workers', 4)
    persistent_workers = getattr(training_cfg, 'persistent_workers', True) and num_workers > 0
    prefetch_factor = getattr(training_cfg, 'prefetch_factor', 2) if num_workers > 0 else None
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


# ============================================================================
# 推理与回测共用的模型输入工具（原 model_inputs.py）
# ============================================================================

FEATURE_GROUP_NAMES: Tuple[str, ...] = ("bull", "bear", "friction", "macro")


def filter_feature_groups(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """仅保留当前 DataFrame 中真实存在的特征列。"""
    available_cols = set(df.columns)
    return {
        group: [col for col in feature_groups.get(group, []) if col in available_cols]
        for group in FEATURE_GROUP_NAMES
    }


def resolve_model_input_dims(
    model: torch.nn.Module,
    feature_groups: Dict[str, List[str]],
) -> Dict[str, int]:
    """
    解析模型期望的输入维度。

    优先读模型显式属性；如果模型没有声明，就回退到特征列数量。
    """
    dims = {}

    if hasattr(model, "bull_dim"):
        dims["bull"] = getattr(model, "bull_dim", None)
        dims["bear"] = getattr(model, "bear_dim", None)
        dims["friction"] = getattr(model, "friction_dim", getattr(model, "heat_dim", None))
        dims["macro"] = getattr(model, "macro_dim", None)

    if not all(dims.get(group) for group in FEATURE_GROUP_NAMES):
        dims = {
            "bull": len(feature_groups.get("bull", [])) or 1,
            "bear": len(feature_groups.get("bear", [])) or 1,
            "friction": len(feature_groups.get("friction", [])) or 1,
            "macro": len(feature_groups.get("macro", [])) or 1,
        }

    return {group: int(dims[group]) for group in FEATURE_GROUP_NAMES}


def prepare_feature_arrays(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    expected_dims: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """把 DataFrame 转成模型输入矩阵，并自动做补零或截断。"""
    valid_groups = filter_feature_groups(df, feature_groups)
    arrays: Dict[str, np.ndarray] = {}

    for group in FEATURE_GROUP_NAMES:
        cols = valid_groups[group]
        expected_dim = max(1, expected_dims.get(group, len(cols) or 1))

        if cols:
            data = df[cols].to_numpy(dtype=np.float32, copy=True)
        else:
            data = np.zeros((len(df), expected_dim), dtype=np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        actual_dim = data.shape[1]
        if actual_dim < expected_dim:
            padding = np.zeros((len(df), expected_dim - actual_dim), dtype=np.float32)
            data = np.concatenate([data, padding], axis=1)
        elif actual_dim > expected_dim:
            data = data[:, :expected_dim]

        arrays[group] = data

    return arrays, valid_groups


def slice_feature_tensors(
    feature_arrays: Dict[str, np.ndarray],
    start: int,
    end: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 numpy 切片转换成模型前向所需的四组 tensor。"""
    return (
        torch.from_numpy(feature_arrays["bull"][start:end]).to(device),
        torch.from_numpy(feature_arrays["bear"][start:end]).to(device),
        torch.from_numpy(feature_arrays["friction"][start:end]).to(device),
        torch.from_numpy(feature_arrays["macro"][start:end]).to(device),
    )


def concat_tensor_components(*component_dicts: Dict) -> Dict[str, torch.Tensor]:
    """拼接多个 batch 前向结果中的张量字段。"""
    merged: Dict[str, List[torch.Tensor]] = {}

    for component_dict in component_dicts:
        for key, value in component_dict.items():
            if torch.is_tensor(value) and value.dim() >= 1:
                merged.setdefault(key, []).append(value)

    return {key: torch.cat(values, dim=0) for key, values in merged.items() if values}


def describe_group_availability(
    feature_groups: Dict[str, List[str]],
    valid_groups: Dict[str, List[str]],
) -> Dict[str, str]:
    """返回特征配置列数与真实可用列数的对照摘要。"""
    summary = {}
    for group in FEATURE_GROUP_NAMES:
        configured = len(feature_groups.get(group, []))
        available = len(valid_groups.get(group, []))
        summary[group] = f"{available}/{configured}"
    return summary
