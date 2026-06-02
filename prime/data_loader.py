"""
================================================================================
数据加载与预处理模块 (修复版 v2)
================================================================================
修复内容：
1. 添加宏观Embedding预计算功能（解决维度不匹配问题）
2. 确保数据严格按时间排序（防止数据泄露）
3. 增强数据验证
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass, field
import hashlib
import json
import time
import warnings

warnings.filterwarnings('ignore')

from market_profiles import get_market_preset, harmonize_panel_schema, has_required_date_coverage

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def fast_zscore_by_group(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
        """Numba 加速的按组 z-score。"""
        result = np.empty_like(values)
        result[:] = np.nan

        for g in prange(n_groups):
            mask = group_ids == g
            group_vals = values[mask]
            if len(group_vals) == 0:
                continue

            valid_mask = ~np.isnan(group_vals)
            valid_vals = group_vals[valid_mask]
            if len(valid_vals) == 0:
                result[mask] = 0.0
                continue

            mean = np.mean(valid_vals)
            std = np.std(valid_vals)
            if std > 1e-8:
                result[mask] = (group_vals - mean) / std
            else:
                result[mask] = group_vals - mean

        return np.nan_to_num(result, nan=0.0)
else:
    def fast_zscore_by_group(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
        """无 numba 时的回退实现。"""
        result = np.empty_like(values)
        result[:] = np.nan

        for g in range(n_groups):
            mask = group_ids == g
            group_vals = values[mask]
            if len(group_vals) == 0:
                continue

            valid_mask = ~np.isnan(group_vals)
            valid_vals = group_vals[valid_mask]
            if len(valid_vals) == 0:
                result[mask] = 0.0
                continue

            mean = np.mean(valid_vals)
            std = np.std(valid_vals)
            if std > 1e-8:
                result[mask] = (group_vals - mean) / std
            else:
                result[mask] = group_vals - mean

        return np.nan_to_num(result, nan=0.0)


class DataCache:
    """简单的数据缓存管理。"""

    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_key(self, prefix: str, df: pd.DataFrame, config_str: str) -> str:
        sample_hash = hashlib.md5(
            pd.util.hash_pandas_object(df.head(2000), index=True).values.tobytes()
        ).hexdigest()[:10]
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:10]
        return f"{prefix}_{sample_hash}_{config_hash}"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        cache_path = self.cache_dir / f'{key}.pkl'
        if cache_path.exists():
            print(f"  ✓ 从缓存加载: {cache_path}")
            return pd.read_pickle(cache_path)
        return None

    def set(self, key: str, df: pd.DataFrame):
        cache_path = self.cache_dir / f'{key}.pkl'
        df.to_pickle(cache_path)
        print(f"  ✓ 保存缓存: {cache_path}")


class ProcessedArtifactStore:
    """完整预处理资产缓存：full_df + macro_embeddings + metadata。"""

    CACHE_VERSION = 'processed_v2'

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_key(
        self,
        config,
        source_path: Path,
        fast: bool,
        use_gpu: bool,
        allow_source_fallback: bool,
    ) -> str:
        feature_groups = config.get_feature_groups()
        source_ref = self._source_fingerprint(source_path)
        payload = {
            'version': self.CACHE_VERSION,
            'mode': config.mode,
            'market_profile': getattr(config.data, 'market_profile', 'unknown'),
            'source': source_ref,
            'fast': fast,
            'use_gpu': use_gpu,
            'allow_source_fallback': allow_source_fallback,
            'forward_days': config.data.forward_days,
            'crash_threshold': config.data.active_crash_threshold,
            'drawdown_threshold': config.data.active_drawdown_threshold,
            'crash_relative_to_benchmark': getattr(config.data, 'crash_relative_to_benchmark', False),
            'macro_lag': config.data.macro_lag,
            'mad_threshold': config.data.mad_threshold,
            'macro_seq_len': config.model.macro_seq_len,
            'train_start': config.data.train_start,
            'train_end': config.data.train_end,
            'valid_start': config.data.valid_start,
            'valid_end': config.data.valid_end,
            'test_start': config.data.test_start,
            'test_end': config.data.test_end,
            'feature_groups': feature_groups,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:16]

    def load(self, key: str) -> Optional[Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
        meta_path = self.cache_dir / f'{key}.meta.json'
        frame_parquet_path = self.cache_dir / f'{key}.frame.parquet'
        frame_pickle_path = self.cache_dir / f'{key}.frame.pkl'
        embeddings_path = self.cache_dir / f'{key}.embeddings.pkl'

        if not meta_path.exists() or not embeddings_path.exists():
            return None

        if frame_parquet_path.exists():
            df = pd.read_parquet(frame_parquet_path)
        elif frame_pickle_path.exists():
            df = pd.read_pickle(frame_pickle_path)
        else:
            return None

        embeddings = pd.read_pickle(embeddings_path)
        print(f"✓ 从预处理资产缓存加载: {self.cache_dir / key}")
        return df, embeddings

    def save(self, key: str, df: pd.DataFrame, macro_embeddings: Dict[str, np.ndarray], metadata: Dict[str, object]):
        meta_path = self.cache_dir / f'{key}.meta.json'
        frame_parquet_path = self.cache_dir / f'{key}.frame.parquet'
        frame_pickle_path = self.cache_dir / f'{key}.frame.pkl'
        embeddings_path = self.cache_dir / f'{key}.embeddings.pkl'

        saved_frame_path = frame_parquet_path
        try:
            df.to_parquet(frame_parquet_path, index=False)
            if frame_pickle_path.exists():
                frame_pickle_path.unlink()
        except Exception:
            df.to_pickle(frame_pickle_path)
            saved_frame_path = frame_pickle_path
            if frame_parquet_path.exists():
                frame_parquet_path.unlink()

        pd.to_pickle(macro_embeddings, embeddings_path)
        meta_payload = dict(metadata)
        meta_payload.update({
            'cache_version': self.CACHE_VERSION,
            'frame_path': saved_frame_path.name,
            'embeddings_path': embeddings_path.name,
            'rows': int(len(df)),
            'columns': list(df.columns),
            'macro_embedding_dim': int(len(next(iter(macro_embeddings.values())))) if macro_embeddings else 0,
        })
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        print(f"✓ 保存预处理资产缓存: {saved_frame_path}")

    def _source_fingerprint(self, source_path: Path) -> Dict[str, object]:
        source_path = Path(source_path)
        if source_path.exists() and source_path.is_file():
            stat = source_path.stat()
            return {
                'path': str(source_path),
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
            }
        return {'path': str(source_path), 'exists': False}


@dataclass
class DataSplit:
    """数据集划分结果"""
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    macro_embeddings: Optional[Dict[str, np.ndarray]] = None
    
    def __repr__(self):
        return f"DataSplit(train={len(self.train)}, valid={len(self.valid)}, test={len(self.test)})"


class ChinaStockDataLoader:
    """
    中国市场数据源加载器。

    优先使用已经存在的 panel 文件；若 panel 不存在，再从分散目录拼接。
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.dirs = {
            'stock': self.data_root / 'stock',
            'valuation': self.data_root / 'valuation',
            'financial': self.data_root / 'financial',
            'capital_flow': self.data_root / 'capital_flow',
            'chip': self.data_root / 'chip',
            'margin': self.data_root / 'margin',
            'northbound': self.data_root / 'northbound',
            'macro': self.data_root / 'macro',
            'index': self.data_root / 'index',
            'industry': self.data_root / 'industry',
            'market': self.data_root / 'market',
            'meta': self.data_root / 'meta',
            'panel': self.data_root / 'panel',
        }

    def load_all(
        self,
        start_date: str = '2008-01-01',
        end_date: str = '2024-12-31',
        use_cache: bool = True,
    ) -> pd.DataFrame:
        panel_file = self._find_panel_file()
        if panel_file is not None:
            print(f"\n>>> 找到预处理的面板数据: {panel_file}")
            df = self._load_file(panel_file)
        else:
            print("\n>>> 未找到预处理面板数据，从各目录加载...")
            df = self._load_stock_data()
            if df is None or df.empty:
                raise ValueError("无法加载股票数据，请检查 stock/ 目录")

            print(f"    基础数据: {df.shape}")
            df = self._merge_valuation(df)
            df = self._merge_capital_flow(df)
            df = self._merge_northbound(df)
            df = self._merge_margin(df)
            df = self._merge_macro(df)
            df = self._merge_index(df)

        df = self._standardize_columns(df)
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
        if {'ts_code', 'trade_date'}.issubset(df.columns):
            df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

        print(f"\n>>> 最终数据: {df.shape}")
        if 'trade_date' in df.columns:
            print(f"    日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        if 'ts_code' in df.columns:
            print(f"    股票数量: {df['ts_code'].nunique()}")
        print(f"    特征列: {len(df.columns)}")
        return df

    def _find_panel_file(self) -> Optional[Path]:
        panel_dir = self.dirs['panel']
        if not panel_dir.exists():
            return None

        candidates = []
        for pattern in ('*.parquet', '*.csv', '*.h5', '*.feather', '*.pkl'):
            candidates.extend(panel_dir.glob(pattern))

        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_size)

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif suffix == '.csv':
            df = pd.read_csv(file_path, low_memory=False)
        elif suffix == '.h5':
            df = pd.read_hdf(file_path)
        elif suffix == '.feather':
            df = pd.read_feather(file_path)
        elif suffix == '.pkl':
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f'不支持的文件格式: {suffix}')
        return self._standardize_columns(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_map = {
            'code': 'ts_code',
            'symbol': 'ts_code',
            'stock_code': 'ts_code',
            'date': 'trade_date',
            'datetime': 'trade_date',
            'vol': 'volume',
            'amount': 'turnover',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def _load_stock_data(self) -> Optional[pd.DataFrame]:
        stock_dir = self.dirs['stock']
        if not stock_dir.exists():
            return None

        dfs = []
        for pattern in ('*.parquet', '*.csv', '*.feather'):
            files = list(stock_dir.glob(pattern))
            if not files:
                continue
            print(f"    从 stock/ 加载 {len(files)} 个 {pattern} 文件...")
            for file_path in files:
                try:
                    dfs.append(self._load_file(file_path))
                except Exception as exc:
                    print(f"      警告: 加载 {file_path.name} 失败: {exc}")
            break

        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def _load_dir_data(self, dir_path: Path, max_files: int = 10) -> Optional[pd.DataFrame]:
        dfs = []
        for pattern in ('*.parquet', '*.csv', '*.feather'):
            files = list(dir_path.glob(pattern))
            if not files:
                continue
            for file_path in files[:max_files]:
                try:
                    dfs.append(self._load_file(file_path))
                except Exception:
                    continue
            break

        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def _merge_on_keys(
        self,
        base_df: pd.DataFrame,
        source_df: Optional[pd.DataFrame],
        merge_keys: List[str],
        keep_cols: List[str],
        desc: str,
    ) -> pd.DataFrame:
        if source_df is None:
            return base_df

        keep_cols = [col for col in keep_cols if col in source_df.columns]
        if len(keep_cols) <= len(merge_keys):
            return base_df

        source_df = source_df[keep_cols].drop_duplicates(merge_keys)
        merged = base_df.merge(source_df, on=merge_keys, how='left')
        print(f"    合并{desc}: +{len(keep_cols) - len(merge_keys)} 列")
        return merged

    def _merge_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        val_df = self._load_dir_data(self.dirs['valuation'])
        cols = ['ts_code', 'trade_date', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'total_mv', 'circ_mv']
        return self._merge_on_keys(df, val_df, ['ts_code', 'trade_date'], cols, '估值数据')

    def _merge_capital_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        cf_df = self._load_dir_data(self.dirs['capital_flow'])
        if cf_df is None:
            return df
        cols = ['ts_code', 'trade_date'] + [
            col for col in cf_df.columns
            if any(key in col.lower() for key in ['buy', 'sell', 'net', 'flow', 'amount'])
        ]
        return self._merge_on_keys(df, cf_df, ['ts_code', 'trade_date'], cols, '资金流向')

    def _merge_northbound(self, df: pd.DataFrame) -> pd.DataFrame:
        nb_df = self._load_dir_data(self.dirs['northbound'])
        if nb_df is None:
            return df
        if 'ts_code' in nb_df.columns:
            cols = ['ts_code', 'trade_date'] + [
                col for col in nb_df.columns
                if any(key in col.lower() for key in ['north', 'hk', 'hold', 'ratio'])
            ]
            return self._merge_on_keys(df, nb_df, ['ts_code', 'trade_date'], cols, '北向资金')

        cols = ['trade_date'] + [
            col for col in nb_df.columns
            if any(key in col.lower() for key in ['north', 'hk', 'hold', 'ratio'])
        ]
        return self._merge_on_keys(df, nb_df, ['trade_date'], cols, '北向资金')

    def _merge_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        margin_df = self._load_dir_data(self.dirs['margin'])
        if margin_df is None:
            return df
        cols = ['ts_code', 'trade_date'] + [
            col for col in margin_df.columns
            if any(key in col.lower() for key in ['rzye', 'rqye', 'rzmre', 'rzrqye', 'margin'])
        ]
        return self._merge_on_keys(df, margin_df, ['ts_code', 'trade_date'], cols, '融资融券')

    def _merge_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        macro_df = self._load_dir_data(self.dirs['macro'])
        if macro_df is None:
            return df
        numeric_cols = [
            col for col in macro_df.columns
            if col != 'trade_date' and pd.api.types.is_numeric_dtype(macro_df[col])
        ]
        cols = ['trade_date'] + numeric_cols
        return self._merge_on_keys(df, macro_df, ['trade_date'], cols, '宏观数据')

    def _merge_index(self, df: pd.DataFrame) -> pd.DataFrame:
        index_df = self._load_dir_data(self.dirs['index'])
        if index_df is None or 'close' not in index_df.columns or 'trade_date' not in index_df.columns:
            return df

        index_df = index_df.sort_values('trade_date').copy()
        index_df['market_ret'] = index_df['close'].pct_change()
        index_df['market_ret_5d'] = index_df['close'].pct_change(5)
        index_df['market_ret_20d'] = index_df['close'].pct_change(20)
        cols = ['trade_date', 'market_ret', 'market_ret_5d', 'market_ret_20d']
        return self._merge_on_keys(df, index_df, ['trade_date'], cols, '指数数据')


def create_panel_data(
    data_root: str,
    output_file: Optional[str] = None,
    start_date: str = '2008-01-01',
    end_date: str = '2024-12-31',
) -> pd.DataFrame:
    """便捷函数：从中国市场目录创建 panel 数据。"""
    loader = ChinaStockDataLoader(data_root)
    df = loader.load_all(start_date=start_date, end_date=end_date)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            if output_path.suffix != '.parquet':
                output_path = output_path.with_suffix('.parquet')
            df.to_parquet(output_path, index=False)
        print(f"\n>>> 数据已保存至: {output_path}")

    return df


class MacroEmbeddingPrecomputer:
    """
    宏观Embedding预计算器
    
    解决问题：避免在Dataset中实时构建宏观序列，提高效率
    方案：预先计算每个日期的宏观状态向量，存为字典
    """
    
    def __init__(
        self,
        macro_features: List[str],
        seq_len: int = 12,
        method: str = 'rolling_mean'
    ):
        self.macro_features = macro_features
        self.seq_len = seq_len
        self.method = method
        self.mean_ = None
        self.std_ = None
        self.embeddings = {}
    
    def fit_transform(self, df: pd.DataFrame, date_col: str = 'trade_date') -> Dict[str, np.ndarray]:
        """计算每个日期的宏观状态Embedding"""
        df = df.sort_values(date_col).copy()
        existing_cols = [c for c in self.macro_features if c in df.columns]
        
        if not existing_cols:
            print("  警告: 没有找到宏观特征，使用零向量")
            dates = df[date_col].unique()
            self.embeddings = {str(d): np.zeros(len(self.macro_features) * 3) for d in dates}
            return self.embeddings
        
        # 按日期聚合
        macro_daily = df.groupby(date_col)[existing_cols].mean().reset_index()
        macro_daily = macro_daily.sort_values(date_col)
        macro_values = macro_daily[existing_cols].values
        
        # 标准化
        self.mean_ = np.nanmean(macro_values, axis=0, keepdims=True)
        self.std_ = np.nanstd(macro_values, axis=0, keepdims=True) + 1e-8
        macro_normalized = (macro_values - self.mean_) / self.std_
        macro_normalized = np.nan_to_num(macro_normalized, nan=0.0)
        
        # 计算Embedding
        for i, row in enumerate(macro_daily.itertuples()):
            date = row[1]
            date_str = str(date)[:10]
            
            if i < self.seq_len:
                window = macro_normalized[:i+1]
            else:
                window = macro_normalized[i-self.seq_len+1:i+1]
            
            embedding = self._compute_rolling_embedding(window, existing_cols)
            self.embeddings[date_str] = embedding.astype(np.float32)
        
        print(f"  ✓ 预计算宏观Embedding: {len(self.embeddings)} 个日期, 维度={len(embedding)}")
        return self.embeddings
    
    def _compute_rolling_embedding(self, window: np.ndarray, cols: List[str]) -> np.ndarray:
        """计算滚动统计Embedding：均值 + 趋势 + 波动"""
        if len(window) == 0:
            return np.zeros(len(cols) * 3)
        
        mean_vec = np.mean(window, axis=0)
        mid = len(window) // 2
        trend_vec = np.mean(window[mid:], axis=0) - np.mean(window[:max(1,mid)], axis=0) if mid > 0 else np.zeros_like(mean_vec)
        vol_vec = np.std(window, axis=0) if len(window) > 1 else np.zeros_like(mean_vec)
        
        return np.concatenate([mean_vec, trend_vec, vol_vec])
    
    def get_embedding(self, date) -> np.ndarray:
        """获取指定日期的Embedding"""
        date_str = str(date)[:10]
        if date_str in self.embeddings:
            return self.embeddings[date_str]
        
        all_dates = sorted(self.embeddings.keys())
        for d in reversed(all_dates):
            if d <= date_str:
                return self.embeddings[d]
        return self.embeddings[all_dates[0]] if all_dates else np.zeros(len(self.macro_features) * 3)
    
    def get_embedding_dim(self) -> int:
        if self.embeddings:
            return len(next(iter(self.embeddings.values())))
        return len(self.macro_features) * 3


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.data
        
    def load_panel_data(self, allow_source_fallback: bool = False) -> pd.DataFrame:
        """加载面板数据"""
        panel_path = self._resolve_panel_path()
        if not panel_path.exists():
            if allow_source_fallback:
                print(f"面板数据不存在，尝试从原始目录构建: {panel_path}")
                try:
                    if getattr(self.data_config, 'mode', 'stock') != 'stock':
                        raise FileNotFoundError("ETF 模式未提供原始目录拼接逻辑")

                    china_loader = ChinaStockDataLoader(str(self.data_config.active_data_root))
                    return china_loader.load_all(
                        start_date=self.data_config.train_start,
                        end_date=self.data_config.test_end,
                    )
                except Exception as exc:
                    raise FileNotFoundError(
                        f"无法从原始目录构建面板数据: {self.data_config.active_data_root}"
                    ) from exc
            raise FileNotFoundError(f"面板数据不存在: {panel_path}")
        
        if panel_path.suffix == '.parquet':
            df = pd.read_parquet(panel_path)
        elif panel_path.suffix == '.h5':
            df = pd.read_hdf(panel_path, key='data')
        else:
            df = pd.read_csv(panel_path, parse_dates=['trade_date'], low_memory=False)

        df = harmonize_panel_schema(df, self.config)
        if 'trade_date' in df.columns:
            lower_bound = pd.to_datetime(self.data_config.train_start) - pd.Timedelta(days=180)
            upper_bound = pd.to_datetime(self.data_config.test_end) + pd.Timedelta(days=30)
            before_shape = df.shape
            df = df[(df['trade_date'] >= lower_bound) & (df['trade_date'] <= upper_bound)].copy()
            if df.shape != before_shape:
                print(f"  - 按训练窗口裁剪原始面板: {before_shape} -> {df.shape}")
        print(f"✓ 加载面板数据: {panel_path} -> {df.shape}")
        return df

    def _resolve_panel_path(self) -> Path:
        """解析实际可用的面板文件路径。"""
        mode = getattr(self.data_config, 'mode', 'stock')

        if mode == 'etf':
            primary_path = getattr(self.data_config, 'etf_panel_path', self.data_config.panel_path)
            panel_dir = Path(getattr(self.data_config, 'etf_data_root', self.data_config.data_root)) / 'panel'
        else:
            primary_path = getattr(self.data_config, 'panel_path')
            panel_dir = Path(getattr(self.data_config, 'data_root'))
            if panel_dir.name != 'panel':
                panel_dir = panel_dir / 'panel'

        candidates = [
            Path(primary_path),
            panel_dir / 'panel_data_complete.parquet',
            panel_dir / 'panel_data_complete.csv',
            panel_dir / 'panel_data.parquet',
            panel_dir / 'panel_data.csv',
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return Path(primary_path)


def _bind_runtime_panel_path(config, source_path: Path):
    """将运行时生成的 panel 显式绑定到当前配置，避免覆盖或误读真实面板文件。"""
    source_path = Path(source_path)
    if getattr(config.data, 'mode', 'stock') == 'etf':
        config.data.etf_panel_path = source_path
    else:
        config.data.panel_path = source_path


class DataPreprocessor:
    """数据预处理器 (修复版)"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.data
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整预处理流程"""
        print(">>> 数据预处理...")
        
        # 0. 【关键】确保按时间排序
        df = self._ensure_time_sorted(df)
        
        # 1. 宏观数据滞后
        df = self._apply_macro_lag(df)
        
        # 2. 缺失值处理
        df = self._handle_missing(df)
        
        # 3. 异常值处理
        df = self._handle_outliers(df)
        
        # 4. 横截面标准化
        df = self._cross_sectional_zscore(df)
        
        # 5. 生成标签
        df = self._generate_labels(df)
        
        print(f"✓ 预处理完成: {df.shape}")
        return df
    
    def _ensure_time_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数据严格按时间排序"""
        if 'trade_date' not in df.columns:
            return df
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        if 'ts_code' in df.columns:
            df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        else:
            df = df.sort_values('trade_date').reset_index(drop=True)
        
        print(f"  - 数据已按时间排序: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        return df
    
    def _apply_macro_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """宏观数据滞后处理"""
        macro_cols = resolve_raw_macro_features(self.config)
        lag_months = self.data_config.macro_lag
        
        existing_macro = [c for c in macro_cols if c in df.columns]
        
        if existing_macro and 'trade_date' in df.columns:
            lag_steps = max(1, int(lag_months * 21))
            macro_daily = (
                df.groupby('trade_date', as_index=False)[existing_macro]
                .mean()
                .sort_values('trade_date')
            )
            macro_lagged = macro_daily.copy()
            macro_lagged[existing_macro] = macro_lagged[existing_macro].shift(lag_steps)
            macro_lagged = macro_lagged.rename(
                columns={col: f'{col}_lagged' for col in existing_macro}
            )

            df = df.merge(macro_lagged, on='trade_date', how='left')
            for col in existing_macro:
                lagged_col = f'{col}_lagged'
                df[col] = df[lagged_col].combine_first(df[col])
            df = df.drop(columns=[f'{col}_lagged' for col in existing_macro])
        
        print(f"  - 宏观数据滞后 {lag_months} 个月")
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """缺失值处理"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        excluded_cols = {
            'fwd_return',
            'fwd_return_5d',
            'fwd_drawdown',
            'max_dd_10d',
            'crash_label',
        }
        process_cols = [col for col in numeric_cols if col not in excluded_cols]
        
        if 'ts_code' in df.columns:
            df[process_cols] = df.groupby('ts_code')[process_cols].ffill()
        else:
            df[process_cols] = df[process_cols].ffill()
        
        df[process_cols] = df[process_cols].fillna(0)
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """异常值处理 - MAD方法"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        threshold = self.data_config.mad_threshold
        
        exclude_cols = ['trade_date', 'ts_code', 'fwd_return', 'fwd_return_5d', 'fwd_drawdown', 'max_dd_10d', 'crash_label']
        process_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        for col in process_cols:
            if col in df.columns:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                if mad > 0:
                    lower = median - threshold * mad
                    upper = median + threshold * mad
                    df[col] = df[col].clip(lower, upper)
        
        return df
    
    def _cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """横截面Z-Score标准化"""
        if 'trade_date' not in df.columns:
            return df
        
        all_features = (self.config.feature.bull_features + 
                       self.config.feature.bear_features + 
                       self.config.feature.friction_features)
        
        existing_features = [f for f in all_features if f in df.columns]
        
        def zscore(x):
            mean, std = x.mean(), x.std()
            return (x - mean) / std if std > 0 else x - mean
        
        for col in existing_features:
            df[f'{col}_zscore'] = df.groupby('trade_date')[col].transform(zscore)
        
        print(f"  - 横截面Z-Score: {len(existing_features)} 个特征")
        return df
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成预测标签"""
        if 'ts_code' not in df.columns:
            return df

        df = df.sort_values(['ts_code', 'trade_date']).copy()
        forward_days = self.data_config.forward_days

        if 'fwd_return' not in df.columns:
            existing_return_col = f'fwd_return_{forward_days}d'
            if existing_return_col in df.columns:
                df['fwd_return'] = df[existing_return_col]
            elif 'close' in df.columns:
                df['fwd_return'] = df.groupby('ts_code')['close'].transform(
                    lambda x: x.shift(-forward_days) / x - 1
                )
            else:
                return df

        if 'fwd_drawdown' not in df.columns:
            if 'max_dd_10d' in df.columns:
                df['fwd_drawdown'] = df['max_dd_10d']
            elif 'close' in df.columns:
                def calc_max_drawdown(prices, window=10):
                    result = np.zeros(len(prices))
                    prices = prices.values if hasattr(prices, 'values') else prices
                    for i in range(len(prices) - window):
                        future = prices[i:i+window+1]
                        peak = future[0]
                        max_dd = 0
                        for p in future[1:]:
                            if p > peak:
                                peak = p
                            dd = (peak - p) / peak if peak > 0 else 0
                            max_dd = max(max_dd, dd)
                        result[i] = max_dd
                    return result

                df['fwd_drawdown'] = df.groupby('ts_code')['close'].transform(calc_max_drawdown)

        crash_threshold = self.data_config.active_crash_threshold
        drawdown_threshold = self.data_config.active_drawdown_threshold

        if 'daily_return' not in df.columns and 'close' in df.columns:
            df['daily_return'] = df.groupby('ts_code')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )

        benchmark_5d = None
        if getattr(self.data_config, 'crash_relative_to_benchmark', False):
            if 'market_ret_5d' in df.columns:
                benchmark_5d = df['market_ret_5d']
            elif 'benchmark_return_5d' in df.columns:
                benchmark_5d = df['benchmark_return_5d']
            elif 'daily_return' in df.columns:
                benchmark_5d = (
                    df.groupby('trade_date')['daily_return']
                    .transform('mean')
                    .rolling(5, min_periods=1)
                    .sum()
                )

        crash_return = df['fwd_return'] if benchmark_5d is None else (df['fwd_return'] - benchmark_5d)
        df['crash_label'] = ((crash_return < crash_threshold) |
                            (df['fwd_drawdown'] > drawdown_threshold)).astype(int)

        print(f"  - 标签生成: 崩盘比例={df['crash_label'].mean():.2%}")
        return df


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.data
    
    def split(self, df: pd.DataFrame) -> DataSplit:
        """按时间划分数据集"""
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        data_start = df['trade_date'].min()
        data_end = df['trade_date'].max()
        
        train_start = pd.to_datetime(self.data_config.train_start)
        train_end = pd.to_datetime(self.data_config.train_end)
        valid_start = pd.to_datetime(self.data_config.valid_start)
        valid_end = pd.to_datetime(self.data_config.valid_end)
        test_start = pd.to_datetime(self.data_config.test_start)
        test_end = pd.to_datetime(self.data_config.test_end)
        
        train = df[(df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)].copy()
        valid = df[(df['trade_date'] >= valid_start) & (df['trade_date'] <= valid_end)].copy()
        test = df[(df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)].copy()
        
        if train_start < data_start or test_end > data_end:
            print(
                f"  - 数据实际范围: {data_start.date()} ~ {data_end.date()}，"
                f"配置区间外的部分会自动截断"
            )

        print(f">>> 数据集划分: train={len(train)}, valid={len(valid)}, test={len(test)}")
        return DataSplit(train=train, valid=valid, test=test)


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, config):
        self.config = config
    
    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取所有特征（确保按时间排序，防止数据泄露）"""
        print(">>> 特征提取...")
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        df = self._extract_bull_features(df)
        df = self._extract_bear_features(df)
        df = self._extract_friction_features(df)
        
        print(f"✓ 特征提取完成: {df.shape}")
        return df
    
    def _extract_bull_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取动力特征
        
        【关键修复】所有特征使用T-1数据，避免数据泄露
        """
        # 先创建滞后一天的收盘价
        df['_close_lag1'] = df.groupby('ts_code')['close'].shift(1)
        
        for period in [5, 10, 20]:
            col_name = f'momentum_{period}d'
            if col_name not in df.columns:
                # 正确：使用T-1的收盘价
                df[col_name] = df.groupby('ts_code')['_close_lag1'].transform(lambda x: x.pct_change(period))
        
        for period in [6, 14]:
            col_name = f'rsi_{period}'
            if col_name not in df.columns:
                # 正确：使用T-1的收盘价
                df[col_name] = df.groupby('ts_code')['_close_lag1'].transform(lambda x: self._calc_rsi(x, period))
        
        if 'price_vs_ma20' not in df.columns:
            ma20_lag1 = df.groupby('ts_code')['_close_lag1'].transform(lambda x: x.rolling(20).mean())
            df['price_vs_ma20'] = (df['_close_lag1'] - ma20_lag1) / (ma20_lag1 + 1e-8)
        
        if 'volume_ratio' not in df.columns and 'volume' in df.columns:
            df['_vol_lag1'] = df.groupby('ts_code')['volume'].shift(1)
            vol_ma = df.groupby('ts_code')['_vol_lag1'].transform(lambda x: x.rolling(20).mean())
            df['volume_ratio'] = df['_vol_lag1'] / (vol_ma + 1e-8)
            df.drop(columns=['_vol_lag1'], inplace=True)
        
        df.drop(columns=['_close_lag1'], inplace=True)
        
        return df
    
    def _extract_bear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取阻力特征
        
        【关键修复】所有特征使用T-1数据
        """
        df['_close_lag1'] = df.groupby('ts_code')['close'].shift(1)
        
        for period in [20, 60]:
            col_name = f'bias_{period}d'
            if col_name not in df.columns:
                ma_lag1 = df.groupby('ts_code')['_close_lag1'].transform(lambda x: x.rolling(period).mean())
                df[col_name] = (df['_close_lag1'] - ma_lag1) / (ma_lag1 + 1e-8) * 100
        
        if 'pe_ttm' in df.columns and 'pe_rank' not in df.columns:
            # PE数据本身已是历史数据，但也做滞后处理
            df['_pe_lag1'] = df.groupby('ts_code')['pe_ttm'].shift(1)
            df['pe_rank'] = df.groupby('ts_code')['_pe_lag1'].transform(
                lambda x: x.rolling(252, min_periods=60).apply(
                    lambda y: (y.iloc[-1] > y[:-1]).mean() if len(y) > 1 else 0.5
                )
            )
            df.drop(columns=['_pe_lag1'], inplace=True)
        
        df.drop(columns=['_close_lag1'], inplace=True)
        
        return df
    
    def _extract_friction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取摩擦特征
        
        【关键修复】所有特征使用T-1数据
        """
        df['_close_lag1'] = df.groupby('ts_code')['close'].shift(1)
        
        for period in [20, 60]:
            col_name = f'volatility_{period}d'
            if col_name not in df.columns:
                df[col_name] = df.groupby('ts_code')['_close_lag1'].transform(
                    lambda x: x.pct_change().rolling(period).std() * np.sqrt(252)
                )
        
        if 'high' in df.columns and 'low' in df.columns:
            if 'amplitude' not in df.columns:
                df['_high_lag1'] = df.groupby('ts_code')['high'].shift(1)
                df['_low_lag1'] = df.groupby('ts_code')['low'].shift(1)
                df['amplitude'] = (df['_high_lag1'] - df['_low_lag1']) / (df['_close_lag1'] + 1e-8) * 100
                df.drop(columns=['_high_lag1', '_low_lag1'], inplace=True)
        
        if 'turnover' in df.columns and 'turnover_20d_avg' not in df.columns:
            df['_turn_lag1'] = df.groupby('ts_code')['turnover'].shift(1)
            df['turnover_20d_avg'] = df.groupby('ts_code')['_turn_lag1'].transform(lambda x: x.rolling(20).mean())
            df.drop(columns=['_turn_lag1'], inplace=True)
        
        df.drop(columns=['_close_lag1'], inplace=True)
        
        return df
    
    @staticmethod
    def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


class FastFeatureExtractor(FeatureExtractor):
    """更偏向向量化实现的特征提取器。"""

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        print(">>> 快速特征提取...")
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        df = self._extract_features_vectorized(df)
        print(f"✓ 特征提取完成: {df.shape}, 耗时 {time.time() - start_time:.1f}s")
        return df

    def _extract_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        df['close_lag1'] = df.groupby('ts_code')['close'].shift(1)
        if 'volume' in df.columns:
            df['volume_lag1'] = df.groupby('ts_code')['volume'].shift(1)
        if 'high' in df.columns:
            df['high_lag1'] = df.groupby('ts_code')['high'].shift(1)
        if 'low' in df.columns:
            df['low_lag1'] = df.groupby('ts_code')['low'].shift(1)
        if 'turnover' in df.columns:
            df['turnover_lag1'] = df.groupby('ts_code')['turnover'].shift(1)

        for period in (5, 10, 20):
            col_name = f'momentum_{period}d'
            if col_name not in df.columns:
                df[col_name] = df.groupby('ts_code')['close_lag1'].pct_change(period)

        for period in (6, 14):
            col_name = f'rsi_{period}'
            if col_name not in df.columns:
                df[col_name] = self._calc_rsi_vectorized(df, period)

        returns_lag1 = df.groupby('ts_code')['close_lag1'].pct_change()
        for period in (20, 60):
            vol_col = f'volatility_{period}d'
            if vol_col not in df.columns:
                df[vol_col] = returns_lag1.groupby(df['ts_code']).transform(
                    lambda x: x.rolling(period, min_periods=max(5, period // 2)).std()
                ) * np.sqrt(252)

            bias_col = f'bias_{period}d'
            if bias_col not in df.columns:
                ma_lag1 = df.groupby('ts_code')['close_lag1'].transform(
                    lambda x: x.rolling(period, min_periods=max(5, period // 2)).mean()
                )
                df[bias_col] = (df['close_lag1'] - ma_lag1) / (ma_lag1 + 1e-8) * 100

        if 'price_vs_ma20' not in df.columns:
            ma20 = df.groupby('ts_code')['close_lag1'].transform(lambda x: x.rolling(20, min_periods=10).mean())
            df['price_vs_ma20'] = (df['close_lag1'] - ma20) / (ma20 + 1e-8)

        if 'volume_lag1' in df.columns and 'volume_ratio' not in df.columns:
            vol_ma = df.groupby('ts_code')['volume_lag1'].transform(lambda x: x.rolling(20, min_periods=10).mean())
            df['volume_ratio'] = df['volume_lag1'] / (vol_ma + 1e-8)

        if 'turnover_lag1' in df.columns and 'turnover_20d_avg' not in df.columns:
            df['turnover_20d_avg'] = df.groupby('ts_code')['turnover_lag1'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )

        if {'high_lag1', 'low_lag1'}.issubset(df.columns) and 'amplitude' not in df.columns:
            df['amplitude'] = (df['high_lag1'] - df['low_lag1']) / (df['close_lag1'] + 1e-8) * 100

        for temp_col in ('close_lag1', 'volume_lag1', 'high_lag1', 'low_lag1', 'turnover_lag1'):
            if temp_col in df.columns:
                df.drop(columns=[temp_col], inplace=True)

        return df

    def _calc_rsi_vectorized(self, df: pd.DataFrame, period: int) -> pd.Series:
        temp_close = df.groupby('ts_code')['close'].shift(1)
        delta = temp_close.groupby(df['ts_code']).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.groupby(df['ts_code']).transform(
            lambda x: x.rolling(period, min_periods=max(3, period // 2)).mean()
        )
        avg_loss = loss.groupby(df['ts_code']).transform(
            lambda x: x.rolling(period, min_periods=max(3, period // 2)).mean()
        )

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))


class AcceleratedDataPreprocessor(DataPreprocessor):
    """统一的可选加速预处理器。"""

    def __init__(self, config, use_gpu: bool = True, use_cache: bool = True):
        super().__init__(config)
        self.use_gpu = use_gpu
        self.use_cache = use_cache
        self.cache = DataCache() if use_cache else None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        print(">>> 数据预处理 (统一加速版)...")

        cache_key = None
        if self.use_cache and self.cache is not None:
            config_str = (
                f"{self.data_config.forward_days}_"
                f"{self.data_config.active_crash_threshold}_"
                f"{self.data_config.active_drawdown_threshold}_"
                f"{self.data_config.macro_lag}_"
                f"{len(df.columns)}_{df.shape[0]}"
            )
            cache_key = self.cache._build_key('preprocess', df, config_str)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        df = self._ensure_time_sorted(df)
        df = self._apply_macro_lag(df)
        df = self._handle_missing(df)
        df = self._handle_outliers(df)

        if self.use_gpu and HAS_CUDF:
            df = self._gpu_cross_sectional_zscore(df)
        else:
            df = self._numba_cross_sectional_zscore(df)

        df = self._generate_labels(df)

        if self.use_cache and self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, df)

        print(f"✓ 预处理完成: {df.shape}, 耗时 {time.time() - start_time:.1f}s")
        return df

    def _gpu_cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        gdf = cudf.DataFrame.from_pandas(df)
        all_features = (
            self.config.feature.bull_features +
            self.config.feature.bear_features +
            self.config.feature.friction_features
        )
        existing_features = [feature for feature in all_features if feature in gdf.columns]

        for col in existing_features:
            mean = gdf.groupby('trade_date')[col].transform('mean')
            std = gdf.groupby('trade_date')[col].transform('std').replace(0, 1)
            gdf[f'{col}_zscore'] = (gdf[col] - mean) / std

        print(f"  - GPU横截面Z-Score: {len(existing_features)} 个特征")
        return gdf.to_pandas()

    def _numba_cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        all_features = (
            self.config.feature.bull_features +
            self.config.feature.bear_features +
            self.config.feature.friction_features
        )
        existing_features = [feature for feature in all_features if feature in df.columns]
        if not existing_features:
            print("  - 横截面Z-Score: 0 个特征")
            return df

        if HAS_NUMBA and len(existing_features) > 5:
            feature_matrix = df[existing_features].to_numpy(dtype=np.float64, copy=True)
            date_ids, unique_dates = pd.factorize(df['trade_date'])
            for idx, col in enumerate(existing_features):
                df[f'{col}_zscore'] = fast_zscore_by_group(feature_matrix[:, idx], date_ids, len(unique_dates))
        else:
            df = self._cross_sectional_zscore(df)
            return df

        print(f"  - Numba横截面Z-Score: {len(existing_features)} 个特征")
        return df


def create_data_pipeline(
    config,
    fast: bool = False,
    use_gpu: bool = True,
    use_cache: bool = True,
    allow_source_fallback: bool = False,
) -> Tuple[DataSplit, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    创建统一数据流水线。

    `fast=False` 使用基础 pandas 版本；
    `fast=True` 使用向量化 / Numba / 可选 cuDF 加速版本。
    """
    loader = DataLoader(config)
    source_path = config.data.active_panel_path
    artifact_store = ProcessedArtifactStore(config.data.processed_cache_dir) if use_cache else None
    artifact_key = None

    if artifact_store is not None:
        artifact_key = artifact_store.build_key(
            config=config,
            source_path=source_path,
            fast=fast,
            use_gpu=use_gpu,
            allow_source_fallback=allow_source_fallback,
        )
        cached_artifact = artifact_store.load(artifact_key)
        if cached_artifact is not None:
            cached_df, cached_embeddings = cached_artifact
            splitter = DatasetSplitter(config)
            splits = splitter.split(cached_df)
            splits.macro_embeddings = cached_embeddings
            return splits, cached_df, cached_embeddings
    
    try:
        df = loader.load_panel_data(allow_source_fallback=allow_source_fallback or fast)
    except FileNotFoundError:
        if getattr(config.data, 'auto_mock_if_missing', True):
            source_path = ensure_mock_market_dataset(config, reason='missing_panel')
            _bind_runtime_panel_path(config, source_path)
            if artifact_store is not None:
                artifact_key = artifact_store.build_key(
                    config=config,
                    source_path=source_path,
                    fast=fast,
                    use_gpu=use_gpu,
                    allow_source_fallback=allow_source_fallback,
                )
                cached_artifact = artifact_store.load(artifact_key)
                if cached_artifact is not None:
                    cached_df, cached_embeddings = cached_artifact
                    splitter = DatasetSplitter(config)
                    splits = splitter.split(cached_df)
                    splits.macro_embeddings = cached_embeddings
                    return splits, cached_df, cached_embeddings
            df = loader.load_panel_data(allow_source_fallback=False)
        else:
            raise FileNotFoundError("面板数据不存在")

    if not has_required_date_coverage(df, config):
        if getattr(config.data, 'auto_mock_if_missing', True):
            print("  - 当前数据无法覆盖目标训练窗口，切换到 market-aware mock 数据")
            source_path = ensure_mock_market_dataset(config, reason='insufficient_date_coverage')
            _bind_runtime_panel_path(config, source_path)
            if artifact_store is not None:
                artifact_key = artifact_store.build_key(
                    config=config,
                    source_path=source_path,
                    fast=fast,
                    use_gpu=use_gpu,
                    allow_source_fallback=allow_source_fallback,
                )
                cached_artifact = artifact_store.load(artifact_key)
                if cached_artifact is not None:
                    cached_df, cached_embeddings = cached_artifact
                    splitter = DatasetSplitter(config)
                    splits = splitter.split(cached_df)
                    splits.macro_embeddings = cached_embeddings
                    return splits, cached_df, cached_embeddings
            df = loader.load_panel_data(allow_source_fallback=False)
        else:
            raise FileNotFoundError("数据时间范围不足以覆盖目标训练窗口")

    extractor_cls = FastFeatureExtractor if fast else FeatureExtractor
    preprocessor_cls = AcceleratedDataPreprocessor if fast else DataPreprocessor

    extractor = extractor_cls(config)
    df = extractor.extract_all(df)

    if fast:
        preprocessor = preprocessor_cls(config, use_gpu=use_gpu, use_cache=use_cache)
    else:
        preprocessor = preprocessor_cls(config)
    df = preprocessor.preprocess(df)
    
    print(">>> 预计算宏观Embedding...")
    macro_precomputer = MacroEmbeddingPrecomputer(
        macro_features=resolve_raw_macro_features(config),
        seq_len=config.model.macro_seq_len,
        method='rolling_mean'
    )
    macro_embeddings = macro_precomputer.fit_transform(df)
    
    df = _add_macro_embedding_to_df(df, macro_embeddings, macro_precomputer.get_embedding_dim())
    
    splitter = DatasetSplitter(config)
    splits = splitter.split(df)
    splits.macro_embeddings = macro_embeddings

    if artifact_store is not None and artifact_key is not None:
        artifact_store.save(
            artifact_key,
            df,
            macro_embeddings,
            metadata={
                'source_path': str(source_path),
                'fast': fast,
                'use_gpu': use_gpu,
                'allow_source_fallback': allow_source_fallback,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
        )
    
    return splits, df, macro_embeddings


def create_fast_data_pipeline(
    config,
    use_gpu: bool = True,
    use_cache: bool = True,
) -> Tuple[DataSplit, pd.DataFrame, Dict[str, np.ndarray]]:
    """兼容旧接口：快速数据流水线。"""
    return create_data_pipeline(
        config,
        fast=True,
        use_gpu=use_gpu,
        use_cache=use_cache,
        allow_source_fallback=True,
    )


def _add_macro_embedding_to_df(df: pd.DataFrame, embeddings: Dict[str, np.ndarray], embed_dim: int) -> pd.DataFrame:
    """将预计算的宏观Embedding添加到DataFrame"""
    embedding_cols = [f'macro_embed_{i}' for i in range(embed_dim)]

    if not embeddings:
        for col in embedding_cols:
            df[col] = 0.0
        return df

    embedding_frame = pd.DataFrame(
        [
            {'trade_date': pd.to_datetime(date_str), **{col: embedding[i] for i, col in enumerate(embedding_cols)}}
            for date_str, embedding in embeddings.items()
        ]
    )
    df = df.merge(embedding_frame, on='trade_date', how='left')
    df[embedding_cols] = df[embedding_cols].fillna(0.0)

    print(f"  ✓ 添加宏观Embedding列: {len(embedding_cols)} 维")
    return df


def resolve_raw_macro_features(config) -> List[str]:
    """
    解析用于宏观滞后和 embedding 预计算的原始宏观列。

    训练后保存的 checkpoint 配置里，`config.feature.macro_features`
    可能已经被替换成 `macro_embed_*` 模型输入列；这里需要回退到
    stock/etf 模板中的原始宏观特征，避免重复嵌入。
    """
    macro_cols = list(getattr(config.feature, "macro_features", []))
    if macro_cols and not all(str(col).startswith("macro_embed_") for col in macro_cols):
        return macro_cols

    from config import FeatureConfig

    base_feature_cfg = FeatureConfig()
    return base_feature_cfg.get_feature_groups(getattr(config.data, "mode", "stock"))["macro"]


# =============================================================================
# 合并自 mock_market_data.py 的市场 mock 生成逻辑
# =============================================================================

def ensure_mock_market_dataset(config, reason: str = "missing_dataset") -> Path:
    """在目标市场缺少有效面板时，生成一份隔离的 mock 数据集。"""
    preset = get_market_preset(config.data.market_profile)
    mock_key = (
        f"{preset.key}_{config.mode}_"
        f"{str(config.data.train_start).replace('-', '')}_"
        f"{str(config.data.test_end).replace('-', '')}"
    )
    panel_dir = config.data.active_data_root / "panel" / "mock_generated" / mock_key
    panel_dir.mkdir(parents=True, exist_ok=True)

    base_name = "etf_panel_complete" if config.mode == "etf" else "panel_data_complete"
    csv_path = panel_dir / f"{base_name}.csv"
    parquet_path = panel_dir / f"{base_name}.parquet"
    meta_path = panel_dir / f"{base_name}.mock_meta.json"

    df = generate_market_mock_panel(config)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    metadata = {
        "market_profile": preset.key,
        "market_label": preset.label,
        "asset_type": config.mode,
        "benchmark": preset.benchmark,
        "reason": reason,
        "generated_dir": str(panel_dir),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "train_window": [config.data.train_start, config.data.train_end],
        "valid_window": [config.data.valid_start, config.data.valid_end],
        "test_window": [config.data.test_start, config.data.test_end],
        "notes": preset.notes,
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"✓ 已生成 {preset.label} / {config.mode} mock 数据: {csv_path}")
    return csv_path


def generate_market_mock_panel(config) -> pd.DataFrame:
    """根据市场 preset 生成保留风格差异的 mock 面板数据。"""
    preset = get_market_preset(config.data.market_profile)
    rng = np.random.default_rng(getattr(config, "seed", 42))

    dates = pd.bdate_range(config.data.train_start, config.data.test_end)
    n_assets = preset.mock_etf_count if config.mode == "etf" else preset.mock_equity_count

    tickers = _build_mock_tickers(preset.key, config.mode, n_assets)
    sectors = _sample_mock_sectors(rng, preset.sector_weights, n_assets)
    market_state = _build_mock_market_state(rng, preset, dates)
    sector_returns = _build_mock_sector_returns(rng, preset, sectors, dates, market_state["market_return"])
    frames = []

    for idx, ticker in enumerate(tickers):
        sector = sectors[idx]
        params = _mock_asset_params(rng, preset, config.mode)
        close = _simulate_mock_close_path(
            rng,
            preset,
            market_state["market_return"],
            sector_returns[sector],
            params["beta"],
            params["idio_scale"],
            base_price=params["base_price"],
        )
        frame = _build_mock_asset_frame(
            ticker=ticker,
            sector=sector,
            dates=dates,
            close=close,
            market_state=market_state,
            params=params,
            config=config,
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df = _compute_mock_common_features(df)

    if config.mode == "etf":
        df = _compute_mock_etf_fields(df, preset)
    else:
        df = _compute_mock_stock_fields(df)

    df = _compute_mock_labels(df, config)
    return df


def _build_mock_tickers(market_key: str, mode: str, n_assets: int) -> np.ndarray:
    prefix_map = {
        "csi500": ("CN", "ETF.CN"),
        "sp500": ("US", "ETF.US"),
        "nikkei225": ("JP", "ETF.JP"),
        "stoxx600": ("EU", "ETF.EU"),
    }
    stock_prefix, etf_prefix = prefix_map[market_key]
    prefix = etf_prefix if mode == "etf" else stock_prefix
    return np.array([f"{prefix}{i:04d}" for i in range(1, n_assets + 1)])


def _sample_mock_sectors(rng, sector_weights: Dict[str, float], n_assets: int) -> np.ndarray:
    sectors = list(sector_weights.keys())
    probs = np.array(list(sector_weights.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(sectors, size=n_assets, p=probs)


def _build_mock_market_state(rng, preset, dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    n = len(dates)
    style = preset.mock_style

    noise = rng.normal(0.0, style["daily_vol"], size=n)
    regime = np.sin(np.linspace(0, 3.8 * np.pi, n)) * style["daily_vol"] * 0.35
    market_return = np.zeros(n)
    for i in range(1, n):
        market_return[i] = 0.18 * market_return[i - 1] + regime[i] + noise[i]

    liquidity = 100 + np.cumsum(rng.normal(0, 0.3, size=n))
    market_turnover = 5e8 * style["turnover_scale"] * (1.0 + np.maximum(0.2, np.abs(market_return)) * 18)
    market_turnover = market_turnover * (1.0 + rng.normal(0, 0.08, size=n))
    market_turnover = np.maximum(market_turnover, 1e7)

    pmi = 50 + 1.8 * np.sin(np.linspace(0, 3 * np.pi, n)) + rng.normal(0, 0.35, size=n)
    cpi = 2.0 + 0.8 * np.sin(np.linspace(0, 2 * np.pi, n) + 0.3) + rng.normal(0, 0.15, size=n)
    ppi = 0.5 + 1.2 * np.sin(np.linspace(0, 2.6 * np.pi, n) - 0.4) + rng.normal(0, 0.2, size=n)

    if preset.key == "sp500":
        policy_rate = 5.1 + 0.15 * np.sin(np.linspace(0, 1.6 * np.pi, n))
    elif preset.key == "nikkei225":
        policy_rate = 0.1 + 0.03 * np.sin(np.linspace(0, 2 * np.pi, n))
    elif preset.key == "stoxx600":
        policy_rate = 3.8 + 0.12 * np.sin(np.linspace(0, 1.4 * np.pi, n))
    else:
        policy_rate = 2.2 + 0.18 * np.sin(np.linspace(0, 2 * np.pi, n))

    benchmark_20d = pd.Series(market_return).rolling(20, min_periods=5).sum().to_numpy() * 100.0
    market_volatility = pd.Series(market_return).rolling(20, min_periods=5).std().to_numpy() * np.sqrt(252) * 100.0
    market_liquidity_change = pd.Series(market_turnover).pct_change().to_numpy() * 100.0

    return {
        "market_return": market_return,
        "market_turnover": market_turnover,
        "market_volatility": np.nan_to_num(market_volatility, nan=np.nanmean(market_volatility)),
        "market_liquidity_change": np.nan_to_num(market_liquidity_change, nan=0.0),
        "cpi_yoy": cpi,
        "ppi_yoy": ppi,
        "pmi": pmi,
        "lpr_1y": policy_rate,
        "real_rate_proxy": policy_rate - cpi,
        "csi500_ret_20d": np.nan_to_num(benchmark_20d, nan=0.0),
        "liquidity_index": liquidity,
    }


def _build_mock_sector_returns(rng, preset, sectors: np.ndarray, dates: pd.DatetimeIndex, market_return: np.ndarray) -> Dict[str, np.ndarray]:
    unique_sectors = sorted(set(sectors))
    sector_returns = {}
    for sector in unique_sectors:
        shock = rng.normal(0.0, preset.mock_style["sector_vol"], size=len(dates))
        sector_returns[sector] = market_return * (0.4 + rng.uniform(0.3, 0.9)) + shock
    return sector_returns


def _mock_asset_params(rng, preset, mode: str) -> Dict[str, float]:
    return {
        "base_price": float(rng.uniform(8.0, 140.0) if mode == "stock" else rng.uniform(0.8, 6.0)),
        "beta": float(rng.normal(1.0, 0.18)),
        "idio_scale": float(abs(rng.normal(preset.mock_style["idio_scale"], preset.mock_style["idio_scale"] * 0.25))),
        "profitability": float(rng.normal(12.0, 6.0)),
        "revenue_growth": float(rng.normal(8.0, 9.0)),
        "debt_ratio": float(np.clip(rng.normal(45.0, 15.0), 5.0, 90.0)),
        "goodwill_risk": float(np.clip(rng.normal(0.08, 0.07), 0.0, 0.45)),
        "turnover_rate": float(np.clip(rng.normal(2.5, 1.1), 0.2, 12.0)),
        "share_base": float(rng.uniform(5e7, 8e8)),
        "pe_base": float(np.clip(rng.normal(18.0, 7.0), 6.0, 50.0)),
        "pb_base": float(np.clip(rng.normal(2.2, 0.9), 0.5, 8.0)),
    }


def _simulate_mock_close_path(rng, preset, market_return: np.ndarray, sector_return: np.ndarray, beta: float, idio_scale: float, base_price: float) -> np.ndarray:
    n = len(market_return)
    returns = beta * market_return + 0.55 * sector_return + rng.normal(0.0, idio_scale, size=n)
    close = np.empty(n, dtype=float)
    close[0] = base_price
    for i in range(1, n):
        close[i] = max(0.3, close[i - 1] * (1.0 + returns[i]))
    return close


def _build_mock_asset_frame(ticker: str, sector: str, dates: pd.DatetimeIndex, close: np.ndarray, market_state: Dict[str, np.ndarray], params: Dict[str, float], config) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash((ticker, config.seed))) % (2**32))
    n = len(dates)

    pre_close = np.roll(close, 1)
    pre_close[0] = close[0]
    daily_ret = close / np.where(pre_close == 0, close, pre_close) - 1.0

    intraday_range = np.maximum(0.004, np.abs(daily_ret) * 1.8 + rng.uniform(0.004, 0.02, size=n))
    open_px = pre_close * (1.0 + rng.normal(0.0, np.maximum(0.001, intraday_range * 0.25), size=n))
    high = np.maximum(open_px, close) * (1.0 + intraday_range * rng.uniform(0.15, 0.6, size=n))
    low = np.minimum(open_px, close) * (1.0 - intraday_range * rng.uniform(0.15, 0.6, size=n))

    volume = params["share_base"] * (params["turnover_rate"] / 100.0)
    volume = volume * (1.0 + np.abs(daily_ret) * 12.0) * rng.uniform(0.6, 1.5, size=n)
    amount = volume * close

    return pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ticker,
            "industry": sector,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pre_close,
            "change": close - pre_close,
            "pct_chg": daily_ret * 100.0,
            "volume": volume,
            "vol": volume,
            "amount": amount,
            "turnover": amount,
            "adj_factor": 1.0,
            "adj_close": close,
            "pe_ttm": np.clip(params["pe_base"] * (1 + rng.normal(0.0, 0.03, size=n)), 5.0, 60.0),
            "pb": np.clip(params["pb_base"] * (1 + rng.normal(0.0, 0.04, size=n)), 0.4, 10.0),
            "turnover_rate": np.clip(params["turnover_rate"] * (1 + np.abs(daily_ret) * 8.0), 0.05, 50.0),
            "total_mv": close * params["share_base"] / 1e4,
            "circ_mv": close * params["share_base"] * 0.72 / 1e4,
            "cpi_yoy": market_state["cpi_yoy"],
            "ppi_yoy": market_state["ppi_yoy"],
            "pmi": market_state["pmi"],
            "lpr_1y": market_state["lpr_1y"],
            "real_rate_proxy": market_state["real_rate_proxy"],
            "market_ret": market_state["market_return"],
            "benchmark_return": market_state["market_return"],
            "csi500_ret_20d": market_state["csi500_ret_20d"],
            "market_volatility": market_state["market_volatility"],
            "market_turnover": market_state["market_turnover"],
            "market_liquidity_change": market_state["market_liquidity_change"],
            "north_net_flow": rng.normal(0.0, amount.mean() * 0.0015, size=n),
            "main_net_inflow": rng.normal(0.0, amount.mean() * 0.0009, size=n),
            "roe_growth": np.clip(params["profitability"] + rng.normal(0.0, 3.0, size=n), -10.0, 40.0),
            "revenue_growth": np.clip(params["revenue_growth"] + rng.normal(0.0, 4.0, size=n), -20.0, 50.0),
            "profit_ratio": np.clip(params["profitability"] + rng.normal(0.0, 2.0, size=n), -10.0, 35.0),
            "debt_ratio": params["debt_ratio"],
            "goodwill_risk": params["goodwill_risk"],
        }
    )


def _compute_mock_common_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = df.groupby("ts_code")

    for period in (5, 10, 20):
        df[f"momentum_{period}d"] = grouped["close"].pct_change(period) * 100.0

    df["rsi_6"] = _mock_rsi(df, 6)
    df["rsi_14"] = _mock_rsi(df, 14)

    ma20 = grouped["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    ma60 = grouped["close"].transform(lambda x: x.rolling(60, min_periods=10).mean())
    vol_ma20 = grouped["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    df["price_vs_ma20"] = (df["close"] - ma20) / (ma20 + 1e-8) * 100.0
    df["bias_20d"] = df["price_vs_ma20"]
    df["bias_60d"] = (df["close"] - ma60) / (ma60 + 1e-8) * 100.0
    df["volume_ratio"] = df["volume"] / (vol_ma20 + 1e-8)

    daily_ret = grouped["close"].pct_change()
    df["ret_1d"] = daily_ret * 100.0
    df["ret_3d"] = grouped["close"].pct_change(3) * 100.0
    df["ret_5d"] = grouped["close"].pct_change(5) * 100.0
    df["ret_10d"] = grouped["close"].pct_change(10) * 100.0
    df["ret_20d"] = grouped["close"].pct_change(20) * 100.0

    df["volatility_20d"] = daily_ret.groupby(df["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).std()) * np.sqrt(252) * 100.0
    df["volatility_60d"] = daily_ret.groupby(df["ts_code"]).transform(lambda x: x.rolling(60, min_periods=10).std()) * np.sqrt(252) * 100.0

    df["amplitude"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 100.0
    df["amplitude_20d_avg"] = grouped["amplitude"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["turnover_20d_avg"] = grouped["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / ((df["high"] - df["low"]).replace(0, np.nan))
    df["clv_20d_avg"] = clv.groupby(df["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["upside_volume_ratio_20d"] = ((df["ret_1d"] > 0).astype(float) * df["volume"]).groupby(df["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).sum()) / (df["volume"].groupby(df["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).sum()) + 1e-8)

    rolling_low_60 = grouped["close"].transform(lambda x: x.rolling(60, min_periods=10).min())
    rolling_high_60 = grouped["close"].transform(lambda x: x.rolling(60, min_periods=10).max())
    span = (rolling_high_60 - rolling_low_60).replace(0, np.nan)

    df["winner_rate"] = ((df["close"] - rolling_low_60) / (span + 1e-8)).clip(0.0, 1.0)
    df["trapped_ratio"] = ((rolling_high_60 - df["close"]) / (span + 1e-8)).clip(0.0, 1.0)
    df["chip_support"] = (1.0 - df["trapped_ratio"]).clip(0.0, 1.0)
    df["dist_to_resistance"] = (rolling_high_60 / df["close"].replace(0, np.nan) - 1.0) * 100.0

    avg_cost = grouped["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["asr"] = avg_cost
    df["avg_cost_deviation"] = (df["close"] - avg_cost) / (avg_cost + 1e-8) * 100.0
    df["cost_pressure"] = (df["close"] < avg_cost).astype(float)
    df["chip_concentration_change"] = grouped["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).std())
    df["vol_price_divergence"] = (df["volume_ratio"] - 1.0) * (df["ret_1d"] / 100.0)

    df["pe_rank"] = df.groupby("trade_date")["pe_ttm"].rank(pct=True)
    df["pb_rank"] = df.groupby("trade_date")["pb"].rank(pct=True)
    return df


def _compute_mock_stock_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["pe"] = df["pe_ttm"]
    df["ps_ttm"] = np.clip(df["pe_ttm"] / 5.5, 0.5, 15.0)
    df["chip_support"] = df["chip_support"].fillna(0.5)
    df["winner_rate"] = df["winner_rate"].fillna(0.5)
    return df


def _compute_mock_etf_fields(df: pd.DataFrame, preset) -> pd.DataFrame:
    style = preset.mock_style
    grouped = df.groupby("ts_code")

    share_change = grouped["close"].diff().fillna(0.0) * 1e6
    premium_rate = np.tanh(grouped["ret_1d"].transform(lambda x: x.fillna(0.0)) / 3.0) * style["premium_noise"] * 100.0
    tracking_error = grouped["ret_1d"].transform(lambda x: x.rolling(20, min_periods=5).std()).fillna(0.0) * 100.0 + style["tracking_error"] * 100.0

    df["fd_share"] = 1e8 + share_change.cumsum()
    df["index_close"] = df["close"] / (1.0 + premium_rate / 100.0)
    df["index_pe"] = np.clip(df["pe_ttm"] * 0.92, 6.0, 40.0)
    df["index_pb"] = np.clip(df["pb"] * 0.92, 0.4, 8.0)
    df["unit_nav"] = df["index_close"]
    df["accum_nav"] = grouped["index_close"].cummax()
    df["adj_nav"] = df["unit_nav"]
    df["premium_rate"] = premium_rate
    df["premium_abs"] = np.abs(premium_rate)
    df["premium_positive_freq"] = ((df["premium_rate"] > 0).astype(float).groupby(df["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).mean()))
    df["_share_change"] = share_change
    df["fund_flow_daily"] = share_change * df["close"] / 1e6
    df["fund_flow_5d"] = grouped["fund_flow_daily"].transform(lambda x: x.rolling(5, min_periods=3).sum())
    df["fund_flow_20d"] = grouped["fund_flow_daily"].transform(lambda x: x.rolling(20, min_periods=5).sum())
    df["share_growth_20d"] = grouped["fd_share"].pct_change(20) * 100.0
    df["share_outflow_rate"] = (-share_change.clip(upper=0)).astype(float) / (np.abs(share_change) + 1e-8)
    df["share_change_volatility"] = share_change.groupby(df["ts_code"]).transform(lambda x: x.rolling(20, min_periods=5).std()) / 1e6
    df["nav_growth_rate"] = grouped["unit_nav"].pct_change(20) * 100.0
    nav_drawdown = grouped["unit_nav"].transform(lambda x: (x / x.rolling(60, min_periods=10).max() - 1.0).rolling(60, min_periods=10).min())
    df["nav_max_drawdown_60d"] = -nav_drawdown.fillna(0.0) * 100.0
    index_ret_20d = grouped["index_close"].pct_change(20) * 100.0
    df["excess_return_20d"] = df["ret_20d"] - index_ret_20d
    df["tracking_error_20d"] = tracking_error
    df["alpha_60d"] = grouped["ret_1d"].transform(lambda x: x.rolling(60, min_periods=10).mean()) * 20.0
    df["weighted_avg_cost"] = grouped["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["index_pe_rank"] = df.groupby("trade_date")["index_pe"].rank(pct=True)
    df["index_pb_rank"] = df.groupby("trade_date")["index_pb"].rank(pct=True)
    return df


def _compute_mock_labels(df: pd.DataFrame, config) -> pd.DataFrame:
    grouped = df.groupby("ts_code")
    forward_days = config.data.forward_days

    df["fwd_return_5d"] = grouped["close"].shift(-forward_days) / df["close"] - 1.0
    df["fwd_ret_5d"] = df["fwd_return_5d"]
    df["daily_return"] = grouped["close"].shift(-1) / df["close"] - 1.0
    if "market_ret" in df.columns:
        df["market_ret_5d"] = df.groupby("trade_date")["market_ret"].transform("first").rolling(forward_days, min_periods=1).sum()

    future_min = grouped["low"].transform(lambda x: x.iloc[::-1].rolling(10, min_periods=1).min().iloc[::-1].shift(-1))
    df["max_dd_10d"] = np.clip((df["close"] - future_min) / df["close"].replace(0, np.nan), 0.0, 1.0)

    crash_threshold = config.data.active_crash_threshold
    drawdown_threshold = config.data.active_drawdown_threshold
    crash_return = df["fwd_return_5d"]
    if getattr(config.data, "crash_relative_to_benchmark", False) and "market_ret_5d" in df.columns:
        crash_return = crash_return - df["market_ret_5d"]
    df["crash_label"] = ((crash_return < crash_threshold) | (df["max_dd_10d"] > drawdown_threshold)).astype(int)
    return df


def _mock_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    delta = df.groupby("ts_code")["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.groupby(df["ts_code"]).transform(lambda x: x.rolling(period, min_periods=max(3, period // 2)).mean())
    avg_loss = loss.groupby(df["ts_code"]).transform(lambda x: x.rolling(period, min_periods=max(3, period // 2)).mean())
    rs = avg_gain / (avg_loss + 1e-8)
    return (100 - (100 / (1 + rs))).fillna(50.0)


# =============================================================================
# 合并自 data_validator.py 的特征验证与清洗逻辑
# =============================================================================

@dataclass
class FeatureReport:
    """特征检测与数据质量报告。"""

    total_columns: int = 0
    numeric_columns: int = 0
    bull_features: List[str] = field(default_factory=list)
    bear_features: List[str] = field(default_factory=list)
    friction_features: List[str] = field(default_factory=list)
    macro_features: List[str] = field(default_factory=list)
    missing_bull: List[str] = field(default_factory=list)
    missing_bear: List[str] = field(default_factory=list)
    missing_friction: List[str] = field(default_factory=list)
    missing_macro: List[str] = field(default_factory=list)
    nan_ratio: Dict[str, float] = field(default_factory=dict)
    high_nan_columns: List[str] = field(default_factory=list)
    has_close: bool = False
    has_trade_date: bool = False
    has_ts_code: bool = False
    has_volume: bool = False

    def print_report(self):
        print("\n" + "=" * 70)
        print("  数据验证报告")
        print("=" * 70)
        print(f"\n>>> 基础信息:")
        print(f"    总列数: {self.total_columns}")
        print(f"    数值列: {self.numeric_columns}")
        print(f"    必要列: close={self.has_close}, trade_date={self.has_trade_date}, ts_code={self.has_ts_code}, volume={self.has_volume}")
        print(f"\n>>> 可用特征:")
        print(f"    Bull特征: {len(self.bull_features)} 个")
        print(f"    Bear特征: {len(self.bear_features)} 个")
        print(f"    Friction特征: {len(self.friction_features)} 个")
        print(f"    Macro特征: {len(self.macro_features)} 个")
        if self.missing_bull or self.missing_bear or self.missing_friction:
            print(f"\n>>> 缺失特征 (配置中有但数据中无):")
            if self.missing_bull:
                print(f"    Bull: {self.missing_bull[:5]}{'...' if len(self.missing_bull) > 5 else ''}")
            if self.missing_bear:
                print(f"    Bear: {self.missing_bear[:5]}{'...' if len(self.missing_bear) > 5 else ''}")
            if self.missing_friction:
                print(f"    Friction: {self.missing_friction[:5]}{'...' if len(self.missing_friction) > 5 else ''}")
        if self.high_nan_columns:
            print(f"\n>>> 高缺失率列 (>50% NaN):")
            for col in self.high_nan_columns[:10]:
                print(f"    {col}: {self.nan_ratio[col]:.1%}")
        print("=" * 70)


class FeaturePatterns:
    """通过列名模式识别 bull / bear / friction / macro 特征。"""

    BULL_PATTERNS = [
        'momentum', 'rsi', 'north', 'main_net', 'roe', 'revenue', 'profit',
        'price_vs_ma', 'volume_ratio', 'macd', 'kdj', 'obv', 'cci',
        'willr', 'adx', 'positive', 'bull', 'up', 'buy', 'inflow',
        'support', 'winner', 'chip_support', 'clv', 'upside_volume',
    ]
    BEAR_PATTERNS = [
        'pe_rank', 'pb_rank', 'ps_rank', 'pe_ttm', 'pb', 'ps',
        'bias', 'trapped', 'margin_sell', 'cost_pressure', 'debt',
        'goodwill', 'resistance', 'sell', 'outflow', 'negative',
        'bear', 'down', 'short', 'avg_cost', 'valuation', 'dist_to',
    ]
    FRICTION_PATTERNS = [
        'turnover', 'volatility', 'amplitude', 'asr', 'chip',
        'vol_price', 'liquidity', 'spread', 'impact', 'slippage',
        'concentration', 'dispersion', 'std', 'var',
    ]
    MACRO_PATTERNS = [
        'macro', 'cpi', 'ppi', 'pmi', 'm2', 'lpr', 'shibor',
        'gdp', 'inflation', 'interest', 'exchange', 'market_',
        'index_', 'csi', 'sse', 'szse', 'real_rate', 'liquidity',
    ]
    EXCLUDE_PATTERNS = [
        'trade_date', 'ts_code', 'code', 'symbol', 'name', 'industry',
        'label', 'target', 'return', 'fwd_', 'crash', 'index',
    ]


class DataValidator:
    """根据当前数据自动校验并补齐特征分组。"""

    def __init__(self, config=None):
        self.config = config
        self.patterns = FeaturePatterns()

    def validate_and_adapt(self, df: pd.DataFrame) -> Tuple[Dict, FeatureReport]:
        report = FeatureReport()
        report.total_columns = len(df.columns)
        report.numeric_columns = len(df.select_dtypes(include=[np.number]).columns)
        report.has_close = 'close' in df.columns
        report.has_trade_date = 'trade_date' in df.columns
        report.has_ts_code = 'ts_code' in df.columns or 'code' in df.columns
        report.has_volume = 'volume' in df.columns
        report.nan_ratio = self._check_nan_ratio(df)
        report.high_nan_columns = [col for col, ratio in report.nan_ratio.items() if ratio > 0.5]

        if self.config is not None:
            feature_groups, report = self._validate_config_features(df, report)
        else:
            feature_groups = self._auto_detect_features(df)
            report.bull_features = feature_groups['bull']
            report.bear_features = feature_groups['bear']
            report.friction_features = feature_groups['friction']
            report.macro_features = feature_groups['macro']
        return feature_groups, report

    def _validate_config_features(self, df: pd.DataFrame, report: FeatureReport) -> Tuple[Dict, FeatureReport]:
        available_cols = set(df.columns)
        config_bull = self.config.feature.bull_features if hasattr(self.config, 'feature') else []
        report.bull_features, report.missing_bull = self._filter_available(config_bull, available_cols)
        config_bear = self.config.feature.bear_features if hasattr(self.config, 'feature') else []
        report.bear_features, report.missing_bear = self._filter_available(config_bear, available_cols)
        config_friction = self.config.feature.friction_features if hasattr(self.config, 'feature') else []
        report.friction_features, report.missing_friction = self._filter_available(config_friction, available_cols)
        config_macro = self.config.feature.macro_features if hasattr(self.config, 'feature') else []
        report.macro_features, report.missing_macro = self._filter_available(config_macro, available_cols)

        if not report.bull_features:
            report.bull_features = self._detect_by_pattern(df, self.patterns.BULL_PATTERNS)
        if not report.bear_features:
            report.bear_features = self._detect_by_pattern(df, self.patterns.BEAR_PATTERNS)
        if not report.friction_features:
            report.friction_features = self._detect_by_pattern(df, self.patterns.FRICTION_PATTERNS)
        if not report.macro_features:
            report.macro_features = self._detect_by_pattern(df, self.patterns.MACRO_PATTERNS)

        feature_groups = {
            'bull': report.bull_features,
            'bear': report.bear_features,
            'friction': report.friction_features,
            'macro': report.macro_features,
        }
        return feature_groups, report

    def _filter_available(self, config_features: List[str], available_cols: Set[str]) -> Tuple[List[str], List[str]]:
        available = []
        missing = []
        for feat in config_features:
            if feat in available_cols or f'{feat}_zscore' in available_cols:
                available.append(feat)
            else:
                missing.append(feat)
        return available, missing

    def _auto_detect_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if not any(exc in col.lower() for exc in self.patterns.EXCLUDE_PATTERNS)]
        bull = self._detect_by_pattern(df, self.patterns.BULL_PATTERNS, feature_cols)
        bear = self._detect_by_pattern(df, self.patterns.BEAR_PATTERNS, feature_cols)
        friction = self._detect_by_pattern(df, self.patterns.FRICTION_PATTERNS, feature_cols)
        macro = self._detect_by_pattern(df, self.patterns.MACRO_PATTERNS, feature_cols)

        classified = set(bull + bear + friction + macro)
        unclassified = [col for col in feature_cols if col not in classified]
        groups = [bull, bear, friction]
        for col in unclassified:
            min_group = min(groups, key=len)
            min_group.append(col)

        return {'bull': bull, 'bear': bear, 'friction': friction, 'macro': macro}

    def _detect_by_pattern(self, df: pd.DataFrame, patterns: List[str], candidate_cols: Optional[List[str]] = None) -> List[str]:
        candidate_cols = candidate_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        detected = []
        for col in candidate_cols:
            if any(pattern in col.lower() for pattern in patterns):
                detected.append(col)
        return detected

    def _check_nan_ratio(self, df: pd.DataFrame) -> Dict[str, float]:
        nan_ratio = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                nan_ratio[col] = df[col].isna().mean()
        return nan_ratio


class DataCleaner:
    """统一的特征清洗器，用于 train/valid/test 的一致处理。"""

    def __init__(self, nan_threshold: float = 0.8, fill_method: str = 'median', clip_outliers: bool = True, outlier_std: float = 5.0):
        self.nan_threshold = nan_threshold
        self.fill_method = fill_method
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.column_stats = {}

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy()
        df = self._replace_inf(df, feature_cols)
        valid_cols = []
        for col in feature_cols:
            if col in df.columns:
                nan_ratio = df[col].isna().mean()
                if nan_ratio < self.nan_threshold:
                    valid_cols.append(col)
                else:
                    print(f"    移除高NaN列: {col} ({nan_ratio:.1%} NaN)")

        for col in valid_cols:
            self.column_stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
            }
        df = self._fill_nan(df, valid_cols)
        if self.clip_outliers:
            df = self._clip_outliers(df, valid_cols)
        return df, valid_cols

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        valid_cols = [c for c in feature_cols if c in df.columns]
        df = self._replace_inf(df, valid_cols)
        df = self._fill_nan(df, valid_cols)
        if self.clip_outliers:
            df = self._clip_outliers(df, valid_cols)
        return df

    def _replace_inf(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        inf_count = 0
        for col in cols:
            if col in df.columns:
                col_inf = np.isinf(df[col]).sum()
                if col_inf > 0:
                    inf_count += col_inf
                    finite_values = df[col][np.isfinite(df[col])]
                    if len(finite_values) > 0:
                        max_val = finite_values.quantile(0.99)
                        min_val = finite_values.quantile(0.01)
                        df[col] = df[col].replace([np.inf], max_val)
                        df[col] = df[col].replace([-np.inf], min_val)
                    else:
                        df[col] = df[col].replace([np.inf, -np.inf], 0)
        if inf_count > 0:
            print(f"    处理Inf值: {inf_count} 个")
        return df

    def _fill_nan(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col not in df.columns:
                continue
            if self.fill_method == 'zero':
                df[col] = df[col].fillna(0)
            elif self.fill_method == 'mean':
                df[col] = df[col].fillna(self.column_stats.get(col, {}).get('mean', 0))
            elif self.fill_method == 'median':
                df[col] = df[col].fillna(self.column_stats.get(col, {}).get('median', 0))
            elif self.fill_method == 'ffill':
                if 'ts_code' in df.columns:
                    df[col] = df.groupby('ts_code')[col].ffill().fillna(0)
                else:
                    df[col] = df[col].ffill().fillna(0)
        return df

    def _clip_outliers(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col not in df.columns:
                continue
            stats = self.column_stats.get(col, {})
            mean = stats.get('mean', df[col].mean())
            std = stats.get('std', df[col].std())
            if std > 0:
                lower = mean - self.outlier_std * std
                upper = mean + self.outlier_std * std
                df[col] = df[col].clip(lower, upper)
        return df


class AdaptiveFeatureConfig:
    """根据当前数据自动生成特征分组、维度与清洗器。"""

    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

    def create_config_from_data(self, df: pd.DataFrame, config=None) -> Dict:
        self.validator.config = config
        feature_groups, report = self.validator.validate_and_adapt(df)
        report.print_report()

        feature_groups = self._ensure_minimum_features(df, feature_groups)
        all_features = feature_groups['bull'] + feature_groups['bear'] + feature_groups['friction'] + feature_groups['macro']

        print("\n>>> 数据清洗...")
        df_cleaned, valid_features = self.cleaner.fit_transform(df, all_features)

        valid_set = set(valid_features)
        feature_groups = {
            'bull': [f for f in feature_groups['bull'] if f in valid_set],
            'bear': [f for f in feature_groups['bear'] if f in valid_set],
            'friction': [f for f in feature_groups['friction'] if f in valid_set],
            'macro': [f for f in feature_groups['macro'] if f in valid_set],
        }
        feature_dims = {
            'bull': max(1, len(feature_groups['bull'])),
            'bear': max(1, len(feature_groups['bear'])),
            'friction': max(1, len(feature_groups['friction'])),
            'macro': max(1, len(feature_groups['macro'])),
        }

        print(f"\n>>> 最终特征配置:")
        print(f"    Bull: {feature_dims['bull']} 维")
        print(f"    Bear: {feature_dims['bear']} 维")
        print(f"    Friction: {feature_dims['friction']} 维")
        print(f"    Macro: {feature_dims['macro']} 维")
        print(f"    总计: {sum(feature_dims.values())} 维")

        return {
            'feature_groups': feature_groups,
            'feature_dims': feature_dims,
            'cleaned_df': df_cleaned,
            'report': report,
            'cleaner': self.cleaner,
        }

    def _ensure_minimum_features(self, df: pd.DataFrame, feature_groups: Dict) -> Dict:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fallback_features = {
            'bull': ['close', 'volume', 'open', 'high'],
            'bear': ['close', 'low', 'volume'],
            'friction': ['volume', 'close', 'high', 'low'],
            'macro': [],
        }
        for group_name, features in feature_groups.items():
            if not features and group_name != 'macro':
                for fallback in fallback_features[group_name]:
                    if fallback in numeric_cols:
                        feature_groups[group_name].append(fallback)
                        print(f"    {group_name}组为空，添加fallback特征: {fallback}")
                        break
                if not feature_groups[group_name] and numeric_cols:
                    feature_groups[group_name].append(numeric_cols[0])
                    print(f"    {group_name}组为空，使用默认列: {numeric_cols[0]}")
        return feature_groups


def validate_data(df: pd.DataFrame, config=None) -> Tuple[Dict, FeatureReport]:
    """快速验证当前 DataFrame 的特征可用性。"""
    validator = DataValidator(config)
    return validator.validate_and_adapt(df)


def prepare_data_for_training(df: pd.DataFrame, config=None) -> Dict:
    """一站式生成训练前的特征配置与清洗结果。"""
    adapter = AdaptiveFeatureConfig()
    return adapter.create_config_from_data(df, config)
