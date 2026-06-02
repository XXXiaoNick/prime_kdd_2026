"""
================================================================================
风险守卫者模块 (修复版 v3)
================================================================================

修复内容：
1. 修复难例挖掘的逻辑错误（分数方向问题）
2. 增强调试信息
3. 改进阈值调整逻辑
================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
import warnings

warnings.filterwarnings('ignore')

# 可选依赖
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


class HardNegativeMiner:
    """
    难例挖掘器 (修复版)
    
    找出EBM的"盲区"：那些EBM认为很好但实际崩盘的股票
    
    修复：score越高表示EBM越看好，所以应该选高分位数
    """
    
    def __init__(self, 
                 score_threshold_percentile: float = 80,  # 【修复】改为80分位（高分）
                 crash_label_col: str = 'crash_label',
                 min_hard_negatives: int = 100):  # 【新增】最少难例数量
        self.score_threshold_percentile = score_threshold_percentile
        self.crash_label_col = crash_label_col
        self.min_hard_negatives = min_hard_negatives
    
    def mine(self, df: pd.DataFrame, scores: np.ndarray, return_weights: bool = True):
        """
        挖掘难例并计算样本权重
        
        Args:
            df: 数据DataFrame（必须包含crash_label列）
            scores: EBM分数（越高越看好，即 -energy）
            return_weights: 是否返回样本权重
        """
        if self.crash_label_col not in df.columns:
            print(f"  警告: {self.crash_label_col} 列不存在，跳过难例挖掘")
            return np.array([]), np.ones(len(df))
        
        crash_labels = df[self.crash_label_col].values
        
        # 【关键修复】处理NaN和异常分数
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 【调试信息】
        print(f"  分数统计: min={scores.min():.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, std={scores.std():.4f}")
        print(f"  崩盘标签: 总数={len(crash_labels)}, 崩盘={crash_labels.sum():.0f}, "
              f"比例={crash_labels.mean():.2%}")
        
        # 【修复】检查分数是否有效
        if scores.std() < 1e-6:
            print(f"  ⚠ 分数方差过小 ({scores.std():.6f})，使用随机难例")
            # 随机选择崩盘样本作为难例
            crash_indices = np.where(crash_labels == 1)[0]
            n_hard = min(self.min_hard_negatives, len(crash_indices))
            hard_indices = np.random.choice(crash_indices, n_hard, replace=False) if len(crash_indices) > 0 else np.array([])
            weights = self._compute_sample_weights(crash_labels, np.zeros(len(crash_labels), dtype=bool))
            return hard_indices, weights
        
        # 【修复】选择高分股票（EBM看好的）
        score_threshold = np.percentile(scores, self.score_threshold_percentile)
        ebm_recommended = scores >= score_threshold  # 【关键修复】>= 而不是 <=
        
        # 难例 = EBM看好 但实际崩盘
        hard_negatives = ebm_recommended & (crash_labels == 1)
        
        n_recommended = ebm_recommended.sum()
        n_hard = hard_negatives.sum()
        
        print(f"  难例挖掘: 高分股票={n_recommended} (>{self.score_threshold_percentile}%), "
              f"其中崩盘={n_hard} ({n_hard/max(1,n_recommended):.1%})")
        
        # 【新增】如果难例太少，降低阈值
        if n_hard < self.min_hard_negatives:
            print(f"  难例数量不足，尝试降低阈值...")
            for percentile in [70, 60, 50, 40, 30]:
                threshold = np.percentile(scores, percentile)
                recommended = scores >= threshold
                hard = recommended & (crash_labels == 1)
                if hard.sum() >= self.min_hard_negatives:
                    ebm_recommended = recommended
                    hard_negatives = hard
                    n_hard = hard.sum()
                    print(f"    使用 {percentile}% 阈值, 难例={n_hard}")
                    break
        
        hard_indices = np.where(hard_negatives)[0]
        
        if return_weights:
            weights = self._compute_sample_weights(crash_labels, hard_negatives)
            return hard_indices, weights
        return hard_indices
    
    def _compute_sample_weights(self, crash_labels: np.ndarray, hard_negatives: np.ndarray) -> np.ndarray:
        """计算样本权重"""
        weights = np.ones(len(crash_labels))
        
        # 平衡正负样本
        crash_ratio = crash_labels.mean()
        if 0 < crash_ratio < 1:
            crash_weight = (1 - crash_ratio) / crash_ratio
            crash_weight = min(crash_weight, 10.0)  # 限制最大权重
            weights[crash_labels == 1] = crash_weight
        
        # 难例额外加权
        if hard_negatives.sum() > 0:
            weights[hard_negatives] = weights[hard_negatives] * 3.0  # 【增强】提高难例权重
        
        weights = weights / weights.mean()
        
        return weights


class RiskGuardian:
    """
    风险守卫者 (修复版 v3)
    
    作为外层循环的安全门控，支持Hard Negative Mining
    """
    
    def __init__(self, model_type: str = 'lightgbm', threshold: float = 0.3, config: Optional[dict] = None):
        self.model_type = model_type
        self.threshold = threshold
        self.config = config or {}
        
        self.model = None
        self.feature_names = None
        self.hard_miner = HardNegativeMiner()
        
        self._init_model_params()
    
    def _init_model_params(self):
        """初始化模型参数"""
        if self.model_type == 'lightgbm':
            self.model_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
                'n_jobs': -1,
                # 【新增】防止过拟合
                'min_data_in_leaf': 100,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
            }
        elif self.model_type == 'xgboost':
            self.model_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42,
                'n_jobs': -1,
            }
        else:
            self.model_params = {}
        
        self.model_params.update(self.config)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        sample_weights: Optional[np.ndarray] = None,
        num_boost_round: int = 200,  # 【增加】轮数
        early_stopping_rounds: int = 30  # 【增加】早停
    ) -> 'RiskGuardian':
        """训练Guardian模型"""
        self.feature_names = feature_names
        
        pos_ratio = y_train.mean()
        print(f">>> 训练Guardian ({self.model_type}), 正样本: {pos_ratio:.2%}")
        
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self._fit_lightgbm(X_train, y_train, X_valid, y_valid, sample_weights, 
                             num_boost_round, early_stopping_rounds)
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            self._fit_xgboost(X_train, y_train, X_valid, y_valid, sample_weights,
                            num_boost_round, early_stopping_rounds)
        else:
            self._fit_sklearn(X_train, y_train, sample_weights)
        
        # 评估
        if X_valid is not None:
            y_pred = self.predict_proba(X_valid)
            recall = recall_score(y_valid, y_pred > self.threshold)
            try:
                auc = roc_auc_score(y_valid, y_pred)
            except:
                auc = 0.5
            print(f"  验证集: Recall={recall:.4f}, AUC={auc:.4f}")
        
        return self
    
    def fit_with_hard_mining(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        feature_cols: List[str],
        energy_scores: np.ndarray,
        crash_col: str = 'crash_label',
        **kwargs
    ) -> 'RiskGuardian':
        """使用难例挖掘进行训练"""
        print(">>> 使用难例挖掘训练Guardian")
        
        # 【关键修复】过滤只使用数据中存在的特征列
        available_cols = set(df_train.columns)
        valid_feature_cols = [c for c in feature_cols if c in available_cols]
        
        if len(valid_feature_cols) < len(feature_cols):
            missing = set(feature_cols) - set(valid_feature_cols)
            print(f"    警告: {len(missing)} 个特征不存在，已跳过")
        
        if not valid_feature_cols:
            raise ValueError("没有可用的特征列！请检查数据和配置。")
        
        print(f"    使用 {len(valid_feature_cols)} 个特征训练Guardian")
        
        # 【关键修复】确保scores方向正确
        # energy_scores 应该是 -energy，即分数越高越好
        self.hard_miner.crash_label_col = crash_col
        _, sample_weights = self.hard_miner.mine(df_train, energy_scores)
        
        X_train = df_train[valid_feature_cols].fillna(0).values
        y_train = df_train[crash_col].values
        X_valid = df_valid[valid_feature_cols].fillna(0).values
        y_valid = df_valid[crash_col].values
        
        return self.fit(X_train, y_train, X_valid, y_valid, valid_feature_cols, sample_weights, **kwargs)
    
    def _fit_lightgbm(self, X_train, y_train, X_valid, y_valid, sample_weights, num_boost_round, early_stopping_rounds):
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names, weight=sample_weights)
        
        valid_sets = [train_data]
        if X_valid is not None:
            valid_sets.append(lgb.Dataset(X_valid, label=y_valid))
        
        self.model = lgb.train(
            self.model_params, train_data, num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
    
    def _fit_xgboost(self, X_train, y_train, X_valid, y_valid, sample_weights, num_boost_round, early_stopping_rounds):
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        evals = [(dtrain, 'train')]
        if X_valid is not None:
            evals.append((xgb.DMatrix(X_valid, label=y_valid), 'valid'))
        
        self.model = xgb.train(
            self.model_params, dtrain, num_boost_round=num_boost_round,
            evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False
        )
    
    def _fit_sklearn(self, X_train, y_train, sample_weights):
        if self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=200,
                random_state=42,
            )
            self.model.fit(X_train, y_train)
            return

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=100,
            n_jobs=-1, random_state=42
        )
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            return self.model.predict(X)
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            return self.model.predict(xgb.DMatrix(X))
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)
    
    def get_safe_mask(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """返回安全股票的掩码（预测崩盘概率低于阈值）"""
        crash_proba = self.predict_proba(X)
        return crash_proba < self.threshold
    
    def tune_threshold(
        self,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        target_recall: float = 0.8,
        min_precision: float = 0.1,
        max_threshold: float = 0.5  # 【新增】最大阈值限制
    ) -> float:
        """
        根据验证集调整阈值
        
        【修复】添加max_threshold限制，防止阈值过高导致风险控制失效
        """
        proba = self.predict_proba(X_valid)
        
        best_threshold = min(self.threshold, max_threshold)  # 【修复】初始值也要限制
        best_f1 = 0
        
        # 【修改】限制搜索范围在[0.1, max_threshold]
        search_range = np.arange(0.1, min(0.9, max_threshold + 0.01), 0.05)
        
        for threshold in search_range:
            pred = (proba > threshold).astype(int)
            
            if pred.sum() == 0:
                continue
            
            recall = recall_score(y_valid, pred, zero_division=0)
            precision = precision_score(y_valid, pred, zero_division=0)
            
            if recall >= target_recall and precision >= min_precision:
                f1 = f1_score(y_valid, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        # 如果没有满足条件的阈值，在限制范围内最大化F1
        if best_f1 == 0:
            for threshold in search_range:  # 【修复】使用相同的搜索范围
                pred = (proba > threshold).astype(int)
                if pred.sum() > 0:
                    f1 = f1_score(y_valid, pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        # 【修复】最终确保不超过max_threshold
        self.threshold = min(best_threshold, max_threshold)
        print(f"  阈值调整: {self.threshold:.2f} (F1={best_f1:.4f}, max={max_threshold:.2f})")
        
        return self.threshold
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取特征重要性"""
        if self.model is None or self.feature_names is None:
            return None
        
        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            importance = self.model.feature_importance(importance_type='gain')
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            importance = list(self.model.get_score(importance_type='gain').values())
        elif self.model_type == 'mlp':
            return None
        else:
            importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'threshold': self.threshold,
                'feature_names': self.feature_names,
                'model': self.model
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'RiskGuardian':
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        guardian = cls(model_type=data['model_type'], threshold=data['threshold'])
        guardian.feature_names = data['feature_names']
        guardian.model = data['model']
        return guardian


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  RiskGuardian 模块测试")
    print("=" * 70)
    
    np.random.seed(42)
    n = 10000
    
    # 模拟数据
    X = np.random.randn(n, 20)
    # 崩盘概率与某些特征相关
    crash_proba = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] - 1)))
    y = (np.random.rand(n) < crash_proba).astype(int)
    
    # 模拟EBM分数（高分=看好）
    scores = -X[:, 0] + np.random.randn(n) * 0.5  # 与崩盘因子负相关
    
    print(f"\n测试数据: X={X.shape}, 崩盘比例={y.mean():.2%}")
    
    # 测试难例挖掘
    df = pd.DataFrame({'crash_label': y})
    miner = HardNegativeMiner()
    hard_indices, weights = miner.mine(df, scores)
    
    print(f"\n难例数量: {len(hard_indices)}")
    print(f"权重范围: {weights.min():.2f} ~ {weights.max():.2f}")
    
    # 测试Guardian
    guardian = RiskGuardian(model_type='lightgbm')
    
    # 划分数据
    train_idx = np.random.choice(n, int(n*0.7), replace=False)
    valid_idx = np.array([i for i in range(n) if i not in train_idx])
    
    guardian.fit(
        X[train_idx], y[train_idx],
        X[valid_idx], y[valid_idx],
        sample_weights=weights[train_idx]
    )
    
    # 测试预测
    safe_mask = guardian.get_safe_mask(X[valid_idx])
    print(f"\n安全股票比例: {safe_mask.mean():.2%}")
    print(f"安全股票中崩盘比例: {y[valid_idx][safe_mask].mean():.2%}")
    print(f"危险股票中崩盘比例: {y[valid_idx][~safe_mask].mean():.2%}")


# ============================================================================
# 集成和自适应Guardian
# ============================================================================

class EnsembleGuardian:
    """
    集成风险守卫者
    
    组合多个Guardian的预测结果
    """
    
    def __init__(self, guardians: List[RiskGuardian], weights: Optional[List[float]] = None):
        self.guardians = guardians
        self.weights = weights or [1.0] * len(guardians)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probas = [g.predict_proba(X) * w for g, w in zip(self.guardians, self.weights)]
        return np.sum(probas, axis=0)
    
    def get_safe_mask(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.3) -> np.ndarray:
        return self.predict_proba(X) <= threshold
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.3) -> np.ndarray:
        return (self.predict_proba(X) > threshold).astype(int)


class AdaptiveGuardian:
    """
    自适应风险守卫者
    
    根据市场状态动态调整阈值
    """
    
    def __init__(self, base_guardian: RiskGuardian, market_feature_cols: List[str] = None):
        self.base_guardian = base_guardian
        self.market_feature_cols = market_feature_cols or []
        self.base_threshold = base_guardian.threshold
        
        # 市场状态到阈值的映射
        self.threshold_map = {
            'bull': 0.4,    # 牛市放宽
            'bear': 0.2,    # 熊市收紧
            'neutral': 0.3  # 中性
        }
    
    def _detect_market_state(self, df: pd.DataFrame) -> str:
        """检测市场状态"""
        if not self.market_feature_cols:
            return 'neutral'
        
        # 简单的市场状态检测
        market_features = df[self.market_feature_cols].mean()
        
        # 假设第一个特征是市场收益率
        if len(market_features) > 0:
            market_ret = market_features.iloc[0] if hasattr(market_features, 'iloc') else market_features[0]
            if market_ret > 0.02:
                return 'bull'
            elif market_ret < -0.02:
                return 'bear'
        
        return 'neutral'
    
    def get_adaptive_threshold(self, df: pd.DataFrame) -> float:
        """获取自适应阈值"""
        market_state = self._detect_market_state(df)
        return self.threshold_map.get(market_state, self.base_threshold)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.base_guardian.predict_proba(X)
    
    def get_safe_mask(self, X: Union[pd.DataFrame, np.ndarray], df: pd.DataFrame = None) -> np.ndarray:
        threshold = self.get_adaptive_threshold(df) if df is not None else self.base_threshold
        return self.predict_proba(X) <= threshold
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], df: pd.DataFrame = None) -> np.ndarray:
        threshold = self.get_adaptive_threshold(df) if df is not None else self.base_threshold
        return (self.predict_proba(X) > threshold).astype(int)
