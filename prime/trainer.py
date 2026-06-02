"""
================================================================================
训练器模块 - 课程学习 (Curriculum Learning)
================================================================================
四阶段训练流程：
1. Stage 1: 线性基线 (Sanity Check) - 验证因子有效性
2. Stage 2: EBM排序学习 - 训练Bull/Bear网络
3. Stage 3: Guardian训练 - 风控模型（含难例挖掘）
4. Stage 4: 全系统联合推断
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json

from models.energy_model import GameEnergyModel, MacroConditionedGameEBM, SimplifiedEBM
from models.risk_guardian import RiskGuardian
from models.losses import PairwiseRankingLoss, ICLoss, CombinedLoss, ICRankingLoss
from dataset import (
    concat_tensor_components,
    describe_group_availability,
    prepare_feature_arrays,
    resolve_model_input_dims,
    slice_feature_tensors,
)

# 导入可视化数据记录器
try:
    from visualization import (
        TrainingHistoryRecorder, 
        GamePhaseSpaceRecorder, 
        GuardianDistributionRecorder,
        CorrelationAnalyzer,
        EnergyLandscapeDataRecorder,
        EnergyLandscapeVisualizer,
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    CorrelationAnalyzer = None
    EnergyLandscapeDataRecorder = None
    EnergyLandscapeVisualizer = None

HAS_CORRELATION = HAS_VISUALIZATION and CorrelationAnalyzer is not None
HAS_ENERGY_LANDSCAPE = HAS_VISUALIZATION and EnergyLandscapeDataRecorder is not None

# 尝试导入V3模型和损失
try:
    from models.energy_model import GameEnergyModelV3
    from models.losses import CombinedLossV3
    HAS_V3_MODEL = True
except ImportError:
    HAS_V3_MODEL = False

# 尝试导入改进版模型
try:
    from models.improved_ebm import ImprovedGameEnergyModel, ResidualGameEBM, DirectICLoss
    HAS_IMPROVED_EBM = True
except ImportError:
    HAS_IMPROVED_EBM = False


class EarlyStopping:
    """早停机制（优化：在GPU上保存模型状态，避免CPU-GPU同步）"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        if self.best_score is None:
            self.best_score = score
            if model:
                # 直接在原设备上clone，不转移到CPU
                self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == 'max' else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            if model:
                self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_model(self, model: nn.Module):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.history = {}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append((epoch, v))
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[int, float]:
        if metric not in self.history:
            return 0, 0.0
        values = self.history[metric]
        if mode == 'max':
            return max(values, key=lambda x: x[1])
        return min(values, key=lambda x: x[1])
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


class Stage1Trainer:
    """
    阶段一：线性基线 (Sanity Check)
    
    目的：验证因子有效性，IC > 0.02才进入下一阶段
    """
    
    def __init__(self, config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.model = None
    
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        feature_dims: Dict[str, int]
    ) -> Tuple[bool, float]:
        print("\n" + "=" * 60)
        print("    Stage 1: 线性基线 (Sanity Check)")
        print("=" * 60)

        total_dim = sum(feature_dims.values())
        self.model = SimplifiedEBM(total_dim).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
        criterion = nn.MSELoss()

        # AMP混合精度
        use_amp = getattr(self.config.training, 'use_amp', True) and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        epochs = self.config.training.stage1_epochs

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # 批量GPU传输：先移动再拼接
                bull = batch['bull'].to(self.device, non_blocking=True)
                bear = batch['bear'].to(self.device, non_blocking=True)
                friction = batch['friction'].to(self.device, non_blocking=True)
                macro = batch['macro'].to(self.device, non_blocking=True)
                x = torch.cat([bull, bear, friction, macro], dim=-1)
                y = batch['label'].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = self.model(x).squeeze()
                    loss = criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证IC
            valid_ic = self._compute_ic(valid_loader, use_amp=use_amp)

            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Valid IC={valid_ic:.4f}")

        final_ic = self._compute_ic(valid_loader, use_amp=use_amp)
        passed = final_ic > self.config.training.ic_threshold

        print(f"\n  最终IC: {final_ic:.4f}, 阈值: {self.config.training.ic_threshold}")
        print(f"  {'✓ 通过' if passed else '✗ 未通过'}")

        return passed, final_ic
    
    def _compute_ic(self, loader: DataLoader, use_amp: bool = False) -> float:
        """计算IC（优化：累积在GPU上，最后一次性转移）"""
        self.model.eval()
        all_pred_list, all_target_list = [], []

        with torch.no_grad():
            for batch in loader:
                bull = batch['bull'].to(self.device, non_blocking=True)
                bear = batch['bear'].to(self.device, non_blocking=True)
                friction = batch['friction'].to(self.device, non_blocking=True)
                macro = batch['macro'].to(self.device, non_blocking=True)
                x = torch.cat([bull, bear, friction, macro], dim=-1)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = self.model(x).squeeze()
                all_pred_list.append(pred)
                all_target_list.append(batch['label'].to(self.device, non_blocking=True))

        # 一次性转移回CPU
        all_pred = torch.cat(all_pred_list).cpu().numpy()
        all_target = torch.cat(all_target_list).cpu().numpy()

        mask = ~(np.isnan(all_pred) | np.isnan(all_target))
        if mask.sum() < 10:
            return 0.0

        return np.corrcoef(all_pred[mask], all_target[mask])[0, 1]


class Stage2Trainer:
    """
    阶段二：EBM排序学习（改进版 V2）
    
    支持两种模型版本：
    - V1: 原版GameEnergyModel（Concat方式）
    - V2: GameEnergyModelV2（宏观调制 + 刚性Bear + 势能陷阱）
    
    支持两种损失函数：
    - V1: ICRankingLoss（IC + Pairwise + ListNet）
    - V2: CombinedLossV2（IC + Pointwise + TopK + NDCG）
    """
    
    def __init__(self, config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.model = None
        
        # 判断使用哪个版本
        self.model_version = getattr(config.model, 'model_version', 'v1')
        self.loss_version = getattr(config.training, 'loss_version', 'v1')
        
        # 【新增】Correlation分析器
        self.correlation_analyzer = None
        
        # 【新增】Energy Landscape数据记录器
        self.landscape_recorder = None
        self.feature_groups = None
    
    def train(
        self,
        pairwise_loader: DataLoader,
        valid_loader: DataLoader,
        feature_dims: Dict[str, int],
        use_ic_loss: bool = True,
        feature_groups: Optional[Dict[str, List[str]]] = None,  # 【新增】特征分组
        enable_landscape: bool = True,  # 【新增】是否启用能量地形图数据收集
        aux_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """训练EBM模型"""
        print("\n" + "=" * 60)
        print("    Stage 2: EBM 排序学习")
        print("=" * 60)
        print(f"  模型版本: {self.model_version}, 损失版本: {self.loss_version}")
        
        # 【新增】保存特征分组
        self.feature_groups = feature_groups
        
        # 【新增】初始化Energy Landscape数据记录器
        if enable_landscape and HAS_ENERGY_LANDSCAPE:
            self.landscape_recorder = EnergyLandscapeDataRecorder(output_dir='outputs/landscape')
            print("  ✓ Energy Landscape数据记录器已启用")
        
        # 【新增】初始化Correlation分析器
        if HAS_CORRELATION:
            self.correlation_analyzer = CorrelationAnalyzer(output_dir='outputs/correlation')
            if feature_groups:
                self.correlation_analyzer.set_feature_names(feature_groups)
        
        # 创建模型
        self._create_model(feature_dims)
        
        # 创建损失函数
        criterion = self._create_loss()
        
        # 训练（传入valid_loader用于收集数据）
        trained_model = self._train_loop(pairwise_loader, valid_loader, criterion, aux_loader=aux_loader)
        
        # 【新增】训练结束后保存能量地形数据
        if self.landscape_recorder is not None:
            print("\n>>> 保存Energy Landscape数据...")
            self.landscape_recorder.save()
        
        # 【新增】训练结束后收集correlation数据
        if HAS_CORRELATION and self.correlation_analyzer is not None:
            print("\n>>> 收集Correlation Matrix数据...")
            self._collect_correlation_data(valid_loader)
        
        return trained_model
    
    def _create_model(self, feature_dims: Dict[str, int]):
        """创建模型"""
        model_cfg = self.config.model
        
        if self.model_version == 'v4':
            # 检查是否使用V4.2（物理场修复版）
            use_v42 = getattr(model_cfg, 'use_v42', True)  # 默认使用V4.2
            
            if use_v42:
                # 使用V4.2物理场修复版
                from models.energy_model import GameEnergyModelV4_2
                
                # 【关键修复】读取消融配置
                enable_bull = getattr(model_cfg, 'enable_bull', True)
                enable_bear = getattr(model_cfg, 'enable_bear', True)
                enable_friction = getattr(model_cfg, 'enable_friction', True)
                enable_macro_modulation = getattr(model_cfg, 'enable_macro_modulation', True)
                fixed_alpha = getattr(model_cfg, 'fixed_alpha', None)
                fixed_beta = getattr(model_cfg, 'fixed_beta', None)
                fixed_gamma = getattr(model_cfg, 'fixed_gamma', None)
                input_noise_std = getattr(model_cfg, 'input_noise_std', 0.03)
                
                self.model = GameEnergyModelV4_2(
                    bull_dim=feature_dims['bull'],
                    bear_dim=feature_dims['bear'],
                    heat_dim=feature_dims['friction'],  # friction -> heat
                    macro_dim=feature_dims['macro'],
                    encoder_hidden=getattr(model_cfg, 'encoder_hidden', 32),
                    encoder_output=getattr(model_cfg, 'encoder_output', 16),
                    dropout=model_cfg.dropout,
                    alpha_range=(0.5, 2.0),
                    beta_range=(0.5, 2.0),
                    gamma_range=(0.0, 1.0),
                    heat_cap=getattr(model_cfg, 'heat_cap', 1.0),
                    heat_penalty=getattr(model_cfg, 'heat_penalty', 0.5),
                    direction_reg_weight=getattr(model_cfg, 'direction_reg_weight', 0.01),
                    # 【关键修复】传入消融参数
                    enable_bull=enable_bull,
                    enable_bear=enable_bear,
                    enable_friction=enable_friction,
                    enable_macro_modulation=enable_macro_modulation,
                    fixed_alpha=fixed_alpha,
                    fixed_beta=fixed_beta,
                    fixed_gamma=fixed_gamma,
                    input_noise_std=input_noise_std,
                    aggregation_mode=getattr(model_cfg, 'aggregation_mode', 'potential_game'),
                    aggregation_hidden_dim=getattr(model_cfg, 'aggregation_hidden_dim', 12),
                    energy_target_mean=getattr(model_cfg, 'energy_target_mean', 0.5),
                    energy_target_std=getattr(model_cfg, 'energy_target_std', 0.15),
                    energy_bound_low=getattr(model_cfg, 'energy_bound_low', 0.1),
                    energy_bound_high=getattr(model_cfg, 'energy_bound_high', 0.9),
                    energy_bound_mode=getattr(model_cfg, 'energy_bound_mode', 'fixed'),
                    energy_target_std_min=getattr(model_cfg, 'energy_target_std_min', 0.08),
                    energy_target_std_max=getattr(model_cfg, 'energy_target_std_max', 0.25),
                    energy_bound_adaptive_scale=getattr(model_cfg, 'energy_bound_adaptive_scale', 0.5),
                ).to(self.device)
                
                print(f"  模型: GameEnergyModelV4.2 (物理场修复版)")
                print(f"    - 编码器隐藏层: {getattr(model_cfg, 'encoder_hidden', 32)}")
                print(f"    - 编码器输出: {getattr(model_cfg, 'encoder_output', 16)}")
                print(f"    - 宏观调制: α∈[0.5,2.0], β∈[0.5,2.0], γ∈[0.0,1.0]")
                print(f"    - Heat有益上限: {getattr(model_cfg, 'heat_cap', 1.0)}")
                print(f"    - Heat过高惩罚: {getattr(model_cfg, 'heat_penalty', 0.5)}")
                print(f"    - 聚合模式: {getattr(model_cfg, 'aggregation_mode', 'potential_game')}")
                # 打印消融配置
                print(f"  消融配置:")
                print(f"    - enable_bull: {enable_bull}")
                print(f"    - enable_bear: {enable_bear}")
                print(f"    - enable_friction: {enable_friction}")
                print(f"    - enable_macro_modulation: {enable_macro_modulation}")
                print(f"    - input_noise_std: {input_noise_std}")
            else:
                # 使用V4.1自适应方向+双系数调制模型
                from models.energy_model import GameEnergyModelV4_1
                
                self.model = GameEnergyModelV4_1(
                    bull_dim=feature_dims['bull'],
                    bear_dim=feature_dims['bear'],
                    friction_dim=feature_dims['friction'],
                    macro_dim=feature_dims['macro'],
                    encoder_hidden=getattr(model_cfg, 'encoder_hidden', 32),
                    encoder_output=getattr(model_cfg, 'encoder_output', 16),
                    dropout=model_cfg.dropout,
                    macro_coef_min=getattr(model_cfg, 'macro_coef_min', 0.3),
                    macro_coef_max=getattr(model_cfg, 'macro_coef_max', 3.0),
                    friction_weight=getattr(model_cfg, 'friction_weight', 0.1)
                ).to(self.device)
                
                print(f"  模型: GameEnergyModelV4.1 (自适应方向+双系数调制)")
                print(f"    - 编码器隐藏层: {getattr(model_cfg, 'encoder_hidden', 32)}")
                print(f"    - 编码器输出: {getattr(model_cfg, 'encoder_output', 16)}")
                print(f"    - 宏观调制范围: [{getattr(model_cfg, 'macro_coef_min', 0.3)}, {getattr(model_cfg, 'macro_coef_max', 3.0)}]")
                print(f"    - Friction权重: {getattr(model_cfg, 'friction_weight', 0.1)}")
        
        elif self.model_version == 'v3':
            # 使用V3平衡版模型
            from models.energy_model import GameEnergyModelV3
            
            self.model = GameEnergyModelV3(
                bull_dim=feature_dims['bull'],
                bear_dim=feature_dims['bear'],
                friction_dim=feature_dims['friction'],
                macro_dim=feature_dims['macro'],
                hidden_dims=getattr(model_cfg, 'hidden_dims', [128, 64, 32]),
                bear_enhance_scale=getattr(model_cfg, 'bear_enhance_scale', 2.0),
                min_energy_output=getattr(model_cfg, 'min_energy_output', 0.5),
                max_energy_output=getattr(model_cfg, 'max_energy_output', 10.0),
                macro_coef_min=getattr(model_cfg, 'macro_coef_min', 0.5),
                macro_coef_max=getattr(model_cfg, 'macro_coef_max', 2.0),
                friction_weight=getattr(model_cfg, 'friction_weight', 0.1),
                dropout=model_cfg.dropout,
                target_bear_ratio=getattr(model_cfg, 'target_bear_ratio', 0.3)
            ).to(self.device)
            
            print(f"  模型: GameEnergyModelV3 (平衡版)")
            print(f"    - Bear特征增强: {getattr(model_cfg, 'bear_enhance_scale', 2.0)}")
            print(f"    - 能量输出范围: [{getattr(model_cfg, 'min_energy_output', 0.5)}, {getattr(model_cfg, 'max_energy_output', 10.0)}]")
            print(f"    - 宏观调制范围: [{getattr(model_cfg, 'macro_coef_min', 0.5)}, {getattr(model_cfg, 'macro_coef_max', 2.0)}]")
            print(f"    - 目标Bear比例: {getattr(model_cfg, 'target_bear_ratio', 0.3)}")
        
        elif self.model_version == 'v2':
            # 使用改进版V2模型
            from models.energy_model import GameEnergyModelV2
            
            self.model = GameEnergyModelV2(
                bull_dim=feature_dims['bull'],
                bear_dim=feature_dims['bear'],
                friction_dim=feature_dims['friction'],
                macro_dim=feature_dims['macro'],
                bull_hidden=model_cfg.bull_hidden_dims,
                bear_hidden=model_cfg.bear_hidden_dims,
                rigidity_beta=getattr(model_cfg, 'bear_rigidity_beta', 5.0),
                volatility_penalty=getattr(model_cfg, 'volatility_penalty', 0.1),
                target_volatility=getattr(model_cfg, 'target_volatility', 0.0),
                dropout=model_cfg.dropout
            ).to(self.device)
            
            print(f"  模型: GameEnergyModelV2")
            print(f"    - Bear刚性β: {getattr(model_cfg, 'bear_rigidity_beta', 5.0)}")
            print(f"    - 波动率惩罚: {getattr(model_cfg, 'volatility_penalty', 0.1)}")
            print(f"    - 宏观调制范围: [{getattr(model_cfg, 'macro_coef_min', 0.3)}, {getattr(model_cfg, 'macro_coef_max', 3.0)}]")
        else:
            # 使用原版V1模型
            bear_weight = getattr(model_cfg, 'bear_weight', 1.0)
            
            self.model = GameEnergyModel(
                bull_dim=feature_dims['bull'],
                bear_dim=feature_dims['bear'],
                friction_dim=feature_dims['friction'],
                macro_dim=feature_dims['macro'],
                bull_hidden=model_cfg.bull_hidden_dims,
                bear_hidden=model_cfg.bear_hidden_dims,
                interaction_lambda=model_cfg.interaction_lambda,
                pure_friction_lambda=model_cfg.pure_friction_lambda,
                bear_weight=bear_weight,
                dropout=model_cfg.dropout
            ).to(self.device)
            
            print(f"  模型: GameEnergyModel (V1)")
            print(f"    - Bear权重: {bear_weight}")
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"    - 参数量: {param_count:,}")
    
    def _create_loss(self):
        """创建损失函数"""
        train_cfg = self.config.training
        model_cfg = self.config.model if hasattr(self.config, 'model') else self.config
        
        if self.loss_version == 'v4':
            # 检查是否使用V4.2损失（分层IC）
            use_v42 = getattr(model_cfg, 'use_v42', True)
            
            if use_v42:
                from models.losses import CombinedLossV4_2
                
                criterion = CombinedLossV4_2(
                    ic_weight=getattr(train_cfg, 'ic_loss_weight', 1.0),
                    rank_weight=getattr(train_cfg, 'rank_loss_weight', 0.5),  # 增加Rank IC权重
                    physics_weight=getattr(train_cfg, 'physics_weight', 0.1),
                    vol_neutral_weight=getattr(train_cfg, 'vol_neutral_weight', 0.2),  # 波动率中性化
                    direction_reg_weight=getattr(train_cfg, 'direction_reg_weight', 0.01),
                    n_tiers=getattr(train_cfg, 'n_tiers', 5),
                    tiered_ic_weight=0.0  # 禁用分层IC！
                )
                
                print(f"  损失函数: CombinedLossV4.2 (IC + RankIC + 波动率中性化)")
                print(f"    - IC权重: {getattr(train_cfg, 'ic_loss_weight', 1.0)}")
                print(f"    - Rank IC权重: {getattr(train_cfg, 'rank_loss_weight', 0.5)}")
                print(f"    - 波动率中性化权重: {getattr(train_cfg, 'vol_neutral_weight', 0.2)}")
                print(f"    - 物理约束权重: {getattr(train_cfg, 'physics_weight', 0.1)}")
            else:
                from models.losses import CombinedLossV4
                
                criterion = CombinedLossV4(
                    ic_weight=getattr(train_cfg, 'ic_loss_weight', 1.0),
                    rank_weight=getattr(train_cfg, 'rank_loss_weight', 0.5),
                    reg_weight=getattr(train_cfg, 'reg_loss_weight', 0.01),
                    rank_temperature=getattr(train_cfg, 'rank_temperature', 1.0)
                )
                
                print(f"  损失函数: CombinedLossV4 (IC + Rank IC)")
                print(f"    - IC权重: {getattr(train_cfg, 'ic_loss_weight', 1.0)}")
                print(f"    - Rank权重: {getattr(train_cfg, 'rank_loss_weight', 0.5)}")
        
        elif self.loss_version == 'v3':
            from models.losses import CombinedLossV3
            
            criterion = CombinedLossV3(
                ic_weight=getattr(train_cfg, 'ic_loss_weight', 1.0),
                pointwise_weight=getattr(train_cfg, 'pointwise_loss_weight', 0.3),
                bear_constraint_weight=getattr(train_cfg, 'bear_constraint_weight', 2.0),
                macro_balance_weight=getattr(train_cfg, 'macro_balance_weight', 0.5),
                reg_weight=getattr(train_cfg, 'reg_loss_weight', 0.01),
                target_bear_ratio=getattr(self.config.model, 'target_bear_ratio', 0.3),
                min_bear_ratio=getattr(self.config.model, 'min_bear_ratio', 0.1)
            )
            
            print(f"  损失函数: CombinedLossV3 (Bear约束版)")
            print(f"    - IC权重: {getattr(train_cfg, 'ic_loss_weight', 1.0)}")
            print(f"    - Bear约束权重: {getattr(train_cfg, 'bear_constraint_weight', 2.0)}")
            print(f"    - 目标Bear比例: {getattr(self.config.model, 'target_bear_ratio', 0.3)}")
        
        elif self.loss_version == 'v2':
            from models.losses import CombinedLossV2
            
            criterion = CombinedLossV2(
                ic_weight=getattr(train_cfg, 'ic_loss_weight', 1.0),
                pointwise_weight=getattr(train_cfg, 'pointwise_loss_weight', 0.5),
                topk_weight=getattr(train_cfg, 'topk_loss_weight', 0.3),
                ndcg_weight=getattr(train_cfg, 'ndcg_loss_weight', 0.2),
                reg_weight=getattr(train_cfg, 'reg_loss_weight', 0.01),
                top_k=self.config.backtest.top_k
            )
            
            print(f"  损失函数: CombinedLossV2")
            print(f"    - IC权重: {train_cfg.ic_loss_weight}")
            print(f"    - Pointwise权重: {train_cfg.pointwise_loss_weight}")
            print(f"    - TopK权重: {train_cfg.topk_loss_weight}")
        else:
            criterion = ICRankingLoss(
                ic_weight=getattr(train_cfg, 'ic_loss_weight', 1.0),
                ranking_weight=getattr(train_cfg, 'ranking_loss_weight', 0.3),
                listnet_weight=getattr(train_cfg, 'listnet_loss_weight', 0.5),
                margin=self.config.model.energy_margin
            )
            
            print(f"  损失函数: ICRankingLoss (V1)")
        
        return criterion
    
    def _train_loop(
        self,
        pairwise_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        aux_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """训练循环（优化：AMP + 批量GPU传输 + 减少验证开销）"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.training.stage2_epochs
        )

        early_stop = EarlyStopping(patience=self.config.training.patience, mode='max')
        best_ic = -1.0
        best_rank_ic = -1.0
        best_combined = -1.0

        # AMP混合精度
        use_amp = getattr(self.config.training, 'use_amp', True) and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if use_amp:
            print(f"  ✓ AMP混合精度训练已启用 (GPU加速2-3x)")

        # 初始化训练历史记录器
        if HAS_VISUALIZATION:
            self.training_history = TrainingHistoryRecorder()

        total_epochs = self.config.training.stage2_epochs
        crash_aux_weight = float(getattr(self.config.training, 'crash_aux_weight', 0.0) or 0.0)
        crash_aux_criterion = nn.BCEWithLogitsLoss()

        for epoch in range(total_epochs):
            self.model.train()
            train_loss = 0.0
            train_ic = 0.0
            n_batches = 0
            crash_aux_epoch = 0.0
            aux_iter = iter(aux_loader) if aux_loader is not None and crash_aux_weight > 0 else None

            pbar = tqdm(pairwise_loader, desc=f"Epoch {epoch+1}", leave=False)

            for batch in pbar:
                # 批量GPU传输：使用non_blocking=True异步传输
                w_bull = batch['winner_bull'].to(self.device, non_blocking=True)
                w_bear = batch['winner_bear'].to(self.device, non_blocking=True)
                w_friction = batch['winner_friction'].to(self.device, non_blocking=True)
                w_macro = batch['winner_macro'].to(self.device, non_blocking=True)
                w_return = batch.get('winner_label', torch.zeros(w_bull.size(0), device=self.device)).to(self.device, non_blocking=True)

                l_bull = batch['loser_bull'].to(self.device, non_blocking=True)
                l_bear = batch['loser_bear'].to(self.device, non_blocking=True)
                l_friction = batch['loser_friction'].to(self.device, non_blocking=True)
                l_macro = batch['loser_macro'].to(self.device, non_blocking=True)
                l_return = batch.get('loser_label', torch.zeros(l_bull.size(0), device=self.device)).to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # AMP autocast
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # 计算能量
                    if self.loss_version == 'v3':
                        e_winner, comp_winner = self.model(w_bull, w_bear, w_friction, w_macro, return_components=True)
                        e_loser, comp_loser = self.model(l_bull, l_bear, l_friction, l_macro, return_components=True)

                        all_energy = torch.cat([e_winner, e_loser])
                        all_returns = torch.cat([w_return, l_return])

                        all_components = {}
                        for key in comp_winner.keys():
                            if torch.is_tensor(comp_winner[key]) and comp_winner[key].dim() >= 1:
                                all_components[key] = torch.cat([comp_winner[key], comp_loser[key]])

                        loss_dict = criterion(all_energy, all_returns, all_components)

                    elif self.loss_version == 'v4':
                        e_winner, comp_winner = self.model(
                            w_bull, w_bear, w_friction, w_macro, return_components=True
                        )
                        e_loser, comp_loser = self.model(
                            l_bull, l_bear, l_friction, l_macro, return_components=True
                        )

                        all_energy = torch.cat([e_winner, e_loser])
                        all_returns = torch.cat([w_return, l_return])
                        all_components = concat_tensor_components(comp_winner, comp_loser)
                        volatility = all_components.get('E_heat', all_components.get('E_friction'))

                        if getattr(self.config.model, 'use_v42', True):
                            loss_dict = criterion(
                                all_energy,
                                all_returns,
                                components=all_components,
                                model=self.model,
                                volatility=volatility,
                            )
                        else:
                            loss_dict = criterion(all_energy, all_returns, components=all_components)

                    else:
                        e_winner = self.model(w_bull, w_bear, w_friction, w_macro)
                        e_loser = self.model(l_bull, l_bear, l_friction, l_macro)

                        all_energy = torch.cat([e_winner, e_loser])
                        all_returns = torch.cat([w_return, l_return])

                        if self.loss_version == 'v2':
                            loss_dict = criterion(all_energy, all_returns)
                        else:
                            loss_dict = criterion(
                                pred_scores=all_energy,
                                true_returns=all_returns,
                                e_winner=e_winner,
                                e_loser=e_loser
                            )

                    if aux_iter is not None:
                        try:
                            aux_batch = next(aux_iter)
                        except StopIteration:
                            aux_iter = iter(aux_loader)
                            aux_batch = next(aux_iter)

                        aux_bull = aux_batch['bull'].to(self.device, non_blocking=True)
                        aux_bear = aux_batch['bear'].to(self.device, non_blocking=True)
                        aux_friction = aux_batch['friction'].to(self.device, non_blocking=True)
                        aux_macro = aux_batch['macro'].to(self.device, non_blocking=True)
                        aux_crash = aux_batch['crash_label'].to(self.device, non_blocking=True).float()
                        aux_energy = self.model(aux_bull, aux_bear, aux_friction, aux_macro).squeeze()
                        aux_logits = (aux_energy - aux_energy.mean()) / (aux_energy.std(unbiased=False) + 1e-6)
                        crash_aux_loss = crash_aux_criterion(aux_logits, aux_crash)
                        loss_dict['crash_aux'] = crash_aux_loss
                        loss_dict['total'] = loss_dict['total'] + crash_aux_weight * crash_aux_loss

                loss = loss_dict['total']
                batch_ic = loss_dict['ic'].item() if 'ic' in loss_dict else 0.0

                # AMP反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_ic += batch_ic
                n_batches += 1
                if 'crash_aux' in loss_dict:
                    crash_aux_epoch += float(loss_dict['crash_aux'].item())

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ic': f'{batch_ic:.4f}'})

            scheduler.step()

            # 验证 - 合并IC计算和统计打印为单次遍历
            valid_ic, valid_rank_ic, v4_stats = self._compute_ic_and_stats(valid_loader, use_amp=use_amp)

            # 收集Energy Landscape数据（仅在首尾epoch）
            if self.landscape_recorder is not None and (epoch == 0 or epoch == total_epochs - 1):
                self._collect_landscape_epoch_data(valid_loader, epoch, valid_ic)

            # 打印宏观调制系数
            if self.model_version in ['v2', 'v3'] and v4_stats.get('alpha_mean') is not None:
                print(f"    宏观调制: α_market={v4_stats['alpha_mean']:.3f}, β_risk={v4_stats['beta_mean']:.3f}")

            # V3额外打印Bear贡献比例
            if self.loss_version == 'v3' and 'bear_ratio' in loss_dict:
                bear_ratio = loss_dict['bear_ratio'].item() if torch.is_tensor(loss_dict['bear_ratio']) else loss_dict['bear_ratio']
                print(f"    Bear贡献比例: {bear_ratio:.4f}")

            # V4额外打印Rank IC损失
            if self.loss_version == 'v4' and 'rank_ic' in loss_dict:
                rank_ic_train = loss_dict['rank_ic'].item() if torch.is_tensor(loss_dict['rank_ic']) else loss_dict['rank_ic']
                print(f"    训练Rank IC: {rank_ic_train:.4f}")
            else:
                rank_ic_train = 0.0
            if crash_aux_weight > 0 and n_batches > 0:
                print(f"    Crash Aux Loss: {crash_aux_epoch / n_batches:.4f}")

            # V4打印宏观调制和特征方向
            if self.model_version == 'v4' and v4_stats:
                output = f"    宏观调制:"
                if v4_stats.get('alpha_mean') is not None:
                    output += f" α={v4_stats['alpha_mean']:.3f}(std={v4_stats['alpha_std']:.3f})"
                    output += f" β={v4_stats['beta_mean']:.3f}(std={v4_stats['beta_std']:.3f})"
                if v4_stats.get('gamma_mean') is not None:
                    output += f" γ={v4_stats['gamma_mean']:.3f}(std={v4_stats['gamma_std']:.3f})"
                print(output)

                # 记录训练历史数据
                if HAS_VISUALIZATION and hasattr(self, 'training_history'):
                    self.training_history.record_epoch(
                        epoch=epoch + 1,
                        alpha_mean=v4_stats.get('alpha_mean', 1.0),
                        alpha_std=v4_stats.get('alpha_std', 0.0),
                        beta_mean=v4_stats.get('beta_mean', 1.0),
                        beta_std=v4_stats.get('beta_std', 0.0),
                        gamma_mean=v4_stats.get('gamma_mean', 0.5),
                        gamma_std=v4_stats.get('gamma_std', 0.0),
                        train_loss=train_loss / n_batches,
                        valid_ic=valid_ic,
                        valid_rank_ic=valid_rank_ic,
                        train_rank_ic=rank_ic_train
                    )

            # 使用IC和Rank IC的组合作为早停标准
            combined_score = 0.5 * valid_ic + 0.5 * valid_rank_ic

            if combined_score > best_combined:
                best_combined = combined_score
                best_ic = valid_ic
                best_rank_ic = valid_rank_ic

            print(f"  Epoch {epoch+1}: Loss={train_loss/n_batches:.4f}, "
                  f"IC={valid_ic:.4f}, RankIC={valid_rank_ic:.4f}, Best={best_ic:.4f}")

            if early_stop(combined_score, self.model):
                print(f"  早停于 epoch {epoch+1}")
                break

        early_stop.load_best_model(self.model)

        final_ic, final_rank_ic, _ = self._compute_ic_and_stats(valid_loader, use_amp=use_amp)
        print(f"\n  ✓ Stage 2 完成")
        print(f"    最佳 IC: {best_ic:.4f}")
        print(f"    最佳 Rank IC: {best_rank_ic:.4f}")
        print(f"    最终 IC: {final_ic:.4f}, Rank IC: {final_rank_ic:.4f}")

        if HAS_VISUALIZATION and hasattr(self, 'training_history'):
            viz_dir = Path('outputs/viz_data')
            viz_dir.mkdir(parents=True, exist_ok=True)
            self.training_history.save(str(viz_dir / 'training_history.json'))

        return self.model
    
    def _print_modulation_stats(self, loader: DataLoader):
        """打印宏观调制系数统计（仅V2）"""
        self.model.eval()
        alphas, betas = [], []
        
        with torch.no_grad():
            for batch in loader:
                z_macro = batch['macro'].to(self.device)
                coeffs = self.model.get_modulation_coefficients(z_macro)
                alphas.extend(coeffs['alpha_market'].cpu().numpy().flatten())
                betas.extend(coeffs['beta_risk'].cpu().numpy().flatten())
        
        alpha_mean = np.mean(alphas)
        beta_mean = np.mean(betas)
        print(f"    宏观调制: α_market={alpha_mean:.3f}, β_risk={beta_mean:.3f}")
    
    def _print_v4_stats(self, loader: DataLoader):
        """打印V4/V4.1/V4.2模型的宏观调制和特征方向统计"""
        self.model.eval()
        alphas, betas, gammas = [], [], []
        
        # 收集宏观调制系数
        with torch.no_grad():
            for batch in loader:
                z_macro = batch['macro'].to(self.device)
                
                # V4.1/V4.2有get_modulation_coefficients方法
                if hasattr(self.model, 'get_modulation_coefficients'):
                    coeffs = self.model.get_modulation_coefficients(z_macro)
                    alphas.extend(coeffs['alpha_market'].cpu().numpy().flatten())
                    betas.extend(coeffs['beta_risk'].cpu().numpy().flatten())
                    if 'gamma_heat' in coeffs:
                        gammas.extend(coeffs['gamma_heat'].cpu().numpy().flatten())
                # V4只有macro_modulation返回scale
                elif hasattr(self.model, 'macro_modulation'):
                    scale = self.model.macro_modulation(z_macro)
                    alphas.extend(scale.cpu().numpy().flatten())
        
        # 打印调制系数
        if alphas and betas:
            alpha_mean = np.mean(alphas)
            beta_mean = np.mean(betas)
            output = f"    宏观调制: α_market={alpha_mean:.3f} (std={np.std(alphas):.3f}), β_risk={beta_mean:.3f} (std={np.std(betas):.3f})"
            
            if gammas:
                gamma_mean = np.mean(gammas)
                output += f", γ_heat={gamma_mean:.3f} (std={np.std(gammas):.3f})"
            
            print(output)
        elif alphas:
            scale_mean = np.mean(alphas)
            scale_std = np.std(alphas)
            print(f"    宏观调制scale: mean={scale_mean:.4f}, std={scale_std:.4f}, "
                  f"range=[{min(alphas):.3f}, {max(alphas):.3f}]")
        
        # 打印特征方向统计
        if hasattr(self.model, 'get_feature_directions'):
            directions = self.model.get_feature_directions()
            for group_name, weights in directions.items():
                if group_name == 'heat':  # 跳过重复的heat（已经有friction）
                    continue
                    
                direction = weights['direction'].cpu().numpy()
                importance = weights['importance'].cpu().numpy()
                
                # 统计
                n_reversed = (direction < 0).sum()
                n_strong = (importance > 0.7).sum()
                n_weak = (importance < 0.3).sum()
                
                # 打印方向值的分布
                dir_mean = np.mean(direction)
                dir_std = np.std(direction)
                
                print(f"    {group_name}特征方向: {n_reversed}/{len(direction)}个反转 (均值={dir_mean:.2f}, std={dir_std:.2f}), "
                      f"重要性: {n_strong}个强(>0.7), {n_weak}个弱(<0.3)")
    
    def _compute_ic_and_stats(
        self,
        loader: DataLoader,
        use_amp: bool = False
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        合并计算IC、Rank IC和V4统计（单次遍历，替代原来3次遍历）

        Returns:
            (ic, rank_ic, v4_stats_dict)
        """
        from scipy.stats import spearmanr

        self.model.eval()
        all_scores_gpu, all_returns_gpu = [], []
        alphas_gpu, betas_gpu, gammas_gpu = [], [], []

        with torch.no_grad():
            for batch in loader:
                x_bull = batch['bull'].to(self.device, non_blocking=True)
                x_bear = batch['bear'].to(self.device, non_blocking=True)
                x_friction = batch['friction'].to(self.device, non_blocking=True)
                x_macro = batch['macro'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    energy = self.model(x_bull, x_bear, x_friction, x_macro)
                scores = -energy.squeeze()
                all_scores_gpu.append(scores)
                all_returns_gpu.append(labels)

                # 同时收集调制系数（如果有）
                if hasattr(self.model, 'get_modulation_coefficients'):
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        coeffs = self.model.get_modulation_coefficients(x_macro)
                    alphas_gpu.append(coeffs['alpha_market'].squeeze())
                    betas_gpu.append(coeffs['beta_risk'].squeeze())
                    if 'gamma_heat' in coeffs:
                        gammas_gpu.append(coeffs['gamma_heat'].squeeze())
                elif hasattr(self.model, 'macro_modulation'):
                    scale = self.model.macro_modulation(x_macro)
                    alphas_gpu.append(scale.squeeze())

        # 一次性转移到CPU
        all_scores = torch.cat(all_scores_gpu).cpu().numpy()
        all_returns = torch.cat(all_returns_gpu).cpu().numpy()

        mask = ~(np.isnan(all_scores) | np.isnan(all_returns))
        if mask.sum() < 10:
            ic, rank_ic = 0.0, 0.0
        else:
            scores_valid = all_scores[mask]
            returns_valid = all_returns[mask]
            ic = np.corrcoef(scores_valid, returns_valid)[0, 1]
            rank_ic, _ = spearmanr(scores_valid, returns_valid)

        # 调制系数统计
        stats = {}
        if alphas_gpu:
            alphas_arr = torch.cat(alphas_gpu).cpu().numpy()
            stats['alpha_mean'] = float(np.mean(alphas_arr))
            stats['alpha_std'] = float(np.std(alphas_arr))
        if betas_gpu:
            betas_arr = torch.cat(betas_gpu).cpu().numpy()
            stats['beta_mean'] = float(np.mean(betas_arr))
            stats['beta_std'] = float(np.std(betas_arr))
        if gammas_gpu:
            gammas_arr = torch.cat(gammas_gpu).cpu().numpy()
            stats['gamma_mean'] = float(np.mean(gammas_arr))
            stats['gamma_std'] = float(np.std(gammas_arr))

        return ic, rank_ic, stats

    def _collect_v4_stats(self, loader: DataLoader) -> Dict[str, float]:
        """
        收集V4/V4.1/V4.2模型的宏观调制系数统计
        返回用于可视化的数据
        """
        self.model.eval()
        alphas, betas, gammas = [], [], []

        with torch.no_grad():
            for batch in loader:
                z_macro = batch['macro'].to(self.device, non_blocking=True)

                if hasattr(self.model, 'get_modulation_coefficients'):
                    coeffs = self.model.get_modulation_coefficients(z_macro)
                    alphas.append(coeffs['alpha_market'].squeeze())
                    betas.append(coeffs['beta_risk'].squeeze())
                    if 'gamma_heat' in coeffs:
                        gammas.append(coeffs['gamma_heat'].squeeze())
                elif hasattr(self.model, 'macro_modulation'):
                    scale = self.model.macro_modulation(z_macro)
                    alphas.append(scale.squeeze())

        # 一次性转移到CPU
        def _to_numpy(tensors):
            if not tensors:
                return np.array([])
            return torch.cat(tensors).cpu().numpy()

        stats = {
            'alpha_mean': float(np.mean(_to_numpy(alphas))) if alphas else 1.0,
            'alpha_std': float(np.std(_to_numpy(alphas))) if alphas else 0.0,
            'beta_mean': float(np.mean(_to_numpy(betas))) if betas else 1.0,
            'beta_std': float(np.std(_to_numpy(betas))) if betas else 0.0,
            'gamma_mean': float(np.mean(_to_numpy(gammas))) if gammas else 0.5,
            'gamma_std': float(np.std(_to_numpy(gammas))) if gammas else 0.0,
        }
        return stats

    def _compute_ic_extended(self, loader: DataLoader) -> Tuple[float, float]:
        """计算IC和Rank IC（保留兼容接口）"""
        ic, rank_ic, _ = self._compute_ic_and_stats(loader)
        return ic, rank_ic

    def _collect_landscape_epoch_data(
        self,
        valid_loader: DataLoader,
        epoch: int,
        valid_ic: float,
        max_batches: int = 50
    ):
        """收集每个epoch的Energy Landscape数据（优化：累积在GPU上）"""
        if self.landscape_recorder is None:
            return

        self.model.eval()

        # GPU上累积tensor
        gpu_scores, gpu_E_bull, gpu_E_bear, gpu_E_heat, gpu_E_total = [], [], [], [], []
        gpu_alpha, gpu_beta, gpu_gamma = [], [], []
        gpu_returns, gpu_features = [], []

        batch_count = 0

        with torch.no_grad():
            for batch in valid_loader:
                if batch_count >= max_batches:
                    break

                x_bull = batch['bull'].to(self.device, non_blocking=True)
                x_bear = batch['bear'].to(self.device, non_blocking=True)
                x_friction = batch['friction'].to(self.device, non_blocking=True)
                x_macro = batch['macro'].to(self.device, non_blocking=True)
                labels = batch.get('label')

                energy, components = self.model(
                    x_bull, x_bear, x_friction, x_macro,
                    return_components=True
                )

                gpu_scores.append(-energy.squeeze())
                gpu_E_bull.append(components.get('E_bull', torch.zeros_like(energy)).squeeze())
                gpu_E_bear.append(components.get('E_bear', torch.zeros_like(energy)).squeeze())
                E_heat_val = components.get('E_heat', components.get('E_friction', torch.zeros_like(energy)))
                gpu_E_heat.append(E_heat_val.squeeze())
                gpu_E_total.append(energy.squeeze())

                gpu_alpha.append(components.get('alpha_market', torch.ones_like(energy)).squeeze())
                gpu_beta.append(components.get('beta_risk', torch.ones_like(energy)).squeeze())
                gpu_gamma.append(components.get('gamma_heat', torch.zeros_like(energy)).squeeze())

                if labels is not None:
                    gpu_returns.append(labels.to(self.device, non_blocking=True))
                else:
                    gpu_returns.append(torch.full((energy.squeeze().shape[0],), float('nan'), device=self.device))

                gpu_features.append(torch.cat([x_bull, x_bear, x_friction, x_macro], dim=-1))

                batch_count += 1

        # 一次性转移到CPU并转为numpy
        def _cat_to_np(tensors):
            return torch.cat(tensors).cpu().numpy() if tensors else np.array([])

        all_scores = _cat_to_np(gpu_scores)
        all_E_bull = _cat_to_np(gpu_E_bull)
        all_E_bear = _cat_to_np(gpu_E_bear)
        all_E_heat = _cat_to_np(gpu_E_heat)
        all_E_total = _cat_to_np(gpu_E_total)
        all_alpha = _cat_to_np(gpu_alpha)
        all_beta = _cat_to_np(gpu_beta)
        all_gamma = _cat_to_np(gpu_gamma)
        all_returns = _cat_to_np(gpu_returns)
        all_features = torch.cat(gpu_features).cpu().numpy() if gpu_features else np.array([])

        # 确定市场状态
        ret_mean = np.nanmean(all_returns)
        if ret_mean > 0.02:
            market_state = 'bull'
        elif ret_mean < -0.02:
            market_state = 'bear'
        else:
            market_state = 'neutral'

        # 只在第一个epoch和最后一个epoch收集全量股票能量数据
        is_first_or_last = (epoch == 0 or epoch == self.config.training.stage2_epochs - 1)

        if is_first_or_last:
            self.landscape_recorder.record_batch_energy(
                E_bull=all_E_bull,
                E_bear=all_E_bear,
                E_heat=all_E_heat,
                E_total=all_E_total,
                alpha=all_alpha,
                beta=all_beta,
                gamma=all_gamma,
                returns=all_returns,
                market_state=market_state
            )

            self.landscape_recorder.record_batch_features(
                all_features, all_E_total, all_returns
            )

        self.landscape_recorder.record_market_coefficients(
            np.mean(all_alpha), np.mean(all_beta), np.mean(all_gamma), market_state
        )

        top_k = max(10, int(len(all_scores) * 0.2))
        top_indices = np.argsort(all_scores)[-top_k:]
        top_features = all_features[top_indices]
        top_energy_mean = all_E_total[top_indices].mean()
        
        # 记录训练轨迹
        self.landscape_recorder.record_train_epoch(
            epoch=epoch,
            top_features=top_features,
            top_energy_mean=top_energy_mean,
            valid_ic=valid_ic
        )
    
    def _collect_correlation_data(self, valid_loader: DataLoader, max_batches: int = 100):
        """
        【新增】收集Correlation Matrix数据
        
        在训练结束后调用
        """
        if self.correlation_analyzer is None:
            return
        
        self.model.eval()
        batch_count = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                if batch_count >= max_batches:
                    break
                
                x_bull = batch['bull'].to(self.device)
                x_bear = batch['bear'].to(self.device)
                x_friction = batch['friction'].to(self.device)
                x_macro = batch['macro'].to(self.device)
                labels = batch.get('label')
                
                # 获取能量分量
                energy, components = self.model(
                    x_bull, x_bear, x_friction, x_macro,
                    return_components=True
                )
                
                # 确定市场状态
                if labels is not None:
                    ret_mean = labels.numpy().mean()
                    if ret_mean > 0.02:
                        market_state = 'bull'
                    elif ret_mean < -0.02:
                        market_state = 'bear'
                    else:
                        market_state = 'neutral'
                else:
                    market_state = 'unknown'
                
                # 记录能量分量
                self.correlation_analyzer.record_energy_components(
                    E_bull=components.get('E_bull', torch.zeros_like(energy)).cpu().numpy(),
                    E_bear=components.get('E_bear', torch.zeros_like(energy)).cpu().numpy(),
                    E_heat=components.get('E_heat', components.get('E_friction', torch.zeros_like(energy))).cpu().numpy(),
                    E_total=energy.cpu().numpy(),
                    alpha=components.get('alpha_market', torch.ones_like(energy)).cpu().numpy(),
                    beta=components.get('beta_risk', torch.ones_like(energy)).cpu().numpy(),
                    gamma=components.get('gamma_heat', torch.zeros_like(energy)).cpu().numpy(),
                    returns=labels.numpy() if labels is not None else None,
                    market_state=market_state
                )
                
                # 记录特征组
                if batch_count < 50:
                    self.correlation_analyzer.record_feature_groups(
                        bull_features=x_bull.cpu().numpy(),
                        bear_features=x_bear.cpu().numpy(),
                        friction_features=x_friction.cpu().numpy(),
                        macro_features=x_macro.cpu().numpy()
                    )
                
                batch_count += 1
        
        # 记录MA-PI-GEBM预测
        all_scores = []
        with torch.no_grad():
            for batch in valid_loader:
                x_bull = batch['bull'].to(self.device)
                x_bear = batch['bear'].to(self.device)
                x_friction = batch['friction'].to(self.device)
                x_macro = batch['macro'].to(self.device)
                
                energy = self.model(x_bull, x_bear, x_friction, x_macro)
                scores = -energy.squeeze().cpu().numpy()
                all_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else [scores])
        
        self.correlation_analyzer.record_model_prediction('MA-PI-GEBM', np.array(all_scores))
        
        print(f"  ✓ 收集了 {len(self.correlation_analyzer.energy_data['E_bull'])} 条Correlation数据")
    
    def get_landscape_recorder(self) -> Optional['EnergyLandscapeDataRecorder']:
        """获取能量地形数据记录器"""
        return self.landscape_recorder
    
    def get_correlation_analyzer(self) -> Optional['CorrelationAnalyzer']:
        """获取correlation分析器"""
        return self.correlation_analyzer


class Stage3Trainer:
    """
    阶段三：Guardian训练（含难例挖掘）
    
    目的：训练外层风控模型，重点识别EBM的盲区
    """
    
    def __init__(self, config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
    
    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: List[str],
        ebm_model: GameEnergyModel,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        use_hard_mining: bool = True
    ) -> RiskGuardian:
        print("\n" + "=" * 60)
        print("    Stage 3: Guardian 训练")
        print("=" * 60)
        
        # 如果提供了特征分组，使用它；否则自动推断
        if feature_groups is None:
            feature_groups = self._infer_feature_groups(feature_cols, train_df)
        
        # 计算EBM能量分数
        energy_scores = self._compute_energy_scores(train_df, ebm_model, feature_groups)
        
        # 创建Guardian
        guardian = RiskGuardian(
            model_type=self.config.model.guardian_type,
            threshold=self.config.model.guardian_threshold
        )
        
        # 【关键修复】过滤只使用数据中存在的特征列
        available_cols = set(train_df.columns)
        valid_feature_cols = [c for c in feature_cols if c in available_cols]
        
        if use_hard_mining:
            guardian.fit_with_hard_mining(
                train_df, valid_df, valid_feature_cols, energy_scores, 'crash_label'
            )
        else:
            X_train = train_df[valid_feature_cols].fillna(0).values
            y_train = train_df['crash_label'].values
            X_valid = valid_df[valid_feature_cols].fillna(0).values
            y_valid = valid_df['crash_label'].values
            
            guardian.fit(X_train, y_train, X_valid, y_valid, valid_feature_cols)
        
        # 调整阈值（限制最大阈值为config中设置的值）
        X_valid = valid_df[valid_feature_cols].fillna(0).values
        y_valid = valid_df['crash_label'].values
        guardian.tune_threshold(
            X_valid, y_valid, 
            target_recall=self.config.model.guardian_target_recall,
            max_threshold=self.config.model.guardian_threshold  # 【新增】限制最大阈值
        )
        
        # 【新增】记录Guardian分布数据（用于可视化）
        if HAS_VISUALIZATION:
            guardian_recorder = GuardianDistributionRecorder()
            # 获取验证集的预测分数
            scores = guardian.predict_proba(X_valid)
            guardian_recorder.record(scores, y_valid)
            
            # 保存数据
            viz_dir = Path('outputs/viz_data')
            viz_dir.mkdir(parents=True, exist_ok=True)
            guardian_recorder.save(str(viz_dir / 'guardian_distribution.json'))
        
        print(f"\n  ✓ Stage 3 完成, 阈值: {guardian.threshold:.2f}")
        
        return guardian
    
    def _infer_feature_groups(self, feature_cols: List[str], df: pd.DataFrame) -> Dict[str, List[str]]:
        """自动推断特征分组"""
        # 按特征类型分组（与config.py保持一致）
        bull_cols = [c for c in feature_cols if any(x in c.lower() for x in 
                    ['momentum', 'rsi', 'north', 'main', 'roe', 'price_vs', 'volume_ratio', 
                     'revenue', 'profit', 'winner', 'support', 'clv', 'upside'])]
        bear_cols = [c for c in feature_cols if any(x in c.lower() for x in 
                    ['pe_rank', 'pb_rank', 'ps_rank', 'bias', 'trapped', 'margin_sell', 
                     'cost', 'debt', 'goodwill', 'resistance', 'avg_cost'])]
        friction_cols = [c for c in feature_cols if any(x in c.lower() for x in 
                    ['turnover', 'volatility', 'amplitude', 'asr', 'chip', 'vol_price', 'concentration'])]
        macro_cols = [c for c in df.columns if c.startswith('macro_embed')]
        
        # 如果某组为空，使用默认分配
        all_assigned = set(bull_cols + bear_cols + friction_cols + macro_cols)
        unassigned = [c for c in feature_cols if c not in all_assigned and not c.startswith('macro_embed')]
        
        if not bull_cols:
            bull_cols = unassigned[:max(1, len(unassigned)//3)]
            unassigned = unassigned[len(bull_cols):]
        if not bear_cols:
            bear_cols = unassigned[:max(1, len(unassigned)//2)]
            unassigned = unassigned[len(bear_cols):]
        if not friction_cols:
            friction_cols = unassigned if unassigned else ['volatility_20d'] if 'volatility_20d' in df.columns else []
        
        return {
            'bull': bull_cols,
            'bear': bear_cols,
            'friction': friction_cols,
            'macro': macro_cols
        }
    
    def _compute_energy_scores(
        self,
        df: pd.DataFrame,
        model: nn.Module,
        feature_groups: Dict[str, List[str]]
    ) -> np.ndarray:
        """
        计算EBM能量分数
        
        【重要修复】
        1. 修复维度不匹配问题（从模型获取正确维度）
        2. 添加NaN检测和处理
        3. 返回scores（-energy），越高越好
        """
        model.eval()
        expected_dims = resolve_model_input_dims(model, feature_groups)
        feature_arrays, valid_groups = prepare_feature_arrays(df, feature_groups, expected_dims)

        availability = describe_group_availability(feature_groups, valid_groups)
        print(
            "  特征可用性: "
            f"bull={availability['bull']}, bear={availability['bear']}, "
            f"friction={availability['friction']}, macro={availability['macro']}"
        )
        print(
            "  模型期望维度: "
            f"bull={expected_dims['bull']}, bear={expected_dims['bear']}, "
            f"friction={expected_dims['friction']}, macro={expected_dims['macro']}"
        )
        
        scores = []
        batch_size = 1000
        nan_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(df), batch_size):
                end = min(i + batch_size, len(df))
                x_bull, x_bear, x_friction, x_macro = slice_feature_tensors(
                    feature_arrays, i, end, self.device
                )
                
                energy = model(x_bull, x_bear, x_friction, x_macro)
                batch_scores = -energy.squeeze().cpu().numpy()
                
                # 检测输出NaN
                if np.any(np.isnan(batch_scores)):
                    nan_batches += 1
                    batch_scores = np.nan_to_num(batch_scores, nan=0.0)
                
                scores.append(batch_scores)
        
        result = np.concatenate(scores)
        
        # 最终NaN检查
        final_nan = np.isnan(result).sum()
        if final_nan > 0 or nan_batches > 0:
            print(f"  ⚠ 输出异常: NaN批次={nan_batches}, 最终NaN={final_nan}")
            result = np.nan_to_num(result, nan=0.0)
        
        # 输出统计
        print(f"  分数统计: min={result.min():.4f}, max={result.max():.4f}, "
              f"mean={result.mean():.4f}, std={result.std():.4f}")
        
        return result


class Stage4Integrator:
    """
    阶段四：全系统联合推断
    
    整合EBM和Guardian进行选股
    """
    
    def __init__(self, config, game_ebm: GameEnergyModel, guardian: RiskGuardian, device: str = 'cpu'):
        self.config = config
        self.game_ebm = game_ebm
        self.guardian = guardian
        self.device = torch.device(device)
        active_costs = self.config.backtest.get_costs(getattr(self.config, 'mode', 'stock'))
        self.single_limit = active_costs['single_limit']
        self.sector_limit = active_costs['sector_limit']
        
        # 【新增】模型版本（用于select_stocks中判断输出格式）
        model_cfg = config.model if hasattr(config, 'model') else config
        self.model_version = getattr(model_cfg, 'model_version', 'v2')
        
        # 【新增】宏观择时参数（从config读取）
        self.enable_macro_timing = getattr(model_cfg, 'enable_macro_timing', True)
        self.energy_threshold = getattr(model_cfg, 'timing_energy_threshold', 0.5)
        self.min_position_ratio = getattr(model_cfg, 'timing_min_position', 0.3)
        self.max_position_ratio = getattr(model_cfg, 'timing_max_position', 1.0)
    
    def _calculate_market_position(self, mean_energy: float, energy_std: float) -> float:
        """
        根据市场平均能量计算仓位比例
        
        逻辑：
        - 能量越高（越危险），仓位越低
        - 使用Sigmoid函数平滑过渡
        """
        if not self.enable_macro_timing:
            return 1.0
        
        # 标准化能量（相对于阈值）
        normalized = (self.energy_threshold - mean_energy) / (energy_std + 1e-6)
        
        # Sigmoid映射到[min_ratio, max_ratio]
        sigmoid = 1.0 / (1.0 + np.exp(-normalized))
        position_ratio = self.min_position_ratio + sigmoid * (self.max_position_ratio - self.min_position_ratio)
        
        return float(np.clip(position_ratio, self.min_position_ratio, self.max_position_ratio))
    
    def select_stocks(
        self,
        df: pd.DataFrame,
        bull_cols: List[str],
        bear_cols: List[str],
        friction_cols: List[str],
        macro_cols: List[str],
        guardian_cols: List[str] = None,
        top_k: int = 50,
        date_col: str = 'trade_date',
        code_col: str = 'ts_code',
        record_phase_space: bool = False  # 【新增】是否记录博弈相平面数据
    ) -> pd.DataFrame:
        """
        选股流程：
        1. EBM计算能量 → 选Top N
        2. Guardian过滤 → 剔除高风险
        3. 【新增】宏观择时 → 动态调整仓位
        4. Softmax权重分配
        """
        self.game_ebm.eval()
        
        # 【新增】初始化博弈相平面记录器
        if record_phase_space and HAS_VISUALIZATION:
            self.phase_space_recorder = GamePhaseSpaceRecorder()
        
        if guardian_cols is None:
            guardian_cols = bull_cols + bear_cols + friction_cols + macro_cols
        
        available_cols = set(df.columns)
        guardian_cols_valid = [c for c in guardian_cols if c in available_cols]
        expected_dims = resolve_model_input_dims(
            self.game_ebm,
            {
                'bull': bull_cols,
                'bear': bear_cols,
                'friction': friction_cols,
                'macro': macro_cols,
            },
        )
        
        results = []
        position_ratios = []  # 记录每日仓位比例
        
        for date, group in df.groupby(date_col):
            feature_arrays, _ = prepare_feature_arrays(
                group,
                {
                    'bull': bull_cols,
                    'bear': bear_cols,
                    'friction': friction_cols,
                    'macro': macro_cols,
                },
                expected_dims,
            )
            X_bull, X_bear, X_friction, X_macro = slice_feature_tensors(
                feature_arrays, 0, len(group), self.device
            )
            
            with torch.no_grad():
                # 【修复】根据模型版本处理不同的输出格式
                model_version = self.model_version  # 使用统一的版本标识
                
                if model_version == 'v4':
                    # V4/V4.1模型：使用return_components获取编码向量
                    energy, components = self.game_ebm(
                        X_bull, X_bear, X_friction, X_macro, 
                        return_components=True
                    )
                    raw_energy = energy.squeeze().cpu().numpy()
                    scores = -raw_energy
                    
                    # V4.1有E_bull/E_bear/E_friction
                    if 'E_bull' in components:
                        E_bull = components['E_bull'].cpu().numpy()
                        E_bear = components['E_bear'].cpu().numpy()
                        E_friction = components.get('E_friction', torch.zeros_like(energy)).squeeze().cpu().numpy()
                    else:
                        # 原版V4使用编码向量的L2范数
                        h_bull = components['h_bull']
                        h_bear = components['h_bear']
                        h_friction = components['h_friction']
                        
                        E_bull = torch.norm(h_bull, dim=-1).cpu().numpy()
                        E_bear = torch.norm(h_bear, dim=-1).cpu().numpy()
                        E_friction = torch.norm(h_friction, dim=-1).cpu().numpy()
                else:
                    # V1/V2/V3模型：使用return_components获取能量分解
                    energy, components = self.game_ebm(
                        X_bull, X_bear, X_friction, X_macro, 
                        return_components=True
                    )
                    raw_energy = energy.squeeze().cpu().numpy()
                    scores = -raw_energy
                    
                    # 提取能量组件（用于归因分析）
                    E_bull = components['E_bull'].squeeze().cpu().numpy()
                    E_bear = components['E_bear'].squeeze().cpu().numpy()
                    E_friction = components.get('E_pure_friction', 
                                 components.get('friction_heat', torch.zeros_like(energy))).squeeze().cpu().numpy()
            
            # 【新增】宏观择时：计算市场平均能量
            mean_energy = np.mean(raw_energy)
            energy_std = np.std(raw_energy) + 1e-6
            position_ratio = self._calculate_market_position(mean_energy, energy_std)
            position_ratios.append({'date': date, 'position_ratio': position_ratio, 'mean_energy': mean_energy})
            
            group = group.copy()
            group['ebm_score'] = scores
            group['raw_energy'] = raw_energy  # 【新增】原始能量
            # 【新增】添加能量组件列
            group['E_bull'] = E_bull
            group['E_bear'] = E_bear
            group['E_friction'] = E_friction
            
            # 【修改】根据仓位比例调整实际选股数量
            effective_top_k = max(1, int(top_k * position_ratio))
            
            # 选Top N
            top_n = min(effective_top_k * 2, len(group))
            top_indices = np.argsort(scores)[-top_n:]
            candidates = group.iloc[top_indices]
            
            # Guardian过滤
            # 【修复】检查是否为虚拟Guardian（enable_guardian=False时）
            is_dummy_guardian = getattr(self.guardian, '_is_dummy', False)
            
            if is_dummy_guardian:
                # 虚拟Guardian：所有股票都安全
                safe_mask = np.ones(len(candidates), dtype=bool)
            elif guardian_cols_valid:
                X_guardian = candidates[guardian_cols_valid].fillna(0).values
                # 【修复】处理Inf值
                X_guardian = np.nan_to_num(X_guardian, nan=0.0, posinf=0.0, neginf=0.0)
                safe_mask = self.guardian.get_safe_mask(X_guardian)
            else:
                X_guardian = np.zeros((len(candidates), 1))
                safe_mask = self.guardian.get_safe_mask(X_guardian)
            
            safe_stocks = candidates[safe_mask]
            
            if len(safe_stocks) > effective_top_k:
                safe_stocks = safe_stocks.nlargest(effective_top_k, 'ebm_score')
            
            # 【新增】记录博弈相平面数据
            if record_phase_space and HAS_VISUALIZATION and hasattr(self, 'phase_space_recorder'):
                # 创建选中股票的mask
                selected_codes = set(safe_stocks[code_col].values) if len(safe_stocks) > 0 else set()
                selected_mask = np.array([code in selected_codes for code in group[code_col].values])
                
                # 获取5日收益（如果存在）
                returns = group['return_5d'].values if 'return_5d' in group.columns else None
                
                self.phase_space_recorder.record_day(
                    date=str(date),
                    ts_codes=group[code_col].tolist(),
                    E_bull=E_bull,
                    E_bear=E_bear,
                    E_friction=E_friction,
                    total_energy=raw_energy,
                    selected_mask=selected_mask,
                    returns=returns
                )
            
            # 分配权重
            if len(safe_stocks) > 0:
                scores_safe = safe_stocks['ebm_score'].values
                
                # 【新增】波动率倒数加权（风险平价思想）
                if 'volatility_20d_zscore' in safe_stocks.columns:
                    vol = safe_stocks['volatility_20d_zscore'].values
                    vol = np.abs(vol) + 0.1  # 防止除零，并转换为正值
                    vol_inv_weights = 1.0 / vol
                    vol_inv_weights = vol_inv_weights / vol_inv_weights.sum()
                elif 'volatility_20d' in safe_stocks.columns:
                    vol = safe_stocks['volatility_20d'].values
                    vol = np.abs(vol) + 0.01  # 防止除零
                    vol_inv_weights = 1.0 / vol
                    vol_inv_weights = vol_inv_weights / vol_inv_weights.sum()
                else:
                    vol_inv_weights = np.ones(len(safe_stocks)) / len(safe_stocks)
                
                # 【修改】结合EBM分数和波动率倒数
                scores_safe = np.clip(scores_safe, -10, 10)
                score_weights = np.exp(scores_safe) / np.exp(scores_safe).sum()
                
                # 混合权重：从config读取比例
                alpha = getattr(self.config.model, 'weight_score_ratio', 0.7)
                weights = alpha * score_weights + (1 - alpha) * vol_inv_weights
                
                # 【新增】行业限制（如果有行业数据）
                if 'sw_industry' in safe_stocks.columns or 'industry' in safe_stocks.columns:
                    industry_col = 'sw_industry' if 'sw_industry' in safe_stocks.columns else 'industry'
                    industry_limit = self.sector_limit
                    
                    safe_stocks_temp = safe_stocks.copy()
                    safe_stocks_temp['raw_weight'] = weights
                    
                    # 按行业分组，限制权重
                    for industry in safe_stocks_temp[industry_col].unique():
                        mask = safe_stocks_temp[industry_col] == industry
                        industry_weight = safe_stocks_temp.loc[mask, 'raw_weight'].sum()
                        
                        if industry_weight > industry_limit:
                            # 按比例缩减该行业的权重
                            scale = industry_limit / industry_weight
                            safe_stocks_temp.loc[mask, 'raw_weight'] *= scale
                    
                    weights = safe_stocks_temp['raw_weight'].values
                    weights = weights / weights.sum()  # 重新归一化
                
                weights = np.minimum(weights, self.single_limit)
                weights = weights / weights.sum()
                
                # 【新增】应用仓位比例（剩余部分假设持有现金）
                weights = weights * position_ratio
                
                safe_stocks = safe_stocks.copy()
                safe_stocks['weight'] = weights
                safe_stocks['position_ratio'] = position_ratio  # 记录仓位比例
                
                results.append(safe_stocks)
        
        # 【新增】输出仓位统计
        if position_ratios:
            ratios_df = pd.DataFrame(position_ratios)
            avg_ratio = ratios_df['position_ratio'].mean()
            min_ratio = ratios_df['position_ratio'].min()
            if avg_ratio < 0.95:  # 只在有显著择时时输出
                print(f"  宏观择时: 平均仓位={avg_ratio:.1%}, 最低仓位={min_ratio:.1%}")
        
        # 【新增】保存博弈相平面数据
        if record_phase_space and HAS_VISUALIZATION and hasattr(self, 'phase_space_recorder'):
            viz_dir = Path('outputs/viz_data')
            viz_dir.mkdir(parents=True, exist_ok=True)
            self.phase_space_recorder.save(str(viz_dir / 'game_phase_space.csv'))
        
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
    
    def get_portfolio(self, df: pd.DataFrame, date, **kwargs) -> Dict[str, float]:
        """获取指定日期的投资组合"""
        df_day = df[df['trade_date'] == date]
        selected = self.select_stocks(df_day, **kwargs)
        
        if selected.empty:
            return {}
        
        return dict(zip(selected['ts_code'], selected['weight']))


class CurriculumTrainer:
    """
    课程学习管理器
    
    协调四个阶段的训练
    
    【新增】训练结束后自动生成：
    - Energy Landscape可视化（能量地形图）
    - Correlation Matrix可视化（相关性矩阵）
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
        
        self.stage1_trainer = None
        self.stage2_trainer = None
        self.stage3_trainer = None
        
        self.game_ebm = None
        self.guardian = None
        self.integrator = None
        
        # 【新增】可视化数据记录器
        self.landscape_recorder = None
        self.correlation_analyzer = None
    
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        pairwise_loader: DataLoader,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_dims: Dict[str, int],
        feature_cols: List[str],
        feature_groups: Optional[Dict[str, List[str]]] = None,
        skip_stage1: bool = False,
        use_ic_loss: bool = True,  # 是否使用IC Loss
        enable_landscape: bool = True,  # 【新增】是否启用能量地形图
        enable_correlation: bool = True  # 【新增】是否启用相关性分析
    ) -> Stage4Integrator:
        """执行完整课程学习"""
        print("\n" + "=" * 70)
        print("    宏观感知物理博弈能量模型 - 课程学习训练")
        print("=" * 70)
        
        # 保存特征分组供后续使用
        self.feature_groups = feature_groups
        
        # Stage 1
        if not skip_stage1:
            self.stage1_trainer = Stage1Trainer(self.config, str(self.device))
            passed, ic = self.stage1_trainer.train(train_loader, valid_loader, feature_dims)
            
            if not passed:
                print("\n警告: Stage 1 未通过，建议检查数据质量")
        
        # Stage 2 【改进】使用IC Loss + 传入可视化参数
        self.stage2_trainer = Stage2Trainer(self.config, str(self.device))
        self.game_ebm = self.stage2_trainer.train(
            pairwise_loader, valid_loader, feature_dims,
            use_ic_loss=use_ic_loss,
            feature_groups=feature_groups,
            enable_landscape=enable_landscape,  # 传入能量地形图参数
            aux_loader=train_loader,
        )
        
        # 【新增】获取可视化记录器
        if enable_landscape and HAS_ENERGY_LANDSCAPE:
            self.landscape_recorder = self.stage2_trainer.get_landscape_recorder()
        if enable_correlation and HAS_CORRELATION:
            self.correlation_analyzer = self.stage2_trainer.get_correlation_analyzer()
        
        # Stage 3 【修复】检查enable_guardian配置
        enable_guardian = getattr(self.config.training, 'enable_guardian', True)
        if enable_guardian:
            self.stage3_trainer = Stage3Trainer(self.config, str(self.device))
            self.guardian = self.stage3_trainer.train(
                train_df, valid_df, feature_cols, self.game_ebm,
                feature_groups=self.feature_groups, use_hard_mining=True
            )
        else:
            print("\n>>> Guardian已禁用 (enable_guardian=False)")
            # 创建一个空的Guardian（总是返回安全）
            from models.risk_guardian import RiskGuardian
            self.guardian = RiskGuardian(model_type='lightgbm', threshold=0.0)
            self.guardian._is_dummy = True  # 标记为虚拟Guardian
        
        # Stage 4
        self.integrator = Stage4Integrator(
            self.config, self.game_ebm, self.guardian, str(self.device)
        )
        
        # 【新增】生成Energy Landscape图表
        if enable_landscape and self.landscape_recorder is not None and HAS_ENERGY_LANDSCAPE:
            print("\n>>> 生成Energy Landscape可视化...")
            visualizer = EnergyLandscapeVisualizer(output_dir='outputs/landscape')
            visualizer.load_data(self.landscape_recorder.data_dir)
            visualizer.plot_all()
        
        # 【新增】生成Correlation Matrix图表
        if enable_correlation and self.correlation_analyzer is not None:
            print("\n>>> 生成Correlation Matrix可视化...")
            self.correlation_analyzer.save_data()
            self.correlation_analyzer.plot_all()
        
        print("\n" + "=" * 70)
        print("    ✓ 课程学习完成")
        print("=" * 70)
        
        return self.integrator
    
    def save(self, save_dir: Path):
        """保存模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存EBM
        torch.save(self.game_ebm.state_dict(), save_dir / 'game_ebm.pt')
        
        # 保存Guardian
        self.guardian.save(save_dir / 'guardian')
        
        # 保存配置
        self.config.save(save_dir / 'config.json')
        
        print(f"✓ 模型保存至: {save_dir}")
    
    @classmethod
    def load(cls, save_dir: Path, config) -> 'CurriculumTrainer':
        """加载模型"""
        save_dir = Path(save_dir)
        
        trainer = cls(config)
        
        # 加载Guardian
        trainer.guardian = RiskGuardian.load(save_dir / 'guardian')
        
        print(f"✓ 模型加载自: {save_dir}")
        
        return trainer
