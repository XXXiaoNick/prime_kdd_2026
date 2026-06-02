"""
================================================================================
核心模型组件 - 宏观感知物理博弈能量模型 (v2)
================================================================================

修复与改进内容：
1. 解决摩擦力"零乘效应"问题 - 添加独立摩擦项
2. 支持预计算宏观Embedding
3. 增强能量方程的物理可解释性
4. 【新增】支持多种宏观调制方式：Concat / FiLM / Hypernetwork

宏观调制方式说明：
- Concat (原始): 简单拼接，宏观作为额外输入 f(x, z) = g([x; z])
- FiLM (推荐): 特征级调制 h' = γ(z) ⊙ h + β(z)
- Hypernetwork: 宏观生成网络权重 W = W_base + ΔW(z)

理论动机：
- 拼接方式：环境是输入变量
- FiLM/Hyper：环境改变物理法则（牛市中动量重要，熊市中估值重要）
================================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Part 1: 基础组件
# ============================================================================

class MacroEncoder(nn.Module):
    """
    宏观环境编码器 - 用于sequence模式
    
    输入：[batch, seq_len, macro_dim]
    输出：[batch, embed_dim]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embed_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, macro_seq: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(macro_seq)
        return self.output_proj(hidden[-1])


class MacroEmbeddingProjector(nn.Module):
    """
    宏观Embedding投影器 - 用于embedding模式（预计算）
    
    输入：[batch, macro_embed_dim] 预计算的embedding
    输出：[batch, embed_dim]
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class FrictionEncoder(nn.Module):
    """摩擦力编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# Part 2: 原始力场网络 (Concat方式)
# ============================================================================

class ForceNetwork(nn.Module):
    """力场网络基类 - Concat方式"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        macro_dim: int = 0,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.macro_dim = macro_dim
        total_input = input_dim + macro_dim
        
        layers = []
        prev_dim = total_input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, z_macro: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.macro_dim > 0 and z_macro is not None:
            x = torch.cat([x, z_macro], dim=-1)
        elif self.macro_dim > 0:
            z_macro = torch.zeros(x.size(0), self.macro_dim, device=x.device)
            x = torch.cat([x, z_macro], dim=-1)
        
        return self.network(x)


class BullNet(ForceNetwork):
    """
    多头动力网络
    
    物理含义：计算推动价格上涨的"动力势能"
    高动力 = 强上涨潜力 = 低能量态（好）
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 macro_dim: int = 0, dropout: float = 0.2):
        super().__init__(input_dim, hidden_dims, macro_dim, dropout)
        self.activation = nn.Softplus()
    
    def forward(self, x_bull: torch.Tensor, z_macro: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw = super().forward(x_bull, z_macro)
        return self.activation(raw)


class BearNet(ForceNetwork):
    """
    空头阻力网络
    
    物理含义：计算抑制价格上涨的"阻力势能"
    高阻力 = 强下跌压力 = 高能量态（差）
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 macro_dim: int = 0, dropout: float = 0.2):
        super().__init__(input_dim, hidden_dims, macro_dim, dropout)
        self.activation = nn.Softplus()
    
    def forward(self, x_bear: torch.Tensor, z_macro: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw = super().forward(x_bear, z_macro)
        return self.activation(raw)


# ============================================================================
# Part 3: FiLM调制组件 (Feature-wise Linear Modulation)
# ============================================================================

class FiLMGenerator(nn.Module):
    """
    FiLM参数生成器
    
    从宏观Embedding生成调制参数 (γ, β)
    h' = γ ⊙ h + β
    
    参数量：O(macro_dim × hidden_dim) 远小于 Hypernetwork
    """
    
    def __init__(self, macro_dim: int, hidden_dim: int):
        super().__init__()
        
        self.film_gen = nn.Sequential(
            nn.Linear(macro_dim, macro_dim * 2),
            nn.GELU(),
            nn.Linear(macro_dim * 2, hidden_dim * 2)
        )
        
        # 初始化：γ≈1, β≈0 (接近恒等变换)
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        self.film_gen[-1].bias.data[:hidden_dim] = 1.0  # γ初始化为1
    
    def forward(self, z_macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_macro: [batch, macro_dim]
        Returns:
            gamma: [batch, hidden_dim] - 缩放因子
            beta: [batch, hidden_dim] - 偏移因子
        """
        film_params = self.film_gen(z_macro)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma, beta


class FiLMModulatedLayer(nn.Module):
    """
    FiLM调制的网络层
    
    h' = γ ⊙ LayerNorm(Linear(x)) + β
    """
    
    def __init__(self, in_dim: int, out_dim: int, macro_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.film_gen = FiLMGenerator(macro_dim, out_dim)
    
    def forward(self, x: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.linear(x))
        gamma, beta = self.film_gen(z_macro)
        h = gamma * h + beta
        return self.dropout(F.gelu(h))


class FiLMForceNetwork(nn.Module):
    """
    FiLM调制的力场网络
    
    每一层都受宏观条件调制，实现"环境改变物理法则"
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        macro_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(
                FiLMModulatedLayer(prev_dim, hidden_dim, macro_dim, dropout)
            )
            prev_dim = hidden_dim
        
        self.output = nn.Linear(prev_dim, 1)
        self.activation = nn.Softplus()
    
    def forward(self, x: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, z_macro)
        return self.activation(self.output(h))


# ============================================================================
# Part 4: Hypernetwork调制组件
# ============================================================================

class LowRankHypernetwork(nn.Module):
    """
    低秩Hypernetwork
    
    生成权重增量：ΔW = U(z) @ V.T
    最终权重：W = W_base + α * ΔW
    
    类似LoRA的思想，参数量可控
    """
    
    def __init__(
        self,
        macro_dim: int,
        weight_shape: Tuple[int, int],
        rank: int = 8,
        scale: float = 0.1
    ):
        super().__init__()
        
        out_features, in_features = weight_shape
        
        self.hyper_u = nn.Linear(macro_dim, out_features * rank)
        self.hyper_v = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features
        self.scale = scale
        
        nn.init.normal_(self.hyper_u.weight, std=0.01)
        nn.init.zeros_(self.hyper_u.bias)
    
    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        batch_size = z_macro.size(0)
        U = self.hyper_u(z_macro).view(batch_size, self.out_features, self.rank)
        delta_W = torch.bmm(U, self.hyper_v.T.unsqueeze(0).expand(batch_size, -1, -1))
        return self.scale * delta_W


class HypernetworkLayer(nn.Module):
    """
    Hypernetwork调制的网络层
    
    W_effective = W_base + ΔW(z_macro)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        macro_dim: int,
        rank: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.weight_base = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.hyper = LowRankHypernetwork(macro_dim, (out_dim, in_dim), rank)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.weight_base, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        delta_W = self.hyper(z_macro)
        W_effective = self.weight_base.unsqueeze(0) + delta_W
        h = torch.bmm(W_effective, x.unsqueeze(-1)).squeeze(-1) + self.bias
        return self.dropout(F.gelu(self.norm(h)))


class HyperForceNetwork(nn.Module):
    """
    Hypernetwork调制的力场网络
    
    网络权重本身由宏观条件生成，实现真正的"环境改变物理法则"
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        macro_dim: int,
        rank: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(
                HypernetworkLayer(prev_dim, hidden_dim, macro_dim, rank, dropout)
            )
            prev_dim = hidden_dim
        
        self.output = nn.Linear(prev_dim, 1)
        self.activation = nn.Softplus()
    
    def forward(self, x: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, z_macro)
        return self.activation(self.output(h))


# ============================================================================
# Part 5: 原始博弈能量模型 (Concat方式)
# ============================================================================

class GameEnergyModel(nn.Module):
    """
    博弈能量模型 (修复版 v3) - 使用Concat方式
    
    【修复】摩擦力的"零乘效应"问题
    【新增】Bear权重（"刹车"机制）
    
    修正后的能量方程：
    E = bear_weight * E_bear - E_bull + λ1 * |E_bull * E_bear| * friction + λ2 * friction
    
    bear_weight > 1.0 会让模型更保守（更重视阻力信号）
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_dim: int = 0,
        bull_hidden: List[int] = [128, 64, 32],
        bear_hidden: List[int] = [128, 64, 32],
        interaction_lambda: float = 0.1,
        pure_friction_lambda: float = 0.05,
        bear_weight: float = 1.0,  # 【新增】Bear权重
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.interaction_lambda = interaction_lambda
        self.pure_friction_lambda = pure_friction_lambda
        self.bear_weight = bear_weight  # 【新增】
        self.macro_dim = macro_dim
        
        self.bull_net = BullNet(bull_dim, bull_hidden, macro_dim, dropout)
        self.bear_net = BearNet(bear_dim, bear_hidden, macro_dim, dropout)
        self.friction_encoder = FrictionEncoder(friction_dim)
        
        self.interaction_weight = nn.Parameter(torch.tensor(interaction_lambda))
        self.pure_friction_weight = nn.Parameter(torch.tensor(pure_friction_lambda))
        # 【新增】可学习的Bear权重
        self.bear_weight_param = nn.Parameter(torch.tensor(bear_weight))
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        z_macro: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        # 输入已在Dataset层清洗，仅做一次安全检查
        E_bull = self.bull_net(x_bull, z_macro)
        E_bear = self.bear_net(x_bear, z_macro)
        friction_heat = self.friction_encoder(x_friction)
        
        # 【修改】应用Bear权重（"刹车"）
        E_base = self.bear_weight_param * E_bear - E_bull
        interaction_term = torch.abs(E_bull * E_bear) * friction_heat
        E_interaction = self.interaction_weight * interaction_term
        E_pure_friction = self.pure_friction_weight * friction_heat
        
        energy = E_base + E_interaction + E_pure_friction
        
        energy = E_base + E_interaction + E_pure_friction

        if return_components:
            return energy, {
                'E_bull': E_bull,
                'E_bear': E_bear,
                'E_base': E_base,
                'E_interaction': E_interaction,
                'E_pure_friction': E_pure_friction,
                'friction_heat': friction_heat,
                'bear_weight': self.bear_weight_param.item(),  # 【新增】
            }
        
        return energy
    
    def get_energy_score(self, x_bull, x_bear, x_friction, z_macro=None) -> torch.Tensor:
        return -self.forward(x_bull, x_bear, x_friction, z_macro)


# ============================================================================
# Part 6: 改进版博弈能量模型 (支持多种调制方式)
# ============================================================================

class ImprovedGameEnergyModel(nn.Module):
    """
    改进版物理博弈能量模型
    
    支持三种宏观调制方式：
    1. 'concat': 简单拼接 (原始方式)
    2. 'film': FiLM调制 (推荐)
    3. 'hyper': Hypernetwork (实验)
    
    能量方程：
    E = E_bear - E_bull + λ1 * |E_bull * E_bear| * heat + λ2 * heat
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        modulation_type: str = 'film',  # 'concat', 'film', 'hyper'
        hyper_rank: int = 8,
        interaction_lambda: float = 0.1,
        pure_friction_lambda: float = 0.05,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.modulation_type = modulation_type
        self.macro_dim = macro_dim
        
        # 宏观Embedding投影
        self.macro_proj = nn.Sequential(
            nn.Linear(macro_dim, 64),
            nn.GELU(),
            nn.Linear(64, macro_dim)
        )
        
        # 根据调制类型选择网络结构
        if modulation_type == 'concat':
            self.bull_net = self._make_concat_network(bull_dim + macro_dim, hidden_dims, dropout)
            self.bear_net = self._make_concat_network(bear_dim + macro_dim, hidden_dims, dropout)
        elif modulation_type == 'film':
            self.bull_net = FiLMForceNetwork(bull_dim, hidden_dims, macro_dim, dropout)
            self.bear_net = FiLMForceNetwork(bear_dim, hidden_dims, macro_dim, dropout)
        elif modulation_type == 'hyper':
            self.bull_net = HyperForceNetwork(bull_dim, hidden_dims, macro_dim, hyper_rank, dropout)
            self.bear_net = HyperForceNetwork(bear_dim, hidden_dims, macro_dim, hyper_rank, dropout)
        else:
            raise ValueError(f"Unknown modulation_type: {modulation_type}")
        
        # 摩擦力网络
        self.friction_net = nn.Sequential(
            nn.Linear(friction_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.interaction_weight = nn.Parameter(torch.tensor(interaction_lambda))
        self.pure_friction_weight = nn.Parameter(torch.tensor(pure_friction_lambda))
        
        self._init_weights()
    
    def _make_concat_network(self, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        x_macro: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        z_macro = self.macro_proj(x_macro)
        
        if self.modulation_type == 'concat':
            bull_input = torch.cat([x_bull, z_macro], dim=-1)
            bear_input = torch.cat([x_bear, z_macro], dim=-1)
            E_bull = self.bull_net(bull_input)
            E_bear = self.bear_net(bear_input)
        else:
            E_bull = self.bull_net(x_bull, z_macro)
            E_bear = self.bear_net(x_bear, z_macro)
        
        heat = self.friction_net(x_friction)
        
        E_base = E_bear - E_bull
        E_interaction = self.interaction_weight * torch.abs(E_bull * E_bear) * heat
        E_pure_friction = self.pure_friction_weight * heat
        
        energy = E_base + E_interaction + E_pure_friction
        
        if return_components:
            return energy, {
                'E_bull': E_bull,
                'E_bear': E_bear,
                'E_interaction': E_interaction,
                'E_pure_friction': E_pure_friction,
                'heat': heat,
                'z_macro': z_macro
            }
        
        return energy
    
    def get_modulation_analysis(
        self,
        x_bull: torch.Tensor,
        z_macro_bull: torch.Tensor,
        z_macro_bear: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """分析宏观调制效果"""
        if self.modulation_type == 'concat':
            bull_input_1 = torch.cat([x_bull, z_macro_bull], dim=-1)
            bull_input_2 = torch.cat([x_bull, z_macro_bear], dim=-1)
            E_bull_bull = self.bull_net(bull_input_1)
            E_bull_bear = self.bull_net(bull_input_2)
        else:
            E_bull_bull = self.bull_net(x_bull, z_macro_bull)
            E_bull_bear = self.bull_net(x_bull, z_macro_bear)
        
        return {
            'E_bull_in_bull_market': E_bull_bull,
            'E_bull_in_bear_market': E_bull_bear,
            'modulation_effect': E_bull_bull - E_bull_bear
        }


# ============================================================================
# Part 7: 完整的宏观条件模型
# ============================================================================

class MacroConditionedGameEBM(nn.Module):
    """
    完整的宏观条件博弈能量模型 (修复版)
    
    支持两种模式：
    - embedding模式：输入预计算的macro embedding向量
    - sequence模式：输入宏观序列，使用GRU编码
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_input_dim: int,
        macro_embed_dim: int = 32,
        macro_mode: str = 'embedding',
        macro_seq_len: int = 12,
        macro_hidden_dim: int = 64,
        bull_hidden: List[int] = [128, 64, 32],
        bear_hidden: List[int] = [128, 64, 32],
        interaction_lambda: float = 0.1,
        pure_friction_lambda: float = 0.05,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.macro_mode = macro_mode
        self.macro_embed_dim = macro_embed_dim
        
        if macro_mode == 'sequence':
            self.macro_encoder = MacroEncoder(
                macro_input_dim, macro_hidden_dim, 2, macro_embed_dim, dropout
            )
        else:
            self.macro_encoder = MacroEmbeddingProjector(
                macro_input_dim, macro_embed_dim, dropout
            )
        
        self.game_ebm = GameEnergyModel(
            bull_dim, bear_dim, friction_dim, macro_embed_dim,
            bull_hidden, bear_hidden,
            interaction_lambda, pure_friction_lambda, dropout
        )
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        macro_input: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        z_macro = self.macro_encoder(macro_input)
        return self.game_ebm(x_bull, x_bear, x_friction, z_macro, return_components)
    
    def encode_macro(self, macro_input: torch.Tensor) -> torch.Tensor:
        return self.macro_encoder(macro_input)


# ============================================================================
# Part 8: 简化版模型 (用于Sanity Check)
# ============================================================================

class SimplifiedEBM(nn.Module):
    """简化版EBM - 用于阶段一Sanity Check"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# Part 9: 消融实验用的EBM变体
# ============================================================================

class EBMNoFriction(nn.Module):
    """无摩擦项的EBM: E = E_bear - E_bull"""
    
    def __init__(self, bull_dim: int, bear_dim: int, macro_dim: int, 
                 hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        
        self.bull_net = self._make_network(bull_dim + macro_dim, hidden_dims, dropout)
        self.bear_net = self._make_network(bear_dim + macro_dim, hidden_dims, dropout)
    
    def _make_network(self, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())
        return nn.Sequential(*layers)
    
    def forward(self, x_bull, x_bear, x_friction, x_macro, return_components=False):
        bull_input = torch.cat([x_bull, x_macro], dim=-1)
        bear_input = torch.cat([x_bear, x_macro], dim=-1)
        
        E_bull = self.bull_net(bull_input)
        E_bear = self.bear_net(bear_input)
        
        energy = E_bear - E_bull
        
        if return_components:
            return energy, {'E_bull': E_bull, 'E_bear': E_bear, 'E_friction': torch.zeros_like(E_bull)}
        return energy


class EBMNoMacro(nn.Module):
    """无宏观条件的EBM"""
    
    def __init__(self, bull_dim: int, bear_dim: int, friction_dim: int,
                 hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        
        self.bull_net = self._make_network(bull_dim, hidden_dims, dropout)
        self.bear_net = self._make_network(bear_dim, hidden_dims, dropout)
        self.friction_net = nn.Sequential(
            nn.Linear(friction_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.interaction_weight = nn.Parameter(torch.tensor(0.1))
        self.pure_friction_weight = nn.Parameter(torch.tensor(0.05))
    
    def _make_network(self, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())
        return nn.Sequential(*layers)
    
    def forward(self, x_bull, x_bear, x_friction, x_macro, return_components=False):
        E_bull = self.bull_net(x_bull)
        E_bear = self.bear_net(x_bear)
        heat = self.friction_net(x_friction)
        
        E_interaction = self.interaction_weight * torch.abs(E_bull * E_bear) * heat
        E_pure_friction = self.pure_friction_weight * heat
        
        energy = E_bear - E_bull + E_interaction + E_pure_friction
        
        if return_components:
            return energy, {'E_bull': E_bull, 'E_bear': E_bear, 'E_friction': E_interaction + E_pure_friction}
        return energy


class EBMSingleFriction(nn.Module):
    """单摩擦项EBM: E = E_bear - E_bull + λ * |E_bull * E_bear| * heat"""
    
    def __init__(self, bull_dim: int, bear_dim: int, friction_dim: int, macro_dim: int,
                 hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        
        self.bull_net = self._make_network(bull_dim + macro_dim, hidden_dims, dropout)
        self.bear_net = self._make_network(bear_dim + macro_dim, hidden_dims, dropout)
        self.friction_net = nn.Sequential(
            nn.Linear(friction_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.interaction_weight = nn.Parameter(torch.tensor(0.1))
    
    def _make_network(self, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())
        return nn.Sequential(*layers)
    
    def forward(self, x_bull, x_bear, x_friction, x_macro, return_components=False):
        bull_input = torch.cat([x_bull, x_macro], dim=-1)
        bear_input = torch.cat([x_bear, x_macro], dim=-1)
        
        E_bull = self.bull_net(bull_input)
        E_bear = self.bear_net(bear_input)
        heat = self.friction_net(x_friction)
        
        E_friction = self.interaction_weight * torch.abs(E_bull * E_bear) * heat
        energy = E_bear - E_bull + E_friction
        
        if return_components:
            return energy, {'E_bull': E_bull, 'E_bear': E_bear, 'E_friction': E_friction}
        return energy


# ============================================================================
# Part 10: 测试代码
# ============================================================================

# ============================================================================
# Part 9: 改进版博弈能量模型 V2 - 基于物理场优化
# ============================================================================
# 
# 核心改进（基于用户建议的批判性分析）：
# 1. 宏观环境作为"介质系数"：α_market调制Bull，β_risk调制Bear
# 2. BearNet使用"刚性"激活：Softplus(β=5)，让阻力有指数特性
# 3. 加入势能陷阱正则化：惩罚高波动率股票
# 4. 支持Pointwise回归损失：直接拟合收益率
#
# ============================================================================

class MacroModulationNetwork(nn.Module):
    """
    宏观调制网络 - 将宏观环境转换为物理介质系数
    
    核心思想：宏观环境不是简单的输入特征，而是改变力的传导效率
    
    输出：
    - α_market: 市场情绪系数，调制Bull的影响力
    - β_risk: 风险厌恶系数，调制Bear的影响力
    
    物理解释：
    - 牛市：α > 1, β < 1 → 动力放大，阻力减小
    - 熊市：α < 1, β > 1 → 动力减小，阻力放大
    """
    
    def __init__(
        self,
        macro_dim: int,
        hidden_dim: int = 32,
        min_coef: float = 0.3,  # 最小系数（防止完全消失）
        max_coef: float = 3.0,  # 最大系数（防止爆炸）
    ):
        super().__init__()
        
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.coef_range = max_coef - min_coef
        
        # 从宏观特征生成调制系数
        self.modulation_net = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # 输出 [α_market, β_risk]
        )
        
        # 初始化：让α和β初始接近1
        self._init_weights()
    
    def _init_weights(self):
        # 最后一层初始化为接近0，让sigmoid输出接近0.5
        nn.init.zeros_(self.modulation_net[-1].weight)
        nn.init.zeros_(self.modulation_net[-1].bias)
    
    def forward(self, z_macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_macro: [batch, macro_dim] 宏观Embedding
        
        Returns:
            alpha_market: [batch, 1] 市场情绪系数
            beta_risk: [batch, 1] 风险厌恶系数
        """
        raw = self.modulation_net(z_macro)
        
        # 使用sigmoid限制在[min_coef, max_coef]范围
        coefficients = torch.sigmoid(raw) * self.coef_range + self.min_coef
        
        alpha_market = coefficients[:, 0:1]  # [batch, 1]
        beta_risk = coefficients[:, 1:2]      # [batch, 1]
        
        return alpha_market, beta_risk


class RigidBearNet(nn.Module):
    """
    刚性阻力网络 - 使用指数特性的激活函数
    
    核心改进：当估值过高时，阻力应呈指数级上升，而不是线性上升
    
    【修复】添加正偏置确保输出不为0
    - 问题：Softplus(β=5)当输入为负时输出接近0
    - 解决：给最后一层添加正偏置，确保raw输出不会太负
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],  # 简化网络结构
        macro_dim: int = 0,
        dropout: float = 0.3,
        rigidity_beta: float = 5.0,  # 刚性系数，越大越"硬"
        min_output: float = 0.1  # 最小输出值
    ):
        super().__init__()
        
        self.macro_dim = macro_dim
        self.rigidity_beta = rigidity_beta
        self.min_output = min_output
        
        total_input = input_dim + macro_dim
        
        layers = []
        prev_dim = total_input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 最后一层
        self.output_layer = nn.Linear(prev_dim, 1)
        self.network = nn.Sequential(*layers)
        
        # 【修复】使用较温和的Softplus
        # beta=1更稳定，然后通过缩放实现"刚性"效果
        self.base_activation = nn.Softplus(beta=1.0)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用xavier初始化，输出更稳定
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 【关键】给输出层添加正偏置，确保初始输出为正
        nn.init.constant_(self.output_layer.bias, 0.5)
    
    def forward(self, x_bear: torch.Tensor, z_macro: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.macro_dim > 0 and z_macro is not None:
            x = torch.cat([x_bear, z_macro], dim=-1)
        elif self.macro_dim > 0:
            z_macro = torch.zeros(x_bear.size(0), self.macro_dim, device=x_bear.device)
            x = torch.cat([x_bear, z_macro], dim=-1)
        else:
            x = x_bear
        
        h = self.network(x)
        raw = self.output_layer(h)
        
        # 使用Softplus确保输出为正，再加上最小值确保不为0
        output = self.base_activation(raw) + self.min_output
        
        # 【可选】应用刚性缩放：当输出大时，指数放大
        if self.rigidity_beta > 1.0:
            # 对于大于1的输出，进行指数放大
            large_mask = output > 1.0
            output = torch.where(
                large_mask,
                1.0 + (output - 1.0) ** self.rigidity_beta,
                output
            )
        
        return output


class SimplifiedBullNet(nn.Module):
    """
    简化的动力网络
    
    使用更简单的结构防止过拟合
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        macro_dim: int = 0,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.macro_dim = macro_dim
        total_input = input_dim + macro_dim
        
        layers = []
        prev_dim = total_input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self.activation = nn.Softplus()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_bull: torch.Tensor, z_macro: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.macro_dim > 0 and z_macro is not None:
            x = torch.cat([x_bull, z_macro], dim=-1)
        elif self.macro_dim > 0:
            z_macro = torch.zeros(x_bull.size(0), self.macro_dim, device=x_bull.device)
            x = torch.cat([x_bull, z_macro], dim=-1)
        else:
            x = x_bull
        
        raw = self.network(x)
        return self.activation(raw)


class GameEnergyModelV2(nn.Module):
    """
    改进版博弈能量模型 V2
    
    核心改进：
    1. 宏观调制：α_market * E_bull, β_risk * E_bear
    2. 刚性Bear：使用Softplus(β=5)让阻力有指数特性
    3. 简化结构：减少隐藏层维度防止过拟合
    4. 势能陷阱：惩罚高波动率股票
    
    能量方程：
    E = β_risk * E_bear - α_market * E_bull + γ * (vol - target_vol)²
    
    物理解释：
    - 牛市：α > β → 动力主导，选高动量股
    - 熊市：β > α → 阻力主导，选低估值股
    - 势能陷阱：惩罚过于疯狂（高波动）或死寂（低波动）的股票
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_dim: int = 27,
        bull_hidden: List[int] = [64, 32],  # 简化
        bear_hidden: List[int] = [64, 32],  # 简化
        rigidity_beta: float = 5.0,  # Bear刚性系数
        volatility_penalty: float = 0.1,  # 势能陷阱系数
        target_volatility: float = 0.0,  # 目标波动率（z-score）
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.macro_dim = macro_dim
        self.volatility_penalty = volatility_penalty
        self.target_volatility = target_volatility
        
        # 【新增】保存维度信息，供外部获取
        self.bull_dim = bull_dim
        self.bear_dim = bear_dim
        self.friction_dim = friction_dim
        
        # 1. 宏观调制网络
        self.macro_modulation = MacroModulationNetwork(
            macro_dim=macro_dim,
            hidden_dim=32,
            min_coef=0.3,
            max_coef=3.0
        )
        
        # 2. 简化的Bull网络（不再拼接macro，因为macro通过调制系数影响）
        self.bull_net = SimplifiedBullNet(
            input_dim=bull_dim,
            hidden_dims=bull_hidden,
            macro_dim=0,  # 不拼接macro
            dropout=dropout
        )
        
        # 3. 刚性Bear网络
        self.bear_net = RigidBearNet(
            input_dim=bear_dim,
            hidden_dims=bear_hidden,
            macro_dim=0,  # 不拼接macro
            dropout=dropout,
            rigidity_beta=rigidity_beta
        )
        
        # 4. 摩擦编码器
        self.friction_encoder = FrictionEncoder(friction_dim)
        
        # 5. 可学习参数
        self.friction_weight = nn.Parameter(torch.tensor(0.1))
        
        # 6. 用于势能陷阱的波动率索引（在friction特征中）
        # 假设volatility_20d_zscore是friction特征的第3个
        self.vol_feature_idx = 2  # 可调整
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        z_macro: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        能量方程：E = β_risk * E_bear - α_market * E_bull + friction + volatility_trap
        """
        # 输入已在Dataset层清洗，仅对z_macro做安全检查
        # 1. 计算宏观调制系数
        if z_macro is not None:
            alpha_market, beta_risk = self.macro_modulation(z_macro)
        else:
            # 默认系数为1
            batch_size = x_bull.size(0)
            alpha_market = torch.ones(batch_size, 1, device=x_bull.device)
            beta_risk = torch.ones(batch_size, 1, device=x_bull.device)
        
        # 2. 计算原始能量
        E_bull_raw = self.bull_net(x_bull)  # [batch, 1]
        E_bear_raw = self.bear_net(x_bear)  # [batch, 1]
        friction_heat = self.friction_encoder(x_friction)  # [batch, 1]
        
        # 3. 应用宏观调制
        E_bull = alpha_market * E_bull_raw
        E_bear = beta_risk * E_bear_raw
        
        # 4. 博弈能量：阻力 - 动力 + 摩擦
        E_game = E_bear - E_bull + self.friction_weight * friction_heat
        
        # 5. 势能陷阱：惩罚偏离目标波动率的股票
        if self.volatility_penalty > 0 and x_friction.size(-1) > self.vol_feature_idx:
            volatility = x_friction[:, self.vol_feature_idx:self.vol_feature_idx+1]
            volatility_trap = self.volatility_penalty * (volatility - self.target_volatility) ** 2
        else:
            volatility_trap = torch.zeros_like(E_game)
        
        # 6. 最终能量
        energy = E_game + volatility_trap
        
        # 输出清洗
        energy = energy.squeeze(-1)  # [batch]

        if return_components:
            return energy, {
                'E_bull': E_bull.squeeze(-1),
                'E_bear': E_bear.squeeze(-1),
                'E_bull_raw': E_bull_raw.squeeze(-1),
                'E_bear_raw': E_bear_raw.squeeze(-1),
                'E_game': E_game.squeeze(-1),
                'friction_heat': friction_heat.squeeze(-1),
                'volatility_trap': volatility_trap.squeeze(-1),
                'alpha_market': alpha_market.squeeze(-1),
                'beta_risk': beta_risk.squeeze(-1),
            }

        return energy

    def get_energy_score(self, x_bull, x_bear, x_friction, z_macro=None) -> torch.Tensor:
        """获取分数（负能量）"""
        return -self.forward(x_bull, x_bear, x_friction, z_macro)
    
    def get_modulation_coefficients(self, z_macro: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取宏观调制系数（用于可视化）"""
        alpha, beta = self.macro_modulation(z_macro)
        return {
            'alpha_market': alpha,
            'beta_risk': beta,
            'alpha_beta_ratio': alpha / (beta + 1e-6)
        }


# ============================================================================
# Part 7: V3 组件 - 平衡版本（解决Bear被关闭问题）
# ============================================================================

class BearFeatureEnhancer(nn.Module):
    """
    Bear特征增强模块
    
    对高估值信号做指数放大，让BearNet能够输出更大的阻力值
    
    设计原理：
    - pe_rank=0.5 时，增强后≈1.5（略微放大）
    - pe_rank=0.9 时，增强后≈5.0（显著放大）
    - pe_rank=1.0 时，增强后≈7.4（极大放大）
    
    这模拟了"估值泡沫"效应：估值越高，风险指数级增长
    """
    
    def __init__(self, input_dim: int, scale: float = 2.0):
        super().__init__()
        self.scale = scale
        
        # 可学习的增强权重（每个特征不同的增强程度）
        self.enhance_weights = nn.Parameter(torch.ones(input_dim) * 0.5)
    
    def forward(self, x_bear: torch.Tensor) -> torch.Tensor:
        """
        增强Bear特征
        
        公式: enhanced = x + w * (exp(x * scale) - 1)
        - 当x小时，增强幅度小
        - 当x大时，增强幅度指数级增长
        """
        # Sigmoid确保权重在0-1之间
        weights = torch.sigmoid(self.enhance_weights)
        
        # 指数增强（对正值特别敏感）
        exp_term = torch.exp(torch.clamp(x_bear * self.scale, max=5)) - 1
        
        # 加权增强
        enhanced = x_bear + weights * exp_term
        
        # 数值稳定性
        enhanced = torch.clamp(enhanced, -10, 10)
        
        return enhanced


class BalancedBearNet(nn.Module):
    """
    平衡的阻力网络
    
    关键改进：
    1. 内置特征增强
    2. 更深的网络（捕捉非线性）
    3. 保证输出在合理范围（0.5-10）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        enhance_scale: float = 2.0,
        min_output: float = 0.5,
        max_output: float = 10.0
    ):
        super().__init__()
        
        self.min_output = min_output
        self.max_output = max_output
        
        # 特征增强
        self.enhancer = BearFeatureEnhancer(input_dim, enhance_scale)
        
        # 主网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        
        # 最后一层偏置设为正值，确保初始输出为正
        last_linear = self.network[-1]
        nn.init.constant_(last_linear.bias, 1.0)
    
    def forward(self, x_bear: torch.Tensor) -> torch.Tensor:
        # 1. 特征增强
        x_enhanced = self.enhancer(x_bear)
        
        # 2. 网络计算
        raw = self.network(x_enhanced)
        
        # 3. 输出限制在 [min_output, max_output]
        output = self.min_output + (self.max_output - self.min_output) * torch.sigmoid(raw)
        
        return output


class BalancedBullNet(nn.Module):
    """
    平衡的动力网络
    
    与BearNet对称设计，确保输出范围相似
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        min_output: float = 0.5,
        max_output: float = 10.0
    ):
        super().__init__()
        
        self.min_output = min_output
        self.max_output = max_output
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        
        last_linear = self.network[-1]
        nn.init.constant_(last_linear.bias, 1.0)
    
    def forward(self, x_bull: torch.Tensor) -> torch.Tensor:
        raw = self.network(x_bull)
        output = self.min_output + (self.max_output - self.min_output) * torch.sigmoid(raw)
        return output


class ConstrainedMacroModulation(nn.Module):
    """
    受约束的宏观调制网络
    
    关键改进：
    1. α和β必须在更窄的范围内 [0.5, 2.0]
    2. 添加平衡约束：|α - β| 不能太大
    3. 初始化接近1，让模型从平衡状态开始
    """
    
    def __init__(
        self,
        macro_dim: int,
        hidden_dim: int = 32,
        min_coef: float = 0.5,
        max_coef: float = 2.0,
        balance_penalty: float = 0.1
    ):
        super().__init__()
        
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.balance_penalty = balance_penalty
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 分别输出α和β
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化为接近1的输出
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
    
    def forward(self, z_macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(z_macro)
        
        alpha_raw = self.alpha_head(h)
        beta_raw = self.beta_head(h)
        
        # 映射到目标范围
        alpha = self.min_coef + (self.max_coef - self.min_coef) * torch.sigmoid(alpha_raw)
        beta = self.min_coef + (self.max_coef - self.min_coef) * torch.sigmoid(beta_raw)
        
        return alpha, beta
    
    def get_balance_loss(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """计算平衡损失，惩罚α和β差距过大"""
        diff = (alpha - beta).abs()
        return self.balance_penalty * diff.mean()


class GameEnergyModelV3(nn.Module):
    """
    博弈能量模型 V3 - 平衡版本
    
    核心改进：
    1. Bull和Bear网络对称设计，输出范围相同
    2. Bear特征内置增强，无需预处理
    3. 宏观调制受约束，防止极端值
    4. 添加Bear贡献比例监控
    
    能量方程：
    E = β * E_bear - α * E_bull + friction
    
    物理解释：
    - E_bear: 阻力能量（高估值→高阻力）
    - E_bull: 动力能量（高动量→高动力）
    - 低能量 = 高动力 + 低阻力 = 好股票
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_dim: int = 27,
        hidden_dims: List[int] = [128, 64, 32],
        bear_enhance_scale: float = 2.0,
        min_energy_output: float = 0.5,
        max_energy_output: float = 10.0,
        macro_coef_min: float = 0.5,
        macro_coef_max: float = 2.0,
        friction_weight: float = 0.1,
        dropout: float = 0.3,
        target_bear_ratio: float = 0.3
    ):
        super().__init__()
        
        # 保存维度（用于Stage3等外部调用）
        self.bull_dim = bull_dim
        self.bear_dim = bear_dim
        self.friction_dim = friction_dim
        self.macro_dim = macro_dim
        
        self.target_bear_ratio = target_bear_ratio
        self.friction_weight_value = friction_weight
        
        # 1. 平衡的Bull网络
        self.bull_net = BalancedBullNet(
            input_dim=bull_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            min_output=min_energy_output,
            max_output=max_energy_output
        )
        
        # 2. 平衡的Bear网络（内置特征增强）
        self.bear_net = BalancedBearNet(
            input_dim=bear_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            enhance_scale=bear_enhance_scale,
            min_output=min_energy_output,
            max_output=max_energy_output
        )
        
        # 3. 摩擦编码器
        self.friction_encoder = nn.Sequential(
            nn.Linear(friction_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # 4. 受约束的宏观调制
        self.macro_modulation = ConstrainedMacroModulation(
            macro_dim=macro_dim,
            hidden_dim=32,
            min_coef=macro_coef_min,
            max_coef=macro_coef_max
        )
        
        # 5. 可学习的摩擦权重
        self.friction_weight = nn.Parameter(torch.tensor(friction_weight))
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        z_macro: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            energy: 能量值（越低越好）
            components: 各组件详情（如果return_components=True）
        """
        # 输入已在Dataset层清洗，仅对z_macro做安全检查
        # 1. 计算原始能量
        E_bull_raw = self.bull_net(x_bull)
        E_bear_raw = self.bear_net(x_bear)
        friction_heat = self.friction_encoder(x_friction)
        
        # 2. 计算宏观调制系数
        if z_macro is not None:
            z_macro = torch.nan_to_num(z_macro, nan=0.0, posinf=0.0, neginf=0.0)
            alpha_market, beta_risk = self.macro_modulation(z_macro)
        else:
            batch_size = x_bull.size(0)
            alpha_market = torch.ones(batch_size, 1, device=x_bull.device)
            beta_risk = torch.ones(batch_size, 1, device=x_bull.device)
        
        # 3. 应用宏观调制
        E_bull = alpha_market * E_bull_raw
        E_bear = beta_risk * E_bear_raw
        
        # 4. 博弈能量：阻力 - 动力 + 摩擦
        E_game = E_bear - E_bull + self.friction_weight * friction_heat
        
        # 5. 最终能量
        energy = E_game.squeeze(-1)

        if return_components:
            return energy, {
                'E_bull': E_bull.squeeze(-1),
                'E_bear': E_bear.squeeze(-1),
                'E_bull_raw': E_bull_raw.squeeze(-1),
                'E_bear_raw': E_bear_raw.squeeze(-1),
                'E_game': E_game.squeeze(-1),
                'friction_heat': friction_heat.squeeze(-1),
                'alpha_market': alpha_market.squeeze(-1),
                'beta_risk': beta_risk.squeeze(-1),
            }
        
        return energy
    
    def get_bear_contribution_ratio(
        self, 
        x_bull: torch.Tensor, 
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        z_macro: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算Bear在总能量中的贡献比例"""
        _, components = self.forward(x_bull, x_bear, x_friction, z_macro, return_components=True)
        
        E_bull = components['E_bull'].abs()
        E_bear = components['E_bear'].abs()
        
        total = E_bull + E_bear + 1e-6
        bear_ratio = E_bear / total
        
        return bear_ratio.mean()
    
    def get_modulation_coefficients(self, z_macro: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取宏观调制系数"""
        alpha, beta = self.macro_modulation(z_macro)
        return {
            'alpha_market': alpha,
            'beta_risk': beta,
            'alpha_beta_ratio': alpha / (beta + 1e-6)
        }


def test_all_models():
    """测试所有模型变体"""
    print("\n" + "=" * 70)
    print("    测试能量模型组件")
    print("=" * 70)
    
    batch_size = 16
    bull_dim, bear_dim, friction_dim, macro_dim = 8, 6, 4, 12
    
    x_bull = torch.randn(batch_size, bull_dim)
    x_bear = torch.randn(batch_size, bear_dim)
    x_friction = torch.randn(batch_size, friction_dim)
    x_macro = torch.randn(batch_size, macro_dim)
    
    # 测试原始GameEnergyModel
    print("\n>>> 测试 GameEnergyModel (Concat)")
    model = GameEnergyModel(bull_dim, bear_dim, friction_dim, macro_dim)
    energy, components = model(x_bull, x_bear, x_friction, x_macro, return_components=True)
    print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    能量输出: {energy.shape}")
    
    # 测试改进版模型
    for mod_type in ['concat', 'film', 'hyper']:
        print(f"\n>>> 测试 ImprovedGameEnergyModel ({mod_type})")
        model = ImprovedGameEnergyModel(
            bull_dim, bear_dim, friction_dim, macro_dim,
            modulation_type=mod_type
        )
        energy, components = model(x_bull, x_bear, x_friction, x_macro, return_components=True)
        print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"    能量输出: {energy.shape}")
    
    # 测试消融变体
    print("\n>>> 测试 EBMNoFriction")
    model = EBMNoFriction(bull_dim, bear_dim, macro_dim)
    energy = model(x_bull, x_bear, x_friction, x_macro)
    print(f"    能量输出: {energy.shape}")
    
    print("\n>>> 测试 EBMNoMacro")
    model = EBMNoMacro(bull_dim, bear_dim, friction_dim)
    energy = model(x_bull, x_bear, x_friction, x_macro)
    print(f"    能量输出: {energy.shape}")
    
    print("\n>>> 测试 EBMSingleFriction")
    model = EBMSingleFriction(bull_dim, bear_dim, friction_dim, macro_dim)
    energy = model(x_bull, x_bear, x_friction, x_macro)
    print(f"    能量输出: {energy.shape}")
    
    # 测试V3模型
    print("\n>>> 测试 GameEnergyModelV3 (平衡版)")
    model_v3 = GameEnergyModelV3(bull_dim, bear_dim, friction_dim, macro_dim)
    energy, components = model_v3(x_bull, x_bear, x_friction, x_macro, return_components=True)
    print(f"    参数量: {sum(p.numel() for p in model_v3.parameters()):,}")
    print(f"    能量输出: {energy.shape}")
    print(f"    E_bull: mean={components['E_bull'].mean():.4f}, std={components['E_bull'].std():.4f}")
    print(f"    E_bear: mean={components['E_bear'].mean():.4f}, std={components['E_bear'].std():.4f}")
    
    # 验证Bear贡献比例
    E_bull_abs = components['E_bull'].abs()
    E_bear_abs = components['E_bear'].abs()
    bear_ratio = E_bear_abs / (E_bull_abs + E_bear_abs + 1e-6)
    print(f"    Bear贡献比例: {bear_ratio.mean():.4f}")
    
    assert components['E_bear'].min() >= 0.4, f"E_bear最小值过低: {components['E_bear'].min():.4f}"
    print("    ✓ V3模型E_bear输出正常")
    
    # 测试V4模型
    print("\n>>> 测试 GameEnergyModelV4 (自适应方向)")
    model_v4 = GameEnergyModelV4(bull_dim, bear_dim, friction_dim, macro_dim)
    energy, components = model_v4(x_bull, x_bear, x_friction, x_macro, return_components=True)
    print(f"    参数量: {sum(p.numel() for p in model_v4.parameters()):,}")
    print(f"    能量输出: {energy.shape}")
    print(f"    Macro scale: mean={components['macro_scale'].mean():.4f}")
    
    # 查看特征方向
    directions = model_v4.get_feature_directions()
    print(f"    Bull特征方向: {directions['bull']['direction'][:3].numpy()}")
    print("    ✓ V4模型输出正常")
    
    print("\n✓ 所有模型测试通过")


# ============================================================================
# Part 8: V4 组件 - 自适应方向学习
# ============================================================================

class AdaptiveFeatureEncoder(nn.Module):
    """
    自适应特征编码器
    
    关键特性：
    1. 可学习的特征方向（每个特征一个符号参数）
    2. 可学习的特征重要性权重
    3. 特征交互层
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 可学习的特征方向（初始化为+1，让模型自己学习是否需要反转）
        self.direction = nn.Parameter(torch.ones(input_dim))
        
        # 可学习的特征重要性
        self.importance = nn.Parameter(torch.ones(input_dim))
        
        # 编码网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用方向和重要性
        direction = torch.tanh(self.direction)
        importance = torch.sigmoid(self.importance)
        
        x_adjusted = x * direction * importance
        
        return self.encoder(x_adjusted)
    
    def get_feature_weights(self) -> Dict[str, torch.Tensor]:
        """获取学习到的特征方向和重要性"""
        return {
            'direction': torch.tanh(self.direction).detach(),
            'importance': torch.sigmoid(self.importance).detach()
        }


class UnifiedEnergyNetwork(nn.Module):
    """
    统一能量网络
    
    不再区分Bull/Bear/Friction，将所有特征统一处理
    """
    
    def __init__(
        self,
        total_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = total_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleMacroModulation(nn.Module):
    """简化的宏观调制"""
    
    def __init__(
        self,
        macro_dim: int,
        hidden_dim: int = 32,
        min_scale: float = 0.5,
        max_scale: float = 2.0
    ):
        super().__init__()
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.network = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        raw = self.network(z_macro)
        scale = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(raw)
        return scale


class GameEnergyModelV4(nn.Module):
    """
    博弈能量模型 V4 - 自适应方向学习
    
    核心设计：
    1. 不做Bull/Bear方向假设
    2. 使用AdaptiveFeatureEncoder让模型学习特征方向
    3. 统一能量网络处理所有特征
    
    能量方程：
    E = scale(macro) * EnergyNet(concat(encode(bull), encode(bear), encode(friction)))
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_dim: int = 27,
        encoder_hidden: int = 64,
        encoder_output: int = 32,
        energy_hidden: List[int] = [96, 48],
        dropout: float = 0.3,
        macro_scale_min: float = 0.5,
        macro_scale_max: float = 2.0
    ):
        super().__init__()
        
        self.bull_dim = bull_dim
        self.bear_dim = bear_dim
        self.friction_dim = friction_dim
        self.macro_dim = macro_dim
        
        # 自适应特征编码器
        self.bull_encoder = AdaptiveFeatureEncoder(
            bull_dim, encoder_hidden, encoder_output, dropout
        )
        self.bear_encoder = AdaptiveFeatureEncoder(
            bear_dim, encoder_hidden, encoder_output, dropout
        )
        self.friction_encoder = AdaptiveFeatureEncoder(
            friction_dim, encoder_hidden, encoder_output, dropout
        )
        
        # 统一能量网络
        total_encoded_dim = encoder_output * 3
        self.energy_network = UnifiedEnergyNetwork(
            total_encoded_dim, energy_hidden, dropout
        )
        
        # 简化宏观调制
        self.macro_modulation = SimpleMacroModulation(
            macro_dim, hidden_dim=32,
            min_scale=macro_scale_min,
            max_scale=macro_scale_max
        )
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        z_macro: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """前向传播（输入已在Dataset层清洗，减少冗余nan_to_num）"""
        h_bull = self.bull_encoder(x_bull)
        h_bear = self.bear_encoder(x_bear)
        h_friction = self.friction_encoder(x_friction)

        h_concat = torch.cat([h_bull, h_bear, h_friction], dim=-1)

        energy_raw = self.energy_network(h_concat)

        if z_macro is not None:
            macro_scale = self.macro_modulation(z_macro)
        else:
            macro_scale = torch.ones_like(energy_raw)

        energy = macro_scale * energy_raw
        energy = energy.squeeze(-1)
        
        if return_components:
            return energy, {
                'energy_raw': energy_raw.squeeze(-1),
                'macro_scale': macro_scale.squeeze(-1),
                'h_bull': h_bull,
                'h_bear': h_bear,
                'h_friction': h_friction,
            }
        
        return energy
    
    def get_feature_directions(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取各组特征的学习方向"""
        return {
            'bull': self.bull_encoder.get_feature_weights(),
            'bear': self.bear_encoder.get_feature_weights(),
            'friction': self.friction_encoder.get_feature_weights()
        }


class DualMacroModulation(nn.Module):
    """
    双系数宏观调制（用于V4.1）
    
    输出alpha和beta两个调制系数：
    - alpha: 动力调制系数（牛市时放大）
    - beta: 阻力调制系数（熊市时放大）
    """
    
    def __init__(
        self,
        macro_dim: int,
        hidden_dim: int = 32,
        min_coef: float = 0.3,
        max_coef: float = 3.0
    ):
        super().__init__()
        
        self.min_coef = min_coef
        self.max_coef = max_coef
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Alpha头（动力调制）
        self.alpha_head = nn.Linear(hidden_dim, 1)
        
        # Beta头（阻力调制）
        self.beta_head = nn.Linear(hidden_dim, 1)
        
        # 初始化使输出接近1
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
    
    def forward(self, z_macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回(alpha, beta)"""
        h = self.shared(z_macro)
        
        alpha_raw = self.alpha_head(h)
        beta_raw = self.beta_head(h)
        
        # Sigmoid映射到[min_coef, max_coef]
        alpha = self.min_coef + (self.max_coef - self.min_coef) * torch.sigmoid(alpha_raw)
        beta = self.min_coef + (self.max_coef - self.min_coef) * torch.sigmoid(beta_raw)
        
        return alpha, beta


class GameEnergyModelV4_1(nn.Module):
    """
    博弈能量模型 V4.1 - 自适应方向 + 双系数调制
    
    结合V4的自适应方向学习和V2的alpha/beta调制：
    1. AdaptiveFeatureEncoder学习特征方向和重要性
    2. DualMacroModulation输出alpha和beta
    3. 能量方程保留Bull-Bear博弈结构
    
    能量方程：
    E = beta * ||h_bear|| - alpha * ||h_bull|| + friction_weight * ||h_friction||
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        friction_dim: int,
        macro_dim: int = 27,
        encoder_hidden: int = 32,
        encoder_output: int = 16,
        dropout: float = 0.5,
        macro_coef_min: float = 0.3,
        macro_coef_max: float = 3.0,
        friction_weight: float = 0.1
    ):
        super().__init__()
        
        self.bull_dim = bull_dim
        self.bear_dim = bear_dim
        self.friction_dim = friction_dim
        self.macro_dim = macro_dim
        self.friction_weight = friction_weight
        
        # 自适应特征编码器
        self.bull_encoder = AdaptiveFeatureEncoder(
            bull_dim, encoder_hidden, encoder_output, dropout
        )
        self.bear_encoder = AdaptiveFeatureEncoder(
            bear_dim, encoder_hidden, encoder_output, dropout
        )
        self.friction_encoder = AdaptiveFeatureEncoder(
            friction_dim, encoder_hidden, encoder_output, dropout
        )
        
        # 能量投影层（将编码向量投影到标量能量）
        self.bull_energy_proj = nn.Sequential(
            nn.Linear(encoder_output, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.bear_energy_proj = nn.Sequential(
            nn.Linear(encoder_output, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.friction_energy_proj = nn.Sequential(
            nn.Linear(encoder_output, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # 双系数宏观调制
        self.macro_modulation = DualMacroModulation(
            macro_dim, hidden_dim=32,
            min_coef=macro_coef_min,
            max_coef=macro_coef_max
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.bull_energy_proj, self.bear_energy_proj, self.friction_energy_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_friction: torch.Tensor,
        z_macro: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """前向传播（输入已在Dataset层清洗）"""
        # 自适应编码
        h_bull = self.bull_encoder(x_bull)
        h_bear = self.bear_encoder(x_bear)
        h_friction = self.friction_encoder(x_friction)

        # 能量投影
        E_bull_raw = self.bull_energy_proj(h_bull)
        E_bear_raw = self.bear_energy_proj(h_bear)
        E_friction_raw = self.friction_energy_proj(h_friction)

        # 宏观调制
        if z_macro is not None:
            alpha, beta = self.macro_modulation(z_macro)
        else:
            batch_size = x_bull.size(0)
            alpha = torch.ones(batch_size, 1, device=x_bull.device)
            beta = torch.ones(batch_size, 1, device=x_bull.device)
        
        # 应用调制
        E_bull = alpha * E_bull_raw
        E_bear = beta * E_bear_raw
        
        # 博弈能量：阻力 - 动力 + 摩擦
        energy = E_bear - E_bull + self.friction_weight * E_friction_raw

        energy = energy.squeeze(-1)
        
        if return_components:
            return energy, {
                'E_bull': E_bull.squeeze(-1),
                'E_bear': E_bear.squeeze(-1),
                'E_bull_raw': E_bull_raw.squeeze(-1),
                'E_bear_raw': E_bear_raw.squeeze(-1),
                'E_friction': E_friction_raw.squeeze(-1),
                'alpha_market': alpha.squeeze(-1),
                'beta_risk': beta.squeeze(-1),
                'h_bull': h_bull,
                'h_bear': h_bear,
                'h_friction': h_friction,
            }
        
        return energy
    
    def get_feature_directions(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取各组特征的学习方向"""
        return {
            'bull': self.bull_encoder.get_feature_weights(),
            'bear': self.bear_encoder.get_feature_weights(),
            'friction': self.friction_encoder.get_feature_weights()
        }
    
    def get_modulation_coefficients(self, z_macro: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取宏观调制系数（用于可视化）"""
        alpha, beta = self.macro_modulation(z_macro)
        return {
            'alpha_market': alpha,
            'beta_risk': beta,
            'alpha_beta_ratio': alpha / (beta + 1e-6)
        }


# ================================================================================
# V4.2 物理场修复版
# ================================================================================

class AdaptiveFeatureEncoderV2(nn.Module):
    """
    自适应特征编码器 V2 - 增强方向学习
    
    改进：
    1. 使用更大的初始方向偏差，促进学习
    2. 添加方向正则化损失
    3. 重要性使用更宽的范围
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        dropout: float = 0.5,
        init_direction_std: float = 0.5  # 初始方向标准差
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 可学习的特征方向（-1到+1）
        # 初始化为随机正负，而不是全0
        self.direction = nn.Parameter(torch.randn(input_dim) * init_direction_std)
        
        # 可学习的特征重要性（0到1）
        # 初始化为0，sigmoid后约0.5
        self.importance = nn.Parameter(torch.zeros(input_dim))
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用方向（tanh限制在[-1, 1]）
        direction = torch.tanh(self.direction)
        
        # 应用重要性（sigmoid限制在[0, 1]，但使用更宽范围）
        importance = torch.sigmoid(self.importance * 2)  # 乘2使梯度更大
        
        # 调整特征
        x_adjusted = x * direction * importance
        
        return self.encoder(x_adjusted)
    
    def get_feature_weights(self) -> Dict[str, torch.Tensor]:
        """获取学习到的特征方向和重要性"""
        return {
            'direction': torch.tanh(self.direction).detach(),
            'importance': torch.sigmoid(self.importance * 2).detach(),
            'raw_direction': self.direction.detach(),
            'raw_importance': self.importance.detach()
        }
    
    def get_direction_regularization(self) -> torch.Tensor:
        """方向正则化：鼓励方向偏离0"""
        direction = torch.tanh(self.direction)
        # 惩罚接近0的方向（鼓励明确的正负）
        return -torch.abs(direction).mean()


class TripleMacroModulation(nn.Module):
    """
    三系数宏观调制
    
    输出alpha, beta, gamma三个调制系数：
    - alpha: 动力调制（牛市放大Bull）
    - beta: 阻力调制（熊市放大Bear）
    - gamma: 热度调制（控制Heat的影响）
    """
    
    def __init__(
        self,
        macro_dim: int,
        hidden_dim: int = 32,
        alpha_range: Tuple[float, float] = (0.5, 2.0),
        beta_range: Tuple[float, float] = (0.5, 2.0),
        gamma_range: Tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__()
        
        self.alpha_min, self.alpha_max = alpha_range
        self.beta_min, self.beta_max = beta_range
        self.gamma_min, self.gamma_max = gamma_range
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 三个调制头
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.gamma_head = nn.Linear(hidden_dim, 1)
        
        # 初始化使输出接近中间值
        for head in [self.alpha_head, self.beta_head, self.gamma_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
    
    def forward(self, z_macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回(alpha, beta, gamma)"""
        h = self.shared(z_macro)
        
        alpha_raw = self.alpha_head(h)
        beta_raw = self.beta_head(h)
        gamma_raw = self.gamma_head(h)
        
        # Sigmoid映射到各自范围
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(alpha_raw)
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta_raw)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(gamma_raw)
        
        return alpha, beta, gamma


class GameEnergyModelV4_2(nn.Module):
    """
    博弈能量模型 V4.2 - 物理场修复版
    
    核心改进：
    1. 所有能量分量强制非负（Softplus）
    2. Heat有上限约束
    3. 过高Heat会被惩罚
    4. 添加方向正则化
    
    能量方程（修复版）：
    E = β * E_bear - α * E_bull + λ * E_heat
    
    【关键修复】：Heat不再降低能量，只作为惩罚项！
    
    物理直觉：
    - 高Bear = 高能量 = 差股票 ✓
    - 高Bull = 低能量 = 好股票 ✓
    - 高Heat = 高能量 = 惩罚高波动 ✓（不再是有益的！）
    """
    
    def __init__(
        self,
        bull_dim: int,
        bear_dim: int,
        heat_dim: int,  # 改名：friction -> heat
        macro_dim: int = 27,
        encoder_hidden: int = 32,
        encoder_output: int = 16,
        dropout: float = 0.5,
        alpha_range: Tuple[float, float] = (0.5, 2.0),
        beta_range: Tuple[float, float] = (0.5, 2.0),
        gamma_range: Tuple[float, float] = (0.0, 1.0),
        heat_cap: float = 1.0,           # 保留参数但不使用
        heat_penalty: float = 0.5,       # Heat惩罚系数（改为直接作用）
        direction_reg_weight: float = 0.01,  # 方向正则化权重
        # ============== 消融实验参数 ==============
        enable_bull: bool = True,              # 是否启用Bull能量
        enable_bear: bool = True,              # 是否启用Bear能量
        enable_friction: bool = True,          # 是否启用Friction/Heat能量
        enable_macro_modulation: bool = True,  # 是否启用宏观调制
        fixed_alpha: Optional[float] = None,   # 固定α值（None=自适应）
        fixed_beta: Optional[float] = None,    # 固定β值（None=自适应）
        fixed_gamma: Optional[float] = None,   # 固定γ值（None=自适应）
        input_noise_std: float = 0.03,         # 输入噪声标准差
        aggregation_mode: str = "potential_game",
        aggregation_hidden_dim: int = 12,
        energy_target_mean: float = 0.5,
        energy_target_std: float = 0.15,
        energy_bound_low: float = 0.1,
        energy_bound_high: float = 0.9,
        energy_bound_mode: str = "fixed",
        energy_target_std_min: float = 0.08,
        energy_target_std_max: float = 0.25,
        energy_bound_adaptive_scale: float = 0.5,
    ):
        super().__init__()
        
        self.bull_dim = bull_dim
        self.bear_dim = bear_dim
        self.friction_dim = heat_dim  # 保持接口兼容
        self.heat_dim = heat_dim
        self.macro_dim = macro_dim
        self.heat_penalty = heat_penalty
        self.direction_reg_weight = direction_reg_weight
        self.encoder_output = encoder_output  # 保存用于消融时创建零张量
        
        # ============== 保存消融设置 ==============
        self.enable_bull = enable_bull
        self.enable_bear = enable_bear
        self.enable_friction = enable_friction
        self.enable_macro_modulation = enable_macro_modulation
        self.fixed_alpha = fixed_alpha if fixed_alpha is not None else 1.0
        self.fixed_beta = fixed_beta if fixed_beta is not None else 1.0
        self.fixed_gamma = fixed_gamma if fixed_gamma is not None else 0.5
        self.input_noise_std = input_noise_std
        self.aggregation_mode = aggregation_mode
        self.energy_target_mean = energy_target_mean
        self.energy_target_std = energy_target_std
        self.energy_bound_low = energy_bound_low
        self.energy_bound_high = energy_bound_high
        self.energy_bound_mode = energy_bound_mode
        self.energy_target_std_min = energy_target_std_min
        self.energy_target_std_max = energy_target_std_max
        self.energy_bound_adaptive_scale = energy_bound_adaptive_scale
        
        # 打印消融配置
        print(f"\n>>> GameEnergyModelV4_2 消融配置:")
        print(f"    enable_bull: {self.enable_bull}")
        print(f"    enable_bear: {self.enable_bear}")
        print(f"    enable_friction: {self.enable_friction}")
        print(f"    enable_macro_modulation: {self.enable_macro_modulation}")
        print(f"    input_noise_std: {self.input_noise_std}")
        if not self.enable_macro_modulation:
            print(f"    fixed_alpha: {self.fixed_alpha}")
            print(f"    fixed_beta: {self.fixed_beta}")
            print(f"    fixed_gamma: {self.fixed_gamma}")
        
        # 使用V2编码器（增强方向学习）
        self.bull_encoder = AdaptiveFeatureEncoderV2(
            bull_dim, encoder_hidden, encoder_output, dropout
        )
        self.bear_encoder = AdaptiveFeatureEncoderV2(
            bear_dim, encoder_hidden, encoder_output, dropout
        )
        self.heat_encoder = AdaptiveFeatureEncoderV2(
            heat_dim, encoder_hidden, encoder_output, dropout
        )
        
        # 能量投影层（输出通过Softplus保证非负）
        self.bull_energy_proj = nn.Sequential(
            nn.Linear(encoder_output, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.bear_energy_proj = nn.Sequential(
            nn.Linear(encoder_output, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.heat_energy_proj = nn.Sequential(
            nn.Linear(encoder_output, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # 三系数宏观调制
        self.macro_modulation = TripleMacroModulation(
            macro_dim, hidden_dim=32,
            alpha_range=alpha_range,
            beta_range=beta_range,
            gamma_range=gamma_range
        )

        if aggregation_mode == "learnable_scalars":
            self.global_alpha_raw = nn.Parameter(torch.tensor(0.0))
            self.global_beta_raw = nn.Parameter(torch.tensor(0.0))
            self.global_gamma_raw = nn.Parameter(torch.tensor(0.0))
        elif aggregation_mode == "attention":
            attention_input = macro_dim if macro_dim > 0 else 3
            self.attention_aggregator = nn.Sequential(
                nn.Linear(attention_input, aggregation_hidden_dim),
                nn.GELU(),
                nn.Linear(aggregation_hidden_dim, 3),
            )
        elif aggregation_mode == "mlp":
            self.mlp_aggregator = nn.Sequential(
                nn.Linear(6, aggregation_hidden_dim),
                nn.GELU(),
                nn.Linear(aggregation_hidden_dim, 1),
            )

        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.bull_energy_proj, self.bear_energy_proj, self.heat_energy_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _batch_normalize(self, x: torch.Tensor, target_mean: float = 0.5, target_std: float = 0.15) -> torch.Tensor:
        """
        批内归一化：将能量值归一化到目标分布
        
        V4.2.4版本：使用sigmoid进行软剪裁，保持梯度平滑
        """
        # 计算批内统计量
        mean = x.mean()
        std = x.std() + 1e-6  # 避免除零
        
        # 归一化到标准正态
        x_norm = (x - mean) / std
        
        # 缩放到目标分布
        x_scaled = x_norm * target_std + target_mean

        # 软剪裁到目标边界范围，保持梯度
        bound_span = max(self.energy_bound_high - self.energy_bound_low, 1e-4)
        center = (self.energy_bound_low + self.energy_bound_high) / 2.0
        x_clipped = torch.sigmoid((x_scaled - center) * 10) * bound_span + self.energy_bound_low

        return x_clipped

    def _resolve_target_std(self, x_heat: torch.Tensor) -> float:
        target_std = float(self.energy_target_std)
        if self.energy_bound_mode != "adaptive_vol":
            return target_std

        heat_proxy = float(x_heat.abs().mean().detach().item()) if x_heat.numel() else target_std
        scaled = target_std * (1.0 + self.energy_bound_adaptive_scale * np.tanh(heat_proxy / 10.0))
        return float(np.clip(scaled, self.energy_target_std_min, self.energy_target_std_max))

    def _resolve_aggregation(
        self,
        E_bull: torch.Tensor,
        E_bear: torch.Tensor,
        E_heat: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        z_macro: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.aggregation_mode == "fixed_equal":
            one = torch.ones_like(alpha)
            energy = E_bear - E_bull + self.heat_penalty * E_heat
            return energy, one, one, one

        if self.aggregation_mode == "learnable_scalars":
            alpha_eff = 0.5 + 1.5 * torch.sigmoid(self.global_alpha_raw)
            beta_eff = 0.5 + 1.5 * torch.sigmoid(self.global_beta_raw)
            gamma_eff = torch.sigmoid(self.global_gamma_raw)
            alpha_tensor = torch.ones_like(alpha) * alpha_eff
            beta_tensor = torch.ones_like(beta) * beta_eff
            gamma_tensor = torch.ones_like(gamma) * gamma_eff
            energy = beta_tensor * E_bear - alpha_tensor * E_bull + self.heat_penalty * gamma_tensor * E_heat
            return energy, alpha_tensor, beta_tensor, gamma_tensor

        if self.aggregation_mode == "attention":
            context = z_macro if z_macro is not None else torch.cat([E_bull, E_bear, E_heat], dim=-1)
            weights = torch.softmax(self.attention_aggregator(context), dim=-1)
            alpha_tensor = weights[:, 0:1]
            beta_tensor = weights[:, 1:2]
            gamma_tensor = weights[:, 2:3]
            energy = beta_tensor * E_bear - alpha_tensor * E_bull + self.heat_penalty * gamma_tensor * E_heat
            return energy, alpha_tensor, beta_tensor, gamma_tensor

        if self.aggregation_mode == "mlp":
            features = torch.cat([E_bull, E_bear, E_heat, alpha, beta, gamma], dim=-1)
            energy = self.mlp_aggregator(features)
            return energy, alpha, beta, gamma

        energy = beta * E_bear - alpha * E_bull + self.heat_penalty * gamma * E_heat
        return energy, alpha, beta, gamma
    
    def forward(
        self,
        x_bull: torch.Tensor,
        x_bear: torch.Tensor,
        x_heat: torch.Tensor,  # 兼容x_friction
        z_macro: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        前向传播 - 支持消融实验
        
        消融逻辑：
        - enable_bull=False: E_bull = 0
        - enable_bear=False: E_bear = 0
        - enable_friction=False: E_heat = 0
        - enable_macro_modulation=False: 使用固定的α/β/γ
        """
        batch_size = x_bull.size(0)
        device = x_bull.device

        # 输入已在Dataset层清洗，仅在训练时添加噪声增强
        if self.training and self.input_noise_std > 0:
            x_bull = x_bull + torch.randn_like(x_bull) * self.input_noise_std
            x_bear = x_bear + torch.randn_like(x_bear) * self.input_noise_std
            x_heat = x_heat + torch.randn_like(x_heat) * self.input_noise_std
        
        # ============== Bull能量计算（支持消融） ==============
        target_std = self._resolve_target_std(x_heat)

        if self.enable_bull:
            h_bull = self.bull_encoder(x_bull)
            E_bull_raw = self.bull_energy_proj(h_bull)
            E_bull_sp = F.softplus(E_bull_raw, beta=1.0)
            E_bull = self._batch_normalize(E_bull_sp, target_mean=self.energy_target_mean, target_std=target_std)
        else:
            # 消融：Bull能量置零
            h_bull = torch.zeros(batch_size, self.encoder_output, device=device)
            E_bull = torch.zeros(batch_size, 1, device=device)
        
        # ============== Bear能量计算（支持消融） ==============
        if self.enable_bear:
            h_bear = self.bear_encoder(x_bear)
            E_bear_raw = self.bear_energy_proj(h_bear)
            E_bear_sp = F.softplus(E_bear_raw, beta=1.0)
            E_bear = self._batch_normalize(E_bear_sp, target_mean=self.energy_target_mean, target_std=target_std)
        else:
            # 消融：Bear能量置零
            h_bear = torch.zeros(batch_size, self.encoder_output, device=device)
            E_bear = torch.zeros(batch_size, 1, device=device)
        
        # ============== Heat能量计算（支持消融） ==============
        if self.enable_friction:
            h_heat = self.heat_encoder(x_heat)
            E_heat_raw = self.heat_energy_proj(h_heat)
            E_heat_sp = F.softplus(E_heat_raw, beta=1.0)
            E_heat = self._batch_normalize(E_heat_sp, target_mean=self.energy_target_mean, target_std=target_std)
        else:
            # 消融：Heat能量置零
            h_heat = torch.zeros(batch_size, self.encoder_output, device=device)
            E_heat = torch.zeros(batch_size, 1, device=device)
        
        # ============== 宏观调制（支持消融） ==============
        if self.enable_macro_modulation and z_macro is not None:
            alpha, beta, gamma = self.macro_modulation(z_macro)
        else:
            # 消融：使用固定系数
            alpha = torch.ones(batch_size, 1, device=device) * self.fixed_alpha
            beta = torch.ones(batch_size, 1, device=device) * self.fixed_beta
            gamma = torch.ones(batch_size, 1, device=device) * self.fixed_gamma
        
        energy, alpha_eff, beta_eff, gamma_eff = self._resolve_aggregation(
            E_bull,
            E_bear,
            E_heat,
            alpha,
            beta,
            gamma,
            z_macro,
        )
        E_bull_mod = alpha_eff * E_bull
        E_bear_mod = beta_eff * E_bear
        E_heat_penalty = self.heat_penalty * gamma_eff * E_heat

        energy = energy.squeeze(-1)

        if return_components:
            return energy, {
                'E_bull': E_bull_mod.squeeze(-1),
                'E_bear': E_bear_mod.squeeze(-1),
                'E_heat': E_heat.squeeze(-1),
                'E_heat_penalty': E_heat_penalty.squeeze(-1),
                'E_friction': E_heat.squeeze(-1),  # 兼容旧接口
                'alpha_market': alpha_eff.squeeze(-1),
                'beta_risk': beta_eff.squeeze(-1),
                'gamma_heat': gamma_eff.squeeze(-1),
                'h_bull': h_bull,
                'h_bear': h_bear,
                'h_friction': h_heat,
                'h_heat': h_heat,
                # 消融状态信息
                '_ablation_enable_bull': self.enable_bull,
                '_ablation_enable_bear': self.enable_bear,
                '_ablation_enable_friction': self.enable_friction,
            }
        
        return energy
    
    def get_feature_directions(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取各组特征的学习方向"""
        return {
            'bull': self.bull_encoder.get_feature_weights(),
            'bear': self.bear_encoder.get_feature_weights(),
            'friction': self.heat_encoder.get_feature_weights(),  # 兼容
        }
    
    def get_modulation_coefficients(self, z_macro: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取宏观调制系数"""
        if self.aggregation_mode == "learnable_scalars":
            alpha = torch.ones((z_macro.size(0), 1), device=z_macro.device) * (0.5 + 1.5 * torch.sigmoid(self.global_alpha_raw))
            beta = torch.ones((z_macro.size(0), 1), device=z_macro.device) * (0.5 + 1.5 * torch.sigmoid(self.global_beta_raw))
            gamma = torch.ones((z_macro.size(0), 1), device=z_macro.device) * torch.sigmoid(self.global_gamma_raw)
        elif self.aggregation_mode == "fixed_equal":
            alpha = torch.ones((z_macro.size(0), 1), device=z_macro.device)
            beta = torch.ones((z_macro.size(0), 1), device=z_macro.device)
            gamma = torch.ones((z_macro.size(0), 1), device=z_macro.device)
        elif self.aggregation_mode == "attention":
            weights = torch.softmax(self.attention_aggregator(z_macro), dim=-1)
            alpha = weights[:, 0:1]
            beta = weights[:, 1:2]
            gamma = weights[:, 2:3]
        else:
            alpha, beta, gamma = self.macro_modulation(z_macro)
        return {
            'alpha_market': alpha,
            'beta_risk': beta,
            'gamma_heat': gamma
        }
    
    def get_direction_regularization(self) -> torch.Tensor:
        """获取方向正则化损失"""
        reg_bull = self.bull_encoder.get_direction_regularization()
        reg_bear = self.bear_encoder.get_direction_regularization()
        reg_heat = self.heat_encoder.get_direction_regularization()
        return self.direction_reg_weight * (reg_bull + reg_bear + reg_heat)


if __name__ == "__main__":
    test_all_models()
