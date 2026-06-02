"""
================================================================================
损失函数模块
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PairwiseRankingLoss(nn.Module):
    """
    成对排序损失
    
    目标：使winner的能量 < loser的能量
    L = max(0, E_winner - E_loser + margin)
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, e_winner: torch.Tensor, e_loser: torch.Tensor) -> torch.Tensor:
        loss = F.relu(e_winner - e_loser + self.margin)
        return loss.mean()


class MarginRankingLoss(nn.Module):
    """
    自适应Margin排序损失
    
    margin随收益差距自适应调整
    """
    
    def __init__(self, base_margin: float = 0.3, scale: float = 1.0):
        super().__init__()
        self.base_margin = base_margin
        self.scale = scale
    
    def forward(
        self,
        e_winner: torch.Tensor,
        e_loser: torch.Tensor,
        winner_return: Optional[torch.Tensor] = None,
        loser_return: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if winner_return is not None and loser_return is not None:
            return_diff = torch.abs(winner_return - loser_return)
            margin = self.base_margin + self.scale * return_diff
        else:
            margin = self.base_margin
        
        loss = F.relu(e_winner - e_loser + margin)
        return loss.mean()


class ICLoss(nn.Module):
    """
    IC损失
    
    最大化预测与实际收益的相关性
    Loss = 1 - IC
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = pred.flatten()
        target = target.flatten()
        
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        cov = (pred_centered * target_centered).mean()
        pred_std = pred.std() + 1e-8
        target_std = target.std() + 1e-8
        
        ic = cov / (pred_std * target_std)
        loss = 1 - ic
        
        return loss, ic


class EnergyRegularization(nn.Module):
    """能量正则化"""
    
    def __init__(self, target_mean: float = 0.0, target_std: float = 1.0, weight: float = 0.01):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std
        self.weight = weight
    
    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        mean_penalty = (energy.mean() - self.target_mean) ** 2
        std_penalty = (energy.std() - self.target_std) ** 2
        return self.weight * (mean_penalty + std_penalty)


class CombinedLoss(nn.Module):
    """
    组合损失
    
    ranking_weight * RankingLoss + ic_weight * ICLoss + reg_weight * Regularization
    """
    
    def __init__(
        self,
        ranking_weight: float = 1.0,
        ic_weight: float = 0.5,
        reg_weight: float = 0.1,
        margin: float = 0.5
    ):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.ic_weight = ic_weight
        self.reg_weight = reg_weight
        
        self.ranking_loss = PairwiseRankingLoss(margin)
        self.ic_loss = ICLoss()
        self.reg_loss = EnergyRegularization(weight=reg_weight)
    
    def forward(
        self,
        e_winner: torch.Tensor,
        e_loser: torch.Tensor,
        pred: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # 排序损失
        ranking = self.ranking_loss(e_winner, e_loser)
        losses['ranking'] = ranking
        
        # IC损失
        if pred is not None and target is not None:
            ic_loss, ic = self.ic_loss(pred, target)
            losses['ic_loss'] = ic_loss
            losses['ic'] = ic
        else:
            losses['ic_loss'] = torch.tensor(0.0, device=e_winner.device)
            losses['ic'] = torch.tensor(0.0, device=e_winner.device)
        
        # 正则化
        all_energy = torch.cat([e_winner, e_loser])
        losses['reg'] = self.reg_loss(all_energy)
        
        # 总损失
        total = (self.ranking_weight * losses['ranking'] + 
                self.ic_weight * losses['ic_loss'] + 
                losses['reg'])
        losses['total'] = total
        
        return losses


class ListNetLoss(nn.Module):
    """
    ListNet损失 - 直接优化排序
    
    基于概率分布的排序损失，比Pairwise更有效
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: 预测分数 (batch,)
        target: 实际收益 (batch,)
        """
        pred = pred.flatten() / self.temperature
        target = target.flatten()
        
        # 转换为概率分布
        pred_prob = F.softmax(pred, dim=0)
        target_prob = F.softmax(target, dim=0)
        
        # 交叉熵损失
        loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-10))
        
        return loss


class PhysicsConsistencyLoss(nn.Module):
    """
    物理一致性损失
    
    确保能量组件的物理语义正确：
    - E_bull 应该与动量正相关（动力越强，能量贡献越正）
    - E_bear 应该与估值负相关（估值越高，阻力越大）
    - friction 应该与波动率正相关
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        E_bull: torch.Tensor,
        E_bear: torch.Tensor,
        friction: torch.Tensor,
        momentum_proxy: torch.Tensor,
        risk_proxy: torch.Tensor,
        volatility_proxy: torch.Tensor
    ) -> torch.Tensor:
        """
        确保物理一致性：
        - corr(E_bull, momentum) > 0
        - corr(E_bear, risk) > 0  
        - corr(friction, volatility) > 0
        """
        def soft_corr(x, y):
            """可微分的相关系数"""
            x = x.flatten()
            y = y.flatten()
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            cov = (x_centered * y_centered).mean()
            std_x = x.std() + 1e-8
            std_y = y.std() + 1e-8
            return cov / (std_x * std_y)
        
        # 物理约束：希望这些相关性为正
        bull_consistency = -soft_corr(E_bull, momentum_proxy)  # 负号因为我们要最小化
        bear_consistency = -soft_corr(E_bear, risk_proxy)
        friction_consistency = -soft_corr(friction, volatility_proxy)
        
        # 只惩罚负相关的情况
        bull_loss = F.relu(bull_consistency)
        bear_loss = F.relu(bear_consistency)
        friction_loss = F.relu(friction_consistency)
        
        return self.weight * (bull_loss + bear_loss + friction_loss)


class ICRankingLoss(nn.Module):
    """
    IC + Ranking 组合损失
    
    同时优化：
    1. IC（预测与收益的相关性）
    2. Ranking（相对顺序）
    
    这是让EBM有效的关键！
    """
    
    def __init__(
        self,
        ic_weight: float = 1.0,
        ranking_weight: float = 0.5,
        listnet_weight: float = 0.5,
        margin: float = 0.3
    ):
        super().__init__()
        self.ic_weight = ic_weight
        self.ranking_weight = ranking_weight
        self.listnet_weight = listnet_weight
        
        self.ic_loss = ICLoss()
        self.ranking_loss = PairwiseRankingLoss(margin)
        self.listnet_loss = ListNetLoss()
    
    def forward(
        self,
        pred_scores: torch.Tensor,
        true_returns: torch.Tensor,
        e_winner: Optional[torch.Tensor] = None,
        e_loser: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        pred_scores: EBM预测分数（-energy）
        true_returns: 实际收益
        e_winner, e_loser: 用于pairwise loss（可选）
        """
        losses = {}
        
        # IC损失 - 最重要！
        ic_loss, ic = self.ic_loss(-pred_scores, true_returns)  # 注意：能量越低分数越高
        losses['ic_loss'] = ic_loss
        losses['ic'] = ic
        
        # ListNet损失
        listnet = self.listnet_loss(-pred_scores, true_returns)
        losses['listnet'] = listnet
        
        # Pairwise损失（可选）
        if e_winner is not None and e_loser is not None:
            ranking = self.ranking_loss(e_winner, e_loser)
            losses['ranking'] = ranking
        else:
            losses['ranking'] = torch.tensor(0.0, device=pred_scores.device)
        
        # 总损失
        total = (self.ic_weight * ic_loss + 
                self.listnet_weight * listnet +
                self.ranking_weight * losses['ranking'])
        losses['total'] = total
        
        return losses


# ============================================================================
# 改进版损失函数 - 基于用户建议
# ============================================================================

class PointwiseRegressionLoss(nn.Module):
    """
    点对点回归损失 - 强迫模型理解"绝对价值"
    
    核心思想：直接拟合收益率，让能量分布与收益率分布一致
    
    L_reg = HuberLoss(E(x), -R_future)
    
    注意：能量越低越好，收益越高越好，所以目标是拟合负收益
    
    使用Huber Loss而非MSE，对极端值更鲁棒
    """
    
    def __init__(self, delta: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.delta = delta
        self.weight = weight
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
    
    def forward(self, energy: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energy: 预测能量 [batch]
            returns: 实际收益 [batch]
        
        Returns:
            loss: 回归损失
        """
        energy = energy.flatten()
        returns = returns.flatten()
        
        # 目标：能量 ≈ -收益（能量低=收益高）
        target = -returns
        
        loss = self.huber(energy, target)
        return self.weight * loss


class TopKListwiseLoss(nn.Module):
    """
    Top-K 列表排序损失 - 关注头部股票
    
    核心思想：不关心垃圾股之间的排序，只关心能否选出头部金股
    
    操作：
    1. 从每个batch中选出Top-K和Bottom-K
    2. 只计算这些样本的排序损失
    
    物理含义：像选拔赛，只在乎谁是冠军
    """
    
    def __init__(
        self,
        top_k: int = 50,
        bottom_k: int = 50,
        temperature: float = 1.0,
        weight: float = 1.0
    ):
        super().__init__()
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.temperature = temperature
        self.weight = weight
    
    def forward(self, energy: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energy: 预测能量 [batch]
            returns: 实际收益 [batch]
        
        Returns:
            loss: Top-K排序损失
        """
        energy = energy.flatten()
        returns = returns.flatten()
        
        batch_size = len(returns)
        k = min(self.top_k, batch_size // 4)  # 确保k不超过batch的1/4
        
        if k < 5:
            # batch太小，退化为普通ListNet
            pred_prob = F.softmax(-energy / self.temperature, dim=0)
            target_prob = F.softmax(returns / self.temperature, dim=0)
            return self.weight * (-torch.sum(target_prob * torch.log(pred_prob + 1e-10)))
        
        # 选出收益最高的Top-K和最低的Bottom-K
        _, top_indices = torch.topk(returns, k)
        _, bottom_indices = torch.topk(returns, k, largest=False)
        
        # 合并索引
        selected_indices = torch.cat([top_indices, bottom_indices])
        
        # 选取对应的能量和收益
        selected_energy = energy[selected_indices]
        selected_returns = returns[selected_indices]
        
        # 计算ListNet损失（只在selected样本上）
        pred_prob = F.softmax(-selected_energy / self.temperature, dim=0)
        target_prob = F.softmax(selected_returns / self.temperature, dim=0)
        
        loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-10))
        
        return self.weight * loss


class NDCGLoss(nn.Module):
    """
    NDCG损失 - 信息检索中的排序指标
    
    NDCG对头部排序错误惩罚更大
    
    使用ApproxNDCG的可微分近似
    """
    
    def __init__(self, temperature: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
    
    def forward(self, energy: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        计算可微分的NDCG损失
        """
        scores = -energy.flatten()  # 分数 = -能量
        relevance = returns.flatten()
        
        # 归一化收益作为相关性（0-1之间）
        relevance_min = relevance.min()
        relevance_max = relevance.max()
        if relevance_max - relevance_min > 1e-6:
            relevance_norm = (relevance - relevance_min) / (relevance_max - relevance_min)
        else:
            relevance_norm = torch.zeros_like(relevance)
        
        # 计算软排序位置
        n = len(scores)
        scores_expanded = scores.unsqueeze(1)  # [n, 1]
        scores_diff = scores_expanded - scores.unsqueeze(0)  # [n, n]
        soft_ranks = torch.sigmoid(scores_diff / self.temperature).sum(dim=1)  # [n]
        
        # 折扣因子：1 / log2(rank + 1)
        discount = 1.0 / torch.log2(soft_ranks + 2)  # +2 因为rank从1开始
        
        # DCG
        dcg = (relevance_norm * discount).sum()
        
        # Ideal DCG（理想排序）
        sorted_relevance, _ = torch.sort(relevance_norm, descending=True)
        ideal_ranks = torch.arange(1, n + 1, dtype=torch.float32, device=scores.device)
        ideal_discount = 1.0 / torch.log2(ideal_ranks + 1)
        idcg = (sorted_relevance * ideal_discount).sum()
        
        # NDCG
        ndcg = dcg / (idcg + 1e-10)
        
        # 损失 = 1 - NDCG
        loss = 1.0 - ndcg
        
        return self.weight * loss


class CombinedLossV2(nn.Module):
    """
    改进版组合损失 V2
    
    结合多种损失：
    1. IC损失 - 直接优化相关性
    2. Pointwise回归损失 - 学习绝对价值
    3. TopK Listwise损失 - 关注头部排序
    4. NDCG损失 - 信息检索风格的排序
    5. 能量正则化 - 稳定训练
    
    权重配置建议：
    - ic_weight=1.0（核心）
    - pointwise_weight=0.5（辅助回归）
    - topk_weight=0.3（头部关注）
    - ndcg_weight=0.2（排序质量）
    """
    
    def __init__(
        self,
        ic_weight: float = 1.0,
        pointwise_weight: float = 0.5,
        topk_weight: float = 0.3,
        ndcg_weight: float = 0.2,
        reg_weight: float = 0.01,
        top_k: int = 50,
        huber_delta: float = 1.0
    ):
        super().__init__()
        
        self.ic_weight = ic_weight
        self.pointwise_weight = pointwise_weight
        self.topk_weight = topk_weight
        self.ndcg_weight = ndcg_weight
        self.reg_weight = reg_weight
        
        self.ic_loss = ICLoss()
        self.pointwise_loss = PointwiseRegressionLoss(delta=huber_delta)
        self.topk_loss = TopKListwiseLoss(top_k=top_k)
        self.ndcg_loss = NDCGLoss()
        self.reg_loss = EnergyRegularization(weight=reg_weight)
    
    def forward(
        self,
        energy: torch.Tensor,
        returns: torch.Tensor,
        e_winner: Optional[torch.Tensor] = None,
        e_loser: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            energy: 预测能量 [batch]
            returns: 实际收益 [batch]
            e_winner, e_loser: Pairwise样本（可选）
        
        Returns:
            losses: 各项损失的字典
        """
        losses = {}
        
        energy = energy.flatten()
        returns = returns.flatten()
        
        # 1. IC损失（核心）
        # 注意：IC计算用的是分数(-energy)与收益的相关性
        ic_loss, ic = self.ic_loss(-energy, returns)
        losses['ic_loss'] = ic_loss
        losses['ic'] = ic
        
        # 2. Pointwise回归损失
        pointwise = self.pointwise_loss(energy, returns)
        losses['pointwise'] = pointwise
        
        # 3. TopK Listwise损失
        topk = self.topk_loss(energy, returns)
        losses['topk'] = topk
        
        # 4. NDCG损失
        ndcg = self.ndcg_loss(energy, returns)
        losses['ndcg'] = ndcg
        
        # 5. 能量正则化
        reg = self.reg_loss(energy)
        losses['reg'] = reg
        
        # 总损失
        total = (
            self.ic_weight * ic_loss +
            self.pointwise_weight * pointwise +
            self.topk_weight * topk +
            self.ndcg_weight * ndcg +
            reg
        )
        losses['total'] = total
        
        return losses


class AdaptiveMarginRankingLoss(nn.Module):
    """
    自适应Margin排序损失 - 改进版
    
    改进点：
    1. Margin随收益差距动态调整
    2. 对头部样本给予更大权重
    3. 忽略"垃圾vs垃圾"的pair
    """
    
    def __init__(
        self,
        base_margin: float = 0.3,
        margin_scale: float = 2.0,
        min_return_diff: float = 0.01,  # 忽略收益差距小于1%的pair
        head_weight: float = 2.0  # 头部样本权重
    ):
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.min_return_diff = min_return_diff
        self.head_weight = head_weight
    
    def forward(
        self,
        e_winner: torch.Tensor,
        e_loser: torch.Tensor,
        winner_return: torch.Tensor,
        loser_return: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            e_winner, e_loser: Winner和Loser的能量
            winner_return, loser_return: 对应的收益
        
        Returns:
            loss: 加权排序损失
        """
        return_diff = winner_return - loser_return
        
        # 1. 过滤掉"垃圾vs垃圾"的pair
        valid_mask = return_diff > self.min_return_diff
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=e_winner.device, requires_grad=True)
        
        # 2. 动态margin：收益差距越大，margin越大
        adaptive_margin = self.base_margin + self.margin_scale * return_diff[valid_mask]
        
        # 3. 计算排序损失
        ranking_loss = F.relu(
            e_winner[valid_mask] - e_loser[valid_mask] + adaptive_margin
        )
        
        # 4. 头部加权：winner收益越高，权重越大
        # 使用softmax转换收益为权重
        winner_weights = F.softmax(winner_return[valid_mask], dim=0)
        weighted_loss = (ranking_loss * winner_weights * len(winner_weights)).mean()
        
        return weighted_loss


# ============================================================================
# V3 损失函数组件 - 包含Bear贡献约束
# ============================================================================

class ICLossImproved(nn.Module):
    """
    改进的IC Loss
    
    改进点：
    1. 使用per-batch归一化，更稳定
    2. 添加方向一致性奖励
    """
    
    def __init__(self, direction_bonus: float = 0.1):
        super().__init__()
        self.direction_bonus = direction_bonus
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        计算负IC作为损失
        
        IC = corr(pred, target)
        Loss = -IC + direction_penalty
        """
        # 去除NaN
        mask = ~(torch.isnan(predictions) | torch.isnan(targets))
        if mask.sum() < 10:
            return torch.tensor(0.0, device=predictions.device), 0.0
        
        pred = predictions[mask]
        tgt = targets[mask]
        
        # 标准化
        pred_centered = pred - pred.mean()
        tgt_centered = tgt - tgt.mean()
        
        pred_std = pred_centered.std() + 1e-8
        tgt_std = tgt_centered.std() + 1e-8
        
        # IC = 相关系数
        ic = (pred_centered * tgt_centered).mean() / (pred_std * tgt_std)
        
        # 方向一致性：预测和目标同号的比例
        direction_match = ((pred > pred.median()) == (tgt > tgt.median())).float().mean()
        direction_bonus = self.direction_bonus * (direction_match - 0.5)
        
        # 损失 = 负IC - 方向奖励
        loss = -ic - direction_bonus
        
        return loss, ic.item()


class BearContributionConstraint(nn.Module):
    """
    Bear贡献约束
    
    确保Bear在总能量中占有一定比例，防止被"关闭"
    
    核心思想：
    - 如果Bear比例低于min_ratio，施加重惩罚
    - 否则轻微惩罚偏离目标比例
    """
    
    def __init__(
        self, 
        target_ratio: float = 0.3,
        min_ratio: float = 0.1,
        weight: float = 1.0
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.min_ratio = min_ratio
        self.weight = weight
    
    def forward(self, E_bull: torch.Tensor, E_bear: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        计算Bear贡献约束损失
        
        Args:
            E_bull: Bull能量 [batch]
            E_bear: Bear能量 [batch]
        
        Returns:
            loss: 约束损失
            ratio: 当前Bear贡献比例
        """
        E_bull_abs = E_bull.abs()
        E_bear_abs = E_bear.abs()
        
        total = E_bull_abs + E_bear_abs + 1e-6
        bear_ratio = (E_bear_abs / total).mean()
        
        # 如果Bear比例低于最小值，重惩罚
        if bear_ratio < self.min_ratio:
            loss = self.weight * 10.0 * (self.min_ratio - bear_ratio) ** 2
        else:
            # 否则，轻微惩罚偏离目标
            loss = self.weight * (bear_ratio - self.target_ratio) ** 2
        
        return loss, bear_ratio.item()


class MacroModulationBalance(nn.Module):
    """
    宏观调制平衡约束
    
    防止α和β差距过大，确保Bull和Bear都有贡献
    """
    
    def __init__(self, max_diff: float = 1.0, weight: float = 0.5):
        super().__init__()
        self.max_diff = max_diff
        self.weight = weight
    
    def forward(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        计算平衡约束损失
        """
        diff = (alpha - beta).abs().mean()
        
        # 如果差距超过max_diff，施加惩罚
        excess = F.relu(diff - self.max_diff)
        loss = self.weight * excess ** 2
        
        return loss


class CombinedLossV3(nn.Module):
    """
    V3综合损失函数
    
    组成：
    1. IC Loss - 核心，最大化预测与收益的相关性
    2. Pointwise Loss - 学习绝对价值
    3. Bear贡献约束 - 确保Bear不被关闭（关键改进）
    4. 宏观平衡约束 - 防止α/β极端值
    5. 能量正则化 - 防止数值爆炸
    
    与V2的区别：
    - 添加Bear贡献约束，权重较大
    - 添加α/β平衡约束
    - 移除NDCG损失（对IC帮助不大）
    """
    
    def __init__(
        self,
        ic_weight: float = 1.0,
        pointwise_weight: float = 0.3,
        bear_constraint_weight: float = 2.0,  # 较大权重，确保Bear不被关闭
        macro_balance_weight: float = 0.5,
        reg_weight: float = 0.01,
        target_bear_ratio: float = 0.3,
        min_bear_ratio: float = 0.1,
        huber_delta: float = 1.0
    ):
        super().__init__()
        
        self.ic_weight = ic_weight
        self.pointwise_weight = pointwise_weight
        self.bear_constraint_weight = bear_constraint_weight
        self.macro_balance_weight = macro_balance_weight
        self.reg_weight = reg_weight
        
        self.ic_loss = ICLossImproved()
        self.pointwise_loss = PointwiseRegressionLoss(delta=huber_delta)
        self.bear_constraint = BearContributionConstraint(
            target_ratio=target_bear_ratio,
            min_ratio=min_bear_ratio,
            weight=bear_constraint_weight
        )
        self.macro_balance = MacroModulationBalance(
            max_diff=1.0,
            weight=macro_balance_weight
        )
    
    def forward(
        self,
        energy: torch.Tensor,
        returns: torch.Tensor,
        components: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            energy: 能量值 [batch]
            returns: 未来收益 [batch]
            components: 能量组件字典（来自model的return_components=True）
        
        Returns:
            losses: 各项损失的字典
        """
        losses = {}
        
        energy = energy.flatten()
        returns = returns.flatten()
        scores = -energy  # 分数 = 负能量
        
        total_loss = torch.tensor(0.0, device=energy.device)
        
        # 1. IC Loss
        ic_loss, ic_value = self.ic_loss(scores, returns)
        total_loss = total_loss + self.ic_weight * ic_loss
        losses['ic_loss'] = ic_loss
        losses['ic'] = torch.tensor(ic_value)
        
        # 2. Pointwise Loss
        pointwise_loss = self.pointwise_loss(energy, returns)
        total_loss = total_loss + self.pointwise_weight * pointwise_loss
        losses['pointwise'] = pointwise_loss
        
        # 3. Bear贡献约束
        if components is not None and 'E_bull' in components and 'E_bear' in components:
            bear_loss, bear_ratio = self.bear_constraint(
                components['E_bull'], 
                components['E_bear']
            )
            total_loss = total_loss + bear_loss
            losses['bear_constraint'] = bear_loss
            losses['bear_ratio'] = torch.tensor(bear_ratio)
        
        # 4. 宏观平衡约束
        if components is not None and 'alpha_market' in components and 'beta_risk' in components:
            macro_loss = self.macro_balance(
                components['alpha_market'],
                components['beta_risk']
            )
            total_loss = total_loss + macro_loss
            losses['macro_balance'] = macro_loss
        
        # 5. 能量正则化
        reg_loss = self.reg_weight * energy.pow(2).mean()
        total_loss = total_loss + reg_loss
        losses['reg'] = reg_loss
        
        losses['total'] = total_loss
        
        return losses


def test_losses():
    """测试损失函数"""
    print("\n" + "=" * 70)
    print("    测试损失函数")
    print("=" * 70)
    
    batch_size = 100
    energy = torch.randn(batch_size) * 5
    returns = torch.randn(batch_size) * 0.1
    
    # 测试V2损失
    print("\n>>> 测试 CombinedLossV2")
    loss_v2 = CombinedLossV2()
    result_v2 = loss_v2(energy, returns)
    print(f"    IC: {result_v2['ic'].item():.4f}")
    print(f"    Total Loss: {result_v2['total'].item():.4f}")
    
    # 测试V3损失
    print("\n>>> 测试 CombinedLossV3")
    loss_v3 = CombinedLossV3()
    
    # 模拟组件（正常情况）
    components_normal = {
        'E_bull': torch.abs(torch.randn(batch_size)) * 5,
        'E_bear': torch.abs(torch.randn(batch_size)) * 3,
        'alpha_market': torch.ones(batch_size) * 1.2,
        'beta_risk': torch.ones(batch_size) * 0.9,
    }
    
    # 模拟组件（Bear被关闭）
    components_bad = {
        'E_bull': torch.abs(torch.randn(batch_size)) * 8,
        'E_bear': torch.abs(torch.randn(batch_size)) * 0.03,
        'alpha_market': torch.ones(batch_size) * 2.8,
        'beta_risk': torch.ones(batch_size) * 0.3,
    }
    
    result_normal = loss_v3(energy, returns, components_normal)
    result_bad = loss_v3(energy, returns, components_bad)
    
    print(f"\n  正常情况:")
    print(f"    Bear ratio: {result_normal['bear_ratio'].item():.4f}")
    print(f"    Total Loss: {result_normal['total'].item():.4f}")
    
    print(f"\n  Bear被关闭时:")
    print(f"    Bear ratio: {result_bad['bear_ratio'].item():.4f}")
    print(f"    Total Loss: {result_bad['total'].item():.4f}")
    
    assert result_bad['total'] > result_normal['total'], "异常情况损失应该更大"
    print(f"\n  ✓ V3损失函数正确惩罚了Bear被关闭的情况")
    print(f"    异常/正常 = {result_bad['total'].item() / result_normal['total'].item():.2f}x")
    
    print("\n✓ 所有损失函数测试通过")


if __name__ == "__main__":
    test_losses()


# ============================================================================
# V4 损失函数组件 - 直接IC和Rank IC优化
# ============================================================================

class DirectICLossV4(nn.Module):
    """
    直接IC损失 V4
    
    简化版本，只关注IC最大化
    """
    
    def __init__(self, reg_weight: float = 0.01):
        super().__init__()
        self.reg_weight = reg_weight
    
    def forward(
        self,
        energy: torch.Tensor,
        returns: torch.Tensor,
        components: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        
        energy = energy.flatten()
        returns = returns.flatten()
        
        mask = ~(torch.isnan(energy) | torch.isnan(returns))
        if mask.sum() < 10:
            losses['total'] = torch.tensor(0.0, device=energy.device, requires_grad=True)
            losses['ic'] = torch.tensor(0.0)
            return losses
        
        energy_clean = energy[mask]
        returns_clean = returns[mask]
        
        scores = -energy_clean
        
        scores_centered = scores - scores.mean()
        returns_centered = returns_clean - returns_clean.mean()
        
        scores_std = scores_centered.std() + 1e-8
        returns_std = returns_centered.std() + 1e-8
        
        ic = (scores_centered * returns_centered).mean() / (scores_std * returns_std)
        
        ic_loss = -ic
        reg_loss = self.reg_weight * energy_clean.pow(2).mean()
        
        total_loss = ic_loss + reg_loss
        
        losses['ic_loss'] = ic_loss
        losses['ic'] = ic
        losses['reg'] = reg_loss
        losses['total'] = total_loss
        
        return losses


class RankICLossV4(nn.Module):
    """
    Rank IC损失 V4（使用可微分的排序代理）
    
    优化排序相关性而不是线性相关性
    """
    
    def __init__(self, temperature: float = 1.0, reg_weight: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.reg_weight = reg_weight
    
    def soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """可微分的软排序"""
        n = x.size(0)
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        soft_greater = torch.sigmoid(diff / self.temperature)
        ranks = soft_greater.sum(dim=1)
        return ranks
    
    def forward(
        self,
        energy: torch.Tensor,
        returns: torch.Tensor,
        components: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """计算Rank IC损失"""
        losses = {}
        
        energy = energy.flatten()
        returns = returns.flatten()
        
        mask = ~(torch.isnan(energy) | torch.isnan(returns))
        if mask.sum() < 10:
            losses['total'] = torch.tensor(0.0, device=energy.device, requires_grad=True)
            losses['ic'] = torch.tensor(0.0)
            losses['rank_ic'] = torch.tensor(0.0)
            return losses
        
        energy_clean = energy[mask]
        returns_clean = returns[mask]
        
        scores = -energy_clean
        
        score_ranks = self.soft_rank(scores)
        return_ranks = self.soft_rank(returns_clean)
        
        score_ranks_centered = score_ranks - score_ranks.mean()
        return_ranks_centered = return_ranks - return_ranks.mean()
        
        score_std = score_ranks_centered.std() + 1e-8
        return_std = return_ranks_centered.std() + 1e-8
        
        rank_ic = (score_ranks_centered * return_ranks_centered).mean() / (score_std * return_std)
        
        scores_centered = scores - scores.mean()
        returns_centered = returns_clean - returns_clean.mean()
        ic = (scores_centered * returns_centered).mean() / (scores_centered.std() + 1e-8) / (returns_centered.std() + 1e-8)
        
        rank_loss = -rank_ic
        reg_loss = self.reg_weight * energy_clean.pow(2).mean()
        
        total_loss = rank_loss + reg_loss
        
        losses['rank_ic_loss'] = rank_loss
        losses['rank_ic'] = rank_ic
        losses['ic'] = ic
        losses['reg'] = reg_loss
        losses['total'] = total_loss
        
        return losses


class CombinedLossV4(nn.Module):
    """
    V4组合损失
    
    简化版本：
    - IC损失（主要）
    - Rank IC损失（辅助）
    - 正则化
    
    不再包含Bear约束等，因为V4模型不区分Bull/Bear
    """
    
    def __init__(
        self,
        ic_weight: float = 1.0,
        rank_weight: float = 0.5,
        reg_weight: float = 0.01,
        rank_temperature: float = 1.0
    ):
        super().__init__()
        
        self.ic_weight = ic_weight
        self.rank_weight = rank_weight
        self.reg_weight = reg_weight
        
        self.ic_loss_fn = DirectICLossV4(reg_weight=0)
        self.rank_loss_fn = RankICLossV4(temperature=rank_temperature, reg_weight=0)
    
    def forward(
        self,
        energy: torch.Tensor,
        returns: torch.Tensor,
        components: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """计算组合损失"""
        ic_result = self.ic_loss_fn(energy, returns)
        rank_result = self.rank_loss_fn(energy, returns)
        
        reg_loss = self.reg_weight * energy.pow(2).mean()
        
        total = (
            self.ic_weight * ic_result['ic_loss'] +
            self.rank_weight * rank_result['rank_ic_loss'] +
            reg_loss
        )
        
        return {
            'total': total,
            'ic_loss': ic_result['ic_loss'],
            'rank_loss': rank_result['rank_ic_loss'],
            'ic': ic_result['ic'],
            'rank_ic': rank_result['rank_ic'],
            'reg': reg_loss
        }


# ================================================================================
# V4.2 损失函数 - 鲁棒性增强版
# ================================================================================

class TieredICLoss(nn.Module):
    """
    分层IC损失 - 已弃用，会导致Rank IC变负
    保留代码供参考，但不推荐使用
    """
    
    def __init__(
        self,
        n_tiers: int = 5,
        tier_weights: Optional[list] = None,
        global_ic_weight: float = 0.5
    ):
        super().__init__()
        
        self.n_tiers = n_tiers
        
        # 默认权重：各层等权
        if tier_weights is None:
            self.tier_weights = [1.0 / n_tiers] * n_tiers
        else:
            self.tier_weights = tier_weights
        
        self.global_ic_weight = global_ic_weight
    
    def forward(
        self,
        scores: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算分层IC损失
        
        Args:
            scores: 模型预测分数 (越高越好)
            returns: 实际收益率
        
        Returns:
            包含total_loss和各分层IC的字典
        """
        scores = scores.flatten()
        returns = returns.flatten()
        
        # 清洗数据
        mask = ~(torch.isnan(scores) | torch.isnan(returns))
        if mask.sum() < 20:
            return {
                'total': torch.tensor(0.0, device=scores.device, requires_grad=True),
                'global_ic': torch.tensor(0.0),
                'tier_ics': [torch.tensor(0.0)] * self.n_tiers
            }
        
        scores_clean = scores[mask]
        returns_clean = returns[mask]
        
        n_samples = len(scores_clean)
        
        # 1. 计算全局IC
        global_ic = self._compute_ic(scores_clean, returns_clean)
        
        # 2. 计算分层IC
        # 按收益率分层
        _, indices = torch.sort(returns_clean)
        tier_size = n_samples // self.n_tiers
        
        tier_ics = []
        for i in range(self.n_tiers):
            start_idx = i * tier_size
            end_idx = start_idx + tier_size if i < self.n_tiers - 1 else n_samples
            
            tier_indices = indices[start_idx:end_idx]
            
            if len(tier_indices) < 10:
                tier_ics.append(torch.tensor(0.0, device=scores.device))
                continue
            
            tier_scores = scores_clean[tier_indices]
            tier_returns = returns_clean[tier_indices]
            
            tier_ic = self._compute_ic(tier_scores, tier_returns)
            tier_ics.append(tier_ic)
        
        # 3. 加权平均
        tiered_ic = sum(w * ic for w, ic in zip(self.tier_weights, tier_ics))
        
        # 4. 组合损失
        total_ic = self.global_ic_weight * global_ic + (1 - self.global_ic_weight) * tiered_ic
        
        # IC越大越好，所以损失取负
        total_loss = -total_ic
        
        return {
            'total': total_loss,
            'global_ic': global_ic.detach(),
            'tiered_ic': tiered_ic.detach(),
            'tier_ics': [ic.detach() for ic in tier_ics]
        }
    
    def _compute_ic(self, scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算IC（Pearson相关系数）"""
        scores_centered = scores - scores.mean()
        returns_centered = returns - returns.mean()
        
        scores_std = scores_centered.std() + 1e-8
        returns_std = returns_centered.std() + 1e-8
        
        ic = (scores_centered * returns_centered).mean() / (scores_std * returns_std)
        
        return ic


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失
    
    确保能量分量符合物理直觉：
    1. E_bull, E_bear, E_heat >= 0
    2. Heat不应该主导能量（贡献有上限）
    3. 宏观调制系数应该有意义地变化
    """
    
    def __init__(
        self,
        heat_contribution_cap: float = 0.3,  # Heat贡献不超过总能量的30%
        modulation_variance_target: float = 0.1  # 调制系数应该有一定方差
    ):
        super().__init__()
        
        self.heat_contribution_cap = heat_contribution_cap
        self.modulation_variance_target = modulation_variance_target
    
    def forward(
        self,
        components: Dict[str, torch.Tensor],
        energy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算物理约束损失"""
        losses = {}
        
        # 1. 非负约束（应该已经通过Softplus满足，这里是额外检查）
        if 'E_bull' in components:
            neg_bull = F.relu(-components['E_bull']).mean()
            neg_bear = F.relu(-components['E_bear']).mean()
            neg_heat = F.relu(-components.get('E_heat', torch.zeros_like(energy))).mean()
            losses['non_negative'] = neg_bull + neg_bear + neg_heat
        
        # 2. Heat贡献约束
        if 'E_heat' in components and 'E_bull' in components:
            total_magnitude = (
                components['E_bull'].abs() + 
                components['E_bear'].abs() + 
                components['E_heat'].abs() + 1e-6
            )
            heat_ratio = components['E_heat'].abs() / total_magnitude
            
            # 惩罚Heat贡献过大
            excess_heat = F.relu(heat_ratio - self.heat_contribution_cap)
            losses['heat_cap'] = excess_heat.mean()
        
        # 3. 调制系数方差约束
        if 'alpha_market' in components:
            alpha_var = components['alpha_market'].var()
            beta_var = components['beta_risk'].var()
            
            # 鼓励调制系数有一定变化（不是常数）
            var_deficit = F.relu(self.modulation_variance_target - alpha_var)
            var_deficit = var_deficit + F.relu(self.modulation_variance_target - beta_var)
            losses['modulation_variance'] = var_deficit
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


class VolatilityNeutralLoss(nn.Module):
    """
    波动率中性化损失
    
    惩罚预测分数和波动率的相关性，防止模型简单地选择高波动股票
    """
    
    def __init__(self, target_correlation: float = 0.0):
        super().__init__()
        self.target_correlation = target_correlation
    
    def forward(
        self,
        scores: torch.Tensor,
        volatility: torch.Tensor
    ) -> torch.Tensor:
        """计算波动率中性化损失"""
        scores = scores.flatten()
        volatility = volatility.flatten()
        
        # 清洗数据
        mask = ~(torch.isnan(scores) | torch.isnan(volatility))
        if mask.sum() < 10:
            return torch.tensor(0.0, device=scores.device)
        
        scores_clean = scores[mask]
        vol_clean = volatility[mask]
        
        # 计算相关性
        scores_centered = scores_clean - scores_clean.mean()
        vol_centered = vol_clean - vol_clean.mean()
        
        corr = (scores_centered * vol_centered).mean() / (
            scores_centered.std() + 1e-8
        ) / (vol_centered.std() + 1e-8)
        
        # 惩罚偏离目标相关性（通常为0，即中性）
        loss = (corr - self.target_correlation).pow(2)
        
        return loss


class CombinedLossV4_2(nn.Module):
    """
    V4.2 组合损失 - 修复版
    
    组成：
    1. IC损失（主要）
    2. Rank IC损失（关键！确保排序正确）
    3. 物理约束损失
    4. 波动率中性化损失（新增）
    5. 方向正则化损失
    
    【重要修复】：移除分层IC，使用全局IC+RankIC
    """
    
    def __init__(
        self,
        ic_weight: float = 1.0,
        rank_weight: float = 0.5,        # 增加Rank IC权重！
        physics_weight: float = 0.1,
        vol_neutral_weight: float = 0.2,  # 新增波动率中性化
        direction_reg_weight: float = 0.01,
        n_tiers: int = 5,                 # 保留参数但不使用
        tiered_ic_weight: float = 0.0     # 禁用分层IC！
    ):
        super().__init__()
        
        self.ic_weight = ic_weight
        self.rank_weight = rank_weight
        self.physics_weight = physics_weight
        self.vol_neutral_weight = vol_neutral_weight
        self.direction_reg_weight = direction_reg_weight
        
        # 不再使用分层IC
        self.physics_loss = PhysicsConstraintLoss()
        self.vol_neutral_loss = VolatilityNeutralLoss()
    
    def forward(
        self,
        energy: torch.Tensor,
        returns: torch.Tensor,
        components: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        volatility: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """计算组合损失"""
        losses = {}
        
        # 1. 分数 = -能量（能量越低越好）
        scores = -energy
        
        # 2. 全局IC损失（不再使用分层）
        ic = self._compute_ic(scores, returns)
        losses['ic'] = ic
        losses['ic_loss'] = -ic * self.ic_weight
        losses['global_ic'] = ic.detach()
        losses['tiered_ic'] = ic.detach()  # 兼容旧接口
        
        # 3. Rank IC损失（关键！）
        rank_ic = self._compute_rank_ic(scores, returns)
        losses['rank_ic'] = rank_ic
        losses['rank_loss'] = -rank_ic * self.rank_weight
        
        # 4. 物理约束损失
        if components is not None and self.physics_weight > 0:
            physics_result = self.physics_loss(components, energy)
            losses['physics_loss'] = physics_result['total'] * self.physics_weight
        else:
            losses['physics_loss'] = torch.tensor(0.0, device=energy.device)
        
        # 5. 波动率中性化损失（新增）
        if volatility is not None and self.vol_neutral_weight > 0:
            losses['vol_neutral_loss'] = self.vol_neutral_loss(scores, volatility) * self.vol_neutral_weight
        else:
            # 如果没有波动率数据，尝试从components中获取E_heat作为代理
            if components is not None and 'E_heat' in components:
                # E_heat可以作为波动率的代理
                losses['vol_neutral_loss'] = self.vol_neutral_loss(scores, components['E_heat']) * self.vol_neutral_weight
            else:
                losses['vol_neutral_loss'] = torch.tensor(0.0, device=energy.device)
        
        # 6. 方向正则化损失
        if model is not None and hasattr(model, 'get_direction_regularization'):
            losses['direction_reg'] = model.get_direction_regularization()
        else:
            losses['direction_reg'] = torch.tensor(0.0, device=energy.device)
        
        # 7. 总损失
        losses['total'] = (
            losses['ic_loss'] +
            losses['rank_loss'] +
            losses['physics_loss'] +
            losses['vol_neutral_loss'] +
            losses['direction_reg']
        )
        
        return losses
    
    def _compute_ic(self, scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算IC（Pearson相关系数）"""
        scores = scores.flatten()
        returns = returns.flatten()
        
        mask = ~(torch.isnan(scores) | torch.isnan(returns))
        if mask.sum() < 10:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        scores_clean = scores[mask]
        returns_clean = returns[mask]
        
        scores_centered = scores_clean - scores_clean.mean()
        returns_centered = returns_clean - returns_clean.mean()
        
        scores_std = scores_centered.std() + 1e-8
        returns_std = returns_centered.std() + 1e-8
        
        ic = (scores_centered * returns_centered).mean() / (scores_std * returns_std)
        
        return ic
    
    def _compute_rank_ic(self, scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算Rank IC（Spearman相关）"""
        scores = scores.flatten()
        returns = returns.flatten()
        
        mask = ~(torch.isnan(scores) | torch.isnan(returns))
        if mask.sum() < 10:
            return torch.tensor(0.0, device=scores.device)
        
        scores_clean = scores[mask]
        returns_clean = returns[mask]
        
        # 软排序
        score_ranks = self._soft_rank(scores_clean)
        return_ranks = self._soft_rank(returns_clean)
        
        # 计算相关
        score_ranks_centered = score_ranks - score_ranks.mean()
        return_ranks_centered = return_ranks - return_ranks.mean()
        
        rank_ic = (score_ranks_centered * return_ranks_centered).mean() / (
            score_ranks_centered.std() + 1e-8
        ) / (return_ranks_centered.std() + 1e-8)
        
        return rank_ic
    
    def _soft_rank(self, x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """可微分的软排序"""
        n = x.size(0)
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        soft_greater = torch.sigmoid(diff / temperature)
        ranks = soft_greater.sum(dim=1)
        return ranks
