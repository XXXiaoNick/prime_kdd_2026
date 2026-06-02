"""
baseline 模型实现集合。

这里收纳论文中需要复现的序列模型、图模型与强化学习近似基线，
并保持统一的前向接口，方便 runner 复用同一套训练和回测框架。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, GPT2Config, GPT2Model
except Exception:  # pragma: no cover
    AutoModel = None
    GPT2Config = None
    GPT2Model = None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)


def patchify_time(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected [B, T, F], got {tuple(x.shape)}")
    bsz, seq_len, feat_dim = x.shape
    if seq_len < patch_len:
        pad = torch.zeros(bsz, patch_len - seq_len, feat_dim, dtype=x.dtype, device=x.device)
        x = torch.cat([pad, x], dim=1)
        seq_len = patch_len
    x = x.transpose(1, 2)  # [B, F, T]
    patches = x.unfold(dimension=-1, size=patch_len, step=stride)  # [B, F, N, patch]
    patches = patches.permute(0, 2, 1, 3).contiguous()
    return patches.view(bsz, patches.size(1), feat_dim * patch_len)


def _make_transformer_encoder(
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def _build_hf_backbone(
    backbone_name: Optional[str],
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    freeze_backbone: bool,
) -> Tuple[nn.Module, int, str]:
    if AutoModel is not None and backbone_name:
        try:
            model = AutoModel.from_pretrained(backbone_name)
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            hidden_size = getattr(model.config, "hidden_size", hidden_dim)
            return model, int(hidden_size), f"pretrained:{backbone_name}"
        except Exception:
            pass

    if GPT2Model is None or GPT2Config is None:
        layer = _make_transformer_encoder(hidden_dim, num_heads, num_layers, 0.1)
        return layer, hidden_dim, "fallback:transformer"

    gpt_config = GPT2Config(
        vocab_size=1,
        n_embd=hidden_dim,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=512,
        n_ctx=512,
    )
    model = GPT2Model(gpt_config)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    return model, hidden_dim, "fallback:random-gpt2"


def _run_hf_backbone(backbone: nn.Module, inputs_embeds: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "forward") and backbone.__class__.__name__ == "TransformerEncoder":
        return backbone(inputs_embeds)
    outputs = backbone(inputs_embeds=inputs_embeds)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        last_hidden = self.norm(h[:, -1])
        return self.head(last_hidden).squeeze(-1), {}


class ALSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = TemporalAttentionPool(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        pooled = self.norm(self.pool(h))
        return self.head(pooled).squeeze(-1), {}


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        h, _ = self.gru(x)
        pooled = self.norm(h[:, -1])
        return self.head(pooled).squeeze(-1), {}


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        self.encoder = _make_transformer_encoder(hidden_dim, num_heads, num_layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        pooled = self.norm(z[:, -1])
        return self.head(pooled).squeeze(-1), {}


class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        patch_len: int,
        stride: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        self.encoder = _make_transformer_encoder(hidden_dim, num_heads, num_layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor):
        bsz, seq_len, feat_dim = x.shape
        if seq_len < self.patch_len:
            pad = torch.zeros(bsz, self.patch_len - seq_len, feat_dim, dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=1)
        channel_first = x.transpose(1, 2)  # [B, F, T]
        patches = channel_first.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(bsz * feat_dim, patches.size(-2), self.patch_len)
        z = self.patch_proj(patches)
        z = self.pos(z)
        z = self.encoder(z)
        z = self.norm(z.mean(dim=1)).view(bsz, feat_dim, -1).mean(dim=1)
        return self.head(z).squeeze(-1), {}


class GPT4TSBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        patch_len: int,
        stride: int,
        dropout: float,
        hf_backbone: Optional[str],
        freeze_backbone: bool,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.backbone, backbone_dim, self.backbone_source = _build_hf_backbone(
            hf_backbone,
            hidden_dim,
            num_heads,
            num_layers,
            freeze_backbone,
        )
        self.input_proj = nn.Linear(input_dim * patch_len, backbone_dim)
        self.pos = PositionalEncoding(backbone_dim)
        self.norm = nn.LayerNorm(backbone_dim)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        patches = patchify_time(x, self.patch_len, self.stride)
        z = self.input_proj(patches)
        z = self.pos(z)
        z = _run_hf_backbone(self.backbone, z)
        pooled = self.norm(z.mean(dim=1))
        return self.head(pooled).squeeze(-1), {"backbone_source": self.backbone_source}


class TimeLLMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        patch_len: int,
        stride: int,
        prompt_length: int,
        dropout: float,
        hf_backbone: Optional[str],
        freeze_backbone: bool,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.prompt_length = prompt_length
        self.backbone, backbone_dim, self.backbone_source = _build_hf_backbone(
            hf_backbone,
            hidden_dim,
            num_heads,
            num_layers,
            freeze_backbone,
        )
        self.reprogram = nn.Sequential(
            nn.Linear(input_dim * patch_len, backbone_dim),
            nn.GELU(),
            nn.Linear(backbone_dim, backbone_dim),
        )
        self.prompt = nn.Parameter(torch.randn(1, prompt_length, backbone_dim) * 0.02)
        self.pos = PositionalEncoding(backbone_dim)
        self.norm = nn.LayerNorm(backbone_dim)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        patches = patchify_time(x, self.patch_len, self.stride)
        tokens = self.reprogram(patches)
        prompt = self.prompt.expand(tokens.size(0), -1, -1)
        z = torch.cat([prompt, tokens], dim=1)
        z = self.pos(z)
        z = _run_hf_backbone(self.backbone, z)
        pooled = self.norm(z[:, self.prompt_length :].mean(dim=1))
        return self.head(pooled).squeeze(-1), {"backbone_source": self.backbone_source}


def build_similarity_graph(node_repr: torch.Tensor, industry_id: Optional[torch.Tensor] = None) -> torch.Tensor:
    normed = F.normalize(node_repr, dim=-1)
    adj = torch.relu(normed @ normed.transpose(0, 1))
    if industry_id is not None:
        same_industry = (industry_id.unsqueeze(0) == industry_id.unsqueeze(1)).float()
        adj = adj + 0.3 * same_industry
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj


class GraphMixBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.nei_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_repr: torch.Tensor, industry_id: Optional[torch.Tensor] = None):
        adj = build_similarity_graph(node_repr, industry_id=industry_id)
        mixed = self.self_proj(node_repr) + adj @ self.nei_proj(node_repr)
        mixed = self.norm(node_repr + self.dropout(F.gelu(mixed)))
        return mixed, adj


class AlphaStockBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.asset_encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.temporal_pool = TemporalAttentionPool(hidden_dim)
        self.asset_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, industry_id: Optional[torch.Tensor] = None):
        h, _ = self.asset_encoder(x)
        asset_repr = self.temporal_pool(h)  # [N, H]
        attn_input = asset_repr.unsqueeze(0)
        attn_output, attn_weights = self.asset_attn(attn_input, attn_input, attn_input, need_weights=True)
        fused = self.norm(attn_output.squeeze(0) + asset_repr)
        score = self.score_head(fused).squeeze(-1)
        return score, {"attention": attn_weights.squeeze(0)}


class DeepTraderBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.asset_encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.market_encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.graph_block = GraphMixBlock(hidden_dim, dropout)
        self.asset_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.market_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, industry_id: Optional[torch.Tensor] = None):
        asset_hidden, _ = self.asset_encoder(x)
        asset_repr = asset_hidden[:, -1]
        market_sequence = x.mean(dim=0, keepdim=True)
        market_hidden, _ = self.market_encoder(market_sequence)
        market_repr = market_hidden[:, -1]
        gated_asset = asset_repr * (1.0 + self.market_gate(market_repr))
        graph_repr, adj = self.graph_block(gated_asset, industry_id=industry_id)
        attn_out, attn_weights = self.asset_attn(
            graph_repr.unsqueeze(0),
            graph_repr.unsqueeze(0),
            graph_repr.unsqueeze(0),
            need_weights=True,
        )
        fused = graph_repr + attn_out.squeeze(0)
        score = self.score_head(fused).squeeze(-1)
        value = self.value_head(fused.mean(dim=0, keepdim=True)).squeeze(-1)
        return score, {"attention": attn_weights.squeeze(0), "value": value, "adjacency": adj}


class DeepPocketBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_pool = TemporalAttentionPool(input_dim)
        self.graph_block_1 = GraphMixBlock(hidden_dim, dropout)
        self.graph_block_2 = GraphMixBlock(hidden_dim, dropout)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, industry_id: Optional[torch.Tensor] = None):
        pooled = self.temporal_pool(x)
        node_repr = self.encoder(pooled)
        node_repr, adj_1 = self.graph_block_1(node_repr, industry_id=industry_id)
        node_repr, adj_2 = self.graph_block_2(node_repr, industry_id=industry_id)
        score = self.actor(node_repr).squeeze(-1)
        value = self.critic(node_repr.mean(dim=0, keepdim=True)).squeeze(-1)
        smoothness = ((adj_2 @ node_repr) - node_repr).pow(2).mean()
        return score, {"value": value, "adjacency": adj_2, "smoothness": smoothness, "adjacency_1": adj_1}


class AlphaGATBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.trend_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.seasonal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.cross_asset_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.graph_block = GraphMixBlock(hidden_dim, dropout)
        self.factor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cf = x.transpose(1, 2)
        trend = F.avg_pool1d(x_cf, kernel_size=3, stride=1, padding=1)
        seasonal = x_cf - trend
        return trend, seasonal

    def forward(self, x: torch.Tensor, industry_id: Optional[torch.Tensor] = None):
        trend, seasonal = self._decompose(x)
        trend_repr = self.trend_conv(trend).mean(dim=-1)
        seasonal_repr = self.seasonal_conv(seasonal).mean(dim=-1)
        factor_repr = self.factor_head(trend_repr + seasonal_repr)
        attn_out, attn_weights = self.cross_asset_attn(
            factor_repr.unsqueeze(0),
            factor_repr.unsqueeze(0),
            factor_repr.unsqueeze(0),
            need_weights=True,
        )
        graph_repr, adj = self.graph_block(factor_repr + attn_out.squeeze(0), industry_id=industry_id)
        score = self.policy_head(graph_repr).squeeze(-1)
        return score, {"attention": attn_weights.squeeze(0), "adjacency": adj}


class StockFormerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_industries: int = 128,
    ):
        super().__init__()
        self.low_proj = nn.Linear(input_dim, hidden_dim)
        self.high_proj = nn.Linear(input_dim, hidden_dim)
        self.low_pos = PositionalEncoding(hidden_dim)
        self.high_pos = PositionalEncoding(hidden_dim)
        self.low_encoder = _make_transformer_encoder(hidden_dim, num_heads, num_layers, dropout)
        self.high_encoder = _make_transformer_encoder(hidden_dim, num_heads, num_layers, dropout)
        self.industry_embedding = nn.Embedding(max_industries, hidden_dim)
        self.asset_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cf = x.transpose(1, 2)
        low = F.avg_pool1d(x_cf, kernel_size=3, stride=1, padding=1)
        high = x_cf - low
        return low.transpose(1, 2), high.transpose(1, 2)

    def forward(self, x: torch.Tensor, industry_id: Optional[torch.Tensor] = None):
        low, high = self._decompose(x)
        low_repr = self.low_encoder(self.low_pos(self.low_proj(low))).mean(dim=1)
        high_repr = self.high_encoder(self.high_pos(self.high_proj(high))).mean(dim=1)
        asset_repr = low_repr + high_repr
        if industry_id is not None:
            clipped_industry = torch.clamp(industry_id, min=0, max=self.industry_embedding.num_embeddings - 1)
            asset_repr = asset_repr + self.industry_embedding(clipped_industry)
        attn_input = asset_repr.unsqueeze(0)
        attn_output, attn_weights = self.asset_attn(attn_input, attn_input, attn_input, need_weights=True)
        fused = self.norm(attn_output.squeeze(0) + asset_repr)
        score = self.score_head(fused).squeeze(-1)
        direction_logit = self.direction_head(fused).squeeze(-1)
        return score, {"direction_logit": direction_logit, "attention": attn_weights.squeeze(0)}


def build_baseline_model(spec, input_dim: int) -> nn.Module:
    key = spec.name
    if key == "lstm":
        return LSTMBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.dropout)
    if key == "alstm":
        return ALSTMBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.dropout)
    if key == "gru":
        return GRUBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.dropout)
    if key == "transformer":
        return TransformerBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.num_heads, spec.dropout)
    if key == "patchtst":
        return PatchTSTBaseline(
            input_dim,
            spec.hidden_dim,
            spec.num_layers,
            spec.num_heads,
            spec.patch_len,
            spec.stride,
            spec.dropout,
        )
    if key == "gpt4ts":
        return GPT4TSBaseline(
            input_dim,
            spec.hidden_dim,
            spec.num_layers,
            spec.num_heads,
            spec.patch_len,
            spec.stride,
            spec.dropout,
            spec.hf_backbone,
            spec.freeze_backbone,
        )
    if key == "time_llm":
        return TimeLLMBaseline(
            input_dim,
            spec.hidden_dim,
            spec.num_layers,
            spec.num_heads,
            spec.patch_len,
            spec.stride,
            spec.prompt_length,
            spec.dropout,
            spec.hf_backbone,
            spec.freeze_backbone,
        )
    if key == "alphastock":
        return AlphaStockBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.num_heads, spec.dropout)
    if key == "deeptrader":
        return DeepTraderBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.num_heads, spec.dropout)
    if key == "deeppocket":
        return DeepPocketBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.num_heads, spec.dropout)
    if key == "alphagat":
        return AlphaGATBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.num_heads, spec.dropout)
    if key == "stockformer":
        return StockFormerBaseline(input_dim, spec.hidden_dim, spec.num_layers, spec.num_heads, spec.dropout)
    raise KeyError(f"Unknown baseline model: {spec.name}")
