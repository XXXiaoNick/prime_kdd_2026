"""
================================================================================
配置管理模块 (v3) — 股票 + ETF 双模式
================================================================================
新增:
  1. mode 字段: "stock" (CSI500) 或 "etf" (A股ETF)
  2. ETF 专属数据路径
  3. ETF 特征配置（44个特征，与股票版维度对齐，语义替换）
  4. ETF 回测参数（佣金更低、无印花税）
  5. ETF crash_label 阈值调整（ETF波动率低于个股）

用法:
  # 股票模式（默认，与原版完全兼容）
  config = Config()

  # ETF 模式
  config = Config.etf_mode()

  # 动态切换
  config = Config()
  config.switch_to_etf()
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from market_profiles import (
    BASE_DATA_ROOT,
    apply_market_profile as apply_market_profile_to_config,
    available_market_profiles,
)


# ============================================================================
#  数据配置
# ============================================================================
@dataclass
class DataConfig:
    """数据配置 — 支持 stock / etf 双模式"""

    # 运行模式: "stock" 或 "etf"
    mode: str = "stock"
    market_profile: str = "csi500"
    market_label: str = "CSI 500"
    auto_mock_if_missing: bool = True

    # ======================== 股票模式路径 ========================
    data_root: Path = BASE_DATA_ROOT / "china"
    panel_path: Path = BASE_DATA_ROOT / "china" / "panel" / "panel_data_complete.parquet"

    stock_dir: Path = BASE_DATA_ROOT / "china" / "stock"
    valuation_dir: Path = BASE_DATA_ROOT / "china" / "valuation"
    financial_dir: Path = BASE_DATA_ROOT / "china" / "financial"
    capital_flow_dir: Path = BASE_DATA_ROOT / "china" / "capital_flow"
    chip_dir: Path = BASE_DATA_ROOT / "china" / "chip"
    margin_dir: Path = BASE_DATA_ROOT / "china" / "margin"
    northbound_dir: Path = BASE_DATA_ROOT / "china" / "northbound"
    macro_dir: Path = BASE_DATA_ROOT / "china" / "macro"
    index_dir: Path = BASE_DATA_ROOT / "china" / "index"
    industry_dir: Path = BASE_DATA_ROOT / "china" / "industry"
    market_dir: Path = BASE_DATA_ROOT / "china" / "market"
    meta_dir: Path = BASE_DATA_ROOT / "china" / "meta"

    # ======================== ETF 模式路径 ========================
    etf_data_root: Path = BASE_DATA_ROOT / "china_etf"
    etf_panel_path: Path = BASE_DATA_ROOT / "china_etf" / "panel" / "etf_panel_complete.parquet"

    etf_daily_dir: Path = BASE_DATA_ROOT / "china_etf" / "etf"
    etf_nav_dir: Path = BASE_DATA_ROOT / "china_etf" / "nav"
    etf_share_dir: Path = BASE_DATA_ROOT / "china_etf" / "share"
    etf_benchmark_dir: Path = BASE_DATA_ROOT / "china_etf" / "benchmark"
    etf_macro_dir: Path = BASE_DATA_ROOT / "china_etf" / "macro"
    etf_index_dir: Path = BASE_DATA_ROOT / "china_etf" / "index"
    etf_meta_dir: Path = BASE_DATA_ROOT / "china_etf" / "meta"

    # ======================== 时间划分 ========================
    train_start: str = "2008-01-01"
    train_end: str = "2017-12-31"
    valid_start: str = "2018-01-01"
    valid_end: str = "2019-12-31"
    test_start: str = "2020-01-01"
    test_end: str = "2020-12-31"

    # ======================== 标签定义 ========================
    forward_days: int = 5
    crash_threshold: float = -0.10       # 股票: -10%
    drawdown_threshold: float = 0.15     # 股票: 15%
    etf_crash_threshold: float = -0.08   # ETF: -8% (ETF波动率低于个股)
    etf_drawdown_threshold: float = 0.12 # ETF: 12%
    crash_relative_to_benchmark: bool = False

    # ======================== 预处理参数 ========================
    macro_lag: int = 1
    mad_threshold: float = 5.0

    # ======================== 便捷属性 ========================
    @property
    def active_panel_path(self) -> Path:
        """当前模式的面板数据路径"""
        primary = self.etf_panel_path if self.mode == "etf" else self.panel_path
        root = self.etf_data_root if self.mode == "etf" else self.data_root
        panel_dir = root / "panel"
        candidates = [
            primary,
            panel_dir / "etf_panel_complete.parquet",
            panel_dir / "etf_panel_complete.csv",
            panel_dir / "panel_data_complete.parquet",
            panel_dir / "panel_data_complete.csv",
            panel_dir / "panel_data.parquet",
            panel_dir / "panel_data.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return primary

    @property
    def active_data_root(self) -> Path:
        return self.etf_data_root if self.mode == "etf" else self.data_root

    @property
    def processed_cache_dir(self) -> Path:
        return self.active_data_root / "panel" / "processed_cache"

    @property
    def active_crash_threshold(self) -> float:
        return self.etf_crash_threshold if self.mode == "etf" else self.crash_threshold

    @property
    def active_drawdown_threshold(self) -> float:
        return self.etf_drawdown_threshold if self.mode == "etf" else self.drawdown_threshold


# ============================================================================
#  特征配置
# ============================================================================
@dataclass
class FeatureConfig:
    """
    特征配置 — 股票 / ETF 双模式

    设计原则:
      1. Bull/Bear/Friction/Macro 四组结构不变，维度完全对齐 (16+10+9+9=44)
      2. 模型架构 (GameEnergyModel) 不需要任何修改
      3. 只需要切换 feature_groups 中的字段名

    ETF 替换逻辑:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 股票特征              → ETF 替换                → 金融直觉          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ north_net_flow        → fund_flow_5d            │ 份额增=资金流入   │
    │ main_net_inflow       → fund_flow_20d           │ 20日累计流入      │
    │ roe_growth            → nav_growth_rate          │ 净值增长=表现好   │
    │ revenue_growth        → excess_return_20d        │ 超额收益=选对了   │
    │ profit_ratio          → alpha_60d                │ alpha>0=有能力    │
    │ chip_support          → share_growth_20d         │ 份额增长=支撑强   │
    │ winner_rate           → premium_positive_freq    │ 溢价频率=看涨     │
    │ pe_rank               → index_pe_rank            │ 标的指数PE排名    │
    │ pb_rank               → index_pb_rank            │ 标的指数PB排名    │
    │ trapped_ratio         → tracking_error_20d       │ 跟踪误差=风险     │
    │ dist_to_resistance    → premium_abs              │ 溢折价=定价异常   │
    │ debt_ratio            → share_outflow_rate       │ 资金撤出=看跌     │
    │ goodwill_risk         → nav_max_drawdown_60d     │ 净值回撤=风险     │
    │ asr                   → weighted_avg_cost        │ 加权成本=摩擦     │
    │ chip_concentration_ch → share_change_volatility  │ 份额波动=摩擦     │
    └──────────────────────────────────────────────────────────────────────┘
    """

    # ======================== 股票特征 (默认) ========================
    bull_features: List[str] = field(default_factory=lambda: [
        'momentum_5d', 'momentum_10d', 'momentum_20d',
        'rsi_6', 'rsi_14',
        'north_net_flow', 'main_net_inflow',
        'roe_growth', 'revenue_growth', 'profit_ratio',
        'price_vs_ma20', 'volume_ratio',
        'chip_support', 'winner_rate',
        'clv_20d_avg', 'upside_volume_ratio_20d',
    ])

    bear_features: List[str] = field(default_factory=lambda: [
        'pe_rank', 'pb_rank',
        'bias_20d', 'bias_60d',
        'trapped_ratio', 'dist_to_resistance',
        'cost_pressure', 'avg_cost_deviation',
        'debt_ratio', 'goodwill_risk',
    ])

    friction_features: List[str] = field(default_factory=lambda: [
        'turnover_rate', 'turnover_20d_avg',
        'volatility_20d', 'volatility_60d',
        'amplitude', 'amplitude_20d_avg',
        'asr', 'chip_concentration_change',
        'vol_price_divergence',
    ])

    macro_features: List[str] = field(default_factory=lambda: [
        'cpi_yoy', 'ppi_yoy', 'pmi', 'lpr_1y',
        'csi500_ret_20d', 'market_turnover', 'market_volatility',
        'market_liquidity_change', 'real_rate_proxy',
    ])

    # ======================== ETF 特征 ========================
    etf_bull_features: List[str] = field(default_factory=lambda: [
        'momentum_5d', 'momentum_10d', 'momentum_20d',       # 同股票
        'rsi_6', 'rsi_14',                                     # 同股票
        'fund_flow_5d', 'fund_flow_20d',                       # ← 替换资金流
        'nav_growth_rate', 'excess_return_20d', 'alpha_60d',   # ← 替换财务成长
        'price_vs_ma20', 'volume_ratio',                       # 同股票
        'share_growth_20d', 'premium_positive_freq',           # ← 替换筹码
        'clv_20d_avg', 'upside_volume_ratio_20d',              # 同股票
    ])

    etf_bear_features: List[str] = field(default_factory=lambda: [
        'index_pe_rank', 'index_pb_rank',                      # ← 替换估值排名
        'bias_20d', 'bias_60d',                                 # 同股票
        'tracking_error_20d', 'premium_abs',                   # ← 替换套牢/阻力
        'cost_pressure', 'avg_cost_deviation',                  # 同股票(量价加权)
        'share_outflow_rate', 'nav_max_drawdown_60d',          # ← 替换财务风险
    ])

    etf_friction_features: List[str] = field(default_factory=lambda: [
        'turnover_rate', 'turnover_20d_avg',                    # 同股票
        'volatility_20d', 'volatility_60d',                     # 同股票
        'amplitude', 'amplitude_20d_avg',                       # 同股票
        'weighted_avg_cost', 'share_change_volatility',        # ← 替换筹码摩擦
        'vol_price_divergence',                                 # 同股票
    ])

    etf_macro_features: List[str] = field(default_factory=lambda: [
        'cpi_yoy', 'ppi_yoy', 'pmi', 'lpr_1y',                # 同股票
        'csi500_ret_20d', 'market_turnover', 'market_volatility',  # 同股票
        'market_liquidity_change', 'real_rate_proxy',           # 同股票
    ])

    def get_feature_groups(self, mode: str = "stock") -> Dict[str, List[str]]:
        """
        根据模式返回特征分组。
        模型代码中直接使用此方法获取特征列表，无需关心是股票还是ETF。
        """
        if mode == "etf":
            return {
                'bull': list(self.etf_bull_features),
                'bear': list(self.etf_bear_features),
                'friction': list(self.etf_friction_features),
                'macro': list(self.etf_macro_features),
            }
        else:
            return {
                'bull': list(self.bull_features),
                'bear': list(self.bear_features),
                'friction': list(self.friction_features),
                'macro': list(self.macro_features),
            }

    def get_all_features(self, mode: str = "stock") -> List[str]:
        """返回当前模式的全部特征名"""
        fg = self.get_feature_groups(mode)
        return fg['bull'] + fg['bear'] + fg['friction'] + fg['macro']

    def get_feature_dims(self, mode: str = "stock") -> Dict[str, int]:
        """返回各组特征维度（模型初始化用）"""
        fg = self.get_feature_groups(mode)
        return {k: len(v) for k, v in fg.items()}

    def print_mapping(self):
        """打印股票→ETF特征映射表"""
        stock_fg = self.get_feature_groups("stock")
        etf_fg = self.get_feature_groups("etf")

        print("\n" + "="*80)
        print("股票 → ETF 特征映射")
        print("="*80)
        for group in ['bull', 'bear', 'friction', 'macro']:
            print(f"\n[{group.upper()}] ({len(stock_fg[group])} 个)")
            print("-"*70)
            for s, e in zip(stock_fg[group], etf_fg[group]):
                marker = "  " if s == e else "→ "
                print(f"  {marker}{s:35s} {'==' if s==e else '→':>2s} {e}")
        print("="*80)


# ============================================================================
#  模型配置（与原版完全一致，无需修改）
# ============================================================================
@dataclass
class ModelConfig:
    """模型配置 — 架构对 stock/etf 通用"""
    macro_mode: str = "embedding"
    macro_seq_len: int = 12
    macro_hidden_dim: int = 64
    macro_embed_dim: int = 32
    macro_num_layers: int = 2

    model_version: str = "v4"
    bull_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    bear_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    bear_rigidity_beta: float = 5.0
    volatility_penalty: float = 0.1
    target_volatility: float = 0.0

    macro_coef_min: float = 0.3
    macro_coef_max: float = 3.0

    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    bear_enhance_scale: float = 2.0
    min_energy_output: float = 0.5
    max_energy_output: float = 10.0
    target_bear_ratio: float = 0.3
    min_bear_ratio: float = 0.1
    friction_weight: float = 0.1

    encoder_hidden: int = 32
    encoder_output: int = 16
    energy_hidden: List[int] = field(default_factory=lambda: [48, 24])
    macro_scale_min: float = 0.5
    macro_scale_max: float = 2.0
    v4_l2_weight: float = 0.01

    use_v42: bool = True
    heat_cap: float = 1.0
    heat_penalty: float = 1.0
    direction_reg_weight: float = 0.01
    aggregation_mode: str = "potential_game"
    aggregation_hidden_dim: int = 12
    energy_target_mean: float = 0.5
    energy_target_std: float = 0.15
    energy_bound_low: float = 0.1
    energy_bound_high: float = 0.9
    energy_bound_mode: str = "fixed"
    energy_target_std_min: float = 0.08
    energy_target_std_max: float = 0.25
    energy_bound_adaptive_scale: float = 0.5

    bear_weight: float = 1.5
    interaction_lambda: float = 0.1
    pure_friction_lambda: float = 0.1

    dropout: float = 0.3
    energy_margin: float = 0.5

    guardian_type: str = "lightgbm"
    guardian_threshold: float = 0.50
    guardian_target_recall: float = 0.6

    enable_macro_timing: bool = True
    timing_energy_threshold: float = 0.5
    timing_min_position: float = 0.5
    timing_max_position: float = 1.0
    weight_score_ratio: float = 0.7

    # 消融
    enable_macro_modulation: bool = True
    fixed_alpha: Optional[float] = None
    fixed_beta: Optional[float] = None
    fixed_gamma: Optional[float] = None
    enable_bull: bool = True
    enable_bear: bool = True
    enable_friction: bool = True
    input_noise_std: float = 0.03


# ============================================================================
#  训练配置
# ============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True  # 自动混合精度训练，2-3倍GPU加速

    stage1_epochs: int = 10
    stage2_epochs: int = 30
    stage3_epochs: int = 20
    stage4_epochs: int = 0

    ic_threshold: float = 0.02
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    patience: int = 3
    min_delta: float = 1e-4
    gradient_clip: float = 1.0

    loss_version: str = "v4"

    ic_loss_weight: float = 1.0
    pointwise_loss_weight: float = 0.5
    topk_loss_weight: float = 0.3
    ndcg_loss_weight: float = 0.2
    reg_loss_weight: float = 0.01

    bear_constraint_weight: float = 2.0
    macro_balance_weight: float = 0.5

    rank_loss_weight: float = 0.3
    rank_temperature: float = 0.5
    v4_l2_weight: float = 0.01

    tiered_ic_weight: float = 0.0
    physics_weight: float = 0.1
    direction_reg_weight: float = 0.01
    n_tiers: int = 5
    vol_neutral_weight: float = 0.2

    ranking_loss_weight: float = 0.3
    listnet_loss_weight: float = 0.5

    pairs_per_day: int = 100
    head_sampling_ratio: float = 0.5
    min_return_diff: float = 0.01

    enable_guardian: bool = True
    rank_ic_weight: float = 0.5
    volatility_weight: float = 0.2
    crash_aux_weight: float = 0.0


# ============================================================================
#  回测配置
# ============================================================================
@dataclass
class BacktestConfig:
    """回测配置 — 股票 / ETF 双模式"""
    top_k: int = 50
    rebalance_freq: str = "weekly"

    # ======================== 股票交易成本 ========================
    commission_rate: float = 0.001
    slippage: float = 0.001
    stamp_tax: float = 0.001          # 卖出印花税

    # ======================== ETF 交易成本（更低！）========================
    etf_commission_rate: float = 0.0003   # ETF 佣金更低
    etf_slippage: float = 0.0005          # ETF 流动性好，滑点小
    etf_stamp_tax: float = 0.0            # ETF 无印花税！

    t_plus_1: bool = True
    single_stock_limit: float = 0.10
    industry_limit: float = 0.30

    # ETF 集中度限制（ETF数量少，可适当放宽）
    etf_single_limit: float = 0.15        # 单只ETF最大15%
    etf_sector_limit: float = 0.40        # 同板块ETF最大40%

    enable_stop_loss: bool = True
    stop_loss_threshold: float = -0.10
    etf_stop_loss_threshold: float = -0.08  # ETF 止损更紧（波动低）

    initial_capital: float = 1000000.0
    enable_macro_timing: bool = True

    def get_costs(self, mode: str = "stock") -> Dict[str, float]:
        """根据模式返回交易成本"""
        if mode == "etf":
            return {
                'commission': self.etf_commission_rate,
                'slippage': self.etf_slippage,
                'stamp_tax': self.etf_stamp_tax,
                'single_limit': self.etf_single_limit,
                'sector_limit': self.etf_sector_limit,
                'stop_loss': self.etf_stop_loss_threshold,
            }
        return {
            'commission': self.commission_rate,
            'slippage': self.slippage,
            'stamp_tax': self.stamp_tax,
            'single_limit': self.single_stock_limit,
            'sector_limit': self.industry_limit,
            'stop_loss': self.stop_loss_threshold,
        }


# ============================================================================
#  总配置
# ============================================================================
@dataclass
class Config:
    """
    总配置类 — 支持 stock / etf 模式切换

    用法:
      config = Config()                 # 默认股票模式
      config = Config.etf_mode()        # ETF模式
      config.switch_to_etf()            # 动态切换
      fg = config.feature.get_feature_groups(config.data.mode)
    """
    experiment_name: str = "macro_game_ebm_v2"

    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    seed: int = 42
    device: str = "cuda"

    # ======================== 模式切换 ========================
    @classmethod
    def market_mode(cls, market_profile: str, asset_type: str = "stock") -> 'Config':
        """按市场预设创建配置"""
        config = cls()
        config.apply_market_profile(market_profile, asset_type)
        return config

    @classmethod
    def etf_mode(cls) -> 'Config':
        """创建 ETF 模式的配置（默认 CSI500 ETF）"""
        return cls.market_mode("csi500", "etf")

    def apply_market_profile(self, market_profile: str, asset_type: str = "stock"):
        """应用市场预设，不改变训练主架构，仅调整路径和超参数"""
        apply_market_profile_to_config(self, market_profile, asset_type)

    def switch_to_etf(self):
        """切换到 ETF 模式"""
        self.apply_market_profile(self.data.market_profile, "etf")

    def switch_to_stock(self):
        """切换回股票模式"""
        self.apply_market_profile(self.data.market_profile, "stock")

    # ======================== 便捷方法 ========================
    @property
    def mode(self) -> str:
        return self.data.mode

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """获取当前模式的特征分组（模型初始化直接用）"""
        return self.feature.get_feature_groups(self.data.mode)

    def get_feature_dims(self) -> Dict[str, int]:
        """获取当前模式的特征维度（模型初始化直接用）"""
        return self.feature.get_feature_dims(self.data.mode)

    def print_config_summary(self):
        """打印配置摘要"""
        fg = self.get_feature_groups()
        print(f"\n{'='*60}")
        print(f"  模式: {self.mode.upper()}")
        print(f"  市场: {self.data.market_label} ({self.data.market_profile})")
        print(f"  实验: {self.experiment_name}")
        print(f"  设备: {self.device}")
        print(f"  数据: {self.data.active_panel_path}")
        print(f"  特征: Bull={len(fg['bull'])} Bear={len(fg['bear'])} "
              f"Friction={len(fg['friction'])} Macro={len(fg['macro'])} "
              f"Total={sum(len(v) for v in fg.values())}")
        print(f"  训练: {self.data.train_start} ~ {self.data.train_end}")
        print(f"  验证: {self.data.valid_start} ~ {self.data.valid_end}")
        print(f"  测试: {self.data.test_start} ~ {self.data.test_end}")
        print(f"  TopK: {self.backtest.top_k}")
        costs = self.backtest.get_costs(self.mode)
        print(f"  佣金: {costs['commission']:.4f}  滑点: {costs['slippage']:.4f}  "
              f"印花税: {costs['stamp_tax']:.4f}")
        print(f"  Crash: ret<{self.data.active_crash_threshold:.0%} "
              f"OR dd>{self.data.active_drawdown_threshold:.0%}")
        print(f"{'='*60}")

    # ======================== 序列化 ========================
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
        print(f"✓ 配置保存至: {path}")

    @classmethod
    def load(cls, path: Path) -> 'Config':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls._from_dict(config_dict)

    def _to_dict(self) -> Dict[str, Any]:
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        return convert(self)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> 'Config':
        return cls(
            experiment_name=d.get('experiment_name', 'macro_game_ebm_v2'),
            data=DataConfig(**{k: Path(v) if ('path' in k.lower() or 'dir' in k.lower() or 'root' in k.lower()) and isinstance(v, str) else v
                              for k, v in d.get('data', {}).items()}),
            feature=FeatureConfig(**d.get('feature', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            backtest=BacktestConfig(**d.get('backtest', {})),
            seed=d.get('seed', 42),
            device=d.get('device', 'cuda'),
        )


# ============================================================================
#  默认配置实例
# ============================================================================
DEFAULT_CONFIG = Config()


# ============================================================================
#  快速验证
# ============================================================================
if __name__ == "__main__":
    for profile in available_market_profiles():
        print(f">>> {profile} / stock")
        Config.market_mode(profile, "stock").print_config_summary()
