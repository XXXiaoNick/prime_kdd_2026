"""
================================================================================
主入口文件 - 宏观感知物理博弈能量模型 (修复版 v2)
================================================================================

使用方法：
    # 完整训练流程
    python main.py --mode train
    
    # 使用演示数据测试
    python main.py --mode test
    
    # 回测已训练模型
    python main.py --mode backtest --checkpoint checkpoints/best_model
    
日志文件保存在 logs/ 目录，命名格式: {mode}_{params}_{datetime}.log
================================================================================
"""

import argparse
import shutil
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Config
from data_loader import AdaptiveFeatureConfig, DatasetSplitter, create_data_pipeline
from dataset import (
    StockDataset,
    create_training_loaders,
    prepare_feature_arrays,
    resolve_model_input_dims,
    slice_feature_tensors,
)
from experiment_catalog import available_experiments, list_experiment_lines
from market_profiles import available_market_profiles
from models import GameEnergyModel, RiskGuardian
from baselines import (
    BaselineRunConfig,
    BaselineRunner,
    available_baselines,
    baseline_registry_lines,
)
from trainer import CurriculumTrainer, Stage4Integrator
from backtest import Backtester, save_backtest_result
from utils import set_seed, get_device, LogManager

try:
    from visualization import ModelVisualizer
    HAS_MODEL_VISUALIZER = True
except Exception:
    HAS_MODEL_VISUALIZER = False


def parse_args():
    parser = argparse.ArgumentParser(description='宏观感知物理博弈能量模型 v2')
    
    # 基础参数
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['train', 'backtest', 'test', 'main_exp', 'ablation', 'robustness', 'baseline'],
                       help='运行模式: train/backtest/test/main_exp/ablation/robustness/baseline')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--no_log', action='store_true', help='禁用日志文件保存')
    
    # 数据路径参数
    parser.add_argument('--data_root', type=str, default=None,
                       help='数据根目录 (默认: <repo>/data/<market>，或由环境变量 PRIME_DATA_ROOT 指定；数据缺失时自动生成 mock 数据)')
    parser.add_argument('--data_path', type=str, default=None, 
                       help='面板数据文件路径 (如果指定，覆盖data_root下的panel路径)')
    parser.add_argument('--panel_file', type=str, default=None,
                       help='panel目录下的数据文件名（默认自动检测 panel_data_complete.* / panel_data.*）')
    parser.add_argument('--market_profile', type=str, default=None,
                       choices=available_market_profiles(),
                       help='市场预设: csi500/sp500/nikkei225/stoxx600')
    parser.add_argument('--asset_type', type=str, default=None,
                       choices=['stock', 'etf'],
                       help='资产类型: stock 或 etf')
    parser.add_argument('--main_exp_type', type=str, default='all',
                       choices=['all'] + available_experiments('main'),
                       help='主实验类型 (仅 main_exp 模式)')
    parser.add_argument('--ablation_type', type=str, default='all',
                       choices=['all'] + available_experiments('ablation'),
                       help='消融实验类型 (仅 ablation 模式)')
    parser.add_argument('--baseline_name', type=str, default='all',
                       choices=['all'] + available_baselines(),
                       help='baseline 模型名: all/' + '/'.join(available_baselines()))
    parser.add_argument('--baseline_epochs', type=int, default=None,
                       help='覆盖 baseline 默认训练轮数')
    parser.add_argument('--baseline_seq_len', type=int, default=None,
                       help='覆盖 baseline 默认序列长度')
    parser.add_argument('--baseline_backbone', type=str, default=None,
                       help='覆盖 GPT4TS/TIME-LLM 的 Hugging Face backbone')
    parser.add_argument('--list_baselines', action='store_true',
                       help='打印 baseline 列表并退出')
    parser.add_argument('--quick_baseline', action='store_true',
                       help='快速 baseline 烟测模式（减少 epochs）')
    
    # 数据加载加速
    parser.add_argument('--fast', action='store_true', 
                       help='使用GPU/Numba加速数据预处理')
    parser.add_argument('--no_cache', action='store_true',
                       help='禁用数据缓存')
    
    # 训练阶段控制
    parser.add_argument('--skip_stage1', action='store_true', help='跳过Stage1 Sanity Check')
    parser.add_argument('--stage1_epochs', type=int, default=None, help='Stage1训练轮数 (默认10)')
    parser.add_argument('--stage2_epochs', type=int, default=None, help='Stage2训练轮数 (默认50)')
    parser.add_argument('--stage3_epochs', type=int, default=None, help='Stage3训练轮数 (默认20)')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小 (默认512)')
    parser.add_argument('--lr', type=float, default=None, help='学习率 (默认1e-3)')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值 (默认10)')
    
    # 回测参数
    parser.add_argument('--top_k', type=int, default=None, help='持仓股票数 (默认50)')
    
    # 消融实验参数
    parser.add_argument('--n_repeats', type=int, default=3, help='消融实验重复次数')
    
    # 鲁棒性实验参数
    parser.add_argument('--robustness_type', type=str, default='all',
                       choices=['all'] + available_experiments('robustness'),
                       help='鲁棒性实验类型 (仅robustness模式)')
    parser.add_argument('--list_main_experiments', action='store_true',
                       help='打印主实验注册表并退出')
    parser.add_argument('--list_ablations', action='store_true',
                       help='打印消融实验注册表并退出')
    parser.add_argument('--list_robustness', action='store_true',
                       help='打印鲁棒性实验注册表并退出')
    
    return parser.parse_args()


def apply_args_to_config(config: Config, args) -> Config:
    """将命令行参数应用到配置"""
    if args.market_profile is not None or args.asset_type is not None:
        market_profile = args.market_profile or config.data.market_profile
        asset_type = args.asset_type or config.data.mode
        config.apply_market_profile(market_profile, asset_type)

    # 数据路径
    if args.data_root is not None:
        data_root = Path(args.data_root)
        if config.data.mode == 'etf':
            config.data.etf_data_root = data_root
            config.data.etf_panel_path = data_root / 'panel' / (args.panel_file or 'etf_panel_complete.csv')
            config.data.etf_daily_dir = data_root / 'etf'
            config.data.etf_nav_dir = data_root / 'nav'
            config.data.etf_share_dir = data_root / 'share'
            config.data.etf_benchmark_dir = data_root / 'benchmark'
            config.data.etf_macro_dir = data_root / 'macro'
            config.data.etf_index_dir = data_root / 'index'
            config.data.etf_meta_dir = data_root / 'meta'
        else:
            config.data.data_root = data_root
            config.data.panel_path = data_root / 'panel' / (args.panel_file or 'panel_data_complete.csv')
            # 更新各子目录路径
            config.data.capital_flow_dir = data_root / 'capital_flow'
            config.data.chip_dir = data_root / 'chip'
            config.data.financial_dir = data_root / 'financial'
            config.data.index_dir = data_root / 'index'
            config.data.industry_dir = data_root / 'industry'
            config.data.macro_dir = data_root / 'macro'
            config.data.market_dir = data_root / 'market'
            config.data.margin_dir = data_root / 'margin'
            config.data.meta_dir = data_root / 'meta'
            config.data.northbound_dir = data_root / 'northbound'
            config.data.stock_dir = data_root / 'stock'
            config.data.valuation_dir = data_root / 'valuation'
    
    if args.data_path is not None:
        if config.data.mode == 'etf':
            config.data.etf_panel_path = Path(args.data_path)
        else:
            config.data.panel_path = Path(args.data_path)
    
    # 训练轮数
    if args.stage1_epochs is not None:
        config.training.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        config.training.stage2_epochs = args.stage2_epochs
    if args.stage3_epochs is not None:
        config.training.stage3_epochs = args.stage3_epochs
    
    # 训练超参数
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.patience is not None:
        config.training.patience = args.patience
    
    # 回测参数
    if args.top_k is not None:
        config.backtest.top_k = args.top_k
    
    return config


def print_experiment_registry(category: str, title: str):
    print(title)
    for line in list_experiment_lines(category):
        print(line)


def print_baseline_registry():
    print("可用 baseline 预设:")
    for line in baseline_registry_lines():
        print(line)


def resolve_runtime_device(config: Config) -> torch.device:
    """根据当前配置解析并同步运行设备。"""
    device = get_device(prefer_gpu=(config.device == 'cuda'))
    config.device = str(device)
    return device


def normalize_runtime_feature_config(config: Config) -> Config:
    """
    规范化运行时特征配置。

    训练保存的 checkpoint config 里，`feature.macro_features` 可能已经被
    动态替换成 `macro_embed_*` 模型输入列。这里在重新进入数据管道前将其
    恢复为 stock/etf 模板里的原始宏观列，避免重复生成 embedding。
    """
    macro_cols = list(getattr(config.feature, "macro_features", []))
    if macro_cols and all(str(col).startswith("macro_embed_") for col in macro_cols):
        from config import FeatureConfig

        base_feature_cfg = FeatureConfig()
        config.feature.macro_features = base_feature_cfg.get_feature_groups(config.mode)["macro"]
    return config


def _get_viz_source_dir() -> Path | None:
    candidates = [
        PROJECT_ROOT / 'outputs' / 'viz_data',
        Path.cwd() / 'outputs' / 'viz_data',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def export_training_visualizations(run_dir: Path, config: Config):
    """将训练阶段产生的可视化原始数据和图表归档到当前实验目录。"""
    source_dir = _get_viz_source_dir()
    if source_dir is None:
        return

    raw_dir = Path(run_dir) / 'viz_data'
    raw_dir.mkdir(parents=True, exist_ok=True)

    history_path = source_dir / 'training_history.json'
    phase_space_path = source_dir / 'game_phase_space.csv'
    guardian_path = source_dir / 'guardian_distribution.json'

    copied = []
    for src in (history_path, phase_space_path, guardian_path):
        if src.exists():
            dst = raw_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    if not copied or not HAS_MODEL_VISUALIZER:
        return

    if history_path.exists() and phase_space_path.exists() and guardian_path.exists():
        figure_dir = Path(run_dir) / 'figures'
        visualizer = ModelVisualizer(output_dir=str(figure_dir))
        visualizer.plot_all(
            history_path=str(raw_dir / 'training_history.json'),
            phase_space_path=str(raw_dir / 'game_phase_space.csv'),
            guardian_path=str(raw_dir / 'guardian_distribution.json'),
            guardian_threshold=config.model.guardian_threshold,
            show=False,
        )


def create_demo_data(config: Config):
    """创建演示数据"""
    print("\n>>> 创建演示数据...")
    
    np.random.seed(42)
    demo_start = pd.to_datetime(config.data.train_start) - pd.Timedelta(days=120)
    demo_end = pd.to_datetime(config.data.test_end)
    dates = pd.date_range(demo_start, demo_end, freq='B')
    codes = [f'{i:06d}' for i in range(1, 101)]
    
    data = []
    for date in dates:
        for code in codes:
            base = {
                'trade_date': date,
                'ts_code': code,
                'close': np.random.randn() * 10 + 50,
                'volume': np.random.rand() * 1e8,
                # Bull特征
                'momentum_5d': np.random.randn(),
                'momentum_10d': np.random.randn(),
                'momentum_20d': np.random.randn(),
                'rsi_14': np.random.rand() * 100,
                'north_net_flow': np.random.randn() * 1e8,
                'main_net_inflow': np.random.randn() * 1e7,
                'price_vs_ma20': np.random.randn() * 0.1,
                'volume_ratio': np.random.rand() * 2 + 0.5,
                # Bear特征
                'pe_rank': np.random.rand(),
                'pb_rank': np.random.rand(),
                'bias_20d': np.random.randn() * 5,
                'bias_60d': np.random.randn() * 8,
                'trapped_ratio': np.random.rand() * 30,
                'margin_sell_pressure': np.random.rand(),
                # Friction特征
                'turnover_rate': np.random.rand() * 10,
                'turnover_20d_avg': np.random.rand() * 8,
                'volatility_20d': np.random.rand() * 0.3,
                'volatility_60d': np.random.rand() * 0.25,
                'amplitude': np.random.rand() * 5,
                # Macro特征
                'cpi_yoy': np.random.randn() * 0.5 + 2,
                'ppi_yoy': np.random.randn() * 2,
                'pmi': np.random.randn() * 2 + 50,
                'm2_yoy': np.random.randn() * 1 + 8,
            }
            
            # 收益标签（与特征弱相关）
            signal = (base['momentum_20d'] * 0.3 - base['pe_rank'] * 0.2 + 
                     base['north_net_flow'] / 1e8 * 0.1 - base['volatility_20d'] * 0.1)
            base['fwd_return'] = signal * 0.01 + np.random.randn() * 0.03
            base['crash_label'] = int(base['fwd_return'] < -0.10 or np.random.rand() < 0.05)
            base['csi500_ret'] = np.random.randn() * 0.015
            
            data.append(base)
    
    df = pd.DataFrame(data)
    
    # 添加预计算的宏观Embedding
    embed_dim = 12
    for i in range(embed_dim):
        df[f'macro_embed_{i}'] = np.random.randn(len(df)) * 0.1
    
    # 划分数据集
    splitter = DatasetSplitter(config)
    splits = splitter.split(df)
    
    # 特征列
    feature_cols = {
        'bull': ['momentum_5d', 'momentum_10d', 'momentum_20d', 'rsi_14', 
                'north_net_flow', 'main_net_inflow', 'price_vs_ma20', 'volume_ratio'],
        'bear': ['pe_rank', 'pb_rank', 'bias_20d', 'bias_60d', 'trapped_ratio', 'margin_sell_pressure'],
        'friction': ['turnover_rate', 'turnover_20d_avg', 'volatility_20d', 'volatility_60d', 'amplitude'],
        'macro': [f'macro_embed_{i}' for i in range(embed_dim)]
    }
    
    print(f"演示数据: {len(df)} 行, {len(dates)} 天, {len(codes)} 股票")
    
    return splits, df, feature_cols


def run_data_pipeline(config: Config, fast: bool = False, no_cache: bool = False):
    """
    数据加载管道（供robustness.py和ablation.py调用）
    
    Returns:
        splits: DataSplits对象 (train, valid, test)
        full_df: 完整数据DataFrame
        feature_cols: 特征列字典 {'bull': [...], 'bear': [...], 'friction': [...], 'macro': [...]}
    """
    # 加载数据
    try:
        if fast:
            print(">>> 使用统一快速数据流水线 (GPU/Numba加速)")
        splits, full_df, macro_embeddings = create_data_pipeline(
            config,
            fast=fast,
            use_gpu=True,
            use_cache=not no_cache,
            allow_source_fallback=fast,
        )
        
        # 【关键】使用数据验证器自动检测和适配特征
        adapter = AdaptiveFeatureConfig()
        data_config = adapter.create_config_from_data(full_df, config)
        
        feature_cols = data_config['feature_groups']
        feature_dims = data_config['feature_dims']
        
        # 添加macro embedding列到feature_cols
        macro_embed_cols = [c for c in full_df.columns if c.startswith('macro_embed_')]
        if macro_embed_cols:
            feature_cols['macro'] = macro_embed_cols
            feature_dims['macro'] = len(macro_embed_cols)
        
        # 更新splits中的数据为清洗后的数据
        cleaner = data_config['cleaner']
        all_features = (feature_cols['bull'] + feature_cols['bear'] + 
                       feature_cols['friction'] + feature_cols['macro'])
        
        # 清洗各数据集
        splits.train = cleaner.transform(splits.train, all_features)
        splits.valid = cleaner.transform(splits.valid, all_features)
        splits.test = cleaner.transform(splits.test, all_features)
        
    except FileNotFoundError:
        print("数据文件不存在，使用演示数据...")
        splits, full_df, feature_cols = create_demo_data(config)
    
    return splits, full_df, feature_cols


def run_training(config: Config, args):
    """运行完整训练流程"""
    print("\n" + "=" * 70)
    print("    宏观感知物理博弈能量模型 v2 - 训练模式")
    print("=" * 70)
    
    set_seed(args.seed)
    resolve_runtime_device(config)
    
    # 加载数据（使用统一的数据管道）
    splits, full_df, feature_cols = run_data_pipeline(
        config, 
        fast=args.fast, 
        no_cache=args.no_cache
    )
    
    # 更新config中的特征列表（用于后续模块）
    config.feature.bull_features = feature_cols['bull']
    config.feature.bear_features = feature_cols['bear']
    config.feature.friction_features = feature_cols['friction']
    config.feature.macro_features = feature_cols['macro']
    
    # 创建数据加载器
    print("\n>>> 创建数据加载器...")
    dataset_bundle, train_loader_simple, valid_loader_simple, pairwise_loader = create_training_loaders(
        splits.train, splits.valid, config
    )
    
    actual_dims = dataset_bundle.get_feature_dims()
    print(f"实际特征维度: {actual_dims}")
    
    all_feature_cols = (feature_cols['bull'] + feature_cols['bear'] + 
                       feature_cols['friction'] + feature_cols['macro'])
    
    # 【调试】验证数据完整性
    print("\n>>> 数据验证:")
    print(f"    train_df shape: {splits.train.shape}")
    print(f"    train_df 包含macro_embed列: {any(c.startswith('macro_embed') for c in splits.train.columns)}")
    macro_cols_in_train = [c for c in splits.train.columns if c.startswith('macro_embed')]
    print(f"    macro_embed列数量: {len(macro_cols_in_train)}")
    
    # 检查NaN
    for group_name, cols in feature_cols.items():
        exist_cols = [c for c in cols if c in splits.train.columns]
        if exist_cols:
            nan_count = splits.train[exist_cols].isna().sum().sum()
            print(f"    {group_name}特征: {len(exist_cols)}列存在, NaN数量={nan_count}")
        else:
            print(f"    {group_name}特征: 0列存在! (期望{len(cols)}列)")
    
    # 课程学习训练
    trainer = CurriculumTrainer(config)
    
    # 【改进】使用IC Loss进行训练
    use_ic_loss = getattr(config.training, 'use_ic_loss', True)
    
    integrator = trainer.train(
        train_loader=train_loader_simple,
        valid_loader=valid_loader_simple,
        pairwise_loader=pairwise_loader,
        train_df=splits.train,
        valid_df=splits.valid,
        feature_dims=actual_dims,  # 使用实际维度
        feature_cols=all_feature_cols,
        feature_groups=feature_cols,
        skip_stage1=args.skip_stage1,
        use_ic_loss=use_ic_loss  # 【新增】传入IC Loss参数
    )
    
    # 保存模型
    save_dir = Path(args.output_dir) / 'checkpoints' / config.experiment_name
    trainer.save(save_dir)
    
    # 回测
    print("\n>>> 测试集回测")
    run_backtest(config, splits.test, integrator, feature_cols, str(save_dir))
    export_training_visualizations(save_dir, config)
    
    print("\n✓ 训练完成！")


def run_backtest(config: Config, test_df: pd.DataFrame, integrator: Stage4Integrator, feature_cols: dict, output_dir: str):
    """
    运行回测
    
    【关键修复】数据流：
    1. compute_ebm_scores计算所有股票的分数和能量组件
    2. select_stocks仅用于获取仓位比例和生成可视化数据
    3. backtest使用完整的分数数据 + 仓位比例
    """
    # 【修复】第1步：计算所有股票的EBM分数和能量组件
    print("\n>>> 计算所有股票的EBM分数...")
    test_df = compute_ebm_scores(test_df, integrator, feature_cols)
    
    # 【调试】检查能量组件
    print(f"\n>>> 能量组件检查:")
    for col in ['E_bull', 'E_bear', 'E_friction', 'ebm_score']:
        if col in test_df.columns:
            vals = test_df[col].dropna()
            print(f"    {col}: count={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}")
        else:
            print(f"    {col}: 不存在")
    
    # 【修复】第2步：调用select_stocks获取仓位比例和可视化数据
    # 这里不使用select_stocks的选股结果，只用于：
    # - 计算每日仓位比例（position_ratio）
    # - 生成博弈相平面可视化数据
    print("\n>>> 计算仓位比例...")
    result_df = integrator.select_stocks(
        test_df,
        bull_cols=feature_cols['bull'],
        bear_cols=feature_cols['bear'],
        friction_cols=feature_cols['friction'],
        macro_cols=feature_cols['macro'],
        top_k=config.backtest.top_k,
        record_phase_space=True  # 记录博弈相平面数据
    )
    
    # 提取仓位比例
    position_ratios = None
    if not result_df.empty and 'position_ratio' in result_df.columns:
        # 按日期提取仓位比例
        position_ratios = result_df.groupby('trade_date')['position_ratio'].first().to_dict()
        avg_ratio = np.mean(list(position_ratios.values()))
        min_ratio = np.min(list(position_ratios.values()))
        max_ratio = np.max(list(position_ratios.values()))
        print(f"    仓位比例: 平均={avg_ratio:.1%}, 最低={min_ratio:.1%}, 最高={max_ratio:.1%}")
    
    # 【修复】第3步：回测，使用完整的分数数据
    backtester = Backtester(config)
    
    energy_components = {
        'E_bull': 'E_bull',
        'E_bear': 'E_bear', 
        'E_friction': 'E_friction'
    }
    
    # 使用日度收益而不是5日收益进行回测
    backtest_return_col = 'daily_return' if 'daily_return' in test_df.columns else 'fwd_return'
    if backtest_return_col == 'fwd_return':
        print("  警告: daily_return列不存在，使用fwd_return（收益可能被高估）")
    
    backtest_result = backtester.run(
        test_df,
        signal_col='ebm_score',
        return_col=backtest_return_col,
        date_col='trade_date',
        code_col='ts_code',
        top_k=config.backtest.top_k,
        energy_components=energy_components,
        position_ratios=position_ratios  # 【新增】传入仓位比例
    )

    backtest_dir = Path(output_dir)
    if backtest_dir.name != 'backtest':
        backtest_dir = backtest_dir / 'backtest'
    save_backtest_result(backtest_result, backtest_dir)
    
    return backtest_result


def compute_ebm_scores(df: pd.DataFrame, integrator: Stage4Integrator, feature_cols: dict) -> pd.DataFrame:
    """
    批量计算EBM得分和能量组件
    
    【修复】计算所有股票的分数，不受position_ratio限制
    """
    df = df.copy()
    
    integrator.game_ebm.eval()
    device = integrator.device
    expected_dims = resolve_model_input_dims(integrator.game_ebm, feature_cols)
    feature_arrays, _ = prepare_feature_arrays(df, feature_cols, expected_dims)
    
    scores = []
    E_bull_list = []
    E_bear_list = []
    E_friction_list = []
    batch_size = 1000
    
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
            X_bull, X_bear, X_friction, X_macro = slice_feature_tensors(
                feature_arrays, i, end, device
            )
            
            # 【修复】使用return_components获取能量组件
            energy, components = integrator.game_ebm(
                X_bull, X_bear, X_friction, X_macro,
                return_components=True
            )
            
            scores.append(-energy.squeeze().cpu().numpy())
            
            # 提取能量组件
            if 'E_bull' in components:
                E_bull_list.append(components['E_bull'].squeeze().cpu().numpy())
                E_bear_list.append(components['E_bear'].squeeze().cpu().numpy())
                E_friction_list.append(components.get('E_friction', 
                    components.get('E_pure_friction', 
                    torch.zeros_like(energy))).squeeze().cpu().numpy())
            else:
                # V4模型使用编码向量的L2范数
                E_bull_list.append(torch.norm(components['h_bull'], dim=-1).cpu().numpy())
                E_bear_list.append(torch.norm(components['h_bear'], dim=-1).cpu().numpy())
                E_friction_list.append(torch.norm(components['h_friction'], dim=-1).cpu().numpy())
    
    df['ebm_score'] = np.concatenate(scores)
    df['E_bull'] = np.concatenate(E_bull_list)
    df['E_bear'] = np.concatenate(E_bear_list)
    df['E_friction'] = np.concatenate(E_friction_list)
    
    return df


def run_test(config: Config, args):
    """运行模块测试"""
    print("\n" + "=" * 70)
    print("    模块测试模式")
    print("=" * 70)
    
    set_seed(args.seed)
    device = resolve_runtime_device(config)
    
    # 创建演示数据
    splits, full_df, feature_cols = create_demo_data(config)
    
    # 测试数据集
    print("\n>>> 测试 StockDataset...")
    dataset = StockDataset(
        splits.train,
        bull_features=feature_cols['bull'],
        bear_features=feature_cols['bear'],
        friction_features=feature_cols['friction'],
        macro_features=feature_cols['macro'],
        use_zscore=False
    )
    print(f"数据集大小: {len(dataset)}")
    print(f"特征维度: {dataset.get_feature_dims()}")
    
    # 测试模型
    print("\n>>> 测试 GameEnergyModel...")
    dims = dataset.get_feature_dims()
    
    model = GameEnergyModel(
        bull_dim=dims['bull'],
        bear_dim=dims['bear'],
        friction_dim=dims['friction'],
        macro_dim=dims['macro'],
        interaction_lambda=0.1,
        pure_friction_lambda=0.05
    ).to(device)
    
    batch_size = 32
    x_bull = torch.randn(batch_size, dims['bull']).to(device)
    x_bear = torch.randn(batch_size, dims['bear']).to(device)
    x_friction = torch.randn(batch_size, dims['friction']).to(device)
    x_macro = torch.randn(batch_size, dims['macro']).to(device)
    
    energy, components = model(x_bull, x_bear, x_friction, x_macro, return_components=True)
    print(f"能量输出: {energy.shape}")
    print(f"能量范围: [{energy.min().item():.4f}, {energy.max().item():.4f}]")
    print(f"E_pure_friction (修复项): {components['E_pure_friction'].mean().item():.4f}")
    
    # 测试Guardian
    print("\n>>> 测试 RiskGuardian...")
    guardian = RiskGuardian(model_type='sklearn', threshold=0.3)
    
    X_train = splits.train[feature_cols['bull'] + feature_cols['bear']].fillna(0).values[:1000]
    y_train = splits.train['crash_label'].values[:1000]
    X_test = splits.valid[feature_cols['bull'] + feature_cols['bear']].fillna(0).values[:500]
    y_test = splits.valid['crash_label'].values[:500]
    
    guardian.fit(X_train, y_train)
    y_pred = guardian.predict(X_test)
    print(f"Guardian Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Guardian Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Guardian Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    
    # 测试回测
    print("\n>>> 测试 Backtester...")
    test_df = splits.test.copy()
    test_df['ebm_score'] = np.random.randn(len(test_df))
    
    backtester = Backtester(config)
    result = backtester.run(test_df, top_k=20)
    
    print("\n✓ 所有模块测试通过！")


def main():
    args = parse_args()

    if args.list_main_experiments:
        print_experiment_registry("main", "可用主实验:")
        return
    if args.list_ablations:
        print_experiment_registry("ablation", "可用消融实验:")
        return
    if args.list_robustness:
        print_experiment_registry("robustness", "可用鲁棒性实验:")
        return
    if args.list_baselines:
        print_baseline_registry()
        return
    
    # 加载配置
    if args.config:
        config = Config.load(Path(args.config))
    else:
        config = Config()
    
    # 应用命令行参数到配置
    config = apply_args_to_config(config, args)
    config = normalize_runtime_feature_config(config)
    
    if args.data_path:
        if config.data.mode == 'etf':
            config.data.etf_panel_path = Path(args.data_path)
        else:
            config.data.panel_path = Path(args.data_path)
    
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        config.device = 'cuda'
    else:
        config.device = 'cpu'

    config.print_config_summary()
    
    # 准备日志参数 - 包含训练轮数等关键信息
    extra_params = {
        'seed': args.seed,
        'gpu': args.gpu,
        'market': config.data.market_profile,
        'asset': config.data.mode,
        's1': config.training.stage1_epochs,  # Stage1轮数
        's2': config.training.stage2_epochs,  # Stage2轮数
        's3': config.training.stage3_epochs,  # Stage3轮数
        'bs': config.training.batch_size,     # batch_size
        'lr': config.training.learning_rate,  # 学习率
        'topk': config.backtest.top_k,        # 持仓数
    }
    
    # 添加可选标记
    if args.skip_stage1:
        extra_params['skip_s1'] = True
    if args.data_path:
        extra_params['data'] = Path(args.data_path).stem[:10]  # 数据文件名前10字符

    if args.mode == 'main_exp':
        extra_params = {
            'exp': args.main_exp_type,
            'market': config.data.market_profile,
            'asset': config.data.mode,
            'repeats': args.n_repeats,
            'seed': args.seed,
        }
    
    # 消融实验特殊参数
    if args.mode == 'ablation':
        extra_params = {
            'exp': args.ablation_type,
            'epochs': config.training.stage2_epochs,
            'bs': config.training.batch_size,
            'repeats': args.n_repeats,
            'seed': args.seed,
        }
    
    if args.mode == 'robustness':
        extra_params = {
            'exp': args.robustness_type,
            'epochs': config.training.stage2_epochs,
            'bs': config.training.batch_size,
            'seed': args.seed,
        }
    if args.mode == 'baseline':
        extra_params = {
            'baseline': args.baseline_name,
            'market': config.data.market_profile,
            'asset': config.data.mode,
            'seed': args.seed,
        }
        if args.baseline_epochs is not None:
            extra_params['epochs'] = args.baseline_epochs
        if args.quick_baseline:
            extra_params['quick'] = True
    
    # 使用日志管理器
    if not args.no_log:
        log_manager = LogManager(
            mode=args.mode,
            log_dir=args.log_dir,
            extra_params=extra_params,
            experiment_name=config.experiment_name
        )
        log_manager.start()
    
    try:
        if args.mode == 'train':
            run_training(config, args)
        elif args.mode == 'backtest':
            if not args.checkpoint:
                print("错误: 回测模式需要指定 --checkpoint")
                return
            run_backtest_mode(config, args)
        elif args.mode == 'test':
            run_test(config, args)
        elif args.mode == 'main_exp':
            run_main_experiments(config, args)
        elif args.mode == 'ablation':
            run_ablation(config, args)
        elif args.mode == 'robustness':
            run_robustness(config, args)
        elif args.mode == 'baseline':
            run_baselines(config, args)
    finally:
        # 确保日志被保存
        if not args.no_log:
            log_manager.stop()


def run_ablation(config: Config, args):
    """运行消融实验"""
    from experiment_runner import AblationRunner, AblationConfig

    device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    ablation_config = AblationConfig(
        ablation_type=args.ablation_type,
        output_dir=str(Path(args.output_dir) / 'ablation'),
        repeats=args.n_repeats,
        fast=args.fast,
        no_cache=args.no_cache,
    )

    runner = AblationRunner(config, ablation_config, device)
    runner.run_all(seed_base=args.seed)


def run_main_experiments(config: Config, args):
    """运行主实验。"""
    from experiment_runner import MainExperimentConfig, MainExperimentRunner

    device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    experiment_config = MainExperimentConfig(
        main_exp_type=args.main_exp_type,
        output_dir=str(Path(args.output_dir) / 'main_experiments'),
        repeats=args.n_repeats,
        fast=args.fast,
        no_cache=args.no_cache,
    )

    runner = MainExperimentRunner(config, experiment_config, device)
    runner.run(seed_base=args.seed)


def run_robustness(config: Config, args):
    """运行鲁棒性实验"""
    from experiment_runner import RobustnessRunner, RobustnessConfig
    
    device = 'cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    
    robustness_config = RobustnessConfig(
        robustness_type=args.robustness_type,
        output_dir=str(Path(args.output_dir) / 'robustness'),
        repeats=max(1, args.n_repeats),
        fast=args.fast,
        no_cache=args.no_cache,
    )
    
    runner = RobustnessRunner(config, robustness_config, device)
    runner.run_all(seed_base=args.seed)


def run_backtest_mode(config: Config, args):
    """回测模式入口"""
    print(">>> 回测模式")
    print(f"    加载模型: {args.checkpoint}")
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_config = checkpoint_dir / 'config.json'
    if checkpoint_config.exists() and args.config is None:
        print(f"    使用检查点配置: {checkpoint_config}")
        config = Config.load(checkpoint_config)
        config = apply_args_to_config(config, args)
        config = normalize_runtime_feature_config(config)
        if args.data_path:
            if config.data.mode == 'etf':
                config.data.etf_panel_path = Path(args.data_path)
            else:
                config.data.panel_path = Path(args.data_path)
        if args.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            config.device = 'cuda'
        else:
            config.device = 'cpu'

    device = resolve_runtime_device(config)

    splits, full_df, feature_cols = run_data_pipeline(
        config,
        fast=args.fast,
        no_cache=args.no_cache,
    )

    config.feature.bull_features = feature_cols['bull']
    config.feature.bear_features = feature_cols['bear']
    config.feature.friction_features = feature_cols['friction']
    config.feature.macro_features = feature_cols['macro']

    sample_dataset = StockDataset(
        splits.train.head(min(len(splits.train), 128)),
        bull_features=feature_cols['bull'],
        bear_features=feature_cols['bear'],
        friction_features=feature_cols['friction'],
        macro_features=feature_cols['macro'],
    )
    feature_dims = sample_dataset.get_feature_dims()

    from trainer import Stage2Trainer
    model_builder = Stage2Trainer(config, str(device))
    model_builder._create_model(feature_dims)
    game_ebm_path = checkpoint_dir / 'game_ebm.pt'
    if not game_ebm_path.exists():
        raise FileNotFoundError(f"未找到 EBM 权重: {game_ebm_path}")
    state_dict = torch.load(game_ebm_path, map_location=device)
    model_builder.model.load_state_dict(state_dict)
    game_ebm = model_builder.model

    guardian_path = checkpoint_dir / 'guardian'
    if guardian_path.exists():
        guardian = RiskGuardian.load(guardian_path)
    else:
        print("    警告: 未找到 Guardian，使用空安全门控")
        guardian = RiskGuardian(model_type='sklearn', threshold=1.0)
        guardian._is_dummy = True

    integrator = Stage4Integrator(config, game_ebm, guardian, str(device))
    run_backtest(config, splits.test, integrator, feature_cols, str(checkpoint_dir))


def run_baselines(config: Config, args):
    """运行 baseline 复现实验。"""
    print("\n" + "=" * 70)
    print("    Baseline 复现实验模式")
    print("=" * 70)

    set_seed(args.seed)
    device = resolve_runtime_device(config)

    baseline_output_dir = args.output_dir
    if baseline_output_dir == 'outputs':
        baseline_output_dir = str(PROJECT_ROOT / 'outputs_baselines' / f"{config.data.market_profile}_{config.data.mode}")
    elif Path(baseline_output_dir).name == 'outputs_baselines':
        baseline_output_dir = str(Path(baseline_output_dir) / f"{config.data.market_profile}_{config.data.mode}")

    splits, full_df, feature_cols = run_data_pipeline(
        config,
        fast=args.fast,
        no_cache=args.no_cache,
    )

    run_config = BaselineRunConfig(
        baseline_name=args.baseline_name,
        output_dir=baseline_output_dir,
        epochs_override=args.baseline_epochs,
        seq_len_override=args.baseline_seq_len,
        hf_backbone_override=args.baseline_backbone,
        quick=args.quick_baseline,
    )
    runner = BaselineRunner(config, run_config, str(device))
    summaries = runner.run(splits, feature_cols)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        print("\n>>> Baseline 汇总:")
        print(summary_df[['baseline', 'valid_ic', 'test_ic', 'annual_return', 'sharpe_ratio', 'max_drawdown']])
        print(f"\n结果已保存到: {baseline_output_dir}")


if __name__ == "__main__":
    main()
