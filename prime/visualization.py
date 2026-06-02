"""
宏观感知物理博弈能量模型 - 可视化模块

包含三个核心可视化：
1. 宏观自适应学习曲线 - α, β, γ 随训练轮次的变化
2. 博弈相平面 - Bull Force vs Bear Force 散点图
3. 守护者判别力 - 安全股vs崩盘股分布

Author: Macro-Game EBM Team
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization disabled")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# 图片保存辅助函数
# =============================================================================

def save_figure(
    fig_or_path: Union[str, Path],
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 150,
    bbox_inches: str = 'tight',
    facecolor: str = 'white',
    transparent: bool = False
):
    """
    同时保存图片为多种格式（PNG和PDF）
    
    Args:
        fig_or_path: 保存路径（不含扩展名）或完整路径
        formats: 要保存的格式列表，默认['png', 'pdf']
        dpi: PNG分辨率
        bbox_inches: 边界框设置
        facecolor: 背景色
        transparent: 是否透明背景（PDF推荐False）
    
    Returns:
        List[Path]: 保存的文件路径列表
    """
    if not HAS_MATPLOTLIB:
        return []
    
    # 处理路径
    path = Path(fig_or_path)
    
    # 如果路径包含扩展名，移除它
    if path.suffix.lower() in ['.png', '.pdf', '.svg', '.jpg', '.jpeg']:
        base_path = path.parent / path.stem
    else:
        base_path = path
    
    saved_paths = []
    
    for fmt in formats:
        fmt = fmt.lower().strip('.')
        save_path = base_path.parent / f"{base_path.name}.{fmt}"
        
        try:
            if fmt == 'pdf':
                # PDF使用矢量格式，不需要高DPI
                plt.savefig(
                    save_path, 
                    format='pdf',
                    bbox_inches=bbox_inches, 
                    facecolor=facecolor,
                    transparent=transparent
                )
            else:
                # PNG等栅格格式使用高DPI
                plt.savefig(
                    save_path, 
                    format=fmt,
                    dpi=dpi, 
                    bbox_inches=bbox_inches, 
                    facecolor=facecolor,
                    transparent=transparent
                )
            saved_paths.append(save_path)
        except Exception as e:
            print(f"  ⚠ 保存{fmt.upper()}失败: {e}")
    
    return saved_paths


def print_saved_paths(paths: List[Path], prefix: str = "  ✓ 保存"):
    """打印保存的文件路径"""
    if not paths:
        return
    
    # 按扩展名分组
    by_ext = {}
    for p in paths:
        ext = p.suffix.lower()
        if ext not in by_ext:
            by_ext[ext] = []
        by_ext[ext].append(p)
    
    # 打印基础名称和格式
    base_name = paths[0].stem
    formats_str = ', '.join(sorted(by_ext.keys()))
    print(f"{prefix}: {paths[0].parent / base_name} ({formats_str})")


class TrainingHistoryRecorder:
    """
    训练历史记录器
    
    记录每个epoch的：
    - α (alpha_market): Bull能量的宏观调制系数
    - β (beta_risk): Bear能量的宏观调制系数  
    - γ (gamma_heat): Friction能量的宏观调制系数
    - IC, Rank IC等指标
    """
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'alpha_mean': [],
            'alpha_std': [],
            'beta_mean': [],
            'beta_std': [],
            'gamma_mean': [],
            'gamma_std': [],
            'train_loss': [],
            'valid_ic': [],
            'valid_rank_ic': [],
            'train_rank_ic': [],
        }
    
    def record_epoch(
        self,
        epoch: int,
        alpha_mean: float,
        alpha_std: float,
        beta_mean: float,
        beta_std: float,
        gamma_mean: float,
        gamma_std: float,
        train_loss: float = 0.0,
        valid_ic: float = 0.0,
        valid_rank_ic: float = 0.0,
        train_rank_ic: float = 0.0
    ):
        """记录一个epoch的数据"""
        # 【修复】确保所有值都是Python原生类型
        def to_python(v):
            if hasattr(v, 'item'):  # numpy类型
                return v.item()
            return float(v) if v is not None else 0.0
        
        self.history['epoch'].append(int(epoch))
        self.history['alpha_mean'].append(to_python(alpha_mean))
        self.history['alpha_std'].append(to_python(alpha_std))
        self.history['beta_mean'].append(to_python(beta_mean))
        self.history['beta_std'].append(to_python(beta_std))
        self.history['gamma_mean'].append(to_python(gamma_mean))
        self.history['gamma_std'].append(to_python(gamma_std))
        self.history['train_loss'].append(to_python(train_loss))
        self.history['valid_ic'].append(to_python(valid_ic))
        self.history['valid_rank_ic'].append(to_python(valid_rank_ic))
        self.history['train_rank_ic'].append(to_python(train_rank_ic))
    
    def save(self, path: str):
        """保存训练历史到JSON文件"""
        # 【修复】将numpy类型转换为Python原生类型
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [
                float(v) if hasattr(v, 'item') else v for v in values
            ]
        
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"  ✓ 训练历史保存至: {path}")
    
    def load(self, path: str):
        """从JSON文件加载训练历史"""
        with open(path, 'r') as f:
            self.history = json.load(f)
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame(self.history)


class GamePhaseSpaceRecorder:
    """
    博弈相平面数据记录器
    
    记录每只股票的：
    - E_bull: Bull能量（动力）
    - E_bear: Bear能量（阻力）
    - E_friction: Friction能量（摩擦）
    - total_energy: 总能量
    - is_selected: 是否被选中（Top K）
    - return_5d: 5日收益（用于验证）
    """
    
    def __init__(self):
        self.data = []
    
    def record_day(
        self,
        date: str,
        ts_codes: List[str],
        E_bull: np.ndarray,
        E_bear: np.ndarray,
        E_friction: np.ndarray,
        total_energy: np.ndarray,
        selected_mask: np.ndarray,
        returns: Optional[np.ndarray] = None
    ):
        """记录一天的博弈相平面数据"""
        n = len(ts_codes)
        for i in range(n):
            record = {
                'date': date,
                'ts_code': ts_codes[i],
                'E_bull': float(E_bull[i]),
                'E_bear': float(E_bear[i]),
                'E_friction': float(E_friction[i]),
                'total_energy': float(total_energy[i]),
                'is_selected': bool(selected_mask[i]),
                'return_5d': float(returns[i]) if returns is not None else None
            }
            self.data.append(record)
    
    def save(self, path: str):
        """保存到CSV文件"""
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)
        print(f"  ✓ 博弈相平面数据保存至: {path}")
    
    def load(self, path: str):
        """从CSV加载"""
        df = pd.read_csv(path)
        self.data = df.to_dict('records')
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame(self.data)


class GuardianDistributionRecorder:
    """
    守护者分布数据记录器
    
    记录：
    - safe_scores: 安全股票的Guardian分数分布
    - crash_scores: 崩盘股票的Guardian分数分布
    """
    
    def __init__(self):
        self.safe_scores = []
        self.crash_scores = []
        self.all_scores = []
        self.all_labels = []
    
    def record(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ):
        """记录分数和标签"""
        # 【修复】确保转换为Python原生类型
        self.all_scores = [float(x) for x in scores]
        self.all_labels = [int(x) for x in labels]
        
        # 分离安全股和崩盘股
        self.safe_scores = [float(x) for x in scores[labels == 0]]
        self.crash_scores = [float(x) for x in scores[labels == 1]]
    
    def save(self, path: str):
        """保存到JSON"""
        data = {
            'safe_scores': self.safe_scores,
            'crash_scores': self.crash_scores,
            'all_scores': self.all_scores,
            'all_labels': self.all_labels
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  ✓ Guardian分布数据保存至: {path}")
    
    def load(self, path: str):
        """从JSON加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.safe_scores = data['safe_scores']
        self.crash_scores = data['crash_scores']
        self.all_scores = data.get('all_scores', [])
        self.all_labels = data.get('all_labels', [])
        return self


class ModelVisualizer:
    """
    模型可视化器
    
    生成三个核心图表：
    1. 宏观自适应学习曲线
    2. 博弈相平面
    3. 守护者判别力
    """
    
    def __init__(self, output_dir: str = 'outputs/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体和样式
        if HAS_MATPLOTLIB:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_learning_curves(
        self,
        history: TrainingHistoryRecorder,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        图1: 宏观自适应学习曲线
        
        【修改】生成4个独立的图表文件
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return
        
        df = history.to_dataframe()
        epochs = df['epoch'].values
        
        # 获取保存目录
        if save_path:
            save_dir = Path(save_path).parent
            base_name = Path(save_path).stem
        else:
            save_dir = self.output_dir
            base_name = 'learning_curves'
        
        # ========== 图1-1: α, β, γ 的收敛曲线 ==========
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        
        ax1.plot(epochs, df['alpha_mean'], 'b-', linewidth=2.5, label=r'$\alpha$ (Bull Modulation)')
        ax1.fill_between(epochs, 
                        df['alpha_mean'] - df['alpha_std'],
                        df['alpha_mean'] + df['alpha_std'],
                        alpha=0.2, color='blue')
        
        ax1.plot(epochs, df['beta_mean'], 'r-', linewidth=2.5, label=r'$\beta$ (Bear Modulation)')
        ax1.fill_between(epochs,
                        df['beta_mean'] - df['beta_std'],
                        df['beta_mean'] + df['beta_std'],
                        alpha=0.2, color='red')
        
        ax1.plot(epochs, df['gamma_mean'], 'g-', linewidth=2.5, label=r'$\gamma$ (Friction Modulation)')
        ax1.fill_between(epochs,
                        df['gamma_mean'] - df['gamma_std'],
                        df['gamma_mean'] + df['gamma_std'],
                        alpha=0.2, color='green')
        
        # 添加最终收敛值标注
        final_alpha = df['alpha_mean'].iloc[-1]
        final_beta = df['beta_mean'].iloc[-1]
        final_gamma = df['gamma_mean'].iloc[-1]
        
        ax1.axhline(y=final_alpha, color='blue', linestyle='--', alpha=0.5)
        ax1.axhline(y=final_beta, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=final_gamma, color='green', linestyle='--', alpha=0.5)
        
        ax1.text(epochs[-1] + 0.3, final_alpha, f'{final_alpha:.2f}', 
                color='blue', fontsize=12, va='center', fontweight='bold')
        ax1.text(epochs[-1] + 0.3, final_beta, f'{final_beta:.2f}', 
                color='red', fontsize=12, va='center', fontweight='bold')
        ax1.text(epochs[-1] + 0.3, final_gamma, f'{final_gamma:.2f}', 
                color='green', fontsize=12, va='center', fontweight='bold')
        
        ax1.set_xlabel('Training Epoch', fontsize=14)
        ax1.set_ylabel('Modulation Coefficient', fontsize=14)
        ax1.set_title('Macro-Adaptive Learning Curves\n' + 
                     r'$\alpha$, $\beta$, $\gamma$ Convergence During Training', 
                     fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(epochs[0] - 0.5, epochs[-1] + 1.5)
        
        plt.tight_layout()
        path1 = save_dir / f'{base_name}_1_modulation'
        saved = save_figure(path1, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 调制系数收敛图保存至")
        if show:
            plt.show()
        plt.close()
        
        # ========== 图1-2: IC曲线 ==========
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        
        ax2.plot(epochs, df['valid_ic'], 'b-', linewidth=2.5, label='Validation IC', marker='o', markersize=6)
        ax2.plot(epochs, df['valid_rank_ic'], 'r--', linewidth=2.5, label='Validation Rank IC', marker='s', markersize=6)
        if 'train_rank_ic' in df.columns and df['train_rank_ic'].sum() > 0:
            ax2.plot(epochs, df['train_rank_ic'], 'g:', linewidth=2.5, label='Train Rank IC', marker='^', markersize=6)
        
        ax2.set_xlabel('Training Epoch', fontsize=14)
        ax2.set_ylabel('IC Value', fontsize=14)
        ax2.set_title('Information Coefficient Over Training\n' +
                     'Measuring Prediction-Return Correlation', 
                     fontsize=16, fontweight='bold')
        ax2.legend(loc='best', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        # 添加最终值标注
        final_valid_ic = df['valid_ic'].iloc[-1]
        final_valid_rank_ic = df['valid_rank_ic'].iloc[-1]
        ax2.annotate(f'Final: {final_valid_ic:.4f}', 
                    xy=(epochs[-1], final_valid_ic),
                    xytext=(epochs[-1]-1, final_valid_ic + 0.05),
                    fontsize=11, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        
        plt.tight_layout()
        path2 = save_dir / f'{base_name}_2_ic'
        saved = save_figure(path2, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ IC曲线图保存至")
        if show:
            plt.show()
        plt.close()
        
        # ========== 图1-3: 训练损失 ==========
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        
        ax3.plot(epochs, df['train_loss'], 'purple', linewidth=2.5, marker='o', markersize=6)
        ax3.fill_between(epochs, df['train_loss'], alpha=0.2, color='purple')
        
        ax3.set_xlabel('Training Epoch', fontsize=14)
        ax3.set_ylabel('Training Loss', fontsize=14)
        ax3.set_title('Training Loss Convergence\n' +
                     'Energy-Based Contrastive Learning', 
                     fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加起始和结束值
        start_loss = df['train_loss'].iloc[0]
        end_loss = df['train_loss'].iloc[-1]
        ax3.annotate(f'Start: {start_loss:.4f}', 
                    xy=(epochs[0], start_loss),
                    xytext=(epochs[0]+0.5, start_loss + 0.05),
                    fontsize=11, color='purple')
        ax3.annotate(f'End: {end_loss:.4f}', 
                    xy=(epochs[-1], end_loss),
                    xytext=(epochs[-1]-1.5, end_loss + 0.05),
                    fontsize=11, color='purple')
        
        plt.tight_layout()
        path3 = save_dir / f'{base_name}_3_loss'
        saved = save_figure(path3, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 损失收敛图保存至")
        if show:
            plt.show()
        plt.close()
        
        # ========== 图1-4: 参数收敛分析（条形图）==========
        fig4, ax4 = plt.subplots(figsize=(10, 7))
        
        params = [r'$\alpha$ (Bull)', r'$\beta$ (Bear)', r'$\gamma$ (Friction)']
        initial_values = [df['alpha_mean'].iloc[0], df['beta_mean'].iloc[0], df['gamma_mean'].iloc[0]]
        final_values = [final_alpha, final_beta, final_gamma]
        
        x = np.arange(len(params))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, initial_values, width, label='Initial', 
                       color='lightgray', edgecolor='black', linewidth=1.5)
        bars2 = ax4.bar(x + width/2, final_values, width, label='Final', 
                       color=['#1f77b4', '#d62728', '#2ca02c'], alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        
        ax4.set_ylabel('Coefficient Value', fontsize=14)
        ax4.set_title('Parameter Convergence: Initial vs Final\n' +
                     'Model Self-Discovered Market Physics', 
                     fontsize=16, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(params, fontsize=13)
        ax4.legend(fontsize=12, loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars1, initial_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, color='gray')
        for bar, val in zip(bars2, final_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加变化箭头和百分比
        for i, (init, final) in enumerate(zip(initial_values, final_values)):
            change_pct = (final - init) / init * 100
            color = 'green' if change_pct < 0 else 'red'
            ax4.annotate(f'{change_pct:+.0f}%', 
                        xy=(i, max(init, final) + 0.15),
                        fontsize=11, ha='center', color=color, fontweight='bold')
        
        plt.tight_layout()
        path4 = save_dir / f'{base_name}_4_convergence'
        saved = save_figure(path4, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 参数收敛对比图保存至")
        if show:
            plt.show()
        plt.close()
        
        print(f"\n  ✓ 学习曲线图表已分别保存（共4个，每个含PNG和PDF格式）")
    
    def plot_game_phase_space(
        self,
        phase_data: GamePhaseSpaceRecorder,
        sample_date: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        图2: 博弈相平面 (The Game Phase Space)
        
        X轴: Bull Force (动力)
        Y轴: Bear Force (阻力)
        颜色: 总能量 (越绿越好)
        蓝圈: 选中的Top K股票
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return
        
        df = phase_data.to_dataframe()
        
        # 如果指定日期，只绘制该日期的数据
        if sample_date:
            df = df[df['date'] == sample_date]
        else:
            # 取最后一天的数据
            dates = df['date'].unique()
            if len(dates) > 0:
                sample_date = dates[-1]
                df = df[df['date'] == sample_date]
        
        if df.empty:
            print("No data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 分离选中和未选中的股票
        selected = df[df['is_selected'] == True]
        not_selected = df[df['is_selected'] == False]
        
        # 创建自定义颜色映射（红-黄-绿，能量从高到低）
        colors = ['#d62728', '#ff7f0e', '#ffff00', '#98df8a', '#2ca02c']
        cmap = LinearSegmentedColormap.from_list('energy', colors[::-1])  # 反转，低能量为绿
        
        # 绘制未选中的股票
        scatter1 = ax.scatter(
            not_selected['E_bull'],
            not_selected['E_bear'],
            c=not_selected['total_energy'],
            cmap=cmap,
            s=50,
            alpha=0.6,
            edgecolors='none'
        )
        
        # 绘制选中的股票（蓝色空心圆）
        ax.scatter(
            selected['E_bull'],
            selected['E_bear'],
            c=selected['total_energy'],
            cmap=cmap,
            s=120,
            alpha=0.9,
            edgecolors='blue',
            linewidths=2,
            marker='o'
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter1, ax=ax, shrink=0.8)
        cbar.set_label('Total Energy (Lower is Better)', fontsize=12)
        
        # 添加对角线参考线（Bull = Bear）
        max_val = max(df['E_bull'].max(), df['E_bear'].max()) * 1.1
        min_val = min(df['E_bull'].min(), df['E_bear'].min()) * 0.9
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
        
        # 添加有效前沿参考线
        # 斜率0.5和1.5的参考线，显示trade-off区域
        ax.plot([min_val, max_val], [min_val*0.5, max_val*0.5], 'gray', linestyle='--', alpha=0.3)
        ax.plot([min_val, max_val], [min_val*1.5, max_val*1.5], 'gray', linestyle='--', alpha=0.3)
        
        # 设置标签和标题
        ax.set_xlabel('Bull Force (Momentum/Flow) →', fontsize=14)
        ax.set_ylabel('Bear Force (Valuation/Risk) →', fontsize=14)
        ax.set_title('The Game Phase Space: Balancing Greed & Fear', fontsize=16, fontweight='bold')
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=2, label=f'Selected (Top {len(selected)})')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        # 添加统计信息文本框
        stats_text = (
            f"Date: {sample_date}\n"
            f"Total Stocks: {len(df)}\n"
            f"Selected: {len(selected)}\n"
            f"Avg Bull (Selected): {selected['E_bull'].mean():.3f}\n"
            f"Avg Bear (Selected): {selected['E_bear'].mean():.3f}\n"
            f"Avg Energy (Selected): {selected['total_energy'].mean():.3f}"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            # 移除扩展名以便同时保存多种格式
            save_base = Path(save_path)
            if save_base.suffix.lower() in ['.png', '.pdf']:
                save_base = save_base.parent / save_base.stem
            saved = save_figure(save_base, formats=['png', 'pdf'])
            print_saved_paths(saved, "  ✓ 博弈相平面图保存至")
        else:
            default_path = self.output_dir / 'game_phase_space'
            saved = save_figure(default_path, formats=['png', 'pdf'])
            print_saved_paths(saved, "  ✓ 博弈相平面图保存至")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_guardian_distribution(
        self,
        guardian_data: GuardianDistributionRecorder,
        threshold: float = 0.5,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        图3: 守护者判别力
        
        【修改】生成2个独立的图表文件
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return
        
        safe_scores = np.array(guardian_data.safe_scores)
        crash_scores = np.array(guardian_data.crash_scores)
        
        # 获取保存目录
        if save_path:
            save_dir = Path(save_path).parent
            base_name = Path(save_path).stem
        else:
            save_dir = self.output_dir
            base_name = 'guardian_distribution'
        
        # 计算统计指标
        cohens_d = 0
        tpr = 0
        fpr = 0
        if len(safe_scores) > 0 and len(crash_scores) > 0:
            pooled_std = np.sqrt((np.var(safe_scores) + np.var(crash_scores)) / 2)
            cohens_d = (np.mean(crash_scores) - np.mean(safe_scores)) / (pooled_std + 1e-6)
            tpr = np.mean(crash_scores >= threshold)
            fpr = np.mean(safe_scores >= threshold)
        
        # ========== 图3-1: 分布重叠图 (KDE) ==========
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        if len(safe_scores) > 0:
            sns.kdeplot(safe_scores, ax=ax1, color='green', fill=True, alpha=0.4, 
                       label=f'Safe Stocks (n={len(safe_scores):,})', linewidth=2.5)
        if len(crash_scores) > 0:
            sns.kdeplot(crash_scores, ax=ax1, color='red', fill=True, alpha=0.4,
                       label=f'Crash Stocks (n={len(crash_scores):,})', linewidth=2.5)
        
        # 添加阈值线
        ax1.axvline(x=threshold, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Threshold = {threshold:.2f}')
        
        ax1.set_xlabel('Guardian Risk Score', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.set_title('Guardian Discriminability: Safe vs Crash Stocks\n' +
                     'Risk Score Distribution Overlap', 
                     fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        if len(safe_scores) > 0 and len(crash_scores) > 0:
            stats_text = (
                f"Cohen's d: {cohens_d:.2f}\n"
                f"TPR (Recall): {tpr:.2%}\n"
                f"FPR: {fpr:.2%}\n"
                f"Safe Mean: {np.mean(safe_scores):.3f}\n"
                f"Crash Mean: {np.mean(crash_scores):.3f}"
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        path1 = save_dir / f'{base_name}_1_kde'
        saved = save_figure(path1, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ Guardian KDE分布图保存至")
        if show:
            plt.show()
        plt.close()
        
        # ========== 图3-2: 箱线图比较 ==========
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        box_data = []
        box_labels = []
        box_colors = []
        
        if len(safe_scores) > 0:
            box_data.append(safe_scores)
            box_labels.append('Safe\nStocks')
            box_colors.append('#2ca02c')
        if len(crash_scores) > 0:
            box_data.append(crash_scores)
            box_labels.append('Crash\nStocks')
            box_colors.append('#d62728')
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
            
            # 设置中位线颜色
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            
            # 添加阈值线
            ax2.axhline(y=threshold, color='blue', linestyle='--', linewidth=2.5,
                       label=f'Threshold = {threshold:.2f}')
            
            # 添加均值点
            means = [np.mean(d) for d in box_data]
            positions = range(1, len(box_data) + 1)
            ax2.scatter(positions, means, color='white', s=100, zorder=5, 
                       edgecolors='black', linewidths=2, marker='D', label='Mean')
        
        ax2.set_ylabel('Guardian Risk Score', fontsize=14)
        ax2.set_title('Risk Score Distribution Comparison\n' +
                     'Box Plot: Safe vs Crash Stocks', 
                     fontsize=16, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加样本量标注
        for i, (label, n) in enumerate(zip(box_labels, [len(safe_scores), len(crash_scores)])):
            ax2.text(i + 1, ax2.get_ylim()[0] - 0.02, f'n={n:,}', 
                    ha='center', va='top', fontsize=11, color='gray')
        
        plt.tight_layout()
        path2 = save_dir / f'{base_name}_2_boxplot'
        saved = save_figure(path2, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ Guardian箱线图保存至")
        if show:
            plt.show()
        plt.close()
        
        print(f"\n  ✓ Guardian判别力图表已分别保存（共2个，每个含PNG和PDF格式）")
    
    def plot_all(
        self,
        history_path: str,
        phase_space_path: str,
        guardian_path: str,
        guardian_threshold: float = 0.5,
        show: bool = True
    ):
        """
        一次性绘制所有三个图表
        """
        print("\n" + "=" * 60)
        print("    生成可视化图表")
        print("=" * 60)
        
        # 加载数据
        history = TrainingHistoryRecorder().load(history_path)
        phase_data = GamePhaseSpaceRecorder().load(phase_space_path)
        guardian_data = GuardianDistributionRecorder().load(guardian_path)
        
        # 绘制图表
        self.plot_learning_curves(history, show=show)
        self.plot_game_phase_space(phase_data, show=show)
        self.plot_guardian_distribution(guardian_data, threshold=guardian_threshold, show=show)
        
        print("\n  ✓ 所有图表生成完成!")


def create_sample_visualization():
    """
    创建示例可视化（用于测试）
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    # 创建模拟的训练历史数据
    history = TrainingHistoryRecorder()
    for epoch in range(1, 31):
        # 模拟α, β, γ的收敛过程
        alpha = 1.0 - 0.3 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.02)
        beta = 1.0 - 0.2 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.02)
        gamma = 0.5 - 0.3 * (1 - np.exp(-epoch/8)) + np.random.normal(0, 0.02)
        
        history.record_epoch(
            epoch=epoch,
            alpha_mean=alpha,
            alpha_std=0.1 - 0.05 * epoch/30,
            beta_mean=beta,
            beta_std=0.15 - 0.08 * epoch/30,
            gamma_mean=max(0.1, gamma),
            gamma_std=0.1 - 0.05 * epoch/30,
            train_loss=-0.5 - 0.3 * (1 - np.exp(-epoch/5)),
            valid_ic=0.03 + 0.02 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.005),
            valid_rank_ic=0.01 + 0.015 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.003),
            train_rank_ic=0.2 + 0.3 * (1 - np.exp(-epoch/5)) + np.random.normal(0, 0.02)
        )
    
    # 创建模拟的博弈相平面数据
    phase_data = GamePhaseSpaceRecorder()
    n_stocks = 500
    
    # 生成相关的Bull和Bear能量
    E_bull = np.random.beta(2, 3, n_stocks) * 1.2 - 0.1
    E_bear = 0.3 * E_bull + np.random.beta(2, 2, n_stocks) * 0.8
    E_friction = np.random.beta(2, 3, n_stocks) * 0.6
    
    # 总能量 = Bull - Bear + Friction
    total_energy = -E_bull + E_bear + 0.3 * E_friction
    
    # 选择能量最低的50只股票
    top_k = 50
    selected_indices = np.argsort(total_energy)[:top_k]
    selected_mask = np.zeros(n_stocks, dtype=bool)
    selected_mask[selected_indices] = True
    
    phase_data.record_day(
        date='2020-06-15',
        ts_codes=[f'stock_{i:03d}' for i in range(n_stocks)],
        E_bull=E_bull,
        E_bear=E_bear,
        E_friction=E_friction,
        total_energy=total_energy,
        selected_mask=selected_mask
    )
    
    # 创建模拟的Guardian分布数据
    guardian_data = GuardianDistributionRecorder()
    
    # 安全股票：均值0.3，标准差0.15
    safe_scores = np.random.beta(2, 5, 5000) * 0.8 + 0.1
    # 崩盘股票：均值0.6，标准差0.2
    crash_scores = np.random.beta(3, 2, 500) * 0.7 + 0.2
    
    all_scores = np.concatenate([safe_scores, crash_scores])
    all_labels = np.concatenate([np.zeros(len(safe_scores)), np.ones(len(crash_scores))])
    
    guardian_data.record(all_scores, all_labels)
    
    # 保存数据
    output_dir = Path('outputs/viz_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history.save(str(output_dir / 'training_history.json'))
    phase_data.save(str(output_dir / 'game_phase_space.csv'))
    guardian_data.save(str(output_dir / 'guardian_distribution.json'))
    
    # 绘制图表
    visualizer = ModelVisualizer()
    visualizer.plot_learning_curves(history, show=False)
    visualizer.plot_game_phase_space(phase_data, show=False)
    visualizer.plot_guardian_distribution(guardian_data, threshold=0.45, show=False)
    
    print("\n  ✓ 示例可视化完成!")


# =============================================================================
# 消融实验可视化
# =============================================================================

class AblationVisualizer:
    """
    消融实验可视化器
    
    生成消融实验相关图表：
    1. 性能对比柱状图（年化收益、IC、夏普比率）
    2. 组件贡献度分析（饼图+柱状图）
    3. 多维雷达图
    4. 热力图（不同seed下的性能）
    """
    
    def __init__(self, output_dir: str = 'outputs/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                try:
                    plt.style.use('ggplot')
                except:
                    pass
        
        # 配色方案
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#28A745',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#17A2B8',
        }
        
        self.ablation_colors = {
            'baseline': '#2E86AB',
            'no_bull': '#C73E1D',
            'no_bear': '#F18F01',
            'no_friction': '#A23B72',
            'no_guardian': '#6C757D',
            'no_macro_modulation': '#28A745',
            'no_macro_timing': '#17A2B8',
            'bull_only': '#E74C3C',
            'bear_only': '#9B59B6',
        }
    
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """加载消融实验数据"""
        data_path = Path(data_dir)
        csv_files = list(data_path.glob('ablation_results_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"未找到消融实验结果文件: {data_path}")
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"  加载数据: {latest_csv}")
        return pd.read_csv(latest_csv)
    
    def plot_comparison(self, df: pd.DataFrame, metric: str = 'annual_return', 
                       save_path: Optional[str] = None, show: bool = False):
        """图1: 性能对比柱状图"""
        if not HAS_MATPLOTLIB:
            return
        
        grouped = df.groupby('exp_name').agg({
            metric: ['mean', 'std'],
            'description': 'first'
        }).reset_index()
        grouped.columns = ['exp_name', 'mean', 'std', 'description']
        grouped = grouped.sort_values('mean', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(grouped))
        colors = [self.ablation_colors.get(name, '#6C757D') for name in grouped['exp_name']]
        
        bars = ax.barh(y_pos, grouped['mean'], xerr=grouped['std'], 
                       color=colors, alpha=0.8, capsize=5, ecolor='gray')
        
        for i, (mean, std) in enumerate(zip(grouped['mean'], grouped['std'])):
            if metric == 'annual_return':
                label = f'{mean:.1%} ± {std:.1%}'
            elif metric == 'sharpe_ratio':
                label = f'{mean:.2f} ± {std:.2f}'
            else:
                label = f'{mean:.4f} ± {std:.4f}'
            ax.text(mean + std + 0.02, i, label, va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['exp_name']}" for _, row in grouped.iterrows()], fontsize=10)
        
        metric_names = {
            'annual_return': 'Annual Return',
            'sharpe_ratio': 'Sharpe Ratio',
            'test_ic': 'Information Coefficient (IC)',
        }
        ax.set_xlabel(metric_names.get(metric, metric), fontsize=12)
        ax.set_title(f'Ablation Study: {metric_names.get(metric, metric)} Comparison', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'ablation_comparison_{metric}'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_contribution(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图2: 组件贡献度分析"""
        if not HAS_MATPLOTLIB:
            return
        
        grouped = df.groupby('exp_name')['annual_return'].mean()
        baseline = grouped.get('no_guardian', grouped.max())
        
        contributions = {}
        mapping = {
            'no_bull': 'Bull Energy',
            'no_bear': 'Bear Energy',
            'no_friction': 'Friction Energy',
            'no_macro_modulation': 'Macro Modulation',
        }
        
        for exp_name, display_name in mapping.items():
            if exp_name in grouped.index:
                contribution = baseline - grouped[exp_name]
                contributions[display_name] = max(0, contribution)
        
        if not contributions:
            print("  ⚠ 无法计算组件贡献度")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        labels = list(contributions.keys())
        sizes = list(contributions.values())
        colors = [self.colors['primary'], self.colors['secondary'], 
                  self.colors['warning'], self.colors['success']][:len(labels)]
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title('Component Contribution Distribution', fontsize=12, fontweight='bold')
        
        bars = ax2.bar(labels, [c * 100 for c in sizes], color=colors, alpha=0.8)
        ax2.set_ylabel('Contribution (% Return)', fontsize=11)
        ax2.set_title('Absolute Contribution', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / 'ablation_contribution'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图3: 热力图"""
        if not HAS_MATPLOTLIB:
            return
        
        pivot = df.pivot_table(index='exp_name', columns='seed', values='annual_return', aggfunc='mean')
        if pivot.empty:
            print("  ⚠ 数据不足，跳过热力图")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=pivot.values.mean() * 100, ax=ax,
                   cbar_kws={'label': 'Annual Return (%)'})
        ax.set_title('Ablation: Performance Across Seeds', fontsize=14, fontweight='bold')
        ax.set_xlabel('Random Seed', fontsize=12)
        ax.set_ylabel('Experiment', fontsize=12)
        
        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / 'ablation_heatmap'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_all(self, data_dir: str, show: bool = False):
        """生成所有消融实验图表"""
        print("\n>>> 生成消融实验可视化图表...")
        df = self.load_data(data_dir)
        
        self.plot_comparison(df, metric='annual_return', show=show)
        self.plot_comparison(df, metric='test_ic', show=show)
        self.plot_comparison(df, metric='sharpe_ratio', show=show)
        self.plot_contribution(df, show=show)
        self.plot_heatmap(df, show=show)
        
        print(f"\n✓ 消融实验图表保存至: {self.output_dir}")


# =============================================================================
# 鲁棒性实验可视化
# =============================================================================

class RobustnessVisualizer:
    """
    鲁棒性实验可视化器
    
    生成鲁棒性实验相关图表：
    1. 噪声敏感性曲线
    2. Top-K敏感性曲线
    3. 学习率敏感性曲线
    4. 训练轮数敏感性曲线
    5. 汇总图
    """
    
    def __init__(self, output_dir: str = 'outputs/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                try:
                    plt.style.use('ggplot')
                except:
                    pass
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#28A745',
            'warning': '#F18F01',
            'danger': '#C73E1D',
        }
    
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """加载鲁棒性实验数据"""
        data_path = Path(data_dir)
        csv_files = list(data_path.glob('robustness_results_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"未找到鲁棒性实验结果文件: {data_path}")
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"  加载数据: {latest_csv}")
        return pd.read_csv(latest_csv)
    
    def plot_noise_sensitivity(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图1: 噪声敏感性"""
        if not HAS_MATPLOTLIB:
            return
        
        noise_df = df[df['exp_type'] == 'noise'].sort_values('param_value')
        if noise_df.empty:
            print("  ⚠ 无噪声敏感性数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('annual_return', 'Annual Return (%)', axes[0, 0], True),
            ('sharpe_ratio', 'Sharpe Ratio', axes[0, 1], False),
            ('test_ic', 'IC', axes[1, 0], False),
            ('max_drawdown', 'Max Drawdown (%)', axes[1, 1], True)
        ]
        
        for metric, title, ax, is_pct in metrics:
            x = noise_df['param_value'] * 100
            y = noise_df[metric] * 100 if is_pct else noise_df[metric]
            
            ax.plot(x, y, 'o-', linewidth=2, markersize=8, color=self.colors['primary'])
            ax.fill_between(x, y, alpha=0.2, color=self.colors['primary'])
            
            for xi, yi in zip(x, y):
                label = f'{yi:.1f}%' if is_pct else f'{yi:.4f}'
                ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
            
            ax.set_xlabel('Noise Level (%)', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title} vs Noise', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Noise Sensitivity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'robustness_noise'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_topk_sensitivity(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图2: Top-K敏感性"""
        if not HAS_MATPLOTLIB:
            return
        
        topk_df = df[df['exp_type'] == 'topk'].sort_values('param_value')
        if topk_df.empty:
            print("  ⚠ 无Top-K敏感性数据")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x = topk_df['param_value']
        y1 = topk_df['annual_return'] * 100
        y2 = topk_df['sharpe_ratio']
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(x, y1, 'o-', linewidth=2.5, markersize=10, 
                       color=self.colors['primary'], label='Annual Return')
        line2 = ax2.plot(x, y2, 's--', linewidth=2.5, markersize=10,
                        color=self.colors['secondary'], label='Sharpe Ratio')
        
        ax.set_xlabel('Top-K', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12, color=self.colors['primary'])
        ax2.set_ylabel('Sharpe Ratio', fontsize=12, color=self.colors['secondary'])
        ax.tick_params(axis='y', labelcolor=self.colors['primary'])
        ax2.tick_params(axis='y', labelcolor=self.colors['secondary'])
        
        for xi, yi in zip(x, y1):
            ax.annotate(f'{yi:.1f}%', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        
        ax.set_title('Top-K Sensitivity Analysis', fontsize=14, fontweight='bold')
        lines = line1 + line2
        ax.legend(lines, [l.get_label() for l in lines], loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / 'robustness_topk'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_lr_sensitivity(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图3: 学习率敏感性"""
        if not HAS_MATPLOTLIB:
            return
        
        lr_df = df[df['exp_type'] == 'lr'].sort_values('param_value')
        if lr_df.empty:
            print("  ⚠ 无学习率敏感性数据")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = [
            ('annual_return', 'Annual Return (%)', axes[0], True),
            ('sharpe_ratio', 'Sharpe Ratio', axes[1], False),
            ('test_ic', 'IC', axes[2], False)
        ]
        
        for metric, title, ax, is_pct in metrics:
            x = lr_df['param_value']
            y = lr_df[metric] * 100 if is_pct else lr_df[metric]
            
            ax.semilogx(x, y, 'o-', linewidth=2.5, markersize=10, color=self.colors['warning'])
            
            for xi, yi in zip(x, y):
                label = f'{yi:.1f}%' if is_pct else f'{yi:.3f}'
                ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
            
            ax.set_xlabel('Learning Rate (log)', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title} vs LR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Learning Rate Sensitivity', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'robustness_lr'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_epochs_sensitivity(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图4: 训练轮数敏感性"""
        if not HAS_MATPLOTLIB:
            return
        
        epochs_df = df[df['exp_type'] == 'epochs'].sort_values('param_value')
        if epochs_df.empty:
            print("  ⚠ 无训练轮数敏感性数据")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x = epochs_df['param_value']
        y1 = epochs_df['annual_return'] * 100
        y2 = epochs_df['test_ic']
        
        ax2 = ax.twinx()
        
        width = (x.max() - x.min()) / len(x) * 0.4 if len(x) > 1 else 3
        bars = ax.bar(x, y1, width=width, alpha=0.7, color=self.colors['success'], label='Annual Return')
        line = ax2.plot(x, y2, 's-', linewidth=2.5, markersize=10, color=self.colors['danger'], label='IC')
        
        ax.set_xlabel('Training Epochs', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12, color=self.colors['success'])
        ax2.set_ylabel('IC', fontsize=12, color=self.colors['danger'])
        ax.tick_params(axis='y', labelcolor=self.colors['success'])
        ax2.tick_params(axis='y', labelcolor=self.colors['danger'])
        
        for xi, yi in zip(x, y1):
            ax.annotate(f'{yi:.1f}%', (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)
        
        ax.set_title('Epochs Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / 'robustness_epochs'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_summary(self, df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
        """图5: 汇总图"""
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 噪声
        noise_df = df[df['exp_type'] == 'noise'].sort_values('param_value')
        if not noise_df.empty:
            ax = axes[0, 0]
            ax.plot(noise_df['param_value'] * 100, noise_df['annual_return'] * 100, 
                   'o-', linewidth=2, markersize=8, color=self.colors['primary'])
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel('Annual Return (%)')
            ax.set_title('Noise Sensitivity', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Top-K
        topk_df = df[df['exp_type'] == 'topk'].sort_values('param_value')
        if not topk_df.empty:
            ax = axes[0, 1]
            ax.plot(topk_df['param_value'], topk_df['annual_return'] * 100, 
                   'o-', linewidth=2, markersize=8, color=self.colors['secondary'])
            ax.set_xlabel('Top-K')
            ax.set_ylabel('Annual Return (%)')
            ax.set_title('Top-K Sensitivity', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 学习率
        lr_df = df[df['exp_type'] == 'lr'].sort_values('param_value')
        if not lr_df.empty:
            ax = axes[1, 0]
            ax.semilogx(lr_df['param_value'], lr_df['annual_return'] * 100, 
                       'o-', linewidth=2, markersize=8, color=self.colors['warning'])
            ax.set_xlabel('Learning Rate (log)')
            ax.set_ylabel('Annual Return (%)')
            ax.set_title('LR Sensitivity', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 训练轮数
        epochs_df = df[df['exp_type'] == 'epochs'].sort_values('param_value')
        if not epochs_df.empty:
            ax = axes[1, 1]
            width = 5 if len(epochs_df) > 1 else 3
            ax.bar(epochs_df['param_value'], epochs_df['annual_return'] * 100, 
                  width=width, alpha=0.7, color=self.colors['success'])
            ax.set_xlabel('Training Epochs')
            ax.set_ylabel('Annual Return (%)')
            ax.set_title('Epochs Sensitivity', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Robustness Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'robustness_summary'
        else:
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.png', '.pdf']:
                save_path = save_path.parent / save_path.stem
        saved = save_figure(save_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")
        if show:
            plt.show()
        plt.close()
    
    def plot_all(self, data_dir: str, show: bool = False):
        """生成所有鲁棒性实验图表"""
        print("\n>>> 生成鲁棒性实验可视化图表...")
        df = self.load_data(data_dir)
        
        self.plot_noise_sensitivity(df, show=show)
        self.plot_topk_sensitivity(df, show=show)
        self.plot_lr_sensitivity(df, show=show)
        self.plot_epochs_sensitivity(df, show=show)
        self.plot_summary(df, show=show)
        
        print(f"\n✓ 鲁棒性实验图表保存至: {self.output_dir}")

# =============================================================================
# 统一实验表格渲染
# =============================================================================

def _fmt_stat(mean, std=None, pct: bool = False, digits: int = 2) -> str:
    if mean is None or pd.isna(mean):
        return "-"
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    mean_str = f"{mean * scale:.{digits}f}{suffix}"
    if std is None or pd.isna(std) or abs(std) < 1e-12:
        return mean_str
    return f"{mean_str} +/- {std * scale:.{digits}f}{suffix}"


def _rows_to_markdown(title: str, headers: List[str], rows) -> str:
    body = [f"## {title}", ""]
    body.append("| " + " | ".join(headers) + " |")
    body.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        body.append("| " + " | ".join(str(cell) for cell in row) + " |")
    body.append("")
    return "\n".join(body)


def _aggregate_variant_payload(payload) -> pd.DataFrame:
    df = pd.DataFrame(payload)
    if df.empty:
        return df
    return df.groupby("variant_name").agg(
        description=("description", "first"),
        market_profile=("market_profile", "first"),
        asset_type=("asset_type", "first"),
        annual_return=("annual_return", "mean"),
        annual_return_std=("annual_return", "std"),
        sharpe_ratio=("sharpe_ratio", "mean"),
        sharpe_ratio_std=("sharpe_ratio", "std"),
        calmar_ratio=("calmar_ratio", "mean"),
        calmar_ratio_std=("calmar_ratio", "std"),
        max_drawdown=("max_drawdown", "mean"),
        max_drawdown_std=("max_drawdown", "std"),
        test_ic=("test_ic", "mean"),
        test_ic_std=("test_ic", "std"),
        rank_ic=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
        kendall_tau=("kendall_tau", "mean"),
        kendall_tau_std=("kendall_tau", "std"),
        crash_ratio=("crash_ratio", "mean"),
        crash_ratio_std=("crash_ratio", "std"),
        guardian_recall=("guardian_recall", "mean"),
        guardian_recall_std=("guardian_recall", "std"),
        guardian_fpr=("guardian_fpr", "mean"),
        guardian_fpr_std=("guardian_fpr", "std"),
        excess_alpha=("excess_alpha", "mean"),
        excess_alpha_std=("excess_alpha", "std"),
        benchmark_annual_return=("benchmark_annual_return", "mean"),
        sigma_target=("sigma_target", "first"),
    ).reset_index()


def render_experiment_table(category: str, experiment_name: str, payload) -> str:
    from experiment_catalog import get_experiment_spec

    title = get_experiment_spec(category, experiment_name)["title"]

    if category == "main" and experiment_name == "market_suite":
        agg = _aggregate_variant_payload(payload)
        if agg.empty:
            return _rows_to_markdown(title, ["Status"], [["No results"]])
        headers = ["Market", "Asset", "ARR(%)", "SR", "MDD", "Rank IC"]
        rows = [
            [
                row["market_profile"],
                row["asset_type"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["max_drawdown"], row["max_drawdown_std"], digits=3),
                _fmt_stat(row["rank_ic"], row["rank_ic_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "main" and experiment_name == "etf_generalization":
        df = pd.DataFrame(payload)
        headers = ["Model", "ARR(%)", "SR", "Rank IC", "Excess alpha"]
        rows = [
            [
                row["model"],
                _fmt_stat(row.get("annual_return"), pct=True),
                _fmt_stat(row.get("sharpe_ratio"), digits=3),
                _fmt_stat(row.get("rank_ic"), digits=3),
                _fmt_stat(row.get("excess_alpha"), pct=True),
            ]
            for _, row in df.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "main" and experiment_name == "cross_geography":
        df = pd.DataFrame(payload)
        headers = ["Universe", "Model", "ARR(%)", "SR", "MDD"]
        rows = [
            [
                row["market_label"],
                row["model"],
                _fmt_stat(row["annual_return"], row.get("annual_return_std"), pct=True),
                _fmt_stat(row["sharpe_ratio"], row.get("sharpe_ratio_std"), digits=3),
                _fmt_stat(row.get("max_drawdown"), row.get("max_drawdown_std"), digits=3),
            ]
            for _, row in df.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "main" and experiment_name == "case_study":
        cases = payload.get("cases", [])
        headers = ["Type", "Stock", "Date", "E_bull", "E_bear", "E_heat", "E_total", "P_crash", "5d Return", "MaxDD"]
        rows = []
        for case in cases:
            rows.append(
                [
                    case.get("case_type", "-"),
                    case.get("ts_code", "-"),
                    case.get("trade_date", "-"),
                    _fmt_stat(case.get("E_bull"), digits=3),
                    _fmt_stat(case.get("E_bear"), digits=3),
                    _fmt_stat(case.get("E_heat"), digits=3),
                    _fmt_stat(case.get("ebm_score"), digits=3),
                    _fmt_stat(case.get("crash_proba"), digits=3),
                    _fmt_stat(case.get("fwd_return"), pct=True),
                    _fmt_stat(case.get("max_dd_10d"), pct=True),
                ]
            )
        return _rows_to_markdown(title, headers, rows or [["No cases found"]])

    if category == "ablation" and experiment_name == "module":
        agg = _aggregate_variant_payload(payload)
        headers = ["Variant", "ARR(%)", "SR", "MDD", "Rank IC"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["max_drawdown"], row["max_drawdown_std"], digits=3),
                _fmt_stat(row["rank_ic"], row["rank_ic_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "ablation" and experiment_name == "feature_grouping":
        agg = _aggregate_variant_payload(payload)
        headers = ["Grouping Strategy", "ARR(%)", "SR", "Rank IC"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["rank_ic"], row["rank_ic_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "ablation" and experiment_name == "aggregation":
        agg = _aggregate_variant_payload(payload)
        headers = ["Aggregation", "ARR(%)", "SR", "MDD", "Kendall tau"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["max_drawdown"], row["max_drawdown_std"], digits=3),
                _fmt_stat(row["kendall_tau"], row["kendall_tau_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "ablation" and experiment_name == "guardian":
        agg = _aggregate_variant_payload(payload)
        headers = ["Guardian", "ARR(%)", "SR", "Crash Recall", "FPR"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["guardian_recall"], row["guardian_recall_std"], pct=True, digits=1),
                _fmt_stat(row["guardian_fpr"], row["guardian_fpr_std"], pct=True, digits=1),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "robustness" and experiment_name in {"noise", "topk", "lr", "epochs"}:
        agg = _aggregate_variant_payload(payload)
        headers = ["Setting", "ARR(%)", "SR", "MDD", "Rank IC"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["max_drawdown"], row["max_drawdown_std"], digits=3),
                _fmt_stat(row["rank_ic"], row["rank_ic_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "robustness" and experiment_name == "crash_label":
        agg = _aggregate_variant_payload(payload)
        headers = ["Crash Definition", "Crash%", "ARR(%)", "SR", "Calmar", "Rank IC"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["crash_ratio"], row["crash_ratio_std"], pct=True, digits=1),
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["calmar_ratio"], row["calmar_ratio_std"], digits=3),
                _fmt_stat(row["rank_ic"], row["rank_ic_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "robustness" and experiment_name == "rolling_window":
        agg = _aggregate_variant_payload(payload)
        headers = ["Window", "ARR(%)", "SR", "Index ARR(%)", "Excess alpha"]
        rows = [
            [
                row["description"],
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["benchmark_annual_return"], pct=True),
                _fmt_stat(row["excess_alpha"], row["excess_alpha_std"], pct=True),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "robustness" and experiment_name == "macro_degradation":
        agg = _aggregate_variant_payload(payload)
        baseline_sr = None
        if "default" in set(agg["variant_name"]):
            baseline_sr = float(agg.loc[agg["variant_name"] == "default", "sharpe_ratio"].iloc[0])
        headers = ["Macro Condition", "ARR(%)", "SR", "Delta SR"]
        rows = []
        for _, row in agg.iterrows():
            delta = None if baseline_sr is None else row["sharpe_ratio"] - baseline_sr
            rows.append(
                [
                    row["description"],
                    _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                    _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                    _fmt_stat(delta, digits=3),
                ]
            )
        return _rows_to_markdown(title, headers, rows)

    if category == "robustness" and experiment_name == "energy_bounds":
        agg = _aggregate_variant_payload(payload)
        headers = ["Strategy", "sigma_target", "ARR(%)", "SR", "MDD", "Calmar"]
        rows = [
            [
                row["description"],
                row["sigma_target"] or "-",
                _fmt_stat(row["annual_return"], row["annual_return_std"], pct=True),
                _fmt_stat(row["sharpe_ratio"], row["sharpe_ratio_std"], digits=3),
                _fmt_stat(row["max_drawdown"], row["max_drawdown_std"], digits=3),
                _fmt_stat(row["calmar_ratio"], row["calmar_ratio_std"], digits=3),
            ]
            for _, row in agg.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    if category == "robustness" and experiment_name == "stress_period":
        df = pd.DataFrame(payload)
        headers = ["Market", "Stress Period", "Market Index(%)", "Strongest Baseline(%)", "PRIME(%)", "PRIME-Baseline"]
        rows = [
            [
                row["market_label"],
                row["stress_label"],
                _fmt_stat(row["market_index_return"], pct=True),
                _fmt_stat(row["baseline_return"], pct=True),
                _fmt_stat(row["prime_return"], pct=True),
                _fmt_stat(row["prime_minus_baseline"], pct=True),
            ]
            for _, row in df.iterrows()
        ]
        return _rows_to_markdown(title, headers, rows)

    return _rows_to_markdown(title, ["Status"], [["No renderer implemented"]])


class CorrelationAnalyzer:
    """轻量统一版相关性分析器。"""

    def __init__(self, output_dir: str = 'outputs/correlation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reset_data()
        self.cmap_diverging = 'RdBu_r'

    def _reset_data(self):
        self.model_predictions = {}
        self.energy_data = {
            'E_bull': [], 'E_bear': [], 'E_heat': [], 'E_total': [],
            'alpha': [], 'beta': [], 'gamma': [], 'returns': [],
            'dates': [], 'ts_codes': [],
        }
        self.market_states = []
        self.feature_groups_data = {'bull': [], 'bear': [], 'friction': [], 'macro': []}
        self.feature_names = {'bull': [], 'bear': [], 'friction': [], 'macro': []}

    def set_feature_names(self, feature_groups: Dict[str, List[str]]):
        for group, names in feature_groups.items():
            if group in self.feature_names:
                self.feature_names[group] = list(names)

    def record_model_prediction(self, model_name: str, predictions: np.ndarray):
        values = np.asarray(predictions).reshape(-1).tolist()
        self.model_predictions.setdefault(model_name, []).extend(values)

    def record_energy_components(
        self,
        E_bull: np.ndarray,
        E_bear: np.ndarray,
        E_heat: np.ndarray,
        E_total: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        returns: Optional[np.ndarray] = None,
        dates: Optional[List[str]] = None,
        ts_codes: Optional[List[str]] = None,
        market_state: Optional[str] = None,
    ):
        n = len(np.asarray(E_bull).reshape(-1))

        def to_list(arr, fill_value=np.nan):
            if arr is None:
                return [fill_value] * n
            return np.asarray(arr).reshape(-1).tolist()

        self.energy_data['E_bull'].extend(to_list(E_bull))
        self.energy_data['E_bear'].extend(to_list(E_bear))
        self.energy_data['E_heat'].extend(to_list(E_heat))
        self.energy_data['E_total'].extend(to_list(E_total))
        self.energy_data['alpha'].extend(to_list(alpha))
        self.energy_data['beta'].extend(to_list(beta))
        self.energy_data['gamma'].extend(to_list(gamma))
        self.energy_data['returns'].extend(to_list(returns))
        self.energy_data['dates'].extend(dates if dates and len(dates) == n else [''] * n)
        self.energy_data['ts_codes'].extend(ts_codes if ts_codes and len(ts_codes) == n else [''] * n)
        self.market_states.extend([market_state or 'unknown'] * n)

    def record_feature_groups(
        self,
        bull_features: np.ndarray,
        bear_features: np.ndarray,
        friction_features: np.ndarray,
        macro_features: np.ndarray,
    ):
        self.feature_groups_data['bull'].append(np.asarray(bull_features))
        self.feature_groups_data['bear'].append(np.asarray(bear_features))
        self.feature_groups_data['friction'].append(np.asarray(friction_features))
        self.feature_groups_data['macro'].append(np.asarray(macro_features))

    def save_data(self, save_dir: Optional[str] = None):
        save_dir = Path(save_dir) if save_dir else self.output_dir / 'data'
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model_predictions:
            pd.DataFrame(self.model_predictions).to_csv(save_dir / 'model_predictions.csv', index=False)

        energy_df = self.get_energy_dataframe()
        if not energy_df.empty:
            energy_df.to_csv(save_dir / 'energy_components.csv', index=False)

        if self.feature_groups_data['bull']:
            np.savez_compressed(
                save_dir / 'feature_groups.npz',
                bull=np.vstack(self.feature_groups_data['bull']),
                bear=np.vstack(self.feature_groups_data['bear']),
                friction=np.vstack(self.feature_groups_data['friction']),
                macro=np.vstack(self.feature_groups_data['macro']),
            )

        with open(save_dir / 'feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False, indent=2)

    def load_data(self, load_dir: str):
        load_dir = Path(load_dir)
        self._reset_data()

        predictions_path = load_dir / 'model_predictions.csv'
        if predictions_path.exists():
            self.model_predictions = pd.read_csv(predictions_path).to_dict('list')

        energy_path = load_dir / 'energy_components.csv'
        if energy_path.exists():
            df = pd.read_csv(energy_path)
            for col in self.energy_data:
                if col in df.columns:
                    self.energy_data[col] = df[col].tolist()
            if 'market_state' in df.columns:
                self.market_states = df['market_state'].tolist()

        feature_path = load_dir / 'feature_groups.npz'
        if feature_path.exists():
            data = np.load(feature_path)
            for key in self.feature_groups_data:
                if key in data:
                    self.feature_groups_data[key] = [data[key]]

        names_path = load_dir / 'feature_names.json'
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
        return self

    def get_energy_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.energy_data)
        if self.market_states:
            df['market_state'] = self.market_states[:len(df)]
        return df

    def _save_plot(self, base_path: Path):
        saved = save_figure(base_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")

    def plot_model_prediction_correlation(self, show: bool = False):
        if not HAS_MATPLOTLIB or not self.model_predictions:
            return None
        df = pd.DataFrame(self.model_predictions)
        if df.empty or df.shape[1] < 1:
            return None
        corr = df.corr(method='spearman')
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap=self.cmap_diverging, center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
        ax.set_title('Model Prediction Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        self._save_plot(self.output_dir / 'correlation_model_predictions')
        if show:
            plt.show()
        plt.close(fig)
        return corr

    def plot_energy_component_correlation(self, show: bool = False):
        if not HAS_MATPLOTLIB or not self.energy_data['E_bull']:
            return None
        energy_df = self.get_energy_dataframe()
        cols = ['E_bull', 'E_bear', 'E_heat', 'E_total', 'alpha', 'beta', 'gamma']
        if 'returns' in energy_df.columns and energy_df['returns'].notna().sum() > 0:
            cols.append('returns')
        df_corr = energy_df[[c for c in cols if c in energy_df.columns]].dropna()
        if len(df_corr) < 10:
            return None
        corr = df_corr.corr(method='spearman')
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap=self.cmap_diverging, center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
        ax.set_title('Energy Component Correlation', fontweight='bold')
        plt.tight_layout()
        self._save_plot(self.output_dir / 'correlation_energy_components')
        if show:
            plt.show()
        plt.close(fig)
        return corr

    def plot_market_state_correlation(self, show: bool = False):
        if not HAS_MATPLOTLIB or not self.market_states:
            return None
        energy_df = self.get_energy_dataframe()
        if energy_df.empty or 'market_state' not in energy_df.columns:
            return None
        grouped = energy_df.groupby('market_state')[['alpha', 'beta', 'gamma', 'E_total']].mean()
        if grouped.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped[['alpha', 'beta', 'gamma']].plot(kind='bar', ax=ax)
        ax.set_title('Average Modulation by Market State', fontweight='bold')
        ax.set_xlabel('Market State')
        ax.set_ylabel('Average coefficient')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        self._save_plot(self.output_dir / 'correlation_market_states')
        if show:
            plt.show()
        plt.close(fig)
        return grouped

    def plot_feature_group_correlation(self, show: bool = False):
        if not HAS_MATPLOTLIB or not any(self.feature_groups_data.values()):
            return None
        aggregated = {}
        for group, values in self.feature_groups_data.items():
            if values:
                block = np.vstack(values)
                aggregated[group] = np.nanmean(block, axis=1)
        if not aggregated:
            return None
        df = pd.DataFrame(aggregated)
        corr = df.corr(method='spearman')
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap=self.cmap_diverging, center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
        ax.set_title('Feature Group Correlation', fontweight='bold')
        plt.tight_layout()
        self._save_plot(self.output_dir / 'correlation_feature_groups')
        if show:
            plt.show()
        plt.close(fig)
        return corr

    def plot_all(self, show: bool = False):
        self.plot_model_prediction_correlation(show=show)
        self.plot_energy_component_correlation(show=show)
        self.plot_market_state_correlation(show=show)
        self.plot_feature_group_correlation(show=show)


class EnergyLandscapeDataRecorder:
    """轻量统一版能量地形数据记录器。"""

    def __init__(self, output_dir: str = 'outputs/landscape'):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._reset_data()

    def _reset_data(self):
        self.stock_energy_data = {
            'E_bull': [], 'E_bear': [], 'E_heat': [], 'E_total': [],
            'alpha': [], 'beta': [], 'gamma': [], 'returns': [],
            'market_state': [], 'ts_code': [], 'date': [],
        }
        self.feature_data = {'features': [], 'E_total': [], 'returns': []}
        self.train_trajectory = []
        self.test_trajectory = []
        self.market_coefficients = {
            'bull': {'alpha': [], 'beta': [], 'gamma': []},
            'neutral': {'alpha': [], 'beta': [], 'gamma': []},
            'bear': {'alpha': [], 'beta': [], 'gamma': []},
            'unknown': {'alpha': [], 'beta': [], 'gamma': []},
        }

    def record_batch_energy(self, E_bull: np.ndarray, E_bear: np.ndarray, E_heat: np.ndarray, E_total: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, returns: Optional[np.ndarray] = None, market_state: str = 'unknown', ts_codes: Optional[List[str]] = None, dates: Optional[List[str]] = None):
        n = len(np.asarray(E_bull).reshape(-1))
        self.stock_energy_data['E_bull'].extend(np.asarray(E_bull).reshape(-1).tolist())
        self.stock_energy_data['E_bear'].extend(np.asarray(E_bear).reshape(-1).tolist())
        self.stock_energy_data['E_heat'].extend(np.asarray(E_heat).reshape(-1).tolist())
        self.stock_energy_data['E_total'].extend(np.asarray(E_total).reshape(-1).tolist())
        self.stock_energy_data['alpha'].extend(np.asarray(alpha).reshape(-1).tolist())
        self.stock_energy_data['beta'].extend(np.asarray(beta).reshape(-1).tolist())
        self.stock_energy_data['gamma'].extend(np.asarray(gamma).reshape(-1).tolist())
        self.stock_energy_data['returns'].extend(np.asarray(returns).reshape(-1).tolist() if returns is not None else [np.nan] * n)
        self.stock_energy_data['market_state'].extend([market_state] * n)
        self.stock_energy_data['ts_code'].extend(list(ts_codes) if ts_codes else [''] * n)
        self.stock_energy_data['date'].extend(list(dates) if dates else [''] * n)

    def record_batch_features(self, features: np.ndarray, E_total: np.ndarray, returns: Optional[np.ndarray] = None):
        features = np.asarray(features)
        if features.size == 0:
            return
        self.feature_data['features'].append(features)
        self.feature_data['E_total'].extend(np.asarray(E_total).reshape(-1).tolist())
        self.feature_data['returns'].extend(np.asarray(returns).reshape(-1).tolist() if returns is not None else [np.nan] * len(E_total))

    def record_train_epoch(self, epoch: int, top_features: np.ndarray, top_energy_mean: float, valid_ic: float = 0.0):
        self.train_trajectory.append({
            'epoch': int(epoch),
            'top_features': np.asarray(top_features).tolist(),
            'energy_mean': float(top_energy_mean),
            'valid_ic': float(valid_ic),
        })

    def record_test_date(self, date: str, selected_features: np.ndarray, selected_energy_mean: float):
        self.test_trajectory.append({
            'date': str(date),
            'selected_features': np.asarray(selected_features).tolist(),
            'energy_mean': float(selected_energy_mean),
        })

    def record_market_coefficients(self, alpha: float, beta: float, gamma: float, market_state: str):
        bucket = self.market_coefficients.setdefault(market_state, {'alpha': [], 'beta': [], 'gamma': []})
        bucket['alpha'].append(float(alpha))
        bucket['beta'].append(float(beta))
        bucket['gamma'].append(float(gamma))

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.stock_energy_data).to_csv(self.data_dir / 'stock_energy.csv', index=False)
        if self.feature_data['features']:
            np.savez_compressed(
                self.data_dir / 'feature_data.npz',
                features=np.vstack(self.feature_data['features']),
                E_total=np.asarray(self.feature_data['E_total']),
                returns=np.asarray(self.feature_data['returns']),
            )
        with open(self.data_dir / 'train_trajectory.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_trajectory, f, ensure_ascii=False, indent=2)
        with open(self.data_dir / 'test_trajectory.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_trajectory, f, ensure_ascii=False, indent=2)
        with open(self.data_dir / 'market_coefficients.json', 'w', encoding='utf-8') as f:
            json.dump(self.market_coefficients, f, ensure_ascii=False, indent=2)


class EnergyLandscapeVisualizer:
    """轻量统一版能量地形可视化。"""

    def __init__(self, output_dir: str = 'outputs/landscape'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stock_df = pd.DataFrame()
        self.feature_matrix = None
        self.feature_energy = None
        self.feature_returns = None
        self.train_trajectory = []
        self.test_trajectory = []
        self.market_coefficients = {}

    def load_data(self, data_dir: str):
        data_dir = Path(data_dir)
        stock_path = data_dir / 'stock_energy.csv'
        if stock_path.exists():
            self.stock_df = pd.read_csv(stock_path)
        feature_path = data_dir / 'feature_data.npz'
        if feature_path.exists():
            data = np.load(feature_path)
            self.feature_matrix = data['features']
            self.feature_energy = data['E_total']
            self.feature_returns = data['returns']
        train_path = data_dir / 'train_trajectory.json'
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_trajectory = json.load(f)
        test_path = data_dir / 'test_trajectory.json'
        if test_path.exists():
            with open(test_path, 'r', encoding='utf-8') as f:
                self.test_trajectory = json.load(f)
        coeff_path = data_dir / 'market_coefficients.json'
        if coeff_path.exists():
            with open(coeff_path, 'r', encoding='utf-8') as f:
                self.market_coefficients = json.load(f)
        return self

    def _save_plot(self, base_path: Path):
        saved = save_figure(base_path, formats=['png', 'pdf'])
        print_saved_paths(saved, "  ✓ 保存")

    def plot_theoretical_landscape(self, show: bool = False):
        if not HAS_MATPLOTLIB or self.stock_df.empty:
            return
        alpha = float(self.stock_df['alpha'].mean()) if 'alpha' in self.stock_df else 1.0
        beta = float(self.stock_df['beta'].mean()) if 'beta' in self.stock_df else 1.0
        gamma = float(self.stock_df['gamma'].mean()) if 'gamma' in self.stock_df else 0.5
        bull = np.linspace(0.0, 1.0, 120)
        bear = np.linspace(0.0, 1.0, 120)
        xx, yy = np.meshgrid(bull, bear)
        zz = beta * yy - alpha * xx + gamma * (xx - yy) ** 2
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xx, yy, zz, levels=24, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Energy')
        ax.set_xlabel('E_bull')
        ax.set_ylabel('E_bear')
        ax.set_title('Theoretical Energy Landscape', fontweight='bold')
        plt.tight_layout()
        self._save_plot(self.output_dir / 'landscape_theoretical')
        if show:
            plt.show()
        plt.close(fig)

    def plot_empirical_trajectory(self, show: bool = False):
        if not HAS_MATPLOTLIB or self.feature_matrix is None or len(self.feature_matrix) == 0:
            return
        if HAS_SKLEARN and self.feature_matrix.shape[1] >= 2:
            scaler = StandardScaler()
            features = scaler.fit_transform(np.nan_to_num(self.feature_matrix, nan=0.0, posinf=0.0, neginf=0.0))
            pca = PCA(n_components=2)
            coords = pca.fit_transform(features)
        else:
            coords = np.asarray(self.feature_matrix)[:, :2]
            if coords.shape[1] < 2:
                coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
            scaler = None
            pca = None

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=np.asarray(self.feature_energy) if self.feature_energy is not None else np.zeros(len(coords)),
            s=18,
            alpha=0.45,
            cmap='viridis_r',
        )
        plt.colorbar(scatter, ax=ax, label='E_total')

        train_points = []
        for traj in self.train_trajectory:
            top_features = np.asarray(traj.get('top_features', []))
            if top_features.size == 0:
                continue
            if pca is not None and scaler is not None:
                point = pca.transform(scaler.transform(np.nan_to_num(top_features, nan=0.0, posinf=0.0, neginf=0.0))).mean(axis=0)
            else:
                point = np.nanmean(top_features[:, :coords.shape[1]], axis=0)
                if len(point) < 2:
                    point = np.pad(point, (0, 2 - len(point)))
            train_points.append(point[:2])
        if train_points:
            train_points = np.asarray(train_points)
            ax.plot(train_points[:, 0], train_points[:, 1], 'o-', color='crimson', linewidth=2, markersize=4, label='Train trajectory')
            ax.legend(loc='best')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Empirical Feature-Space Energy Trajectory', fontweight='bold')
        plt.tight_layout()
        self._save_plot(self.output_dir / 'landscape_empirical')
        if show:
            plt.show()
        plt.close(fig)

    def plot_market_state_comparison(self, show: bool = False):
        if not HAS_MATPLOTLIB or not self.market_coefficients:
            return
        rows = []
        for state, coeffs in self.market_coefficients.items():
            if not coeffs['alpha']:
                continue
            rows.append({
                'market_state': state,
                'alpha': float(np.mean(coeffs['alpha'])),
                'beta': float(np.mean(coeffs['beta'])),
                'gamma': float(np.mean(coeffs['gamma'])),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        df.set_index('market_state')[['alpha', 'beta', 'gamma']].plot(kind='bar', ax=ax)
        ax.set_title('Macro Coefficients by Market State', fontweight='bold')
        ax.set_xlabel('Market State')
        ax.set_ylabel('Coefficient')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        self._save_plot(self.output_dir / 'landscape_market_states')
        if show:
            plt.show()
        plt.close(fig)

    def plot_combined(self, show: bool = False):
        if not HAS_MATPLOTLIB or self.stock_df.empty:
            return
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].scatter(self.stock_df['E_bull'], self.stock_df['E_bear'], c=self.stock_df['E_total'], s=12, alpha=0.5, cmap='viridis_r')
        axes[0].set_xlabel('E_bull')
        axes[0].set_ylabel('E_bear')
        axes[0].set_title('Bull/Bear Plane')
        self.stock_df[['alpha', 'beta', 'gamma']].plot(kind='hist', bins=30, alpha=0.6, ax=axes[1])
        axes[1].set_title('Coefficient Distribution')
        if 'market_state' in self.stock_df:
            self.stock_df.groupby('market_state')['E_total'].mean().plot(kind='bar', ax=axes[2], color='#4c72b0')
        axes[2].set_title('Average E_total by Market State')
        axes[2].set_xlabel('Market State')
        axes[2].grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        self._save_plot(self.output_dir / 'landscape_summary')
        if show:
            plt.show()
        plt.close(fig)

    def plot_all(self, show: bool = False):
        self.plot_theoretical_landscape(show=show)
        self.plot_empirical_trajectory(show=show)
        self.plot_market_state_comparison(show=show)
        self.plot_combined(show=show)


if __name__ == '__main__':
    create_sample_visualization()
