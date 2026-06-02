"""
================================================================================
回测与评估模块 (修复版 v2)
================================================================================
修复内容：
1. 添加能量归因分析 (Energy Attribution)
2. 增加Bull/Bear/Friction贡献度分解
================================================================================
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class EnergyAttribution:
    """能量归因结果"""
    bull_contribution: float = 0.0
    bear_contribution: float = 0.0
    friction_contribution: float = 0.0
    return_from_low_bear: float = 0.0
    return_from_high_bull: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    excess_returns: pd.Series
    portfolio_nav: pd.Series
    benchmark_nav: pd.Series
    metrics: Dict[str, float]
    holdings: pd.DataFrame
    trades: pd.DataFrame
    energy_attribution: Optional[EnergyAttribution] = None


class PerformanceMetrics:
    """绩效指标计算"""
    
    @staticmethod
    def calculate_all(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        metrics = {}
        
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (periods_per_year / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(periods_per_year)
        metrics['downside_vol'] = returns[returns < 0].std() * np.sqrt(periods_per_year) if (returns < 0).any() else 0
        
        excess_return = metrics['annual_return'] - risk_free_rate
        metrics['sharpe_ratio'] = excess_return / metrics['volatility'] if metrics['volatility'] > 0 else 0
        metrics['sortino_ratio'] = excess_return / metrics['downside_vol'] if metrics['downside_vol'] > 0 else 0
        
        nav = (1 + returns).cumprod()
        drawdown = (nav - nav.cummax()) / nav.cummax()
        metrics['max_drawdown'] = drawdown.min()
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        metrics['win_rate'] = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        metrics['profit_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0
        
        if benchmark_returns is not None:
            excess = returns - benchmark_returns
            metrics['excess_return'] = (1 + excess).prod() - 1
            metrics['excess_annual'] = (1 + metrics['excess_return']) ** (periods_per_year / len(excess)) - 1
            metrics['tracking_error'] = excess.std() * np.sqrt(periods_per_year)
            metrics['info_ratio'] = metrics['excess_annual'] / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
        
        return metrics


class EnergyAttributionAnalyzer:
    """能量归因分析器"""
    
    def analyze(
        self,
        holdings_df: pd.DataFrame,
        energy_components: Dict[str, str] = None
    ) -> EnergyAttribution:
        if holdings_df.empty:
            return EnergyAttribution()
        
        attr = EnergyAttribution()
        
        # 自动检测日期列名（兼容'date'和'trade_date'）
        date_col = 'date' if 'date' in holdings_df.columns else 'trade_date'
        n_days = len(holdings_df[date_col].unique()) if date_col in holdings_df.columns else 1
        
        if 'E_bull' in holdings_df.columns and 'weight' in holdings_df.columns:
            attr.bull_contribution = (holdings_df['weight'] * holdings_df['E_bull']).sum() / n_days
        
        if 'E_bear' in holdings_df.columns and 'weight' in holdings_df.columns:
            attr.bear_contribution = (holdings_df['weight'] * holdings_df['E_bear']).sum() / n_days
        
        if 'E_friction' in holdings_df.columns and 'weight' in holdings_df.columns:
            attr.friction_contribution = (holdings_df['weight'] * holdings_df['E_friction']).sum() / n_days
        
        if 'E_bear' in holdings_df.columns and 'return' in holdings_df.columns:
            median_bear = holdings_df['E_bear'].median()
            low_bear_mask = holdings_df['E_bear'] <= median_bear
            if low_bear_mask.any():
                attr.return_from_low_bear = (holdings_df.loc[low_bear_mask, 'weight'] * holdings_df.loc[low_bear_mask, 'return']).sum()
        
        if 'E_bull' in holdings_df.columns and 'return' in holdings_df.columns:
            median_bull = holdings_df['E_bull'].median()
            high_bull_mask = holdings_df['E_bull'] >= median_bull
            if high_bull_mask.any():
                attr.return_from_high_bull = (holdings_df.loc[high_bull_mask, 'weight'] * holdings_df.loc[high_bull_mask, 'return']).sum()
        
        return attr


class Backtester:
    """策略回测器"""
    
    def __init__(self, config):
        self.config = config
        self.backtest_config = config.backtest

        active_costs = self.backtest_config.get_costs(getattr(config, 'mode', 'stock'))
        self.commission = active_costs['commission']
        self.slippage = active_costs['slippage']
        self.stamp_tax = active_costs['stamp_tax']
        self.single_limit = active_costs['single_limit']
        self.sector_limit = active_costs['sector_limit']
        self.stop_loss_threshold = active_costs['stop_loss']
        self.t_plus_1 = self.backtest_config.t_plus_1
        
        self.attribution_analyzer = EnergyAttributionAnalyzer()
    
    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = 'ebm_score',
        return_col: str = 'fwd_return',
        date_col: str = 'trade_date',
        code_col: str = 'ts_code',
        benchmark_col: Optional[str] = 'csi500_ret',
        top_k: int = 50,
        energy_components: Optional[Dict[str, str]] = None,
        position_ratios: Optional[Dict] = None  # 【新增】每日仓位比例 {date: ratio}
    ) -> BacktestResult:
        print("\n>>> 运行策略回测")
        print(f"    持仓: {top_k}, 成本: 佣金={self.commission:.3%}, 滑点={self.slippage:.3%}")
        
        df = df.sort_values([date_col, code_col]).copy()
        dates = df[date_col].unique()
        
        rebalance_freq = self.backtest_config.rebalance_freq
        if rebalance_freq == 'weekly':
            rebalance_dates = dates[::5]
        elif rebalance_freq == 'monthly':
            rebalance_dates = dates[::21]
        else:
            rebalance_dates = dates
        
        holdings = {}
        holdings_history = []
        trades_history = []
        portfolio_returns = []
        benchmark_returns = []
        
        # 【新增】止损参数
        stop_loss_threshold = self.stop_loss_threshold
        enable_stop_loss = getattr(self.backtest_config, 'enable_stop_loss', True)
        
        # 【新增】跟踪持仓成本和累计收益
        holding_costs = {}  # {code: entry_price}
        cumulative_returns = {}  # {code: cumulative_return}
        stop_loss_triggered = set()  # 已触发止损的股票
        stop_loss_count = 0
        
        for date in dates:
            df_day = df[df[date_col] == date]
            
            # 【新增】处理止损：在换仓前检查止损
            if enable_stop_loss and holdings:
                stocks_to_remove = []
                for code in list(holdings.keys()):
                    if code in stop_loss_triggered:
                        stocks_to_remove.append(code)
                        continue
                    
                    stock_data = df_day[df_day[code_col] == code]
                    if not stock_data.empty and return_col in stock_data.columns:
                        daily_ret = stock_data[return_col].iloc[0]
                        
                        # 更新累计收益
                        if code not in cumulative_returns:
                            cumulative_returns[code] = 0
                        cumulative_returns[code] = (1 + cumulative_returns[code]) * (1 + daily_ret) - 1
                        
                        # 检查止损条件
                        if cumulative_returns[code] <= stop_loss_threshold:
                            stocks_to_remove.append(code)
                            stop_loss_triggered.add(code)
                            stop_loss_count += 1
                
                # 移除触发止损的股票（次日生效，模拟T+1）
                for code in stocks_to_remove:
                    if code in holdings:
                        del holdings[code]
                        if code in cumulative_returns:
                            del cumulative_returns[code]
                
                # 重新归一化权重
                if holdings:
                    total_weight = sum(holdings.values())
                    if total_weight > 0:
                        holdings = {k: v / total_weight for k, v in holdings.items()}
            
            if date in rebalance_dates:
                new_holdings = self._generate_portfolio(df_day, signal_col, code_col, top_k)
                
                # 【新增】排除已触发止损的股票
                if enable_stop_loss:
                    new_holdings = {k: v for k, v in new_holdings.items() if k not in stop_loss_triggered}
                    if new_holdings:
                        total = sum(new_holdings.values())
                        new_holdings = {k: v / total for k, v in new_holdings.items()}
                
                # 【新增】应用仓位比例（宏观择时）
                current_position_ratio = 1.0
                if position_ratios is not None and date in position_ratios:
                    current_position_ratio = position_ratios[date]
                    # 缩放权重（剩余部分假设持有现金）
                    new_holdings = {k: v * current_position_ratio for k, v in new_holdings.items()}
                
                trades = self._calculate_trades(holdings, new_holdings, date)
                trades_history.extend(trades)
                
                # 【新增】重置新持仓的累计收益
                for code in new_holdings:
                    if code not in holdings:
                        cumulative_returns[code] = 0
                
                holdings = new_holdings
            
            # 【修复】记录仓位比例到holdings_history
            current_position_ratio = 1.0
            if position_ratios is not None and date in position_ratios:
                current_position_ratio = position_ratios[date]
            
            for code, weight in holdings.items():
                record = {'date': date, 'code': code, 'weight': weight, 'position_ratio': current_position_ratio}
                
                stock_data = df_day[df_day[code_col] == code]
                if not stock_data.empty:
                    if signal_col in stock_data.columns:
                        record['ebm_score'] = stock_data[signal_col].iloc[0]
                    if return_col in stock_data.columns:
                        record['return'] = stock_data[return_col].iloc[0]
                    
                    if energy_components:
                        for comp_name, col_name in energy_components.items():
                            if col_name in stock_data.columns:
                                record[comp_name] = stock_data[col_name].iloc[0]
                
                holdings_history.append(record)
            
            port_ret = self._calculate_portfolio_return(df_day, holdings, return_col, code_col)
            
            if date in rebalance_dates and trades_history:
                turnover = sum(abs(t['amount']) for t in trades_history if t['date'] == date)
                cost = turnover * (self.commission * 2 + self.slippage + self.stamp_tax / 2)
                port_ret -= cost
            
            portfolio_returns.append({'date': date, 'return': port_ret})
            
            if benchmark_col and benchmark_col in df_day.columns:
                bench_ret = df_day[benchmark_col].iloc[0]
                if abs(bench_ret) > 1:
                    bench_ret = bench_ret / 100
                benchmark_returns.append({'date': date, 'return': bench_ret})
        
        port_ret_series = pd.DataFrame(portfolio_returns).set_index('date')['return']
        bench_ret_series = pd.DataFrame(benchmark_returns).set_index('date')['return'] if benchmark_returns else pd.Series(0, index=port_ret_series.index)
        excess_ret_series = port_ret_series - bench_ret_series
        
        port_nav = (1 + port_ret_series).cumprod()
        bench_nav = (1 + bench_ret_series).cumprod()
        
        metrics = PerformanceMetrics.calculate_all(port_ret_series, bench_ret_series)
        
        # 【新增】持仓统计
        holdings_df = pd.DataFrame(holdings_history)
        if not holdings_df.empty:
            daily_holdings_count = holdings_df.groupby('date')['code'].nunique()
            metrics['avg_holdings'] = daily_holdings_count.mean()
            metrics['max_holdings'] = daily_holdings_count.max()
            metrics['min_holdings'] = daily_holdings_count.min()
            metrics['std_holdings'] = daily_holdings_count.std()
            
            # 计算平均持仓权重
            avg_weight = holdings_df['weight'].mean()
            max_weight = holdings_df['weight'].max()
            metrics['avg_weight'] = avg_weight
            metrics['max_weight'] = max_weight
            
            # 【新增】计算仓位比例统计（持仓占总资金的比例）
            # position_ratio列在trainer.py的select_stocks中添加
            if 'position_ratio' in holdings_df.columns:
                # 按日期计算每日仓位比例（取第一个值，因为同一天的position_ratio相同）
                daily_position = holdings_df.groupby('date')['position_ratio'].first()
                metrics['avg_position_ratio'] = daily_position.mean()
                metrics['max_position_ratio'] = daily_position.max()
                metrics['min_position_ratio'] = daily_position.min()
                metrics['std_position_ratio'] = daily_position.std()
            else:
                # 如果没有position_ratio列，使用每日总权重作为仓位比例
                daily_total_weight = holdings_df.groupby('date')['weight'].sum()
                metrics['avg_position_ratio'] = daily_total_weight.mean()
                metrics['max_position_ratio'] = daily_total_weight.max()
                metrics['min_position_ratio'] = daily_total_weight.min()
                metrics['std_position_ratio'] = daily_total_weight.std()
        
        # 【新增】止损统计
        if enable_stop_loss:
            metrics['stop_loss_count'] = stop_loss_count
            if stop_loss_count > 0:
                print(f"    止损触发: {stop_loss_count} 次")
        
        ic_metrics = self._calculate_daily_ic(df, signal_col, return_col, date_col)
        metrics.update(ic_metrics)
        
        holdings_df = pd.DataFrame(holdings_history)
        energy_attribution = self.attribution_analyzer.analyze(holdings_df) if not holdings_df.empty else None
        
        result = BacktestResult(
            portfolio_returns=port_ret_series,
            benchmark_returns=bench_ret_series,
            excess_returns=excess_ret_series,
            portfolio_nav=port_nav,
            benchmark_nav=bench_nav,
            metrics=metrics,
            holdings=holdings_df,
            trades=pd.DataFrame(trades_history) if trades_history else pd.DataFrame(),
            energy_attribution=energy_attribution
        )
        
        self._print_summary(result)
        return result
    
    def _generate_portfolio(self, df, signal_col, code_col, top_k):
        df_sorted = df.nlargest(top_k, signal_col)
        weight = 1.0 / len(df_sorted)
        
        weight = min(weight, self.single_limit)
        
        holdings = {row[code_col]: weight for _, row in df_sorted.iterrows()}
        
        total = sum(holdings.values())
        if total > 0:
            holdings = {k: v / total for k, v in holdings.items()}
        
        return holdings
    
    def _calculate_trades(self, old_holdings, new_holdings, date):
        trades = []
        all_codes = set(old_holdings.keys()) | set(new_holdings.keys())
        
        for code in all_codes:
            old_w = old_holdings.get(code, 0)
            new_w = new_holdings.get(code, 0)
            if old_w != new_w:
                trades.append({'date': date, 'code': code, 'old_weight': old_w, 'new_weight': new_w, 'amount': new_w - old_w})
        
        return trades
    
    def _calculate_portfolio_return(self, df, holdings, return_col, code_col):
        if not holdings:
            return 0.0
        
        total_return = 0.0
        for code, weight in holdings.items():
            stock_data = df[df[code_col] == code]
            if not stock_data.empty and return_col in stock_data.columns:
                ret = stock_data[return_col].iloc[0]
                if pd.notna(ret):
                    total_return += weight * ret
        
        return total_return
    
    def _calculate_daily_ic(self, df, signal_col, return_col, date_col):
        ic_list, rank_ic_list = [], []
        
        for date, group in df.groupby(date_col):
            if len(group) < 10:
                continue
            
            pred = group[signal_col]
            ret = group[return_col]
            
            if pred.std() > 0 and ret.std() > 0:
                ic_list.append(pred.corr(ret))
                rank_ic_list.append(pred.rank().corr(ret.rank()))
        
        if ic_list:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            return {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': ic_mean / ic_std if ic_std > 0 else 0,
                'rank_ic_mean': np.mean(rank_ic_list)
            }
        
        return {'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'rank_ic_mean': 0}
    
    def _print_summary(self, result: BacktestResult):
        m = result.metrics
        
        print("\n" + "-" * 60)
        print("    回测结果摘要")
        print("-" * 60)
        print(f"    总收益率:     {m['total_return']:>10.2%}")
        print(f"    年化收益率:   {m['annual_return']:>10.2%}")
        print(f"    年化波动率:   {m['volatility']:>10.2%}")
        print(f"    夏普比率:     {m['sharpe_ratio']:>10.2f}")
        print(f"    索提诺比率:   {m['sortino_ratio']:>10.2f}")
        print(f"    最大回撤:     {m['max_drawdown']:>10.2%}")
        print(f"    Calmar比率:   {m['calmar_ratio']:>10.2f}")
        print(f"    胜率:         {m['win_rate']:>10.2%}")
        
        if 'excess_annual' in m:
            print(f"\n    超额年化收益: {m['excess_annual']:>10.2%}")
            print(f"    信息比率:     {m['info_ratio']:>10.2f}")
        
        if 'ic_mean' in m:
            print(f"\n    IC均值:       {m['ic_mean']:>10.4f}")
            print(f"    ICIR:         {m['icir']:>10.4f}")
            print(f"    Rank IC均值:  {m['rank_ic_mean']:>10.4f}")
        
        # 【新增】持仓统计输出
        if 'avg_holdings' in m:
            print("\n" + "-" * 60)
            print("    持仓统计")
            print("-" * 60)
            print(f"    平均持仓数:   {m['avg_holdings']:>10.1f}")
            print(f"    最大持仓数:   {m['max_holdings']:>10.0f}")
            print(f"    最小持仓数:   {m['min_holdings']:>10.0f}")
            print(f"    持仓数标准差: {m['std_holdings']:>10.2f}")
            print(f"    平均单股权重: {m['avg_weight']:>10.2%}")
            print(f"    最大单股权重: {m['max_weight']:>10.2%}")
        
        # 【新增】仓位比例统计（占总资金比例）
        if 'avg_position_ratio' in m:
            print("\n" + "-" * 60)
            print("    仓位比例（占总资金）")
            print("-" * 60)
            print(f"    平均仓位:     {m['avg_position_ratio']:>10.2%}")
            print(f"    最高仓位:     {m['max_position_ratio']:>10.2%}")
            print(f"    最低仓位:     {m['min_position_ratio']:>10.2%}")
            print(f"    仓位标准差:   {m['std_position_ratio']:>10.2%}")
        
        if result.energy_attribution:
            attr = result.energy_attribution
            print("\n" + "-" * 60)
            print("    能量归因分析")
            print("-" * 60)
            print(f"    Bull贡献:     {attr.bull_contribution:>10.4f}")
            print(f"    Bear贡献:     {attr.bear_contribution:>10.4f}")
            print(f"    Friction贡献: {attr.friction_contribution:>10.4f}")


class BacktestVisualizer:
    """回测可视化"""
    
    @staticmethod
    def plot_nav(
        result: BacktestResult,
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        axes[0].plot(result.portfolio_nav.index, result.portfolio_nav.values, label='Strategy')
        axes[0].plot(result.benchmark_nav.index, result.benchmark_nav.values, label='Benchmark', alpha=0.7)
        axes[0].set_ylabel('NAV')
        axes[0].legend()
        axes[0].set_title('Portfolio Performance')
        axes[0].grid(True, alpha=0.3)
        
        excess_nav = (1 + result.excess_returns).cumprod()
        axes[1].fill_between(excess_nav.index, 1, excess_nav.values, where=excess_nav.values >= 1, alpha=0.3, color='green')
        axes[1].fill_between(excess_nav.index, 1, excess_nav.values, where=excess_nav.values < 1, alpha=0.3, color='red')
        axes[1].plot(excess_nav.index, excess_nav.values, 'k-', linewidth=1)
        axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Excess NAV')
        axes[1].set_title('Excess Return')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close(fig)


def _to_json_safe_dict(data: Dict[str, float]) -> Dict[str, float]:
    """将 numpy / pandas 标量安全转换为 JSON 可序列化字典。"""
    safe = {}
    for key, value in data.items():
        if hasattr(value, "item"):
            safe[key] = value.item()
        elif isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def save_backtest_result(
    result: BacktestResult,
    output_dir: Path,
    save_nav_figure: bool = True,
) -> Dict[str, Path]:
    """将回测结果完整落盘，供主实验和 baseline 共用。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = dict(result.metrics)
    if result.energy_attribution is not None:
        metrics.update(
            {
                "bull_contribution": result.energy_attribution.bull_contribution,
                "bear_contribution": result.energy_attribution.bear_contribution,
                "friction_contribution": result.energy_attribution.friction_contribution,
                "return_from_low_bear": result.energy_attribution.return_from_low_bear,
                "return_from_high_bull": result.energy_attribution.return_from_high_bull,
            }
        )

    metrics_path = output_dir / "metrics.json"
    holdings_path = output_dir / "holdings.csv"
    trades_path = output_dir / "trades.csv"
    nav_path = output_dir / "nav.csv"
    nav_fig_path = output_dir / "nav.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe_dict(metrics), f, indent=2, ensure_ascii=False)

    result.holdings.to_csv(holdings_path, index=False)
    result.trades.to_csv(trades_path, index=False)

    nav_df = pd.DataFrame(
        {
            "date": result.portfolio_nav.index,
            "portfolio_return": result.portfolio_returns.values,
            "benchmark_return": result.benchmark_returns.reindex(result.portfolio_nav.index).values,
            "excess_return": result.excess_returns.reindex(result.portfolio_nav.index).values,
            "portfolio_nav": result.portfolio_nav.values,
            "benchmark_nav": result.benchmark_nav.reindex(result.portfolio_nav.index).values,
        }
    )
    nav_df.to_csv(nav_path, index=False)

    if save_nav_figure:
        BacktestVisualizer.plot_nav(result, save_path=nav_fig_path, show=False)

    return {
        "metrics": metrics_path,
        "holdings": holdings_path,
        "trades": trades_path,
        "nav": nav_path,
        "nav_figure": nav_fig_path,
    }
