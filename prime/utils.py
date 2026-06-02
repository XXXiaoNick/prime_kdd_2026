"""
================================================================================
工具函数模块
================================================================================
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    设置随机种子

    Args:
        seed: 随机种子值
        deterministic: 是否启用完全确定性模式（会降低GPU性能）。
                       仅在需要精确复现结果时设为True。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 启用cuDNN benchmark自动调优，固定输入尺寸下可获得10-30%加速
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """获取计算设备"""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


class TeeLogger:
    """
    同时输出到控制台和文件的Logger
    
    捕获所有print和stdout输出，同时写入文件
    """
    
    def __init__(self, log_file: Path, mode: str = 'w'):
        self.terminal = sys.stdout
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, mode, encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # 立即写入，避免丢失
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


class StderrTeeLogger:
    """捕获stderr的Logger"""
    
    def __init__(self, log_file: Path, mode: str = 'a'):
        self.terminal = sys.stderr
        self.log_file = Path(log_file)
        self.file = open(self.log_file, mode, encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


class LogManager:
    """
    日志管理器
    
    用法：
        log_manager = LogManager(args)
        log_manager.start()
        # ... 运行代码 ...
        log_manager.stop()
    
    或者使用上下文管理器：
        with LogManager(args) as log:
            # ... 运行代码 ...
    """
    
    def __init__(
        self,
        mode: str,
        log_dir: str = 'logs',
        extra_params: Optional[Dict] = None,
        experiment_name: Optional[str] = None
    ):
        self.mode = mode
        self.log_dir = Path(log_dir)
        self.extra_params = extra_params or {}
        self.experiment_name = experiment_name
        
        self.log_file = None
        self.stdout_logger = None
        self.stderr_logger = None
        self.original_stdout = None
        self.original_stderr = None
        
    def _generate_log_filename(self) -> str:
        """
        生成日志文件名
        
        格式: {mode}_{experiment}_{关键参数}_{datetime}.log
        示例: train_macro_game_ebm_v2_s1-10_s2-50_s3-20_bs512_seed42_20241219_143052.log
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 构建参数字符串
        param_parts = [self.mode]
        
        # 添加实验名称（简化）
        if self.experiment_name:
            # 截取实验名称，避免太长
            exp_name = self.experiment_name.replace('_', '-')[:20]
            param_parts.append(exp_name)
        
        # 添加训练相关参数（按重要性排序）
        priority_params = ['s1', 's2', 's3', 'bs', 'lr', 'topk', 'seed']
        
        for key in priority_params:
            if key in self.extra_params and self.extra_params[key] is not None:
                value = self.extra_params[key]
                if isinstance(value, float):
                    # 科学计数法简化
                    if value < 0.01:
                        param_parts.append(f"{key}{value:.0e}".replace('-0', '-'))
                    else:
                        param_parts.append(f"{key}{value}")
                else:
                    param_parts.append(f"{key}{value}")
        
        # 添加其他布尔标记
        for key, value in self.extra_params.items():
            if key not in priority_params and value is not None:
                if isinstance(value, bool) and value:
                    param_parts.append(key.replace('_', ''))
                elif isinstance(value, str) and len(value) < 15:
                    param_parts.append(f"{key}-{value}")
        
        param_parts.append(timestamp)
        
        filename = '_'.join(param_parts) + '.log'
        # 清理文件名中的非法字符
        filename = filename.replace('/', '-').replace('\\', '-').replace(':', '-').replace(' ', '')
        
        return filename
    
    def start(self) -> Path:
        """开始日志记录"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self._generate_log_filename()
        self.log_file = self.log_dir / filename
        
        # 保存原始stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 替换为Tee Logger
        self.stdout_logger = TeeLogger(self.log_file, mode='w')
        self.stderr_logger = StderrTeeLogger(self.log_file, mode='a')
        
        sys.stdout = self.stdout_logger
        sys.stderr = self.stderr_logger
        
        # 打印日志头
        self._print_header()
        
        return self.log_file
    
    def stop(self):
        """停止日志记录"""
        if self.stdout_logger:
            self._print_footer()
            
            # 恢复原始stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # 关闭文件
            self.stdout_logger.close()
            self.stderr_logger.close()
            
            print(f"\n日志已保存至: {self.log_file}")
    
    def _print_header(self):
        """打印日志头部信息"""
        print("=" * 80)
        print(f"  宏观感知物理博弈能量模型 (Macro-Aware Game EBM) v4")
        print("=" * 80)
        print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  运行模式: {self.mode}")
        print(f"  日志文件: {self.log_file}")
        
        if self.extra_params:
            print(f"  运行参数:")
            for key, value in self.extra_params.items():
                if value is not None:
                    print(f"    - {key}: {value}")
        
        print(f"  Python版本: {sys.version.split()[0]}")
        
        try:
            import torch
            print(f"  PyTorch版本: {torch.__version__}")
            print(f"  CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print(f"  PyTorch: 未安装")
        
        print("=" * 80)
        print()
    
    def _print_footer(self):
        """打印日志尾部信息"""
        print()
        print("=" * 80)
        print(f"  运行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"\n错误: {exc_type.__name__}: {exc_val}")
        self.stop()
        return False  # 不抑制异常


def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """配置标准logging模块（可选使用）"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_name}_{timestamp}.log'
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算IC"""
    pred = np.array(predictions).flatten()
    tgt = np.array(targets).flatten()
    
    mask = ~(np.isnan(pred) | np.isnan(tgt))
    pred, tgt = pred[mask], tgt[mask]
    
    if len(pred) < 2:
        return 0.0
    
    return np.corrcoef(pred, tgt)[0, 1]


def calculate_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算Rank IC"""
    pred = pd.Series(predictions).rank()
    tgt = pd.Series(targets).rank()
    return calculate_ic(pred, tgt)


def mad_clip(data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """MAD异常值处理"""
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    
    if mad > 0:
        lower = median - threshold * mad
        upper = median + threshold * mad
        return np.clip(data, lower, upper)
    return data


class Timer:
    """计时器"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.name:
            print(f'{self.name}: {elapsed:.2f}s')


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count