"""
baseline 子系统统一导出入口。

这里不放具体实现，只负责把注册表、运行器和配置对象集中导出，
方便主入口与实验框架按统一方式调用 baseline。
"""

from .registry import (
    BaselineSpec,
    available_baselines,
    baseline_registry_lines,
    get_baseline_spec,
)
from .runner import BaselineRunner, BaselineRunConfig

__all__ = [
    "BaselineSpec",
    "BaselineRunConfig",
    "BaselineRunner",
    "available_baselines",
    "baseline_registry_lines",
    "get_baseline_spec",
]
