"""兼容性适配层 - 提供 Ant-Game SDK 的统一接口

本模块提供以下功能：
1. Backend 适配器 - 兼容新旧 API
2. 弃用警告机制 - 提示旧代码迁移

使用方式：
    from ppo_antwar.compat import BackendAdapter
"""

from .adapters import BackendAdapter, MatchRuntimeWrapper

__all__ = [
    "BackendAdapter",
    "MatchRuntimeWrapper",
]
