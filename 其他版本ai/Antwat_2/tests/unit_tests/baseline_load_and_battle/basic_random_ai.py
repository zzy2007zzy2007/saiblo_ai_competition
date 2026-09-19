#!/usr/bin/env python3
"""BasicRandomAI - 随机选择合法动作的AI，使用项目SDK"""
import os
import sys
import random
from datetime import datetime


class BasicRandomAI:
    """随机AI：每次从合法动作中随机选择一个执行"""

    def __init__(self):
        self.original_sys_path = sys.path.copy()
        self.sdk_path = '/root/autodl-tmp/AntWar/ppo_v1/Ant-Game/SDK'
        self._setup_sdk()
        self._load_required_modules()

    def _setup_sdk(self):
        """设置项目SDK路径"""
        if os.path.exists(self.sdk_path):
            sys.path.insert(0, self.sdk_path)

    def _load_required_modules(self):
        """加载必要的模块"""
        try:
            from SDK.backend.state import PythonBackendState
            from SDK.backend.core import load_backend
            from SDK.backend.model import Operation as BackendOperation
            from SDK.utils.constants import OperationType as BackendOperationType
            self.PythonBackendState = PythonBackendState
            self.load_backend = load_backend
            self.BackendOperation = BackendOperation
            self.OperationType = BackendOperationType
        except Exception as e:
            print(f"警告: BasicRandomAI 加载模块失败: {e}")

    def _get_valid_actions(self, state, player):
        """获取当前玩家的所有合法动作"""
        try:
            from SDK.utils.actions import ActionBundle
            catalog_path = os.path.join(self.sdk_path, 'utils', 'actions.py')
            if os.path.exists(catalog_path):
                sys.path.insert(0, self.sdk_path)
                from SDK.utils.actions import ActionCatalog
                catalog = ActionCatalog(max_actions=32)
                bundles = catalog.build(state, player)
                return bundles
        except Exception as e:
            pass
        return []

    def choose_operations(self, state, player):
        """随机选择一个合法动作"""
        try:
            bundles = self._get_valid_actions(state, player)
            if bundles and len(bundles) > 0:
                selected = random.choice(bundles)
                return list(selected.operations)
        except Exception as e:
            print(f"BasicRandomAI choose_operations 错误: {e}")
        return []

    def __del__(self):
        """清理：恢复原始sys.path"""
        if hasattr(self, 'original_sys_path'):
            sys.path = self.original_sys_path