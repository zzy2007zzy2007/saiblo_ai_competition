"""Backend 兼容适配器 - 提供 Ant-Game SDK 的统一 API 接口

主要功能：
1. 检测并适配新旧 API 差异
2. 提供统一的 runtime 创建接口
3. 处理 movement_policy 等新增参数
4. 为旧版 resolve_turn 方法提供兼容支持
"""

from typing import Optional, Any, Dict, List, Tuple


from SDK.backend.engine import DEFAULT_MOVEMENT_POLICY


class MatchRuntimeWrapper:
    """MatchRuntime 包装类 - 提供旧版 API 兼容支持

    主要功能：
    1. 为 resolve_turn 方法提供兼容实现（使用 apply_self_operations、apply_opponent_operations、finish_round）
    2. 代理所有其他方法到原始对象
    """

    def __init__(self, original_runtime):
        self._original = original_runtime

    def __getattr__(self, name):
        # 代理所有未明确重写的方法和属性到原始对象
        return getattr(self._original, name)

    def resolve_turn(self, player_0_operations, player_1_operations):
        """兼容旧版 API 的 resolve_turn 方法

        使用新的 API 来模拟旧的 resolve_turn 行为：
        1. 应用玩家0的操作
        2. 应用玩家1的操作
        3. 完成回合

        Args:
            player_0_operations: 玩家0的操作列表
            player_1_operations: 玩家1的操作列表

        Returns:
            TurnResolution: 回合结果
        """
        # 转换输入以确保它们是列表格式
        ops_0 = player_0_operations if isinstance(player_0_operations, list) else [player_0_operations] if player_0_operations else []
        ops_1 = player_1_operations if isinstance(player_1_operations, list) else [player_1_operations] if player_1_operations else []
        
        # 应用操作
        if self._original.player == 0:
            if ops_0:
                self._original.apply_self_operations(ops_0)
            if ops_1:
                self._original.apply_opponent_operations(ops_1)
        else:
            if ops_1:
                self._original.apply_self_operations(ops_1)
            if ops_0:
                self._original.apply_opponent_operations(ops_0)
        
        self._original.state.advance_round()

        class SimulatedTurnResolution:
            def __init__(self, state):
                self.state = state
                self.terminal = state.terminal
                self.round_index = state.round_index
                if hasattr(state, 'players'):
                    self.players = state.players
        
        return SimulatedTurnResolution(self._original.state)


class BackendAdapter:
    """Backend 兼容适配器

    提供统一的接口来创建游戏运行时，处理新旧 API 的差异。

    使用示例：
        adapter = BackendAdapter()
        runtime = adapter.create_runtime(player=0, seed=42)

        # 或者直接使用静态方法
        runtime = BackendAdapter.create_runtime(player=0, seed=42)
    """

    @staticmethod
    def create_runtime(
        player,
        seed=0,
        prefer_native=False,
        movement_policy=None,
        cold_handle_rule_illegal=False,
    ):
        """创建游戏运行时

        Args:
            player: 玩家 ID (0 或 1)
            seed: 随机种子
            prefer_native: 是否优先使用 Native 后端
            movement_policy: 移动策略（Ant-Game 新增参数）
            cold_handle_rule_illegal: 非法操作处理方式（Ant-Game 新增参数）

        Returns:
            MatchRuntimeWrapper: 包装的游戏运行时实例（提供旧版 API 兼容）

        Raises:
            RuntimeError: 当 SDK 加载失败时
        """
        try:
            from SDK.backend.core import load_backend, PythonBackend, NativeBackend
            from SDK.backend.runtime import MatchRuntime
            from SDK.backend.state import BackendState, PythonBackendState
        except ImportError as e:
            raise RuntimeError(
                "Failed to import SDK. Please ensure Ant-Game SDK is installed. Error: " + str(e)
            )

        backend = load_backend(prefer_native=prefer_native)

        try:
            runtime = MatchRuntime.create(
                player=player,
                seed=seed,
                prefer_native=prefer_native,
                backend=backend,
                movement_policy=movement_policy or DEFAULT_MOVEMENT_POLICY,
                cold_handle_rule_illegal=cold_handle_rule_illegal,
            )
        except TypeError:
            runtime = MatchRuntime.create(
                player=player,
                seed=seed,
                prefer_native=prefer_native,
                backend=backend,
            )

        # 用包装类包装原始 runtime，提供 resolve_turn 兼容支持
        return MatchRuntimeWrapper(runtime)

    @staticmethod
    def supports_native_backend() -> bool:
        """检测是否支持 Native 后端

        Returns:
            bool: 是否支持 Native 后端
        """
        try:
            from SDK import native_antwar  # type: ignore
            return True
        except ImportError:
            return False

    @staticmethod
    def get_backend_info() -> Dict[str, Any]:
        """获取 Backend 信息

        Returns:
            Dict: 包含 backend 支持信息的字典
        """
        info = {
            "python_backend": True,
            "native_backend": BackendAdapter.supports_native_backend(),
        }

        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            info["cuda_available"] = False

        return info


__all__ = ['BackendAdapter', 'MatchRuntimeWrapper']
