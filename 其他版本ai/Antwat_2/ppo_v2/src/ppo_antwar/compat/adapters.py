from typing import Any, Dict, Optional
import inspect
import importlib.util

from SDK.backend.engine import DEFAULT_MOVEMENT_POLICY

_HAS_NATIVE = importlib.util.find_spec("SDK.native_antwar") is not None


class MatchRuntimeWrapper:
    """MatchRuntime 包装类 - 提供兼容的 resolve_turn 方法"""

    def __init__(self, original_runtime):
        self._original = original_runtime

    def __getattr__(self, name):
        return getattr(self._original, name)

    def resolve_turn(self, self_operations, opponent_operations, self_first):
        """SDK 兼容的回合执行。

        Args:
            self_operations: 己方操作列表
            opponent_operations: 对方操作列表
            self_first: 己方是否对应 player 0
        """
        ops_self = self._normalize_ops(self_operations)
        ops_opponent = self._normalize_ops(opponent_operations)
        if self_first:
            if ops_self:
                self._original.apply_self_operations(ops_self)
            if ops_opponent:
                self._original.apply_opponent_operations(ops_opponent)
        else:
            if ops_opponent:
                self._original.apply_self_operations(ops_opponent)
            if ops_self:
                self._original.apply_opponent_operations(ops_self)
        self._original.state.advance_round()
        return _SimulatedTurnResolution(self._original.state)

    @staticmethod
    def _normalize_ops(operations):
        if operations is None:
            return []
        if isinstance(operations, list):
            return operations
        return [operations]


class _SimulatedTurnResolution:
    def __init__(self, state):
        self.state = state
        self.terminal = state.terminal
        self.round_index = state.round_index
        self.players = state.players


class BackendAdapter:
    """Backend 兼容适配器 - 封装 SDK 版本差异，提供统一的运行时创建接口"""

    @staticmethod
    def create_runtime(
        player: int,
        cold_handle_rule_illegal: bool = False,
        prefer_native: bool = False,
        movement_policy: Optional[Any] = None,
    ) -> MatchRuntimeWrapper:
        """创建游戏运行时实例"""
        try:
            from SDK.backend.core import load_backend
            from SDK.backend.runtime import MatchRuntime
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SDK. Please ensure Ant-Game SDK is installed. Error: {e}"
            )

        backend = load_backend(prefer_native=prefer_native)

        kwargs = dict(
            player=player, seed=0, prefer_native=prefer_native, backend=backend
        )
        sig = inspect.signature(MatchRuntime.create)
        if "movement_policy" in sig.parameters:
            kwargs["movement_policy"] = movement_policy or DEFAULT_MOVEMENT_POLICY
        if "cold_handle_rule_illegal" in sig.parameters:
            kwargs["cold_handle_rule_illegal"] = cold_handle_rule_illegal
        runtime = MatchRuntime.create(**kwargs)

        return MatchRuntimeWrapper(runtime)

    @staticmethod
    def supports_native_backend() -> bool:
        """检测是否支持 Native 后端"""
        return _HAS_NATIVE

    @staticmethod
    def get_backend_info() -> Dict[str, Any]:
        """获取 Backend 信息"""
        import torch

        info: Dict[str, Any] = {
            "python_backend": True,
            "native_backend": BackendAdapter.supports_native_backend(),
        }
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
        return info


__all__ = ["BackendAdapter", "MatchRuntimeWrapper"]
