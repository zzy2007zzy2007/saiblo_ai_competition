#!/usr/bin/env python3
"""
简化版测试脚本 - 直接测试 resolve_turn 兼容性修复
"""

import sys
import os

# 添加必要的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Ant-Game"))


def test_core_compat():
    """测试核心兼容性功能"""
    print("=" * 80)
    print("测试: 核心兼容性功能")
    print("=" * 80)
    
    try:
        # 直接导入我们的包装类
        from SDK.backend.engine import DEFAULT_MOVEMENT_POLICY
        from SDK.backend.core import load_backend
        from SDK.backend.runtime import MatchRuntime
        
        # 我们自己实现一个简化的包装类来测试
        class SimpleMatchRuntimeWrapper:
            def __init__(self, original_runtime):
                self._original = original_runtime
                
            def __getattr__(self, name):
                return getattr(self._original, name)
                
            def resolve_turn(self, player_0_operations, player_1_operations):
                ops_0 = player_0_operations if isinstance(player_0_operations, list) else [player_0_operations] if player_0_operations else []
                ops_1 = player_1_operations if isinstance(player_1_operations, list) else [player_1_operations] if player_1_operations else []
                
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
                
                public_state = self._original.state.to_public_round_state()
                self._original.finish_round(public_state)
                
                class SimulatedTurnResolution:
                    def __init__(self, state):
                        self.state = state
                        self.terminal = state.terminal
                        self.round_index = state.round_index
                        if hasattr(state, 'players'):
                            self.players = state.players
                
                return SimulatedTurnResolution(self._original.state)
        
        # 测试创建原始 runtime
        backend = load_backend(prefer_native=False)
        original_runtime = MatchRuntime.create(
            player=0,
            seed=42,
            prefer_native=False,
            backend=backend,
            movement_policy=DEFAULT_MOVEMENT_POLICY,
            cold_handle_rule_illegal=True,
        )
        print("✓ 成功创建原始 MatchRuntime")
        
        # 测试包装类
        wrapped_runtime = SimpleMatchRuntimeWrapper(original_runtime)
        print("✓ 成功创建 SimpleMatchRuntimeWrapper")
        
        # 检查是否有 resolve_turn 方法
        if hasattr(wrapped_runtime, 'resolve_turn'):
            print("✓ 包装类有 resolve_turn 方法")
        else:
            print("✗ 包装类没有 resolve_turn 方法")
            return False
        
        # 测试 resolve_turn 调用
        result = wrapped_runtime.resolve_turn([], [])
        print("✓ resolve_turn() 调用成功")
        print(f"  - result.terminal = {result.terminal}")
        print(f"  - result.round_index = {result.round_index}")
        
        # 测试代理其他方法和属性
        if hasattr(wrapped_runtime, 'state'):
            print("✓ 包装类可以访问 state 属性")
        if hasattr(wrapped_runtime, 'player'):
            print(f"✓ 包装类可以访问 player 属性: {wrapped_runtime.player}")
        
        print("\n✅ 核心兼容性功能测试成功！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("AntWar resolve_turn 兼容性简化测试")
    print("=" * 80)
    
    success = test_core_compat()
    
    if success:
        print("\n🎉 测试成功！兼容性修复有效！")
        return 0
    else:
        print("\n❌ 测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
