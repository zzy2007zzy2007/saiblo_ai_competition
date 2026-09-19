#!/usr/bin/env python3
"""
测试先手后手机制修复
"""

import sys
import os

# 添加必要的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ppo", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Ant-Game"))

import pytest
from ppo_antwar.env.antwar_env import AntWarEnv


class TestTurnOrder:
    """测试先手后手机制"""

    def test_player_0_always_moves_first(self):
        """验证玩家0总是先手"""
        env = AntWarEnv(player_id=0, seed=42)
        obs, info = env.reset()
        
        assert env.current_player == 0, "初始状态应该是玩家0行动"
        assert info.get("to_play") == 0, "info应该指示玩家0行动"
        
        # 玩家0行动（不推进回合）
        obs, rewards, terminated, truncated, info = env.step(0)
        assert env._runtime.state.round_index == 0, "玩家0行动后回合不应推进"
        assert env.current_player == 1, "玩家0行动后应该轮到玩家1"
        assert rewards["player_0"] == 0.0, "玩家0行动后不应有奖励"
        assert rewards["player_1"] == 0.0, "玩家0行动后不应有奖励"
        
        # 玩家1行动（推进回合）
        obs, rewards, terminated, truncated, info = env.step(0)
        assert env._runtime.state.round_index == 1, "玩家1行动后回合应该推进"
        assert env.current_player == 0, "玩家1行动后应该轮到玩家0"
        
        env.close()

    def test_round_advances_only_after_player_1(self):
        """验证仅在玩家1行动后推进回合"""
        env = AntWarEnv(player_id=0, seed=123)
        env.reset()
        
        initial_round = env._runtime.state.round_index
        
        # 玩家0行动
        env.step(0)
        assert env._runtime.state.round_index == initial_round, "玩家0行动后回合不应推进"
        
        # 玩家1行动
        env.step(0)
        assert env._runtime.state.round_index == initial_round + 1, "玩家1行动后回合应该推进"
        
        env.close()

    def test_noop_penalty_applied_to_both_players(self):
        """验证NO-OP惩罚应用于两个玩家"""
        env = AntWarEnv(player_id=0, seed=456)
        env.reset()
        
        # 两个玩家都执行NO-OP（动作0）
        env.step(0)  # 玩家0
        obs, rewards, terminated, truncated, info = env.step(0)  # 玩家1
        
        assert rewards["player_0"] < 0, "玩家0的NO-OP应该被惩罚"
        assert rewards["player_1"] < 0, "玩家1的NO-OP应该被惩罚"
        
        env.close()

    def test_no_noop_penalty_when_not_noop(self):
        """验证非NO-OP动作不被惩罚"""
        env = AntWarEnv(player_id=0, seed=789)
        env.reset()
        
        # 两个玩家都执行非NO-OP动作
        # 注意：需要使用有效的动作ID，动作1通常是有效的
        env.step(1)  # 玩家0
        obs, rewards, terminated, truncated, info = env.step(1)  # 玩家1
        
        # 检查奖励是否不为负（假设执行动作会获得正奖励或至少不被惩罚）
        # 由于环境的随机性，我们只检查NO-OP惩罚没有被应用
        no_op_penalty = -0.3
        assert rewards["player_0"] != no_op_penalty, "玩家0不应受到NO-OP惩罚"
        assert rewards["player_1"] != no_op_penalty, "玩家1不应受到NO-OP惩罚"
        
        env.close()

    def test_context_switching_correctly(self):
        """验证决策上下文正确切换"""
        env = AntWarEnv(player_id=0, seed=999)
        env.reset()
        
        # 初始状态
        assert env._context.to_play == 0
        assert not env._context.settles_after_action
        
        # 玩家0行动后
        env.step(0)
        assert env._context.to_play == 1
        assert env._context.settles_after_action
        
        # 玩家1行动后
        env.step(0)
        assert env._context.to_play == 0
        assert not env._context.settles_after_action
        
        env.close()

    def test_multiple_rounds_consistency(self):
        """验证多个回合的一致性"""
        env = AntWarEnv(player_id=0, seed=111)
        env.reset()
        
        for round_idx in range(5):
            # 验证回合开始时的状态
            assert env._runtime.state.round_index == round_idx
            assert env.current_player == 0
            
            # 玩家0行动
            env.step(1)
            assert env.current_player == 1
            
            # 玩家1行动
            env.step(1)
            assert env.current_player == 0
            assert env._runtime.state.round_index == round_idx + 1
        
        env.close()


if __name__ == "__main__":
    # 直接运行测试
    print("\n" + "=" * 80)
    print("AntWar 先手后手机制测试")
    print("=" * 80)
    
    test = TestTurnOrder()
    
    tests = [
        ("test_player_0_always_moves_first", test.test_player_0_always_moves_first),
        ("test_round_advances_only_after_player_1", test.test_round_advances_only_after_player_1),
        ("test_noop_penalty_applied_to_both_players", test.test_noop_penalty_applied_to_both_players),
        ("test_no_noop_penalty_when_not_noop", test.test_no_noop_penalty_when_not_noop),
        ("test_context_switching_correctly", test.test_context_switching_correctly),
        ("test_multiple_rounds_consistency", test.test_multiple_rounds_consistency),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n测试: {name}")
        try:
            test_func()
            print(f"✅ 通过")
            passed += 1
        except AssertionError as e:
            print(f"❌ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 80}")
    print(f"测试结果: 共 {len(tests)} 个测试，通过 {passed} 个，失败 {failed} 个")
    print(f"{'=' * 80}")
    
    sys.exit(0 if failed == 0 else 1)