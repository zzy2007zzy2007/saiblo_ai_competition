#!/usr/bin/env python3
"""
集成测试：验证与SDK示例的行为一致性
"""

import sys
import os

# 添加必要的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ppo", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Ant-Game"))


def test_sdk_sequential_env_interface():
    """验证AntWarEnv实现了与SDK示例一致的接口"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=42)
    obs, info = env.reset()

    # 验证接口属性
    assert hasattr(env, 'current_player'), "应该有current_player属性"
    assert hasattr(env, 'settles_after_action'), "应该有settles_after_action属性"

    # 验证决策上下文
    assert env.current_player == 0, "初始玩家应该是0"
    assert info.get("to_play") == 0, "info应该包含to_play"

    env.close()
    print("✅ test_sdk_sequential_env_interface 通过")


def test_step_returns_info_with_context():
    """验证step返回的info包含决策上下文"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=123)
    obs, info = env.reset()

    # 玩家0行动
    obs, rewards, terminated, truncated, info = env.step(0)
    assert "to_play" in info, "info应该包含to_play"
    assert "phase" in info, "info应该包含phase"
    assert "settles_after_action" in info, "info应该包含settles_after_action"
    assert info["to_play"] == 1, "当前应该是玩家1"
    assert info["settles_after_action"] == True, "玩家1行动后应该settles"

    # 玩家1行动
    obs, rewards, terminated, truncated, info = env.step(0)
    assert info["to_play"] == 0, "当前应该是玩家0"
    assert info["settles_after_action"] == False, "玩家0行动后不应该settles"

    env.close()
    print("✅ test_step_returns_info_with_context 通过")


def test_round_progression_timing():
    """验证回合推进时机正确"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=456)
    env.reset()

    initial_round = env._runtime.state.round_index

    # 玩家0行动（不推进回合）
    env.step(0)
    assert env._runtime.state.round_index == initial_round, "玩家0行动后回合不应推进"

    # 玩家1行动（推进回合）
    env.step(0)
    assert env._runtime.state.round_index == initial_round + 1, "玩家1行动后回合应该推进"

    env.close()
    print("✅ test_round_progression_timing 通过")


def test_reward_timing():
    """验证奖励只在回合完成时计算"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=789)
    env.reset()

    # 玩家0行动
    obs, rewards_p0, terminated, truncated, info = env.step(0)
    assert rewards_p0["player_0"] == 0.0, "玩家0行动后不应有奖励"
    assert rewards_p0["player_1"] == 0.0, "玩家0行动后不应有奖励"

    # 玩家1行动
    obs, rewards_p1, terminated, truncated, info = env.step(0)
    # 此时应该有奖励（回合完成）
    assert rewards_p1["player_0"] != 0.0 or rewards_p1["player_1"] != 0.0, "玩家1行动后应该有奖励"

    env.close()
    print("✅ test_reward_timing 通过")


def test_noop_penalty_for_both_players():
    """验证两个玩家都会因NO-OP受到惩罚"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=111)
    env.reset()

    # 两个玩家都执行NO-OP
    env.step(0)  # 玩家0 NO-OP
    obs, rewards, terminated, truncated, info = env.step(0)  # 玩家1 NO-OP

    # 验证两个玩家都受到惩罚
    assert rewards["player_0"] < 0, "玩家0应该因NO-OP受到惩罚"
    assert rewards["player_1"] < 0, "玩家1应该因NO-OP受到惩罚"

    env.close()
    print("✅ test_noop_penalty_for_both_players 通过")


def test_full_episode_consistency():
    """验证完整对局的先手后手一致性"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=222)
    env.reset()

    max_rounds = 5
    for round_idx in range(max_rounds):
        # 验证回合开始状态
        assert env._runtime.state.round_index == round_idx, f"回合应该是{round_idx}"
        assert env.current_player == 0, "每回合应该从玩家0开始"

        # 执行一回合
        env.step(1)  # 玩家0
        env.step(1)  # 玩家1

        # 验证回合推进
        if not env._runtime.state.terminal:
            assert env._runtime.state.round_index == round_idx + 1, "回合应该推进"

    env.close()
    print("✅ test_full_episode_consistency 通过")


def test_backward_compatibility_structure():
    """验证返回结构与Gymnasium标准一致"""
    from ppo_antwar.env.antwar_env import AntWarEnv

    env = AntWarEnv(player_id=0, seed=333)
    obs, info = env.reset()

    obs, rewards, terminated, truncated, info = env.step(0)

    # 验证返回结构
    assert isinstance(obs, dict), "observations应该是dict"
    assert isinstance(rewards, dict), "rewards应该是dict"
    assert isinstance(terminated, bool), "terminated应该是bool"
    assert isinstance(truncated, bool), "truncated应该是bool"
    assert isinstance(info, dict), "info应该是dict"

    # 验证observations结构
    assert "player_0" in obs or "board" in obs, "observations应该包含player_0或board"

    env.close()
    print("✅ test_backward_compatibility_structure 通过")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AntWar 先手后手机制集成测试")
    print("=" * 80)

    tests = [
        test_sdk_sequential_env_interface,
        test_step_returns_info_with_context,
        test_round_progression_timing,
        test_reward_timing,
        test_noop_penalty_for_both_players,
        test_full_episode_consistency,
        test_backward_compatibility_structure,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        print(f"\n测试: {test_func.__name__}")
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