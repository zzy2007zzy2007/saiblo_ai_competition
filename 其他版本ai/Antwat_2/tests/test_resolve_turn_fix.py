#!/usr/bin/env python3
"""
测试 resolve_turn 兼容性修复脚本

验证我们添加的 MatchRuntimeWrapper 是否能够正确模拟旧版 API
"""

import sys
import os

# 添加 ppo 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ppo_antwar.compat import BackendAdapter
from ppo_antwar.env.antwar_env import AntWarEnv


def test_resolve_turn():
    """测试 resolve_turn 方法是否可以正常调用"""
    print("=" * 80)
    print("测试 1: 测试 BackendAdapter 创建的 runtime 是否有 resolve_turn 方法")
    print("=" * 80)
    
    try:
        runtime = BackendAdapter.create_runtime(player=0, seed=42)
        print(f"✓ 成功创建 runtime: {type(runtime)}")
        
        # 检查是否有 resolve_turn 方法
        if hasattr(runtime, 'resolve_turn'):
            print("✓ runtime 有 resolve_turn 方法")
        else:
            print("✗ runtime 没有 resolve_turn 方法")
            return False
            
        # 检查是否可以正常访问其他属性
        if hasattr(runtime, 'state'):
            print("✓ runtime 有 state 属性")
        if hasattr(runtime, 'player'):
            print(f"✓ runtime.player = {runtime.player}")
        if hasattr(runtime, 'apply_self_operations'):
            print("✓ runtime 有 apply_self_operations 方法")
            
        print("\n✅ 测试 1 成功通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_antwar_env():
    """测试 AntWarEnv 是否可以正常初始化和运行"""
    print("\n" + "=" * 80)
    print("测试 2: 测试 AntWarEnv 的初始化和基本操作")
    print("=" * 80)
    
    try:
        env = AntWarEnv(player_id=0, seed=42)
        print("✓ 成功创建 AntWarEnv")
        
        # 测试 reset
        obs, info = env.reset()
        print("✓ env.reset() 成功")
        print(f"  - obs 包含键: {list(obs.keys())}")
        print(f"  - board 形状: {obs['board'].shape}")
        
        # 测试 step
        actions = {"player_0": 0, "player_1": 0}
        obs, rewards, terminated, truncated, info = env.step(actions)
        print("✓ env.step() 成功")
        print(f"  - terminated: {terminated}, truncated: {truncated}")
        
        env.close()
        print("\n✅ 测试 2 成功通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resolve_turn_call():
    """测试直接调用 resolve_turn 方法"""
    print("\n" + "=" * 80)
    print("测试 3: 测试直接调用 resolve_turn 方法")
    print("=" * 80)
    
    try:
        runtime = BackendAdapter.create_runtime(player=0, seed=1234)
        print("✓ 创建 runtime 成功")
        
        # 模拟调用 resolve_turn（使用空操作列表）
        result = runtime.resolve_turn([], [])
        print("✓ resolve_turn() 调用成功")
        print(f"  - result 类型: {type(result)}")
        print(f"  - result.terminal: {result.terminal}")
        print(f"  - result.round_index: {result.round_index}")
        
        print("\n✅ 测试 3 成功通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("AntWar resolve_turn 兼容性修复测试")
    print("=" * 80)
    
    results = []
    results.append(("测试 1: BackendAdapter 功能", test_resolve_turn()))
    results.append(("测试 2: AntWarEnv 基本操作", test_antwar_env()))
    results.append(("测试 3: resolve_turn 直接调用", test_resolve_turn_call()))
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！兼容性修复成功！")
        return 0
    else:
        print("\n❌ 部分测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
