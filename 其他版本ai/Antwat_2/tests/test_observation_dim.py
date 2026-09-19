#!/usr/bin/env python3
"""验证 ObservationEncoder 的 global_dim 是否为 30 的测试脚本"""

import sys
from pathlib import Path

# 添加 src 目录到路径
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

print("=" * 80)
print("测试 ObservationEncoder 的 global_dim 和 encode_global 输出")
print("=" * 80)
print()

# ==================== 测试 1: 检查 global_dim ====================
print("[测试 1/2] 检查 ObservationEncoder 的 global_dim...")
try:
    from ppo_antwar.env.observation import ObservationEncoder
    
    encoder = ObservationEncoder()
    print(f"  encoder.global_dim = {encoder.global_dim}")
    
    if encoder.global_dim == 30:
        print("  ✓ global_dim 正确设置为 30")
    else:
        print(f"  ✗ global_dim 错误，应为 30，实际为 {encoder.global_dim}")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 测试 2: 实际运行 encode_global ====================
print("[测试 2/2] 实际运行 encode_global 检查输出维度...")
try:
    from ppo_antwar.env.antwar_env import AntWarEnv
    
    # 创建环境并重置
    env = AntWarEnv(player_id=0, seed=42)
    obs, infos = env.reset()
    
    # 获取 player_0 的观测
    player_0_obs = obs['player_0']
    global_features = player_0_obs['global']
    
    print(f"  global_features.shape = {global_features.shape}")
    print(f"  global_features 维度 = {len(global_features)}")
    
    if len(global_features) == 30:
        print("  ✓ encode_global 输出 30 维，正确！")
    else:
        print(f"  ✗ encode_global 输出维度错误，应为 30，实际为 {len(global_features)}")
        sys.exit(1)
    
    # 打印最后 8 个元素（4个我方超级武器冷却 + 4个对手超级武器冷却）
    print()
    print("  最后 8 个特征（4个我方超级武器 + 4个对手超级武器冷却）:")
    for i in range(-8, 0):
        print(f"    feature[{i}] = {global_features[i]:.4f}")
    
    print()
    print("  ✓ 所有测试通过！")
        
except Exception as e:
    print(f"  ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("所有测试通过！✓")
print("=" * 80)
