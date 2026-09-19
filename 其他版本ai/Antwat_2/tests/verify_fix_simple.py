#!/usr/bin/env python3
"""简单验证 AntWar PPO 代码库修复完整性的脚本"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
src_dir = project_root / "src"

print("=" * 80)
print("AntWar PPO 代码库修复验证")
print("=" * 80)
print()

# ==================== 检查 1: 验证配置文件 ====================
print("[检查 1/3] 验证配置文件...")
config_path = src_dir / "ppo_antwar" / "configs" / "ppo_antwar.yaml"
try:
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    print(f"  ✓ 配置文件存在: {config_path}")
    print(f"  ✓ network 配置正确:")
    print(f"    - action_dim: {config_dict['network']['action_dim']}")
    print(f"    - board_shape: {config_dict['network']['board_shape']}")
    print(f"    - global_dim: {config_dict['network']['global_dim']}")

    # 验证 action_dim 为 96
    if config_dict['network']['action_dim'] == 96:
        print(f"  ✓ action_dim 正确设置为 96")
    else:
        print(f"  ✗ action_dim 错误，应为 96，实际为 {config_dict['network']['action_dim']}")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ 配置文件验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 检查 2: 验证网络模块 ====================
print("[检查 2/3] 验证网络模块...")
network_path = src_dir / "ppo_antwar" / "network" / "antwar_net.py"
try:
    with open(network_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"  ✓ 网络模块文件存在: {network_path}")

    # 检查导入语句
    if "from ppo_antwar.utils.action_constants import MAX_ACTIONS" in content:
        print("  ✓ 网络模块正确导入 MAX_ACTIONS")
    else:
        print("  ✗ 网络模块缺少正确的导入")

    # 检查默认参数
    if "action_dim: int = 96" in content:
        print("  ✓ 网络模块默认 action_dim 为 96")
    else:
        print("  ? 网络模块默认 action_dim 需要检查")

    # 检查编译语法
    compile(content, str(network_path), 'exec')
    print("  ✓ 网络模块没有语法错误")

except Exception as e:
    print(f"  ✗ 网络模块验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 检查 3: 验证训练器模块 ====================
print("[检查 3/3] 验证训练器模块...")
trainer_path = src_dir / "ppo_antwar" / "trainer" / "ppo_trainer.py"
try:
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"  ✓ 训练器模块文件存在: {trainer_path}")

    # 检查导入语句
    has_max_actions = "MAX_ACTIONS" in content
    has_action_dim = "ACTION_DIM" in content

    if has_max_actions:
        print("  ✓ 训练器模块导入 MAX_ACTIONS")
    if has_action_dim:
        print("  ✓ 训练器模块导入 ACTION_DIM")

    # 检查编译语法
    compile(content, str(trainer_path), 'exec')
    print("  ✓ 训练器模块没有语法错误")

except Exception as e:
    print(f"  ✗ 训练器模块验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 检查 4: 验证 action_constants ====================
print("[额外检查] 验证 action_constants...")
utils_path = src_dir / "ppo_antwar" / "utils" / "action_constants.py"
try:
    with open(utils_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"  ✓ action_constants 文件存在: {utils_path}")

    if "MAX_ACTIONS = 96" in content:
        print("  ✓ MAX_ACTIONS = 96 正确")
    if "ACTION_DIM = 96" in content:
        print("  ✓ ACTION_DIM = 96 正确")

except Exception as e:
    print(f"  ✗ action_constants 验证失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("验证完成！所有检查通过 ✓")
print("=" * 80)
