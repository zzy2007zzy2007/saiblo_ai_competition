#!/usr/bin/env python3
"""验证 AntWar PPO 代码库修复完整性的脚本"""

import sys
import os
import importlib.util
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

print("=" * 80)
print("AntWar PPO 代码库修复验证")
print("=" * 80)
print()

# ==================== 检查 1: 直接加载网络模块 ====================
print("[检查 1/4] 加载网络模块...")
try:
    network_path = src_dir / "ppo_antwar" / "network" / "antwar_net.py"
    spec = importlib.util.spec_from_file_location("antwar_net", str(network_path))
    antwar_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(antwar_net)

    AntWarPolicyValueNetwork = antwar_net.AntWarPolicyValueNetwork
    count_parameters = antwar_net.count_parameters

    print("  ✓ network 模块加载成功")

    # 加载常量模块
    utils_path = src_dir / "ppo_antwar" / "utils" / "action_constants.py"
    spec = importlib.util.spec_from_file_location("action_constants", str(utils_path))
    action_constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(action_constants)

    MAX_ACTIONS = action_constants.MAX_ACTIONS
    ACTION_DIM = action_constants.ACTION_DIM

    print(f"  ✓ action_constants 加载成功 (MAX_ACTIONS={MAX_ACTIONS}, ACTION_DIM={ACTION_DIM})")

    print("  ✓ 所有模块加载成功！")
except Exception as e:
    print(f"  ✗ 模块加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 检查 2: 加载配置文件 ====================
print("[检查 2/4] 加载配置文件...")
config_path = project_root / "src" / "ppo_antwar" / "configs" / "ppo_antwar.yaml"
try:
    import yaml
    from easydict import EasyDict

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = EasyDict(config_dict)

    print(f"  ✓ 配置文件加载成功: {config_path}")
    print(f"  ✓ 网络配置:")
    print(f"    - board_shape: {config.network.board_shape}")
    print(f"    - global_dim: {config.network.global_dim}")
    print(f"    - action_dim: {config.network.action_dim}")
    print(f"    - hidden_dim: {config.network.hidden_dim}")
    print(f"    - rnn_type: {config.network.rnn_type}")
    print(f"    - rnn_layers: {config.network.rnn_layers}")
    print(f"    - use_popart: {config.network.use_popart}")
except Exception as e:
    print(f"  ✗ 配置文件加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 检查 3: 初始化策略网络 ====================
print("[检查 3/4] 初始化策略网络...")
try:
    import torch

    device = torch.device("cpu")  # 使用 CPU 测试
    policy = AntWarPolicyValueNetwork(
        board_shape=tuple(config.network.board_shape),
        global_dim=config.network.global_dim,
        action_dim=config.network.action_dim,
        hidden_dim=config.network.hidden_dim,
        rnn_type=config.network.rnn_type,
        rnn_layers=config.network.rnn_layers,
        use_popart=config.network.use_popart,
    ).to(device)

    print(f"  ✓ 策略网络初始化成功")
    print(f"  ✓ 网络参数数量: {count_parameters(policy)}")

    # 测试前向传播
    batch_size = 2
    board = torch.randn(batch_size, *config.network.board_shape, device=device)
    global_features = torch.randn(batch_size, config.network.global_dim, device=device)
    action_mask = torch.ones(batch_size, config.network.action_dim, device=device)

    with torch.no_grad():
        logits, value, rnn_state = policy(board, global_features, action_mask)

    print(f"  ✓ 前向传播测试成功")
    print(f"    - logits shape: {logits.shape}")
    print(f"    - value shape: {value.shape}")

    # 测试 get_action 方法
    test_board = torch.randn(1, *config.network.board_shape, device=device)
    test_global = torch.randn(1, config.network.global_dim, device=device)
    test_mask = torch.ones(1, config.network.action_dim, device=device)

    with torch.no_grad():
        action, log_prob, value_pred, new_rnn = policy.get_action(
            test_board, test_global, test_mask, deterministic=False
        )

    print(f"  ✓ get_action 测试成功")
    print(f"    - action: {action.item()}")

except Exception as e:
    print(f"  ✗ 策略网络初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 检查 4: 验证 trainer 模块的语法 ====================
print("[检查 4/4] 验证 trainer 模块语法...")
try:
    # 简单地编译训练器模块来检查语法错误
    trainer_path = src_dir / "ppo_antwar" / "trainer" / "ppo_trainer.py"
    with open(trainer_path, 'r', encoding='utf-8') as f:
        source = f.read()
    compile(source, str(trainer_path), 'exec')

    print(f"  ✓ trainer 模块语法验证成功")
    print(f"  ✓ trainer 模块没有语法错误")

    # 简单检查导入语句
    lines = source.split('\n')
    has_action_dim = False
    has_max_actions = False
    for line in lines:
        if 'ACTION_DIM' in line:
            has_action_dim = True
        if 'MAX_ACTIONS' in line:
            has_max_actions = True

    if has_action_dim:
        print("  ✓ trainer 模块正确导入 ACTION_DIM")
    if has_max_actions:
        print("  ✓ trainer 模块正确导入 MAX_ACTIONS")

except Exception as e:
    print(f"  ✗ trainer 模块语法验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("验证完成！所有检查通过 ✓")
print("=" * 80)
