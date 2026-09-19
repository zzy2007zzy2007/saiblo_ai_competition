#!/usr/bin/env python3
"""
测试 baseline 模型加载和对战
简化版本，用于在 server5 的 /tmp/ 目录运行

需要文件：
- baselines/ann_593/model.pth
- baselines/ann_593/neural_network.py
- baselines/ann_v1/model.pth (无 neural_network.py)
- baselines/ann_593/SDK/ (完整SDK)
"""

import os
import sys

# 测试配置
TEST_DIR = '/tmp/baseline_test'
BASELINES_DIR = '/root/autodl-tmp/AntWar/baselines'
SDK_DIR = '/root/autodl-tmp/AntWar/baselines/ann_593/SDK'

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_files():
    """检查必要的文件是否存在"""
    print_section("检查文件")

    # 检查 baselines
    for model in ['ann_593', 'ann_v1']:
        model_dir = os.path.join(BASELINES_DIR, model)
        model_file = os.path.join(model_dir, 'model.pth')
        nn_file = os.path.join(model_dir, 'neural_network.py')

        print(f"\n{model}:")
        print(f"  目录存在: {os.path.exists(model_dir)}")
        print(f"  model.pth: {os.path.exists(model_file)}")
        print(f"  neural_network.py: {os.path.exists(nn_file)}")

        if os.path.exists(model_dir):
            print(f"  目录内容: {os.listdir(model_dir)}")

    # 检查 SDK
    print(f"\nSDK:")
    print(f"  SDK 目录: {SDK_DIR}")
    print(f"  存在: {os.path.exists(SDK_DIR)}")
    if os.path.exists(SDK_DIR):
        print(f"  内容: {os.listdir(SDK_DIR)}")

def test_import():
    """测试 SDK 导入"""
    print_section("测试 SDK 导入")

    # 添加 SDK 到路径
    if os.path.exists(SDK_DIR):
        sys.path.insert(0, SDK_DIR)
        print(f"✓ 添加 SDK 到路径: {SDK_DIR}")
    else:
        print(f"✗ SDK 目录不存在: {SDK_DIR}")
        return False

    # 添加 baselines 到路径
    ann593_dir = os.path.join(BASELINES_DIR, 'ann_593')
    sys.path.insert(0, ann593_dir)
    print(f"✓ 添加 ann_593 到路径: {ann593_dir}")

    # 测试导入
    try:
        from SDK.backend.model import Operation, OperationType
        print(f"✓ 成功导入 Operation, OperationType")
        print(f"  Operation: {Operation}")
        print(f"  OperationType: {OperationType}")

        # 测试创建 Operation
        op = Operation(OperationType.BUILD_TOWER, arg0=4, arg1=6)
        print(f"✓ 成功创建 Operation: {op}")
        print(f"  op_type: {op.op_type}")
        print(f"  arg0: {op.arg0}")
        print(f"  arg1: {op.arg1}")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_load(model_name='ann_593'):
    """测试模型加载"""
    print_section(f"测试 {model_name} 模型加载")

    model_dir = os.path.join(BASELINES_DIR, model_name)
    sys.path.insert(0, model_dir)

    try:
        from neural_network import NeuralNetworkAgent
        print(f"✓ 成功导入 NeuralNetworkAgent from {model_name}")

        # 测试不同的 hidden_dim
        for hidden_dim in [64, 128, 256, 512]:
            print(f"\n测试 hidden_dim={hidden_dim}:")
            try:
                agent = NeuralNetworkAgent(10144, 32, hidden_dim=hidden_dim)
                print(f"  ✓ 创建 agent 成功")

                model_path = os.path.join(model_dir, 'model.pth')
                if os.path.exists(model_path):
                    agent.load_model(model_path)
                    print(f"  ✓ 模型加载成功 (hidden_dim={hidden_dim})")
                    return True
                else:
                    print(f"  ✗ 模型文件不存在: {model_path}")
            except Exception as e:
                print(f"  ✗ 失败: {e}")

    except Exception as e:
        print(f"✗ 导入 NeuralNetworkAgent 失败: {e}")
        import traceback
        traceback.print_exc()

    return False

def test_battle():
    """测试对战"""
    print_section("测试对战")

    # 这里简化处理，只测试基本功能
    print("对战测试需要更完整的环境")
    print("建议使用 battle_simulator.py 进行完整测试")

    return True

def main():
    print("=" * 70)
    print("  Baseline 模型加载和对战测试")
    print("=" * 70)
    print(f"测试目录: {TEST_DIR}")
    print(f"Baselines 目录: {BASELINES_DIR}")
    print(f"SDK 目录: {SDK_DIR}")

    # 检查文件
    check_files()

    # 测试导入
    if not test_import():
        print("✗ SDK 导入失败，测试终止")
        return

    # 测试 ann_593 模型加载
    test_model_load('ann_593')

    # 测试 ann_v1 模型加载（ann_v1 没有 neural_network.py）
    print_section("ann_v1 模型测试")
    print("ann_v1 没有 neural_network.py，无法直接加载")
    print("ann_v1 可能是规则型 AI 或需要其他加载方式")

    # 测试对战
    test_battle()

    print_section("测试完成")
    print("测试完成！")

if __name__ == "__main__":
    main()
