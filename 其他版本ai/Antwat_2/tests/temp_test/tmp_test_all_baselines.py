#!/usr/bin/env python3
"""
完整的 Baseline 模型加载和对战测试
测试所有 baseline 模型：ann_593, ann_v1, gen99, gen199, sample
"""

import os
import sys
import torch

# 配置
BASELINES_DIR = '/root/autodl-tmp/AntWar/baselines'
SDK_DIR = '/root/autodl-tmp/AntWar/baselines/ann_593/SDK'

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_sdk_import():
    """测试 SDK 导入"""
    print_section("1. SDK 导入测试")

    sys.path.insert(0, SDK_DIR)
    print(f"✓ 添加 SDK 到路径: {SDK_DIR}")

    try:
        from SDK.backend.model import Operation, OperationType
        print(f"✓ 成功导入 Operation, OperationType")

        op = Operation(OperationType.BUILD_TOWER, arg0=4, arg1=6)
        print(f"✓ 成功创建 Operation: {op}")
        return True
    except Exception as e:
        print(f"✗ SDK 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ann_593():
    """测试 ann_593 模型"""
    print_section("2. Ann593 模型测试")

    model_dir = os.path.join(BASELINES_DIR, 'ann_593')
    sys.path.insert(0, model_dir)

    print(f"目录内容: {os.listdir(model_dir)}")

    # 检查 neural_network.py
    nn_file = os.path.join(model_dir, 'neural_network.py')
    if not os.path.exists(nn_file):
        print(f"✗ neural_network.py 不存在")
        return False

    try:
        from neural_network import NeuralNetworkAgent
        print(f"✓ 导入 NeuralNetworkAgent 成功")

        # 测试 hidden_dim=256 (从之前的错误信息得知)
        agent = NeuralNetworkAgent(10144, 32, hidden_dim=256)
        print(f"✓ 创建 agent 成功 (hidden_dim=256)")

        model_path = os.path.join(model_dir, 'model.pth')
        if os.path.exists(model_path):
            # 检查模型文件
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✓ 加载模型文件成功")
            print(f"  模型类型: {type(checkpoint)}")

            if isinstance(checkpoint, dict):
                print(f"  模型键: {list(checkpoint.keys())}")
            else:
                print(f"  模型形状: {checkpoint.shape if hasattr(checkpoint, 'shape') else 'N/A'}")

            # 尝试加载
            agent.load_model(model_path)
            print(f"✓ 模型加载成功")

            return True
        else:
            print(f"✗ 模型文件不存在: {model_path}")
            return False

    except Exception as e:
        print(f"✗ Ann593 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ann_v1():
    """测试 ann_v1 模型"""
    print_section("3. AnnV1 模型测试")

    model_dir = os.path.join(BASELINES_DIR, 'ann_v1')
    print(f"目录内容: {os.listdir(model_dir)}")

    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        print(f"✗ model.pth 不存在")
        return False

    # 检查模型结构
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✓ 模型文件加载成功")
        print(f"  模型类型: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"  模型键: {list(checkpoint.keys())}")

            # 检查网络结构
            for key, value in checkpoint.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")

        return True
    except Exception as e:
        print(f"✗ AnnV1 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gen99():
    """测试 gen99 规则型 AI"""
    print_section("4. Gen99 规则型 AI 测试")

    ai_dir = os.path.join(BASELINES_DIR, 'gen99')
    print(f"目录内容: {os.listdir(ai_dir)}")

    # 检查 SDK
    sdk_dir = os.path.join(ai_dir, 'SDK')
    if os.path.exists(sdk_dir):
        sys.path.insert(0, sdk_dir)
        print(f"✓ 添加 gen99 SDK 到路径")

    sys.path.insert(0, ai_dir)
    print(f"✓ 添加 gen99 到路径")

    try:
        from ai import AI, create_agent
        print(f"✓ 导入 AI 成功")

        # 创建 agent
        agent = create_agent()
        print(f"✓ 创建 agent 成功: {type(agent)}")

        # 检查方法
        if hasattr(agent, 'choose_bundle'):
            print(f"✓ agent.choose_bundle 方法存在")
        if hasattr(agent, 'choose_operations'):
            print(f"✓ agent.choose_operations 方法存在")

        return True
    except Exception as e:
        print(f"✗ Gen99 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gen199():
    """测试 gen199 规则型 AI"""
    print_section("5. Gen199 规则型 AI 测试")

    ai_dir = os.path.join(BASELINES_DIR, 'gen199')
    print(f"目录内容: {os.listdir(ai_dir)}")

    # 检查 SDK
    sdk_dir = os.path.join(ai_dir, 'SDK')
    if os.path.exists(sdk_dir):
        sys.path.insert(0, sdk_dir)
        print(f"✓ 添加 gen199 SDK 到路径")

    sys.path.insert(0, ai_dir)
    print(f"✓ 添加 gen199 到路径")

    try:
        from ai import AI, create_agent
        print(f"✓ 导入 AI 成功")

        # 创建 agent
        agent = create_agent()
        print(f"✓ 创建 agent 成功: {type(agent)}")

        return True
    except Exception as e:
        print(f"✗ Gen199 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample():
    """测试 sample 规则型 AI"""
    print_section("6. Sample 规则型 AI 测试")

    ai_dir = os.path.join(BASELINES_DIR, 'sample')
    print(f"目录内容: {os.listdir(ai_dir)}")

    # 检查 SDK
    sdk_dir = os.path.join(ai_dir, 'SDK')
    if os.path.exists(sdk_dir):
        sys.path.insert(0, sdk_dir)
        print(f"✓ 添加 sample SDK 到路径")

    sys.path.insert(0, ai_dir)
    print(f"✓ 添加 sample 到路径")

    try:
        from ai import AI, create_agent
        print(f"✓ 导入 AI 成功")

        # 创建 agent
        agent = create_agent()
        print(f"✓ 创建 agent 成功: {type(agent)}")

        return True
    except Exception as e:
        print(f"✗ Sample 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("  完整的 Baseline 模型加载和对战测试")
    print("=" * 70)

    results = {}

    # 1. SDK 导入
    results['SDK'] = test_sdk_import()

    # 2. Ann593
    results['ann_593'] = test_ann_593()

    # 3. AnnV1
    results['ann_v1'] = test_ann_v1()

    # 4. Gen99
    results['gen99'] = test_gen99()

    # 5. Gen199
    results['gen199'] = test_gen199()

    # 6. Sample
    results['sample'] = test_sample()

    # 总结
    print_section("测试结果总结")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name}: {status}")

    print("\n" + "=" * 70)
    print("  测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
