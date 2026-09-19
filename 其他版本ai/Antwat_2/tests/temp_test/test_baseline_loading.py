#!/usr/bin/env python3
"""
Baseline 模型加载测试
测试所有 baseline 模型的正确加载方式
"""

import os
import sys
import torch

print("=" * 70)
print("Baseline 模型加载测试")
print("=" * 70)

# 配置
BASELINES_DIR = '/root/autodl-tmp/AntWar/baselines'

# 检查 SDK 路径
SDK_DIR = '/root/autodl-tmp/AntWar/baselines/ann_593/SDK'
print(f"SDK 目录: {SDK_DIR}")
print(f"SDK 存在: {os.path.exists(SDK_DIR)}")
if os.path.exists(SDK_DIR):
    print(f"SDK 内容: {os.listdir(SDK_DIR)}")

# 添加 SDK 路径
sys.path.insert(0, SDK_DIR)
print(f"✓ 添加 SDK 到 sys.path")

# 测试导入
print("\n测试 SDK 导入...")
try:
    import SDK
    print(f"✓ SDK 模块导入成功")
    print(f"  SDK 内容: {dir(SDK)}")
except Exception as e:
    print(f"✗ SDK 模块导入失败: {e}")

try:
    from SDK.backend.model import Operation, OperationType
    print("✓ 成功导入 Operation, OperationType")
except Exception as e:
    print(f"✗ 导入 Operation, OperationType 失败: {e}")
    import traceback
    traceback.print_exc()

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_ann_593():
    """测试 ann_593 模型加载"""
    print_section("测试 Ann593 模型加载")

    model_dir = os.path.join(BASELINES_DIR, 'ann_593')
    model_path = os.path.join(model_dir, 'model.pth')

    # 检查文件
    print(f"目录内容: {os.listdir(model_dir)}")
    print(f"模型文件存在: {os.path.exists(model_path)}")

    # 添加模型路径
    sys.path.insert(0, model_dir)

    try:
        from neural_network import NeuralNetworkAgent
        print("✓ 成功导入 NeuralNetworkAgent")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 加载模型权重查看结构
    print("\n检查模型结构...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"模型类型: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"模型键: {list(checkpoint.keys())}")
        for key, value in checkpoint.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")

    # 尝试加载
    print("\n尝试创建并加载模型...")
    agent = NeuralNetworkAgent(10144, 32, hidden_dim=256)

    try:
        agent.load_model(model_path)
        print("✓ 模型加载成功")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ann_v1():
    """测试 ann_v1 模型加载"""
    print_section("测试 AnnV1 模型加载")

    model_dir = os.path.join(BASELINES_DIR, 'ann_v1')
    model_path = os.path.join(model_dir, 'model.pth')

    # 检查文件
    print(f"目录内容: {os.listdir(model_dir)}")
    print(f"模型文件存在: {os.path.exists(model_path)}")

    # 检查是否有 neural_network.py
    nn_file = os.path.join(model_dir, 'neural_network.py')
    if os.path.exists(nn_file):
        print("✓ neural_network.py 存在")

        sys.path.insert(0, model_dir)
        try:
            from neural_network import NeuralNetworkAgent
            print("✓ 成功导入 NeuralNetworkAgent")
        except Exception as e:
            print(f"✗ 导入失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 加载模型权重查看结构
        print("\n检查模型结构...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"模型类型: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"模型键: {list(checkpoint.keys())}")
            for key, value in checkpoint.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")

        # 尝试加载
        print("\n尝试创建并加载模型...")
        agent = NeuralNetworkAgent(10144, 32, hidden_dim=256)

        try:
            agent.load_model(model_path)
            print("✓ 模型加载成功")
            return True
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("✗ neural_network.py 不存在")
        print("ann_v1 可能需要使用不同的加载方式")
        return False

def test_gen99():
    """测试 gen99 规则型 AI"""
    print_section("测试 Gen99 规则型 AI")

    ai_dir = os.path.join(BASELINES_DIR, 'gen99')

    # 检查 SDK
    sdk_dir = os.path.join(ai_dir, 'SDK')
    if os.path.exists(sdk_dir):
        sys.path.insert(0, sdk_dir)
        print(f"✓ 添加 gen99 SDK 到路径")

    sys.path.insert(0, ai_dir)
    print(f"✓ 添加 gen99 到路径")

    try:
        from ai import AI, create_agent
        print("✓ 成功导入 AI, create_agent")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        agent = create_agent()
        print(f"✓ 创建 agent 成功: {type(agent)}")

        # 检查方法
        if hasattr(agent, 'choose_bundle'):
            print("✓ agent.choose_bundle 方法存在")
        if hasattr(agent, 'choose_operations'):
            print("✓ agent.choose_operations 方法存在")

        return True
    except Exception as e:
        print(f"✗ 创建 agent 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gen199():
    """测试 gen199 规则型 AI"""
    print_section("测试 Gen199 规则型 AI")

    ai_dir = os.path.join(BASELINES_DIR, 'gen199')

    # 检查 SDK
    sdk_dir = os.path.join(ai_dir, 'SDK')
    if os.path.exists(sdk_dir):
        sys.path.insert(0, sdk_dir)
        print(f"✓ 添加 gen199 SDK 到路径")

    sys.path.insert(0, ai_dir)
    print(f"✓ 添加 gen199 到路径")

    try:
        from ai import AI, create_agent
        print("✓ 成功导入 AI, create_agent")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        agent = create_agent()
        print(f"✓ 创建 agent 成功: {type(agent)}")
        return True
    except Exception as e:
        print(f"✗ 创建 agent 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample():
    """测试 sample 规则型 AI"""
    print_section("测试 Sample 规则型 AI")

    ai_dir = os.path.join(BASELINES_DIR, 'sample')

    # 检查 SDK
    sdk_dir = os.path.join(ai_dir, 'SDK')
    if os.path.exists(sdk_dir):
        sys.path.insert(0, sdk_dir)
        print(f"✓ 添加 sample SDK 到路径")

    sys.path.insert(0, ai_dir)
    print(f"✓ 添加 sample 到路径")

    try:
        from ai import AI, create_agent
        print("✓ 成功导入 AI, create_agent")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        agent = create_agent()
        print(f"✓ 创建 agent 成功: {type(agent)}")
        return True
    except Exception as e:
        print(f"✗ 创建 agent 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    results = {}

    # 测试各个模型
    results['ann_593'] = test_ann_593()
    results['ann_v1'] = test_ann_v1()
    results['gen99'] = test_gen99()
    results['gen199'] = test_gen199()
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
