#!/usr/bin/env python3
"""
测试AnnV1模型加载
重点测试如何正确加载AnnV1模型
"""

import os
import sys
import torch

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../baselines/ann_v1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../baselines/ann_593'))

from neural_network import NeuralNetworkAgent

def check_model_loaded_correctly(model_state_dict):
    """检查模型状态字典的结构"""
    print("    模型状态字典中的键:")
    for key, value in model_state_dict.items():
        print(f"      {key}: {value.shape}")

def load_and_test_ann_v1_model():
    """加载并测试AnnV1模型"""
    print("=" * 60)
    print("测试AnnV1模型加载")
    print("=" * 60)

    model_path = os.path.join(os.path.dirname(__file__), '../../../baselines/ann_v1/model.pth')
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print("错误: 模型文件不存在！")
        return None

    # 首先加载模型状态字典，分析其结构
    print("\n分析模型文件结构:")
    model_state_dict = torch.load(model_path, map_location='cpu')
    check_model_loaded_correctly(model_state_dict)

    # 推断正确的hidden_dim
    # shared_layers.0.weight 的形状是 [hidden_dim, input_dim]
    if 'shared_layers.0.weight' in model_state_dict:
        weight_shape = model_state_dict['shared_layers.0.weight'].shape
        print(f"\n推断hidden_dim:")
        print(f"  shared_layers.0.weight 形状: {weight_shape}")
        if len(weight_shape) == 2:
            inferred_hidden_dim = weight_shape[0]
            inferred_input_dim = weight_shape[1]
            print(f"  推断的 hidden_dim: {inferred_hidden_dim}")
            print(f"  推断的 input_dim: {inferred_input_dim}")

    # 测试不同的hidden_dim参数
    test_hidden_dims = [64, 128, 256, 512]

    correct_hidden_dim = None
    for hidden_dim in test_hidden_dims:
        print(f"\n测试 hidden_dim={hidden_dim}:")

        # 创建网络
        agent = NeuralNetworkAgent(10144, 32, hidden_dim=hidden_dim)
        print(f"  ✓ 成功创建NeuralNetworkAgent (hidden_dim={hidden_dim})")

        # 获取网络的状态字典
        network_state_dict = agent.network.state_dict()
        print(f"  网络状态字典中的键:")
        for key, value in network_state_dict.items():
            print(f"    {key}: {value.shape}")

        # 尝试加载模型
        try:
            agent.load_model(model_path)
            print(f"  ✓ load_model() 调用成功（使用了 strict=False）")
        except Exception as e:
            print(f"  ✗ load_model() 调用失败: {e}")

        # 真正检查：对比权重形状
        print(f"\n  对比关键权重形状:")
        for key in ['shared_layers.0.weight', 'actor_head.0.weight', 'critic_head.0.weight']:
            if key in model_state_dict and key in network_state_dict:
                model_shape = model_state_dict[key].shape
                network_shape = network_state_dict[key].shape
                match = "✓ 匹配" if model_shape == network_shape else "✗ 不匹配"
                print(f"    {key}:")
                print(f"      模型文件: {model_shape}")
                print(f"      网络:     {network_shape}")
                print(f"      {match}")

        # 检查是否所有权重都匹配
        all_match = True
        for key in model_state_dict.keys():
            if key in network_state_dict:
                if model_state_dict[key].shape != network_state_dict[key].shape:
                    all_match = False
                    break

        if all_match:
            print(f"\n  ★ 所有权重形状都匹配！hidden_dim={hidden_dim} 是正确的！")
            correct_hidden_dim = hidden_dim
            break
        else:
            print(f"\n  ✗ 存在权重形状不匹配")

    return correct_hidden_dim

def main():
    print("开始AnnV1模型加载测试...")
    print()

    # 测试: 找出正确的hidden_dim参数
    correct_hidden_dim = load_and_test_ann_v1_model()

    if correct_hidden_dim is None:
        print("\n错误: 无法确定正确的hidden_dim参数！")
        return

    print("\n" + "=" * 60)
    print(f"测试完成！确定的正确hidden_dim参数: {correct_hidden_dim}")
    print("=" * 60)

if __name__ == "__main__":
    main()