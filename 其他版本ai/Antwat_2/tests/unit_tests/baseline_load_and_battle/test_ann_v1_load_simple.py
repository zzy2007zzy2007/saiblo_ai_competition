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

def load_and_test_ann_v1_model():
    """加载并测试AnnV1模型"""
    print("=" * 60)
    print("测试AnnV1模型加载")
    print("=" * 60)

    model_path = os.path.join(os.path.dirname(__file__), '../../../baselines/ann_v1/model.pth')
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")

    # 测试不同的hidden_dim参数
    test_hidden_dims = [64, 128, 256, 512]

    correct_hidden_dim = None
    for hidden_dim in test_hidden_dims:
        print(f"\n测试 hidden_dim={hidden_dim}:")
        try:
            agent = NeuralNetworkAgent(10144, 32, hidden_dim=hidden_dim)
            print("  ✓ 成功创建NeuralNetworkAgent")

            agent.load_model(model_path)
            print("  ✓ 模型加载成功！")
            print(f"  正确的hidden_dim参数是: {hidden_dim}")
            correct_hidden_dim = hidden_dim
            break
        except Exception as e:
            print(f"  ✗ 模型加载失败:")
            print(f"    错误信息: {e}")

    return correct_hidden_dim

def main():
    print("开始AnnV1模型加载测试...")
    print()

    # 测试: 找出正确的hidden_dim参数
    correct_hidden_dim = load_and_test_ann_v1_model()

    if correct_hidden_dim is None:
        print("\n错误: 无法加载AnnV1模型！")
        print("请检查模型文件是否存在以及模型结构是否正确。")
        return

    print("\n" + "=" * 60)
    print(f"测试完成！确定的正确hidden_dim参数: {correct_hidden_dim}")
    print("=" * 60)

if __name__ == "__main__":
    main()