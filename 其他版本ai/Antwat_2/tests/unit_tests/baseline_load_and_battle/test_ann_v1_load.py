#!/usr/bin/env python3
"""
测试AnnV1模型加载
"""

import os
import sys

# 添加baselines目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../..')

def test_ann_v1_model_load():
    """测试AnnV1模型加载"""
    print("开始测试AnnV1模型加载...")
    
    # 测试不同的hidden_dim参数
    test_hidden_dims = [64, 128, 256, 512]
    model_path = os.path.join(os.path.dirname(__file__), '../../../baselines/ann_v1/model.pth')
    
    print(f"模型文件路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    
    for hidden_dim in test_hidden_dims:
        print(f"\n测试 hidden_dim={hidden_dim}:")
        try:
            # 动态导入NeuralNetworkAgent
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../baselines/ann_v1'))
            from neural_network import NeuralNetworkAgent
            
            # 创建代理
            agent = NeuralNetworkAgent(10144, 32, hidden_dim=hidden_dim)
            print("✓ 成功创建NeuralNetworkAgent")
            
            # 尝试加载模型
            agent.load_model(model_path)
            print("✓ 模型加载成功！")
            print(f"  正确的hidden_dim参数是: {hidden_dim}")
            return hidden_dim
        except Exception as e:
            print("✗ 模型加载失败:")
            print(f"  错误信息: {e}")
    
    return None

if __name__ == "__main__":
    correct_hidden_dim = test_ann_v1_model_load()
    print(f"\n测试完成！正确的hidden_dim参数: {correct_hidden_dim}")