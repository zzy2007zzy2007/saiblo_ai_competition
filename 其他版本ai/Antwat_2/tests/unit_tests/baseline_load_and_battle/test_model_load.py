#!/usr/bin/env python3
"""
测试ann_v1和ann_593模型加载的正确hidden_dim参数
"""

import os
import sys

# 使用服务器上的正确路径
antwar_dir = '/root/autodl-tmp/AntWar'

def test_model_load(model_name, model_path, network_dir):
    """测试模型加载"""
    print(f"\n=== 测试 {model_name} 模型 ===")
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    print(f"网络目录: {network_dir}")
    print(f"网络目录存在: {os.path.exists(network_dir)}")
    
    # 动态导入NeuralNetworkAgent
    sys.path.insert(0, network_dir)
    try:
        from neural_network import NeuralNetworkAgent
        print("✓ 成功导入NeuralNetworkAgent")
    except Exception as e:
        print("✗ 导入NeuralNetworkAgent失败:")
        print(f"  错误信息: {e}")
        return None
    
    # 测试不同的hidden_dim参数
    test_hidden_dims = [64, 128, 256, 512]
    
    for hidden_dim in test_hidden_dims:
        print(f"\n测试 hidden_dim={hidden_dim}:")
        try:
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
    print("开始测试模型加载...")
    
    # 测试ann_v1模型
    ann_v1_dir = os.path.join(antwar_dir, 'baselines', 'ann_v1')
    ann_v1_model_path = os.path.join(ann_v1_dir, 'model.pth')
    ann_v1_hidden_dim = test_model_load("ann_v1", ann_v1_model_path, ann_v1_dir)
    
    # 测试ann_593模型
    ann_593_dir = os.path.join(antwar_dir, 'baselines', 'ann_593')
    ann_593_model_path = os.path.join(ann_593_dir, 'model.pth')
    ann_593_hidden_dim = test_model_load("ann_593", ann_593_model_path, ann_593_dir)
    
    print("\n=== 测试结果 ===")
    print(f"ann_v1 正确的hidden_dim参数: {ann_v1_hidden_dim}")
    print(f"ann_593 正确的hidden_dim参数: {ann_593_hidden_dim}")
    print("\n测试完成！")