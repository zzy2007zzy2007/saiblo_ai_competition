#!/usr/bin/env python3
"""
测试 Ann593 模型加载
"""

import os
import sys
import torch

print("=" * 60)
print("测试 Ann593 模型加载")
print("=" * 60)

# 测试路径
base_path = '/root/autodl-tmp/AntWar'
ann_593_path = os.path.join(base_path, 'baselines/ann_593')
model_path = os.path.join(ann_593_path, 'model.pth')

print(f"\n检查路径:")
print(f"  ann_593_path: {ann_593_path}")
print(f"  存在: {os.path.exists(ann_593_path)}")
print(f"  model.pth: {os.path.exists(model_path)}")
if os.path.exists(ann_593_path):
    print(f"\nann_593 目录内容:")
    print("\n".join(os.listdir(ann_593_path)))

# 添加路径
sys.path.insert(0, ann_593_path)
print(f"\n添加 sys.path: {ann_593_path}")

print(f"\n尝试导入 neural_network 模块...")
try:
    from neural_network import NeuralNetworkAgent
    print("✓ 成功导入 NeuralNetworkAgent")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 尝试加载模型
print(f"\n尝试加载模型...")
try:
    agent = NeuralNetworkAgent(input_dim=10144, action_dim=32, hidden_dim=256)
    agent.load_model(model_path)
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试成功！")
print("=" * 60)
