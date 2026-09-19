#!/usr/bin/env python3
"""模型加载器"""

import os
import sys
import torch

class ModelLoader:
    def __init__(self):
        self.loaded_models = {}
    
    def load_model(self, model_config):
        """加载单个模型"""
        name = model_config['name']
        path = model_config['path']
        model_file = model_config['model_file']
        input_dim = model_config['input_dim']
        action_dim = model_config['action_dim']
        candidates = model_config['hidden_dim_candidates']
        
        # 使用 model_config 中指定的 base_path
        if 'base_path' in model_config:
            full_path = os.path.join(model_config['base_path'], path)
            model_path = os.path.join(full_path, model_file)
        else:
            # 回退到默认路径
            base_paths = [
                os.path.join(os.path.dirname(__file__), '../../..'),
                '/root/autodl-tmp/AntWar'
            ]
            full_path = None
            model_path = None
            for base_path in base_paths:
                candidate_path = os.path.join(base_path, path)
                candidate_model_path = os.path.join(candidate_path, model_file)
                if os.path.exists(candidate_model_path):
                    full_path = candidate_path
                    model_path = candidate_model_path
                    break
        
        print(f"\n=== 加载模型: {name} ===")
        print(f"模型路径: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"✗ 模型文件不存在")
            return None, None
        
        sys.path.insert(0, full_path)
        
        try:
            from neural_network import NeuralNetworkAgent
            print("✓ 成功导入 NeuralNetworkAgent")
        except Exception as e:
            print(f"✗ 导入失败: {e}")
            return None, None
        
        for hidden_dim in candidates:
            print(f"\n测试 hidden_dim={hidden_dim}:")
            try:
                agent = NeuralNetworkAgent(input_dim, action_dim, hidden_dim=hidden_dim)
                agent.load_model(model_path)
                
                if self._verify_weights(agent, model_path):
                    print(f"✓ 模型加载成功 (hidden_dim={hidden_dim})")
                    self.loaded_models[name] = {'agent': agent, 'hidden_dim': hidden_dim}
                    return agent, hidden_dim
            except Exception as e:
                print(f"✗ 加载失败: {e}")
        
        return None, None
    
    def _verify_weights(self, agent, model_path):
        """验证权重形状匹配"""
        try:
            model_state_dict = torch.load(model_path, map_location='cpu')
            network_state_dict = agent.network.state_dict()
            for key in model_state_dict.keys():
                if key in network_state_dict:
                    if model_state_dict[key].shape != network_state_dict[key].shape:
                        return False
            return True
        except Exception as e:
            return False
