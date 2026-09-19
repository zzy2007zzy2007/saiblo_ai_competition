#!/usr/bin/env python3
"""模型注册表"""

import os

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config):
        self.models[name] = config

    def get_model_config(self, name):
        return self.models.get(name)

    def list_models(self):
        return list(self.models.keys())

    def load_default_models(self):
        """加载默认模型配置 - 支持本地和服务器环境"""
        base_paths = [
            os.path.join(os.path.dirname(__file__), '../../..'),
            '/root/autodl-tmp/AntWar'
        ]

        default_models = [
            {'name': 'ann_593', 'type': 'neural_network', 'path': 'baselines/ann_593', 'model_file': 'model.pth',
             'input_dim': 10144, 'action_dim': 32, 'hidden_dim_candidates': [256]},
            {'name': 'ann_v1', 'type': 'neural_network', 'path': 'baselines/ann_v1', 'model_file': 'model.pth',
             'input_dim': 10144, 'action_dim': 32, 'hidden_dim_candidates': [256]},
            {'name': 'gen99', 'type': 'rule_based', 'path': 'baselines/gen99',
             'class_name': 'AI', 'module_file': 'ai.py'},
            {'name': 'gen199', 'type': 'rule_based', 'path': 'baselines/gen199',
             'class_name': 'AI', 'module_file': 'ai.py'},
            {'name': 'sample', 'type': 'rule_based', 'path': 'baselines/sample',
             'class_name': 'AI', 'module_file': 'ai.py'}
        ]

        for model in default_models:
            model_found = False
            model_type = model.get('type', 'neural_network')

            if model_type == 'rule_based':
                for base_path in base_paths:
                    full_path = os.path.join(base_path, model['path'])
                    module_file = os.path.join(full_path, model['module_file'])
                    if os.path.exists(module_file):
                        model_config = model.copy()
                        model_config['base_path'] = base_path
                        self.register_model(model['name'], model_config)
                        print(f"✓ 注册规则型AI: {model['name']} ({full_path})")
                        model_found = True
                        break
            else:
                for base_path in base_paths:
                    full_path = os.path.join(base_path, model['path'])
                    model_full_path = os.path.join(full_path, model['model_file'])
                    if os.path.exists(model_full_path):
                        model_config = model.copy()
                        model_config['base_path'] = base_path
                        self.register_model(model['name'], model_config)
                        print(f"✓ 注册神经网络模型: {model['name']} ({full_path})")
                        model_found = True
                        break

            if not model_found:
                print(f"✗ 模型不存在: {model['name']}")