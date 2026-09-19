#!/usr/bin/env python3
"""
测试 baseline 模型使用自己的 SDK

验证方案：让每个 baseline 模型使用自己的 SDK 目录，解决特征维度不匹配问题
"""

import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"基础目录: {BASE_DIR}")

class BaselineSDKTest:
    def __init__(self):
        self.test_results = {}
    
    def _get_baseline_dirs(self):
        """获取所有 baseline 模型目录"""
        baseline_base = os.path.join(BASE_DIR, 'baselines')
        if not os.path.exists(baseline_base):
            print(f"✗ baseline 目录不存在: {baseline_base}")
            return []
        
        dirs = []
        for name in os.listdir(baseline_base):
            full_path = os.path.join(baseline_base, name)
            if os.path.isdir(full_path):
                sdk_path = os.path.join(full_path, 'SDK')
                if os.path.exists(sdk_path):
                    dirs.append({
                        'name': name,
                        'path': full_path,
                        'sdk_path': sdk_path
                    })
        return dirs
    
    def _load_agent_with_own_sdk(self, baseline_info):
        """使用模型自己的 SDK 加载 agent"""
        name = baseline_info['name']
        base_path = baseline_info['path']
        sdk_path = baseline_info['sdk_path']
        
        print(f"\n=== 测试模型: {name} ===")
        print(f"模型路径: {base_path}")
        print(f"SDK路径: {sdk_path}")
        
        original_sys_path = sys.path.copy()
        
        try:
            sys.path = []
            sys.path.insert(0, sdk_path)
            sys.path.insert(0, base_path)
            
            for path in original_sys_path:
                if 'site-packages' in path:
                    sys.path.append(path)
            
            if os.path.exists(os.path.join(base_path, 'ai.py')):
                from ai import AI
                agent = AI()
                print("✓ 成功加载 AI 类")
                
                if hasattr(agent, 'choose_operations'):
                    print("✓ AI 类有 choose_operations 方法")
                if hasattr(agent, 'choose_bundle'):
                    print("✓ AI 类有 choose_bundle 方法")
                
                return agent, None
            elif os.path.exists(os.path.join(base_path, 'main.py')):
                from main import create_agent
                agent = create_agent()
                print("✓ 成功通过 create_agent() 创建 agent")
                return agent, None
            else:
                return None, "找不到 ai.py 或 main.py"
                
        except Exception as e:
            import traceback
            error_msg = f"加载失败: {e}\n{traceback.format_exc()}"
            return None, error_msg
        finally:
            sys.path = original_sys_path
    
    def _test_agent_feature_dimension(self, agent, baseline_info):
        """测试 agent 的特征提取维度"""
        name = baseline_info['name']
        sdk_path = baseline_info['sdk_path']
        
        original_sys_path = sys.path.copy()
        
        try:
            sys.path = []
            sys.path.insert(0, sdk_path)
            
            for path in original_sys_path:
                if 'site-packages' in path or 'saiblo-antwar-sdk' in path.lower():
                    sys.path.append(path)
            
            from SDK.utils.features import FeatureExtractor
            
            backend_paths = [
                '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python',
                os.path.join(BASE_DIR, 'saiblo-antwar-sdk-python')
            ]
            for bp in backend_paths:
                if os.path.exists(bp):
                    sys.path.insert(0, bp)
                    break
            
            from SDK.backend.core import load_backend
            from SDK.backend.state import PythonBackendState
            
            backend = load_backend(prefer_native=False)
            game_state = backend.initial_state(seed=0)
            state = PythonBackendState(game_state)
            
            feature_extractor = FeatureExtractor()
            action_mask = [1] * 32
            
            observation = feature_extractor.encode_observation(state, 0, action_mask)
            flattened = feature_extractor.flatten_observation(observation)
            
            if isinstance(flattened, list):
                flattened = [float(x) for x in flattened]
            
            dim = len(flattened) if hasattr(flattened, '__len__') else -1
            
            print(f"✓ 特征维度: {dim}")
            return dim
            
        except Exception as e:
            import traceback
            print(f"✗ 特征提取失败: {e}\n{traceback.format_exc()}")
            return None
        finally:
            sys.path = original_sys_path
    
    def run_test(self):
        """运行测试"""
        print("=" * 70)
        print("Baseline 模型使用自己 SDK 的测试")
        print("=" * 70)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        baseline_dirs = self._get_baseline_dirs()
        
        if not baseline_dirs:
            print("✗ 没有找到 baseline 模型目录")
            return
        
        print(f"找到 {len(baseline_dirs)} 个 baseline 模型:")
        for info in baseline_dirs:
            print(f"  - {info['name']}: {info['path']}")
        print()
        
        for baseline_info in baseline_dirs:
            name = baseline_info['name']
            
            agent, error = self._load_agent_with_own_sdk(baseline_info)
            
            if agent is None:
                self.test_results[name] = {
                    'success': False,
                    'error': error,
                    'feature_dim': None
                }
                print(f"✗ {name} 加载失败")
                continue
            
            feature_dim = self._test_agent_feature_dimension(agent, baseline_info)
            
            self.test_results[name] = {
                'success': True,
                'error': None,
                'feature_dim': feature_dim
            }
            
            print(f"✓ {name} 测试完成")
        
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)
        
        for name, result in self.test_results.items():
            if result['success']:
                status = "✓ 成功"
                dim_info = f", 特征维度: {result['feature_dim']}" if result['feature_dim'] else ""
                print(f"{name}: {status}{dim_info}")
            else:
                print(f"{name}: ✗ 失败 - {result['error']}")
        
        print("\n测试完成！")

if __name__ == "__main__":
    test = BaselineSDKTest()
    test.run_test()