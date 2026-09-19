#!/usr/bin/env python3
"""测试 baseline 模型使用自己的 SDK"""

import os
import sys

BASE_DIR = '/root/autodl-tmp/AntWar'
baseline_base = os.path.join(BASE_DIR, 'baselines')

print('=' * 60)
print('Baseline SDK 测试')
print('=' * 60)

baseline_dirs = []
for name in os.listdir(baseline_base):
    full_path = os.path.join(baseline_base, name)
    if os.path.isdir(full_path):
        sdk_path = os.path.join(full_path, 'SDK')
        if os.path.exists(sdk_path):
            baseline_dirs.append({'name': name, 'path': full_path, 'sdk_path': sdk_path})

print(f'找到 {len(baseline_dirs)} 个 baseline 模型:')
for d in baseline_dirs:
    print(f'  - {d["name"]}: {d["path"]}')

for info in baseline_dirs:
    name = info['name']
    base_path = info['path']
    sdk_path = info['sdk_path']
    
    print(f'\n=== 测试 {name} ===')
    print(f'模型路径: {base_path}')
    print(f'SDK路径: {sdk_path}')
    
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
            print('✓ 成功加载 AI 类')
            
            if hasattr(agent, 'choose_operations'):
                print('✓ AI 类有 choose_operations 方法')
            if hasattr(agent, 'choose_bundle'):
                print('✓ AI 类有 choose_bundle 方法')
            
            try:
                from SDK.utils.features import FeatureExtractor
                print('✓ 成功导入 FeatureExtractor')
            except Exception as e:
                print(f'✗ 导入 FeatureExtractor 失败: {e}')
        else:
            print('✗ 找不到 ai.py')
            
    except Exception as e:
        import traceback
        print(f'✗ 加载失败: {e}')
        print(traceback.format_exc())
    finally:
        sys.path = original_sys_path

print('\n' + '=' * 60)
print('测试完成')
print('=' * 60)
