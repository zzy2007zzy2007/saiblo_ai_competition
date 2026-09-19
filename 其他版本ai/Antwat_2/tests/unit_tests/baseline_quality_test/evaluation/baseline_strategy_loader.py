"""策略加载器 - 直接加载baselines中的策略"""
import sys
import os
import importlib.util

# 添加baselines目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'baselines/ann_593'))

def load_baseline_strategy(strategy_name):
    """加载baselines中的策略
    
    Args:
        strategy_name: 策略名称（如'ann_593', 'ann_v1'等）
        
    Returns:
        BaseAgent: 策略实例
    """
    strategy_path = os.path.join(project_root, 'baselines', strategy_name, 'ai.py')
    
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"策略文件不存在: {strategy_path}")
    
    # 动态加载策略模块
    spec = importlib.util.spec_from_file_location("ai", strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ai'] = module
    spec.loader.exec_module(module)
    
    # 创建策略实例
    if hasattr(module, 'create_agent'):
        return module.create_agent()
    elif hasattr(module, 'AI'):
        return module.AI()
    else:
        raise AttributeError(f"策略模块 {strategy_name} 中没有找到 create_agent 或 AI 类")

def get_available_baseline_strategies():
    """获取所有可用的baseline策略"""
    baselines_dir = os.path.join(project_root, 'baselines')
    strategies = []
    
    for item in os.listdir(baselines_dir):
        item_path = os.path.join(baselines_dir, item)
        if os.path.isdir(item_path):
            ai_path = os.path.join(item_path, 'ai.py')
            if os.path.exists(ai_path):
                strategies.append(item)
    
    return strategies

if __name__ == '__main__':
    # 测试加载策略
    print("可用的baseline策略:")
    strategies = get_available_baseline_strategies()
    for strategy in strategies:
        print(f"  - {strategy}")
    
    # 测试加载ann_593
    print("\n测试加载ann_593:")
    agent = load_baseline_strategy('ann_593')
    print(f"成功加载: {agent.__class__.__name__}")
