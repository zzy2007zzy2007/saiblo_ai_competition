#!/usr/bin/env python3
"""
BasicRandomAI vs gen99 对战测试脚本
=====================================

测试目标：
1. BasicRandomAI 使用项目SDK的 ActionCatalog
2. gen99 使用自己SDK的 ActionCatalog  
3. 对战引擎使用项目SDK
4. 进行3局对战，输出每局结果和最终胜率
"""

import sys
import os

def run_battle_test():
    # 设置路径
    project_sdk_path = "/root/autodl-tmp/AntWar/ppo_v1/Ant-Game"
    gen99_path = "/root/autodl-tmp/AntWar/baselines/gen99"
    
    # ===== 步骤1：导入gen99的AI（使用gen99自己的SDK）=====
    # 先添加gen99路径，确保gen99使用自己的SDK
    sys.path.insert(0, gen99_path)
    from ai import AI as Gen99AI
    
    # ===== 步骤2：导入项目SDK的核心模块（用于对战引擎和BasicRandomAI）=====
    sys.path.insert(0, project_sdk_path)
    
    from SDK.backend.core import PythonBackend
    from SDK.backend.state import BackendState
    from SDK.utils.actions import ActionCatalog as ProjectActionCatalog
    from SDK.utils.features import FeatureExtractor
    
    # ===== 步骤3：定义BasicRandomAI（使用项目SDK）=====
    class BasicRandomAI:
        def __init__(self, seed=None):
            import random
            self.rng = random.Random(seed)
            self.feature_extractor = FeatureExtractor()
            self.catalog = ProjectActionCatalog(feature_extractor=self.feature_extractor)
        
        def list_bundles(self, state: BackendState, player: int) -> list:
            return self.catalog.build(state, player)
        
        def choose_bundle(self, state: BackendState, player: int) -> object:
            bundles = self.list_bundles(state, player)
            if len(bundles) <= 1:
                return bundles[0]
            pool = bundles[1:] if len(bundles) > 1 else bundles
            return self.rng.choice(pool)
    
    # ===== 步骤4：对战函数 =====
    def run_single_game(seed: int) -> int:
        """运行一局游戏，返回获胜者（0或1），None表示平局"""
        backend = PythonBackend()
        state = backend.initial_state(seed=seed)
        
        # 创建AI实例
        ai0 = BasicRandomAI(seed=seed * 2)
        ai1 = Gen99AI(seed=seed * 2 + 1)
        
        # 游戏主循环
        while not state.terminal and state.round_index < 500:
            # Player 0 (BasicRandomAI)
            bundle0 = ai0.choose_bundle(state, 0)
            state.apply_operation_list(0, bundle0.operations)
            
            # Player 1 (gen99)
            bundle1 = ai1.choose_bundle(state, 1)
            state.apply_operation_list(1, bundle1.operations)
            
            # 推进回合
            state.advance_round()
        
        return state.winner
    
    # ===== 步骤5：进行3局对战 =====
    results = []
    seeds = [42, 123, 456]
    
    print("=" * 60)
    print("BasicRandomAI vs gen99 对战测试")
    print("=" * 60)
    print(f"对战局数: {len(seeds)}")
    print()
    
    for i, seed in enumerate(seeds):
        print(f"第 {i+1} 局 (seed={seed})... ", end="", flush=True)
        winner = run_single_game(seed)
        
        if winner == 0:
            result_str = "BasicRandomAI 胜"
        elif winner == 1:
            result_str = "gen99 胜"
        else:
            result_str = "平局"
        
        print(result_str)
        results.append(winner)
    
    # ===== 步骤6：统计结果 =====
    print()
    print("=" * 60)
    print("对战结果统计")
    print("=" * 60)
    
    ai0_wins = sum(1 for w in results if w == 0)
    ai1_wins = sum(1 for w in results if w == 1)
    draws = sum(1 for w in results if w is None)
    
    print(f"BasicRandomAI 胜: {ai0_wins} 局")
    print(f"gen99 胜: {ai1_wins} 局")
    print(f"平局: {draws} 局")
    print()
    
    total = len(results)
    ai0_rate = (ai0_wins / total) * 100
    ai1_rate = (ai1_wins / total) * 100
    
    print(f"BasicRandomAI 胜率: {ai0_rate:.1f}%")
    print(f"gen99 胜率: {ai1_rate:.1f}%")
    
    return {
        'results': results,
        'ai0_wins': ai0_wins,
        'ai1_wins': ai1_wins,
        'draws': draws,
        'ai0_win_rate': ai0_rate,
        'ai1_win_rate': ai1_rate
    }

if __name__ == "__main__":
    run_battle_test()