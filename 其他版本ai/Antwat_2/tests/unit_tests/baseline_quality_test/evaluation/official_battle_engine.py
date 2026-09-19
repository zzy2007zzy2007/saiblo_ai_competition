"""基于官方SDK的对战引擎 - 直接使用baselines策略"""
import sys
import os

# 添加SDK目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'baselines/ann_593'))

from SDK.backend.core import load_backend
from SDK.utils.actions import ActionCatalog
from tests.evaluation.simple_action_catalog import SimpleActionCatalog

class OfficialBattleEngine:
    """使用官方SDK的对战引擎"""
    
    @staticmethod
    def run_battle(agent1, agent2, episodes=50, max_rounds=500, verbose=True, log_file=None):
        """运行两个智能体的对战
        
        Args:
            agent1: 先手智能体（BaseAgent类型）
            agent2: 后手智能体（BaseAgent类型）
            episodes: 对战轮数
            max_rounds: 每局最大回合数
            verbose: 是否打印详细信息
            log_file: 对战日志文件路径
            
        Returns:
            tuple: (wins1, wins2, draws, avg_rounds) - 智能体1胜场、智能体2胜场、平局数、平均回合数
        """
        wins = [0, 0]
        draws = 0
        total_rounds = 0
        
        # 打开日志文件
        log_handle = None
        if log_file:
            log_handle = open(log_file, 'w')
            log_handle.write(f"=== {agent1.__class__.__name__} vs {agent2.__class__.__name__} 对战日志 ===\n")
            log_handle.write(f"对战轮数: {episodes}\n")
            log_handle.write(f"每局最大回合数: {max_rounds}\n")
            log_handle.write("\n")
        
        # 加载官方后端
        backend = load_backend(prefer_native=False)
        
        # 创建ActionCatalog（使用简化版本，禁用一步前瞻搜索）
        catalog = SimpleActionCatalog()
        
        for episode in range(episodes):
            # 创建初始状态
            state = backend.initial_state(seed=episode)
            
            round_count = 0
            
            if log_handle:
                log_handle.write(f"=== 第 {episode} 局 ===\n")
                log_handle.write(f"初始HP: {agent1.__class__.__name__}={state.bases[0].hp}, {agent2.__class__.__name__}={state.bases[1].hp}\n")
            
            # 运行对战
            while not state.terminal and round_count < max_rounds:
                round_count += 1
                
                # 玩家0（先手）选择动作
                bundles0 = catalog.build(state, 0)
                bundle0 = agent1.choose_bundle(state, 0, bundles0)
                ops0 = list(bundle0.operations)
                
                # 玩家1（后手）选择动作
                bundles1 = catalog.build(state, 1)
                bundle1 = agent2.choose_bundle(state, 1, bundles1)
                ops1 = list(bundle1.operations)
                
                # 执行回合
                state.resolve_turn(ops0, ops1)
                
                if log_handle and round_count % 10 == 0:
                    log_handle.write(f"回合 {round_count}: {agent1.__class__.__name__} HP={state.bases[0].hp}, {agent2.__class__.__name__} HP={state.bases[1].hp}\n")
                    log_handle.flush()  # 立即刷新到磁盘
                
                # 每50回合打印一次进度
                if verbose and round_count % 50 == 0:
                    print(f"  Round {round_count}: {agent1.__class__.__name__} HP={state.bases[0].hp}, {agent2.__class__.__name__} HP={state.bases[1].hp}")
            
            # 判断胜负
            winner = None
            if state.winner == 0:
                wins[0] += 1
                winner = agent1.__class__.__name__
            elif state.winner == 1:
                wins[1] += 1
                winner = agent2.__class__.__name__
            else:
                draws += 1
                winner = "Draw"
            
            total_rounds += round_count
            
            if log_handle:
                log_handle.write(f"结果: {winner}\n")
                log_handle.write(f"最终HP: {agent1.__class__.__name__}={state.bases[0].hp}, {agent2.__class__.__name__}={state.bases[1].hp}\n")
                log_handle.write(f"死亡蚂蚁: {agent1.__class__.__name__}={state.die_count[0]}, {agent2.__class__.__name__}={state.die_count[1]}\n")
                log_handle.write(f"超级武器使用: {agent1.__class__.__name__}={state.super_weapon_usage[0]}, {agent2.__class__.__name__}={state.super_weapon_usage[1]}\n")
                log_handle.write(f"总回合数: {round_count}\n\n")
            
            if verbose and episode % 10 == 0:
                print(f"Episode {episode}: {agent1.__class__.__name__} HP={state.bases[0].hp}, {agent2.__class__.__name__}={state.bases[1].hp}, Rounds={round_count}, Result={winner}")
        
        # 关闭日志文件
        if log_handle:
            log_handle.write(f"=== 对战总结 ===\n")
            log_handle.write(f"{agent1.__class__.__name__} 胜场: {wins[0]}\n")
            log_handle.write(f"{agent2.__class__.__name__} 胜场: {wins[1]}\n")
            log_handle.write(f"平局: {draws}\n")
            log_handle.write(f"平均回合数: {total_rounds / episodes:.2f}\n")
            log_handle.close()
        
        avg_rounds = total_rounds / episodes
        return wins[0], wins[1], draws, avg_rounds
