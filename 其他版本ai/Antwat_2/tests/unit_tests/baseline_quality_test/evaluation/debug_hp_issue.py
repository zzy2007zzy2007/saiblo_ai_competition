"""测试脚本：调查HP一致性问题"""
import sys
import os

# 添加SDK目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'saiblo-antwar-sdk-python'))

from antwar.gamestate import GameState
from antwar.pheromone import generate_init_pheromone

# 添加策略目录到路径
sys.path.insert(0, os.path.join(project_root, 'tests/strategies'))
from ann_v1 import AnnV1
from gen99 import Gen99

def test_pheromone_symmetry():
    """测试信息素对称性"""
    print("=== 测试信息素对称性 ===")
    
    for seed in [0, 1, 2]:
        p0, p1 = generate_init_pheromone(seed)
        
        # 检查信息素是否对称
        is_symmetric = True
        for i in range(19):
            for j in range(19):
                # 检查是否中心对称
                if abs(p0.value[i][j] - p1.value[18-i][18-j]) > 0.001:
                    is_symmetric = False
                    break
            if not is_symmetric:
                break
        
        print(f"Seed {seed}: 信息素对称性 = {is_symmetric}")
        
        # 打印部分信息素值
        print(f"  p0[0][0] = {p0.value[0][0]:.6f}, p1[0][0] = {p1.value[0][0]:.6f}")
        print(f"  p0[9][9] = {p0.value[9][9]:.6f}, p1[9][9] = {p1.value[9][9]:.6f}")
        print(f"  p0[18][18] = {p0.value[18][18]:.6f}, p1[18][18] = {p1.value[18][18]:.6f}")

def test_game_simulation():
    """测试游戏模拟"""
    print("\n=== 测试游戏模拟 ===")
    
    agent1 = AnnV1()
    agent2 = Gen99()
    
    for episode in [0, 1]:
        print(f"\n--- Episode {episode} ---")
        
        game_state = GameState()
        game_state.init_with_seed(episode)
        
        print(f"初始HP: {game_state.hp[0]}, {game_state.hp[1]}")
        print(f"初始金币: {game_state.coin[0]}, {game_state.coin[1]}")
        print(f"初始蚂蚁数: {len(game_state.ants)}")
        print(f"初始防御塔数: {len(game_state.towers)}")
        
        # 模拟前50回合
        for round_num in range(1, 51):
            # 先手智能体选择动作
            actions1 = agent1.select_action(game_state, 0)
            if actions1:
                print(f"回合 {round_num} - {agent1.name} 动作数: {len(actions1)}, 金币: {game_state.coin[0]}")
            for op in actions1:
                result = game_state.apply_operation(0, op)
                print(f"  执行操作: {op.type}, 结果: {result}")
            
            # 后手智能体选择动作
            actions2 = agent2.select_action(game_state, 1)
            if actions2:
                print(f"回合 {round_num} - {agent2.name} 动作数: {len(actions2)}, 金币: {game_state.coin[1]}")
            for op in actions2:
                result = game_state.apply_operation(1, op)
                print(f"  执行操作: {op.type}, 结果: {result}")
            
            # 模拟下一回合
            game_state.simulate_next_round()
            
            print(f"回合 {round_num} 结束: HP={game_state.hp[0]}, {game_state.hp[1]}, 金币={game_state.coin[0]}, {game_state.coin[1]}, 蚂蚁数={len(game_state.ants)}, 防御塔数={len(game_state.towers)}")

if __name__ == "__main__":
    test_pheromone_symmetry()
    test_game_simulation()
