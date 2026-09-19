"""测试策略是否正确执行"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'saiblo-antwar-sdk-python'))

from antwar.gamestate import GameState
from tests.strategies.ann_v1 import AnnV1
from tests.strategies.gen99 import Gen99

# 创建游戏状态
game_state = GameState()
game_state.init_with_seed(0)

# 创建策略
agent1 = AnnV1()
agent2 = Gen99()

print("=== 初始状态 ===")
print(f"玩家0金币: {game_state.coin[0]}")
print(f"玩家1金币: {game_state.coin[1]}")
print(f"玩家0 HP: {game_state.hp[0]}")
print(f"玩家1 HP: {game_state.hp[1]}")
print(f"玩家0防御塔数: {len([t for t in game_state.towers if t.player == 0])}")
print(f"玩家1防御塔数: {len([t for t in game_state.towers if t.player == 1])}")

# 模拟前10回合
for round_num in range(10):
    print(f"\n=== 回合 {round_num + 1} ===")
    
    # 玩家0选择动作
    actions0 = agent1.select_action(game_state, 0)
    print(f"玩家0选择的动作数: {len(actions0)}")
    for i, action in enumerate(actions0):
        print(f"  动作{i+1}: {action}")
    
    # 应用玩家0的动作
    for op in actions0:
        game_state.apply_operation(0, op)
    
    # 玩家1选择动作
    actions1 = agent2.select_action(game_state, 1)
    print(f"玩家1选择的动作数: {len(actions1)}")
    for i, action in enumerate(actions1):
        print(f"  动作{i+1}: {action}")
    
    # 应用玩家1的动作
    for op in actions1:
        game_state.apply_operation(1, op)
    
    # 模拟下一回合
    game_state.simulate_next_round()
    
    print(f"玩家0金币: {game_state.coin[0]}")
    print(f"玩家1金币: {game_state.coin[1]}")
    print(f"玩家0 HP: {game_state.hp[0]}")
    print(f"玩家1 HP: {game_state.hp[1]}")
    print(f"玩家0防御塔数: {len([t for t in game_state.towers if t.player == 0])}")
    print(f"玩家1防御塔数: {len([t for t in game_state.towers if t.player == 1])}")
    print(f"蚂蚁总数: {len(game_state.ants)}")
