"""测试防御塔建造"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'saiblo-antwar-sdk-python'))

from antwar.gamestate import GameState
from antwar.coord import Coord, is_player_highland
from antwar.protocol import build_tower_op
from tests.strategies.gen99 import Gen99

# 创建游戏状态
game_state = GameState()
game_state.init_with_seed(0)

# 创建策略
agent = Gen99()

print("=== 初始状态 ===")
print(f"玩家1金币: {game_state.coin[1]}")
print(f"玩家1防御塔数: {len([t for t in game_state.towers if t.player == 1])}")

# 检查建造防御塔的成本
tower_cost = game_state.build_tower_cost(1)
print(f"建造防御塔成本: {tower_cost}")

# 检查gen99策略想要建造的位置
tower_positions = [[Coord(5, 5), Coord(5, 13)], [Coord(12, 5), Coord(12, 13)]]
print(f"\ngen99策略想要建造的位置:")
for c in tower_positions[1]:
    print(f"  坐标: {c}, 是否为高台: {is_player_highland(c, 1)}")

# 手动尝试建造防御塔
print("\n=== 手动建造防御塔 ===")
coord = Coord(12, 5)
print(f"尝试在 {coord} 建造防御塔")
print(f"是否为高台: {is_player_highland(coord, 1)}")
print(f"该位置是否已有防御塔: {game_state.tower_at(coord)}")

# 尝试建造
op = build_tower_op(coord)
print(f"建造操作: {op}")

# 检查是否可以建造
result = game_state.build_tower(1, coord)
print(f"建造结果: {result}")

if result:
    print(f"建造成功！防御塔ID: {result.id}")
    print(f"防御塔位置: {result.coord}")
    print(f"防御塔玩家: {result.player}")
else:
    print("建造失败！")

print(f"\n当前防御塔数: {len([t for t in game_state.towers if t.player == 1])}")
