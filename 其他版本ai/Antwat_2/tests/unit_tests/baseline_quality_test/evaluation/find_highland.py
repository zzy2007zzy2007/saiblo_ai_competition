"""查找正确的高台位置"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'saiblo-antwar-sdk-python'))

from antwar.coord import Coord, is_player_highland

print("=== 玩家0的高台位置 ===")
for x in range(19):
    for y in range(19):
        c = Coord(x, y)
        if is_player_highland(c, 0):
            print(f"  ({x}, {y})")

print("\n=== 玩家1的高台位置 ===")
for x in range(19):
    for y in range(19):
        c = Coord(x, y)
        if is_player_highland(c, 1):
            print(f"  ({x}, {y})")
