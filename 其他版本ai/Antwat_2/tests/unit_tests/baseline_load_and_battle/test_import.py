#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')

import battle_simulator
print(f"battle_simulator loaded OK")
print(f"MAX_ROUNDS = {battle_simulator.DEFAULT_MAX_ROUNDS}")
print(f"LOG_LEVEL_DEBUG = {battle_simulator.LOG_LEVEL_DEBUG}")
print(f"BATTLE_PRINT_INTERVAL = {battle_simulator.BATTLE_PRINT_INTERVAL}")