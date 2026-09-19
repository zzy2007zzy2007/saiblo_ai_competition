#!/usr/bin/env python3
import sys
import os
import cloudpickle

# 模拟 worker 的过程
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Ant-Game')))

from ppo_antwar.league.baseline_agent import load_agent_via_subprocess

BASE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/baselines"
gen99_dir = os.path.join(BASE_DIR, "gen99")

print("=== Testing load_agent_via_subprocess ===")
serialized = load_agent_via_subprocess(gen99_dir, "Gen99")
print(f"✓ 序列化成功, 长度 = {len(serialized)}")

print("\n=== 现在模拟 worker 的 sys.path 设置 ===")

original_sys_path = sys.path.copy()
original_common_module = sys.modules.get('common')

# 设置 sys.path
sys.path = [gen99_dir]
sdk_dir = os.path.join(gen99_dir, 'SDK')
if os.path.exists(sdk_dir):
    sys.path.insert(0, sdk_dir)

for path in original_sys_path:
    if path not in sys.path:
        sys.path.append(path)

print(f"sys.path = {sys.path}")

# 现在反序列化 agent
print("\n=== 反序列化 agent ===")
agent = cloudpickle.loads(serialized)
print("✓ 反序列化成功！")

print("\n=== 现在测试 agent 的方法调用 ===")
# 模拟一个最简单的 state
from SDK.backend.state import BackendState
from SDK.backend.model import Player

# 创建简单的 state
state = BackendState()
p0 = Player(player_id=0)
p1 = Player(player_id=1)
state.players = [p0, p1]

print("尝试调用 choose_bundle()...")
bundle = agent.choose_bundle(state, 1)
print(f"✓ choose_bundle() 成功！返回: {type(bundle)}, len = {len(bundle.operations) if hasattr(bundle, 'operations') else 'N/A'}")
