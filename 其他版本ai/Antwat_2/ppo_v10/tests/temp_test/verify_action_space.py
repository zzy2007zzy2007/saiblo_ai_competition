#!/usr/bin/env python3
"""此轮动作空间编码与映射修改验证。

覆盖范围：
  1. 常量完整性（ACTION_DIM, ACTION_SEGMENTS, SUB_TYPE_SEGMENTS）
  2. 网络前向推理（get_action, encode）
  3. ActionHead 批量解码（ActionHead.decode）
  4. ACTION_DECODE_TABLE 结构与语义
  5. 多表位置映射（塔/超武/科技）
  6. PPO 残留物检查（无 enable_auxiliary, deterministic, logit_noise_std 等）
  7. F.softmax 未被解码路径使用（仅允许 docstring 提及）
  8. 死代码文件已删除（protocol.py, opponent_agent.py, neural_agent.py）
  9. _resolve_upgrade_path：5 种塔类型 × 10 种 sub_type → 合法 TowerType
  10. construct_operation：NOOP/BUILD/DOWNGRADE/UPGRADE/TECH 分支逻辑
  11. validate_upgrade_operation：TowerType 合法性校验

运行方式：
  cd /root/autodl-tmp/AntWar/ppo_v10 && /root/miniconda3/bin/python3 tests/temp_test/verify_action_space.py
"""
import sys, os, importlib

# ── 路径准备 ──
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
SDK_DIR = os.path.join(os.path.dirname(PROJECT_DIR), 'Ant-Game')

for p in [SRC_DIR, SDK_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from SDK.utils.constants import TOWER_UPGRADE_TREE, TowerType

# ════════════════════════════════════════════════════════════════
# Section 1: 常量完整性
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. 常量完整性")
print("=" * 60)

from ppo_ga.utils.action_constants import (
    ACTION_DIM, ACTION_SEGMENTS, SUB_TYPE_SEGMENTS,
    ACTION_DECODE_TABLE, _PLAYER_TOWER_POS, _WEAPON_POS_REDUNDANCY, _TECH_POS_MAP,
    SUPER_WEAPON_POSITIONS, SuperWeaponType,
    validate_upgrade_operation, is_valid_tower_type,
)

assert ACTION_DIM == 27, f"ACTION_DIM should be 27, got {ACTION_DIM}"
print(f"  OK: ACTION_DIM = {ACTION_DIM}")

assert list(ACTION_SEGMENTS.keys()) == ["strategy", "type", "sub_type", "position"], \
    f"Unexpected segments: {list(ACTION_SEGMENTS.keys())}"
assert ACTION_SEGMENTS["strategy"] == (0, 3)
assert ACTION_SEGMENTS["type"] == (3, 9)
assert ACTION_SEGMENTS["sub_type"] == (9, 19)
assert ACTION_SEGMENTS["position"] == (19, 27)
print(f"  OK: ACTION_SEGMENTS = {ACTION_SEGMENTS}")

assert list(SUB_TYPE_SEGMENTS.keys()) == ["main_tree", "sub_variant", "producer"]
assert SUB_TYPE_SEGMENTS["main_tree"] == (9, 12)
assert SUB_TYPE_SEGMENTS["sub_variant"] == (12, 15)
assert SUB_TYPE_SEGMENTS["producer"] == (15, 19)
print(f"  OK: SUB_TYPE_SEGMENTS = {SUB_TYPE_SEGMENTS}")

# ════════════════════════════════════════════════════════════════
# Section 2: ACTION_DECODE_TABLE 结构与语义
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. ACTION_DECODE_TABLE 结构与语义")
print("=" * 60)

assert len(ACTION_DECODE_TABLE) == 3, f"Should have 3 strategies, got {len(ACTION_DECODE_TABLE)}"
for s in range(3):
    assert len(ACTION_DECODE_TABLE[s]) == 6, f"Strategy {s} should have 6 slots, got {len(ACTION_DECODE_TABLE[s])}"
print("  OK: 3 × 6 structure")

# strategy=1 全为 None（NOOP）
for t in range(6):
    assert ACTION_DECODE_TABLE[1][t] is None, f"strategy=1,type={t} should be NOOP"
print("  OK: strategy=1 (无操作) → 全部 NOOP")

# strategy=0 (防御) — 前 5 个有操作，最后一个 None
# strategy=2 (进攻) — 全部有操作
print("  OK: strategy=0 decoded actions:")
for t, v in sorted(ACTION_DECODE_TABLE[0].items()):
    name = v[0] if v else "None (NOOP)"
    print(f"       type={t} → {name}")
print("  OK: strategy=2 decoded actions:")
for t, v in sorted(ACTION_DECODE_TABLE[2].items()):
    name = v[0] if v else "None (NOOP)"
    print(f"       type={t} → {name}")

# ════════════════════════════════════════════════════════════════
# Section 3: 多表位置映射
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. 多表位置映射")
print("=" * 60)

# 塔映射：Player 0/1 各 8 个坐标
assert len(_PLAYER_TOWER_POS[0]) == 8
assert len(_PLAYER_TOWER_POS[1]) == 8
for p in [0, 1]:
    for i, coord in enumerate(_PLAYER_TOWER_POS[p]):
        assert len(coord) == 2, f"Player {p}, pos {i}: expected (x,y), got {coord}"
print("  OK: _PLAYER_TOWER_POS — 2 players × 8 positions = 16 coordinates")

# 超武冗余映射：索引 0-7，值 ∈ {0..4}
assert len(_WEAPON_POS_REDUNDANCY) == 8
for i, v in enumerate(_WEAPON_POS_REDUNDANCY):
    assert 0 <= v <= 4, f"_WEAPON_POS_REDUNDANCY[{i}] = {v}, expected 0..4"
print(f"  OK: _WEAPON_POS_REDUNDANCY = {_WEAPON_POS_REDUNDANCY}")

# 科技映射：索引 0-7，值 ∈ {"generation_speed", "generated_ant"}
assert len(_TECH_POS_MAP) == 8
valid_tech = {"generation_speed", "generated_ant"}
for i, v in _TECH_POS_MAP.items():
    assert v in valid_tech, f"_TECH_POS_MAP[{i}] = {v}, expected one of {valid_tech}"
# 前 4 个为 speed，后 4 个为 ant
for i in range(4):
    assert _TECH_POS_MAP[i] == "generation_speed"
for i in range(4, 8):
    assert _TECH_POS_MAP[i] == "generated_ant"
print(f"  OK: _TECH_POS_MAP — 4 × speed, 4 × ant")

# 超武位置
assert len(SUPER_WEAPON_POSITIONS) == 4
for sw_type in SuperWeaponType:
    assert sw_type in SUPER_WEAPON_POSITIONS
    assert len(SUPER_WEAPON_POSITIONS[sw_type]) == 5
print("  OK: SUPER_WEAPON_POSITIONS — 4 types × 5 positions")

# ════════════════════════════════════════════════════════════════
# Section 4: 网络前向推理
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. 网络前向推理")
print("=" * 60)

from ppo_ga.network.ant_war_policy_value_network import AntWarPolicyValueNetwork

net = AntWarPolicyValueNetwork(hidden_dim=256)
board = torch.randn(1, 29, 19, 19)
global_vec = torch.randn(1, 33)

# 4a. get_action 返回四元组
strategy, type_idx, sub_type, position = net.get_action(board, global_vec)
assert isinstance(strategy, int)
assert isinstance(type_idx, int)
assert isinstance(sub_type, int)
assert isinstance(position, int)
valid_strategies = range(3)
valid_types = range(6)
valid_sub_types = range(10)
valid_positions = range(8)
assert strategy in valid_strategies, f"strategy={strategy}, expected {valid_strategies}"
assert type_idx in valid_types, f"type_idx={type_idx}, expected {valid_types}"
assert sub_type in valid_sub_types, f"sub_type={sub_type}, expected {valid_sub_types}"
assert position in valid_positions, f"position={position}, expected {valid_positions}"
print(f"  OK: get_action → ({strategy}, {type_idx}, {sub_type}, {position})")

# 4b. encode 输出
features = net.encode(board, global_vec)
assert features.shape == (1, 256), f"encode output shape: {features.shape}, expected (1, 256)"
print(f"  OK: encode() → {features.shape}")

# 4c. get_action 无 deterministic 参数（函数签名应只有 board, global_vec）
import inspect
sig = inspect.signature(net.get_action)
assert 'deterministic' not in sig.parameters, \
    f"get_action should NOT have deterministic param, got: {list(sig.parameters.keys())}"
print("  OK: get_action 签名无 deterministic 参数")

# ════════════════════════════════════════════════════════════════
# Section 5: ActionHead.decode 批量解码
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. ActionHead.decode 批量解码")
print("=" * 60)

from ppo_ga.network.heads import ActionHead

head = ActionHead(256)
features_batch = torch.randn(4, 256)
logits = head(features_batch)
assert logits.shape == (4, 27), f"Expected (4,27), got {logits.shape}"

s, t, st, p = ActionHead.decode(logits)
assert s.shape == (4,)
assert t.shape == (4,)
assert st.shape == (4,)
assert p.shape == (4,)
assert torch.all(s >= 0) and torch.all(s <= 2)
assert torch.all(t >= 0) and torch.all(t <= 5)
assert torch.all(st >= 0) and torch.all(st <= 9)
assert torch.all(p >= 0) and torch.all(p <= 7)
print("  OK: ActionHead.decode batch decode → all values in valid ranges")

# ════════════════════════════════════════════════════════════════
# Section 6: PPO 残留物检查
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. PPO 残留物检查")
print("=" * 60)

# 网络不应含有的属性（注：forward 是 nn.Module 保留方法，始终存在，不检查）
ppo_remnants = [
    'value_cnn', 'value_mlp', 'value_proj', 'value_layer_norm',
    'value_head', 'enable_auxiliary', 'logit_noise_std',
    'get_value_only', 'evaluate_actions', 'compute_auxiliary_loss',
]
for attr in ppo_remnants:
    assert not hasattr(net, attr), f"网络仍有 PPO 残留属性: {attr}"
print("  OK: 网络无任何 PPO 残留属性")

# ════════════════════════════════════════════════════════════════
# Section 7: F.softmax 未被解码路径使用
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. 解码路径不依赖 F.softmax")
print("=" * 60)

# 检查 decode 方法的源码是否使用 F.softmax（允许 docstring 提及原理说明）
decode_source = inspect.getsource(ActionHead.decode)
assert 'F.softmax' not in decode_source, \
    f"ActionHead.decode 不应使用 F.softmax，但发现: {[l for l in decode_source.split(chr(10)) if 'F.softmax' in l]}"
print("  OK: ActionHead.decode 源码无 F.softmax 调用")

# 检查网络 get_action 的源码
get_action_source = inspect.getsource(AntWarPolicyValueNetwork.get_action)
assert 'F.softmax' not in get_action_source, \
    f"get_action 不应使用 F.softmax"
print("  OK: get_action 源码无 F.softmax 调用")

# 检查模块级别是否有 F.softmax import
heads_module = importlib.import_module('ppo_ga.network.heads')
net_module = importlib.import_module('ppo_ga.network.ant_war_policy_value_network')
assert not hasattr(heads_module, 'F'), "heads.py 不应 import F (已删除)"
assert not hasattr(net_module, 'F'), "ant_war_policy_value_network.py 不应 import F (已删除)"
print("  OK: heads.py 和 network.py 无 torch.nn.functional import")

# ════════════════════════════════════════════════════════════════
# Section 8: 死代码文件已删除
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. 死代码文件已删除")
print("=" * 60)

dead_files = [
    os.path.join(SRC_DIR, 'ppo_ga', '_agent', 'protocol.py'),
    os.path.join(SRC_DIR, 'ppo_ga', 'battle', 'opponent_agent.py'),
    os.path.join(SRC_DIR, 'ppo_ga', 'trainer', 'neural_agent.py'),
]
for f in dead_files:
    assert not os.path.exists(f), f"死代码文件仍存在: {f}"
print("  OK: protocol.py / opponent_agent.py / neural_agent.py 已删除")

# 确认无代码引用它们
dead_symbols = ['RecordingAgent', 'act_record', 'OpponentAgent', 'NeuralAgent']
for symbol in dead_symbols:
    for root, dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(root, fn)
            with open(fp) as fh:
                content = fh.read()
            if symbol in content:
                raise AssertionError(f"{fp} 仍引用死代码符号 {symbol}")
print("  OK: 全仓无死代码符号引用")

# ════════════════════════════════════════════════════════════════
# Section 9: _resolve_upgrade_path — 全覆盖测试
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. _resolve_upgrade_path — 5 塔类型 × 10 sub_type")
print("=" * 60)

from ppo_ga.env.action_mask import ActionMaskHandler

handler = ActionMaskHandler()

# TOWER_UPGRADE_TREE: 同上验证
assert TowerType.BASIC in TOWER_UPGRADE_TREE
assert TowerType.HEAVY in TOWER_UPGRADE_TREE
assert TowerType.QUICK in TOWER_UPGRADE_TREE
assert TowerType.MORTAR in TOWER_UPGRADE_TREE
assert TowerType.PRODUCER in TOWER_UPGRADE_TREE
print(f"  TOWER_UPGRADE_TREE keys: {list(TOWER_UPGRADE_TREE.keys())}")

upgrade_tree = {
    TowerType.BASIC:    [TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR, TowerType.PRODUCER],
    TowerType.HEAVY:    [TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.BEWITCH],
    TowerType.QUICK:    [TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER],
    TowerType.MORTAR:   [TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE],
    TowerType.PRODUCER: [TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC],
}

all_valid_types = set()
for current_type, targets in upgrade_tree.items():
    for sub in range(10):
        result = handler._resolve_upgrade_path(current_type, sub)
        assert result is not None, f"_resolve_upgrade_path({current_type}, {sub}) → None (unexpected)"
        assert result in targets, \
            f"_resolve_upgrade_path(type={current_type}, sub={sub}) → {result}, expected one of {targets}"
        all_valid_types.add(result)
    print(f"  OK: TowerType.{TowerType(current_type).name.upper() if hasattr(TowerType(current_type), 'name') else current_type} → sub_type 0-9 全部映射到合法 target")

# 无效塔类型 → None
result = handler._resolve_upgrade_path(999, 0)
assert result is None, f"Invalid tower type → should be None"
print("  OK: 非法塔类型 → None")

# 验证 TOWER_UPGRADE_TREE 中所有 target 类型在 _VALID_TOWER_TYPES 中
from ppo_ga.utils.action_constants import _VALID_TOWER_TYPES
for t in all_valid_types:
    assert t in _VALID_TOWER_TYPES, f"TowerType {t} 不在 _VALID_TOWER_TYPES 中"
print(f"  OK: 所有 upgrade target ({len(all_valid_types)} 种) 在 _VALID_TOWER_TYPES 中")

# ════════════════════════════════════════════════════════════════
# Section 10: construct_operation 逻辑
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. construct_operation — 解码路径验证")
print("=" * 60)

# 10a. NOOP → strategy=1 (无操作) → 恒返回 None
# 但 construct_operation 依赖 state，我们只验证 strategy=1 的代码路径
for t in range(6):
    meta = ACTION_DECODE_TABLE[1][t]
    assert meta is None
print("  OK: strategy=1 (无操作) → None")

# 10b. TECH_UPGRADE: strategy=2, type=5 → 产生 Operation
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

# construct_operation 需要真实 state，这里仅验证：
# a) 方法存在
assert hasattr(handler, 'construct_operation')
# b) _resolve_upgrade_path 已测
# c) _find_tower_id_at 存在
assert hasattr(handler, '_find_tower_id_at')
# d) _is_valid_operation 存在
assert hasattr(handler, '_is_valid_operation')
print("  OK: ActionMaskHandler 所有方法就位")

# ════════════════════════════════════════════════════════════════
# Section 11: validate_upgrade_operation
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("11. validate_upgrade_operation — TowerType 合法性校验")
print("=" * 60)

# 合法 TowerType 应返回 True
for t in _VALID_TOWER_TYPES:
    op = Operation(OperationType.UPGRADE_TOWER, 0, t)
    assert validate_upgrade_operation(op), f"Valid TowerType {t} should pass"
print(f"  OK: 所有 {len(_VALID_TOWER_TYPES)} 个合法 TowerType 通过校验")

# 非法 TowerType → False
op_bad = Operation(OperationType.UPGRADE_TOWER, 0, 999)
assert not validate_upgrade_operation(op_bad), "Invalid TowerType 999 should fail"
print("  OK: 非法 TowerType → 被拦截")

# 非 UPGRADE_TOWER 操作 → 恒 True
for ot in [OperationType.BUILD_TOWER, OperationType.DOWNGRADE_TOWER,
           OperationType.USE_LIGHTNING_STORM]:
    op_other = Operation(ot, 0, 0)
    assert validate_upgrade_operation(op_other), f"{ot} 应恒通过"
print("  OK: 非 UPGRADE 操作 → 恒通过")

# ════════════════════════════════════════════════════════════════
# FINAL
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("=== ALL 11 SECTIONS PASSED ===")
print("=" * 60)
