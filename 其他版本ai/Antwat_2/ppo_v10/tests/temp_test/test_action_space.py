"""Temporary test for action space redesign verification."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Ant-Game'))

import torch
from ppo_ga.network.ant_war_policy_value_network import AntWarPolicyValueNetwork, count_parameters
from ppo_ga.network.heads import ActionHead
from ppo_ga.utils.action_constants import (
    ACTION_DIM, ACTION_DECODE_TABLE, ACTION_SEGMENTS, SUB_TYPE_SEGMENTS,
    _PLAYER_TOWER_POS, _WEAPON_POS_REDUNDANCY, _TECH_POS_MAP,
)
from ppo_ga.env.action_mask import ActionMaskHandler

# 1. Verify constants
assert ACTION_DIM == 27, f'ACTION_DIM should be 27, got {ACTION_DIM}'
print(f'OK: ACTION_DIM = {ACTION_DIM}')
print(f'OK: ACTION_SEGMENTS = {ACTION_SEGMENTS}')
print(f'OK: SUB_TYPE_SEGMENTS = {SUB_TYPE_SEGMENTS}')

# 2. Verify decode table
assert len(ACTION_DECODE_TABLE) == 3, 'Should have 3 strategies'
for s in range(3):
    assert len(ACTION_DECODE_TABLE[s]) == 6, f'Strategy {s} should have 6 type slots'
print('OK: ACTION_DECODE_TABLE structure valid')

# strategy=1 should be all NOOP
for t in range(6):
    assert ACTION_DECODE_TABLE[1][t] is None, f'strategy=1,type={t} should be NOOP'
print('OK: strategy=1 all NOOP')

# 3. Verify position mapping tables
assert len(_PLAYER_TOWER_POS[0]) == 8
assert len(_PLAYER_TOWER_POS[1]) == 8
assert len(_WEAPON_POS_REDUNDANCY) == 8
assert len(_TECH_POS_MAP) == 8
print('OK: Position mapping tables valid')

# 4. Create network
net = AntWarPolicyValueNetwork(hidden_dim=256)
params = count_parameters(net)
print(f'OK: Network created, params={params}')

# 5. Test forward pass
board = torch.randn(1, 29, 19, 19)
global_vec = torch.randn(1, 33)
strategy, type_idx, sub_type, position = net.get_action(board, global_vec)
print(f'OK: get_action output: strategy={strategy}, type_idx={type_idx}, sub_type={sub_type}, position={position}')
assert 0 <= strategy <= 2
assert 0 <= type_idx <= 5
assert 0 <= sub_type <= 9
assert 0 <= position <= 7

# 6. Test ActionHead batch decode
head = ActionHead(256)
features = torch.randn(4, 256)
logits = head(features)
assert logits.shape == (4, 27), f'Expected (4,27), got {logits.shape}'
s, t, st, p = ActionHead.decode(logits)
assert s.shape == (4,)
assert t.shape == (4,)
assert st.shape == (4,)
assert p.shape == (4,)
print(f'OK: ActionHead batch decode shapes: s={s.shape}, t={t.shape}, st={st.shape}, p={p.shape}')

# 7. Test ActionMaskHandler construct_operation (NOOP case)
handler = ActionMaskHandler()
# strategy=1 should always return None (NOOP)
# We can't test with real state without game setup, but verify method exists
assert hasattr(handler, 'construct_operation')
assert hasattr(handler, '_resolve_upgrade_path')
assert hasattr(handler, '_find_tower_id_at')
assert hasattr(handler, '_is_valid_operation')
print('OK: ActionMaskHandler has all required methods')

# 8. Verify no PPO remnants in network
assert not hasattr(net, 'value_cnn')
assert not hasattr(net, 'value_mlp')
assert not hasattr(net, 'value_proj')
assert not hasattr(net, 'value_layer_norm')
assert not hasattr(net, 'value_head')
assert not hasattr(net, 'enable_auxiliary')
assert not hasattr(net, 'logit_noise_std')
assert not hasattr(net, 'forward')
assert not hasattr(net, 'get_value_only')
assert not hasattr(net, 'evaluate_actions')
assert not hasattr(net, 'compute_auxiliary_loss')
print('OK: No PPO remnants in network')

# 9. Verify encode method exists
features = net.encode(board, global_vec)
assert features.shape == (1, 256)
print(f'OK: encode() output shape: {features.shape}')

print('\n=== ALL TESTS PASSED ===')
