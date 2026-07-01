"""Decode network output (action_map + class logits) into game Operations.

The decoder:
1. Masks illegal action classes and positions based on game state
2. For each of 3 policy heads, selects (class_id, x, y) via argmax after masking
3. Maps (class_id, x, y) to a concrete Operation using the intent-level approach
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import torch

from SDK.utils.constants import (
    HIGHLAND_CELLS,
    MAP_SIZE,
    PLAYER_BASES,
    TOWER_UPGRADE_TREE,
    TowerType,
    OperationType,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    LEVEL2_TOWER_UPGRADE_COST,
    LEVEL3_TOWER_UPGRADE_COST,
    tower_build_cost_for_count,
    BASE_UPGRADE_COST,
)
from SDK.backend.state import BackendState
from SDK.backend.model import Operation

# ─── Mapping: channel index → target TowerType ────────────────────────────

CHANNEL_TO_TOWER_TYPE: dict[int, TowerType] = {
    0: TowerType.BASIC,           # Build Basic
    1: TowerType.HEAVY,           # Target: Heavy
    2: TowerType.HEAVY_PLUS,      # Target: Heavy+
    3: TowerType.ICE,             # Target: Ice
    4: TowerType.BEWITCH,         # Target: Bewitch
    5: TowerType.QUICK,           # Target: Quick
    6: TowerType.QUICK_PLUS,      # Target: Quick+
    7: TowerType.DOUBLE,          # Target: Double
    8: TowerType.SNIPER,          # Target: Sniper
    9: TowerType.MORTAR,          # Target: Mortar
    10: TowerType.MORTAR_PLUS,    # Target: Mortar+
    11: TowerType.PULSE,          # Target: Pulse
    12: TowerType.MISSILE,        # Target: Missile
    13: TowerType.PRODUCER_FAST,  # Target: Producer+ (fast)
    14: TowerType.PRODUCER_SIEGE, # Target: Siege
    15: TowerType.PRODUCER_MEDIC, # Target: Medic
}

# Map channel 16 (downgrade) is handled separately
# Map channels 17-20 (super weapons) map to SuperWeaponType
CHANNEL_TO_SUPER_WEAPON: dict[int, SuperWeaponType] = {
    17: SuperWeaponType.LIGHTNING_STORM,
    18: SuperWeaponType.EMP_BLASTER,
    19: SuperWeaponType.DEFLECTOR,
    20: SuperWeaponType.EMERGENCY_EVASION,
}

SUPER_WEAPON_TO_OP_TYPE: dict[SuperWeaponType, OperationType] = {
    SuperWeaponType.LIGHTNING_STORM: OperationType.USE_LIGHTNING_STORM,
    SuperWeaponType.EMP_BLASTER: OperationType.USE_EMP_BLASTER,
    SuperWeaponType.DEFLECTOR: OperationType.USE_DEFLECTOR,
    SuperWeaponType.EMERGENCY_EVASION: OperationType.USE_EMERGENCY_EVASION,
}

NUM_CLASSES = 24  # 0-22 action classes + 23 = HOLD

# ─── Helper: find upgrade step toward target ───────────────────────────────


def upgrade_step(current_type: TowerType, target_type: TowerType) -> TowerType | None:
    """Return the TowerType to upgrade to in one step, toward `target_type`.

    Example: current=BASIC, target=HEAVY_PLUS → returns HEAVY
             current=HEAVY, target=HEAVY_PLUS → returns HEAVY_PLUS
             current=HEAVY_PLUS, target=HEAVY_PLUS → returns None (already there)
    """
    if current_type == target_type:
        return None  # Already the target

    # Check direct upgrade
    upgrades = TOWER_UPGRADE_TREE.get(current_type, ())
    if target_type in upgrades:
        return target_type

    # Find the path: check if any direct upgrade is on the path to target
    for step in upgrades:
        sub_upgrades = TOWER_UPGRADE_TREE.get(step, ())
        if target_type in sub_upgrades or target_type == step:
            return step

    return None  # Can't reach target from current type


# ─── Per-class valid position checker ──────────────────────────────────────


def _check_super_weapon_valid(state: BackendState, player: int, ch: int) -> bool:
    """Check if a super weapon channel is valid."""
    sw = CHANNEL_TO_SUPER_WEAPON[ch]
    stats = SUPER_WEAPON_STATS[sw]
    return (
        state.weapon_cooldowns[player, sw] == 0
        and state.coins[player] >= stats.cost
    )


def make_class_mask(
    state: BackendState,
    player: int,
    *,
    position_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask of shape (23,) indicating which classes are valid.

    Args:
        state: game state
        player: current player
        position_mask: optional precomputed position mask (avoids recomputation)
    """
    if position_mask is None:
        position_mask = make_position_masks(state, player)

    mask = np.zeros(NUM_CLASSES, dtype=bool)

    # Tower classes 0-15: valid if there's at least one valid position
    for ch in range(16):
        mask[ch] = position_mask[ch].any()

    # Downgrade (16): valid if player has at least one tower
    mask[16] = position_mask[16].any()

    # Super weapons (17-20): check cooldown and coins
    for ch in range(17, 21):
        mask[ch] = _check_super_weapon_valid(state, player, ch)

    # Base upgrades (21-22)
    mask[21] = (
        state.bases[player].generation_level < 2
        and state.coins[player] >= BASE_UPGRADE_COST[state.bases[player].generation_level]
    )
    mask[22] = (
        state.bases[player].ant_level < 2
        and state.coins[player] >= BASE_UPGRADE_COST[state.bases[player].ant_level]
    )

    # HOLD (23): always valid
    mask[23] = True

    return mask


def make_position_masks(state: BackendState, player: int) -> np.ndarray:
    """Return boolean mask of shape (23, 19, 19) for each class at each position."""
    mask = np.zeros((NUM_CLASSES, MAP_SIZE, MAP_SIZE), dtype=bool)

    my_tower_positions: dict[tuple[int, int], int] = {}
    for t in state.towers_of(player):
        my_tower_positions[(t.x, t.y)] = t.tower_type

    # Tower classes 0-15: check each highland cell
    for ch in range(16):
        target_type = CHANNEL_TO_TOWER_TYPE[ch]
        for (x, y) in HIGHLAND_CELLS[player]:
            tower_type = my_tower_positions.get((x, y))
            if tower_type is None:
                # Empty highland: can build Basic
                build_cost = state.build_tower_cost(state.tower_count(player))
                mask[ch, x, y] = state.coins[player] >= build_cost
            else:
                # Tower exists: check if we can upgrade toward target
                step = upgrade_step(tower_type, target_type)
                if step is not None:
                    cost = (
                        LEVEL2_TOWER_UPGRADE_COST
                        if tower_type in (TowerType.BASIC, TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR, TowerType.PRODUCER)
                        else LEVEL3_TOWER_UPGRADE_COST
                    )
                    mask[ch, x, y] = state.coins[player] >= cost

    # Downgrade (16): any friendly tower position
    for t in state.towers_of(player):
        mask[16, t.x, t.y] = True

    # Super weapons (17-20): can place at any position (if cooldown/coins allow)
    for ch in range(17, 21):
        if _check_super_weapon_valid(state, player, ch):
            mask[ch, :, :] = True

    # Base upgrades (21-22): no position, all False

    return mask


# ─── Decode one head ────────────────────────────────────────────────────────


def decode_head(
    head_logits: np.ndarray,       # (23,)
    action_map: np.ndarray,         # (23, 19, 19)
    class_mask: np.ndarray,         # (23,)
    position_mask: np.ndarray,      # (23, 19, 19)
    state: BackendState,
    player: int,
) -> Operation | None:
    """Decode one policy head into a single Operation (or None if pass)."""
    # Step 1: Mask class logits and select class
    masked_class = np.where(class_mask, head_logits, -np.inf)
    class_id = int(np.argmax(masked_class))

    if not class_mask[class_id]:
        return None  # No valid action

    # Step 2: HOLD — do nothing this turn
    if class_id == 23:
        return None

    # Step 3: Base upgrades (no position)
    if class_id == 21:
        return Operation(OperationType.UPGRADE_GENERATION_SPEED)
    if class_id == 22:
        return Operation(OperationType.UPGRADE_GENERATED_ANT)

    # Step 4: Super weapons
    if 17 <= class_id <= 20:
        sw_type = CHANNEL_TO_SUPER_WEAPON[class_id]
        op_type = SUPER_WEAPON_TO_OP_TYPE[sw_type]
        pos_mask = position_mask[class_id]
        if not pos_mask.any():
            return None
        channel_map = action_map[class_id]
        masked_map = np.where(pos_mask, channel_map, -np.inf)
        x, y = np.unravel_index(np.argmax(masked_map), masked_map.shape)
        return Operation(op_type, int(x), int(y))

    # Step 4: Tower actions (classes 0-15)
    if 0 <= class_id <= 15:
        pos_mask = position_mask[class_id]
        if not pos_mask.any():
            return None
        channel_map = action_map[class_id]
        masked_map = np.where(pos_mask, channel_map, -np.inf)
        x, y = np.unravel_index(np.argmax(masked_map), masked_map.shape)
        return _decode_tower_action(state, player, class_id, int(x), int(y))

    # Step 5: Downgrade (class 16)
    if class_id == 16:
        pos_mask = position_mask[16]
        if not pos_mask.any():
            return None
        channel_map = action_map[16]
        masked_map = np.where(pos_mask, channel_map, -np.inf)
        x, y = np.unravel_index(np.argmax(masked_map), masked_map.shape)
        tower = state.tower_at(int(x), int(y))
        if tower is not None and tower.player == player:
            return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)

    return None


def _decode_tower_action(
    state: BackendState,
    player: int,
    class_id: int,
    x: int,
    y: int,
) -> Operation | None:
    """Decode a tower intent (build or upgrade toward target type) at (x,y)."""
    target_type = CHANNEL_TO_TOWER_TYPE[class_id]
    tower = state.tower_at(x, y)

    if tower is None:
        # Empty highland: build Basic
        return Operation(OperationType.BUILD_TOWER, x, y)

    if tower.player != player:
        return None  # Enemy tower

    # Friendly tower: try to upgrade toward target
    step = upgrade_step(tower.tower_type, target_type)
    if step is None:
        return None  # Already at target or can't upgrade

    return Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(step))


# ─── Main decode function ────────────────────────────────────────────────────


def decode_network_output(
    network_output: dict[str, torch.Tensor | np.ndarray],
    state: BackendState,
    player: int,
) -> list[Operation]:
    """Decode network output into a list of Operations (up to 3).

    Args:
        network_output: dict with keys:
            - action_map: (1, 23, 19, 19) or (23, 19, 19)
            - head1_logits, ..., headN_logits: (1, 23) or (23,)
            - value: (1, 1) or (1,) — ignored for decoding
        state: current game state
        player: current player (0 or 1)

    Returns:
        list of Operations (0-3 items, to be sent as the turn's bundle)
    """
    # Convert tensors to numpy and squeeze batch dimension
    def _to_np(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        return np.squeeze(t)  # Remove batch dim if present

    action_map = _to_np(network_output["action_map"])  # (23, 19, 19)
    head_keys = sorted(k for k in network_output if k.startswith("head") and k.endswith("_logits"))
    head_logits_list = [_to_np(network_output[k]) for k in head_keys]

    # Compute masks
    position_mask = make_position_masks(state, player)
    class_mask = make_class_mask(state, player, position_mask=position_mask)

    # Decode each head (up to 3 operations)
    operations: list[Operation] = []
    for head_idx, head_logits in enumerate(head_logits_list):
        op = decode_head(head_logits, action_map, class_mask, position_mask, state, player)
        if op is not None:
            # Check if operation is legal (given already selected operations)
            if state.can_apply_operation(player, op, operations):
                operations.append(op)
            # Update masks for remaining heads if needed
            # (optional: invalidate recently used positions)
            if op.op_type == OperationType.BUILD_TOWER:
                position_mask[:, op.arg0, op.arg1] = False

    return operations
