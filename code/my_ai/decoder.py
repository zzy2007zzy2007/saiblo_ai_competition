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
    MAP_PROPERTY,
    PLAYER_BASES,
    TOWER_UPGRADE_TREE,
    Terrain,
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


def _check_super_weapon_valid(state: BackendState, player: int, ch: int,
                              intent_decoding: bool = False) -> bool:
    """Check if a super weapon channel is valid.

    With intent_decoding: only check cooldown (decoder handles gold).
    Without: check both cooldown and gold (original behavior).
    """
    sw = CHANNEL_TO_SUPER_WEAPON[ch]
    if state.weapon_cooldowns[player, sw] != 0:
        return False
    if not intent_decoding and state.coins[player] < SUPER_WEAPON_STATS[sw].cost:
        return False
    return True


def make_class_mask(
    state: BackendState,
    player: int,
    *,
    position_mask: np.ndarray | None = None,
    intent_decoding: bool = False,
) -> np.ndarray:
    """Return boolean mask of shape (23,) indicating which classes are valid.

    Args:
        state: game state
        player: current player
        position_mask: optional precomputed position mask (avoids recomputation)
        intent_decoding: if True, super weapons not masked by gold cost.
    """
    if position_mask is None:
        position_mask = make_position_masks(state, player, intent_decoding=intent_decoding)

    mask = np.zeros(NUM_CLASSES, dtype=bool)

    # Tower classes 0-15: valid if there's at least one valid position
    for ch in range(16):
        mask[ch] = position_mask[ch].any()

    # Downgrade (16): valid if player has at least one tower
    mask[16] = position_mask[16].any()

    # Super weapons (17-20): check cooldown (and coins if not intent_decoding)
    for ch in range(17, 21):
        mask[ch] = _check_super_weapon_valid(state, player, ch, intent_decoding=intent_decoding)

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


def make_position_masks(state: BackendState, player: int,
                        intent_decoding: bool = False) -> np.ndarray:
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

    # Super weapons (17-20): can place at any valid (non-VOID) position
    valid_cells = np.array(MAP_PROPERTY, dtype=np.int32) != Terrain.VOID  # (19,19) bool
    for ch in range(17, 21):
        if _check_super_weapon_valid(state, player, ch, intent_decoding=intent_decoding):
            mask[ch] = valid_cells

    # Base upgrades (21-22): no position, all False

    return mask


def _sample_position(
    channel_map: np.ndarray,     # (19, 19) action_map for a class
    pos_mask: np.ndarray,        # (19, 19) legal cells
    rng: np.random.Generator | None,
    pos_temperature: float,
) -> tuple[int, int, float] | None:
    """Sample (x, y) from a masked raw-value + temperature softmax.

    Distribution is over LEGAL cells only (masked), using raw action_map
    values (no z-score).  The recorded logprob is on this same masked
    base.  Training-time new-policy logπ MUST use the same mask (stored
    in npz) and raw values to keep the PPO ratio consistent.

    Returns (x, y, logprob) — logprob is log π_pos under the masked
    distribution.  Returns None if no legal cells at all.
    """
    if not pos_mask.any():
        return None
    # Fixed-scale normalization (÷100): scale-independent like class heads
    # want, but unlike z-score it doesn't depend on data statistics.
    POS_SCALE = 100.0
    scaled = channel_map / POS_SCALE
    masked = np.where(pos_mask, scaled, -np.inf)
    if pos_temperature > 0 and rng is not None:
        logits = masked / pos_temperature
    else:
        logits = masked
    logits -= logits.max()  # numerical stability
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()           # masked distribution
    if pos_temperature > 0 and rng is not None:
        flat = rng.choice(probs.size, p=probs.ravel())
        x, y = np.unravel_index(flat, probs.shape)
    else:
        x, y = np.unravel_index(np.argmax(masked), masked.shape)
    logprob = float(np.log(probs[x, y] + 1e-12))
    return int(x), int(y), logprob


# ─── Decode one head ────────────────────────────────────────────────────────


def decode_head(
    head_logits: np.ndarray,       # (23,)
    action_map: np.ndarray,         # (23, 19, 19)
    class_mask: np.ndarray,         # (23,)
    position_mask: np.ndarray,      # (23, 19, 19)
    state: BackendState,
    player: int,
    *,
    allowed_classes: list[int] | None = None,
    rng: np.random.Generator | None = None,
    temperature: float = 0.0,
    intent_decoding: bool = False,
    sampled_class_out: list[int] | None = None,
    pos_temperature: float = 0.0,
    sampled_pos_out: list[tuple[int, int, float, np.ndarray]] | None = None,
) -> Operation | None:
    """Decode one policy head into a single Operation (or None if pass).

    Unlike the traditional "mask-then-argmax" approach, this decoder
    checks the RAW (unmasked) argmax first. If the head's top choice
    is illegal, the head is skipped (returns None) instead of falling
    back to the next-best legal action. This prevents heads from
    automatically decaying into wasteful fallback actions (e.g. DOWNGRADE)
    when their preferred action is temporarily unavailable.

    When ``intent_decoding`` is True and a super weapon (17-20) is chosen
    but gold is insufficient, the decoder auto-downgrades a tower using
    class 16's action_map instead of returning None.

    When temperature > 0, uses z-score normalized temperature sampling
    (per-head mean/std) instead of argmax for smoother action selection.

    When pos_temperature > 0, positions are sampled from a per-channel
    z-scored + temperature-softmax distribution over legal cells instead
    of argmax — this gives the position channel a gradient path in PPO.

    ``sampled_class_out`` (optional): a list that receives the sampled
    class id (the actual action this head chose), so callers can build
    on-policy log-prob targets.  Recorded for BOTH argmax and sampled
    paths, before the legality check.

    ``sampled_pos_out`` (optional): a list that receives
    (x, y, logprob, mask) for each head that selected a position-bearing
    action (super weapon / tower / downgrade).  logprob is log π_pos under
    the masked sampling distribution; mask is the legal-cell mask used for
    sampling — both needed so training can reconstruct the same distribution.
    Only appended when a position is sampled.
    """
    # Step 0: Filter by allowed_classes if set
    if allowed_classes is not None:
        restricted = np.ones(len(class_mask), dtype=bool)
        restricted[allowed_classes] = True
        class_mask = class_mask & restricted
        for ch in range(len(position_mask)):
            if ch not in allowed_classes:
                position_mask[ch] = False

    # Step 1: Choose class (argmax or temperature sampling)
    if temperature > 0 and rng is not None:
        # z-score normalize so temperature is scale-invariant (per-head)
        mean = head_logits.mean()
        std = head_logits.std() + 1e-8
        logits = (head_logits - mean) / std
        probs = np.exp(logits / temperature)
        probs /= probs.sum()
        class_id = int(rng.choice(len(probs), p=probs))
    else:
        class_id = int(np.argmax(head_logits))

    if sampled_class_out is not None:
        sampled_class_out.append(class_id)

    if not class_mask[class_id]:
        return None  # Head's top choice is illegal → skip this head

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
        cost = SUPER_WEAPON_STATS[sw_type].cost

        # Intent decoding: gold insufficient → downgrade tower
        if intent_decoding and state.coins[player] < cost:
            dg_map = action_map[16]
            dg_mask = position_mask[16]
            if dg_mask.any():
                masked = np.where(dg_mask, dg_map, -np.inf)
                x, y = np.unravel_index(np.argmax(masked), masked.shape)
                tower = state.tower_at(int(x), int(y))
                if tower is not None and tower.player == player:
                    return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            return None

        pos_mask = position_mask[class_id]
        pos = _sample_position(action_map[class_id], pos_mask, rng, pos_temperature)
        if pos is None:
            return None
        x, y, logprob = pos
        if sampled_pos_out is not None:
            sampled_pos_out.append((x, y, logprob, pos_mask.copy()))
        return Operation(op_type, x, y)

    # Step 4: Tower actions (classes 0-15)
    if 0 <= class_id <= 15:
        pos_mask = position_mask[class_id]
        pos = _sample_position(action_map[class_id], pos_mask, rng, pos_temperature)
        if pos is None:
            return None
        x, y, logprob = pos
        if sampled_pos_out is not None:
            sampled_pos_out.append((x, y, logprob, pos_mask.copy()))
        return _decode_tower_action(state, player, class_id, x, y)

    # Step 5: Downgrade (class 16)
    if class_id == 16:
        pos_mask = position_mask[16]
        pos = _sample_position(action_map[16], pos_mask, rng, pos_temperature)
        if pos is None:
            return None
        x, y, logprob = pos
        if sampled_pos_out is not None:
            sampled_pos_out.append((x, y, logprob, pos_mask.copy()))
        tower = state.tower_at(x, y)
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
    *,
    allowed_classes: list[int] | None = None,
    rng: np.random.Generator | None = None,
    temperature: float = 0.0,
    intent_decoding: bool = False,
    sampled_class_out: list[int] | None = None,
    pos_temperature: float = 0.0,
    sampled_pos_out: list[tuple[int, int, float, np.ndarray]] | None = None,
) -> list[Operation]:
    """Decode network output into a list of Operations (up to 3).

    Args:
        network_output: dict with keys:
            - action_map: (1, 23, 19, 19) or (23, 19, 19)
            - head1_logits, ..., headN_logits: (1, 23) or (23,)
            - value: (1, 1) or (1,) — ignored for decoding
        state: current game state
        player: current player (0 or 1)
        sampled_class_out: optional list — receives the sampled class id
            for each head (in head order), the ACTUAL action this head
            chose.  Same length as head_logits_list.
        pos_temperature: temperature for position sampling (>0 samples,
            else argmax).  Passed to decode_head.
        sampled_pos_out: optional list — receives (x, y, logprob, mask)
            for each head that sampled a position-bearing action.

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
    position_mask = make_position_masks(state, player, intent_decoding=intent_decoding)
    class_mask = make_class_mask(state, player, position_mask=position_mask, intent_decoding=intent_decoding)

    # Decode each head (up to 3 operations)
    operations: list[Operation] = []
    for head_idx, head_logits in enumerate(head_logits_list):
        op = decode_head(head_logits, action_map, class_mask, position_mask, state, player,
                         allowed_classes=allowed_classes, rng=rng, temperature=temperature,
                         intent_decoding=intent_decoding,
                         sampled_class_out=sampled_class_out,
                         pos_temperature=pos_temperature,
                         sampled_pos_out=sampled_pos_out)
        if op is not None:
            # Check if operation is legal (given already selected operations)
            if state.can_apply_operation(player, op, operations):
                operations.append(op)
            # Update masks for remaining heads if needed
            # (optional: invalidate recently used positions)
            if op.op_type == OperationType.BUILD_TOWER:
                position_mask[:, op.arg0, op.arg1] = False

    return operations
