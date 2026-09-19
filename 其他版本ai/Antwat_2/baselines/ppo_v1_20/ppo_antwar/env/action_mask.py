from __future__ import annotations

from typing import Optional

import numpy as np

from ppo_antwar.utils.action_constants import (
    ACTION_DIM,
    TOWER_POSITIONS,
    SUPER_WEAPON_POSITIONS,
    OperationType,
    SuperWeaponType,
)
from SDK.backend.state import BackendState
from SDK.backend.model import Operation
from SDK.utils.constants import TOWER_UPGRADE_TREE


class ActionMaskHandler:
    def __init__(self) -> None:
        self.action_dim = ACTION_DIM

    def get_action_mask(self, state: BackendState, player: int) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.float32)
        mask[0] = 1.0
        for action_id in range(1, self.action_dim):
            if self._action_id_to_op(action_id, state, player) is not None:
                mask[action_id] = 1.0
        return mask

    def action_id_to_operation(self, action_id: int, state: BackendState, player: int) -> Operation | None:
        if action_id == 0:
            return None
        return self._action_id_to_op(action_id, state, player)

    def _action_id_to_op(self, action_id: int, state: BackendState, player: int) -> Operation | None:
        idx = action_id

        if 1 <= idx <= 10:
            x, y = TOWER_POSITIONS[idx - 1][0], TOWER_POSITIONS[idx - 1][1]
            op = Operation(OperationType.BUILD_TOWER, x, y)
            return op if self._is_legal(state, player, op) else None

        idx -= 10

        if 1 <= idx <= 40:
            pos_idx = (idx - 1) // 4
            dir_idx = (idx - 1) % 4
            x, y = TOWER_POSITIONS[pos_idx][0], TOWER_POSITIONS[pos_idx][1]
            tower_id = self._find_tower_id_at(state, player, x, y)
            if tower_id is None:
                return None
            tower = state.tower_by_id(tower_id)
            if tower is None:
                return None
            targets = TOWER_UPGRADE_TREE.get(tower.tower_type, ())
            if dir_idx >= len(targets):
                return None
            op = Operation(OperationType.UPGRADE_TOWER, tower_id, int(targets[dir_idx]))
            return op if self._is_legal(state, player, op) else None

        idx -= 40

        if 1 <= idx <= 10:
            x, y = TOWER_POSITIONS[idx - 1][0], TOWER_POSITIONS[idx - 1][1]
            tower_id = self._find_tower_id_at(state, player, x, y)
            if tower_id is None:
                return None
            op = Operation(OperationType.DOWNGRADE_TOWER, tower_id)
            return op if self._is_legal(state, player, op) else None

        idx -= 10

        if 1 <= idx <= 5:
            positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.LIGHTNING_STORM]
            x, y = positions[idx - 1]
            op = Operation(OperationType.USE_LIGHTNING_STORM, x, y)
            return op if self._is_legal(state, player, op) else None

        idx -= 5

        if 1 <= idx <= 5:
            positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.EMP_BLASTER]
            x, y = positions[idx - 1]
            op = Operation(OperationType.USE_EMP_BLASTER, x, y)
            return op if self._is_legal(state, player, op) else None

        idx -= 5

        if 1 <= idx <= 5:
            positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.DEFLECTOR]
            x, y = positions[idx - 1]
            op = Operation(OperationType.USE_DEFLECTOR, x, y)
            return op if self._is_legal(state, player, op) else None

        idx -= 5

        if 1 <= idx <= 5:
            positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.EMERGENCY_EVASION]
            x, y = positions[idx - 1]
            op = Operation(OperationType.USE_EMERGENCY_EVASION, x, y)
            return op if self._is_legal(state, player, op) else None

        idx -= 5

        if idx == 1:
            op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
            return op if self._is_legal(state, player, op) else None

        if idx == 2:
            op = Operation(OperationType.UPGRADE_GENERATED_ANT)
            return op if self._is_legal(state, player, op) else None

        return None

    @staticmethod
    def _find_tower_id_at(state: BackendState, player: int, x: int, y: int) -> int | None:
        for tower in state.towers_of(player):
            if tower.x == x and tower.y == y:
                return tower.tower_id
        return None

    @staticmethod
    def _upgrade_direction(state: BackendState, tower_id: int) -> int:
        tower = state.tower_by_id(tower_id)
        if tower is None:
            return 0
        targets = TOWER_UPGRADE_TREE.get(tower.tower_type, ())
        return int(targets[0]) if targets else 0

    @staticmethod
    def _is_legal(state: BackendState, player: int, op: Operation) -> bool:
        try:
            return state.can_apply_operation(player, op)
        except Exception:
            return False

    def sample_legal_action(self, state: BackendState, player: int) -> int:
        valid_ids = [0]
        for action_id in range(1, self.action_dim):
            if self._action_id_to_op(action_id, state, player) is not None:
                valid_ids.append(action_id)
        return int(np.random.choice(valid_ids))
