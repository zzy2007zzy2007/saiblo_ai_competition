from typing import Optional

import numpy as np

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType, TOWER_UPGRADE_TREE

from ..utils.action_constants import (
    ACTION_DIM,
    SUPER_WEAPON_POSITIONS,
    TOWER_POSITIONS,
    TYPE_CONFIG,
    SuperWeaponType,
    _UPGRADE_DIRECTIONS,
)

_DOWNGRADE_BAN_ROUNDS = 0


class ActionMaskHandler:
    """动作掩码处理器 - 生成动作合法性掩码，确保策略网络只输出合法动作"""

    def get_action_mask(self, state, player: int) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        mask[0] = 1.0
        for action_id in range(1, ACTION_DIM):
            op = self._action_id_to_op(action_id, state, player)
            if op is not None:
                mask[action_id] = 1.0

        # ======================================================================
        # [INTENTIONAL DESIGN] 前1000回合禁止降级防御塔
        # 这是故意设计的行为，不是bug！
        # 原因：防止模型在早期通过频繁降级防御塔来获取短期奖励，
        # 从而避免模型学会"先建后拆"的投机策略。
        # 1000回合后降级操作恢复正常可用。
        # ======================================================================
        round_idx = getattr(state, "round_index", 0)
        if round_idx < _DOWNGRADE_BAN_ROUNDS:
            for type_cfg in TYPE_CONFIG:
                if type_cfg["name"] == "downgrade_tower":
                    mask[type_cfg["flat_start"] : type_cfg["flat_end"]] = 0.0
                    break

        return mask

    def action_id_to_op(
        self, action_id: int, state, player: int
    ) -> Optional[Operation]:
        """将 action_id 转换为操作对象（公共方法）"""
        return self._action_id_to_op(action_id, state, player)

    def _action_id_to_op(
        self, action_id: int, state, player: int
    ) -> Optional[Operation]:
        if action_id == 0:
            return None

        for type_cfg in TYPE_CONFIG:
            start = type_cfg["flat_start"]
            end = type_cfg["flat_end"]
            if start <= action_id < end:
                target_idx = action_id - start
                type_name = type_cfg["name"]
                break
        else:
            return None

        if type_name == "build_tower":
            if target_idx < len(TOWER_POSITIONS):
                pos = TOWER_POSITIONS[target_idx]
                op = Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
                return op if self._is_valid_operation(state, player, op) else None
        elif type_name == "upgrade_tower":
            # 40 个插槽 = 10 个可建塔位置 × 4 个升级方向
            pos_idx = target_idx // _UPGRADE_DIRECTIONS
            dir_idx = target_idx % _UPGRADE_DIRECTIONS
            if pos_idx < len(TOWER_POSITIONS):
                pos = TOWER_POSITIONS[pos_idx]
                tower_id = self._find_tower_id_at(state, player, pos[0], pos[1])
                if tower_id is not None:
                    tower = state.tower_by_id(tower_id)
                    if tower is not None:
                        targets = TOWER_UPGRADE_TREE.get(tower.tower_type, ())
                        if dir_idx < len(targets):
                            op = Operation(
                                OperationType.UPGRADE_TOWER,
                                tower_id,
                                int(targets[dir_idx]),
                            )
                            return (
                                op
                                if self._is_valid_operation(state, player, op)
                                else None
                            )
        elif type_name == "downgrade_tower":
            if target_idx < len(TOWER_POSITIONS):
                pos = TOWER_POSITIONS[target_idx]
                tower_id = self._find_tower_id_at(state, player, pos[0], pos[1])
                if tower_id is not None:
                    op = Operation(OperationType.DOWNGRADE_TOWER, tower_id)
                    return op if self._is_valid_operation(state, player, op) else None
        elif type_name == "lightning_storm":
            positions = SUPER_WEAPON_POSITIONS.get(SuperWeaponType.LIGHTNING_STORM, [])
            if target_idx < len(positions):
                pos = positions[target_idx]
                op = Operation(OperationType.USE_LIGHTNING_STORM, pos[0], pos[1])
                return op if self._is_valid_operation(state, player, op) else None
        elif type_name == "emp_blaster":
            positions = SUPER_WEAPON_POSITIONS.get(SuperWeaponType.EMP_BLASTER, [])
            if target_idx < len(positions):
                pos = positions[target_idx]
                op = Operation(OperationType.USE_EMP_BLASTER, pos[0], pos[1])
                return op if self._is_valid_operation(state, player, op) else None
        elif type_name == "deflector":
            positions = SUPER_WEAPON_POSITIONS.get(SuperWeaponType.DEFLECTOR, [])
            if target_idx < len(positions):
                pos = positions[target_idx]
                op = Operation(OperationType.USE_DEFLECTOR, pos[0], pos[1])
                return op if self._is_valid_operation(state, player, op) else None
        elif type_name == "evasion":
            positions = SUPER_WEAPON_POSITIONS.get(
                SuperWeaponType.EMERGENCY_EVASION, []
            )
            if target_idx < len(positions):
                pos = positions[target_idx]
                op = Operation(OperationType.USE_EMERGENCY_EVASION, pos[0], pos[1])
                return op if self._is_valid_operation(state, player, op) else None
        elif type_name == "tech_upgrade":
            if target_idx == 0:
                op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
                return op if self._is_valid_operation(state, player, op) else None
            elif target_idx == 1:
                op = Operation(OperationType.UPGRADE_GENERATED_ANT)
                return op if self._is_valid_operation(state, player, op) else None

        return None

    @staticmethod
    def _find_tower_id_at(state, player: int, x: int, y: int) -> Optional[int]:
        """查找指定坐标上的己方塔 ID"""
        for tower in state.towers_of(player):
            if tower.x == x and tower.y == y:
                return tower.tower_id
        return None

    def _is_valid_operation(self, state, player: int, op) -> bool:
        if op is None:
            return True
        try:
            return state.can_apply_operation(player, op)
        except Exception:
            return False

    def flat_mask_to_type_mask(self, flat_mask: np.ndarray) -> np.ndarray:
        """将扁平动作掩码转换为类型掩码（类型合法当且仅当该类型下至少有一个合法动作）"""
        type_mask = np.zeros(len(TYPE_CONFIG), dtype=np.float32)
        # 处理 2D 输入 (batch, dim) → 取第一行
        if flat_mask.ndim == 2:
            flat_mask = flat_mask[0]
        for i, type_cfg in enumerate(TYPE_CONFIG):
            start = type_cfg["flat_start"]
            end = type_cfg["flat_end"]
            if flat_mask[start:end].sum() > 0:
                type_mask[i] = 1.0
        return type_mask

    def flat_mask_to_target_mask(
        self, flat_mask: np.ndarray, type_id: int
    ) -> np.ndarray:
        """从扁平动作掩码中提取指定动作类型的目标掩码"""
        type_cfg = TYPE_CONFIG[type_id]
        start = type_cfg["flat_start"]
        end = type_cfg["flat_end"]
        return flat_mask[start:end].copy()
