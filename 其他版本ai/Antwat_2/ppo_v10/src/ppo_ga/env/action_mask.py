from typing import Optional

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType, TOWER_UPGRADE_TREE, TowerType

from ..utils.action_constants import (
    _PLAYER_TOWER_POS,
    _TECH_POS_MAP,
    _WEAPON_POS_REDUNDANCY,
    ACTION_DECODE_TABLE,
    SUPER_WEAPON_POSITIONS,
    SuperWeaponType,
    validate_upgrade_operation,
)

# 超级武器 action_name → (SuperWeaponType, OperationType) 映射
_WEAPON_MAP = {
    "lightning_storm": (SuperWeaponType.LIGHTNING_STORM, OperationType.USE_LIGHTNING_STORM),
    "emp_blaster":     (SuperWeaponType.EMP_BLASTER,     OperationType.USE_EMP_BLASTER),
    "deflector":       (SuperWeaponType.DEFLECTOR,       OperationType.USE_DEFLECTOR),
    "evasion":         (SuperWeaponType.EMERGENCY_EVASION, OperationType.USE_EMERGENCY_EVASION),
}


class ActionMaskHandler:
    """动作解码处理器 - 将 (strategy, type_idx, sub_type, position) 转换为 Operation。

    不再生成 119 维扁平掩码。合法性检查通过构造 Operation 后调用
    state.can_apply_operation() 完成。
    """

    def construct_operation(
        self,
        strategy: int,
        type_idx: int,
        sub_type: int,
        position: int,
        state,
        player: int,
    ) -> Optional[Operation]:
        """将 (s, t, st, p) 四元组解码为 Operation

        解码流程：
        1. 查 ACTION_DECODE_TABLE[strategy][type_idx] 获取 action_name
        2. 根据 action_name 将 position 映射为具体坐标或科技类型
        3. 构造 Operation 对象
        4. 调用 can_apply_operation 检查合法性

        Args:
            strategy:  {0, 1, 2}
            type_idx:  {0..5}
            sub_type:  {0..9}（仅 UPGRADE_TOWER 使用）
            position:  {0..7}
            state:     游戏状态
            player:    玩家 ID

        Returns:
            Operation 对象，或 None（非法组合 → NOOP）
        """
        action_meta = ACTION_DECODE_TABLE[strategy].get(type_idx)
        if action_meta is None:
            return None

        action_name, _flags = action_meta

        # === 塔操作：BUILD / UPGRADE / DOWNGRADE ===
        if action_name in ("build_tower", "upgrade_tower", "downgrade_tower"):
            pos = _PLAYER_TOWER_POS[player][position]
            x, y = pos

            if action_name == "build_tower":
                if state.tower_at(x, y) is not None:
                    return None
                op = Operation(OperationType.BUILD_TOWER, x, y)
                return op if self._is_valid_operation(state, player, op) else None

            elif action_name == "upgrade_tower":
                tower_id = self._find_tower_id_at(state, player, x, y)
                if tower_id is None:
                    return None
                tower = state.tower_by_id(tower_id)
                if tower is None:
                    return None
                upgrade_target = self._resolve_upgrade_path(tower.tower_type, sub_type)
                if upgrade_target is None:
                    return None
                op = Operation(OperationType.UPGRADE_TOWER, tower_id, upgrade_target)
                if not validate_upgrade_operation(op):
                    return None
                return op if self._is_valid_operation(state, player, op) else None

            elif action_name == "downgrade_tower":
                tower_id = self._find_tower_id_at(state, player, x, y)
                if tower_id is None:
                    return None
                op = Operation(OperationType.DOWNGRADE_TOWER, tower_id)
                return op if self._is_valid_operation(state, player, op) else None

        # === 超级武器 ===
        elif action_name in _WEAPON_MAP:
            weapon_type, op_type = _WEAPON_MAP[action_name]
            weapon_positions = SUPER_WEAPON_POSITIONS[weapon_type]
            mapped_idx = _WEAPON_POS_REDUNDANCY[position]
            pos = weapon_positions[mapped_idx]
            op = Operation(op_type, pos[0], pos[1])
            return op if self._is_valid_operation(state, player, op) else None

        # === 科技升级 ===
        elif action_name == "tech_upgrade":
            tech_type = _TECH_POS_MAP[position]
            if tech_type == "generation_speed":
                op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
            else:
                op = Operation(OperationType.UPGRADE_GENERATED_ANT)
            return op if self._is_valid_operation(state, player, op) else None

        return None

    def _resolve_upgrade_path(self, current_tower_type: int, sub_type: int) -> Optional[int]:
        """根据当前塔类型和 sub_type 确定升级目标 TowerType

        sub_type 10 维编码：
            [0:3)  main_tree   → HEAVY/QUICK/MORTAR
            [3:6)  sub_variant → variant_A/B/C
            [6:10) producer    → PRODUCER/FAST/SIEGE/MEDIC

        升级逻辑：
            - BASIC → main_tree / producer 决定一级升级方向
            - combat_type → sub_variant 决定二级子类
            - PRODUCER → producer sub 决定二级子类
        """
        targets = TOWER_UPGRADE_TREE.get(current_tower_type, ())
        if not targets:
            return None

        if current_tower_type == TowerType.BASIC:
            # BASIC → 4 targets: HEAVY, QUICK, MORTAR, PRODUCER
            # main_tree [0:3) → targets[0:3], producer [6:10) → targets[3]
            if sub_type < 3:
                return int(targets[sub_type])
            return int(targets[3])

        # 非 BASIC → 3 targets（二级升级）
        # sub_variant [3:6) → targets[0:3]
        # producer sub [7:10) → targets[0:3]
        # main_tree [0:3) → 冗余映射到 targets[0:3]
        if sub_type < 3:
            idx = sub_type
        elif sub_type < 6:
            idx = sub_type - 3
        elif sub_type == 6:
            idx = 0
        else:
            idx = sub_type - 7
        return int(targets[idx])

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
