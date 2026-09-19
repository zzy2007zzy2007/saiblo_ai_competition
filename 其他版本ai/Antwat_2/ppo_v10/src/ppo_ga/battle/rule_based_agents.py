from typing import Any, Dict, List, Tuple

import numpy as np
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

from ..utils.action_constants import HIGHLAND_CELLS, TOWER_POSITIONS
from .ant_war_agent import AntWarAgent

# ── BUILD_ORDERS ──────────────────────────────────────────────────────────────
# 玩家专属建塔优先级顺序
# 排序原则：
#   1. 中线位（y=9）优先
#   2. 到战场中心 (9,9) 的欧几里得距离升序
#   3. 同距离时 y 坐标更接近 9 的优先
BUILD_ORDERS: Dict[int, List[Tuple[int, int]]] = {
    0: [
        (4, 9),   # ① 中线高地，控制中央纵向通路
        (5, 9),   # ② 中线辅助位
        (7, 8),   # ③ 近战场中心 (dist=2.24)，中央通道关键位
        (8, 7),   # ④ 近战场中心 (dist=2.24)
        (6, 7),   # ⑤ 中左，形成交叉火力
        (5, 6),   # ⑥ 中左，衔接上下区域
        (6, 14),  # ⑦ 右下扩展
        (4, 2),   # ⑧ 左上方角落，保护基地侧翼
    ],
    1: [
        (12, 9),  # ① 中线高地，控制中央通路
        (13, 9),  # ② 中线偏右
        (14, 9),  # ③ 中右下方
        (10, 7),  # ④ 近战场中心 (dist=2.24)，中央偏右
        (12, 6),  # ⑤ 中右
        (11, 5),  # ⑥ 右上方
        (11, 14), # ⑦ 右下方
        (13, 15), # ⑧ 右下角
    ],
}


def validate_build_orders() -> None:
    """验证 BUILD_ORDERS 的全量正确性。

    验证项：
      1. 每个玩家恰好包含 8 个唯一位置
      2. 双方位置集合的并集恰好覆盖全部 TOWER_POSITIONS
      3. 每个位置属于对应玩家的高地（HIGHLAND_CELLS）
      4. 双方无共享位置（不重叠）
    """
    # 1. 每个玩家恰好 8 个唯一位置
    for p in (0, 1):
        order = BUILD_ORDERS[p]
        assert len(order) == 8, \
            f"Player {p}: expected 8 positions, got {len(order)}"
        assert len(set(order)) == 8, \
            f"Player {p}: has duplicate positions"

    # 2. 双方并集恰好覆盖全部 TOWER_POSITIONS
    all_positions = set(BUILD_ORDERS[0]) | set(BUILD_ORDERS[1])
    assert all_positions == set(TOWER_POSITIONS), \
        f"Coverage mismatch. " \
        f"Missing: {set(TOWER_POSITIONS) - all_positions}. " \
        f"Extra: {all_positions - set(TOWER_POSITIONS)}"

    # 3. 每个位置属于对应玩家的高地（而不是对方的高地）
    for p in (0, 1):
        for pos in BUILD_ORDERS[p]:
            assert pos in HIGHLAND_CELLS[p], \
                f"Player {p} position {pos} is not in Player {p}'s highlands " \
                f"(HIGHLAND_CELLS[{p}])"

    # 4. 双方位置不重叠
    assert set(BUILD_ORDERS[0]).isdisjoint(set(BUILD_ORDERS[1])), \
        f"Overlapping positions: {set(BUILD_ORDERS[0]) & set(BUILD_ORDERS[1])}"

    print(f"[BUILD_ORDERS] Validation OK. "
          f"{len(TOWER_POSITIONS)} total positions, "
          f"Player 0: {len(BUILD_ORDERS[0])}, Player 1: {len(BUILD_ORDERS[1])}, "
          f"no overlap \u2713")


# 模块加载时自动验证
validate_build_orders()


class BasicRandomAI(AntWarAgent):
    """随机选择合法动作的 Agent"""

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self._mask_handler = None

    def choose_operations(self, state) -> List[Any]:
        if self._mask_handler is None:
            from ..env.action_mask import ActionMaskHandler

            self._mask_handler = ActionMaskHandler()
        # 随机生成四元组，尝试构造合法 Operation
        for _ in range(50):
            strategy = int(np.random.randint(0, 3))
            type_idx = int(np.random.randint(0, 6))
            sub_type = int(np.random.randint(0, 10))
            position = int(np.random.randint(0, 8))
            op = self._mask_handler.construct_operation(
                strategy, type_idx, sub_type, position, state, self.player_id
            )
            if op is not None:
                return [op]
        return []


class BasicTowerAI(AntWarAgent):
    """简单策略：优先建塔，然后升级"""

    def choose_operations(self, state) -> List[Any]:
        operations = []

        if BUILD_ORDERS[self.player_id]:
            center = BUILD_ORDERS[self.player_id][0]
            lightning = Operation(
                OperationType.USE_LIGHTNING_STORM, center[0], center[1]
            )
            if state.can_apply_operation(self.player_id, lightning):
                operations.append(lightning)
            else:
                emp = Operation(OperationType.USE_EMP_BLASTER, center[0], center[1])
                if state.can_apply_operation(self.player_id, emp):
                    operations.append(emp)

        if not operations:
            operations.extend(self._try_build_towers(state))
        if not operations:
            operations.extend(self._try_upgrade_towers(state))
        if not operations:
            operations.extend(self._try_tech_upgrade(state))

        return operations

    def _try_build_towers(self, state) -> List[Any]:
        for pos in BUILD_ORDERS[self.player_id]:
            tower = state.tower_at(pos[0], pos[1])
            if tower is None:
                op = Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
                if state.can_apply_operation(self.player_id, op):
                    return [op]
        return []

    def _try_upgrade_towers(self, state) -> List[Any]:
        from SDK.utils.constants import TOWER_UPGRADE_TREE

        for pos in BUILD_ORDERS[self.player_id]:
            tower = state.tower_at(pos[0], pos[1])
            if tower is None or tower.player != self.player_id:
                continue
            # 从升级树中获取当前塔的可升级目标类型
            targets = TOWER_UPGRADE_TREE.get(tower.tower_type, ())
            if not targets:
                continue
            # 对每个可行的升级目标尝试构造操作
            for target_type in targets:
                op = Operation(
                    OperationType.UPGRADE_TOWER,
                    tower.tower_id,  # 正确：塔的唯一 ID
                    int(target_type),  # 正确：目标 TowerType（来自 TOWER_UPGRADE_TREE）
                )
                try:
                    if state.can_apply_operation(self.player_id, op):
                        return [op]
                except Exception:
                    continue  # SDK 内部异常，跳过当前升级目标
        return []

    def _try_tech_upgrade(self, state) -> List[Any]:
        op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
        if state.can_apply_operation(self.player_id, op):
            return [op]
        op = Operation(OperationType.UPGRADE_GENERATED_ANT)
        if state.can_apply_operation(self.player_id, op):
            return [op]
        return []


class NoviceAI(AntWarAgent):
    """新手 AI：只会按优先级顺序建塔，不会升级、不会放超级武器、不会升科技。

    强度定位：
      - 显著强于 BasicRandomAI（建在正确位置 vs 随机位置）
      - 显著弱于 BasicTowerAI（无超级武器、无塔升级、无科技升级、塔数上限 4）

    策略：
      1. 钱够 → 在 BUILD_ORDERS 优先级最高的空位上建塔（最多 4 座）
      2. 其他情况 → NO_OP
    """

    _MAX_TOWERS = 4

    def choose_operations(self, state) -> List[Any]:
        operations = self._try_build(state)
        return [operations] if operations is not None else []

    def _try_build(self, state):
        tower_count = len(state.towers_of(self.player_id))
        if tower_count >= self._MAX_TOWERS:
            return None

        cost = state.build_tower_cost(tower_count)
        if state.coins[self.player_id] < cost:
            return None

        positions = BUILD_ORDERS.get(self.player_id, [])
        occupied = {(t.x, t.y) for t in state.towers_of(self.player_id)}
        for pos in positions:
            if pos not in occupied:
                op = Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
                if state.can_apply_operation(self.player_id, op):
                    return op
        return None


class MediumRuleAI(BasicTowerAI):
    """中级规则策略：多阶段决策链

    优先级链（按顺序尝试，成功即返回）：
      1. _try_early_build     — 前 3 座塔的早期建造
      2. _try_upgrade_gen_speed_l1 — 生产速度 L0→L1 (200g)
      3. _try_upgrade_tower_l2     — BASIC→HEAVY/QUICK (60g)
      4. _try_upgrade_gen_speed_l2 — 生产速度 L1→L2 (250g)
      5. _try_lightning_storm      — 闪电风暴 (90g, 回合≥50)
      6. _try_emp                  — EMP 冲击炮 (135g, 敌方塔≥3)
      7. _try_mid_build            — 第 4~6 座塔中期建造 (250g 门槛)
      8. _try_upgrade_ant_hp_l1    — 蚂蚁血量 L0→L1 (250g)
      9. _try_upgrade_tower_l3     — HEAVY→HEAVY_PLUS / QUICK→SNIPER (300g)
     10. _try_deflector            — 偏折护盾 (60g, 回合≥80)
     11. _try_late_build           — 第 7+ 座塔后期建造 (350g 门槛)
    """

    _WEAPON_COST = {
        'lightning_storm': 90,
        'emp_blaster': 135,
        'deflector': 60,
    }

    _TOWER_L2_COST = 60
    _TOWER_L3_COST = 200
    _BASE_UPGRADE_COST = (200, 250)

    def __init__(self, player_id: int):
        super().__init__(player_id)
        from SDK.utils.constants import TowerType, SuperWeaponType

        self._build_positions: Dict[int, List[Tuple[int, int]]] = BUILD_ORDERS
        self._TowerType = TowerType
        self._SuperWeaponType = SuperWeaponType
        self._l2_upgraded_towers: set = set()
        self._l3_upgraded_towers: set = set()
        self._ls_positions: Dict[int, List[Tuple[int, int]]] = {
            0: [(9, 9), (8, 9), (10, 9)],
            1: [(9, 9), (10, 9), (8, 9)],
        }
        self._emp_positions: Dict[int, List[Tuple[int, int]]] = {
            0: [(10, 9), (11, 9)],
            1: [(8, 9), (7, 9)],
        }
        self._deflector_positions: Dict[int, List[Tuple[int, int]]] = {
            0: [(6, 9)],
            1: [(12, 9)],
        }

    def reset_for_episode(self, episode_num: int) -> None:
        """每局开始时清空已升级塔记录"""
        self._l2_upgraded_towers.clear()
        self._l3_upgraded_towers.clear()

    # ── choose_operations ─────────────────────────────────────────────────

    def choose_operations(self, state) -> List[Any]:
        p = self.player_id

        op = (
            self._try_early_build(state)
            or self._try_upgrade_gen_speed_l1(state)
            or self._try_upgrade_tower_l2(state)
            or self._try_upgrade_gen_speed_l2(state)
            or self._try_lightning_storm(state)
            or self._try_emp(state)
            or self._try_mid_build(state)
            or self._try_upgrade_ant_hp_l1(state)
            or self._try_upgrade_tower_l3(state)
            or self._try_deflector(state)
            or self._try_late_build(state)
        )
        return [op] if op is not None else []

    # ── 游戏状态查询 ──────────────────────────────────────────────────────

    def _get_coins(self, state) -> int:
        return state.coins[self.player_id]

    def _get_tower_count(self, state) -> int:
        return len(state.towers_of(self.player_id))

    def _get_enemy_tower_count(self, state) -> int:
        return len(state.towers_of(1 - self.player_id))

    def _get_round(self, state) -> int:
        return state.round_index

    def _get_gen_level(self, state) -> int:
        return state.bases[self.player_id].generation_level

    def _get_ant_level(self, state) -> int:
        return state.bases[self.player_id].ant_level

    def _get_build_cost(self, state, tower_count: int) -> int:
        """通过 SDK 计算当前建塔费用"""
        return state.build_tower_cost(tower_count)

    def _weapon_ready(self, state, weapon_key: str) -> bool:
        """检查超级武器是否冷却就绪"""
        idx_map = {'lightning_storm': 1, 'emp_blaster': 2, 'deflector': 3}
        idx = idx_map.get(weapon_key, -1)
        if idx < 0:
            return False
        try:
            cd = state.weapon_cooldowns[self.player_id][idx]
            return int(cd) <= 0
        except (AttributeError, IndexError, TypeError):
            return False

    # ── 位置选择 ──────────────────────────────────────────────────────────

    def _select_build_position(self, state) -> Tuple[int, int] | None:
        """从 BUILD_ORDERS 中选择下一个可建塔的空位"""
        positions = self._build_positions.get(self.player_id, [])
        if not positions:
            return None
        occupied = {(t.x, t.y) for t in state.towers_of(self.player_id)}
        for pos in positions:
            if pos not in occupied:
                op = Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
                if state.can_apply_operation(self.player_id, op):
                    return pos
        return None

    def _select_basic_tower(self, state):
        """选择优先级最高的 BASIC 塔用于 L2 升级"""
        positions = self._build_positions.get(self.player_id, [])
        rank = {pos: i for i, pos in enumerate(positions)}
        candidates = [
            t for t in state.towers_of(self.player_id)
            if t.tower_type == self._TowerType.BASIC
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: rank.get((t.x, t.y), 999))
        return candidates[0]

    def _select_l2_tower(self, state):
        """选择优先级最高的 L2 塔用于 L3 升级（排除已升级的）"""
        positions = self._build_positions.get(self.player_id, [])
        rank = {pos: i for i, pos in enumerate(positions)}
        l2_types = {
            self._TowerType.HEAVY, self._TowerType.QUICK,
            self._TowerType.MORTAR, self._TowerType.PRODUCER,
        }
        candidates = [
            t for t in state.towers_of(self.player_id)
            if t.tower_type in l2_types and t.tower_id not in self._l3_upgraded_towers
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: rank.get((t.x, t.y), 999))
        return candidates[0]

    def _heavies_count(self, state) -> int:
        return sum(
            1 for t in state.towers_of(self.player_id)
            if t.tower_type == self._TowerType.HEAVY
        )

    # ── 决策方法：每步返回 Operation | None ──────────────────────────────

    def _try_early_build(self, state):
        """前 3 座塔的早期建造"""
        tower_count = self._get_tower_count(state)
        if tower_count > 3:
            return None
        cost = self._get_build_cost(state, tower_count)
        if self._get_coins(state) < cost:
            return None
        pos = self._select_build_position(state)
        if pos is None:
            return None
        return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])

    def _try_upgrade_gen_speed_l1(self, state):
        """生产速度 L0 → L1"""
        if self._get_gen_level(state) != 0:
            return None
        if self._get_coins(state) < self._BASE_UPGRADE_COST[0]:
            return None
        op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
        if state.can_apply_operation(self.player_id, op):
            return op
        return None

    def _try_upgrade_tower_l2(self, state):
        """BASIC → HEAVY（或第二个起 QUICK）"""
        if self._get_coins(state) < self._TOWER_L2_COST:
            return None
        tower = self._select_basic_tower(state)
        if tower is None:
            return None
        if tower.tower_id in self._l2_upgraded_towers:
            return None
        heavy_count = self._heavies_count(state)
        target_type = int(self._TowerType.QUICK) if heavy_count >= 1 else int(self._TowerType.HEAVY)
        op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, target_type)
        if not state.can_apply_operation(self.player_id, op):
            return None
        self._l2_upgraded_towers.add(tower.tower_id)
        return op

    def _try_upgrade_gen_speed_l2(self, state):
        """生产速度 L1 → L2"""
        if self._get_gen_level(state) != 1:
            return None
        if self._get_coins(state) < self._BASE_UPGRADE_COST[1]:
            return None
        op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
        if state.can_apply_operation(self.player_id, op):
            return op
        return None

    def _try_lightning_storm(self, state):
        """闪电风暴"""
        if not self._weapon_ready(state, 'lightning_storm'):
            return None
        if self._get_coins(state) < self._WEAPON_COST['lightning_storm']:
            return None
        if self._get_round(state) < 50:
            return None
        positions = self._ls_positions.get(self.player_id, [(9, 9)])
        for pos in positions:
            op = Operation(OperationType.USE_LIGHTNING_STORM, pos[0], pos[1])
            if state.can_apply_operation(self.player_id, op):
                return op
        return None

    def _try_emp(self, state):
        """EMP 冲击炮：敌方塔 ≥ 3 时使用"""
        if not self._weapon_ready(state, 'emp_blaster'):
            return None
        if self._get_coins(state) < self._WEAPON_COST['emp_blaster']:
            return None
        if self._get_enemy_tower_count(state) < 3:
            return None
        positions = self._emp_positions.get(self.player_id, [(9, 9)])
        for pos in positions:
            op = Operation(OperationType.USE_EMP_BLASTER, pos[0], pos[1])
            if state.can_apply_operation(self.player_id, op):
                return op
        return None

    def _try_mid_build(self, state):
        """第 4~6 座塔的中期建造"""
        tower_count = self._get_tower_count(state)
        if tower_count > 6:
            return None
        if tower_count < 4:
            return None
        if self._get_coins(state) < 250:
            return None
        cost = self._get_build_cost(state, tower_count)
        if self._get_coins(state) < cost:
            return None
        pos = self._select_build_position(state)
        if pos is None:
            return None
        return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])

    def _try_upgrade_ant_hp_l1(self, state):
        """蚂蚁血量 L0 → L1"""
        if self._get_ant_level(state) != 0:
            return None
        if self._get_coins(state) < 250:
            return None
        op = Operation(OperationType.UPGRADE_GENERATED_ANT)
        if state.can_apply_operation(self.player_id, op):
            return op
        return None

    def _try_upgrade_tower_l3(self, state):
        """L2 塔升级到 L3：HEAVY→HEAVY_PLUS，QUICK→SNIPER"""
        if self._get_coins(state) < self._TOWER_L3_COST:
            return None
        tower = self._select_l2_tower(state)
        if tower is None:
            return None
        if tower.tower_id in self._l3_upgraded_towers:
            return None
        tower_type = getattr(tower, 'tower_type', None)
        if tower_type == self._TowerType.HEAVY:
            target_type = int(self._TowerType.HEAVY_PLUS)
        elif tower_type == self._TowerType.QUICK:
            target_type = int(self._TowerType.SNIPER)
        else:
            return None
        op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, target_type)
        if not state.can_apply_operation(self.player_id, op):
            return None
        self._l3_upgraded_towers.add(tower.tower_id)
        return op

    def _try_deflector(self, state):
        """偏折护盾（回合 ≥ 80）"""
        if not self._weapon_ready(state, 'deflector'):
            return None
        if self._get_coins(state) < self._WEAPON_COST['deflector']:
            return None
        if self._get_round(state) < 80:
            return None
        positions = self._deflector_positions.get(self.player_id, [(9, 9)])
        for pos in positions:
            op = Operation(OperationType.USE_DEFLECTOR, pos[0], pos[1])
            if state.can_apply_operation(self.player_id, op):
                return op
        return None

    def _try_late_build(self, state):
        """第 7+ 座塔的后期建造"""
        tower_count = self._get_tower_count(state)
        if tower_count < 7:
            return None
        if self._get_coins(state) < 350:
            return None
        cost = self._get_build_cost(state, tower_count)
        if self._get_coins(state) < cost:
            return None
        pos = self._select_build_position(state)
        if pos is None:
            return None
        return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])


__all__ = ["BasicRandomAI", "BasicTowerAI", "NoviceAI", "MediumRuleAI"]
