#!/usr/bin/env python3
"""基于规则的AI - 只在最边上建塔和升级"""
from __future__ import annotations
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
ANT_GAME_ROOT = SUBMISSION_DIR.parent / "Ant-Game"
sys.path.insert(0, str(SUBMISSION_DIR))
sys.path.insert(1, str(ANT_GAME_ROOT))

from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.backend.model import Operation, OperationType
from SDK.utils.constants import SuperWeaponType, TowerType, STRATEGIC_BUILD_ORDER, MAX_ACTIONS
from common import BaseAgent
from score_utils import calculate_operation_score


class SimpleActionCatalog(ActionCatalog):
    """简化的ActionCatalog，禁用一步前瞻搜索以提高性能"""

    def build(self, state: BackendState, player: int) -> list[ActionBundle]:
        """构建候选动作列表，但不进行一步前瞻搜索"""
        bundles: list[ActionBundle] = [ActionBundle(name="hold", score=0.0, tags=("noop",))]
        bundles.extend(self._build_candidates(state, player))
        bundles.extend(self._upgrade_candidates(state, player))
        bundles.extend(self._downgrade_candidates(state, player))
        bundles.extend(self._base_upgrade_candidates(state, player))
        bundles.extend(self._superweapon_candidates(state, player))
        bundles.extend(self._paired_candidates(state, player, bundles[1:]))

        # 去重
        unique: dict[tuple[tuple[int, int, int], ...], ActionBundle] = {}
        for bundle in bundles:
            key = tuple((int(op.op_type), op.arg0, op.arg1) for op in bundle.operations)
            if key not in unique or bundle.score > unique[key]:
                unique[key] = bundle

        # 排序并返回（不进行一步前瞻搜索）
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        return ordered[:self.max_actions]


def is_edge_position(x: int, y: int) -> bool:
    """判断位置是否在最边上"""
    return y == 1 or y == 17


class RuleBasedAI(BaseAgent):
    """基于规则的AI - 只在最边上建塔和升级"""
    
    LIGHTNING_COST = 90
    LOG_FILE = SUBMISSION_DIR / "ai_decisions.log"

    def __init__(self, seed: int | None = None, max_actions: int = MAX_ACTIONS) -> None:
        super().__init__(seed=seed, max_actions=max_actions)
        # 替换为 SimpleActionCatalog（禁用一步前瞻搜索，大幅加速）
        self.catalog = SimpleActionCatalog(
            max_actions=max_actions,
            feature_extractor=self.feature_extractor,
        )
    
    @classmethod
    def reset_log(cls):
        """清空日志文件"""
        with open(cls.LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=== AI决策日志 ===\n\n")
    
    @classmethod
    def log(cls, message):
        """写日志"""
        with open(cls.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    def get_lightning_bundle(self, bundles: list[ActionBundle]) -> ActionBundle | None:
        """获取评分最高的闪电风暴"""
        lightning = []
        for b in bundles:
            for op in b.operations:
                if op.op_type == OperationType.USE_LIGHTNING_STORM:
                    lightning.append(b)
                    break
        if lightning:
            return max(lightning, key=lambda x: x.score)
        return None
    
    def op_to_bundle(self, op: Operation, state: BackendState, player: int) -> ActionBundle:
        """把单个操作包装成bundle"""
        score = calculate_operation_score(state, player, op)
        name = f"custom_{OperationType(op.op_type).name}"
        return ActionBundle(name=name, operations=(op,), score=score)
    
    def get_edge_upgrade_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        """获取在最边上升级basic到producer的bundles"""
        result = []
        towers = state.towers_of(player)
        
        for tower in towers:
            if not is_edge_position(tower.x, tower.y):
                continue
            if tower.tower_type != TowerType.BASIC:
                continue
            
            for target in {TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC}:
                op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
                if state.can_apply_operation(player, op):
                    result.append(self.op_to_bundle(op, state, player))
        
        return result
    
    def get_edge_build_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        """获取在最边上建basic塔的bundles - 使用自定义评分（安全优先）"""
        result = []
        from SDK.utils.constants import PLAYER_BASES
        enemy_base = PLAYER_BASES[1 - player]
        
        for x, y in STRATEGIC_BUILD_ORDER[player]:
            if not is_edge_position(x, y):
                continue
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if state.can_apply_operation(player, op):
                # 自定义评分：越安全越好，离敌人越远越好
                score = self._calc_build_score_for_production(state, player, x, y, enemy_base)
                bundle = ActionBundle(
                    name=f"build_edge({x},{y})", 
                    operations=(op,), 
                    score=score,
                    tags=("build",)
                )
                result.append(bundle)
        return result
    
    def _calc_build_score_for_production(self, state: BackendState, player: int, x: int, y: int, enemy_base) -> float:
        """计算适合生产的建塔评分（安全优先）"""
        from SDK.utils.geometry import hex_distance
        
        # 1. 敌人压力（周围6格内的敌人）- 敌人越多分数越低
        enemy_pressure = 0.0
        for ant in state.ants_of(1 - player):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= 6:
                enemy_pressure += max(0.0, 6.5 - distance) * (1.0 + ant.level * 0.4)
        
        # 2. 离敌人基地距离 - 越远越好
        dist_to_enemy = hex_distance(x, y, enemy_base[0], enemy_base[1])
        
        # 评分：离敌人越远越高，敌人压力越低越高
        score = dist_to_enemy * 2.0 - enemy_pressure * 5.0
        
        return score
    
    def get_downgrade_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        """获取拆塔bundles - 当闪电快好时，允许拆producer塔"""
        result = []
        towers = state.towers_of(player)
        lightning_cooldown = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        coins = state.coins[player]
        
        for tower in towers:
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if state.can_apply_operation(player, op):
                # 如果闪电快好且钱不够，连producer塔也可以拆
                if lightning_cooldown <= 2 and coins < self.LIGHTNING_COST:
                    result.append(self.op_to_bundle(op, state, player))
                # 否则只拆非producer塔
                elif tower.tower_type not in {TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC}:
                    result.append(self.op_to_bundle(op, state, player))
        
        return result
    
    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        
        lightning_cooldown = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        coins = state.coins[player]
        
        self.log(f"\n玩家{player}决策开始：闪电冷却={lightning_cooldown} 金币={coins}")
        
        # 策略1：如果闪电冷却0 且 金币够，就闪电风暴
        lightning = self.get_lightning_bundle(bundles)
        if lightning_cooldown == 0 and coins >= self.LIGHTNING_COST and lightning is not None:
            self.log(f"✓ 策略1：选闪电风暴")
            return lightning
        self.log(f"✗ 策略1：不满足条件")
        
        # 策略2：当闪电冷却 <=2 且 金币不够闪电风暴时，拆塔
        if lightning_cooldown <= 2 and coins < self.LIGHTNING_COST:
            self.log(f"策略2：闪电冷却快好，需要拆塔攒钱")
            downgrade_bundles = self.get_downgrade_bundles(state, player)
            if downgrade_bundles:
                self.log(f"✓ 策略2：找到拆塔操作")
                return max(downgrade_bundles, key=lambda x: x.score)
            self.log(f"✗ 策略2：没有找到可拆的塔")
        else:
            self.log(f"✗ 策略2：不满足条件")
        
        # 策略3：其他情况
        # 如果闪电冷却快好（0-2），优先攒钱放闪电
        if lightning_cooldown <= 2:
            self.log(f"策略3：闪电冷却快好，优先hold攒钱")
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
        
        # 正常发展阶段（闪电冷却>=3）
        # 3.1 优先升级边上的basic到producer
        edge_upgrades = self.get_edge_upgrade_bundles(state, player)
        if edge_upgrades:
            self.log(f"✓ 策略3.1：找到边上升级操作（{len(edge_upgrades)}个）")
            return max(edge_upgrades, key=lambda x: x.score)
        self.log(f"✗ 策略3.1：没有边上升级操作")
        
        # 3.2 检查：如果边上有basic塔但不够升级，就hold攒钱
        towers = state.towers_of(player)
        has_edge_basic_towers = False
        for tower in towers:
            if is_edge_position(tower.x, tower.y) and tower.tower_type == TowerType.BASIC:
                has_edge_basic_towers = True
                break
        if has_edge_basic_towers:
            self.log(f"✓ 策略3.2：边上有basic塔，hold攒钱")
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
        self.log(f"✗ 策略3.2：边上没有basic塔")
        
        # 3.3 在边上建塔
        edge_builds = self.get_edge_build_bundles(state, player)
        if edge_builds:
            self.log(f"✓ 策略3.3：找到边上建塔操作（{len(edge_builds)}个）")
            chosen = max(edge_builds, key=lambda x: x.score)
            for op in chosen.operations:
                if op.op_type == OperationType.BUILD_TOWER:
                    self.log(f"    选边上建塔：({op.arg0},{op.arg1})")
            return chosen
        self.log(f"✗ 策略3.3：没有边上建塔操作")
        
        # 3.4 实在不行就hold
        self.log(f"✓ 策略3.4：hold")
        return ActionBundle(name="hold", score=0.0, tags=("noop",))


class AI(RuleBasedAI):
    pass


def create_agent():
    return AI()
