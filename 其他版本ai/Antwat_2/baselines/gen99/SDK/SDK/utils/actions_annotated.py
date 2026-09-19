#!/usr/bin/env python3
"""
ActionCatalog 评估函数详解
==========================

本文件包含所有操作评估函数，标注了所有可调参数。
可以通过调整这些参数来优化AI表现。

参数调优建议：
1. 先理解每个参数的作用
2. 每次只调整1-2个参数
3. 使用测试环境验证效果
4. 记录每次调整的结果

作者: AI Assistant
日期: 2026-04-03
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from SDK.utils.constants import (
    ANT_GENERATION_CYCLE,
    ANT_MAX_HP,
    MAX_ACTIONS,
    OperationType,
    PLAYER_BASES,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    TowerType,
)
from SDK.utils.features import FeatureExtractor
from SDK.utils.geometry import hex_distance
from SDK.backend.state import BackendState
from SDK.backend.model import Operation, Tower


@dataclass(slots=True)
class ActionBundle:
    """操作包：包含一个或多个操作的组合"""
    name: str                                    # 操作名称（用于调试）
    operations: tuple[Operation, ...] = ()       # 操作列表
    score: float = 0.0                           # 评估分数（越高越好）
    tags: tuple[str, ...] = field(default_factory=tuple)  # 标签（用于分类）

    def protocol_lines(self) -> list[list[int]]:
        return [op.to_protocol_tokens() for op in self.operations]


class ActionCatalog:
    """
    操作目录：生成并评估所有可能的操作
    
    可调参数汇总：
    ====================
    【建造塔】
    - PRESSURE_WEIGHT: 敌方压力权重 (默认: 2.5)
    - BUILD_COST_PENALTY: 建造成本惩罚系数 (默认: 0.03)
    
    【升级塔】
    - LEVEL_BONUS: 塔等级奖励系数 (默认: 1.5)
    - POSITION_WEIGHT: 位置优先级权重 (默认: 0.15)
    
    【降级/出售】
    - REFUND_VALUE: 退款价值系数 (默认: 0.04)
    - POSITION_PENALTY: 位置惩罚系数 (默认: 0.3)
    - LEVEL_PENALTY: 等级惩罚系数 (默认: 3.0)
    - PRESSURE_THRESHOLD: 压力阈值 (默认: 1.5)
    
    【基地升级-蚂蚁】
    - ANT_BASE_SCORE: 基础分 (默认: 8.0)
    - HP_GAIN_WEIGHT: 生命值增益权重 (默认: 1.4)
    - FRONTLINE_WEIGHT_ANT: 前线距离权重 (默认: 0.22)
    - ROUND_DECAY_ANT: 回合衰减系数 (默认: 0.01)
    - LEVEL_PENALTY_ANT: 等级惩罚 (默认: 1.2)
    
    【基地升级-生产速度】
    - GEN_BASE_SCORE: 基础分 (默认: 10.0)
    - TEMPO_GAIN_WEIGHT: 节奏增益权重 (默认: 14.0)
    - NEAREST_ANT_WEIGHT: 最近蚂蚁距离权重 (默认: 0.12)
    - ROUND_DECAY_GEN: 回合衰减系数 (默认: 0.015)
    
    【组合操作】
    - SECOND_ACTION_WEIGHT: 第二个操作权重 (默认: 0.9)
    
    【重新排序】
    - ROLLOUT_WEIGHT: 推演价值权重 (默认: 0.2)
    
    【塔类型适配度】
    - 各种塔类型有不同的权重参数
    
    【超级武器】
    - 各种武器有不同的成本惩罚系数
    """
    
    # ==================== 可调参数定义 ====================
    
    # 【建造塔参数】
    PRESSURE_WEIGHT = 2.5           # 敌方压力权重：越高越重视防守
    BUILD_COST_PENALTY = 0.03       # 建造成本惩罚：越高越节省金币
    
    # 【升级塔参数】
    LEVEL_BONUS = 1.5               # 等级奖励：越高越喜欢升级高等级塔
    POSITION_WEIGHT = 0.15          # 位置权重：越高越重视战略位置
    
    # 【降级/出售参数】
    REFUND_VALUE = 0.04             # 退款价值系数：越高越喜欢卖塔
    POSITION_PENALTY = 0.3          # 位置惩罚：越高越不卖重要位置的塔
    LEVEL_PENALTY = 3.0             # 等级惩罚：越高越不卖高等级塔
    PRESSURE_THRESHOLD = 1.5        # 压力阈值：超过此值不考虑出售
    
    # 【基地升级-蚂蚁参数】
    ANT_BASE_SCORE = 8.0            # 蚂蚁升级基础分
    HP_GAIN_WEIGHT = 1.4            # 生命值增益权重
    FRONTLINE_WEIGHT_ANT = 0.22     # 前线距离权重：前线压力大时更想升级
    ROUND_DECAY_ANT = 0.01          # 回合衰减：后期升级价值降低
    LEVEL_PENALTY_ANT = 1.2         # 等级惩罚：高等级时升级成本考虑
    
    # 【基地升级-生产速度参数】
    GEN_BASE_SCORE = 10.0           # 生产速度升级基础分
    TEMPO_GAIN_WEIGHT = 14.0        # 节奏增益权重：生产周期缩短的价值
    NEAREST_ANT_WEIGHT = 0.12       # 最近蚂蚁距离权重
    ROUND_DECAY_GEN = 0.015         # 回合衰减系数
    
    # 【组合操作参数】
    SECOND_ACTION_WEIGHT = 0.9      # 第二个操作权重：避免过度乐观
    
    # 【重新排序参数】
    ROLLOUT_WEIGHT = 0.2            # 推演价值权重：考虑一步后的状态变化
    
    # 【超级武器阈值】
    STORM_THRESHOLD = 2.5           # 闪电风暴使用阈值
    EMP_THRESHOLD = 2.0             # EMP使用阈值
    DEFLECTOR_THRESHOLD = 1.5       # 护盾使用阈值
    EVASION_THRESHOLD = 1.0         # 紧急回避使用阈值
    
    # 【超级武器成本惩罚】
    STORM_COST_PENALTY = 0.03       # 闪电风暴成本惩罚
    EMP_COST_PENALTY = 0.025        # EMP成本惩罚
    DEFLECTOR_COST_PENALTY = 0.02   # 护盾成本惩罚
    EVASION_COST_PENALTY = 0.02     # 紧急回避成本惩罚
    
    # 【塔类型适配度参数】
    HEAVY_DENSITY_WEIGHT = 1.1      # 重型塔-密度权重
    HEAVY_DISTANCE_PENALTY = 0.1    # 重型塔-距离惩罚
    ICE_DENSITY_WEIGHT = 1.3        # 冰冻塔-密度权重
    MORTAR_DENSITY_WEIGHT = 0.85    # 迫击炮-密度权重
    MORTAR_DISTANCE_WEIGHT = 12.0   # 迫击炮-距离权重基准
    QUICK_DENSITY_WEIGHT = 0.9      # 快速塔-密度权重
    QUICK_BASE_BONUS = 4.0          # 快速塔-基础奖励
    SNIPER_DISTANCE_WEIGHT = 18.0   # 狙击塔-距离权重基准
    SNIPER_DENSITY_WEIGHT = 0.4     # 狙击塔-密度权重
    PRODUCER_DENSITY_BONUS = 10.0   # 生产塔-密度奖励基准
    PRODUCER_DENSITY_WEIGHT = 0.75  # 生产塔-密度权重
    PRODUCER_DISTANCE_WEIGHT = 16.0 # 生产塔-距离权重基准
    PRODUCER_FORWARD_WEIGHT = 0.22  # 生产塔-前线权重
    
    # 【生产塔分支奖励】
    PRODUCER_BRANCH_BONUS = 0.0     # 基础生产塔分支奖励
    PRODUCER_FAST_BONUS = 0.5       # 快速生产塔分支奖励
    PRODUCER_SIEGE_BONUS = 1.1      # 攻城生产塔分支奖励
    PRODUCER_MEDIC_BONUS = 1.0      # 医疗生产塔分支奖励

    def __init__(self, max_actions: int = MAX_ACTIONS, feature_extractor: FeatureExtractor | None = None) -> None:
        self.max_actions = max_actions
        self.feature_extractor = feature_extractor or FeatureExtractor(max_actions=max_actions)

    def build(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        构建所有可能的操作候选
        
        流程：
        1. 生成各类操作候选（建造、升级、降级、基地升级、超级武器、组合）
        2. 去重（相同操作保留分数最高的）
        3. 按分数排序
        4. 使用一步推演重新排序
        5. 返回前max_actions个
        """
        # 1. 生成基础操作（hold + 各类候选）
        bundles: list[ActionBundle] = [ActionBundle(name="hold", score=0.0, tags=("noop",))]
        bundles.extend(self._build_candidates(state, player))      # 建造塔
        bundles.extend(self._upgrade_candidates(state, player))    # 升级塔
        bundles.extend(self._downgrade_candidates(state, player))  # 降级/出售
        bundles.extend(self._base_upgrade_candidates(state, player))  # 基地升级
        bundles.extend(self._superweapon_candidates(state, player))   # 超级武器
        bundles.extend(self._paired_candidates(state, player, bundles[1:]))  # 组合操作
        
        # 2. 去重：相同操作保留分数最高的
        unique: dict[tuple[tuple[int, int, int], ...], ActionBundle] = {}
        for bundle in bundles:
            key = tuple((int(op.op_type), op.arg0, op.arg1) for op in bundle.operations)
            if key not in unique or bundle.score > unique[key].score:
                unique[key] = bundle
        
        # 3. 按分数排序
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        
        # 4. 使用一步推演重新排序（考虑操作后的状态变化）
        reranked = self._rerank_with_one_step_rollout(
            state, player, 
            ordered[: min(len(ordered), self.max_actions * 2)]
        )
        
        # 5. 返回前max_actions个
        return reranked[: self.max_actions]

    def action_mask(self, bundles: list[ActionBundle]) -> np.ndarray:
        """生成动作掩码（用于神经网络）"""
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mask[: len(bundles)] = 1
        return mask

    def bundle_for_index(self, bundles: list[ActionBundle], action_index: int) -> ActionBundle:
        """根据索引获取操作包"""
        if 0 <= action_index < len(bundles):
            return bundles[action_index]
        return bundles[0]

    def _build_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        【建造塔候选生成】
        
        评估公式：
        score = lane_bonus + pressure * PRESSURE_WEIGHT - build_cost * BUILD_COST_PENALTY
        
        参数说明：
        - lane_bonus: 位置战略价值（预定义的战略建造顺序）
        - pressure: 当地敌方蚂蚁压力（距离越近、等级越高压力越大）
        - build_cost: 当前建造成本（塔越多成本越高）
        
        可调参数：
        - PRESSURE_WEIGHT (默认2.5): 敌方压力权重
          * 增大：更重视防守，在压力大的地方建塔
          * 减小：更重视经济，优先在战略位置建塔
        
        - BUILD_COST_PENALTY (默认0.03): 建造成本惩罚
          * 增大：更节省金币，少建塔
          * 减小：更愿意花钱建塔
        """
        results: list[ActionBundle] = []
        tower_count = state.tower_count(player)
        build_cost = state.build_tower_cost(tower_count)
        
        # 金币不足直接返回
        if state.coins[player] < build_cost:
            return results
        
        # 遍历所有战略建造位置
        for x, y in STRATEGIC_BUILD_ORDER[player]:
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if not state.can_apply_operation(player, op):
                continue
            
            # 计算当地敌方压力
            pressure = self._local_enemy_pressure(state, player, x, y)
            # 获取位置战略价值
            lane_bonus = state.slot_priority(player, x, y)
            
            # 评估分数 = 位置价值 + 压力权重 - 成本惩罚
            score = lane_bonus + pressure * self.PRESSURE_WEIGHT - build_cost * self.BUILD_COST_PENALTY
            
            results.append(ActionBundle(
                name=f"build@{x},{y}", 
                operations=(op,), 
                score=score, 
                tags=("build",)
            ))
        return results

    def _upgrade_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        【升级塔候选生成】
        
        评估公式：
        score = fit + tower.level * LEVEL_BONUS + slot_priority * POSITION_WEIGHT
        
        参数说明：
        - fit: 塔类型与当地环境的适配度（由_tower_type_fit计算）
        - tower.level: 当前塔等级
        - slot_priority: 位置战略优先级
        
        可调参数：
        - LEVEL_BONUS (默认1.5): 等级奖励系数
          * 增大：更喜欢升级高等级塔
          * 减小：对等级不敏感，更看适配度
        
        - POSITION_WEIGHT (默认0.15): 位置权重
          * 增大：更重视战略位置的塔
          * 减小：位置因素不重要
        """
        results: list[ActionBundle] = []
        enemy_base = PLAYER_BASES[1 - player]  # 敌方基地位置
        
        for tower in state.towers_of(player):
            # 计算当地敌方密度（压力）
            local_density = self._local_enemy_pressure(state, player, tower.x, tower.y)
            
            # 遍历所有可升级的目标类型
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ()):
                op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
                if not state.can_apply_operation(player, op):
                    continue
                
                # 计算塔类型适配度
                fit = self._tower_type_fit(
                    target, 
                    local_density, 
                    hex_distance(tower.x, tower.y, *enemy_base)
                )
                
                # 评估分数 = 适配度 + 等级奖励 + 位置权重
                score = (
                    fit + 
                    tower.level * self.LEVEL_BONUS + 
                    state.slot_priority(player, tower.x, tower.y) * self.POSITION_WEIGHT
                )
                
                results.append(ActionBundle(
                    name=f"upgrade#{tower.tower_id}->{int(target)}",
                    operations=(op,),
                    score=score,
                    tags=("upgrade", f"tower:{int(target)}"),
                ))
        return results

    def _downgrade_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        【降级/出售塔候选生成】
        
        评估公式：
        score = refund * REFUND_VALUE - slot_priority * POSITION_PENALTY - tower.level * LEVEL_PENALTY
        
        参数说明：
        - refund: 出售获得的退款金额
        - slot_priority: 位置战略优先级（重要位置惩罚高）
        - tower.level: 塔等级（高等级惩罚高）
        
        可调参数：
        - REFUND_VALUE (默认0.04): 退款价值系数
          * 增大：更愿意卖塔换钱
          * 减小：不重视退款，保留塔
        
        - POSITION_PENALTY (默认0.3): 位置惩罚
          * 增大：不轻易卖战略位置的塔
          * 减小：位置因素不重要
        
        - LEVEL_PENALTY (默认3.0): 等级惩罚
          * 增大：不轻易卖高等级塔
          * 减小：等级因素不重要
        
        - PRESSURE_THRESHOLD (默认1.5): 压力阈值
          * 超过此压力值的位置不考虑出售
          * 增大：更愿意在压力区卖塔
          * 减小：只在安全区卖塔
        """
        results: list[ActionBundle] = []
        
        for tower in state.towers_of(player):
            # 计算当地压力
            pressure = self._local_enemy_pressure(state, player, tower.x, tower.y)
            
            # 压力太大，不适合出售（防守需要）
            if pressure > self.PRESSURE_THRESHOLD:
                continue
            
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not state.can_apply_operation(player, op):
                continue
            
            # 计算退款金额
            refund = state.operation_income(player, op)
            
            # 评估分数 = 退款价值 - 位置惩罚 - 等级惩罚
            score = (
                refund * self.REFUND_VALUE - 
                state.slot_priority(player, tower.x, tower.y) * self.POSITION_PENALTY - 
                tower.level * self.LEVEL_PENALTY
            )
            
            results.append(ActionBundle(
                name=f"downgrade#{tower.tower_id}", 
                operations=(op,), 
                score=score, 
                tags=("sell",)
            ))
        return results

    def _base_upgrade_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        【基地升级候选生成】
        
        包含两种升级：蚂蚁升级 和 生产速度升级
        
        【蚂蚁升级】
        评估公式：
        score = ANT_BASE_SCORE + hp_gain * HP_GAIN_WEIGHT + frontline_distance * FRONTLINE_WEIGHT_ANT - round_index * ROUND_DECAY_ANT - level * LEVEL_PENALTY_ANT
        
        可调参数：
        - ANT_BASE_SCORE (默认8.0): 基础分
        - HP_GAIN_WEIGHT (默认1.4): 生命值增益权重
        - FRONTLINE_WEIGHT_ANT (默认0.22): 前线距离权重
          * 前线压力大时更想升级蚂蚁（增加生存能力）
        - ROUND_DECAY_ANT (默认0.01): 回合衰减
          * 后期升级价值降低（游戏快结束了）
        - LEVEL_PENALTY_ANT (默认1.2): 等级惩罚
          * 高等级时升级成本更高
        
        【生产速度升级】
        评估公式：
        score = GEN_BASE_SCORE + tempo_gain * TEMPO_GAIN_WEIGHT + nearest_ant_distance * NEAREST_ANT_WEIGHT - round_index * ROUND_DECAY_GEN
        
        可调参数：
        - GEN_BASE_SCORE (默认10.0): 基础分
        - TEMPO_GAIN_WEIGHT (默认14.0): 节奏增益权重
          * 生产周期缩短的价值（越大越重视经济）
        - NEAREST_ANT_WEIGHT (默认0.12): 最近蚂蚁距离权重
        - ROUND_DECAY_GEN (默认0.015): 回合衰减系数
        """
        results: list[ActionBundle] = []
        
        # ========== 蚂蚁升级 ==========
        if state.bases[player].ant_level < 2:  # 最高2级
            level = state.bases[player].ant_level
            # 计算生命值增益
            hp_gain = ANT_MAX_HP[level + 1] - ANT_MAX_HP[level]
            
            if hp_gain > 0:
                op = Operation(OperationType.UPGRADE_GENERATED_ANT)
                if state.can_apply_operation(player, op):
                    # 前线距离：前线压力大时更想升级
                    frontline_dist = state.frontline_distance(player)
                    
                    score = (
                        self.ANT_BASE_SCORE +                           # 基础分
                        hp_gain * self.HP_GAIN_WEIGHT +                 # 生命值增益价值
                        frontline_dist * self.FRONTLINE_WEIGHT_ANT -    # 前线距离权重
                        state.round_index * self.ROUND_DECAY_ANT -      # 回合衰减
                        level * self.LEVEL_PENALTY_ANT                  # 等级惩罚
                    )
                    
                    results.append(ActionBundle(
                        "upgrade-ant", (op,), score, ("base", "offense")
                    ))
        
        # ========== 生产速度升级 ==========
        if state.bases[player].generation_level < 2:  # 最高2级
            level = state.bases[player].generation_level
            current_cycle = ANT_GENERATION_CYCLE[level]
            next_cycle = ANT_GENERATION_CYCLE[level + 1]
            
            if next_cycle < current_cycle - 1e-6:  # 确实有提升
                op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
                if state.can_apply_operation(player, op):
                    # 节奏增益 = 生产周期缩短量
                    tempo_gain = current_cycle - next_cycle
                    
                    score = (
                        self.GEN_BASE_SCORE +                           # 基础分
                        tempo_gain * self.TEMPO_GAIN_WEIGHT +           # 节奏增益价值
                        state.nearest_ant_distance(player) * self.NEAREST_ANT_WEIGHT -  # 最近蚂蚁距离
                        state.round_index * self.ROUND_DECAY_GEN        # 回合衰减
                    )
                    
                    results.append(ActionBundle(
                        "upgrade-gen", (op,), score, ("base", "tempo")
                    ))
        
        return results

    def _superweapon_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        【超级武器候选生成】
        
        包含4种超级武器，每种都有使用阈值和成本惩罚
        
        【闪电风暴】
        - 用途：范围伤害敌方蚂蚁
        - 阈值：STORM_THRESHOLD (默认2.5)
        - 成本惩罚：STORM_COST_PENALTY (默认0.03)
        
        【EMP冲击】
        - 用途：瘫痪敌方塔
        - 阈值：EMP_THRESHOLD (默认2.0)
        - 成本惩罚：EMP_COST_PENALTY (默认0.025)
        
        【护盾】
        - 用途：保护我方蚂蚁
        - 阈值：DEFLECTOR_THRESHOLD (默认1.5)
        - 成本惩罚：DEFLECTOR_COST_PENALTY (默认0.02)
        
        【紧急回避】
        - 用途：紧急传送我方蚂蚁
        - 阈值：EVASION_THRESHOLD (默认1.0)
        - 成本惩罚：EVASION_COST_PENALTY (默认0.02)
        
        可调参数：
        - 降低阈值：更频繁使用武器
        - 提高阈值：更谨慎使用武器
        - 调整成本惩罚：平衡经济和发展
        """
        results: list[ActionBundle] = []
        enemy = 1 - player
        enemy_ants = state.ants_of(enemy)
        my_ants = state.ants_of(player)
        enemy_towers = state.towers_of(enemy)

        # ========== 闪电风暴 ==========
        if (enemy_ants and 
            state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0 and
            state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost):
            
            # 找到最佳目标（能消灭最多/最有价值的蚂蚁）
            best = max(
                ((ant.x, ant.y, self._storm_value(state, player, ant.x, ant.y)) 
                 for ant in enemy_ants),
                key=lambda item: item[2],
                default=None,
            )
            
            # 超过阈值才使用
            if best and best[2] > self.STORM_THRESHOLD:
                op = Operation(OperationType.USE_LIGHTNING_STORM, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"storm@{best[0]},{best[1]}", 
                        (op,), 
                        best[2], 
                        ("weapon", "storm")
                    ))

        # ========== EMP冲击 ==========
        if (enemy_towers and 
            state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER] == 0 and
            state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost):
            
            centers = {(tower.x, tower.y) for tower in enemy_towers}
            scored = [(x, y, self._emp_value(state, player, x, y)) for x, y in centers]
            best = max(scored, key=lambda item: item[2], default=None)
            
            if best and best[2] > self.EMP_THRESHOLD:
                op = Operation(OperationType.USE_EMP_BLASTER, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"emp@{best[0]},{best[1]}", 
                        (op,), 
                        best[2], 
                        ("weapon", "emp")
                    ))

        # ========== 护盾 ==========
        if (my_ants and 
            state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR] == 0 and
            state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost):
            
            best = max(
                ((ant.x, ant.y, self._deflector_value(state, player, ant.x, ant.y)) 
                 for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            
            if best and best[2] > self.DEFLECTOR_THRESHOLD:
                op = Operation(OperationType.USE_DEFLECTOR, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"deflect@{best[0]},{best[1]}", 
                        (op,), 
                        best[2], 
                        ("weapon", "shield")
                    ))

        # ========== 紧急回避 ==========
        if (my_ants and 
            state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION] == 0 and
            state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost):
            
            best = max(
                ((ant.x, ant.y, self._evasion_value(state, player, ant.x, ant.y)) 
                 for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            
            if best and best[2] > self.EVASION_THRESHOLD:
                op = Operation(OperationType.USE_EMERGENCY_EVASION, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"evasion@{best[0]},{best[1]}", 
                        (op,), 
                        best[2], 
                        ("weapon", "panic")
                    ))

        return results

    def _paired_candidates(self, state: BackendState, player: int, singles: list[ActionBundle]) -> list[ActionBundle]:
        """
        【组合操作候选生成】
        
        将两个单操作组合在一起（如：卖塔+建塔，升级+升级等）
        
        评估公式：
        score = first.score + second.score * SECOND_ACTION_WEIGHT
        
        可调参数：
        - SECOND_ACTION_WEIGHT (默认0.9): 第二个操作权重
          * 避免过度乐观（两个操作不一定都能成功）
          * 增大：更乐观，更愿意组合操作
          * 减小：更保守，倾向于单操作
        
        限制：
        - 只考虑前8个单操作（减少计算量）
        - 最多2个操作组合
        - 必须都能合法执行
        """
        results: list[ActionBundle] = []
        
        # 只考虑sell/build/upgrade/base类型的操作
        left = [bundle for bundle in singles 
                if bundle.tags and bundle.tags[0] in {"sell", "build", "upgrade", "base"}]
        
        # 取前8个（减少组合数量）
        left = sorted(left, key=lambda item: item.score, reverse=True)[:8]
        
        # 两两组合
        for first in left:
            for second in left:
                if first is second:
                    continue
                
                operations = first.operations + second.operations
                if len(operations) > 2:
                    continue
                
                # 验证组合是否合法
                trial = state.clone()
                accepted: list[Operation] = []
                legal = True
                
                for op in operations:
                    if not trial.can_apply_operation(player, op, accepted):
                        legal = False
                        break
                    accepted.append(op)
                
                if not legal:
                    continue
                
                # 评估分数 = 第一个操作 + 第二个操作 * 权重
                score = first.score + second.score * self.SECOND_ACTION_WEIGHT
                name = f"{first.name}+{second.name}"
                
                results.append(ActionBundle(
                    name=name, 
                    operations=tuple(operations), 
                    score=score, 
                    tags=("combo",)
                ))
        
        return results

    def _rerank_with_one_step_rollout(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> list[ActionBundle]:
        """
        【使用一步推演重新排序】
        
        评估公式：
        new_score = original_score + rollout_value * ROLLOUT_WEIGHT
        
        其中 rollout_value = evaluate(after_state) - evaluate(current_state)
        
        作用：
        - 不仅看操作的即时收益，还看操作后的状态变化
        - 避免短视的决策
        
        可调参数：
        - ROLLOUT_WEIGHT (默认0.2): 推演价值权重
          * 增大：更重视长远影响
          * 减小：更重视即时收益
          * 设为0：禁用推演，只用原始分数
        
        注意：
        - 需要克隆状态并模拟执行，计算成本较高
        - 只对前max_actions*2个候选进行推演（减少计算量）
        """
        # 计算当前状态的基线评估值
        baseline = self.feature_extractor.evaluate(state, player)
        
        reranked: list[ActionBundle] = []
        
        for bundle in bundles:
            # 克隆状态并执行操作
            trial = state.clone()
            trial.apply_operation_list(player, bundle.operations)
            trial.advance_round()  # 推进一回合
            
            # 计算推演后的评估值变化
            rollout_value = self.feature_extractor.evaluate(trial, player) - baseline
            
            # 新分数 = 原分数 + 推演价值 * 权重
            new_score = bundle.score + rollout_value * self.ROLLOUT_WEIGHT
            
            reranked.append(ActionBundle(
                bundle.name, 
                bundle.operations, 
                new_score, 
                bundle.tags
            ))
        
        # 按新分数排序
        reranked.sort(key=lambda item: item.score, reverse=True)
        
        if not reranked:
            return [ActionBundle(name="hold")]
        
        return reranked

    def _local_enemy_pressure(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        【计算当地敌方压力】
        
        计算指定位置周围的敌方蚂蚁威胁程度
        
        公式：
        pressure = sum(max(0, 6.5 - distance) * (1.0 + ant.level * 0.4))
        
        参数：
        - 6.5: 有效距离上限（距离超过6.5的不计入）
        - 0.4: 等级权重（高等级蚂蚁威胁更大）
        
        返回值越大表示压力越大，需要更多防守
        """
        pressure = 0.0
        
        for ant in state.ants_of(1 - player):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= 6:
                # 距离越近威胁越大，等级越高威胁越大
                pressure += max(0.0, 6.5 - distance) * (1.0 + ant.level * 0.4)
        
        return pressure

    def _tower_type_fit(self, tower_type: TowerType, local_density: float, forward_distance: int) -> float:
        """
        【计算塔类型与环境的适配度】
        
        根据塔的类型、当地敌方密度、距离敌方基地的距离，计算适配度
        
        可调参数（各种塔类型的权重）：
        
        【重型塔】HEAVY, HEAVY_PLUS, BEWITCH
        - 适合：高密度区域
        - 不适合：太靠前的位置
        - 公式：local_density * 1.1 - forward_distance * 0.1
        
        【冰冻/脉冲塔】ICE, PULSE
        - 适合：高密度区域（控制效果）
        - 公式：local_density * 1.3
        
        【迫击炮/导弹塔】MORTAR, MORTAR_PLUS, MISSILE
        - 适合：中等密度，中等距离
        - 公式：local_density * 0.85 + max(0, 12 - forward_distance)
        
        【快速塔】QUICK, QUICK_PLUS, DOUBLE
        - 适合：任何位置（快速攻击）
        - 有基础分
        - 公式：local_density * 0.9 + 4.0
        
        【狙击塔】SNIPER
        - 适合：远距离（射程远）
        - 公式：max(0, 18 - forward_distance) + local_density * 0.4
        
        【生产塔】PRODUCER系列
        - 适合：低密度（安全），不要太靠前
        - 不同分支有不同奖励
        - 公式较复杂，包含密度、距离、生产速度、分支奖励
        """
        
        # 重型塔：适合防守，不要太靠前
        if tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS, TowerType.BEWITCH):
            return local_density * self.HEAVY_DENSITY_WEIGHT - forward_distance * self.HEAVY_DISTANCE_PENALTY
        
        # 冰冻/脉冲塔：适合高密度区域（控制）
        if tower_type in (TowerType.ICE, TowerType.PULSE):
            return local_density * self.ICE_DENSITY_WEIGHT
        
        # 迫击炮/导弹塔：适合中等距离
        if tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS, TowerType.MISSILE):
            return local_density * self.MORTAR_DENSITY_WEIGHT + max(0.0, self.MORTAR_DISTANCE_WEIGHT - forward_distance)
        
        # 快速塔：通用，有基础分
        if tower_type in (TowerType.QUICK, TowerType.QUICK_PLUS, TowerType.DOUBLE):
            return local_density * self.QUICK_DENSITY_WEIGHT + self.QUICK_BASE_BONUS
        
        # 狙击塔：适合远距离
        if tower_type == TowerType.SNIPER:
            return max(0.0, self.SNIPER_DISTANCE_WEIGHT - forward_distance) + local_density * self.SNIPER_DENSITY_WEIGHT
        
        # 生产塔：复杂计算
        if tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC):
            stats = TOWER_STATS[tower_type]
            
            # 生产速度 = 12 / 生产间隔
            cadence = 12.0 / max(stats.spawn_interval, 1)
            
            # 密度奖励：低密度区域更适合（安全）
            density_bonus = max(0.0, self.PRODUCER_DENSITY_BONUS - local_density) * self.PRODUCER_DENSITY_WEIGHT
            
            # 前线奖励：不要太靠前
            forward_bonus = max(0.0, self.PRODUCER_DISTANCE_WEIGHT - forward_distance) * self.PRODUCER_FORWARD_WEIGHT
            
            # 分支奖励：不同升级路线有不同价值
            branch_bonus = {
                TowerType.PRODUCER: self.PRODUCER_BRANCH_BONUS,
                TowerType.PRODUCER_FAST: self.PRODUCER_FAST_BONUS,
                TowerType.PRODUCER_SIEGE: self.PRODUCER_SIEGE_BONUS,
                TowerType.PRODUCER_MEDIC: self.PRODUCER_MEDIC_BONUS,
            }[tower_type]
            
            return density_bonus + forward_bonus + cadence + branch_bonus
        
        return 0.0

    def _storm_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        【闪电风暴价值评估】
        
        评估在指定位置使用闪电风暴的价值
        
        公式：
        value = sum(kill_reward + (4 - distance) * 0.5) - cost * STORM_COST_PENALTY
        
        考虑因素：
        - 能消灭的蚂蚁价值
        - 距离中心越近的蚂蚁受到的伤害越大
        - 使用成本
        """
        enemy = 1 - player
        total = 0.0
        
        for ant in state.ants_of(enemy):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].attack_range:
                # 消灭奖励 + 距离修正（越近伤害越高）
                total += ant.kill_reward + (4 - distance) * 0.5
        
        # 减去成本惩罚
        return total - SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost * self.STORM_COST_PENALTY

    def _emp_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        【EMP冲击价值评估】
        
        评估在指定位置使用EMP的价值
        
        公式：
        value = sum(3.0 + tower.level * 2.5) - cost * EMP_COST_PENALTY
        
        考虑因素：
        - 能瘫痪的塔数量
        - 塔的等级（高等级塔更有价值）
        - 使用成本
        """
        total = 0.0
        
        for tower in state.towers_of(1 - player):
            distance = hex_distance(x, y, tower.x, tower.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].attack_range:
                # 基础价值 + 等级价值
                total += 3.0 + tower.level * 2.5
        
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost * self.EMP_COST_PENALTY

    def _deflector_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        【护盾价值评估】
        
        评估在指定位置使用护盾的价值
        
        公式：
        value = sum(0.8 + ant.level * 0.8) + max(0, 7 - nearest_distance) * 0.5 - cost * DEFLECTOR_COST_PENALTY
        
        考虑因素：
        - 能保护的蚂蚁数量和等级
        - 最近蚂蚁的距离（紧急程度）
        - 使用成本
        """
        total = 0.0
        
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].attack_range:
                # 保护价值 = 基础 + 等级
                total += 0.8 + ant.level * 0.8
        
        # 紧急程度加成（蚂蚁离基地越近越紧急）
        total += max(0.0, 7 - state.nearest_ant_distance(player)) * 0.5
        
        return total - SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost * self.DEFLECTOR_COST_PENALTY

    def _evasion_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        【紧急回避价值评估】
        
        评估在指定位置使用紧急回避的价值
        
        公式：
        value = sum(0.6 + ant.level * 0.7) + max(0, 5 - nearest_distance) - cost * EVASION_COST_PENALTY
        
        考虑因素：
        - 能传送的蚂蚁数量和等级
        - 最近蚂蚁的距离（紧急程度）
        - 使用成本
        
        注意：紧急回避的阈值最低(1.0)，用于危急情况
        """
        total = 0.0
        
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].attack_range:
                total += 0.6 + ant.level * 0.7
        
        # 紧急程度加成
        total += max(0.0, 5 - state.nearest_ant_distance(player))
        
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost * self.EVASION_COST_PENALTY


# ==================== 使用示例 ====================
"""
如何调整参数：

1. 创建自定义的ActionCatalog子类：

class MyActionCatalog(ActionCatalog):
    # 调整建造塔参数（更重视防守）
    PRESSURE_WEIGHT = 3.5  # 原来是2.5
    BUILD_COST_PENALTY = 0.02  # 原来是0.03，更愿意花钱
    
    # 调整升级参数（更重视等级）
    LEVEL_BONUS = 2.0  # 原来是1.5
    
    # 禁用推演（加快计算）
    ROLLOUT_WEIGHT = 0.0  # 原来是0.2

2. 在AI中使用自定义的ActionCatalog：

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.catalog = MyActionCatalog()  # 使用自定义的目录
    
    def list_bundles(self, state, player):
        return self.catalog.build(state, player)

3. 测试不同参数的效果：
   - 每次只改1-2个参数
   - 使用测试环境对比效果
   - 记录胜率变化
"""
