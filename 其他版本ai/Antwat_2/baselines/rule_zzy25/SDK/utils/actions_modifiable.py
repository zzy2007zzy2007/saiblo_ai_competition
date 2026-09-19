#!/usr/bin/env python3
"""
ActionCatalog 可修改公式标记版
==========================

本文件标记了所有可修改的评估公式，每个公式都有详细说明。

标记说明：
- 📝 可修改：可以直接调整参数或公式结构
- 🔄 建议改进：推荐尝试不同的计算方式
- 🚀 强化学习：适合作为强化学习的输入特征

作者: AI Assistant
日期: 2026-04-03
"""

from __future__ import annotations

import json
import os
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
    name: str
    operations: tuple[Operation, ...] = ()
    score: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def protocol_lines(self) -> list[list[int]]:
        return [op.to_protocol_tokens() for op in self.operations]


class ActionCatalog:
    """
    操作目录：生成并评估所有可能的操作
    """
    
    def __init__(self, max_actions: int = MAX_ACTIONS, feature_extractor: FeatureExtractor | None = None, params_file: str | None = None) -> None:
        self.max_actions = max_actions
        self.feature_extractor = feature_extractor or FeatureExtractor(max_actions=max_actions)
        self.params_file = params_file
        self._load_params()
    
    def _load_params(self) -> None:
        """
        从参数文件加载配置
        """
        # 默认参数
        default_params = {
            # 【建造塔参数】
            "PRESSURE_WEIGHT": 2.5,           # 敌方压力权重
            "BUILD_COST_PENALTY": 0.03,       # 建造成本惩罚
            
            # 【升级塔参数】
            "LEVEL_BONUS": 1.5,               # 等级奖励
            "POSITION_WEIGHT": 0.15,          # 位置权重
            
            # 【降级/出售参数】
            "REFUND_VALUE": 0.04,             # 退款价值
            "POSITION_PENALTY": 0.3,          # 位置惩罚
            "LEVEL_PENALTY": 3.0,             # 等级惩罚
            "PRESSURE_THRESHOLD": 1.5,        # 压力阈值
            
            # 【基地升级-蚂蚁参数】
            "ANT_BASE_SCORE": 8.0,            # 基础分
            "HP_GAIN_WEIGHT": 1.4,            # 生命值增益权重
            "FRONTLINE_WEIGHT_ANT": 0.22,     # 前线距离权重
            "ROUND_DECAY_ANT": 0.01,          # 回合衰减
            "LEVEL_PENALTY_ANT": 1.2,         # 等级惩罚
            
            # 【基地升级-生产速度参数】
            "GEN_BASE_SCORE": 10.0,           # 基础分
            "TEMPO_GAIN_WEIGHT": 14.0,        # 节奏增益权重
            "NEAREST_ANT_WEIGHT": 0.12,       # 最近蚂蚁距离权重
            "ROUND_DECAY_GEN": 0.015,         # 回合衰减
            
            # 【组合操作参数】
            "SECOND_ACTION_WEIGHT": 0.9,      # 第二个操作权重
            
            # 【重新排序参数】
            "ROLLOUT_WEIGHT": 0.2,            # 推演价值权重
            
            # 【超级武器阈值】
            "STORM_THRESHOLD": 2.5,           # 闪电风暴阈值
            "EMP_THRESHOLD": 2.0,             # EMP阈值
            "DEFLECTOR_THRESHOLD": 1.5,       # 护盾阈值
            "EVASION_THRESHOLD": 1.0,         # 紧急回避阈值
            
            # 【超级武器成本惩罚】
            "STORM_COST_PENALTY": 0.03,       # 闪电风暴成本惩罚
            "EMP_COST_PENALTY": 0.025,        # EMP成本惩罚
            "DEFLECTOR_COST_PENALTY": 0.02,   # 护盾成本惩罚
            "EVASION_COST_PENALTY": 0.02      # 紧急回避成本惩罚
        }
        
        # 从文件加载参数
        if self.params_file and os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r', encoding='utf-8') as f:
                    loaded_params = json.load(f)
                # 合并参数
                for key, value in loaded_params.items():
                    setattr(self, key, value)
                # 设置剩余的默认参数
                for key, value in default_params.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
            except Exception as e:
                print(f"警告: 无法加载参数文件 {self.params_file}: {e}")
                # 使用默认参数
                for key, value in default_params.items():
                    setattr(self, key, value)
        else:
            # 使用默认参数
            for key, value in default_params.items():
                setattr(self, key, value)

    def build(self, state: BackendState, player: int) -> list[ActionBundle]:
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
            if key not in unique or bundle.score > unique[key].score:
                unique[key] = bundle
        
        # 排序
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        
        # 重新排序（使用一步推演）
        reranked = self._rerank_with_one_step_rollout(
            state, player, 
            ordered[: min(len(ordered), self.max_actions * 2)]
        )
        
        return reranked[: self.max_actions]

    def _build_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        建造塔候选生成
        
        📝 可修改公式：
        score = lane_bonus + pressure * PRESSURE_WEIGHT - build_cost * BUILD_COST_PENALTY
        
        🔄 建议改进：
        - 加入塔类型的影响
        - 考虑未来收益
        - 加入经济状况
        """
        results: list[ActionBundle] = []
        tower_count = state.tower_count(player)
        build_cost = state.build_tower_cost(tower_count)
        
        if state.coins[player] < build_cost:
            return results
        
        for x, y in STRATEGIC_BUILD_ORDER[player]:
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if not state.can_apply_operation(player, op):
                continue
            
            pressure = self._local_enemy_pressure(state, player, x, y)
            lane_bonus = state.slot_priority(player, x, y)
            
            # 📝 可修改公式：建造塔评分
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
        升级塔候选生成
        
        📝 可修改公式：
        score = fit + tower.level * LEVEL_BONUS + slot_priority * POSITION_WEIGHT
        
        🔄 建议改进：
        - 加入升级成本的考虑
        - 考虑升级后的长期收益
        - 加入与其他塔的协同效应
        """
        results: list[ActionBundle] = []
        enemy_base = PLAYER_BASES[1 - player]
        
        for tower in state.towers_of(player):
            local_density = self._local_enemy_pressure(state, player, tower.x, tower.y)
            
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ()):
                op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
                if not state.can_apply_operation(player, op):
                    continue
                
                fit = self._tower_type_fit(
                    target, 
                    local_density, 
                    hex_distance(tower.x, tower.y, *enemy_base)
                )
                
                # 📝 可修改公式：升级塔评分
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
        降级/出售塔候选生成
        
        📝 可修改公式：
        score = refund * REFUND_VALUE - slot_priority * POSITION_PENALTY - tower.level * LEVEL_PENALTY
        
        🔄 建议改进：
        - 加入当前经济状况的影响
        - 考虑出售后的重新投资价值
        - 加入塔的历史贡献
        """
        results: list[ActionBundle] = []
        
        for tower in state.towers_of(player):
            pressure = self._local_enemy_pressure(state, player, tower.x, tower.y)
            
            # 📝 可修改参数：压力阈值
            if pressure > self.PRESSURE_THRESHOLD:
                continue
            
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not state.can_apply_operation(player, op):
                continue
            
            refund = state.operation_income(player, op)
            
            # 📝 可修改公式：降级/出售评分
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
        基地升级候选生成
        
        包含两种升级：蚂蚁升级 和 生产速度升级
        """
        results: list[ActionBundle] = []
        
        # ========== 蚂蚁升级 ==========
        if state.bases[player].ant_level < 2:
            level = state.bases[player].ant_level
            hp_gain = ANT_MAX_HP[level + 1] - ANT_MAX_HP[level]
            
            if hp_gain > 0:
                op = Operation(OperationType.UPGRADE_GENERATED_ANT)
                if state.can_apply_operation(player, op):
                    frontline_dist = state.frontline_distance(player)
                    
                    # 📝 可修改公式：蚂蚁升级评分
                    score = (
                        self.ANT_BASE_SCORE +                           # 基础分
                        hp_gain * self.HP_GAIN_WEIGHT +                 # 生命值增益
                        frontline_dist * self.FRONTLINE_WEIGHT_ANT -    # 前线距离
                        state.round_index * self.ROUND_DECAY_ANT -      # 回合衰减
                        level * self.LEVEL_PENALTY_ANT                  # 等级惩罚
                    )
                    
                    results.append(ActionBundle(
                        "upgrade-ant", (op,), score, ("base", "offense")
                    ))
        
        # ========== 生产速度升级 ==========
        if state.bases[player].generation_level < 2:
            level = state.bases[player].generation_level
            current_cycle = ANT_GENERATION_CYCLE[level]
            next_cycle = ANT_GENERATION_CYCLE[level + 1]
            
            if next_cycle < current_cycle - 1e-6:
                op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
                if state.can_apply_operation(player, op):
                    tempo_gain = current_cycle - next_cycle
                    
                    # 📝 可修改公式：生产速度升级评分
                    score = (
                        self.GEN_BASE_SCORE +                           # 基础分
                        tempo_gain * self.TEMPO_GAIN_WEIGHT +           # 节奏增益
                        state.nearest_ant_distance(player) * self.NEAREST_ANT_WEIGHT -  # 最近蚂蚁距离
                        state.round_index * self.ROUND_DECAY_GEN        # 回合衰减
                    )
                    
                    results.append(ActionBundle(
                        "upgrade-gen", (op,), score, ("base", "tempo")
                    ))
        
        return results

    def _superweapon_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        """
        超级武器候选生成
        
        包含4种超级武器，每种都有使用阈值和成本惩罚
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
            
            best = max(
                ((ant.x, ant.y, self._storm_value(state, player, ant.x, ant.y)) 
                 for ant in enemy_ants),
                key=lambda item: item[2],
                default=None,
            )
            
            # 📝 可修改参数：使用阈值
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
            
            # 📝 可修改参数：使用阈值
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
            
            # 📝 可修改参数：使用阈值
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
            
            # 📝 可修改参数：使用阈值
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
        组合操作候选生成
        
        📝 可修改公式：
        score = first.score + second.score * SECOND_ACTION_WEIGHT
        
        🔄 建议改进：
        - 考虑操作之间的协同效应
        - 加入成本的综合考虑
        - 考虑操作顺序的影响
        """
        results: list[ActionBundle] = []
        
        left = [bundle for bundle in singles 
                if bundle.tags and bundle.tags[0] in {"sell", "build", "upgrade", "base"}]
        left = sorted(left, key=lambda item: item.score, reverse=True)[:8]
        
        for first in left:
            for second in left:
                if first is second:
                    continue
                
                operations = first.operations + second.operations
                if len(operations) > 2:
                    continue
                
                # 验证合法性
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
                
                # 📝 可修改公式：组合操作评分
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
        使用一步推演重新排序
        
        📝 可修改公式：
        new_score = original_score + rollout_value * ROLLOUT_WEIGHT
        
        🔄 建议改进：
        - 考虑多步推演
        - 加入不确定性因素
        - 考虑不同对手的反应
        """
        baseline = self.feature_extractor.evaluate(state, player)
        
        reranked: list[ActionBundle] = []
        
        for bundle in bundles:
            trial = state.clone()
            trial.apply_operation_list(player, bundle.operations)
            trial.advance_round()
            
            rollout_value = self.feature_extractor.evaluate(trial, player) - baseline
            
            # 📝 可修改公式：重新排序评分
            new_score = bundle.score + rollout_value * self.ROLLOUT_WEIGHT
            
            reranked.append(ActionBundle(
                bundle.name, 
                bundle.operations, 
                new_score, 
                bundle.tags
            ))
        
        reranked.sort(key=lambda item: item.score, reverse=True)
        
        if not reranked:
            return [ActionBundle(name="hold")]
        
        return reranked

    def _local_enemy_pressure(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        计算当地敌方压力
        
        📝 可修改公式：
        pressure = sum(max(0, 6.5 - distance) * (1.0 + ant.level * 0.4))
        
        🔄 建议改进：
        - 考虑蚂蚁的生命值
        - 考虑蚂蚁的类型
        - 考虑蚂蚁的移动方向
        - 考虑多个蚂蚁的协同效应
        """
        pressure = 0.0
        
        for ant in state.ants_of(1 - player):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= 6:
                # 📝 可修改公式：敌方压力计算
                pressure += max(0.0, 6.5 - distance) * (1.0 + ant.level * 0.4)
        
        return pressure

    def _tower_type_fit(self, tower_type: TowerType, local_density: float, forward_distance: int) -> float:
        """
        计算塔类型与环境的适配度
        
        📝 可修改公式：根据不同塔类型的适配度计算
        
        🔄 建议改进：
        - 加入与其他塔的协同效应
        - 考虑敌人的类型
        - 考虑地形因素
        - 考虑游戏阶段
        """
        
        # 重型塔：适合防守，不要太靠前
        if tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS, TowerType.BEWITCH):
            # 📝 可修改公式：重型塔适配度
            return local_density * 1.1 - forward_distance * 0.1
        
        # 冰冻/脉冲塔：适合高密度区域
        if tower_type in (TowerType.ICE, TowerType.PULSE):
            # 📝 可修改公式：冰冻/脉冲塔适配度
            return local_density * 1.3
        
        # 迫击炮/导弹塔：适合中等距离
        if tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS, TowerType.MISSILE):
            # 📝 可修改公式：迫击炮/导弹塔适配度
            return local_density * 0.85 + max(0.0, 12 - forward_distance)
        
        # 快速塔：通用，有基础分
        if tower_type in (TowerType.QUICK, TowerType.QUICK_PLUS, TowerType.DOUBLE):
            # 📝 可修改公式：快速塔适配度
            return local_density * 0.9 + 4.0
        
        # 狙击塔：适合远距离
        if tower_type == TowerType.SNIPER:
            # 📝 可修改公式：狙击塔适配度
            return max(0.0, 18 - forward_distance) + local_density * 0.4
        
        # 生产塔：复杂计算
        if tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC):
            stats = TOWER_STATS[tower_type]
            
            # 生产速度 = 12 / 生产间隔
            cadence = 12.0 / max(stats.spawn_interval, 1)
            
            # 密度奖励：低密度区域更适合
            density_bonus = max(0.0, 10 - local_density) * 0.75
            
            # 前线奖励：不要太靠前
            forward_bonus = max(0.0, 16 - forward_distance) * 0.22
            
            # 分支奖励：不同升级路线有不同价值
            branch_bonus = {
                TowerType.PRODUCER: 0.0,
                TowerType.PRODUCER_FAST: 0.5,
                TowerType.PRODUCER_SIEGE: 1.1,
                TowerType.PRODUCER_MEDIC: 1.0,
            }[tower_type]
            
            # 📝 可修改公式：生产塔适配度
            return density_bonus + forward_bonus + cadence + branch_bonus
        
        return 0.0

    def _storm_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        闪电风暴价值评估
        
        📝 可修改公式：
        value = sum(kill_reward + (4 - distance) * 0.5) - cost * STORM_COST_PENALTY
        
        🔄 建议改进：
        - 考虑蚂蚁的生命值
        - 考虑蚂蚁的位置重要性
        - 考虑使用时机（比如敌人波次）
        """
        enemy = 1 - player
        total = 0.0
        
        for ant in state.ants_of(enemy):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].attack_range:
                # 📝 可修改公式：闪电风暴价值
                total += ant.kill_reward + (4 - distance) * 0.5
        
        # 📝 可修改参数：成本惩罚
        return total - SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost * self.STORM_COST_PENALTY

    def _emp_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        EMP冲击价值评估
        
        📝 可修改公式：
        value = sum(3.0 + tower.level * 2.5) - cost * EMP_COST_PENALTY
        
        🔄 建议改进：
        - 考虑塔的类型
        - 考虑塔的位置重要性
        - 考虑多个塔的协同效应
        """
        total = 0.0
        
        for tower in state.towers_of(1 - player):
            distance = hex_distance(x, y, tower.x, tower.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].attack_range:
                # 📝 可修改公式：EMP价值
                total += 3.0 + tower.level * 2.5
        
        # 📝 可修改参数：成本惩罚
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost * self.EMP_COST_PENALTY

    def _deflector_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        护盾价值评估
        
        📝 可修改公式：
        value = sum(0.8 + ant.level * 0.8) + max(0, 7 - nearest_distance) * 0.5 - cost * DEFLECTOR_COST_PENALTY
        
        🔄 建议改进：
        - 考虑蚂蚁的生命值
        - 考虑蚂蚁的类型
        - 考虑蚂蚁的位置重要性
        """
        total = 0.0
        
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].attack_range:
                # 📝 可修改公式：护盾价值
                total += 0.8 + ant.level * 0.8
        
        # 紧急程度加成
        total += max(0.0, 7 - state.nearest_ant_distance(player)) * 0.5
        
        # 📝 可修改参数：成本惩罚
        return total - SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost * self.DEFLECTOR_COST_PENALTY

    def _evasion_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        """
        紧急回避价值评估
        
        📝 可修改公式：
        value = sum(0.6 + ant.level * 0.7) + max(0, 5 - nearest_distance) - cost * EVASION_COST_PENALTY
        
        🔄 建议改进：
        - 考虑蚂蚁的生命值
        - 考虑蚂蚁的类型
        - 考虑蚂蚁的位置重要性
        - 考虑紧急程度
        """
        total = 0.0
        
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].attack_range:
                # 📝 可修改公式：紧急回避价值
                total += 0.6 + ant.level * 0.7
        
        # 紧急程度加成
        total += max(0.0, 5 - state.nearest_ant_distance(player))
        
        # 📝 可修改参数：成本惩罚
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost * self.EVASION_COST_PENALTY


# ==================== 特征提取示例 ====================
# 强化学习特征提取示例：
# 
# def extract_features(state, player, action):
#     """
#     提取用于强化学习的特征
#     
#     🚀 强化学习：适合作为神经网络输入
#     """
#     features = {}
#     
#     # 1. 经济特征
#     features['coins'] = state.coins[player]
#     features['tower_count'] = state.tower_count(player)
#     features['build_cost'] = state.build_tower_cost(features['tower_count'])
#     
#     # 2. 压力特征
#     if action.tags and 'build' in action.tags:
#         # 提取建造位置
#         x, y = action.name.split('@')[1].split(',')
#         features['local_pressure'] = _local_enemy_pressure(state, player, int(x), int(y))
#         features['lane_bonus'] = state.slot_priority(player, int(x), int(y))
#     
#     # 3. 塔特征
#     if action.tags and 'upgrade' in action.tags:
#         tower_id = int(action.name.split('#')[1].split('->')[0])
#         tower = state.tower_by_id(tower_id)
#         if tower:
#             features['tower_level'] = tower.level
#             features['tower_type'] = tower.tower_type
#     
#     # 4. 基地特征
#     features['ant_level'] = state.bases[player].ant_level
#     features['generation_level'] = state.bases[player].generation_level
#     features['base_hp'] = state.bases[player].hp
#     
#     # 5. 时间特征
#     features['round_index'] = state.round_index
#     features['game_phase'] = 0 if state.round_index < 170 else (1 if state.round_index < 340 else 2)
#     
#     # 6. 武器特征
#     features['weapon_cooldowns'] = state.weapon_cooldowns[player].tolist()
#     
#     return features

# ==================== 线性模型示例 ====================
# 线性模型评估示例：
# 
# class LinearEvaluator:
#     """
#     线性模型评估器
#     
#     📝 可修改：权重可以通过强化学习训练
#     """
#     def __init__(self, weights=None):
#         self.weights = weights or {
#             'coins': 0.01,
#             'local_pressure': 2.5,
#             'lane_bonus': 1.0,
#             'build_cost': -0.03,
#             'tower_level': 1.5,
#             'ant_level': 0.5,
#             'generation_level': 0.8,
#             'round_index': -0.01,
#             'game_phase': -0.5,
#         }
#     
#     def evaluate(self, state, player, action):
#         features = extract_features(state, player, action)
#         score = sum(features.get(feature, 0) * weight 
#                    for feature, weight in self.weights.items())
#         return score