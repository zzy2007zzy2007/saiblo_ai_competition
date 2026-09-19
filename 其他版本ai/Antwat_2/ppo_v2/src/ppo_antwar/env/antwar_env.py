from typing import Any, Dict, Optional, Tuple

import numpy as np

from loguru import logger

from ..compat.adapters import BackendAdapter
from ..utils.action_constants import (
    MAX_ROUND,
    REWARD_CONFIG,
    TOWER_POSITIONS,
    TYPE_CONFIG,
    _UPGRADE_DIRECTIONS,
)
from .action_mask import ActionMaskHandler
from .observation import ObservationEncoder
from .player_state import PlayerState
from .state_snapshot import StateSnapshot

_POSITION_HASH_MULTIPLIER = 100


class AntWarEnv:
    """AntWar 游戏环境 - 封装游戏后端 SDK，提供 Gymnasium 风格接口。"""

    _ACTION_REWARD_DISPATCH = {
        "build_tower": "_compute_build_tower_reward",
        "upgrade_tower": "_compute_upgrade_tower_reward",
        "downgrade_tower": "_compute_downgrade_tower_reward",
        "lightning_storm": "_compute_fixed_deploy_reward",
        "emp_blaster": "_compute_fixed_deploy_reward",
        "deflector": "_compute_fixed_deploy_reward",
        "evasion": "_compute_fixed_deploy_reward",
        "tech_upgrade": "_compute_tech_upgrade_reward",
    }

    def __init__(
        self,
        player_id: int,
        backend_type: str,
        prefer_native: bool = False,
    ):
        self._player_id = player_id
        self._backend_type = backend_type
        self._prefer_native = prefer_native
        self._runtime = None
        self._tower_build_slot_counts: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
        self._last_reward_detail: Dict[str, Any] = {}

        self.action_mask_handler = ActionMaskHandler()
        self.observation_encoder = ObservationEncoder(self.action_mask_handler)

    def reset(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
        """重置环境到初始状态，返回 (observations, info)"""
        self._runtime = BackendAdapter.create_runtime(player=self._player_id)
        self._tower_build_slot_counts = {0: {}, 1: {}}
        self._last_reward_detail = {}

        state = self._runtime.state
        obs_0 = self.observation_encoder.encode(state, 0)
        obs_1 = self.observation_encoder.encode(state, 1)

        observations = {
            "self": obs_0,
            "opponent": obs_1,
        }
        info = {"round_index": 0}
        return observations, info

    def step(
        self, action_self: int, action_opponent: int, self_first: bool
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Tuple[float, float],
        bool,
        bool,
        Dict[str, Any],
    ]:
        """执行双方动作并推进游戏回合。

        Args:
            action_self: 己方动作 ID
            action_opponent: 对方动作 ID
            self_first: 是否将己方动作映射为 player_0

        Returns:
            (observations, (reward_self, reward_opponent), terminated, truncated, info)
        """
        state = self._runtime.state
        self_player = 0 if self_first else 1

        # 动作 → 操作（直接使用 self_player/opponent_player 映射）
        self_ops = []
        opponent_ops = []

        if action_self != 0:
            op = self.action_mask_handler.action_id_to_op(
                action_self, state, self_player
            )
            if op is not None:
                self_ops.append(op)

        if action_opponent != 0:
            op = self.action_mask_handler.action_id_to_op(
                action_opponent, state, 1 - self_player
            )
            if op is not None:
                opponent_ops.append(op)

        # 动作奖励（只算 self）
        reward_self_action = self._compute_action_reward(action_self, self_player)

        try:
            reward_self, reward_opponent, terminated, truncated, info = (
                self._resolve_turn_from_ops(self_ops, opponent_ops, self_first)
            )
            reward_self += reward_self_action
            if "reward_detail" in info:
                info["reward_detail"]["action_reward"] = round(reward_self_action, 4)
        except Exception as e:
            logger.error(f"[AntWarEnv] _resolve_turn_from_ops failed: {e}")
            # 返回安全默认值，防止训练崩溃
            observations = {}
            reward_self = 0.0
            reward_opponent = 0.0
            terminated = True
            truncated = False
            info = {"round_index": 0, "error": str(e)[:200]}

        if not terminated:
            state = self._runtime.state
            observations = {
                "self": self.observation_encoder.encode(state, self_player),
                "opponent": self.observation_encoder.encode(state, 1 - self_player),
            }
        else:
            observations = {}

        return observations, (reward_self, reward_opponent), terminated, truncated, info

    def get_state_snapshot(self, player: int) -> StateSnapshot:
        """公共接口：获取当前状态的快照，self_state 为请求方的状态。

        Args:
            player: 请求方 player ID (0 或 1)

        Returns:
            StateSnapshot: self_state 为请求方状态，opponent_state 为对方状态
        """
        raw = self._snapshot_state(self._runtime.state)
        if player == 1:
            return StateSnapshot(
                self_state=raw.opponent_state,
                opponent_state=raw.self_state,
                round_index=raw.round_index,
            )
        return raw

    @staticmethod
    def _snapshot_state(state) -> StateSnapshot:
        towers_0 = state.towers_of(0)
        towers_1 = state.towers_of(1)

        tower_hp_0 = {}
        tower_hp_1 = {}
        for tower in getattr(state, "towers", []):
            if tower.player == 0:
                tower_hp_0[tower.tower_id] = tower.hp
            elif tower.player == 1:
                tower_hp_1[tower.tower_id] = tower.hp

        return StateSnapshot(
            self_state=PlayerState(
                hp=state.bases[0].hp,
                coins=state.coins[0],
                tower_hp=tower_hp_0,
                tower_count=len(towers_0),
                tower_levels=[t.level for t in towers_0],
                gen_level=state.bases[0].generation_level,
                ant_level=state.bases[0].ant_level,
                die_count=state.die_count[0],
                build_cost=state.build_tower_cost(state.tower_count(0)),
            ),
            opponent_state=PlayerState(
                hp=state.bases[1].hp,
                coins=state.coins[1],
                tower_hp=tower_hp_1,
                tower_count=len(towers_1),
                tower_levels=[t.level for t in towers_1],
                gen_level=state.bases[1].generation_level,
                ant_level=state.bases[1].ant_level,
                die_count=state.die_count[1],
                build_cost=state.build_tower_cost(state.tower_count(1)),
            ),
            round_index=getattr(state, "round_index", 0),
        )

    def _compute_battle_rewards(
        self,
        old: StateSnapshot,
        new: StateSnapshot,
        terminated: bool,
        self_player: int,
        winner: Optional[int] = None,
        coins_self: Optional[float] = None,
        coins_opponent: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """计算完整回合的 self 视角奖励，返回 (reward_self, components)

        Args:
            old: 回合前状态快照（self 视角）
            new: 回合后状态快照（self 视角）
            terminated: 是否终止
            self_player: self 对应的 SDK player ID (0 或 1)
            winner: SDK 的 winner player ID (0 或 1)
            coins_self: 操作执行后、回合推进前的 self 金币
            coins_opponent: 操作执行后、回合推进前的 opponent 金币

        Returns:
            (reward_self, components)
        """
        # 伤害计算（self 视角：hp_dmg 是对 opponent 造成的伤害）
        hp_dmg = max(0.0, old.opponent_state.hp - new.opponent_state.hp)
        hp_dmg_received = max(0.0, old.self_state.hp - new.self_state.hp)

        tower_dmg = sum(
            max(
                0.0,
                old.opponent_state.tower_hp.get(k, 0)
                - new.opponent_state.tower_hp.get(k, 0),
            )
            for k in old.opponent_state.tower_hp
        )
        tower_dmg_received = sum(
            max(
                0.0,
                old.self_state.tower_hp.get(k, 0) - new.self_state.tower_hp.get(k, 0),
            )
            for k in old.self_state.tower_hp
        )

        # 金币计算（仅计算被动收入）
        if coins_self is not None and coins_opponent is not None:
            coin_gain_self = new.self_state.coins - coins_self
            coin_gain_opponent = new.opponent_state.coins - coins_opponent
        else:
            coin_gain_self = new.self_state.coins - old.self_state.coins
            coin_gain_opponent = new.opponent_state.coins - old.opponent_state.coins

        r = (
            hp_dmg * REWARD_CONFIG["base_hp_attack_weight"]
            + tower_dmg * REWARD_CONFIG["tower_hp_attack_weight"]
            + coin_gain_self * REWARD_CONFIG["own_coin_gain_weight"]
            - coin_gain_opponent * REWARD_CONFIG["enemy_coin_gain_weight"]
        )

        # ── 塔存活奖励 ──
        multipliers = REWARD_CONFIG["tower_survival_level_multipliers"]
        tower_survival_per_tower = REWARD_CONFIG["tower_survival_per_tower"]
        enemy_tower_survival_per_tower = REWARD_CONFIG["enemy_tower_survival_per_tower"]

        own_survival = sum(
            tower_survival_per_tower * multipliers[min(lv, len(multipliers) - 1)]
            for lv in new.self_state.tower_levels
        )
        enemy_survival = new.opponent_state.tower_count * enemy_tower_survival_per_tower
        tower_survival = own_survival - enemy_survival
        r += tower_survival

        # ── 科技加成 ──
        tech = (
            new.self_state.gen_level * REWARD_CONFIG["tech_speed_per_level"]
            + new.self_state.ant_level * REWARD_CONFIG["tech_hp_per_level"]
        )
        r += tech

        # ── 余额奖励 ──
        balance = new.self_state.coins * REWARD_CONFIG["balance_reward_weight"]
        r += balance

        # ── 死亡惩罚 ──
        self_die_delta = new.self_state.die_count - old.self_state.die_count
        opponent_die_delta = new.opponent_state.die_count - old.opponent_state.die_count
        penalty_per_ant = REWARD_CONFIG["own_die_penalty_per_ant"]
        die_penalty = (
            self_die_delta * penalty_per_ant - opponent_die_delta * penalty_per_ant
        )
        r += die_penalty

        # ── 终局奖励 ──
        end_reward = 0.0
        if terminated and winner is not None:
            if winner == self_player:
                end_reward = REWARD_CONFIG["win_reward"]
            else:
                end_reward = REWARD_CONFIG["loss_reward"]
        r += end_reward

        # clip
        clip_val = REWARD_CONFIG["step_reward_clip"]
        r = max(-clip_val, min(clip_val, r))

        # components（无 _self/_opponent 后缀，均为 self 视角）
        components = {
            "hp_dmg_reward": round(hp_dmg * REWARD_CONFIG["base_hp_attack_weight"], 4),
            "tower_dmg_reward": round(
                tower_dmg * REWARD_CONFIG["tower_hp_attack_weight"], 4
            ),
            "coin_gain_reward": round(
                coin_gain_self * REWARD_CONFIG["own_coin_gain_weight"], 4
            ),
            "coin_penalty_reward": round(
                -coin_gain_opponent * REWARD_CONFIG["enemy_coin_gain_weight"], 4
            ),
            "tower_survival": round(tower_survival, 4),
            "tech_bonus": round(tech, 4),
            "balance": round(balance, 4),
            "die_penalty": round(die_penalty, 4),
            "end_reward": round(end_reward, 4),
            "hp_dmg_raw": (round(hp_dmg, 2), round(hp_dmg_received, 2)),
            "tower_dmg_raw": (round(tower_dmg, 2), round(tower_dmg_received, 2)),
            "coin_gain_raw": (round(coin_gain_self, 2), round(coin_gain_opponent, 2)),
            "self_die_delta": self_die_delta,
            "opponent_die_delta": opponent_die_delta,
        }

        return r, components

    def _resolve_turn_from_ops(
        self, self_ops, opponent_ops, self_first
    ) -> Tuple[float, float, bool, bool, Dict]:
        """回合执行核心方法 - 计算完整回合奖励

        Args:
            self_ops: 己方操作列表
            opponent_ops: 对方操作列表
            self_first: 是否将己方映射为 player_0

        Returns:
            (reward_self, reward_opponent, terminated, truncated, info)
        """
        state = self._runtime.state
        self_player = 0 if self_first else 1

        # 获取 self 视角的回合前快照
        old = self.get_state_snapshot(self_player)

        # 在 SDK 调用边界做 player 映射
        runtime = self._runtime
        if self_first:
            # self → player 0, opponent → player 1
            if self_ops:
                runtime.apply_self_operations(self_ops)
            if opponent_ops:
                runtime.apply_opponent_operations(opponent_ops)
        else:
            # opponent → player 0, self → player 1
            if opponent_ops:
                runtime.apply_self_operations(opponent_ops)
            if self_ops:
                runtime.apply_opponent_operations(self_ops)

        # 捕获操作后、回合前的中间状态（塔 HP + 金币）
        mid = self.get_state_snapshot(self_player)
        coins_self = mid.self_state.coins
        coins_opponent = mid.opponent_state.coins

        # 推进回合
        state.advance_round()
        new_state = state
        new = self.get_state_snapshot(self_player)

        # 计算纯蚂蚁+武器造成的塔伤害（mid → new，不含操作效果）
        self_tower_dmg = sum(
            max(0.0, mid.self_state.tower_hp.get(k, 0) - new.self_state.tower_hp.get(k, 0))
            for k in mid.self_state.tower_hp
        )
        opponent_tower_dmg = sum(
            max(0.0, mid.opponent_state.tower_hp.get(k, 0) - new.opponent_state.tower_hp.get(k, 0))
            for k in mid.opponent_state.tower_hp
        )

        terminated = getattr(new_state, "terminal", False) or (
            getattr(new_state, "winner", None) is not None
        )
        winner = getattr(new_state, "winner", None)

        reward_self, components = self._compute_battle_rewards(
            old,
            new,
            terminated,
            self_player=self_player,
            winner=winner,
            coins_self=coins_self,
            coins_opponent=coins_opponent,
        )

        # reward_opponent 仅用于日志（零和假设）
        reward_opponent = -reward_self

        self._last_reward_detail = {
            "reward_self": round(reward_self, 4),
            "reward_opponent": round(reward_opponent, 4),
            **components,
            "action_reward": 0.0,
            "self_expense": round(old.self_state.coins - coins_self, 1),
            "opponent_expense": round(old.opponent_state.coins - coins_opponent, 1),
            "round_index": getattr(new_state, "round_index", 0),
            "self_hp_before": old.self_state.hp,
            "self_hp_after": new.self_state.hp,
            "opponent_hp_before": old.opponent_state.hp,
            "opponent_hp_after": new.opponent_state.hp,
            "self_tower_hp_before": sum(old.self_state.tower_hp.values()),
            "self_tower_hp_after": sum(new.self_state.tower_hp.values()),
            "opponent_tower_hp_before": sum(old.opponent_state.tower_hp.values()),
            "opponent_tower_hp_after": sum(new.opponent_state.tower_hp.values()),
            "self_coins_before": old.self_state.coins,
            "self_coins_after": new.self_state.coins,
            "opponent_coins_before": old.opponent_state.coins,
            "opponent_coins_after": new.opponent_state.coins,
            "ops_self_count": len(self_ops) if self_ops else 0,
            "ops_opponent_count": len(opponent_ops) if opponent_ops else 0,
            "terminated": terminated,
            "self_tower_count": new.self_state.tower_count,
            "opponent_tower_count": new.opponent_state.tower_count,
            "self_gen_level": new.self_state.gen_level,
            "self_ant_level": new.self_state.ant_level,
            "opponent_gen_level": new.opponent_state.gen_level,
            "opponent_ant_level": new.opponent_state.ant_level,
            "self_build_cost": new.self_state.build_cost,
            "opponent_build_cost": new.opponent_state.build_cost,
            "self_tower_dmg": round(self_tower_dmg, 2),
            "opponent_tower_dmg": round(opponent_tower_dmg, 2),
        }

        round_idx = getattr(new_state, "round_index", 0)
        truncated = round_idx >= MAX_ROUND and not terminated
        return (
            reward_self,
            reward_opponent,
            terminated,
            truncated,
            {
                "round_index": round_idx,
                "reward_detail": self._last_reward_detail,
                "winner": winner,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

    # ── 动作奖励计算 ──────────────────────────────────────────────────────────

    def _compute_action_reward(self, action_id: int, player: int) -> float:
        if action_id == 0:
            return REWARD_CONFIG["noop_base_penalty"]


        type_name, target_idx = self._resolve_action_type(action_id)
        if type_name is None:
            return 0.0

        handler_name = self._ACTION_REWARD_DISPATCH.get(type_name)
        if handler_name is None:
            return 0.0

        return getattr(self, handler_name)(action_id, player, type_name, target_idx)

    @staticmethod
    def _resolve_action_type(action_id: int):
        for type_cfg in TYPE_CONFIG:
            if type_cfg["flat_start"] <= action_id < type_cfg["flat_end"]:
                return type_cfg["name"], action_id - type_cfg["flat_start"]
        return None, None

    @staticmethod
    def resolve_action_type(action_id: int):
        """公共接口：将 action_id 解析为 (类型名称, 目标索引)。委托给 _resolve_action_type。"""
        return AntWarEnv._resolve_action_type(action_id)

    def compute_action_reward(self, action_id: int, player: int) -> float:
        """公共接口：计算动作奖励。委托给 _compute_action_reward。"""
        return self._compute_action_reward(action_id, player)

    def _compute_fixed_deploy_reward(self, _action_id, _player, type_name, _target_idx):
        return REWARD_CONFIG[f"deploy_{type_name}"]

    def _compute_build_tower_reward(self, _action_id, player, _type_name, target_idx):
        if target_idx >= len(TOWER_POSITIONS):
            return 0.0
        target_pos = TOWER_POSITIONS[target_idx]
        pos_key = target_pos[0] * _POSITION_HASH_MULTIPLIER + target_pos[1]
        player_counts = self._tower_build_slot_counts[player]
        count = player_counts.get(pos_key, 0)
        tiers = REWARD_CONFIG["build_tower_tiers"]
        tier_idx = min(count, len(tiers) - 1)
        player_counts[pos_key] = count + 1
        return tiers[tier_idx]

    def _compute_upgrade_tower_reward(
        self, _action_id, _player, _type_name, target_idx
    ):
        if target_idx >= len(TOWER_POSITIONS) * _UPGRADE_DIRECTIONS or self._runtime is None:
            return 0.0
        # 64 个插槽 = 16 个位置 × 4 个升级方向
        pos_idx = target_idx // _UPGRADE_DIRECTIONS
        if pos_idx >= len(TOWER_POSITIONS):
            return 0.0
        target_pos = TOWER_POSITIONS[pos_idx]
        tower = self._runtime.state.tower_at(target_pos[0], target_pos[1])
        if tower is not None:
            # 计算在 _resolve_turn_from_ops 之前执行，检查升级前等级
            if tower.level == 1:
                return REWARD_CONFIG["upgrade_tower_l2"]
            elif tower.level == 2:
                return REWARD_CONFIG["upgrade_tower_l3"]
        return 0.0

    def _compute_downgrade_tower_reward(
        self, _action_id, player, _type_name, target_idx
    ):
        reward = REWARD_CONFIG["downgrade_tower_penalty"]
        if target_idx < len(TOWER_POSITIONS):
            target_pos = TOWER_POSITIONS[target_idx]
            pos_key = target_pos[0] * _POSITION_HASH_MULTIPLIER + target_pos[1]
            player_counts = self._tower_build_slot_counts[player]
            count = player_counts.get(pos_key, 0)
            if count > 0:
                player_counts[pos_key] = count - 1
        return reward

    def _compute_tech_upgrade_reward(self, _action_id, player, _type_name, target_idx):
        if self._runtime is None:
            if target_idx == 0:
                return REWARD_CONFIG["upgrade_gen_speed_l1"]
            elif target_idx == 1:
                return REWARD_CONFIG["upgrade_gen_ant_l1"]
            return 0.0
        player_state = self._runtime.state.bases[player]
        if target_idx == 0:
            level = player_state.generation_level
            if level == 0:
                return REWARD_CONFIG["upgrade_gen_speed_l1"]
            elif level == 1:
                return REWARD_CONFIG["upgrade_gen_speed_l1"]
            elif level == 2:
                return REWARD_CONFIG["upgrade_gen_speed_l2"]
            return 0.0
        elif target_idx == 1:
            level = player_state.ant_level
            if level == 0:
                return REWARD_CONFIG["upgrade_gen_ant_l1"]
            elif level == 1:
                return REWARD_CONFIG["upgrade_gen_ant_l1"]
            elif level == 2:
                return REWARD_CONFIG["upgrade_gen_ant_l2"]
            return 0.0
        return 0.0

    # ── 清理 ──────────────────────────────────────────────────────────────────

    def close(self):
        """清理环境资源"""
        self._runtime = None
        self._tower_build_slot_counts.clear()
