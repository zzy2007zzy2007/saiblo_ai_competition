from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from SDK.backend.state import BackendState

from .observation import ObservationEncoder
from .action_mask import ActionMaskHandler
from ..utils.action_constants import REWARD_CONFIG, OperationType, TOWER_POSITIONS


class AntWarEnv:
    def __init__(self) -> None:
        self.observation_encoder = ObservationEncoder()
        self.action_mask_handler = ActionMaskHandler()
        self._runtime = None
        self._pending_player_0_action = None
        self._is_sequential_pending = False
        self._tower_build_slot_counts: dict[int, dict[int, int]] = {
            0: {i: 0 for i in range(8)},
            1: {i: 0 for i in range(8)},
        }
        self._player_noop_streak: dict[int, int] = {0: 0, 1: 0}
        self._last_reward_detail: dict = {}

    def reset(self) -> Tuple[Dict, Dict]:
        from ..compat.adapters import BackendAdapter

        self._runtime = BackendAdapter.create_runtime(player=0)
        self._pending_player_0_action = None
        self._is_sequential_pending = False
        self._tower_build_slot_counts = {
            0: {i: 0 for i in range(8)},
            1: {i: 0 for i in range(8)},
        }
        self._player_noop_streak = {0: 0, 1: 0}
        self._last_reward_detail = {}

        state = self._runtime.state
        obs_0 = self._encode_observation(state, 0)
        obs_1 = self._encode_observation(state, 1)

        return {"player_0": obs_0, "player_1": obs_1}, {"round_index": state.round_index}

    def step(
        self, action: Union[int, Dict[str, int]]
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        if isinstance(action, dict):
            return self._step_batch(action)
        else:
            return self._step_sequential(action)

    def _step_sequential(self, action: int) -> Tuple[Dict, Dict, bool, bool, Dict]:
        if not self._is_sequential_pending:
            self._pending_player_0_action = action
            self._is_sequential_pending = True
            state = self._runtime.state
            obs_0 = self._encode_observation(state, 0)
            obs_1 = self._encode_observation(state, 1)
            return (
                {"player_0": obs_0, "player_1": obs_1},
                {"player_0": 0.0, "player_1": 0.0},
                False,
                False,
                {"round_index": state.round_index},
            )
        else:
            self._is_sequential_pending = False
            return self._resolve_turn(self._pending_player_0_action, action)

    def _step_batch(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        return self._resolve_turn(actions["player_0"], actions["player_1"])

    def _resolve_turn(
        self, player_0_action_id: int, player_1_action_id: int
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        state = self._runtime.state
        player_0_op = self.action_mask_handler.action_id_to_operation(player_0_action_id, state, 0)
        player_1_op = self.action_mask_handler.action_id_to_operation(player_1_action_id, state, 1)
        return self._resolve_turn_from_ops(player_0_op, player_1_op)

    def _action_or_ops_to_op(self, action_id: Optional[int], ops: Optional[list], player_idx: int):
        if ops is not None and len(ops) > 0:
            return ops[0]
        if action_id is not None:
            return self.action_mask_handler.action_id_to_operation(action_id, self._runtime.state, player_idx)
        return None

    def _resolve_turn_from_ids_or_ops(
        self,
        player_0_action_id: Optional[int] = None,
        player_0_ops: Optional[list] = None,
        player_1_action_id: Optional[int] = None,
        player_1_ops: Optional[list] = None,
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        player_0_op = self._action_or_ops_to_op(player_0_action_id, player_0_ops, 0)
        player_1_op = self._action_or_ops_to_op(player_1_action_id, player_1_ops, 1)
        return self._resolve_turn_from_ops(player_0_op, player_1_op)

    def _resolve_turn_from_ops(
        self, player_0_op, player_1_op
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        state = self._runtime.state

        hp_before = tuple(state.bases[p].hp for p in (0, 1))
        die_before = tuple(state.die_count)

        action_reward_0 = self._compute_action_reward(0, player_0_op)
        action_reward_1 = self._compute_action_reward(1, player_1_op)

        ops_0 = [player_0_op] if player_0_op is not None else []
        ops_1 = [player_1_op] if player_1_op is not None else []

        coins_before_ops = tuple(state.coins)

        if ops_0:
            self._runtime.apply_self_operations(ops_0)
        if ops_1:
            self._runtime.apply_opponent_operations(ops_1)

        coins_after_ops = tuple(state.coins)

        tower_hp_before = {
            0: {t.tower_id: t.hp for t in state.towers_of(0)},
            1: {t.tower_id: t.hp for t in state.towers_of(1)},
        }

        state.advance_round()

        state = self._runtime.state
        terminated = state.terminal

        obs_0 = self._encode_observation(state, 0)
        obs_1 = self._encode_observation(state, 1)

        cfg = REWARD_CONFIG
        rewards = {"player_0": action_reward_0, "player_1": action_reward_1}

        for p in (0, 1):
            enemy = 1 - p
            agent = f"player_{p}"

            round_reward = 0.0
            enemy_base_damage = hp_before[enemy] - state.bases[enemy].hp
            enemy_towers_after = {t.tower_id: t.hp for t in state.towers_of(enemy)}
            enemy_tower_damage = 0
            for tid, hp_b4 in tower_hp_before[enemy].items():
                hp_after = enemy_towers_after.get(tid, 0)
                enemy_tower_damage += max(0, hp_b4 - hp_after)
            hp_attack_base_val = enemy_base_damage * cfg["base_hp_attack_weight"]
            hp_attack_tower_val = enemy_tower_damage * cfg["tower_hp_attack_weight"]
            round_reward += hp_attack_base_val + hp_attack_tower_val

            own_income = state.coins[p] - coins_after_ops[p]
            enemy_income = state.coins[enemy] - coins_after_ops[enemy]
            coin_gain_val = own_income * cfg["own_coin_gain_weight"] - enemy_income * cfg["enemy_coin_gain_weight"]
            round_reward += own_income * cfg["own_coin_gain_weight"]
            round_reward -= enemy_income * cfg["enemy_coin_gain_weight"]

            balance_reward_val = state.coins[p] * cfg["balance_reward_weight"]
            round_reward += balance_reward_val

            multipliers = cfg["tower_survival_level_multipliers"]
            own_tower_value = 0
            for t in state.towers_of(p):
                level = min(t.level, len(multipliers) - 1)
                own_tower_value += cfg["tower_survival_per_tower"] * multipliers[level]
            enemy_tower_value = len(state.towers_of(enemy)) * cfg["enemy_tower_survival_per_tower"]
            tower_survival_val = own_tower_value - enemy_tower_value
            round_reward += own_tower_value
            round_reward -= enemy_tower_value

            tech_bonus_val = state.bases[p].generation_level * cfg["tech_speed_per_level"] \
                             + state.bases[p].ant_level * cfg["tech_hp_per_level"]
            round_reward += tech_bonus_val

            own_die_delta = state.die_count[p] - die_before[p]
            enemy_die_delta = state.die_count[enemy] - die_before[enemy]
            die_penalty_val = own_die_delta * cfg["own_die_penalty_per_ant"] - enemy_die_delta * cfg["own_die_penalty_per_ant"]
            round_reward += own_die_delta * cfg["own_die_penalty_per_ant"]
            round_reward -= enemy_die_delta * cfg["own_die_penalty_per_ant"]

            end_reward_val = 0.0
            if terminated:
                if state.winner == p:
                    end_reward_val = cfg["win_reward"]
                    round_reward += cfg["win_reward"]
                elif state.winner == enemy:
                    end_reward_val = cfg["loss_reward"]
                    round_reward += cfg["loss_reward"]

            rewards[agent] += round_reward

            action_reward_val = action_reward_0 if p == 0 else action_reward_1
            own_expense = coins_before_ops[p] - coins_after_ops[p]
            self._last_reward_detail[agent] = {
                'action_reward': round(action_reward_val, 4),
                'hp_attack_base': round(hp_attack_base_val, 4),
                'hp_attack_tower': round(hp_attack_tower_val, 4),
                'coin_gain': round(coin_gain_val, 4),
                'tower_survival': round(tower_survival_val, 4),
                'balance': round(balance_reward_val, 4),
                'tech_bonus': round(tech_bonus_val, 4),
                'die_penalty': round(die_penalty_val, 4),
                'end_reward': round(end_reward_val, 4),
                'own_income': round(own_income, 1),
                'enemy_income': round(enemy_income, 1),
                'coin_balance': round(state.coins[p], 1),
                'own_expense': round(own_expense, 1),
                'enemy_coin_balance': round(state.coins[enemy], 1),
                'tower_count': len(state.towers_of(p)),
                'enemy_tower_count': len(state.towers_of(enemy)),
                'build_tower_cost': state.build_tower_cost(state.tower_count(p)),
                'generation_level': state.bases[p].generation_level,
                'ant_level': state.bases[p].ant_level,
                'game_round': state.round_index,
            }

        clip = cfg["step_reward_clip"]
        for agent in ("player_0", "player_1"):
            rewards[agent] = max(min(rewards[agent], clip), -clip)

        return {"player_0": obs_0, "player_1": obs_1}, rewards, terminated, False, {"round_index": state.round_index}

    def _compute_action_reward(self, player: int, op) -> float:
        cfg = REWARD_CONFIG
        reward = 0.0

        if op is None or op.op_type == OperationType.NO_OP:
            self._player_noop_streak[player] += 1
            n = self._player_noop_streak[player]
            if n > cfg["noop_tolerance"]:
                penalty = cfg["noop_base_penalty"] + cfg["noop_base_penalty"] * ((n - cfg["noop_tolerance"]) // 5)
                reward = max(penalty, cfg["noop_max_penalty"])
            return reward

        self._player_noop_streak[player] = 0

        op_type = op.op_type

        if op_type == OperationType.BUILD_TOWER:
            x, y = op.arg0, op.arg1
            try:
                global_idx = TOWER_POSITIONS.index((x, y))
            except ValueError:
                global_idx = -1
            if global_idx >= 0:
                local_idx = global_idx if player == 0 else global_idx - 8
                if 0 <= local_idx <= 7:
                    count = self._tower_build_slot_counts[player][local_idx]
                    tiers = cfg["build_tower_tiers"]
                    if count < len(tiers):
                        reward += tiers[count]
                    else:
                        reward += tiers[-1]
                    self._tower_build_slot_counts[player][local_idx] += 1

        elif op_type == OperationType.UPGRADE_TOWER:
            target_type = op.arg1
            if target_type >= 100:
                reward += cfg["upgrade_tower_l3"]
            else:
                reward += cfg["upgrade_tower_l2"]

        elif op_type == OperationType.DOWNGRADE_TOWER:
            reward += cfg["downgrade_tower_penalty"]
            tower_id = op.arg0
            state = self._runtime.state
            tower = state.tower_by_id(tower_id) if hasattr(state, 'tower_by_id') else None
            if tower is not None:
                try:
                    global_idx = TOWER_POSITIONS.index((tower.x, tower.y))
                except ValueError:
                    global_idx = -1
                if global_idx >= 0:
                    local_idx = global_idx if player == 0 else global_idx - 8
                    if 0 <= local_idx <= 7:
                        old = self._tower_build_slot_counts[player][local_idx]
                        if old > 0:
                            self._tower_build_slot_counts[player][local_idx] = old - 1

        elif op_type == OperationType.UPGRADE_GENERATION_SPEED:
            level = self._runtime.state.bases[player].generation_level - 1
            if 0 <= level < len(cfg["upgrade_gen_speed"]):
                reward += cfg["upgrade_gen_speed"][level]

        elif op_type == OperationType.UPGRADE_GENERATED_ANT:
            level = self._runtime.state.bases[player].ant_level - 1
            if 0 <= level < len(cfg["upgrade_gen_ant"]):
                reward += cfg["upgrade_gen_ant"][level]

        elif op_type == OperationType.USE_LIGHTNING_STORM:
            reward += cfg["deploy_ls"]
        elif op_type == OperationType.USE_EMP_BLASTER:
            reward += cfg["deploy_emp"]
        elif op_type == OperationType.USE_DEFLECTOR:
            reward += cfg["deploy_deflector"]
        elif op_type == OperationType.USE_EMERGENCY_EVASION:
            reward += cfg["deploy_evasion"]

        return reward

    def _encode_observation(self, state: BackendState, player: int) -> Dict[str, np.ndarray]:
        encoded = self.observation_encoder.encode(state, player)
        action_mask = self.action_mask_handler.get_action_mask(state, player)
        return {
            "board": encoded["board"],
            "global": encoded["global"],
            "action_mask": action_mask,
        }

    def close(self) -> None:
        self._runtime = None
        self._pending_player_0_action = None
        self._is_sequential_pending = False


def make_antwar_env(**kwargs):
    def _factory() -> AntWarEnv:
        return AntWarEnv()

    return _factory
