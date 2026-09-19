from __future__ import annotations

import numpy as np
from typing import Dict

from SDK.backend.state import BackendState
from SDK.utils.features import (
    FeatureExtractor,
    MAX_ACTIONS,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    PLAYER_BASES,
)
from SDK.utils.turns import DecisionContext
from ppo_antwar.utils.action_constants import OBS_NORMALIZATION


class ObservationEncoder:
    def __init__(self) -> None:
        self._extractor = FeatureExtractor(max_actions=MAX_ACTIONS)

    def encode(self, state: BackendState, player: int) -> Dict[str, np.ndarray]:
        board = self._extractor.encode_board(state, player)
        global_features = self._encode_global(state, player)
        return {
            "board": board,
            "global": global_features,
        }

    def _encode_global(self, state: BackendState, player: int) -> np.ndarray:
        enemy = 1 - player
        summary = self._extractor.summarize(state, player).named
        s = OBS_NORMALIZATION

        features = np.array(
            [
                summary["round_ratio"],
                summary["hp_delta"] / s["hp_scale"],
                summary["coin_ratio"],
                summary["safe_coin"] / s["coin_scale"],
                summary["frontline_advantage"] / s["distance_scale"],
                summary["enemy_front_distance"] / s["distance_scale"],
                summary["my_front_distance"] / s["distance_scale"],
                summary["enemy_progress"] / s["progress_scale"],
                summary["my_progress"] / s["progress_scale"],
                summary["tower_count"] / s["tower_count_scale"],
                summary["enemy_tower_count"] / s["tower_count_scale"],
                summary["tower_level_sum"] / s["tower_level_scale"],
                summary["enemy_tower_level_sum"] / s["tower_level_scale"],
                summary["kill_delta"] / s["kill_scale"],
                summary["old_delta"] / s["kill_scale"],
                summary["tower_spread"] / s["tower_spread_scale"],
                summary["slot_fill_ratio"],
                summary["generation_level"],
                summary["ant_level"],
                summary["hostile_distance"] / s["distance_scale"],
                summary["base_arc_coverage"],
                summary["tower_spacing"],
            ],
            dtype=np.float32,
        )

        extras = np.array(
            [
                state.bases[player].hp / s["hp_scale"],
                state.bases[enemy].hp / s["hp_scale"],
                state.coins[player] / s["coin_scale"],
                state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
                / SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cooldown,
                state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cooldown,
                state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR]
                / SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cooldown,
                state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.LIGHTNING_STORM]
                / SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.EMP_BLASTER]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.DEFLECTOR]
                / SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.EMERGENCY_EVASION]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cooldown,
            ],
            dtype=np.float32,
        )

        return np.concatenate([features, extras], dtype=np.float32)
