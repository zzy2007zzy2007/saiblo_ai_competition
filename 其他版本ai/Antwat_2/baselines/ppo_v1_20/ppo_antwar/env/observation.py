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

        features = np.array(
            [
                summary["round_ratio"],
                summary["hp_delta"] / 50.0,
                summary["coin_ratio"],
                summary["safe_coin"] / 1000.0,
                summary["frontline_advantage"] / 19.0,
                summary["enemy_front_distance"] / 19.0,
                summary["my_front_distance"] / 19.0,
                summary["enemy_progress"] / 64.0,
                summary["my_progress"] / 64.0,
                summary["tower_count"] / 20.0,
                summary["enemy_tower_count"] / 20.0,
                summary["tower_level_sum"] / 40.0,
                summary["enemy_tower_level_sum"] / 40.0,
                summary["kill_delta"] / 20.0,
                summary["old_delta"] / 20.0,
                summary["tower_spread"] / 10.0,
                summary["slot_fill_ratio"],
                summary["generation_level"],
                summary["ant_level"],
                summary["hostile_distance"] / 19.0,
                summary["base_arc_coverage"],
                summary["tower_spacing"],
            ],
            dtype=np.float32,
        )

        extras = np.array(
            [
                state.bases[player].hp / 50.0,
                state.bases[enemy].hp / 50.0,
                state.coins[player] / 1000.0,
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
