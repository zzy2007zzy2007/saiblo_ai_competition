from typing import Dict, Optional

import numpy as np

from SDK.utils.features import (
    FeatureExtractor,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
)

from ..utils.action_constants import OBS_NORMALIZATION
from .action_mask import ActionMaskHandler


class ObservationEncoder:
    """观测编码器 - 将游戏状态编码为神经网络可处理的张量格式。

    输出字典：
        board:       numpy.ndarray, shape (28, 19, 19)  棋盘特征图
        global_vec:  numpy.ndarray, shape (33,)           全局特征向量
        action_mask: numpy.ndarray, shape (119,)          动作合法性掩码
    """

    GLOBAL_FEATURE_DIM = 33

    _REQUIRED_OBS_SCALES = [
        "hp_scale",
        "coin_scale",
        "distance_scale",
        "progress_scale",
        "tower_count_scale",
        "tower_level_scale",
        "kill_scale",
        "tower_spread_scale",
        "tower_spacing_scale",
    ]

    def __init__(self, action_mask_handler: Optional[ActionMaskHandler] = None):
        missing = [k for k in self._REQUIRED_OBS_SCALES if k not in OBS_NORMALIZATION]
        if missing:
            raise RuntimeError(
                f"Missing required observation normalization scales: {missing}"
            )
        self._extractor = FeatureExtractor()
        self._action_mask_handler = action_mask_handler or ActionMaskHandler()

    def encode(self, state, player: int) -> Dict[str, np.ndarray]:
        board = self._encode_board(state, player)
        global_vec = self._encode_global(state, player)
        action_mask = self._action_mask_handler.get_action_mask(state, player)

        return {
            "board": board,
            "global": global_vec,
            "action_mask": action_mask,
        }

    def _encode_board(self, state, player_id: int) -> np.ndarray:
        board = self._extractor.encode_board(state, player_id)
        return np.array(board, dtype=np.float32)

    def _encode_global(self, state, player_id: int) -> np.ndarray:
        eid = 1 - player_id
        summary = self._extractor.summarize(state, player_id)

        features = []

        n = summary.named
        features.append(n["round_ratio"])
        features.append(n["hp_delta"] / OBS_NORMALIZATION["hp_scale"])
        features.append(n["coin_ratio"])
        features.append(n["safe_coin"] / OBS_NORMALIZATION["coin_scale"])
        features.append(n["frontline_advantage"] / OBS_NORMALIZATION["distance_scale"])
        features.append(n["enemy_front_distance"] / OBS_NORMALIZATION["distance_scale"])
        features.append(n["my_front_distance"] / OBS_NORMALIZATION["distance_scale"])
        features.append(n["enemy_progress"] / OBS_NORMALIZATION["progress_scale"])
        features.append(n["my_progress"] / OBS_NORMALIZATION["progress_scale"])
        features.append(n["tower_count"] / OBS_NORMALIZATION["tower_count_scale"])
        features.append(n["enemy_tower_count"] / OBS_NORMALIZATION["tower_count_scale"])
        features.append(n["tower_level_sum"] / OBS_NORMALIZATION["tower_level_scale"])
        features.append(
            n["enemy_tower_level_sum"] / OBS_NORMALIZATION["tower_level_scale"]
        )
        features.append(n["kill_delta"] / OBS_NORMALIZATION["kill_scale"])
        features.append(n["old_delta"] / OBS_NORMALIZATION["kill_scale"])
        features.append(n["tower_spread"] / OBS_NORMALIZATION["tower_spread_scale"])
        features.append(n["slot_fill_ratio"])
        features.append(n["generation_level"])
        features.append(n["ant_level"])
        features.append(n["hostile_distance"] / OBS_NORMALIZATION["distance_scale"])
        features.append(n["base_arc_coverage"])
        features.append(n["tower_spacing"] / OBS_NORMALIZATION["tower_spacing_scale"])

        features.append(state.bases[player_id].hp / OBS_NORMALIZATION["hp_scale"])
        features.append(state.bases[eid].hp / OBS_NORMALIZATION["hp_scale"])
        features.append(state.coins[player_id] / OBS_NORMALIZATION["coin_scale"])

        for weapon in [
            SuperWeaponType.LIGHTNING_STORM,
            SuperWeaponType.EMP_BLASTER,
            SuperWeaponType.DEFLECTOR,
            SuperWeaponType.EMERGENCY_EVASION,
        ]:
            cd = SUPER_WEAPON_STATS[weapon].cooldown
            features.append(state.weapon_cooldowns[player_id, weapon] / max(cd, 1))

        for weapon in [
            SuperWeaponType.LIGHTNING_STORM,
            SuperWeaponType.EMP_BLASTER,
            SuperWeaponType.DEFLECTOR,
            SuperWeaponType.EMERGENCY_EVASION,
        ]:
            cd = SUPER_WEAPON_STATS[weapon].cooldown
            features.append(state.weapon_cooldowns[eid, weapon] / max(cd, 1))

        assert len(features) == self.GLOBAL_FEATURE_DIM, (
            f"Expected {self.GLOBAL_FEATURE_DIM} global features, got {len(features)}"
        )
        return np.array(features, dtype=np.float32)
