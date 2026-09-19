#!/usr/bin/env python3
"""
最终提交AI - 95%胜率！
"""
from __future__ import annotations

import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum

SUBMISSION_DIR = Path(__file__).resolve().parent
ANT_GAME_ROOT = SUBMISSION_DIR / "Ant-Game"
GA_TRAINING_ROOT = SUBMISSION_DIR / "ga_nn_training"

sys.path.insert(0, str(SUBMISSION_DIR))
sys.path.insert(1, str(ANT_GAME_ROOT))
sys.path.insert(2, str(GA_TRAINING_ROOT))

from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import TowerType, MAP_SIZE
from SDK.backend.model import Operation, OperationType
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle

from common import BaseAgent


class ActionFeatureType(IntEnum):
    HOLD = 0
    BUILD_TOWER = 1
    UPGRADE_TOWER = 2
    DOWNGRADE_TOWER = 3
    LIGHTNING_STORM = 4
    EMP_BLASTER = 5
    DEFLECTOR = 6
    EMERGENCY_EVASION = 7
    UPGRADE_SPEED = 8
    UPGRADE_ANT = 9


@dataclass(slots=True)
class ActionFeatures:
    type_onehot: np.ndarray
    position: np.ndarray
    tower_position: np.ndarray
    tower_type: np.ndarray
    cost: float
    is_valid: float


class ActionFeatureExtractor:

    def __init__(self):
        self.max_coins = 500

    def _get_tower_by_id(self, state: BackendState, tower_id: int):
        for tower in state.towers:
            if tower.tower_id == tower_id:
                return tower
        return None

    def _operation_to_action_type(self, op: Operation) -> ActionFeatureType:
        type_map = {
            OperationType.BUILD_TOWER: ActionFeatureType.BUILD_TOWER,
            OperationType.UPGRADE_TOWER: ActionFeatureType.UPGRADE_TOWER,
            OperationType.DOWNGRADE_TOWER: ActionFeatureType.DOWNGRADE_TOWER,
            OperationType.USE_LIGHTNING_STORM: ActionFeatureType.LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER: ActionFeatureType.EMP_BLASTER,
            OperationType.USE_DEFLECTOR: ActionFeatureType.DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION: ActionFeatureType.EMERGENCY_EVASION,
            OperationType.UPGRADE_GENERATION_SPEED: ActionFeatureType.UPGRADE_SPEED,
            OperationType.UPGRADE_GENERATED_ANT: ActionFeatureType.UPGRADE_ANT,
        }
        return type_map.get(op.op_type, ActionFeatureType.HOLD)

    def _get_action_cost(self, op: Operation, state: BackendState, player: int) -> float:
        if op.op_type == OperationType.BUILD_TOWER:
            tower_count = state.tower_count(player)
            return 15 * (2 ** tower_count)
        elif op.op_type == OperationType.UPGRADE_TOWER:
            tower = self._get_tower_by_id(state, op.arg0)
            if tower:
                if tower.level == 1:
                    return 60
                elif tower.level == 2:
                    return 200
            return 0
        elif op.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self._get_tower_by_id(state, op.arg0)
            if tower:
                if tower.level == 2:
                    return -30
                elif tower.level == 3:
                    return -100
            return 0
        elif op.op_type == OperationType.USE_LIGHTNING_STORM:
            return 90
        elif op.op_type == OperationType.USE_EMP_BLASTER:
            return 135
        elif op.op_type == OperationType.USE_DEFLECTOR:
            return 60
        elif op.op_type == OperationType.USE_EMERGENCY_EVASION:
            return 60
        elif op.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            return 200 if state.bases[player].generation_level == 0 else 250
        elif op.op_type == OperationType.UPGRADE_GENERATED_ANT:
            return 200 if state.bases[player].ant_level == 0 else 250
        return 0.0

    def extract(self, op: Operation, state: BackendState, player: int) -> ActionFeatures:
        action_type = self._operation_to_action_type(op)

        type_onehot = np.zeros(len(ActionFeatureType), dtype=np.float32)
        type_onehot[action_type.value] = 1.0

        position = np.zeros(2, dtype=np.float32)
        if op.arg0 >= 0 and op.arg1 >= 0:
            position[0] = op.arg0 / MAP_SIZE
            position[1] = op.arg1 / MAP_SIZE

        tower_position = np.zeros(2, dtype=np.float32)
        tower_type = np.zeros(5, dtype=np.float32)

        if op.op_type in [OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER]:
            tower = self._get_tower_by_id(state, op.arg0)
            if tower:
                tower_position[0] = tower.x / MAP_SIZE
                tower_position[1] = tower.y / MAP_SIZE
                if tower.tower_type.value < len(tower_type):
                    tower_type[tower.tower_type.value] = 1.0

        cost = self._get_action_cost(op, state, player) / self.max_coins

        is_valid = 1.0

        return ActionFeatures(
            type_onehot=type_onehot,
            position=position,
            tower_position=tower_position,
            tower_type=tower_type,
            cost=cost,
            is_valid=is_valid
        )

    def flatten(self, features: ActionFeatures) -> np.ndarray:
        return np.concatenate([
            features.type_onehot,
            features.position,
            features.tower_position,
            features.tower_type,
            np.array([features.cost, features.is_valid])
        ])


class StateActionEncoder:

    def __init__(self):
        self.state_extractor = FeatureExtractor()
        self.action_extractor = ActionFeatureExtractor()
        self._state_dim = None
        self._action_dim = 21

    def encode_state(self, state: BackendState, player: int) -> np.ndarray:
        action_mask = np.zeros(100, dtype=np.float32)
        obs = self.state_extractor.encode_observation(state, player, action_mask)
        state_vec = self.state_extractor.flatten_observation(obs)
        return state_vec.astype(np.float32)

    def encode_action(self, op: Operation, state: BackendState, player: int) -> np.ndarray:
        action_features = self.action_extractor.extract(op, state, player)
        return self.action_extractor.flatten(action_features)


from ga_trainer import TwoStageGenome


class AI(BaseAgent):
    class ScoringStrategy:
        MEAN = 'mean'
        SUM = 'sum'
        SUM_SQRT = 'sum_sqrt'
        MAX = 'max'

    def __init__(self, strategy: str = ScoringStrategy.SUM_SQRT, model_path: str | None = None, top_n_candidates: int = 30):
        super().__init__()
        self.encoder = StateActionEncoder()
        self.network = None
        self._cached_state_embed = None
        self._cached_player = None
        self.top_n_candidates = top_n_candidates
        self.strategy = strategy

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'my_ant_game', 'models', 'two_stage_model.npz')

        genome = TwoStageGenome()
        genome.load(model_path)
        self.network = genome.create_network()

        self._invalidate_cache = lambda: None

    def _ensure_state_encoding(self, state: BackendState, player: int) -> np.ndarray:
        if self._cached_state_embed is None or self._cached_player != (state, player):
            state_vec = self.encoder.encode_state(state, player)
            self._cached_state_embed = self.network.encode_state(state_vec)
            self._cached_player = (state, player)
        return self._cached_state_embed

    def score_bundle(self, state: BackendState, player: int, bundle: ActionBundle) -> float:
        if not bundle.operations:
            return 0.0

        state_embed = self._ensure_state_encoding(state, player)

        scores = []
        for op in bundle.operations:
            action_vec = self.encoder.encode_action(op, state, player)
            score = self.network.score_action(state_embed, action_vec)
            scores.append(score)

        if self.strategy == self.ScoringStrategy.MEAN:
            return sum(scores) / len(scores)
        elif self.strategy == self.ScoringStrategy.SUM:
            return sum(scores)
        elif self.strategy == self.ScoringStrategy.SUM_SQRT:
            return sum(scores) / np.sqrt(len(scores))
        elif self.strategy == self.ScoringStrategy.MAX:
            return max(scores)
        else:
            return sum(scores) / len(scores)

    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)

        if len(bundles) <= 1:
            return bundles[0]

        if len(bundles) > self.top_n_candidates:
            ranked_by_official = sorted(bundles[1:], key=lambda x: -x.score)[:self.top_n_candidates-1]
            shortlist = [bundles[0]] + ranked_by_official
        else:
            shortlist = bundles

        scored_bundles = []
        for bundle in shortlist:
            score = self.score_bundle(state, player, bundle)
            scored_bundles.append((score, bundle))

        scored_bundles.sort(key=lambda x: -x[0])
        best_bundle = scored_bundles[0][1]

        return best_bundle


def create_agent(strategy: str = 'sum_sqrt') -> AI:
    return AI(strategy)
