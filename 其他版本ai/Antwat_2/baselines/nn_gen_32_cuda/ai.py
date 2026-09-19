from __future__ import annotations

import sys
import os
import glob
import numpy as np
import torch
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import TowerType, MAP_SIZE
from SDK.backend.model import Operation, OperationType
from SDK.backend.state import BackendState
from SDK.backend.engine import GameState
from SDK.utils.actions import ActionBundle

from common import BaseAgent
from model import ActionScoringNetwork


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


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
            position[0] = op.arg0 / (MAP_SIZE - 1)
            position[1] = op.arg1 / (MAP_SIZE - 1)

        tower_position = np.zeros(2, dtype=np.float32)
        if op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER):
            tower = self._get_tower_by_id(state, op.arg0)
            if tower is not None:
                tower_position[0] = tower.x / (MAP_SIZE - 1)
                tower_position[1] = tower.y / (MAP_SIZE - 1)
        elif op.op_type == OperationType.BUILD_TOWER:
            tower_position = position.copy()

        tower_type = np.zeros(5, dtype=np.float32)
        if op.op_type == OperationType.BUILD_TOWER:
            if 0 <= op.arg1 < 5:
                tower_type[op.arg1] = 1.0
        elif op.op_type == OperationType.UPGRADE_TOWER:
            if 0 <= op.arg1 <= 4:
                tower_type[op.arg1] = 1.0
        elif op.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self._get_tower_by_id(state, op.arg0)
            if tower is not None:
                tower_type[min(tower.tower_type.value, 4)] = 1.0

        cost = self._get_action_cost(op, state, player) / self.max_coins
        is_valid = 1.0 if state.coins[player] >= abs(self._get_action_cost(op, state, player)) else 0.0

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
            [features.cost],
            [features.is_valid],
        ])

    def batch_extract(self, ops: list[Operation], state: BackendState, player: int) -> np.ndarray:
        if not ops:
            return np.zeros((0, 21), dtype=np.float32)
        num_types = len(ActionFeatureType)
        batch_size = len(ops)
        result = np.zeros((batch_size, 21), dtype=np.float32)

        for i, op in enumerate(ops):
            action_type = self._operation_to_action_type(op)
            result[i, action_type.value] = 1.0

            if op.arg0 >= 0 and op.arg1 >= 0:
                result[i, num_types] = op.arg0 / (MAP_SIZE - 1)
                result[i, num_types + 1] = op.arg1 / (MAP_SIZE - 1)

            if op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER):
                tower = self._get_tower_by_id(state, op.arg0)
                if tower is not None:
                    result[i, num_types + 2] = tower.x / (MAP_SIZE - 1)
                    result[i, num_types + 3] = tower.y / (MAP_SIZE - 1)
            elif op.op_type == OperationType.BUILD_TOWER:
                result[i, num_types + 2] = result[i, num_types]
                result[i, num_types + 3] = result[i, num_types + 1]

            tt_start = num_types + 4
            if op.op_type == OperationType.BUILD_TOWER:
                if 0 <= op.arg1 < 5:
                    result[i, tt_start + op.arg1] = 1.0
            elif op.op_type == OperationType.UPGRADE_TOWER:
                if 0 <= op.arg1 <= 4:
                    result[i, tt_start + op.arg1] = 1.0
            elif op.op_type == OperationType.DOWNGRADE_TOWER:
                tower = self._get_tower_by_id(state, op.arg0)
                if tower is not None:
                    result[i, tt_start + min(tower.tower_type.value, 4)] = 1.0

            cost = self._get_action_cost(op, state, player)
            result[i, tt_start + 5] = cost / self.max_coins
            result[i, tt_start + 6] = 1.0 if state.coins[player] >= abs(cost) else 0.0

        return result


class StateActionEncoder:
    def __init__(self):
        self.state_extractor = FeatureExtractor()
        self.action_extractor = ActionFeatureExtractor()
        self._state_dim = None
        self._action_dim = 21

    def _get_state_dim(self, state: BackendState, player: int) -> int:
        if self._state_dim is None:
            action_mask = np.zeros(100, dtype=np.float32)
            obs = self.state_extractor.encode_observation(state, player, action_mask)
            flat = self.state_extractor.flatten_observation(obs)
            self._state_dim = len(flat)
        return self._state_dim

    def encode_state_once(self, state: BackendState, player: int) -> np.ndarray:
        action_mask = np.zeros(100, dtype=np.float32)
        obs = self.state_extractor.encode_observation(state, player, action_mask)
        return self.state_extractor.flatten_observation(obs)

    def encode_batch(self, state: BackendState, player: int,
                     ops_list: list[Operation], state_vec: np.ndarray) -> np.ndarray:
        if not ops_list:
            return np.zeros((0, len(state_vec) + self._action_dim), dtype=np.float32)
        action_batch = self.action_extractor.batch_extract(ops_list, state, player)
        state_batch = np.broadcast_to(state_vec, (len(ops_list), len(state_vec)))
        return np.concatenate([state_batch, action_batch], axis=1).astype(np.float32)

    def get_input_dim(self, state: BackendState, player: int) -> int:
        return self._get_state_dim(state, player) + self._action_dim


def find_model_file() -> str | None:
    if not os.path.exists(MODEL_DIR):
        return None
    npz_files = glob.glob(os.path.join(MODEL_DIR, "*.npz"))
    if not npz_files:
        return None
    if len(npz_files) == 1:
        return npz_files[0]
    latest_file = max(npz_files, key=os.path.getmtime)
    print(f"检测到多个模型文件，选择最新的: {os.path.basename(latest_file)}")
    return latest_file


class NeuralNetworkAgent(BaseAgent):
    def __init__(self, model_path: str | None = None, top_n_candidates: int = 32,
                 network: ActionScoringNetwork | None = None, device: str | torch.device = 'cpu'):
        super().__init__()
        self.encoder = StateActionEncoder()
        self.network = network
        self._input_dim = None
        self.top_n_candidates = top_n_candidates
        self._device = torch.device(device)

        if self.network is None:
            load_path = model_path if (model_path is not None and os.path.exists(model_path)) else None
            if load_path is not None:
                self.network = ActionScoringNetwork(model_path=load_path, device=self._device)
            else:
                self.network = None

    def _init_network(self, state: BackendState, player: int):
        if self.network is None:
            self._input_dim = self.encoder.get_input_dim(state, player)
            self.network = ActionScoringNetwork(input_dim=self._input_dim, device=self._device)
            self.encoder._get_state_dim(state, player)

    def score_bundles_batch(self, state: BackendState, player: int,
                            bundles: list[ActionBundle]) -> list[tuple[float, int]]:
        self._init_network(state, player)

        if not bundles:
            return []

        op_to_bundle: list[tuple[int, Operation]] = []
        for b_idx, bundle in enumerate(bundles):
            for op in bundle.operations:
                op_to_bundle.append((b_idx, op))

        if not op_to_bundle:
            return [(0.0, b_idx) for b_idx in range(len(bundles))]

        all_ops = [item[1] for item in op_to_bundle]
        bundle_indices = [item[0] for item in op_to_bundle]

        state_vec = self.encoder.encode_state_once(state, player)
        batch_input = self.encoder.encode_batch(state, player, all_ops, state_vec)

        x_tensor = torch.from_numpy(batch_input).to(self._device)
        with torch.no_grad():
            scores = self.network.forward_batch(x_tensor)
        scores_np = scores.cpu().numpy()

        bundle_scores = np.zeros(len(bundles), dtype=np.float64)
        bundle_counts = np.zeros(len(bundles), dtype=np.int32)
        for b_idx, score in zip(bundle_indices, scores_np):
            bundle_scores[b_idx] += score
            bundle_counts[b_idx] += 1

        for b_idx in range(len(bundles)):
            if bundle_counts[b_idx] > 0:
                bundle_scores[b_idx] /= bundle_counts[b_idx]

        return [(float(bundle_scores[b_idx]), b_idx) for b_idx in range(len(bundles))]

    def choose_bundle(self, state: BackendState, player: int,
                      bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)

        if len(bundles) <= 1:
            return bundles[0]

        if len(bundles) > self.top_n_candidates:
            ranked_by_official = sorted(bundles[1:], key=lambda b: -b.score)[:self.top_n_candidates - 1]
            shortlist = [bundles[0]] + ranked_by_official
        else:
            shortlist = bundles

        scored = self.score_bundles_batch(state, player, shortlist)
        scored.sort(key=lambda x: -x[0])
        best_bundle = shortlist[scored[0][1]]

        return best_bundle


class AI(NeuralNetworkAgent):
    pass


def create_agent():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return AI(device=device)


if __name__ == "__main__":
    state = GameState.initial(seed=7)
    ai = NeuralNetworkAgent()

    for round_idx in range(5):
        print(f"\n=== 回合 {round_idx} ===")
        print(f"玩家0金币: {state.coins[0]}, 玩家1金币: {state.coins[1]}")

        bundles = ai.list_bundles(state, player=0)
        print(f"候选动作数量: {len(bundles)}")

        chosen_bundle = ai.choose_bundle(state, player=0, bundles=bundles)
        print(f"选择的动作: {[str(op) for op in chosen_bundle.operations]}")

        if chosen_bundle.operations:
            state.apply_operation_list(0, chosen_bundle.operations)

        state.advance_round()

    print("\n测试完成！")
