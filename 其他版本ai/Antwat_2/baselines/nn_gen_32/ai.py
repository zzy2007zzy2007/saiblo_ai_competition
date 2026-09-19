#!/usr/bin/env python3
"""
神经网络AI代理 - 提交版本
"""
from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# 添加SDK路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 导入SDK模块
from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import TowerType, MAP_SIZE
from SDK.backend.model import Operation, OperationType
from SDK.backend.state import BackendState
from SDK.backend.engine import GameState
from SDK.utils.actions import ActionBundle

from common import BaseAgent


class ActionFeatureType(IntEnum):
    """动作类型枚举"""
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
    """动作特征向量"""
    type_onehot: np.ndarray  # 动作类型独热编码 (10维)
    position: np.ndarray     # 位置特征 (2维, 归一化坐标)
    tower_position: np.ndarray  # 塔位置特征 (2维, 归一化坐标)
    tower_type: np.ndarray   # 塔类型独热编码 (5维)
    cost: float              # 动作成本 (归一化金币)
    is_valid: float          # 是否有效动作


class ActionFeatureExtractor:
    """动作特征提取器"""
    
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


class StateActionEncoder:
    """状态-动作联合编码器"""
    
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
    
    def encode(self, state: BackendState, player: int, op: Operation) -> np.ndarray:
        action_mask = np.zeros(100, dtype=np.float32)
        obs = self.state_extractor.encode_observation(state, player, action_mask)
        state_vec = self.state_extractor.flatten_observation(obs)
        
        action_features = self.action_extractor.extract(op, state, player)
        action_vec = self.action_extractor.flatten(action_features)
        
        return np.concatenate([state_vec, action_vec]).astype(np.float32)
    
    def get_input_dim(self, state: BackendState, player: int) -> int:
        return self._get_state_dim(state, player) + self._action_dim


import os
import glob

# 模型存储目录（相对于ai.py的位置）
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "action_scoring_model.npz")


def find_model_file() -> str | None:
    """自动查找models/文件夹中的npz模型文件"""
    if not os.path.exists(MODEL_DIR):
        return None
    
    # 查找所有npz文件
    npz_files = glob.glob(os.path.join(MODEL_DIR, "*.npz"))
    
    if not npz_files:
        return None
    
    # 如果只有一个npz文件，直接返回
    if len(npz_files) == 1:
        return npz_files[0]
    
    # 如果有多个，选择修改时间最新的
    latest_file = max(npz_files, key=os.path.getmtime)
    print(f"检测到多个模型文件，选择最新的: {os.path.basename(latest_file)}")
    return latest_file


class ActionScoringNetwork:
    """动作评分神经网络"""
    
    def __init__(self, input_dim: int = 10171, hidden_dim: int = 512, model_path: str | None = None, auto_load: bool = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 如果提供了模型路径且文件存在，加载预训练模型
        if model_path is not None and os.path.exists(model_path):
            self._load_model(model_path)
        # 如果允许自动加载，尝试自动查找模型文件
        elif auto_load:
            auto_found_path = find_model_file()
            if auto_found_path is not None:
                self._load_model(auto_found_path)
            elif os.path.exists(DEFAULT_MODEL_PATH):
                self._load_model(DEFAULT_MODEL_PATH)
            else:
                self._initialize_random()
        # 否则随机初始化
        else:
            self._initialize_random()
    
    def _initialize_random(self):
        """随机初始化参数"""
        rng = np.random.default_rng(42)
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        scale3 = np.sqrt(2.0 / (self.hidden_dim // 2))
        
        self.w1 = rng.normal(0.0, scale1, (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        
        self.w2 = rng.normal(0.0, scale2, (self.hidden_dim, self.hidden_dim // 2)).astype(np.float32)
        self.b2 = np.zeros(self.hidden_dim // 2, dtype=np.float32)
        
        self.w3 = rng.normal(0.0, scale3, (self.hidden_dim // 2, self.hidden_dim // 4)).astype(np.float32)
        self.b3 = np.zeros(self.hidden_dim // 4, dtype=np.float32)
        
        self.w_out = rng.normal(0.0, np.sqrt(2.0 / (self.hidden_dim // 4)), 
                                (self.hidden_dim // 4, 1)).astype(np.float32)
        self.b_out = np.zeros(1, dtype=np.float32)
        
        # 默认归一化系数
        self.max_abs_score = 2094.42
    
    def _load_model(self, path: str):
        """从文件加载模型参数"""
        data = np.load(path)
        self.input_dim = int(data['input_dim'])
        self.hidden_dim = int(data['hidden_dim'])
        
        # 加载归一化系数（用于还原原始评分）
        self.max_abs_score = float(data.get('max_abs_score', 2094.42))
        
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        
        if 'w3' in data:
            self.w3 = data['w3']
            self.b3 = data['b3']
            self.w_out = data['w_out']
            self.b_out = data['b_out']
        else:
            self.w_out = data['w_out']
            self.b_out = data['b_out']
            self.w3 = None
            self.b3 = None
    
    def forward(self, x: np.ndarray) -> float:
        """前向传播，输出动作评分（已还原为原始范围）"""
        hidden1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        hidden2 = np.maximum(hidden1 @ self.w2 + self.b2, 0.0)
        
        if self.w3 is not None:
            hidden3 = np.maximum(hidden2 @ self.w3 + self.b3, 0.0)
            score = float(np.tanh((hidden3 @ self.w_out + self.b_out)[0]))
        else:
            score = float(np.tanh((hidden2 @ self.w_out + self.b_out)[0]))
        
        # 将归一化评分还原为原始范围
        return score * self.max_abs_score


class NeuralNetworkAgent(BaseAgent):
    """使用神经网络评分的AI代理"""
    
    def __init__(self, model_path: str | None = None, top_n_candidates: int = 32, network: ActionScoringNetwork | None = None):
        super().__init__()
        self.encoder = StateActionEncoder()
        self.network = network
        self._input_dim = None
        self.top_n_candidates = top_n_candidates
        
        if self.network is None:
            load_path = model_path if (model_path is not None and os.path.exists(model_path)) else None
            
            if load_path is not None:
                self.network = ActionScoringNetwork(model_path=load_path)
            else:
                self.network = None
    
    def _init_network(self, state: BackendState, player: int):
        if self.network is None:
            self._input_dim = self.encoder.get_input_dim(state, player)
            self.network = ActionScoringNetwork(input_dim=self._input_dim)
    
    def score_bundle(self, state: BackendState, player: int, bundle: ActionBundle) -> float:
        self._init_network(state, player)
        
        if not bundle.operations:
            return 0.0
        
        scores = []
        for op in bundle.operations:
            input_vec = self.encoder.encode(state, player, op)
            score = self.network.forward(input_vec)
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        
        if len(bundles) <= 1:
            return bundles[0]
        
        # 第一步：使用官方评分筛选前N个候选
        if len(bundles) > self.top_n_candidates:
            ranked_by_official = sorted(bundles[1:], key=lambda b: -b.score)[:self.top_n_candidates-1]
            shortlist = [bundles[0]] + ranked_by_official
        else:
            shortlist = bundles
        
        # 第二步：使用神经网络对筛选后的候选进行评分
        scored_bundles = []
        for bundle in shortlist:
            score = self.score_bundle(state, player, bundle)
            scored_bundles.append((score, bundle))
        
        scored_bundles.sort(key=lambda x: -x[0])
        best_bundle = scored_bundles[0][1]
        
        return best_bundle


# 提交用AI类
class AI(NeuralNetworkAgent):
    pass


# 测试
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


def create_agent():
    return AI()
