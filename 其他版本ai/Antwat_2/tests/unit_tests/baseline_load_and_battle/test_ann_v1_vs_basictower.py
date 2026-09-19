#!/usr/bin/env python3
"""
测试AnnV1模型加载和与BasicTowerAI对战
重点测试如何正确加载AnnV1模型
"""

import os
import sys
import torch
import numpy as np

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'saiblo-antwar-sdk-python', 'saiblo-antwar-sdk-python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'baselines', 'ann_v1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'baselines', 'ann_593'))

from antwar.protocol import Operation, OperationType
from antwar.coord import Coord
from SDK.backend.state import PythonBackendState
from SDK.backend.engine import GameState as BackendGameState
from SDK.backend.core import load_backend
from neural_network import NeuralNetworkAgent

class BasicTowerAI:
    """简单策略：优先建造防御塔"""
    def choose_operations(self, state, player):
        positions = [Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12)]
        for pos in positions:
            if state.coins[player] >= 30:
                from antwar.protocol import build_tower_op
                return [build_tower_op(pos)]
        return []

def load_and_test_ann_v1_model():
    """加载并测试AnnV1模型"""
    print("=" * 60)
    print("测试AnnV1模型加载")
    print("=" * 60)

    model_path = os.path.join(os.path.dirname(__file__), '../../..', 'baselines', 'ann_v1', 'model.pth')
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")

    # 测试不同的hidden_dim参数
    test_hidden_dims = [64, 128, 256, 512]

    for hidden_dim in test_hidden_dims:
        print(f"\n测试 hidden_dim={hidden_dim}:")
        try:
            agent = NeuralNetworkAgent(10144, 32, hidden_dim=hidden_dim)
            print("  ✓ 成功创建NeuralNetworkAgent")

            agent.load_model(model_path)
            print("  ✓ 模型加载成功！")
            print(f"  正确的hidden_dim参数是: {hidden_dim}")
            return hidden_dim
        except Exception as e:
            print(f"  ✗ 模型加载失败:")
            print(f"    错误信息: {e}")

    return None

def test_battle(ann_v1_hidden_dim, episodes=10):
    """测试AnnV1与BasicTowerAI的对战"""
    print("\n" + "=" * 60)
    print(f"测试AnnV1(hidden_dim={ann_v1_hidden_dim}) vs BasicTowerAI")
    print("=" * 60)

    # 创建AnnV1代理
    ann_v1 = NeuralNetworkAgent(10144, 32, hidden_dim=ann_v1_hidden_dim)
    model_path = os.path.join(os.path.dirname(__file__), '../../..', 'baselines', 'ann_v1', 'model.pth')
    ann_v1.load_model(model_path)

    # 创建BasicTowerAI代理
    basic_tower = BasicTowerAI()

    # 统计
    ann_v1_wins = 0
    basic_tower_wins = 0
    draws = 0
    total_rounds = 0

    for episode in range(episodes):
        print(f"\n第 {episode + 1} 场对战:")

        # 初始化游戏
        backend = load_backend(prefer_native=False)
        game_state = backend.initial_state(seed=episode)
        state = PythonBackendState(game_state)

        round_count = 0
        max_rounds = 200

        while state.bases[0].hp > 0 and state.bases[1].hp > 0 and round_count < max_rounds:
            round_count += 1

            # AnnV1选择动作
            action_mask = np.ones(32, dtype=np.float32)
            from SDK.utils.features import FeatureExtractor
            feature_extractor = FeatureExtractor()
            observation = feature_extractor.encode_observation(state, 0, action_mask)
            flattened_obs = feature_extractor.flatten_observation(observation)
            obs_tensor = torch.tensor(flattened_obs, dtype=torch.float32).unsqueeze(0).to(ann_v1.device)

            with torch.no_grad():
                action_probs, _ = ann_v1.network(obs_tensor)

            action_probs = action_probs.cpu()
            from torch.distributions import Categorical
            dist = Categorical(action_probs)
            action_idx = dist.sample().item()

            # AnnV1执行动作（这里简化为建造防御塔）
            if action_idx < 10:
                positions = [
                    Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12),
                    Coord(7, 7), Coord(7, 11), Coord(11, 7), Coord(11, 11),
                    Coord(9, 4), Coord(9, 14)
                ]
                pos = positions[action_idx]
                if state.coins[0] >= 30:
                    from antwar.protocol import build_tower_op
                    from SDK.backend.model import Operation as BackendOperation
                    from SDK.utils.constants import OperationType as BackendOperationType
                    op = build_tower_op(pos)
                    backend_op = BackendOperation(
                        op_type=BackendOperationType(op.type.value),
                        arg0=op.arg0,
                        arg1=op.arg1
                    )
                    state.apply_operation(0, backend_op)

            # BasicTowerAI选择动作
            basic_ops = basic_tower.choose_operations(state, 1)
            for op in basic_ops:
                from SDK.backend.model import Operation as BackendOperation
                from SDK.utils.constants import OperationType as BackendOperationType
                backend_op = BackendOperation(
                    op_type=BackendOperationType(op.type.value),
                    arg0=op.arg0,
                    arg1=op.arg1
                )
                state.apply_operation(1, backend_op)

            # 模拟下一回合
            state.advance_round()

        total_rounds += round_count

        # 判断胜负
        if state.bases[0].hp > 0 and state.bases[1].hp <= 0:
            ann_v1_wins += 1
            result = "AnnV1胜"
        elif state.bases[0].hp <= 0 and state.bases[1].hp > 0:
            basic_tower_wins += 1
            result = "BasicTowerAI胜"
        else:
            draws += 1
            result = "平局"

        print(f"  回合数: {round_count}, 结果: {result}")
        print(f"  AnnV1基地HP: {state.bases[0].hp}, BasicTowerAI基地HP: {state.bases[1].hp}")

    # 打印统计结果
    print("\n" + "=" * 60)
    print("对战统计结果")
    print("=" * 60)
    print(f"总场次: {episodes}")
    print(f"AnnV1胜: {ann_v1_wins} ({ann_v1_wins/episodes*100:.1f}%)")
    print(f"BasicTowerAI胜: {basic_tower_wins} ({basic_tower_wins/episodes*100:.1f}%)")
    print(f"平局: {draws} ({draws/episodes*100:.1f}%)")
    print(f"平均回合数: {total_rounds/episodes:.1f}")

    return {
        'ann_v1_wins': ann_v1_wins,
        'basic_tower_wins': basic_tower_wins,
        'draws': draws,
        'avg_rounds': total_rounds / episodes
    }

def main():
    print("开始AnnV1模型加载和对战测试...")
    print()

    # 测试1: 找出正确的hidden_dim参数
    correct_hidden_dim = load_and_test_ann_v1_model()

    if correct_hidden_dim is None:
        print("\n错误: 无法加载AnnV1模型！")
        print("请检查模型文件是否存在以及模型结构是否正确。")
        return

    print(f"\n确定的正确hidden_dim参数: {correct_hidden_dim}")

    # 测试2: 使用正确的hidden_dim进行对战测试
    test_battle(correct_hidden_dim, episodes=10)

    print("\n测试完成！")

if __name__ == "__main__":
    main()