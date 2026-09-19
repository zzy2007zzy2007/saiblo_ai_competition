#!/usr/bin/env python3
"""
测试 ann_v1 模型加载
使用 ann_593 的网络结构
"""

import os
import sys
import torch
import torch.nn as nn

print("=" * 70)
print("AnnV1 模型加载测试")
print("=" * 70)

# 配置
BASELINES_DIR = '/root/autodl-tmp/AntWar/baselines'

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

class ActorCriticNetwork(nn.Module):
    """复制 ann_593 的网络结构"""
    def __init__(self, input_dim, action_dim, hidden_dim=512):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        features = self.shared_layers(x)
        action_probs = self.actor_head(features)
        value = self.critic_head(features)
        return action_probs, value

    def get_logits(self, x):
        features = self.shared_layers(x)
        logits = self.actor_head[:-1](features)
        return logits


class SimpleNN(nn.Module):
    """简单的 3 层网络"""
    def __init__(self, board_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(board_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NeuralNetworkAgent(nn.Module):
    """使用 ActorCriticNetwork 的智能体"""
    def __init__(self, board_dim=10144, action_dim=32, hidden_dim=256, use_actor_critic=False):
        super().__init__()
        self.board_dim = board_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_actor_critic = use_actor_critic

        if use_actor_critic:
            self.network = ActorCriticNetwork(board_dim, action_dim, hidden_dim)
        else:
            self.network = SimpleNN(board_dim, action_dim, hidden_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def load_model(self, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return False

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'network' in checkpoint:
                state_dict = checkpoint['network']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            print(f"模型键: {list(state_dict.keys())}")
            print(f"第一个权重形状: {list(state_dict.values())[0].shape}")

            self.network.load_state_dict(state_dict)
            self.network.eval()
            print("✓ 模型加载成功")
            return True

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_ann_v1_with_hidden_dim(hidden_dim):
    """测试 ann_v1 使用特定的 hidden_dim"""
    print(f"\n测试 hidden_dim={hidden_dim}:")

    model_dir = os.path.join(BASELINES_DIR, 'ann_v1')
    model_path = os.path.join(model_dir, 'model.pth')

    agent = NeuralNetworkAgent(board_dim=10144, action_dim=32, hidden_dim=hidden_dim, use_actor_critic=True)

    if agent.load_model(model_path):
        return True
    return False


def main():
    model_dir = os.path.join(BASELINES_DIR, 'ann_v1')
    model_path = os.path.join(model_dir, 'model.pth')

    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")

    # 先查看模型结构
    print("\n检查模型结构...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"模型类型: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"模型键: {list(checkpoint.keys())}")
    else:
        print("模型不是字典类型")

    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")

    # 尝试不同的 hidden_dim
    print("\n尝试不同的 hidden_dim:")
    for hidden_dim in [64, 128, 256, 512]:
        test_ann_v1_with_hidden_dim(hidden_dim)


if __name__ == "__main__":
    main()
