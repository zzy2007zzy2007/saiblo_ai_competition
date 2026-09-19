from typing import Dict, List, Optional

import numpy as np
import torch


def _safe_tensor(data, dtype, device):
    """将数据转为 tensor，data 为 None 时返回空 tensor 避免崩溃"""
    if data is None:
        return torch.zeros(0, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, device=device)


class EpisodeBatch:
    """Episode 数据批次 - 收集一局对战的完整轨迹数据"""

    def __init__(self):
        self.observations_board: List[np.ndarray] = []
        self.observations_global: List[np.ndarray] = []
        self.observations_mask: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        self.aux_tower_damage: Optional[np.ndarray] = None
        self.aux_gold_income: Optional[np.ndarray] = None
        self.aux_base_damage: Optional[np.ndarray] = None
        # bool 掩码，标记哪些步有有效的 aux 标签（True=有效，False=被截断需在 loss 中跳过）
        self.aux_valid_mask: Optional[np.ndarray] = None

        # GAE bootstrapping value for truncated episode（截断时作为 V(s_{t+1}) 传入 GAE）
        self.final_value: float = 0.0
        # 每个 episode 的 bootstrap value 列表（merge 后包含所有 episode 的值）
        # 单 episode batch 长度为 1，由 EpisodeCollector.collect 设置
        self.final_values: List[float] = []

    def add_step(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """追加一个时间步的数据"""
        self.observations_board.append(obs["board"])
        self.observations_global.append(obs["global"])
        self.observations_mask.append(obs["action_mask"])
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """将批次数据转换为 PyTorch 张量"""
        tensors = {
            "board": torch.tensor(
                np.array(self.observations_board), dtype=torch.float32, device=device
            ),
            "global": torch.tensor(
                np.array(self.observations_global), dtype=torch.float32, device=device
            ),
            "action_mask": torch.tensor(
                np.array(self.observations_mask), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
            "log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
        }
        tensors["aux_tower_damage"] = _safe_tensor(
            self.aux_tower_damage, torch.float32, device
        )
        tensors["aux_gold_income"] = _safe_tensor(
            self.aux_gold_income, torch.float32, device
        )
        tensors["aux_base_damage"] = _safe_tensor(
            self.aux_base_damage, torch.float32, device
        )
        tensors["aux_valid_mask"] = _safe_tensor(
            self.aux_valid_mask, torch.bool, device
        )
        return tensors

    @staticmethod
    def merge(batches: List["EpisodeBatch"]) -> "EpisodeBatch":
        """合并多个 EpisodeBatch，将数据拼接"""
        merged = EpisodeBatch()
        for b in batches:
            merged.observations_board.extend(b.observations_board)
            merged.observations_global.extend(b.observations_global)
            merged.observations_mask.extend(b.observations_mask)
            merged.actions.extend(b.actions)
            merged.rewards.extend(b.rewards)
            merged.values.extend(b.values)
            merged.log_probs.extend(b.log_probs)
            merged.dones.extend(b.dones)

        if any(b.aux_tower_damage is None for b in batches):
            merged.aux_tower_damage = None
        else:
            merged.aux_tower_damage = np.concatenate(
                [b.aux_tower_damage for b in batches], axis=0
            )
        if any(b.aux_gold_income is None for b in batches):
            merged.aux_gold_income = None
        else:
            merged.aux_gold_income = np.concatenate(
                [b.aux_gold_income for b in batches], axis=0
            )
        if any(b.aux_base_damage is None for b in batches):
            merged.aux_base_damage = None
        else:
            merged.aux_base_damage = np.concatenate(
                [b.aux_base_damage for b in batches], axis=0
            )
        if any(b.aux_valid_mask is None for b in batches):
            merged.aux_valid_mask = None
        else:
            merged.aux_valid_mask = np.concatenate(
                [b.aux_valid_mask for b in batches], axis=0
            )

        # 收集各 component batch 的 final_values，供 GAE 计算访问每个 episode 边界
        for b in batches:
            if b.final_values:
                merged.final_values.extend(b.final_values)
            else:
                # 兼容未设置 final_values 的旧 batch
                merged.final_values.append(b.final_value)
        merged.final_value = batches[-1].final_value if batches else 0.0

        return merged

    def __len__(self) -> int:
        return len(self.actions)
