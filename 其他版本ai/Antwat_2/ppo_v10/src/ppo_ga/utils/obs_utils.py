from typing import Dict

import numpy as np
import torch


def observation_to_tensors(
    observation: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """将 numpy 观测转换为模型输入格式的 PyTorch 张量"""
    return {
        "board": torch.tensor(
            observation["board"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "global": torch.tensor(
            observation["global"], dtype=torch.float32, device=device
        ).unsqueeze(0),
    }
