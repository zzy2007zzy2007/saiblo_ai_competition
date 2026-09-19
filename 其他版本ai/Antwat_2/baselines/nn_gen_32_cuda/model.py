from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import os
import glob


class ActionScoringNetwork(nn.Module):
    """PyTorch 版动作评分神经网络，支持 GPU batch forward

    从 numpy 版迁移，支持 2 层或 3 层隐藏层架构。
    """

    def __init__(self, input_dim: int = 10171, hidden_dim: int = 512,
                 model_path: str | None = None, device: str | torch.device = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_abs_score = 2094.42

        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(input_dim, hidden_dim))
        self.net.add_module('relu1', nn.ReLU())
        self.net.add_module('fc2', nn.Linear(hidden_dim, hidden_dim // 2))
        self.net.add_module('relu2', nn.ReLU())
        self.net.add_module('fc3', nn.Linear(hidden_dim // 2, hidden_dim // 4))
        self.net.add_module('relu3', nn.ReLU())
        self.net.add_module('fc_out', nn.Linear(hidden_dim // 4, 1))

        if model_path is not None and os.path.exists(model_path):
            self.load_npz(model_path)
        else:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            auto_path = self._find_model_file(model_dir)
            if auto_path:
                self.load_npz(auto_path)

        self._device = torch.device(device)
        self.to(self._device)

    @staticmethod
    def _find_model_file(model_dir: str) -> str | None:
        if not os.path.exists(model_dir):
            return None
        npz_files = glob.glob(os.path.join(model_dir, "*.npz"))
        if not npz_files:
            return None
        if len(npz_files) == 1:
            return npz_files[0]
        return max(npz_files, key=os.path.getmtime)

    def load_npz(self, path: str):
        data = np.load(path)
        self.input_dim = int(data['input_dim'])
        self.hidden_dim = int(data['hidden_dim'])
        self.max_abs_score = float(data.get('max_abs_score', 2094.42))
        self._rebuild_from_npz(data)

    def _rebuild_from_npz(self, data: np.lib.npyio.NpzFile):
        w1 = torch.from_numpy(data['w1'].copy())
        b1 = torch.from_numpy(data['b1'].copy())
        w2 = torch.from_numpy(data['w2'].copy())
        b2 = torch.from_numpy(data['b2'].copy())

        if 'w3' in data:
            w3 = torch.from_numpy(data['w3'].copy())
            b3 = torch.from_numpy(data['b3'].copy())
            w_out = torch.from_numpy(data['w_out'].copy())
            b_out = torch.from_numpy(data['b_out'].copy())
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, 1),
            )
            self.net[0].weight.data = w1.t()
            self.net[0].bias.data = b1
            self.net[2].weight.data = w2.t()
            self.net[2].bias.data = b2
            self.net[4].weight.data = w3.t()
            self.net[4].bias.data = b3
            self.net[6].weight.data = w_out.t()
            self.net[6].bias.data = b_out
        else:
            w_out = torch.from_numpy(data['w_out'].copy())
            b_out = torch.from_numpy(data['b_out'].copy())
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )
            self.net[0].weight.data = w1.t()
            self.net[0].bias.data = b1
            self.net[2].weight.data = w2.t()
            self.net[2].bias.data = b2
            self.net[4].weight.data = w_out.t()
            self.net[4].bias.data = b_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        score = torch.tanh(h)
        return score * self.max_abs_score

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).squeeze(-1)
