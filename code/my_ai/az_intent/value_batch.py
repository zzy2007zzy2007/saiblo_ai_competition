"""批量价值函数：把"一堆状态 → 一堆标量"包成一次前向。

动机（2026-10-21）：`mc_score='net'` 的仲裁需要给每个候选的 N 个叶子打分。
原实现逐叶调用 `net_fn`（三网、未批量）⇒ 72 次 × 7.6 ms ≈ 0.55 s，占仲裁成本 80%，
其中 2/3 是白跑的（class/pos 两网的输出被丢弃）。

这里只用**值网**，并且把 N 个叶子**批量**编码 + 一次前向：
  · 单叶 `encode_observation` ≈ 414 µs（空闲）⇒ 批量摊薄
  · 值网 batch=32 时 ≈ 264 µs/条（CPU 实测最优点）
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def make_batched_value_fn(value_model, feature_extractor, *,
                          batch_size: int = 32,
                          tanh: bool = True) -> Callable[[Sequence, int], np.ndarray]:
    """返回 `fn(states, player) -> np.ndarray(shape=(len(states),))`。

    终局状态用 ±1（胜负）代替网络输出，避免"已结束的局面"被网络打出一个中间值。
    """
    import torch

    zeros = np.zeros(96)

    def value_fn(states: Sequence, player: int) -> np.ndarray:
        n = len(states)
        out = np.zeros(n, dtype=np.float64)
        pending_idx, pending_states = [], []
        for i, st in enumerate(states):
            if getattr(st, "terminal", False):
                winner = getattr(st, "winner", None)
                out[i] = 0.0 if winner is None else (1.0 if int(winner) == int(player) else -1.0)
            else:
                pending_idx.append(i)
                pending_states.append(st)
        if not pending_states:
            return out

        for start in range(0, len(pending_states), batch_size):
            chunk = pending_states[start:start + batch_size]
            boards, stats = [], []
            for st in chunk:
                obs = feature_extractor.encode_observation(st, player, zeros)
                boards.append(obs["board"])
                stats.append(obs["stats"])
            board = torch.from_numpy(np.stack(boards)).float()
            stat = torch.from_numpy(np.stack(stats)).float()
            if next(value_model.parameters()).is_cuda:
                board, stat = board.cuda(), stat.cuda()
            with torch.no_grad():
                raw = value_model(board, stat)["value"].squeeze(-1)
                if tanh:
                    raw = torch.tanh(raw)
            vals = raw.detach().cpu().numpy().tolist()
            for offset, val in enumerate(vals):
                out[pending_idx[start + offset]] = float(val)
        return out

    return value_fn
