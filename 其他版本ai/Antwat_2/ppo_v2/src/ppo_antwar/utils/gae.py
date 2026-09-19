from typing import List, Optional, Tuple

import torch


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
    final_value: float = 0.0,
    final_values: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 GAE 优势估计和折扣回报

    Args:
        rewards: 每个时间步的奖励列表
        values: 每个时间步的状态价值估计列表
        dones: 每个时间步的是否终止标记列表
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数
        device: 计算设备
        final_value: 截断 episode 末尾的 bootstrap value V(s_{T})。
                     当末步 dones[T-1]=False（截断）时使用此值替代 0。
                     自然终止的 episode 传 0 即可。
        final_values: 每个 episode 的 bootstrap value 列表。
                     由 EpisodeBatch.merge 收集，按 episode 顺序排列。
                     提供时，GAE 在每个 episode 边界使用对应的 bootstrap value，
                     而非默认的 0 或 final_value。

    Returns:
        (advantages, returns) 两个张量

    Note:
        裁剪操作由调用方根据需要自行处理。
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    T = len(rewards)
    advantages = torch.zeros(T, device=device)
    gae = 0.0

    # 构建 episode 边界到 final_value 的映射
    # done=True 的位置即为 episode 结束，按倒序为每个 episode 分配 final_value
    if final_values is not None and len(final_values) > 0:
        episode_final_values = list(final_values)
    else:
        # 兼容旧接口：所有 episode 共用一个 final_value（仅最后一个 episode）
        done_indices = (dones_t > 0.5).nonzero(as_tuple=True)[0].tolist()
        episode_final_values = []
        for i in range(len(done_indices)):
            if i == len(done_indices) - 1:
                episode_final_values.append(final_value)
            else:
                episode_final_values.append(0.0)  # 自然终止
        if not done_indices:
            episode_final_values.append(final_value)

    # 从最后一个 episode 开始倒推匹配 final_value
    # ep_idx 指向当前 episode 边界对应的 bootstrap value
    ep_idx = len(episode_final_values) - 1

    for t in reversed(range(T)):
        if dones_t[t] > 0.5:
            # episode 边界（包括 t == T-1）：
            # 使用该 episode 的 bootstrap value（自然终止为 0.0，截断为 V(s_{T+1})）
            if 0 <= ep_idx < len(episode_final_values):
                next_value = episode_final_values[ep_idx]
            else:
                next_value = 0.0
            ep_idx -= 1
        else:
            # 非边界：使用下一步的 value 估计
            next_value = values_t[t + 1]

        delta = rewards_t[t] + gamma * next_value - values_t[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
        advantages[t] = gae

    returns = advantages + values_t

    return advantages, returns
