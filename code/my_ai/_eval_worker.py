"""Isolated eval worker — imports torch INSIDE function body (after env var set).

This avoids CUDA DLL loading in mp.Pool subprocesses on Windows spawn.
The file deliberately has NO 'import torch' at module level.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def _eval_worker(
    params_flat: np.ndarray,
    opp_params_flat: np.ndarray,
    seed: int,
    num_heads: int = 3,
    bc_dir=None,
    gen=0,
    ind=0,
    action_dropout: float = 0.0,
    small: bool = False,
    bn_stats: dict | None = None,
) -> dict:
    """Run one match: params vs opponent params, collect game data to .npz."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    # Lazy imports — avoid triggering torch at module level in subprocesses
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent
    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    model = create_model(num_heads=num_heads, small=small)
    model.set_parameters_from_vector(params_flat)
    if bn_stats:
        for name, buf in model.state_dict().items():
            if "running_mean" in name or "running_var" in name:
                buf.copy_(torch.from_numpy(bn_stats[name]))
    agent = NeuralAgent(model=model, action_dropout=action_dropout)

    opp_model = create_model(num_heads=num_heads, small=small)
    opp_model.set_parameters_from_vector(opp_params_flat)
    if bn_stats:
        for name, buf in opp_model.state_dict().items():
            if "running_mean" in name or "running_var" in name:
                buf.copy_(torch.from_numpy(bn_stats[name]))
    opponent = NeuralAgent(model=opp_model)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    bc_boards, bc_stats = [], []
    bc_class_labels, bc_action_maps, bc_head_logits = [], [], []
    hp_traj = []

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops_us = agent._choose_operations(state, our_player)
        if bc_dir and not agent.dropout_this_turn:
            feat = agent.feature_extractor.encode_observation(
                state, our_player, np.zeros(agent.max_actions))
            bc_boards.append(feat["board"].copy())
            bc_stats.append(feat["stats"].copy())

            hp_traj.append((state.bases[our_player].hp,
                            state.bases[opp_player].hp))

            cls_labels = []
            head_logits_list = []
            for hi in range(num_heads):
                logits = agent.last_output[f"head{hi+1}_logits"].squeeze(0)
                cls_labels.append(logits.argmax().item())
                head_logits_list.append(logits.cpu().numpy())
            bc_class_labels.append(np.array(cls_labels))
            bc_head_logits.append(np.stack(head_logits_list, axis=0))

            bc_action_maps.append(
                agent.last_output["action_map"].squeeze(0).cpu().numpy()
            )

        ops_opp = opponent._choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp

    if bc_dir and bc_boards:
        boards_arr = np.stack(bc_boards, axis=0)
        stats_arr = np.stack(bc_stats, axis=0)
        class_arr = np.stack(bc_class_labels, axis=0)
        map_arr = np.stack(bc_action_maps, axis=0)
        logits_arr = np.stack(bc_head_logits, axis=0)
        value_labels = np.array(hp_traj, dtype=np.float32)

        _write_npz(
            Path(bc_dir) / f"gen_{gen:04d}_ind{ind:03d}_seed{seed}.npz",
            boards_arr, stats_arr, class_arr, map_arr, logits_arr,
            value=value_labels,
        )

    if hp_us <= 0 and hp_opp <= 0:
        result = {"score": 0.5, "our_player": our_player}
    elif hp_us > hp_opp:
        result = {"score": 1.0, "our_player": our_player}
    elif hp_opp > hp_us:
        result = {"score": 0.0, "our_player": our_player}
    else:
        result = {"score": 0.5, "our_player": our_player}
    score = result["score"]
    if score == 1.0:
        color = "\033[92m"
    elif score == 0.0:
        color = "\033[91m"
    else:
        color = "\033[93m"
    reset = "\033[0m"
    print(f"{color}.{reset}", end="", flush=True)
    return result


def _write_npz(path, board, stats, class_, action_map, head_logits, value=None):
    """Save game timestep data as compressed .npz (no torch dependency)."""
    F16_MAX = 65504.0
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(
        board=board.astype(np.float16),
        stats=stats.astype(np.float16),
        class_=class_,
        action_map=np.clip(action_map, -F16_MAX, F16_MAX).astype(np.float16),
        head_logits=np.clip(head_logits, -F16_MAX, F16_MAX).astype(np.float16),
    )
    if value is not None:
        kw["value"] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(path, **kw)
