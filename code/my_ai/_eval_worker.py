"""Isolated eval worker — imports torch INSIDE function body (after env var set).

This avoids CUDA DLL loading in mp.Pool subprocesses on Windows spawn.
The file deliberately has NO 'import torch' at module level.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def _ppo_rollout_and_save(
    params_flat: np.ndarray,
    opp_params_flat: np.ndarray,
    seed: int,
    rollout_dir: str,
    temperature: float = 1.0,
    num_heads: int = 3,
    small: bool = False,
    no_bn: bool = False,
    bn_stats: dict | None = None,
    intent_decoding: bool = True,
    pos_temperature: float = 0.0,
) -> dict:
    """Run one game, save trajectory for PPO training.

    Exploration: the agent uses temperature sampling during gameplay
    (stochastic).  The stored ``action_classes`` are argmax (deterministic),
    so logπ_old is always computed from raw logits via log_softmax.
    Entropy bonus in the PPO objective encourages exploration across
    iterations.

    ``bn_stats`` (dict of running_mean/running_var numpy arrays) must be
    provided when the model uses BatchNorm (no_bn=False) — the parameter
    vector from ``set_parameters_from_vector`` does NOT include BN buffers.

    Returns: {'path': str, 'our_player': int, 'T': int}
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent
    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    model = create_model(num_heads=num_heads, small=small, no_bn=no_bn)
    model.set_parameters_from_vector(params_flat)
    if bn_stats:
        for name, buf in model.state_dict().items():
            if "running_mean" in name or "running_var" in name:
                buf.copy_(torch.from_numpy(bn_stats[name]))
    agent = NeuralAgent(model=model, eval_temperature=temperature,
                        intent_decoding=intent_decoding,
                        pos_temperature=pos_temperature)

    opp_model = create_model(num_heads=num_heads, small=small, no_bn=no_bn)
    opp_model.set_parameters_from_vector(opp_params_flat)
    if bn_stats:
        for name, buf in opp_model.state_dict().items():
            if "running_mean" in name or "running_var" in name:
                buf.copy_(torch.from_numpy(bn_stats[name]))
    opponent = NeuralAgent(model=opp_model)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    boards, stats_list = [], []
    action_classes, action_maps, head_logits_list = [], [], []
    pos_record_list, pos_mask_list = [], []
    values, rewards = [], []

    hp_us_prev = state.bases[our_player].hp
    hp_opp_prev = state.bases[opp_player].hp

    for _ in range(MAX_ROUND):
        if state.terminal:
            break

        ops_us = agent._choose_operations(state, our_player)
        output = agent.last_output

        feat = agent.feature_extractor.encode_observation(
            state, our_player, np.zeros(agent.max_actions))
        boards.append(feat["board"].copy())
        stats_list.append(feat["stats"].copy())

        # Store the ACTUAL sampled class per head (from decoder's sampling,
        # via sampled_class_out).  This is the on-policy action that
        # generated the reward — PPO log-probs must match it.
        cls = np.array(agent.last_sampled_classes, dtype=np.int64)
        action_classes.append(cls)

        # Store sampled position + its log-prob + legal-cell mask per head
        # (for on-policy position targets).  -1 = head did not select a
        # position-bearing action (HOLD / base upgrade / illegal).
        # pos_record cols: [channel, x, y, logprob] — channel is the
        # action_map channel the position came from (may differ from the
        # intent class under intent decoding, e.g. super-weapon → downgrade
        # records channel 16).
        pos_xy = np.full((num_heads, 4), -1, dtype=np.float32)
        pos_mask_arr = np.zeros((num_heads, 19, 19), dtype=np.float16)
        for hi, (ch, x, y, lp, mask) in enumerate(agent.last_sampled_positions[:num_heads]):
            pos_xy[hi, 0] = ch
            pos_xy[hi, 1] = x
            pos_xy[hi, 2] = y
            pos_xy[hi, 3] = lp
            pos_mask_arr[hi] = mask.astype(np.float16)
        pos_record_list.append(pos_xy)
        pos_mask_list.append(pos_mask_arr)

        action_maps.append(output["action_map"].squeeze(0).cpu().numpy())
        # Store z-score normalized head logits (per-head mean/std) — the SAME
        # distribution the decoder sampled from.  Storing the normalized form
        # (not raw ±960 logits) keeps rollout/training distributions identical
        # and float16-safe (normalized values are ~[-3, 3]).
        hsl = np.stack([output[f"head{hi+1}_logits"].squeeze(0).cpu().numpy()
                        for hi in range(num_heads)], axis=0)
        norm_hsl = np.stack([
            (hsl[hi] - hsl[hi].mean()) / (hsl[hi].std() + 1e-8)
            for hi in range(num_heads)
        ], axis=0)
        head_logits_list.append(norm_hsl)
        values.append(float(output["value"].squeeze().cpu().numpy()))

        ops_opp = opponent._choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

        hp_us_now = state.bases[our_player].hp
        hp_opp_now = state.bases[opp_player].hp
        reward = (hp_opp_prev - hp_opp_now) - (hp_us_prev - hp_us_now)
        rewards.append(float(reward))
        hp_us_prev, hp_opp_prev = hp_us_now, hp_opp_now

    # Progress dot (same as _eval_worker: green=win, red=loss, yellow=draw)
    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    if hp_us <= 0 and hp_opp <= 0:
        color = "\033[93m"
    elif hp_us > hp_opp:
        color = "\033[92m"
    elif hp_opp > hp_us:
        color = "\033[91m"
    else:
        color = "\033[93m"
    reset = "\033[0m"
    print(f"{color}.{reset}", end="", flush=True)

    # Save to npz
    path = Path(rollout_dir) / f"ppo_seed{seed:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        boards=np.stack(boards, axis=0).astype(np.float16),
        stats=np.stack(stats_list, axis=0).astype(np.float16),
        action_classes=np.stack(action_classes, axis=0),
        action_maps=np.stack(action_maps, axis=0).astype(np.float16),
        head_logits=np.stack(head_logits_list, axis=0).astype(np.float16),
        pos_record=np.stack(pos_record_list, axis=0).astype(np.float32),
        pos_mask=np.stack(pos_mask_list, axis=0).astype(np.float16),
        values=np.array(values, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
    )
    return {"path": str(path), "our_player": our_player, "T": len(rewards)}


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
    no_bn: bool = False,
    bn_stats: dict | None = None,
    eval_temperature: float = 0.0,
    intent_decoding: bool = True,
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

    model = create_model(num_heads=num_heads, small=small, no_bn=no_bn)
    model.set_parameters_from_vector(params_flat)
    if bn_stats:
        for name, buf in model.state_dict().items():
            if "running_mean" in name or "running_var" in name:
                buf.copy_(torch.from_numpy(bn_stats[name]))
    agent = NeuralAgent(model=model, action_dropout=action_dropout, eval_temperature=eval_temperature, intent_decoding=intent_decoding)

    opp_model = create_model(num_heads=num_heads, small=small, no_bn=no_bn)
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
