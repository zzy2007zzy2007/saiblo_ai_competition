"""Show what each head wants to do. Uses NeuralAgent for gameplay, vs ExampleAI."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import numpy as np
from SDK.backend.engine import GameState
from AI.ai_example import AI as ExampleAI
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from my_ai.decoder import decode_head, make_class_mask, make_position_masks
from utils.logger import get_logger

CLASS_SHORT = {
    0: "0Basic", 1: "1Heavy", 2: "11Heavy+", 3: "12Ice", 4: "13Bewitch",
    5: "2Quick", 6: "21Quick+", 7: "22Double", 8: "23Sniper",
    9: "3Mortar", 10: "31Mortar+", 11: "32Pulse", 12: "33Missile",
    13: "41Producer+", 14: "42Siege", 15: "43Medic",
    16: "13Downgrade", 17: "21Lightning", 18: "22EMP", 19: "23Deflector", 20: "24Evasion",
    21: "31Speed", 22: "32Hp", 23: "HOLD"
}


def diagnose_heads(ckpt_path: str, seed: int = 0, num_heads: int | None = None, log=None, small: bool = False, action_dropout: float = 0.0):
    _print = print
    _empty = lambda: _print()
    if log is not None:
        _print = lambda *a, **kw: log.print(*a, timestamp=False, **kw)
        _empty = lambda: log.print(timestamp=False)

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if num_heads is None:
        if "num_heads" in ckpt:
            num_heads = ckpt["num_heads"]
        elif "config" in ckpt:
            num_heads = ckpt["config"].get("num_heads", 3)
        else:
            import re
            hk = [k for k in ckpt.get("model_state", ckpt) if re.match(r"policy_heads\.\d+\.weight", k)]
            num_heads = max(len(hk), 1) if hk else 3

    if "top2_params" in ckpt and len(ckpt["top2_params"]) > 0:
        t = ckpt["top2_params"][0]
        params = t.numpy() if isinstance(t, torch.Tensor) else t
    elif "mean" in ckpt:
        params = ckpt["mean"].numpy()
    else:
        raise ValueError("No params found")

    model = create_model(num_heads=num_heads, small=small)
    model.set_parameters_from_vector(params)
    _print(f"num_heads={num_heads}, params={len(params):,} device={device}")
    agent = NeuralAgent(model=model, action_dropout=action_dropout)
    opponent = ExampleAI(seed=seed)

    stats = {i: {"ops": 0, "rejected": 0, "classes": []} for i in range(num_heads)}
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    total_ops = 0
    total_holds = 0

    for turn in range(MAX_ROUND):
        if state.terminal:
            break

        u = agent._choose_operations(state, 0)
        opp = opponent.choose_operations(state, 1)

        # Peek at what each head wants
        obs = agent.feature_extractor.encode_observation(state, 0, np.zeros(agent.max_actions))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        st = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            output = model(board, st)

        def _to_np(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            return np.squeeze(t)

        action_map = _to_np(output["action_map"])
        pos_mask = make_position_masks(state, 0)
        cls_mask = make_class_mask(state, 0, position_mask=pos_mask)

        for h in range(num_heads):
            key = f"head{h+1}_logits"
            if key not in output:
                continue
            hl = _to_np(output[key])
            op = decode_head(hl, action_map, cls_mask, pos_mask, state, 0)
            cid = int(np.argmax(hl))
            stats[h]["classes"].append(cid)
            okay = op is not None and state.can_apply_operation(0, op, [])
            if okay:
                stats[h]["ops"] += 1
            else:
                stats[h]["rejected"] += 1
            if okay and op.op_type.name == "BUILD_TOWER":
                pos_mask[:, op.arg0, op.arg1] = False

        state.resolve_turn(u, opp)
        total_ops += len(u) if u else 0
        if not u:
            total_holds += 1

    # Print results
    _print(f"Checkpoint: {ckpt_path}")
    _print(f"num_heads={num_heads}, params={len(params):,}")
    r = "WIN" if state.bases[0].hp > state.bases[1].hp else "LOSS"
    _print(f"Seed={seed}, turns={turn+1}, result={r} ({state.bases[0].hp} vs {state.bases[1].hp})")
    _print(f"Bundle: {total_ops} ops, {total_holds} holds across {turn+1} turns")
    _empty()
    _print(f"{'Head':>6} {'Ops':>6} {'Rej':>6}  Top-8 classes (what head wants to do)")
    _print("-" * 55)
    for h in range(num_heads):
        s = stats[h]
        total = s["ops"] + s["rejected"]
        if total == 0:
            _print(f"  H{h+1}:  (inactive)")
            continue
        top3 = sorted(set(s["classes"]), key=lambda c: s["classes"].count(c), reverse=True)[:8]
        t3 = ", ".join(f"{CLASS_SHORT.get(c,'?')}({s['classes'].count(c)})" for c in top3)
        acc = s["ops"] / total * 100
        _print(f"  H{h+1}: {s['ops']:>4d}/{total:<3d}  {acc:3.0f}%  {t3}")

    total_tries = sum(s["ops"] + s["rejected"] for s in stats.values())
    _print(f"\n  Total head-decode attempts: {total_tries} = {total_tries/(turn+1):.2f}/turn")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true", help="use small model (87K params, 1 head)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="log directory (dual output to terminal + file)")
    args = parser.parse_args()
    log = None
    if args.log_dir:
        log = get_logger(Path(args.log_dir) / "diagnose_heads.log", mode="a")
    diagnose_heads(args.ckpt, args.seed, log=log, small=args.small)
