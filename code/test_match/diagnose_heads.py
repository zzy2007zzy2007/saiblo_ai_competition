"""Show what each head wants to do. Uses NeuralAgent for gameplay (identical to diagnose_model.py)."""
from __future__ import annotations
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
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from my_ai.decoder import decode_head, make_class_mask, make_position_masks

CLASS_SHORT = {
    0: "B", 1: "H", 2: "H+", 3: "I", 4: "W", 5: "Q", 6: "Q+", 7: "D",
    8: "S", 9: "M", 10: "M+", 11: "P", 12: "R", 13: "P+", 14: "G", 15: "E",
    16: "\u2193", 17: "\u26a1", 18: "EMP", 19: "Grv", 20: "Evs",
    21: "\u2642\u2191", 22: "\u2665\u2191", 23: "\u2205"
}


def diagnose_heads(ckpt_path: str, seed: int = 0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

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

    model = create_model(num_heads=num_heads)
    model.set_parameters_from_vector(params)
    agent = NeuralAgent(model=model)

    stats = {i: {"ops": 0, "rejected": 0, "classes": []} for i in range(num_heads)}
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    total_ops = 0
    total_holds = 0

    for turn in range(MAX_ROUND):
        if state.terminal:
            break

        # Use NeuralAgent's _choose_operations (identical to diagnose_model.py)
        u = agent._choose_operations(state, 0)

        # Also peek at what each head wants (re-run forward on current state)
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
            cid = int(np.argmax(hl * cls_mask.astype(np.float32)))
            stats[h]["classes"].append(cid)
            okay = op is not None and state.can_apply_operation(0, op, [])
            if okay:
                stats[h]["ops"] += 1
            else:
                stats[h]["rejected"] += 1
            if okay and op.op_type.name == "BUILD_TOWER":
                pos_mask[:, op.arg0, op.arg1] = False

        state.resolve_turn(u, [])
        total_ops += len(u) if u else 0
        if not u:
            total_holds += 1

    # Print results
    print(f"Checkpoint: {ckpt_path}")
    print(f"num_heads={num_heads}, params={len(params):,}")
    r = "WIN" if state.bases[0].hp > state.bases[1].hp else "LOSS"
    print(f"Seed={seed}, turns={turn+1}, result={r} ({state.bases[0].hp} vs {state.bases[1].hp})")
    print(f"Bundle: {total_ops} ops, {total_holds} holds across {turn+1} turns")
    print()
    print(f"{'Head':>6} {'Ops':>6} {'Rej':>6}  Top-3 classes (what head wants to do)")
    print("-" * 55)
    for h in range(num_heads):
        s = stats[h]
        total = s["ops"] + s["rejected"]
        if total == 0:
            print(f"  H{h+1}:  (inactive)")
            continue
        top3 = sorted(set(s["classes"]), key=lambda c: s["classes"].count(c), reverse=True)[:3]
        t3 = ", ".join(f"{CLASS_SHORT.get(c,'?')}({s['classes'].count(c)})" for c in top3)
        acc = s["ops"] / total * 100
        print(f"  H{h+1}: {s['ops']:>4d}/{total:<3d}  {acc:3.0f}%  {t3}")

    total_tries = sum(s["ops"] + s["rejected"] for s in stats.values())
    print(f"\n  Total head-decode attempts: {total_tries} = {total_tries/(turn+1):.2f}/turn")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose_heads(args.ckpt, args.seed)
