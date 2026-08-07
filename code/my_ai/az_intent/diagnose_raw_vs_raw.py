"""Diagnose trained vs pre-training model behavior on identical states.

Plays a reference game with the pre-training model's raw policy (both sides),
recording every decision state.  Then runs BOTH models' raw (argmax) decode on
each recorded state and reports:
  - raw-action agreement rate (overall + by game phase)
  - when they differ: what action class each chose (aggregated table)
  - value-head comparison (mean by phase, correlation)
  - a few concrete example divergences
  - weight difference between the two checkpoints

Usage:
    python code/my_ai/az_intent/diagnose_raw_vs_raw.py \
        --a training_history/az_intent/az_az3.pt \
        --b training_history/az_intent/gen0120_warm.pt --seed 0
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def describe_ops(ops) -> str:
    if not ops:
        return "WAIT"
    return "+".join(f"op{int(o.op_type)}[{o.arg0},{o.arg1}]" for o in ops)


def op_class_name(op_type: int) -> str:
    try:
        from SDK.utils.constants import OperationType
        return f"{OperationType(op_type).name}({op_type})"
    except Exception:
        return f"op{op_type}"


def main() -> None:
    from SDK.backend.engine import GameState
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import load_model_from_ckpt
    from my_ai.decoder import decode_network_output

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="trained checkpoint")
    parser.add_argument("--b", required=True, help="pre-training checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=512)
    args = parser.parse_args()

    torch.set_num_threads(4)
    model_a = load_model_from_ckpt(args.a)
    model_a.eval()
    model_b = load_model_from_ckpt(args.b)
    model_b.eval()

    # weight difference
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    diffs = {k: (sd_a[k] - sd_b[k]).abs() for k in sd_a if k in sd_b}
    all_d = torch.cat([d.flatten() for d in diffs.values()])
    print(f"weight diff: max={all_d.max():.4f} mean={all_d.mean():.5f} "
          f"std={all_d.std():.4f}  (>0.05 count={(all_d > 0.05).sum().item()})")

    feat = FeatureExtractor(max_actions=96)

    # ── reference game: B (pre-training) raw vs raw, record states ──
    state = GameState.initial(seed=args.seed, cold_handle_rule_illegal=True)
    ref_states: list = []  # (state_clone, player)
    for round_idx in range(args.max_rounds):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            ref_states.append((state.clone(), player))
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model_b(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                              torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            ops = decode_network_output(out, state, player, temperature=0.0,
                                        intent_decoding=True)
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()
    print(f"reference game: B raw vs B raw, {len(ref_states)} decisions, "
          f"winner={state.winner} hp={[b.hp for b in state.bases]} rounds={state.round_index}\n")

    # ── evaluate both models on each recorded state ──
    agree = 0
    by_phase = Counter()
    phase_total = Counter()
    diff_classes: Counter = Counter()  # (A class, B class)
    values_a, values_b = [], []
    examples = []

    for i, (s, player) in enumerate(ref_states):
        obs = feat.encode_observation(s, player, np.zeros(96))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            out_a = model_a(board, stats)
            out_b = model_b(board, stats)

        ops_a = decode_network_output(out_a, s, player, temperature=0.0, intent_decoding=True)
        ops_b = decode_network_output(out_b, s, player, temperature=0.0, intent_decoding=True)

        cls_a = int(ops_a[0].op_type) if ops_a else -1
        cls_b = int(ops_b[0].op_type) if ops_b else -1
        values_a.append(float(out_a["value"].squeeze()))
        values_b.append(float(out_b["value"].squeeze()))

        phase = "early" if i < 100 else ("mid" if i < 400 else "late")
        phase_total[phase] += 1
        same = describe_ops(ops_a) == describe_ops(ops_b)
        if same:
            agree += 1
            by_phase[phase] += 1
        else:
            diff_classes[(op_class_name(cls_a), op_class_name(cls_b))] += 1
            if len(examples) < 6:
                examples.append((i, player, round(s.round_index), describe_ops(ops_a),
                                 describe_ops(ops_b), cls_a, cls_b))

    n = len(ref_states)
    print(f"raw action agreement: {agree}/{n} = {agree/n:.1%}")
    for ph in ("early", "mid", "late"):
        if phase_total[ph]:
            print(f"  {ph:5s}: {by_phase[ph]}/{phase_total[ph]} = {by_phase[ph]/phase_total[ph]:.1%}")

    print(f"\nclass preference when they differ (A vs B):")
    for (ca, cb), cnt in diff_classes.most_common(12):
        print(f"  A={ca:20s} B={cb:20s} x{cnt}")

    va, vb = np.array(values_a), np.array(values_b)
    corr = np.corrcoef(va, vb)[0, 1]
    print(f"\nvalue head: A mean={va.mean():+.3f} B mean={vb.mean():+.3f} "
          f"A std={va.std():.3f} B std={vb.std():.3f} corr={corr:.3f}")
    print(f"  value agreement sign: {(np.sign(va) == np.sign(vb)).mean():.1%}")

    print(f"\nexample divergences:")
    for i, player, r, ops_a, ops_b, ca, cb in examples:
        print(f"  decision#{i} r{r} p{player}: A[{ops_a}] vs B[{ops_b}]  "
              f"({op_class_name(ca)} vs {op_class_name(cb)})")

    print(f"\nfinal hp in ref game: {[b.hp for b in state.bases]}  (for context)")


if __name__ == "__main__":
    main()
