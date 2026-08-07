"""Bundle-MCTS AlphaZero training (T2, round 2) — docs/az_weighted_value_retraining_plan.md.

Loads self-play samples collected by az_selfplay.py and trains:
  - policy loss: bundle visit distribution marginalized to per-head intent
    targets (count-weighted, §2), then a DECOMPOSED CE (§3.1): class CE over
    the class marginal + per-class weighted position CE, using the SAME
    distributions the search samples from (z-score + temperatures).
  - anchor loss: MSE of the policy outputs vs the recorded self-play outputs
    (trust-region, §3.2).
  - value loss: MSE against exponentially-weighted future HP-difference labels
    (docs/value_label_future_weighted.md), computed per frame from the stats
    feature (extras offset 22) so every decision gets a distinct value target.

Round-2 data split: the policy head trains on the CURRENT batch's samples only
(policy targets go stale fast); the value head trains on the ACCUMULATED data
(game outcomes are facts that stay useful).  Both are optimized jointly every
step (two mini-batches, one loss) — NOT in phases.

Usage:
    python code/my_ai/az_intent/az_train.py --init training_history/az_intent/gen0120_warm.pt \
        --policy-dir training_history/az_intent/az_selfplay_data/batch4 \
        --data-dir training_history/az_intent/az_selfplay_data \
        --checkpoint training_history/az_intent/az_r2.pt
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from my_ai.az_intent.mcts import HP_SCALE


def load_samples(pkl_paths: list[Path]) -> list[dict]:
    """Load self-play pickles into a flat list of samples."""
    samples: list[dict] = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        samples.extend(d["samples"])
    return samples


def add_weighted_labels(samples: list[dict], tau: float = 20.0,
                        label_scale: float = 1.0) -> None:
    """In-place: set ``value_label`` = exp-weighted future HP-diff (P0 view, /HP_SCALE).

    Per-frame instantaneous HP diff d_t is reconstructed from the stats feature
    (extras offset 22: stats[24] = bases[player].hp/50, stats[25] = bases[enemy].hp/50).
    Relative form: label_t = weighted_future_avg - d_t (0-centered, predicts the
    future advantage CHANGE from here).  ``label_scale`` amplifies the labels so
    the trained value head's output magnitude matches the terminal-scale values
    the search expects (else the PUCT explore term dominates and search starves).
    Backward O(n) recurrence.
    """
    gamma = float(np.exp(-1.0 / tau))
    n = len(samples)
    if n == 0:
        return
    d = np.empty(n, dtype=np.float64)
    for t, s in enumerate(samples):
        if s["player"] == 0:
            d[t] = (float(s["stats"][24]) - float(s["stats"][25])) * 50.0
        else:
            d[t] = (float(s["stats"][25]) - float(s["stats"][24])) * 50.0
    suffix = wsum = 0.0
    for t in range(n - 1, -1, -1):
        suffix = d[t] + gamma * suffix
        wsum = 1.0 + gamma * wsum
        raw = suffix / wsum
        label = float(np.clip((raw - d[t]) / HP_SCALE, -1.0, 1.0)) * label_scale
        # value must be from the SAMPLE's player perspective (features are
        # player-perspective; the search reads it as the current player's value)
        samples[t]["value_label"] = label if samples[t]["player"] == 0 else -label


def _marginalize(s: dict, head: int) -> dict:
    """Count-weighted marginalization: bundle visit -> per-intent target mass."""
    target: dict = {}
    for counts, visit in zip(s["intent_counts"], s["visit"]):
        heads = counts[head]
        total = sum(heads.values())
        if total <= 0:
            continue
        for intent, cnt in heads.items():
            target[intent] = target.get(intent, 0.0) + float(visit) * cnt / total
    return target


def _sample_policy_loss(out: dict, b: int, s: dict, t_class: float, t_pos: float,
                        num_heads: int) -> torch.Tensor:
    """Decomposed CE for one sample (all heads): class CE + position CE."""
    cm = torch.from_numpy(s["class_mask"]).bool()
    pm = torch.from_numpy(s["position_mask"]).bool()
    legal = torch.where(cm)[0]
    legal_list = legal.tolist()

    total = torch.tensor(0.0)
    for h in range(num_heads):
        target = _marginalize(s, h)
        if not target:
            continue
        head_logits = out[f"head{h + 1}_logits"][b]
        z = (head_logits - head_logits.mean()) / (head_logits.std() + 1e-8)
        logits = z / t_class
        lse = torch.logsumexp(logits[legal], dim=0)

        target_class: dict = {}
        target_pos: dict = {}
        for intent, mass in target.items():
            c, x, y = intent
            target_class[c] = target_class.get(c, 0.0) + mass
            if x >= 0:
                target_pos.setdefault(c, {})[(x, y)] = target_pos.get(c, {}).get((x, y), 0.0) + mass

        # class CE over the class marginal
        for c, m in target_class.items():
            if c in legal_list:
                idx = legal_list.index(c)
                total = total + m * -(logits[legal[idx]] - lse)

        # per-class weighted position CE
        for c, pos_mass in target_pos.items():
            if c not in legal_list:
                continue
            total_m = sum(pos_mass.values())
            if total_m <= 0:
                continue
            xs, ys = torch.where(pm[c])
            if len(xs) == 0:
                continue
            am_c = out["action_map"][b, c] / t_pos
            pos_logits = am_c[xs, ys]
            pos_lse = torch.logsumexp(pos_logits, dim=0)
            pos_map = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist()))}
            for (x, y), m in pos_mass.items():
                if (x, y) in pos_map:
                    idx = pos_map[(x, y)]
                    total = total + target_class[c] * (m / total_m) * -(pos_logits[idx] - pos_lse)
    return total


def compute_loss(model, policy_batch: list[dict], value_batch: list[dict], *,
                 t_class: float, t_pos: float, lambda_value: float,
                 lambda_anchor: float) -> tuple:
    """Joint loss over two mini-batches: policy+anchor on policy_batch,
    value MSE on value_batch (both heads optimized together every step)."""
    boards = torch.stack([torch.from_numpy(s["board"]).float() for s in policy_batch])
    stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in policy_batch])
    out = model(boards, stats)

    policy_loss = sum(
        _sample_policy_loss(out, b, s, t_class, t_pos, model.num_heads)
        for b, s in enumerate(policy_batch)
    ) / len(policy_batch)

    am_tgt = torch.stack([torch.from_numpy(s["recorded_action_map"]).float() for s in policy_batch])
    hl_tgt = torch.stack([torch.from_numpy(s["recorded_head_logits"]).float() for s in policy_batch])
    anchor_loss = F.mse_loss(out["action_map"], am_tgt) + F.mse_loss(
        torch.stack([out[f"head{i + 1}_logits"] for i in range(model.num_heads)], dim=1),
        hl_tgt,
    )

    v_boards = torch.stack([torch.from_numpy(s["board"]).float() for s in value_batch])
    v_stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in value_batch])
    v_out = model(v_boards, v_stats)
    v_tgt = torch.as_tensor([s["value_label"] for s in value_batch], dtype=torch.float32)
    value_loss = F.mse_loss(v_out["value"].squeeze(-1), v_tgt)

    loss = policy_loss + lambda_value * value_loss + lambda_anchor * anchor_loss
    return loss, policy_loss, value_loss, anchor_loss


def train(model, policy_samples: list[dict], value_samples: list[dict], *,
          epochs: int = 5, batch_size: int = 32, lr: float = 1e-3,
          t_class: float = 0.5, t_pos: float = 0.3,
          lambda_value: float = 1.0, lambda_anchor: float = 1.0,
          seed: int = 0, checkpoint: str = "") -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    n = len(policy_samples)
    nv = len(value_samples)
    for epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        v_order = list(range(nv))
        rng.shuffle(v_order)
        v_pos = 0
        tl = tp = tv = ta = 0.0
        n_steps = 0
        for start in range(0, n, batch_size):
            p_batch = [policy_samples[i] for i in order[start:start + batch_size]]
            v_batch = [value_samples[v_order[(v_pos + j) % nv]] for j in range(len(p_batch))]
            v_pos += len(p_batch)
            loss, p, v, a = compute_loss(
                model, p_batch, v_batch,
                t_class=t_class, t_pos=t_pos,
                lambda_value=lambda_value, lambda_anchor=lambda_anchor,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tl += loss.detach().item(); tp += p.detach().item()
            tv += v.detach().item(); ta += a.detach().item()
            n_steps += 1
        print(f"  epoch {epoch}: loss={tl/max(n_steps,1):.4f} "
              f"policy={tp/max(n_steps,1):.4f} value={tv/max(n_steps,1):.4f} "
              f"anchor={ta/max(n_steps,1):.4f}  (policy pool {n}, value pool {nv})",
              flush=True)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_heads": model.num_heads,
                "no_bn": model.no_bn,
                "latent_dim": model.LATENT_DIM,
                "num_resblocks": model.num_resblocks,
                "completed_batches": 1,
            },
            checkpoint,
        )
        print(f"[train] saved -> {checkpoint}", flush=True)
    return {"policy": n, "value": nv, "epochs": epochs, "steps": n_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-MCTS AlphaZero training (T2, round 2)")
    parser.add_argument("--init", required=True, help="initial checkpoint (gen0120_warm)")
    parser.add_argument("--data-dir", required=True, help="accumulated self-play data dir (value pool)")
    parser.add_argument("--policy-dir", type=str, default=None,
                        help="current-batch data dir (policy pool); default = --data-dir")
    parser.add_argument("--checkpoint", type=str, default="training_history/az_intent/az_az0.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=0.3)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=20.0,
                        help="time constant for exp-weighted future hp-diff labels (view distance)")
    parser.add_argument("--label-scale", type=float, default=1.0,
                        help="amplify value labels so value-head output magnitude "
                             "matches the terminal scale the search expects (~6)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from my_ai.az_intent.az_selfplay import load_model_from_ckpt

    torch.manual_seed(args.seed)
    model = load_model_from_ckpt(args.init)
    model.train()

    policy_dir = args.policy_dir or args.data_dir
    policy_paths = sorted(Path(policy_dir).rglob("az_selfplay_seed*.pkl"))
    value_paths = sorted(Path(args.data_dir).rglob("az_selfplay_seed*.pkl"))
    print(f"[train] policy pool: {len(policy_paths)} files from {policy_dir}", flush=True)
    print(f"[train] value pool:  {len(value_paths)} files from {args.data_dir}", flush=True)
    policy_samples = load_samples(policy_paths)

    value_samples: list[dict] = []
    for path in value_paths:
        with open(path, "rb") as f:
            game = pickle.load(f)["samples"]
        add_weighted_labels(game, tau=args.tau, label_scale=args.label_scale)
        value_samples.extend(game)
    labels = np.asarray([s["value_label"] for s in value_samples])
    print(f"[train] policy {len(policy_samples)} samples, value {len(value_samples)} samples; "
          f"label mean={labels.mean():+.3f} std={labels.std():.3f} "
          f"range=[{labels.min():.3f},{labels.max():.3f}] tau={args.tau} "
          f"label_scale={args.label_scale}", flush=True)

    metrics = train(
        model, policy_samples, value_samples,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        t_class=args.t_class, t_pos=args.t_pos,
        lambda_value=args.lambda_value, lambda_anchor=args.lambda_anchor,
        seed=args.seed, checkpoint=args.checkpoint,
    )
    print(f"[train] done {metrics}", flush=True)


if __name__ == "__main__":
    main()
