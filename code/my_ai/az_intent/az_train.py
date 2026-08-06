"""Bundle-MCTS AlphaZero training (T2) — docs/az_bundle_alpha_zero_training.md.

Loads self-play samples collected by az_selfplay.py and trains:
  - policy loss: bundle visit distribution marginalized to per-head intent
    targets (count-weighted, §2), then a DECOMPOSED CE (§3.1): class CE over
    the class marginal + per-class weighted position CE, using the SAME
    distributions the search samples from (z-score + temperatures).
  - value loss: MSE on the HP-difference target.
  - anchor loss: MSE of the policy outputs vs the recorded self-play outputs
    (trust-region, §3.2).

Usage:
    python code/my_ai/az_intent/az_train.py --init training_history/az_intent/gen0120_warm.pt \
        --data-dir training_history/az_intent/az_selfplay_data \
        --checkpoint training_history/az_intent/az_az0.pt
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


def load_samples(pkl_paths: list[Path]) -> list[dict]:
    """Load self-play pickles into a flat list of samples."""
    samples: list[dict] = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        samples.extend(d["samples"])
    return samples


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


def compute_loss(model, batch: list[dict], *, t_class: float, t_pos: float,
                 lambda_value: float, lambda_anchor: float) -> tuple:
    boards = torch.stack([torch.from_numpy(s["board"]).float() for s in batch])
    stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in batch])
    out = model(boards, stats)

    policy_loss = sum(
        _sample_policy_loss(out, b, s, t_class, t_pos, model.num_heads)
        for b, s in enumerate(batch)
    ) / len(batch)

    v_tgt = torch.as_tensor([s["value_target"] for s in batch], dtype=torch.float32)
    value_loss = F.mse_loss(out["value"].squeeze(-1), v_tgt)

    am_tgt = torch.stack([torch.from_numpy(s["recorded_action_map"]).float() for s in batch])
    hl_tgt = torch.stack([torch.from_numpy(s["recorded_head_logits"]).float() for s in batch])
    anchor_loss = F.mse_loss(out["action_map"], am_tgt) + F.mse_loss(
        torch.stack([out[f"head{i + 1}_logits"] for i in range(model.num_heads)], dim=1),
        hl_tgt,
    )

    loss = policy_loss + lambda_value * value_loss + lambda_anchor * anchor_loss
    return loss, policy_loss, value_loss, anchor_loss


def train(model, samples: list[dict], *, epochs: int = 5, batch_size: int = 32,
          lr: float = 1e-3, t_class: float = 0.5, t_pos: float = 0.3,
          lambda_value: float = 1.0, lambda_anchor: float = 1.0,
          seed: int = 0, checkpoint: str = "") -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    n = len(samples)
    for epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        tl = tp = tv = ta = 0.0
        n_steps = 0
        for start in range(0, n, batch_size):
            batch = [samples[i] for i in order[start:start + batch_size]]
            loss, p, v, a = compute_loss(
                model, batch, t_class=t_class, t_pos=t_pos,
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
              f"anchor={ta/max(n_steps,1):.4f}", flush=True)
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
    return {"samples": n, "epochs": epochs, "steps": n_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-MCTS AlphaZero training (T2)")
    parser.add_argument("--init", required=True, help="initial checkpoint (gen0120_warm)")
    parser.add_argument("--data-dir", required=True, help="self-play data dir (pickles)")
    parser.add_argument("--checkpoint", type=str, default="training_history/az_intent/az_az0.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=0.3)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from my_ai.az_intent.az_selfplay import load_model_from_ckpt

    torch.manual_seed(args.seed)
    model = load_model_from_ckpt(args.init)
    model.train()

    pkl_paths = sorted(Path(args.data_dir).glob("az_selfplay_seed*.pkl"))
    print(f"[train] loading {len(pkl_paths)} self-play files...", flush=True)
    samples = load_samples(pkl_paths)
    print(f"[train] {len(samples)} samples", flush=True)

    metrics = train(
        model, samples,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        t_class=args.t_class, t_pos=args.t_pos,
        lambda_value=args.lambda_value, lambda_anchor=args.lambda_anchor,
        seed=args.seed, checkpoint=args.checkpoint,
    )
    print(f"[train] done {metrics}", flush=True)


if __name__ == "__main__":
    main()
