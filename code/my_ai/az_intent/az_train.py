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
                        label_scale: float = 1.0,
                        label_mode: str = "rel",
                        mix_alpha: float = 0.5,
                        label_weight: str = "geo") -> None:
    """In-place: set ``value_label`` = exp-weighted future HP-diff (player view, /HP_SCALE).

    Per-frame instantaneous HP diff d_t is taken from stats[1] (``hp_delta`` =
    bases[player].hp - bases[enemy].hp, player perspective).  It is numerically
    identical to the original ``(stats[24]-stats[25])*50`` form — see below.
    NOTE (2026-09-09 correction): the 2026-09-07 commit called the original code
    an "index bug" and claimed the labels had always been garbage.  That was a
    MISDIAGNOSIS.  stats = 22 summarize features + 20 extras, so extras[2]/[3]
    land exactly at indices 24/25 (bases[player].hp/50, bases[enemy].hp/50) — and
    the original player==1 branch flipped the index order, so d_t was already a
    unified P0 view.  Old and new labels are identical (corr 0.999999); the net
    effect of the "fix" was zero.  The value-head problem is the DATA (see
    results doc §26), not the label formula.
    Label forms (docs/value_label_future_weighted.md):
      - "rel" (default): label_t = weighted_future_avg - d_t  (0-centered, predicts
        the future advantage CHANGE from here).  Design-recommended, but the search
        reads value as an absolute evaluation — mismatch (see results doc §19).
      - "abs": label_t = weighted_future_avg (absolute weighted future HP-diff;
        matches how the search consumes value as an absolute advantage).
      - "mix": label_t = mix_alpha * d_t + (1 - mix_alpha) * weighted_future_avg
        (current absolute HP-diff blended with the future trend).
    ``label_scale`` amplifies the labels so the trained value head's output
    magnitude matches the terminal-scale values the search expects (else the
    PUCT explore term dominates and search starves).
    ``label_weight`` selects the weight on the future offset k (mirrors
    value_warmup._exp_weighted_labels):
      "geo"  : w_k = gamma^k    — peak at k=0 (current frame), the original form.
      "kgeo" : w_k = k*gamma^k  — w_0 = 0, so the current frame is EXCLUDED and the
               peak moves to k~tau.  Cuts the "copy the current HP diff from
               stats[1]" shortcut (see docs/az_pool_tau_label_plan.md).
    Backward O(n) recurrence.
    """
    gamma = float(np.exp(-1.0 / tau))
    n = len(samples)
    if n == 0:
        return
    # d[t] in a unified P0 view: P0/P1 decision samples of one round are stored
    # interleaved (P0 +hp_delta, P1 -hp_delta for the same physical position), so
    # a per-player-perspective d[t] would cancel in the weighted average.  This
    # matches the original (stats[24]-stats[25])*50 form, which also unified to P0
    # by flipping the index order for player==1.
    d = np.empty(n, dtype=np.float64)
    for t, s in enumerate(samples):
        hp = float(s["stats"][1])  # hp_delta = player.hp - enemy.hp (player view)
        d[t] = hp if s["player"] == 0 else -hp

    def finish(raw: float, d_t: float, p: int) -> float:
        if label_mode == "abs":
            lab = raw
        elif label_mode == "mix":
            lab = mix_alpha * d_t + (1.0 - mix_alpha) * raw
        else:
            lab = raw - d_t
        label = float(np.clip(lab / HP_SCALE, -1.0, 1.0)) * label_scale
        # value must be from the SAMPLE's player perspective (features are
        # player-perspective; the search reads it as the current player's value)
        return label if p == 0 else -label

    if label_weight == "kgeo":
        # B_t = sum k g^k d_{t+k} = g (B_{t+1} + A_{t+1}); WB analogous.
        A = B = WA = WB = 0.0
        for t in range(n - 1, -1, -1):
            B = gamma * (B + A)
            WB = gamma * (WB + WA)
            A = d[t] + gamma * A
            WA = 1.0 + gamma * WA
            raw = (B / WB) if WB > 1e-9 else d[t]  # last frame: fall back to d_t
            samples[t]["value_label"] = finish(raw, d[t], int(samples[t]["player"]))
        return

    suffix = wsum = 0.0
    for t in range(n - 1, -1, -1):
        suffix = d[t] + gamma * suffix
        wsum = 1.0 + gamma * wsum
        samples[t]["value_label"] = finish(suffix / wsum, d[t], int(samples[t]["player"]))


def _marginalize(s: dict, head: int) -> dict:
    """Count-weighted marginalization: bundle visit -> per-intent target mass.

    ``intent_counts`` is ``None`` for decisions the collector SKIPPED
    (``--skip-hold-search``: no search was run, so there is no visit distribution to
    marginalize).  Such samples carry no policy target — they only keep the anchor
    term — so return an empty target instead of crashing.  (Previously a None here
    would raise on the pkl path, i.e. skip-hold collection was silently unusable
    with any policy training.)
    """
    ic = s.get("intent_counts")
    if not ic:
        return {}
    target: dict = {}
    for counts, visit in zip(ic, s["visit"]):
        heads = counts[head]
        total = sum(heads.values())
        if total <= 0:
            continue
        for intent, cnt in heads.items():
            target[intent] = target.get(intent, 0.0) + float(visit) * cnt / total
    return target


def _sample_policy_loss(out: dict, b: int, s: dict, t_class: float, t_pos: float,
                        num_heads: int, pos_single: bool = False,
                        lambda_class_ce: float = 1.0) -> torch.Tensor:
    """Decomposed CE for one sample (all heads): class CE + position CE.

    ``pos_single`` collapses each class's position target to its argmax
    position (single-point target).  Multi-position targets in a sample are
    often spread across non-adjacent cells (esp. LIGHTNING) which the smooth
    action_map conv cannot match simultaneously — their gradients cancel.
    Collapsing to one position removes the conflict; loss/divergence tests
    show action_map then develops position discrimination much faster.

    ``lambda_class_ce=0`` drops the class term (used by the position-only phase,
    docs/az_three_net_split_plan.md §5): on pos-only data the class target is a
    one-hot on the head's OWN argmax, so its gradient only sharpens whatever the head
    already prefers — i.e. it would deepen the lightning collapse.  With it at 0 the
    head logits are never touched, so ``out`` only needs ``action_map``.
    """
    cm = torch.from_numpy(s["class_mask"]).bool()
    pm = torch.from_numpy(s["position_mask"]).bool()
    legal = torch.where(cm)[0]
    legal_list = legal.tolist()

    total = torch.tensor(0.0)
    for h in range(num_heads):
        target = _marginalize(s, h)
        if not target:
            continue
        target_class: dict = {}
        target_pos: dict = {}
        for intent, mass in target.items():
            c, x, y = intent
            target_class[c] = target_class.get(c, 0.0) + mass
            if x >= 0:
                target_pos.setdefault(c, {})[(x, y)] = target_pos.get(c, {}).get((x, y), 0.0) + mass

        # class CE over the class marginal (skipped entirely when lambda_class_ce == 0)
        if lambda_class_ce > 0:
            head_logits = out[f"head{h + 1}_logits"][b]
            z = (head_logits - head_logits.mean()) / (head_logits.std() + 1e-8)
            logits = z / t_class
            lse = torch.logsumexp(logits[legal], dim=0)
            for c, m in target_class.items():
                if c in legal_list:
                    idx = legal_list.index(c)
                    total = total + lambda_class_ce * m * -(logits[legal[idx]] - lse)

        # per-class weighted position CE
        for c, pos_mass in target_pos.items():
            if c not in legal_list:
                continue
            if pos_single:
                pos_mass = {max(pos_mass.items(), key=lambda kv: kv[1])[0]:
                            max(pos_mass.values())}
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
                 lambda_anchor: float, device: str = "cpu",
                 pos_single: bool = False) -> tuple:
    """Joint loss over two mini-batches: policy+anchor on policy_batch,
    value MSE on value_batch (both heads optimized together every step)."""
    boards = torch.stack([torch.from_numpy(s["board"]).float() for s in policy_batch]).to(device)
    stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in policy_batch]).to(device)
    out = model(boards, stats)

    policy_loss = sum(
        _sample_policy_loss(out, b, s, t_class, t_pos, model.num_heads, pos_single)
        for b, s in enumerate(policy_batch)
    ) / len(policy_batch)

    am_tgt = torch.stack([torch.from_numpy(s["recorded_action_map"]).float() for s in policy_batch]).to(device)
    hl_tgt = torch.stack([torch.from_numpy(s["recorded_head_logits"]).float() for s in policy_batch]).to(device)
    anchor_loss = F.mse_loss(out["action_map"], am_tgt) + F.mse_loss(
        torch.stack([out[f"head{i + 1}_logits"] for i in range(model.num_heads)], dim=1),
        hl_tgt,
    )

    v_boards = torch.stack([torch.from_numpy(s["board"]).float() for s in value_batch]).to(device)
    v_stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in value_batch]).to(device)
    v_out = model(v_boards, v_stats)
    v_tgt = torch.as_tensor([s["value_label"] for s in value_batch], dtype=torch.float32).to(device)
    value_loss = F.mse_loss(v_out["value"].squeeze(-1), v_tgt)

    loss = policy_loss + lambda_value * value_loss + lambda_anchor * anchor_loss
    return loss, policy_loss, value_loss, anchor_loss


def train(model, policy_samples: list[dict], value_samples: list[dict], *,
          epochs: int = 5, batch_size: int = 32, lr: float = 1e-3,
          t_class: float = 0.5, t_pos: float = 0.3,
          lambda_value: float = 1.0, lambda_anchor: float = 1.0,
          seed: int = 0, checkpoint: str = "", device: str = "cpu",
          pos_single: bool = False) -> dict:
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
                device=device, pos_single=pos_single,
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
                "gn": getattr(model, "gn", False),
                "gn_groups": getattr(model, "gn_groups", 8),
                "value_pool": getattr(model, "value_pool", "gap"),
                "latent_dim": model.LATENT_DIM,
                "num_resblocks": model.num_resblocks,
                "completed_batches": 1,
            },
            checkpoint,
        )
        print(f"[train] saved -> {checkpoint}", flush=True)
    return {"policy": n, "value": nv, "epochs": epochs, "steps": n_steps}


def _policy_step(policy_model, p_batch, opt_p, t_class, t_pos, lambda_anchor, device,
                 pos_single: bool = False):
    """One policy step: decomposed CE + anchor. Returns (loss_p, policy_loss, anchor_loss)."""
    boards = torch.stack([torch.from_numpy(s["board"]).float() for s in p_batch]).to(device)
    stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in p_batch]).to(device)
    p_out = policy_model(boards, stats)
    policy_loss = sum(
        _sample_policy_loss(p_out, b, s, t_class, t_pos, policy_model.num_heads, pos_single)
        for b, s in enumerate(p_batch)
    ) / len(p_batch)
    am_tgt = torch.stack([torch.from_numpy(s["recorded_action_map"]).float() for s in p_batch]).to(device)
    hl_tgt = torch.stack([torch.from_numpy(s["recorded_head_logits"]).float() for s in p_batch]).to(device)
    anchor_loss = F.mse_loss(p_out["action_map"], am_tgt) + F.mse_loss(
        torch.stack([p_out[f"head{i + 1}_logits"] for i in range(policy_model.num_heads)], dim=1),
        hl_tgt,
    )
    loss_p = policy_loss + lambda_anchor * anchor_loss
    opt_p.zero_grad()
    loss_p.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 5.0)
    opt_p.step()
    return loss_p, policy_loss, anchor_loss


def _value_step(value_model, v_batch, opt_v, device,
                anchor_model=None, lambda_anchor_v: float = 0.0):
    """One value step: MSE on labels + optional backbone anchor.

    ``anchor_model`` (frozen copy of the value net at training start) anchors the
    value net's shared backbone so value-only training doesn't drift/collapse the
    features — mirrors value_warmup's shared-backbone anchor that produced a good
    value head while az's free value-net backbone drifted. Returns value_loss.
    """
    v_boards = torch.stack([torch.from_numpy(s["board"]).float() for s in v_batch]).to(device)
    v_stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in v_batch]).to(device)
    v_out = value_model(v_boards, v_stats)
    v_tgt = torch.as_tensor([s["value_label"] for s in v_batch], dtype=torch.float32).to(device)
    value_loss = F.mse_loss(v_out["value"].squeeze(-1), v_tgt)
    if anchor_model is not None and lambda_anchor_v > 0:
        with torch.no_grad():
            ref = anchor_model(v_boards, v_stats)["state_emb"]
        value_loss = value_loss + lambda_anchor_v * F.mse_loss(v_out["state_emb"], ref)
    opt_v.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_model.parameters(), 5.0)
    opt_v.step()
    return value_loss


def train_split(policy_model, value_model, policy_samples: list[dict],
                value_samples: list[dict], *, epochs: int = 5, batch_size: int = 32,
                lr: float = 1e-3, t_class: float = 0.5, t_pos: float = 0.3,
                lambda_anchor: float = 1.0, seed: int = 0,
                checkpoint: str = "", device: str = "cpu",
                value_only: bool = False, value_passes: int = 1,
                policy_only: bool = False,
                pos_single: bool = False,
                value_anchor: float = 0.0) -> dict:
    """Independent training: policy net (CE + anchor) and value net (weighted MSE).

    Two separate networks, two optimizers — no shared-backbone coupling.
    Saves a combined checkpoint with both state_dicts.

    Per epoch the policy gets 1 pass over the POLICY pool while the value gets
    ``value_passes`` passes over the VALUE pool (decoupled — the value pool is
    much larger, so tying its steps to the policy pool under-trains it).
    ``value_only`` freezes the policy net (no policy/anchor step).
    """
    opt_p = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=1e-4)
    opt_v = torch.optim.AdamW([p for p in value_model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    n = len(policy_samples)
    nv = len(value_samples)

    # value-only backbone anchor: freeze the value net at its starting weights so
    # value-only training can't drift/collapse the shared feature backbone.
    v_anchor_model = None
    if value_only and value_anchor > 0:
        import copy as _copy
        v_anchor_model = _copy.deepcopy(value_model)
        v_anchor_model.eval()
        for p in v_anchor_model.parameters():
            p.requires_grad_(False)
        v_anchor_model.to(device)

    def _value_pass(acc_tl, acc_tv, acc_vsteps, acc_steps):
        v_order = list(range(nv))
        rng.shuffle(v_order)
        for start in range(0, nv, batch_size):
            v_batch = [value_samples[v_order[(start + j) % nv]] for j in range(min(batch_size, nv - start))]
            value_loss = _value_step(value_model, v_batch, opt_v, device,
                                     anchor_model=v_anchor_model,
                                     lambda_anchor_v=value_anchor)
            acc_tl += value_loss.detach().item()
            acc_tv += value_loss.detach().item()
            acc_vsteps += 1
            acc_steps += 1
        return acc_tl, acc_tv, acc_vsteps, acc_steps

    for epoch in range(epochs):
        tl = tp = tv = ta = 0.0
        n_p = n_v = n_steps = 0
        if value_only:
            for _ in range(max(value_passes, 1)):
                tl, tv, n_v, n_steps = _value_pass(tl, tv, n_v, n_steps)
        elif policy_only:
            # policy only: 1 pass over the policy pool, value net untouched
            order = list(range(n))
            rng.shuffle(order)
            for start in range(0, n, batch_size):
                p_batch = [policy_samples[i] for i in order[start:start + batch_size]]
                loss_p, policy_loss, anchor_loss = _policy_step(
                    policy_model, p_batch, opt_p, t_class, t_pos, lambda_anchor, device,
                    pos_single=pos_single)
                tl += loss_p.detach().item()
                tp += policy_loss.detach().item()
                ta += anchor_loss.detach().item()
                n_p += 1
                n_steps += 1
        else:
            # policy: 1 pass over the policy pool (with one interleaved value batch each)
            order = list(range(n))
            rng.shuffle(order)
            v_order = list(range(nv))
            rng.shuffle(v_order)
            v_pos = 0
            for start in range(0, n, batch_size):
                p_batch = [policy_samples[i] for i in order[start:start + batch_size]]
                v_batch = [value_samples[v_order[(v_pos + j) % nv]] for j in range(len(p_batch))]
                v_pos += len(p_batch)
                loss_p, policy_loss, anchor_loss = _policy_step(
                    policy_model, p_batch, opt_p, t_class, t_pos, lambda_anchor, device,
                    pos_single=pos_single)
                value_loss = _value_step(value_model, v_batch, opt_v, device)
                tl += loss_p.detach().item() + value_loss.detach().item()
                tp += policy_loss.detach().item()
                tv += value_loss.detach().item()
                ta += anchor_loss.detach().item()
                n_p += 1
                n_v += 1
                n_steps += 1
            # value: additional passes over the full value pool (decoupled)
            for _ in range(1, value_passes):
                tl, tv, n_v, n_steps = _value_pass(tl, tv, n_v, n_steps)
        # loss 用总步数平均；policy/anchor 只用 policy 步数平均，value 只用 value 步数平均
        print(f"  epoch {epoch}: loss={tl/max(n_steps,1):.4f} "
              f"policy={tp/max(n_p,1):.4f} value={tv/max(n_v,1):.4f} "
              f"anchor={ta/max(n_p,1):.4f}  (policy pool {n}, value pool {nv}, "
              f"value_passes={value_passes})"
              + (" [value_only]" if value_only else ""), flush=True)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": policy_model.state_dict(),
                "value_state": value_model.state_dict(),
                "num_heads": policy_model.num_heads,
                "no_bn": policy_model.no_bn,
                "gn": getattr(policy_model, "gn", False),
                "gn_groups": getattr(policy_model, "gn_groups", 8),
                "value_pool": getattr(policy_model, "value_pool", "gap"),
                "latent_dim": policy_model.LATENT_DIM,
                "num_resblocks": policy_model.num_resblocks,
                "completed_batches": 1,
            },
            checkpoint,
        )
        print(f"[train] saved -> {checkpoint}", flush=True)
    return {"policy": n, "value": nv, "epochs": epochs, "steps": n_steps}


def save_three_net(path: str, class_model, pos_model, value_model, meta: dict,
                   completed_batches: int = 1) -> None:
    """Write a 3-net checkpoint (class/pos/value) — deliberately NO ``model_state``.

    Writing a stale/untrained ``model_state`` would let the old 1-/2-net loaders read
    the wrong action_map silently; see docs/az_three_net_split_plan.md §3.
    """
    out = {k: v for k, v in meta.items()
           if k not in ("model_state", "value_state", "class_state", "pos_state")}
    out.update({
        "class_state": class_model.state_dict(),
        "pos_state": pos_model.state_dict(),
        "value_state": value_model.state_dict(),
        "three_net": True,
        "num_heads": class_model.num_heads,
        "no_bn": class_model.no_bn,
        "gn": getattr(class_model, "gn", False),
        "gn_groups": getattr(class_model, "gn_groups", 8),
        "latent_dim": class_model.LATENT_DIM,
        "num_resblocks": class_model.num_resblocks,
        "completed_batches": completed_batches,
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"[pos-only] saved -> {path}", flush=True)


def train_pos_only(class_model, pos_model, value_model, samples: list[dict], *,
                   epochs: int, batch_size: int, lr: float, t_class: float, t_pos: float,
                   lambda_anchor: float, pos_single: bool, device: str,
                   out_path: str, meta: dict, anchor_scope: str = "contrib") -> dict:
    """位置-only 阶段：只训位置网的 action_map（docs/az_three_net_split_plan.md §5）。

        L = 位置 CE（lambda_class_ce=0，不碰 head_logits） + λ_anchor · MSE(action_map, 记录值)

    - **类网 / 价值网冻结**，连前向都不做（位置 CE 只需要 action_map）⇒ 类网输出逐位不变；
    - **HOLD 回合在位置 CE 上严格为 0**（实测：HOLD 样本 `requires_grad=False`）——它的目标里
      没有位置。所以位置网只被"出招回合"（约 2.8%）的 CE 训练。

    ``anchor_scope``：
      - ``contrib``（默认）: 只对**有位置目标**的样本施加 anchor ⇒ HOLD 回合 CE 与 anchor
        都跳过，λ 的语义就是"每个出招回合允许改多少"（纯信任域，作用在 CE 真正发力处）；
      - ``all``: 全样本都锚 ⇒ 额外限制"2.8% 的拟合结果泛化泄漏到其余 97% 状态空间"的量。

    归一化：CE 与 anchor **都除以"本 batch 贡献样本数"**（不是 ``len(batch)``）。因为
    batch 里平均只有 ~0.9 个样本带目标，按 ``len(batch)`` 平均会把 CE 稀释 ~36×，
    使 λ 失去可解释性。日志会打出有效 anchor/CE 比。
    """
    for m in (class_model, value_model):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
    for p in pos_model.parameters():
        p.requires_grad = True
    pos_model.train().to(device)
    opt = torch.optim.Adam([p for p in pos_model.parameters() if p.requires_grad], lr=lr)

    # 预计算"该样本是否有位置目标"（只算一次）
    n_all = len(samples)
    contrib = []
    for s in samples:
        if any(x >= 0 for h in range(pos_model.num_heads)
               for (_c, x, _y) in _marginalize(s, h)):
            contrib.append(s)
    print(f"[pos-only] {n_all} 样本中带位置目标的 {len(contrib)} 个 "
          f"({len(contrib) / max(n_all, 1):.1%})", flush=True)
    if anchor_scope == "contrib":
        # HOLD 回合在 CE 上严格为 0；anchor 也只锚有目标的样本 ⇒ 它们对训练完全无用，
        # 直接从数据集里去掉（等价于"只训出招回合"，也让 CE 的归一化天然是 per-贡献样本）。
        samples = contrib
        print(f"[pos-only] anchor_scope=contrib -> 只训这 {len(samples)} 个出招回合"
              f"（HOLD 回合 CE/anchor 都跳过）", flush=True)
    else:
        print("[pos-only] anchor_scope=all -> 全样本参与（HOLD 回合只贡献 anchor，"
              "用于限制泛化泄漏）", flush=True)
    if not samples:
        raise SystemExit("[pos-only] 没有带位置目标的样本，无从训练")

    n_steps = 0
    for epoch in range(epochs):
        random.shuffle(samples)
        tot = tc = ta = 0.0
        ns = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            boards = torch.stack([torch.from_numpy(s["board"]).float() for s in batch]).to(device)
            stats = torch.stack([torch.from_numpy(s["stats"]).float() for s in batch]).to(device)
            out = pos_model(boards, stats)                      # 只需位置网前向
            ce = sum(_sample_policy_loss({"action_map": out["action_map"]}, b, s,
                                         t_class, t_pos, pos_model.num_heads,
                                         pos_single, lambda_class_ce=0.0)
                     for b, s in enumerate(batch)) / len(batch)
            am_tgt = torch.stack([torch.from_numpy(s["recorded_action_map"]).float()
                                  for s in batch]).to(device)
            anc = F.mse_loss(out["action_map"], am_tgt)
            loss = ce + lambda_anchor * anc
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pos_model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); tc += ce.item(); ta += anc.item(); ns += 1; n_steps += 1
        ce_m = tc / max(ns, 1)
        an_m = ta / max(ns, 1)
        an_w = lambda_anchor * an_m
        ratio = (an_w / ce_m) if ce_m > 0 else float("inf")
        print(f"[pos-only] epoch {epoch+1}/{epochs}: loss={tot/max(ns,1):.4f} "
              f"pos_ce={ce_m:.4f} anchor(加权)={an_w:.6f} "
              f"[有效 anchor/CE = {ratio:.4f}]  "
              f"({ns} 步/batch {batch_size}; λ={lambda_anchor}, "
              f"anchor_scope={anchor_scope}, pos_single={pos_single})", flush=True)
        save_three_net(out_path, class_model, pos_model, value_model, meta)
    return {"epochs": epochs, "steps": n_steps, "n_samples_used": len(samples),
            "n_samples_all": n_all}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-MCTS AlphaZero training (T2, round 2)")
    parser.add_argument("--init", required=True, help="initial checkpoint (gen0120_warm)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="accumulated self-play data dir (value pool); "
                             "位置-only 模式不需要")
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
    parser.add_argument("--label-mode", type=str, default="rel",
                        choices=["rel", "abs", "mix", "terminal"],
                        help="value label form: 'rel' = weighted_future_avg - d_t "
                             "(0-centered change, default), 'abs' = weighted_future_avg "
                             "(absolute advantage), 'mix' = blend of current d_t and "
                             "weighted_future_avg (see --label-mix-alpha)")
    parser.add_argument("--label-mix-alpha", type=float, default=0.5,
                        help="for label_mode='mix': weight of current HP-diff d_t "
                             "(1-alpha weights the future average)")
    parser.add_argument("--label-weight", type=str, default="geo",
                        choices=["geo", "kgeo"],
                        help="future-offset weight: 'geo' = gamma^k (original) or "
                             "'kgeo' = k*gamma^k (excludes the current frame, cuts the "
                             "copy-stats[1] shortcut; mirrors value_warmup)")
    parser.add_argument("--split", action="store_true",
                        help="train policy and value as two independent networks "
                             "(docs/az_split_policy_value_plan.md)")
    parser.add_argument("--max-value-batches", type=int, default=None,
                        help="value pool keeps only the most recent N batch dirs "
                             "(keeps training memory bounded over long runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' (cuda if available), 'cuda', or 'cpu'")
    parser.add_argument("--policy-only", action="store_true",
                        help="train only the policy net; the value net is left untouched "
                             "(freeze a good value head and let the policy learn from search)")
    parser.add_argument("--value-only", action="store_true",
                        help="freeze the policy net, train only the value head "
                             "(e.g. re-calibrate on a different tau)")
    parser.add_argument("--value-anchor", type=float, default=0.0,
                        help="value-only: anchor the value net's backbone (state_emb) "
                             "to its frozen starting weights with this MSE weight — "
                             "prevents value-only training from drifting/collapsing "
                             "the features (mirrors value_warmup's shared-backbone anchor)")
    parser.add_argument("--value-passes", type=int, default=1,
                        help="how many full passes over the VALUE pool per epoch "
                             "(decoupled from the policy pool size; value pool is "
                             "much larger so default 1 under-trains it — use 3)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="freeze initial_conv + resblocks on both nets; only the heads "
                             "train (removes the under-constrained backbone drift that makes "
                             "joint training land in arbitrary solutions)")
    parser.add_argument("--pos-single", action="store_true",
                        help="collapse each class's position target to its argmax "
                             "position (single-point); multi-position targets are often "
                             "spread across non-adjacent cells whose gradients cancel, "
                             "making position CE unlearnable at practical step counts")
    parser.add_argument("--pos-only-net", action="store_true",
                        help="位置-only 阶段（三网 ckpt）：只训位置网，类网与价值网冻结且"
                             "不参与前向；损失 = 位置 CE + λ_anchor·MSE(action_map, 记录值)。"
                             "见 docs/az_three_net_split_plan.md §5")
    parser.add_argument("--anchor-scope", type=str, default="contrib",
                        choices=["contrib", "all"],
                        help="位置-only：anchor 作用范围。contrib=只锚有位置目标的样本"
                             "（HOLD 回合 CE 与 anchor 都跳过，λ 语义为'每个出招回合允许改多少'）；"
                             "all=全样本都锚（额外限制泛化泄漏）")
    parser.add_argument("--lambda-class-ce", type=float, default=1.0,
                        help="class CE 的权重；0 = 完全关掉（位置-only 阶段必须为 0，"
                             "否则会加重类头的闪电坍缩）")
    args = parser.parse_args()

    from my_ai.az_intent.az_selfplay import (load_model_from_ckpt, load_split_models,
                                             load_three_models)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[train] device={device}", flush=True)

    torch.manual_seed(args.seed)
    three_meta: dict = {}
    if args.pos_only_net:
        _src = torch.load(args.init, map_location="cpu", weights_only=False)
        if "pos_state" not in _src:
            raise SystemExit("--pos-only-net 需要三网 ckpt（含 pos_state）；"
                             "先用 make_three_net_ckpt.py 拆一份")
        class_model, pos_model, value_model = load_three_models(args.init)
        three_meta = {k: v for k, v in _src.items()
                      if k not in ("model_state", "value_state", "class_state", "pos_state")}
        print(f"[train] pos-only 模式：三网 ckpt，类网/价值网冻结", flush=True)
    elif args.split:
        policy_model, value_model = load_split_models(args.init)
        policy_model.train().to(device)
        value_model.train().to(device)
    else:
        model = load_model_from_ckpt(args.init)
        model.train().to(device)

    if args.freeze_backbone and not args.pos_only_net:
        nets = ([("policy", policy_model), ("value", value_model)] if args.split
                else [("model", model)])
        for tag, m in nets:
            n_frozen = 0
            for name, p in m.named_parameters():
                if name.startswith("initial_conv") or name.startswith("resblocks"):
                    p.requires_grad = False
                    n_frozen += p.numel()
            print(f"[train] {tag} backbone frozen: {n_frozen:,} params", flush=True)

    if args.policy_dir is None and args.data_dir is None:
        raise SystemExit("至少给一个 --policy-dir 或 --data-dir")
    policy_dir = args.policy_dir or args.data_dir
    policy_paths = sorted(Path(policy_dir).rglob("az_selfplay_seed*.pkl"))

    if args.pos_only_net:
        # 位置-only：不需要价值池（价值网冻结），直接进位置训练
        if args.lambda_class_ce != 0.0:
            print(f"[train] 注意：pos-only 模式强制 lambda_class_ce=0"
                  f"（传入的 {args.lambda_class_ce} 被忽略）——pos-only 的类目标是"
                  f"该头自己的 argmax，照训会加重闪电坍缩", flush=True)
        policy_samples = load_samples(policy_paths)
        print(f"[train] pos-only: policy pool {len(policy_samples)} samples from "
              f"{policy_dir}", flush=True)
        if not policy_samples:
            raise SystemExit(f"policy pool 为空（{policy_dir} 下没有 az_selfplay_seed*.pkl）")
        metrics = train_pos_only(
            class_model, pos_model, value_model, policy_samples,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            t_class=args.t_class, t_pos=args.t_pos,
            lambda_anchor=args.lambda_anchor, pos_single=args.pos_single,
            device=device, out_path=args.checkpoint, meta=three_meta,
            anchor_scope=args.anchor_scope)
        print(f"[train] pos-only done: {metrics}", flush=True)
        return

    value_paths = sorted(Path(args.data_dir).rglob("az_selfplay_seed*.pkl"))
    if args.max_value_batches:
        from collections import defaultdict
        by_batch: dict = defaultdict(list)
        for p in value_paths:
            by_batch[p.parent.name].append(p)

        def _batch_key(name: str) -> int:
            num = name.replace("batch", "")
            return int(num) if num.isdigit() else -1

        keep = set(sorted(by_batch, key=_batch_key)[-args.max_value_batches:])
        value_paths = sorted(p for bn, ps in by_batch.items() if bn in keep for p in ps)
        print(f"[train] value pool limited to last {args.max_value_batches} batches "
              f"({len(value_paths)} files)", flush=True)
    print(f"[train] policy pool: {len(policy_paths)} files from {policy_dir}", flush=True)
    print(f"[train] value pool:  {len(value_paths)} files from {args.data_dir}", flush=True)
    policy_samples = load_samples(policy_paths)

    value_samples: list[dict] = []
    for path in value_paths:
        with open(path, "rb") as f:
            game = pickle.load(f)["samples"]
        if args.label_mode == "terminal":
            # 用采集时已存的终局 HP 差标签（value_target，value_warmup 风格，
            # 每局所有样本同值、clip 到 ±1）——对齐被验证有效的 value_warmup。
            for s in game:
                if "value_target" not in s:
                    raise SystemExit(f"terminal mode: {path} 无 value_target 字段")
                s["value_label"] = float(s["value_target"])
        else:
            add_weighted_labels(game, tau=args.tau, label_scale=args.label_scale,
                                label_mode=args.label_mode,
                                mix_alpha=args.label_mix_alpha,
                                label_weight=args.label_weight)
        value_samples.extend(game)
    labels = np.asarray([s["value_label"] for s in value_samples])
    print(f"[train] policy {len(policy_samples)} samples, value {len(value_samples)} samples; "
          f"label mean={labels.mean():+.3f} std={labels.std():.3f} "
          f"range=[{labels.min():.3f},{labels.max():.3f}] tau={args.tau} "
          f"label_scale={args.label_scale} label_mode={args.label_mode}", flush=True)

    if args.split:
        metrics = train_split(
            policy_model, value_model, policy_samples, value_samples,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            t_class=args.t_class, t_pos=args.t_pos,
            lambda_anchor=args.lambda_anchor,
            seed=args.seed, checkpoint=args.checkpoint, device=device,
            value_only=args.value_only, value_passes=args.value_passes,
            policy_only=args.policy_only,
            pos_single=args.pos_single,
            value_anchor=args.value_anchor,
        )
    else:
        metrics = train(
            model, policy_samples, value_samples,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            t_class=args.t_class, t_pos=args.t_pos,
            lambda_value=args.lambda_value, lambda_anchor=args.lambda_anchor,
            seed=args.seed, checkpoint=args.checkpoint, device=device,
            pos_single=args.pos_single,
        )
    print(f"[train] done {metrics}", flush=True)


if __name__ == "__main__":
    main()
