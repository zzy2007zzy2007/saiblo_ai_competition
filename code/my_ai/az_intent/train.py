"""P3: torch training for intent-space AlphaZero (方案 B).

Hot-start: load encoder + policy heads from an existing ES checkpoint, RE-INIT
the value head (ES value heads are broken, ~1.5e7 outputs).  Train on
self-play samples: factored-prior CE (class softmax x per-channel position
softmax) + MSE on the saturated HP-difference value target.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def make_net_fn(model, feature_extractor, max_actions: int = 96,
                value_tanh: bool = True):
    """Wrap the torch model into the net_fn interface used by MCTS/self-play.

    ``value`` is passed through tanh so the search Q is always bounded (the
    value head is re-initialized but may output large values early on).
    ``value_tanh=False`` keeps the raw value-head output (experiment: tanh
    saturates large abs-label-trained value heads, flattening discrimination).
    """
    def net_fn(state, player):
        model.eval()  # inference path: no dropout, fixed BN
        obs = feature_extractor.encode_observation(state, player, np.zeros(max_actions))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            out = model(board, stats)
        heads = [out[f"head{i + 1}_logits"].squeeze(0).numpy() for i in range(model.num_heads)]
        v_t = out["value"].squeeze(0)
        value = float(torch.tanh(v_t).item()) if value_tanh else float(v_t.item())
        return {
            "action_map": out["action_map"].squeeze(0).numpy(),
            "head_logits": heads,
            "value": value,
        }
    return net_fn


def make_split_net_fn(policy_model, value_model, feature_extractor, max_actions: int = 96,
                      value_tanh: bool = True):
    """net_fn over two independent networks (policy + value) — docs/az_split_policy_value_plan.md.

    Same interface as make_net_fn so the MCTS is agnostic.  ``value`` is passed
    through tanh (bounded Q), matching the single-model path.
    ``value_tanh=False`` keeps the raw value-head output (experiment).
    """
    def net_fn(state, player):
        policy_model.eval()
        value_model.eval()
        obs = feature_extractor.encode_observation(state, player, np.zeros(max_actions))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            p_out = policy_model(board, stats)
            v_out = value_model(board, stats)
        heads = [p_out[f"head{i + 1}_logits"].squeeze(0).numpy()
                 for i in range(policy_model.num_heads)]
        v_t = v_out["value"].squeeze(0)
        value = float(torch.tanh(v_t).item()) if value_tanh else float(v_t.item())
        return {
            "action_map": p_out["action_map"].squeeze(0).numpy(),
            "head_logits": heads,
            "value": value,
        }
    return net_fn


def infer_model_config(ckpt: dict) -> dict:
    """Infer the model architecture from a checkpoint's state_dict."""
    sd = ckpt.get("model_state")
    cfg: dict = {"num_heads": int(ckpt.get("num_heads", 1)), "latent_dim": 64, "num_resblocks": 6}
    if sd is None:
        cfg["no_bn"] = bool(ckpt.get("no_bn", False))
        cfg["gn"] = bool(ckpt.get("gn", False))
        cfg["gn_groups"] = int(ckpt.get("gn_groups", 8))
        cfg["value_pool"] = str(ckpt.get("value_pool", "gap"))
        return cfg
    # initial_conv = [Conv, <norm>, ReLU]: BN has running stats, GroupNorm has
    # weight/bias only, no_bn has a parameter-free ReLU at index 1.
    has_bn = "initial_conv.1.running_mean" in sd
    has_gn = (not has_bn) and ("initial_conv.1.weight" in sd)
    cfg["gn"] = bool(ckpt.get("gn", has_gn))
    cfg["gn_groups"] = int(ckpt.get("gn_groups", 8))
    cfg["no_bn"] = (not has_bn) and (not has_gn)
    # value_pool cannot be inferred from shapes (128 vs 192 collide) -> metadata.
    cfg["value_pool"] = str(ckpt.get("value_pool", "gap"))
    w = sd.get("initial_conv.0.weight")
    if w is not None:
        cfg["latent_dim"] = int(w.shape[0])
    cfg["num_resblocks"] = sum(1 for k in sd if k.startswith("resblocks.") and ".conv1.weight" in k)
    return cfg


def load_hotstart(model: nn.Module, ckpt_path: str, fold_bn: bool = True) -> None:
    """Load encoder + policy heads from a checkpoint; re-init the value head.

    Parameter source priority matches eval_checkpoint.py: top2_params[0] (TOP1)
    > mean > model_state.  (top2_params[0] and mean are usually identical and are
    the STRONG individuals; model_state can differ slightly and be weaker.)

    - BN checkpoints are folded into no_bn (all models train without BN, matching
      the strong ES/GA models and avoiding BN mode issues in the search).
    - Checkpoints with fewer policy heads than the model are expanded by copying
      the single head — heads start identical and differentiate through training.
    """
    from my_ai.network import AntWarNetwork, create_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = infer_model_config(ckpt)
    ck_heads = cfg["num_heads"]

    if "top2_params" in ckpt and len(ckpt["top2_params"]) > 0:
        vec = np.asarray(ckpt["top2_params"][0].numpy(), dtype=np.float32)
        source = "top2_params[0] (TOP1)"
    elif "mean" in ckpt:
        vec = np.asarray(ckpt["mean"].numpy(), dtype=np.float32)
        source = "mean"
    else:
        vec = None
        source = "model_state"

    if vec is not None:
        tmp = create_model(
            num_resblocks=cfg["num_resblocks"],
            num_heads=ck_heads,
            latent_dim=cfg["latent_dim"],
            no_bn=cfg["no_bn"],
        )
        tmp.set_parameters_from_vector(vec)
        sd = tmp.state_dict()
    else:
        sd = dict(ckpt["model_state"])

    has_bn = "initial_conv.1.running_mean" in sd
    if fold_bn and has_bn:
        sd = AntWarNetwork.fold_bn_into_state_dict(sd)
        print("[train] hot-start: folded BN -> no_bn")

    if ck_heads < model.num_heads and "policy_heads.1.weight" not in sd:
        w = sd["policy_heads.0.weight"]
        b = sd["policy_heads.0.bias"]
        for i in range(model.num_heads):
            sd[f"policy_heads.{i}.weight"] = w
            sd[f"policy_heads.{i}.bias"] = b
        print(f"[train] hot-start: expanded {ck_heads} head(s) -> {model.num_heads} heads")

    # The value head is re-initialised below, so its keys are dropped first:
    # a different value_pool changes their shapes.  Everything else stays strict.
    sd = {k: v for k, v in sd.items() if not k.startswith("value_head.")}
    _res = model.load_state_dict(sd, strict=False)
    _unexpected = [k for k in _res.unexpected_keys]
    _missing = [k for k in _res.missing_keys if not k.startswith("value_head.")]
    if _unexpected or _missing:
        raise RuntimeError(f"hot-start state_dict mismatch: unexpected={_unexpected} "
                           f"missing(non-value-head)={_missing}")
    with torch.no_grad():
        for module in model.value_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    print(f"[train] hot-start loaded {Path(ckpt_path).name} (source: {source}); value head re-initialized")


class AZTrainer:
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        value_weight: float = 1.0,
        t_class: float = 1.0,
        t_pos: float = 1.0,
        epochs: int = 3,
        batch_size: int = 16,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_weight = value_weight
        self.t_class = t_class
        self.t_pos = t_pos
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def _policy_loss(self, action_map, head_logits, samples: list[dict]) -> torch.Tensor:
        """Factored-prior CE over each sample's stored full intent space."""
        total = torch.tensor(0.0, requires_grad=True)
        count = 0
        for b, s in enumerate(samples):
            h = head_logits[b]  # (24,)
            am = action_map[b]  # (24, 19, 19)
            cls = torch.from_numpy(s["intent_class"]).long()
            xs = torch.from_numpy(s["intent_x"]).long()
            ys = torch.from_numpy(s["intent_y"]).long()
            tgt = torch.from_numpy(s["target"]).float()
            n = cls.shape[0]
            if n == 0:
                continue
            # class marginal: softmax over the sample's legal classes
            legal_classes = torch.unique(cls)
            lse_c = torch.logsumexp(h[legal_classes] / self.t_class, dim=0)
            log_p_class = h[cls] / self.t_class - lse_c  # (n,)
            # position marginal: per (class) logsumexp over that class's positions
            log_p_pos = torch.zeros(n, dtype=torch.float32)
            pos_mask = xs >= 0
            for c in legal_classes.tolist():
                m = (cls == c) & pos_mask
                if not m.any():
                    continue
                if c <= 20:
                    am_c = am[c]  # (19, 19)
                    pos_logits = am_c[xs[m], ys[m]] / self.t_pos
                    lse_pos = torch.logsumexp(pos_logits, dim=0)
                    log_p_pos[m] = pos_logits - lse_pos
                # class-only intents (21/22/23): log_p_pos stays 0
            prior_log = log_p_class + log_p_pos
            total = total + (-(tgt * prior_log).sum())
            count += 1
        return total / max(count, 1)

    def _loss_step(self, samples: list[dict]) -> torch.Tensor:
        self.model.train()  # training path: dropout active
        boards = torch.stack(
            [torch.from_numpy(s["board"]).float() for s in samples]
        )
        stats = torch.stack(
            [torch.from_numpy(s["stats"]).float() for s in samples]
        )
        out = self.model(boards, stats)
        action_map = out["action_map"]  # (B, 24, 19, 19)
        head_logits = torch.stack(
            [out[f"head{s['head_idx'] + 1}_logits"][b] for b, s in enumerate(samples)]
        )  # (B, 24) — each sample uses its own head's logits
        policy_loss = self._policy_loss(action_map, head_logits, samples)
        value_pred = out["value"].squeeze(-1)  # (B,)
        value_tgt = torch.as_tensor(
            [s["value_target"] for s in samples], dtype=torch.float32
        )
        value_loss = F.mse_loss(value_pred, value_tgt)
        return policy_loss + self.value_weight * value_loss, policy_loss, value_loss

    def train_on_games(self, games: list) -> dict:
        samples = [s for game in games for s in game.samples]
        total_loss = total_p = total_v = 0.0
        n_steps = 0
        for epoch in range(self.epochs):
            self.rng.shuffle(samples)
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                loss, p_loss, v_loss = self._loss_step(batch)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                total_loss += float(loss)
                total_p += float(p_loss)
                total_v += float(v_loss)
                n_steps += 1
        return {
            "loss": total_loss / max(n_steps, 1),
            "policy_loss": total_p / max(n_steps, 1),
            "value_loss": total_v / max(n_steps, 1),
            "samples": len(samples),
        }

    def save(self, path: str, completed_batches: int = 0) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "num_heads": self.model.num_heads,
                "no_bn": self.model.no_bn,
                "latent_dim": self.model.LATENT_DIM,
                "num_resblocks": self.model.num_resblocks,
                "config": {
                    "t_class": self.t_class,
                    "t_pos": self.t_pos,
                    "epochs": self.epochs,
                },
                "completed_batches": completed_batches,
            },
            path,
        )
        print(f"[train] checkpoint saved -> {path}", flush=True)


def build_model(ckpt_path: str | None, keep_bn: bool = False,
                value_pool: str | None = None) -> nn.Module:
    """Build a model from a checkpoint.

    ``value_pool`` overrides the checkpoint's value-head pooling (None = keep
    it).  The value head is re-initialised by load_hotstart in either case, so
    changing the pool only changes the backbone-carrying inputs' widths.
    """
    from my_ai.network import create_model

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = infer_model_config(ckpt)
        vp = value_pool if value_pool is not None else cfg.get("value_pool", "gap")
        target_heads = max(cfg["num_heads"], 3)  # our intent tree needs 3 heads
        if cfg.get("gn"):
            # GroupNorm cannot be folded away (per-sample stats) — keep it as-is.
            model = create_model(
                num_resblocks=cfg["num_resblocks"], num_heads=target_heads,
                latent_dim=cfg["latent_dim"], no_bn=False, gn=True,
                gn_groups=cfg.get("gn_groups", 8),
                value_pool=vp,
            )
            load_hotstart(model, ckpt_path, fold_bn=False)
            return model
        model = create_model(
            num_resblocks=cfg["num_resblocks"],
            num_heads=target_heads,
            latent_dim=cfg["latent_dim"],
            no_bn=not keep_bn,  # default: fold BN checkpoints, train without BN
            value_pool=vp,
        )
        load_hotstart(model, ckpt_path, fold_bn=not keep_bn)
    else:
        model = create_model(num_heads=3, no_bn=not keep_bn,
                             value_pool=value_pool or "gap")
    return model


def main() -> None:
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.mcts import IntentMCTS
    from my_ai.az_intent.selfplay import run_game

    parser = argparse.ArgumentParser(description="Intent-space AlphaZero training (方案 B)")
    parser.add_argument("--hotstart", type=str, default=None, help="ES checkpoint for hot-start")
    parser.add_argument("--checkpoint", type=str, default="training_history/az_intent/az_intent.pt")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--games-per-batch", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--max-depth-rounds", type=int, default=2)
    parser.add_argument("--temp-rounds", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--t-class", type=float, default=1.0)
    parser.add_argument("--t-pos", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from a prior AZTrainer checkpoint (our .pt format)")
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save a per-batch snapshot every N batches (0=off)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    from my_ai.network import create_model

    start_batch = 0
    if args.resume and not Path(args.resume).exists():
        print(f"[train] resume checkpoint {args.resume} missing -> cold start", flush=True)
        args.resume = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = create_model(
            num_resblocks=ckpt.get("num_resblocks", 6),
            num_heads=ckpt.get("num_heads", 3),
            latent_dim=ckpt.get("latent_dim", 64),
            no_bn=True,
        )
        model.load_state_dict(ckpt["model_state"])
        start_batch = int(ckpt.get("completed_batches", 0))
        print(f"[train] resumed from {args.resume} at batch {start_batch}", flush=True)
    else:
        model = build_model(args.hotstart)

    feat = FeatureExtractor(max_actions=96)
    net_fn = make_net_fn(model, feat)
    mcts = IntentMCTS(
        net_fn,
        iterations=args.iterations,
        max_depth_rounds=args.max_depth_rounds,
        c_puct=args.c_puct,
        t_class=args.t_class,
        t_pos=args.t_pos,
        seed=args.seed,
    )
    trainer = AZTrainer(
        model,
        lr=args.lr,
        t_class=args.t_class,
        t_pos=args.t_pos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    consecutive_failures = 0
    for batch_idx in range(start_batch, args.batches):
        t0 = time.perf_counter()
        games = []
        for g in range(args.games_per_batch):
            seed = args.seed * 1000 + batch_idx * 100 + g
            try:
                game = run_game(
                    net_fn, feat, mcts, seed,
                    max_rounds=args.max_rounds, temp_rounds=args.temp_rounds,
                )
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                print(f"  batch={batch_idx} game={g} FAILED: {exc!r} "
                      f"(consecutive={consecutive_failures})", flush=True)
                if consecutive_failures >= 3:
                    raise
                continue
            games.append(game)
            winner = "p0" if game.winner == 0 else ("p1" if game.winner == 1 else "draw")
            print(f"  batch={batch_idx} game={g} rounds={game.rounds} winner={winner} "
                  f"hp={game.hp} samples={game.num_samples}", flush=True)
        if not games:
            continue
        metrics = trainer.train_on_games(games)
        trainer.save(args.checkpoint, completed_batches=batch_idx + 1)
        if args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
            snap = f"{Path(args.checkpoint).with_suffix('')}_batch{batch_idx + 1}.pt"
            trainer.save(snap, completed_batches=batch_idx + 1)
        print(f"batch={batch_idx} {metrics} elapsed={time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
