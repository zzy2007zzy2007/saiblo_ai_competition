"""Search cost benchmark: how much does each search_mode actually cost, and WHY.

Times ``BundleMCTS.search`` on a FIXED set of real game states — sampled from a
raw-policy self-play game, so the state/turn mix matches what a real search run
sees — once per candidate-axis mode.  Because the states are identical across
modes, the wall-clock ratio is a clean cost comparison.

Also counts the mechanism counters so the speedup is *attributed* rather than
just observed.  The two cost drivers per tree expansion are
  1. ONE model forward (``net_fn``) — needed to get action_map + head logits,
  2. ``k * sample_mult`` ``sample_bundle`` calls (360 at the default k=24),
so ``#expansions`` is the real dial.  A narrower candidate axis can only help
by making the tree shallower/narrower (fewer expansions), NOT by making an
expansion itself cheaper.

Modes:
  joint              — sample classes AND positions (default)
  class-only         — positions pinned to action_map argmax, classes sampled
  pos-only(argmax)   — class pinned to the raw logits argmax, positions sampled
  pos-only(playable) — class pinned to the highest-logit EXECUTABLE class
  skip-hold          — joint, but skip the search entirely when all 3 heads'
                       masked argmax is HOLD (mirrors az_selfplay --skip-hold-search)

Usage:
    python _tmp_search_speed_bench.py [checkpoint] [n_states] [iterations] [depth]
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch.set_num_threads(1)

from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_initial_state
from my_ai.az_intent import bundle_mcts as bm
from my_ai.az_intent.bundle_mcts import BundleMCTS
from my_ai.decoder import decode_network_output, make_class_mask, make_position_masks
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = sys.argv[1] if len(sys.argv) > 1 else "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
N_STATES = int(sys.argv[2]) if len(sys.argv) > 2 else 96
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 256
DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else 4

feat = FeatureExtractor(max_actions=96)
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()

COUNTERS = {"forwards": 0, "expands": 0, "samples": 0}


def full_net_fn(st, player):
    obs = feat.encode_observation(st, player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
    COUNTERS["forwards"] += 1
    return {"action_map": out["action_map"][0].numpy(),
            "head_logits": [out[h][0].numpy() for h in heads],
            "value": float(out["value"].item())}


# ── instrument the two expansion drivers (no production-code changes) ────────
_orig_expand = BundleMCTS._expand


def _counted_expand(self, node, k):
    COUNTERS["expands"] += 1
    return _orig_expand(self, node, k)


BundleMCTS._expand = _counted_expand

_orig_sample_bundle = bm.sample_bundle


def _counted_sample_bundle(*a, **kw):
    COUNTERS["samples"] += 1
    return _orig_sample_bundle(*a, **kw)


bm.sample_bundle = _counted_sample_bundle


# ── collect a fixed, representative set of states from a raw self-play game ──
def collect_states(seed: int, want: int) -> list[tuple]:
    state = make_initial_state(seed, True)
    snaps: list[tuple] = []
    stride, turn = 3, 0
    for _ in range(600):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            turn += 1
            if turn % stride == 0 and len(snaps) < want:
                snaps.append((state.clone(), player))
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            ops = decode_network_output(out, state, player, temperature=0.0,
                                       intent_decoding=True)
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()
    return snaps


SNAPS = collect_states(3, N_STATES)
print(f"collected {len(SNAPS)} states from a raw self-play game (seed 3)", flush=True)


def is_all_hold(st, player) -> bool:
    obs = feat.encode_observation(st, player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    pm = make_position_masks(st, player, intent_decoding=True)
    cm = make_class_mask(st, player, position_mask=pm, intent_decoding=True)
    votes = []
    for h in range(model.num_heads):
        lg = out[f"head{h + 1}_logits"].squeeze(0).numpy().astype(np.float64)
        votes.append(int(np.argmax(np.where(cm, lg, -1e9))))
    return all(v == 23 for v in votes)


MODES = [("joint", "joint", "argmax"),
         ("class-only", "class-only", "argmax"),
         ("pos-only(argmax)", "pos-only", "argmax"),
         ("pos-only(playable)", "pos-only", "playable"),
         ("skip-hold", "joint", "argmax")]

print(f"\nconfig: ckpt={Path(CK).name} iterations={ITERS} depth={DEPTH} "
      f"k=24 sample_mult=15 (=> 360 samples/expansion)\n", flush=True)
print(f"{'mode':22s} {'total_s':>8s} {'med_ms':>8s} {'mean_ms':>8s} "
      f"{'expands':>8s} {'samples':>9s} {'fwd':>6s} {'root_cands':>10s}")
results = {}
for label, mode, pin in MODES:
    times, expands, samples, fwds, ncand, skipped = [], [], [], [], [], 0
    for i, (st, pl) in enumerate(SNAPS):
        if label == "skip-hold" and is_all_hold(st, pl):
            skipped += 1
            times.append(0.0)
            expands.append(0)
            samples.append(0)
            fwds.append(0)
            ncand.append(1)
            continue
        m = BundleMCTS(full_net_fn, iterations=ITERS, max_depth_rounds=DEPTH, k=24,
                       t_class=0.5, t_pos=0.3, seed=1000 + i,
                       search_mode=mode, pos_pin=pin)
        for key in COUNTERS:
            COUNTERS[key] = 0
        t0 = time.perf_counter()
        res = m.search(st, pl, temperature=0.0)
        dt = time.perf_counter() - t0
        times.append(dt)
        expands.append(COUNTERS["expands"])
        samples.append(COUNTERS["samples"])
        fwds.append(COUNTERS["forwards"])
        ncand.append(len(set(res.bundles)))
    results[label] = (sum(times), statistics.median(times), statistics.mean(times),
                      statistics.mean(expands), statistics.mean(samples),
                      statistics.mean(fwds), statistics.mean(ncand), skipped)
    tot, med, mean, e, s, f, nc, sk = results[label]
    extra = f"  skipped={sk}/{len(SNAPS)}" if label == "skip-hold" else ""
    print(f"{label:22s} {tot:8.1f} {med * 1e3:8.1f} {mean * 1e3:8.1f} "
          f"{e:8.1f} {s:9.1f} {f:6.1f} {nc:10.2f}{extra}", flush=True)

base = results["joint"][0]
print("\n--- 相对 joint 的总耗时比（同一批 state，越小越快）---")
for label, r in results.items():
    print(f"  {label:22s} {r[0] / base:6.2f}x   (中位数 {r[1] / results['joint'][1]:5.2f}x)")
print("\n--- 归因：每局的成本 ≈ expansions × (1 forward + 360 sample_bundle) ---")
for label, r in results.items():
    print(f"  {label:22s} 平均 expansions={r[3]:6.1f}  samples={r[4]:8.1f}  "
          f"forwards={r[5]:6.1f}  root 候选数={r[6]:.2f}")
print("\nspeed bench done")
