"""验证 skip_single_candidate 是严格等价的（不是近似），并实测加速比。

同一批真实局面、同一个 seed、同一份 RNG 流，只切 skip_single_candidate：
  - chosen_bundle / bundles / visit_policy / intent_counts 必须逐位相等；
  - root_value 允许不同（语义从"访问均值"变成"根网络值"，已查无消费方）。

同时计时，检验预测：单候选回合 748ms -> ~4ms（1 前向 + 360 采样）。
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
from my_ai.az_intent.bundle_mcts import BundleMCTS
from my_ai.decoder import decode_network_output
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
ITERS, DEPTH = 256, 4
GAMES = (3, 11, 29)     # 三个种子的 raw 自对弈，stride 2 -> 局面更分散

feat = FeatureExtractor(max_actions=96)
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()


def net_fn(st, player):
    obs = feat.encode_observation(st, player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
    return {"action_map": out["action_map"][0].numpy(),
            "head_logits": [out[h][0].numpy() for h in heads],
            "value": float(out["value"].item())}


def collect(seed: int, want: int) -> list[tuple]:
    state = make_initial_state(seed, True)
    snaps, turn = [], 0
    for _ in range(700):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            turn += 1
            if turn % 2 == 0 and len(snaps) < want:
                snaps.append((state.clone(), player))
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            state.apply_operation_list(player, decode_network_output(
                out, state, player, temperature=0.0, intent_decoding=True))
        if player == 1 and not state.terminal:
            state.advance_round()
    return snaps


SNAPS: list[tuple] = []
for g in GAMES:
    SNAPS += collect(g, 110)
print(f"collected {len(SNAPS)} states from {len(GAMES)} raw self-play games", flush=True)


def new_mcts(i: int, mode: str, skip: bool, seed_base: int = 1000):
    return BundleMCTS(net_fn, iterations=ITERS, max_depth_rounds=DEPTH, k=24,
                      t_class=0.5, t_pos=0.3, seed=seed_base + i,
                      search_mode=mode, skip_single_candidate=skip)


for label, mode in (("joint", "joint"), ("pos-only(argmax)", "pos-only")):
    bad, t_off, t_on = 0, 0.0, 0.0
    off_t, on_t, single = [], [], []
    for i, (st, pl) in enumerate(SNAPS):
        t0 = time.perf_counter(); r_off = new_mcts(i, mode, False).search(st, pl)
        d_off = time.perf_counter() - t0
        t0 = time.perf_counter(); r_on = new_mcts(i, mode, True).search(st, pl)
        d_on = time.perf_counter() - t0
        t_off += d_off; t_on += d_on
        off_t.append(d_off); on_t.append(d_on)
        single.append(len(r_off.bundles) <= 1)
        ok = (
            list(r_off.bundles) == list(r_on.bundles)
            and r_off.chosen_bundle == r_on.chosen_bundle
            and r_off.chosen_index == r_on.chosen_index
            and list(r_off.intent_counts) == list(r_on.intent_counts)
            and np.array_equal(np.asarray(r_off.visit_policy),
                               np.asarray(r_on.visit_policy))
            and abs(float(np.asarray(r_off.visit_policy).sum()) - 1.0) < 1e-6
        )
        if not ok:
            bad += 1
            if bad <= 3:
                print(f"  !! seed-index {i} pl={pl} nb_off={len(r_off.bundles)} "
                      f"off={r_off.chosen_bundle} on={r_on.chosen_bundle}")
    # temperature=1.0 (self-play 探索路径) 也必须一致
    temp_bad = 0
    for i, (st, pl) in enumerate(SNAPS[:60]):
        a = new_mcts(i, mode, False, seed_base=500).search(st, pl, temperature=1.0)
        b = new_mcts(i, mode, True, seed_base=500).search(st, pl, temperature=1.0)
        if a.chosen_bundle != b.chosen_bundle:
            temp_bad += 1

    n, ns = len(SNAPS), sum(single)
    print(f"\n=== search_mode={label} ===")
    print(f"  局面数={n}  单候选局面={ns} ({ns / n:.1%})")
    print(f"  逐位不一致局面数: {bad}    temperature=1.0 不一致: {temp_bad}/60")
    print(f"  总耗时 off={t_off:.1f}s on={t_on:.1f}s  ⇒ {t_off / t_on:.2f}×")
    print(f"  中位 off={statistics.median(off_t) * 1e3:7.1f}ms  "
          f"on={statistics.median(on_t) * 1e3:7.1f}ms")
    if ns:
        print(f"  只看单候选局面: off={statistics.median([off_t[i] for i in range(n) if single[i]]) * 1e3:7.1f}ms"
              f"  on={statistics.median([on_t[i] for i in range(n) if single[i]]) * 1e3:7.1f}ms")
    if n - ns:
        print(f"  只看多候选局面: off={statistics.median([off_t[i] for i in range(n) if not single[i]]) * 1e3:7.1f}ms"
              f"  on={statistics.median([on_t[i] for i in range(n) if not single[i]]) * 1e3:7.1f}ms")
    sys.stdout.flush()

print("\nverify done")
