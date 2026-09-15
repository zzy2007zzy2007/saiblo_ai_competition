"""把 pos-only 搜索的回合按"候选构成"拆开：单候选到底是 HOLD 还是被迫动作？

用户的关键判断：那 97% 单候选回合**都是 HOLD**（而不是"某个被迫的真实动作"），
所以它们与位置轴无关。这里直接测：
  pass        : bundles 为空，或唯一候选是空 bundle（= 这一回合不动作）
  forced_act  : 唯一候选是一个非空 bundle（被迫出招，无位置选择余地）
  choice      : >=2 个候选（有真正的位置选择）
"""
from __future__ import annotations

import sys
from collections import Counter
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
GAMES = (3, 11)
WANT = 110

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
    SNAPS += collect(g, WANT)
print(f"states = {len(SNAPS)} (from raw self-play seeds {GAMES}, stride 2)", flush=True)

for tag, kw in (("pos-only + skip-single", dict(search_mode="pos-only", skip_single_candidate=True)),
                ("pos-only (no skip)", dict(search_mode="pos-only", skip_single_candidate=False)),
                ("joint (no skip)", dict(search_mode="joint", skip_single_candidate=False))):
    kinds = Counter()
    nsizes = Counter()
    for i, (st, pl) in enumerate(SNAPS):
        m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=24,
                       t_class=0.5, t_pos=0.3, seed=1000 + i, **kw)
        res = m.search(st, pl, temperature=0.0)
        nb = len(res.bundles)
        nsizes[nb] += 1
        empty_only = (nb == 0) or (nb == 1 and len(res.bundles[0]) == 0)
        if nb >= 2:
            kinds["choice (>=2 cands)"] += 1
        elif empty_only:
            kinds["pass (single empty)"] += 1
        else:
            kinds["forced_act (single non-empty)"] += 1
    n = len(SNAPS)
    print(f"\n=== {tag} ===")
    for k, v in kinds.most_common():
        print(f"   {k:28s} {v:5d}  ({v / n:6.1%})")
    print(f"   候选数分布: {dict(sorted(nsizes.items())[:6])}")
    sys.stdout.flush()
print("\nbreakdown done")
