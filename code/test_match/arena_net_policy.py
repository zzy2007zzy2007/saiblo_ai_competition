"""把"网络裸策略"（net argmax，**不带任何搜索**）接进快竞技场，和规则策略对打。

为什么：本次的结论都建立在"规则策略"层面（快、N 大）。要把它们和现役的 ML 栈接起来，
第一块砖是"网络自己的策略到底有多强"：
  - 若 net-argmax 打不过几十行规则 ⇒ 现役的强度主要不在网络的策略上（而在搜索/评估/或都没有）；
  - 若 net-argmax 更强 ⇒ 规则结论只能当"经济层面的先验"，不能替代网络。

用法：
  python code/test_match/arena_net_policy.py --duel net:checkpoints/posnet_A_k5_m32.pt ref --seeds 32
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAX_ROUNDS = 512


class NetPolicy:
    """网络裸策略：一次前向 → decode_network_output（temperature=0，最多 3 op）。"""

    def __init__(self, ckpt: str, max_actions: int = 96):
        import numpy as np
        import torch
        from SDK.utils.features import FeatureExtractor
        from my_ai.az_intent.az_selfplay import (ThreeNetPolicy, load_model_from_ckpt,
                                                 load_split_models, load_three_models)
        torch.set_num_threads(1)
        self.torch = torch
        self.model = None
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "class_state" in ck and "pos_state" in ck:
            self.model = ThreeNetPolicy(*load_three_models(ckpt)[:2])
        elif "value_state" in ck:
            self.model = load_split_models(ckpt)[0]
        else:
            self.model = load_model_from_ckpt(ckpt)
        self.model.eval()
        self.feat = FeatureExtractor(max_actions=max_actions)
        self.max_actions = max_actions
        self.ckpt = ckpt

    def act(self, state, player):
        import numpy as np
        from my_ai.decoder import decode_network_output
        try:
            obs = self.feat.encode_observation(state, player, np.zeros(self.max_actions))
            with self.torch.no_grad():
                out = self.model(
                    self.torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    self.torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            return decode_network_output(out, state, player, temperature=0.0,
                                         intent_decoding=True)
        except Exception as exc:  # noqa: BLE001
            print(f"!! NetPolicy.act 异常 round={state.round_index} p={player}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return []


def make(spec: str):
    if spec.startswith("net:"):
        return NetPolicy(spec[4:])
    from test_match.fast_arena import make_policy
    return make_policy(spec)


def play(spec_a: str, spec_b: str, seed: int) -> dict:
    from test_match.fast_arena import _new_state
    a_seat = 0 if seed % 2 == 0 else 1
    pa, pb = make(spec_a), make(spec_b)
    pol = {a_seat: pa, 1 - a_seat: pb}
    st = _new_state(seed)
    for _ in range(MAX_ROUNDS):
        if st.terminal:
            break
        for player in (0, 1):
            if st.terminal:
                break
            ops = pol[player].act(st, player)
            if ops:
                st.apply_operation_list(player, ops)
        if st.terminal:
            break
        st.advance_round()
    hp = (int(st.bases[0].hp), int(st.bases[1].hp))
    return dict(seed=seed, a_seat=a_seat, rounds=int(st.round_index),
                margin=hp[a_seat] - hp[1 - a_seat], hp=hp,
                winner=(None if st.winner is None else int(st.winner)),
                towers=(len(st.towers_of(0)), len(st.towers_of(1))),
                coins=(int(st.coins[0]), int(st.coins[1])))


def _job(a):
    return play(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duel", nargs=2, required=True)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--start-seed", type=int, default=7000)
    a = ap.parse_args()
    jobs = [(a.duel[0], a.duel[1], a.start_seed + i) for i in range(a.seeds)]
    if a.workers <= 1:
        outs = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            outs = list(ex.map(_job, jobs, chunksize=2))
    mg = [o["margin"] for o in outs]
    print(json.dumps(dict(
        a=a.duel[0], b=a.duel[1], n=len(mg),
        margin=statistics.mean(mg),
        se=statistics.pstdev(mg) / max(len(mg) ** 0.5, 1),
        win=sum(1 for x in mg if x > 0), loss=sum(1 for x in mg if x < 0),
        tie=sum(1 for x in mg if x == 0),
        rounds=statistics.mean(o["rounds"] for o in outs),
        towers=statistics.mean(sum(o["towers"]) for o in outs),
        coins=statistics.mean(sum(o["coins"]) for o in outs),
    ), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
