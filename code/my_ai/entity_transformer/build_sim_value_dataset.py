#!/usr/bin/env python3
"""把 128 局实体数据集**投影**成"亚军 sim 能提供"的单边输入。

亚军的 `evaluate_terminal` 只依赖：
  己方塔（tower_full_salvage_value）+ 敌方蚁（worker/combat_threat）+ base_hp + coins
⇒ 单边投影就是它的完整输入。

输入：`entity_ds_128.npz`（含 `is_own` 标志的双方实体 + 42 维 stats + value）
输出：同 schema，但
  - `tower_*` 只留 `is_own>0.5`（我的塔）
  - `ant_*`   只留 `is_own<0.5`（敌方的蚁）
  - `stats`  只填 2 维：`[base_hp/50, coins/300]`（从 extras 的 24/22 位取；其余置 0）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 42 维 stats 的 extras 偏移（前 22 个是 named）
#   22=coins[player]/300  23=coins[enemy]/300  24=bases[player].hp/50  25=bases[enemy].hp/50
IDX_MY_COINS = 22
IDX_MY_BASE_HP = 24


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training_history/distill_top2/entity_ds_128.npz")
    ap.add_argument("--out", default="training_history/distill_top2/sim_value_ds_128.npz")
    a = ap.parse_args()
    d = np.load(_REPO / a.data if not Path(a.data).is_absolute() else a.data)
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    N = d["value"].shape[0]
    print(f"[proj] N={N}", flush=True)

    tower_cont = d["tower_cont"]; tower_type = d["tower_type"]; tower_pos = d["tower_pos"]
    ant_cont = d["ant_cont"]; ant_kind = d["ant_kind"]; ant_behavior = d["ant_behavior"]; ant_pos = d["ant_pos"]
    stats = d["stats"]

    Tn = tower_cont.shape[1]; An = ant_cont.shape[1]
    o_tc = np.zeros_like(tower_cont); o_tt = np.full_like(tower_type, 16)
    o_tp = np.full_like(tower_pos, 361); o_tm = np.zeros_like(tower_cont[:, :, 0], dtype=bool)
    o_ac = np.zeros_like(ant_cont); o_ak = np.full_like(ant_kind, 2)
    o_ab = np.full_like(ant_behavior, 5); o_ap = np.full_like(ant_pos, 361)
    o_am = np.zeros_like(ant_cont[:, :, 0], dtype=bool)
    o_st = np.zeros_like(stats)

    n_my_tw = 0; n_en_ant = 0
    for i in range(N):
        # 我的塔：is_own>0.5
        tw_idx = np.where(tower_cont[i, :, 0] > 0.5)[0]
        k = 0
        for j in tw_idx[:Tn]:
            o_tc[i, k] = tower_cont[i, j]; o_tt[i, k] = tower_type[i, j]
            o_tp[i, k] = tower_pos[i, j]; o_tm[i, k] = True
            k += 1
        n_my_tw += k
        # 敌方的蚁：is_own<0.5
        an_idx = np.where(ant_cont[i, :, 0] < 0.5)[0]
        k = 0
        for j in an_idx[:An]:
            o_ac[i, k] = ant_cont[i, j]; o_ak[i, k] = ant_kind[i, j]
            o_ab[i, k] = ant_behavior[i, j]; o_ap[i, k] = ant_pos[i, j]; o_am[i, k] = True
            k += 1
        n_en_ant += k
        # 2 维标量（sim 能提供的）
        o_st[i, 0] = stats[i, IDX_MY_BASE_HP]
        o_st[i, 1] = stats[i, IDX_MY_COINS]

    np.savez_compressed(
        out,
        tower_cont=o_tc, tower_type=o_tt, tower_pos=o_tp, tower_mask=o_tm,
        ant_cont=o_ac, ant_kind=o_ak, ant_behavior=o_ab, ant_pos=o_ap, ant_mask=o_am,
        stats=o_st, value=d["value"], seed=d["seed"], round=d["round"], player=d["player"])
    print(f"[proj] saved -> {out}", flush=True)
    print(f"[proj] 平均我的塔={n_my_tw/N:.2f} 敌方蚁={n_en_ant/N:.2f}", flush=True)
    v = d["value"]
    print(f"[proj] value mean/std/min/max = {v.mean():.1f}/{v.std():.1f}/{v.min():.1f}/{v.max():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
