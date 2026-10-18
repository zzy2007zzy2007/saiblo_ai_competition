#!/usr/bin/env python3
"""解析亚军 rollout dump（`value_seed*.jsonl`，由 `-DLURE_DUMP_VALUE` 的亚军 exe 写盘）。

每条：`{round, player, base_hp, coins, value, towers:[[x,y,type,hp]], ants:[[x,y,hp,kind,level,age,behavior]]}`
注意：这是**单边视角**（`DefenseSimulator` 里的 = 己方塔 + 敌方蚁），没有 42 维 stats。
所以 model 的 `stats` 通道填 `[base_hp/50, coins/300, round/512, 0...]`（只有前 3 维有效）。

输出 npz（pad 到固定长度）：
  tower_cont (N,Tn,3)  # is_own=1, hp/max_hp, 0(cooldown)
  tower_type (N,Tn) i16, tower_pos (N,Tn) i16, tower_mask (N,Tn) bool
  ant_cont (N,An,4)    # is_own=0, hp/30, level/2, age/64
  ant_kind (N,An) i8, ant_behavior (N,An) i8, ant_pos (N,An) i16, ant_mask (N,An) bool
  stats (N,42) f32, value (N,) f32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.entity_transformer.model import TOWER_TYPE_TO_ID, PAD_CELL  # noqa: E402

# TowerType -> max_hp（Basic=10，其余=15；见 SDK constants）
def _tower_max_hp(tid: int) -> int:
    return 10 if tid == 0 else 15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="training_history/distill_top2/value_dump_test")
    ap.add_argument("--out", default="training_history/distill_top2/rollout_ds.npz")
    ap.add_argument("--max-towers", type=int, default=64)
    ap.add_argument("--max-ants", type=int, default=96)
    ap.add_argument("--max-samples", type=int, default=0, help="0=全部")
    a = ap.parse_args()

    dd = Path(a.dump_dir)
    if not dd.is_absolute():
        dd = _REPO / dd
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out
    files = sorted(dd.glob("value_seed*.jsonl"))
    print(f"[roll] {len(files)} 个 dump 文件", flush=True)

    Tn, An = a.max_towers, a.max_ants
    TC, TT, TP, TM = [], [], [], []
    AC, AK, AB, AP, AM = [], [], [], [], []
    ST, VAL = [], []
    SEED = []
    n_tw_max = n_ant_max = 0
    n_skip = 0
    for fi, f in enumerate(files):
        try:
            seed = int(f.stem.replace("value_seed", ""))
        except ValueError:
            seed = -1
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    n_skip += 1
                    continue
                towers = r.get("towers") or []
                ants = r.get("ants") or []
                n_tw_max = max(n_tw_max, len(towers))
                n_ant_max = max(n_ant_max, len(ants))

                tc = np.zeros((Tn, 3), np.float32)
                tt = np.full(Tn, 16, np.int16)
                tp = np.full(Tn, PAD_CELL, np.int16)
                tmask = np.zeros(Tn, bool)
                for k, t in enumerate(towers[:Tn]):
                    x, y, tid, hp = (int(v) for v in t[:4])
                    tc[k] = [1.0, float(hp) / max(_tower_max_hp(tid), 1), 0.0]
                    tt[k] = TOWER_TYPE_TO_ID.get(tid, 16)
                    tp[k] = x * 19 + y
                    tmask[k] = True

                ac = np.zeros((An, 4), np.float32)
                ak = np.full(An, 2, np.int8)
                ab = np.full(An, 5, np.int8)
                ap = np.full(An, PAD_CELL, np.int16)
                amask = np.zeros(An, bool)
                for k, an in enumerate(ants[:An]):
                    x, y, hp, kind, level, age, beh = (int(v) for v in an[:7])
                    max_hp = 30.0 if kind == 1 else (10.0 + 5.0 * level)
                    ac[k] = [0.0, float(hp) / max_hp, float(level) / 2.0, float(age) / 64.0]
                    ak[k] = min(max(kind, 0), 1)
                    ab[k] = min(max(beh, 0), 4)
                    ap[k] = x * 19 + y
                    amask[k] = True

                stats = np.zeros(42, np.float32)
                stats[0] = float(r.get("base_hp", 50)) / 50.0
                stats[1] = float(r.get("coins", 0)) / 300.0
                stats[2] = float(r.get("round", 0)) / 512.0

                TC.append(tc); TT.append(tt); TP.append(tp); TM.append(tmask)
                AC.append(ac); AK.append(ak); AB.append(ab); AP.append(ap); AM.append(amask)
                ST.append(stats); VAL.append(float(r.get("value", 0.0)))
                SEED.append(seed)
                if a.max_samples and len(VAL) >= a.max_samples:
                    break
        print(f"  [{fi+1}/{len(files)}] {f.name} recs={len(VAL)}", flush=True)
        if a.max_samples and len(VAL) >= a.max_samples:
            break

    if not VAL:
        print("[roll] 没有样本！", flush=True)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tower_cont=np.stack(TC), tower_type=np.stack(TT), tower_pos=np.stack(TP), tower_mask=np.stack(TM),
        ant_cont=np.stack(AC), ant_kind=np.stack(AK), ant_behavior=np.stack(AB), ant_pos=np.stack(AP),
        ant_mask=np.stack(AM), stats=np.stack(ST), value=np.array(VAL, np.float32),
        seed=np.array(SEED, np.int32))
    v = np.array(VAL, np.float32)
    print(f"[roll] saved {len(v)} -> {out}", flush=True)
    print(f"[roll] value mean/std/min/max = {v.mean():.1f}/{v.std():.1f}/{v.min():.1f}/{v.max():.1f}", flush=True)
    print(f"[roll] 实测 max towers={n_tw_max} ants={n_ant_max} (pad {Tn}/{An}), skip={n_skip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
