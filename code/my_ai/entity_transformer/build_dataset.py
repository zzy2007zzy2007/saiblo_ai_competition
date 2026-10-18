#!/usr/bin/env python3
"""解析桥接的 raw dump（`.bin` + `ents_*.jsonl`）→ 实体 token 数据集。

输入（同目录、按 seed 配对）：
  - `raw_seed<seed>.bin`    : 每个决策两个玩家各一条（board+stats+player+round+value）
  - `ents_seed<seed>.jsonl` : 每回合一条（towers/ants/coins/bases_hp）

输出 npz（pad 到固定长度）：
  tower_cont (N,Tn,3) f32   # is_own, hp_ratio, cooldown(=0)
  tower_type (N,Tn) i16      # 0..16，16=pad
  tower_pos  (N,Tn) i16      # x*19+y，361=pad
  tower_mask (N,Tn) bool
  ant_cont   (N,An,4) f32    # is_own, hp_ratio, level, age/64
  ant_kind   (N,An) i8       # 0/1，2=pad
  ant_behavior (N,An) i8     # 0..4，5=pad
  ant_pos    (N,An) i16
  ant_mask   (N,An) bool
  stats (N,42) f32, value (N,) f32, seed (N,) i32, round (N,) i16, player (N,) i8
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

BOARD_SHAPE = (28, 19, 19)
BOARD_BYTES = int(np.prod(BOARD_SHAPE)) * 2      # 20216
STATS_BYTES = 42 * 2                              # 84
REC_BYTES = BOARD_BYTES + STATS_BYTES + 1 + 2 + 4


def _tower_max_hp(tower_type: int) -> int:
    return 10 if int(tower_type) == 0 else 15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="training_history/distill_top2/ru_ent_raw")
    ap.add_argument("--out", default="training_history/distill_top2/entity_ds.npz")
    ap.add_argument("--max-towers", type=int, default=200)
    ap.add_argument("--max-ants", type=int, default=120)
    ap.add_argument("--with-board", action="store_true", help="额外存 board(28,19,19)（供格 token 用）")
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    if not raw.is_absolute():
        raw = _REPO / raw
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out

    bins = sorted(raw.glob("raw_seed*.bin"))
    Tn, An = a.max_towers, a.max_ants
    print(f"[build] {len(bins)} seeds, pad Tn={Tn} An={An}", flush=True)

    TC, TT, TP, TM = [], [], [], []
    AC, AK, AB, AP, AM = [], [], [], [], []
    ST, VAL, SEED, RND, PL = [], [], [], [], []
    BOARDS: list = []
    n_ant_max = n_tw_max = 0

    for bi, bf in enumerate(bins):
        seed = int(bf.stem.replace("raw_seed", ""))
        ef = raw / f"ents_seed{seed:05d}.jsonl"
        if not ef.exists():
            print(f"  [skip] 无 ents jsonl: {ef.name}", flush=True)
            continue
        ents = {}
        with open(ef, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    ents[int(r["round"])] = r
        n_rec = bf.stat().st_size // REC_BYTES
        if n_rec == 0:
            continue
        buf = np.fromfile(bf, dtype=np.uint8)[: n_rec * REC_BYTES].reshape(n_rec, REC_BYTES)
        off = BOARD_BYTES
        stats = buf[:, off:off + STATS_BYTES].copy().view(np.float16).reshape(n_rec, 42)
        off += STATS_BYTES
        pl = buf[:, off].astype(np.int64)
        off += 1
        rnd = buf[:, off:off + 2].copy().view(np.int16).reshape(n_rec)
        off += 2
        val = buf[:, off:off + 4].copy().view(np.float32).reshape(n_rec)

        for i in range(n_rec):
            r = int(rnd[i])
            p = int(pl[i])
            ent = ents.get(r)
            if ent is None:
                continue
            towers = ent.get("towers", [])
            ants = ent.get("ants", [])
            n_tw_max = max(n_tw_max, len(towers))
            n_ant_max = max(n_ant_max, len(ants))

            tc = np.zeros((Tn, 3), np.float32)
            tt = np.full(Tn, 16, np.int16)     # 16 = pad
            tp = np.full(Tn, PAD_CELL, np.int16)
            tmask = np.zeros(Tn, bool)
            for k, tw in enumerate(towers[:Tn]):
                _id, owner, x, y, ttype, hp = (int(v) for v in tw[:6])
                tc[k] = [1.0 if owner == p else 0.0,
                         float(hp) / max(_tower_max_hp(ttype), 1), 0.0]
                tt[k] = TOWER_TYPE_TO_ID.get(ttype, 16)
                tp[k] = x * 19 + y
                tmask[k] = True

            ac = np.zeros((An, 4), np.float32)
            ak = np.full(An, 2, np.int8)
            ab = np.full(An, 5, np.int8)
            ap = np.full(An, PAD_CELL, np.int16)
            amask = np.zeros(An, bool)
            for k, an in enumerate(ants[:An]):
                _id, owner, x, y, hp, max_hp, kind, age, beh = (int(v) for v in an[:9])
                ac[k] = [1.0 if owner == p else 0.0,
                         float(hp) / max(max_hp, 1),
                         float(age) / 64.0,
                         float(kind)]
                ak[k] = min(max(kind, 0), 1)
                ab[k] = min(max(beh, 0), 4)
                ap[k] = x * 19 + y
                amask[k] = True

            TC.append(tc); TT.append(tt); TP.append(tp); TM.append(tmask)
            AC.append(ac); AK.append(ak); AB.append(ab); AP.append(ap); AM.append(amask)
            ST.append(stats[i].astype(np.float32)); VAL.append(float(val[i]))
            SEED.append(seed); RND.append(r); PL.append(p)
            if a.with_board:
                BOARDS.append(buf[i, :BOARD_BYTES].copy().view(np.float16).reshape(28, 19, 19))

        print(f"  [{bi+1}/{len(bins)}] seed={seed} recs={len(TC)}", flush=True)

    if not VAL:
        print("[build] 没有样本！", flush=True)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tower_cont=np.stack(TC), tower_type=np.stack(TT), tower_pos=np.stack(TP), tower_mask=np.stack(TM),
        ant_cont=np.stack(AC), ant_kind=np.stack(AK), ant_behavior=np.stack(AB), ant_pos=np.stack(AP),
        ant_mask=np.stack(AM),
        stats=np.stack(ST), value=np.array(VAL, np.float32),
        seed=np.array(SEED, np.int32), round=np.array(RND, np.int16), player=np.array(PL, np.int8))
    v = np.array(VAL, np.float32)
    print(f"[build] saved {len(v)} -> {out}", flush=True)
    print(f"[build] value mean/std/min/max = {v.mean():.1f}/{v.std():.1f}/{v.min():.1f}/{v.max():.1f}", flush=True)
    print(f"[build] 实测 max towers={n_tw_max} ants={n_ant_max} (pad {Tn}/{An})", flush=True)
    if a.with_board and BOARDS:
        bp = out.with_suffix(".board.npy")
        np.save(bp, np.stack(BOARDS))
        print(f"[build] board -> {bp} {len(BOARDS)}x28x19x19", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
