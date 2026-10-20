#!/usr/bin/env python3
"""从低质量对局的 `.ents.jsonl` 构建**双边实体数据集**（塔+蚁，双方都存），
并与 `lowq_f64` 紧凑集（每局 64 帧）对齐，供 `entity_transformer/train.py` 直接训练。

格式对齐 `build_dataset.py`：
  tower_cont (N,Tn,3)  is_own, hp/max, cooldown(=0)
  tower_type (N,Tn)    0..16（16=pad）
  tower_pos  (N,Tn)    x*19+y（361=pad）
  tower_mask (N,Tn)
  ant_cont   (N,An,4)  is_own, hp/max, level, age/64
  ant_kind   (N,An)    0/1（2=pad）
  ant_behavior (N,An)  0..4（5=pad）
  ant_pos    (N,An)
  ant_mask   (N,An)
  stats (N,42) / value (N,) / seed (N,) = 局号

⚠️ 我们的采集里**没有存蚁的 level**（`ants` 字段只有 9 项），这里把 level 置 0，
   `hp/max` 已隐含了等级带来的上限差异。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "code"))
from my_ai.entity_transformer.model import TOWER_TYPE_TO_ID  # noqa: E402

MAX_TW = 48
MAX_ANT = 96


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="合并目录（含 game_offsets.npy）")
    ap.add_argument("--npz-dir", required=True, help="原始 5000 局目录（含 *.ents.jsonl）")
    ap.add_argument("--compact", required=True, help="lowq_f64 目录（含 frame_idx/stats/label）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src, nd, cp = Path(a.src), Path(a.npz_dir), Path(a.compact)
    offs = np.load(src / "game_offsets.npy")
    fidx = np.load(cp / "frame_idx.npy")
    cstats = np.load(cp / "stats.npy")
    clabel = np.load(cp / "label.npy")
    cgid = np.load(cp / "game_id.npy")
    N = len(fidx)
    print(f"[ent] {N} samples from {len(offs)-1} games", flush=True)

    # 全局行 -> 紧凑集行
    row_to_compact = {int(g): i for i, g in enumerate(fidx)}

    TC = np.zeros((N, MAX_TW, 3), np.float32)
    TT = np.full((N, MAX_TW), 16, np.int16)
    TP = np.full((N, MAX_TW), 361, np.int16)
    TM = np.zeros((N, MAX_TW), bool)
    AC = np.zeros((N, MAX_ANT, 4), np.float32)
    AK = np.full((N, MAX_ANT), 2, np.int8)
    AB = np.full((N, MAX_ANT), 5, np.int8)
    AP = np.full((N, MAX_ANT), 361, np.int16)
    AM = np.zeros((N, MAX_ANT), bool)
    seed = cgid.astype(np.int32)

    ents_files = sorted(nd.glob("*.ents.jsonl"))
    print(f"[ent] {len(ents_files)} ents files", flush=True)
    for gi, ef in enumerate(ents_files):
        s0 = int(offs[gi])
        with open(ef, encoding="utf-8") as fh:
            for j, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                g = s0 + j
                ci = row_to_compact.get(g)
                if ci is None:
                    continue
                r = json.loads(line)
                pl = int(r.get("player", 0))
                k = 0
                for t in (r.get("towers") or []):
                    _tid, owner, x, y, ttype, hp = (int(v) for v in t[:6])
                    if k >= MAX_TW:
                        break
                    TC[ci, k] = (1.0 if owner == pl else 0.0,
                                 hp / (10.0 if ttype == 0 else 15.0), 0.0)
                    TT[ci, k] = TOWER_TYPE_TO_ID.get(ttype, 16)
                    TP[ci, k] = x * 19 + y
                    TM[ci, k] = True
                    k += 1
                k = 0
                for ant in (r.get("ants") or []):
                    _aid, owner, x, y, hp, max_hp, kind, age, beh = (int(v) for v in ant[:9])
                    if k >= MAX_ANT:
                        break
                    AC[ci, k] = (1.0 if owner == pl else 0.0,
                                 hp / max(max_hp, 1), 0.0, min(age, 64) / 64.0)
                    AK[ci, k] = min(max(kind, 0), 1)
                    AB[ci, k] = min(max(beh, 0), 4)
                    AP[ci, k] = x * 19 + y
                    AM[ci, k] = True
                    k += 1
        if (gi + 1) % 1000 == 0:
            print(f"  [{gi+1}/{len(ents_files)}]", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tower_cont=TC, tower_type=TT, tower_pos=TP, tower_mask=TM,
        ant_cont=AC, ant_kind=AK, ant_behavior=AB, ant_pos=AP, ant_mask=AM,
        stats=cstats.astype(np.float16), value=clabel.astype(np.float32),
        seed=seed,
    )
    print(f"[ent] done -> {out}  max_towers={int(TM.sum(1).max())} max_ants={int(AM.sum(1).max())}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
