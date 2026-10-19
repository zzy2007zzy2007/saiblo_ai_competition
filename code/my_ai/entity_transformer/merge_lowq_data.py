#!/usr/bin/env python3
"""把 5000 个低质量对局的 npz + ents.jsonl **合并成大数组**（memmap 预分配）。

为什么合并：训练时流式读取，若每次随机打开 5000 个小 npz（每个 ~10 MB），
10 个 epoch 就是 5 万次小文件随机 I/O。合并成几个大 `.npy` 后，`np.memmap`
按页缓存/顺序读，快一个数量级。

输出（目录 `<src>_merged`）：
  board.npy     (N, 28, 19, 19) float16
  stats.npy     (N, 42) float16
  player.npy    (N,) int8
  value_win.npy (N,) float32          # ±0.5 终局胜负
  hp_us.npy / hp_opp.npy (N,) int16   # 原始基地血量，可导出 HP 差标签
  round.npy     (N,) int16
  tower_type.npy (N,32) int16, tower_pos.npy (N,32) int16, tower_mask.npy (N,32) bool
  tower_hp.npy   (N,32) float16       # hp/max_hp
  ant_kind.npy   (N,64) int8, ant_pos.npy (N,64) int16, ant_mask.npy (N,64) bool
  ant_cont.npy   (N,64,4) float16     # is_own, hp/max_hp, level/2, age/64（敌方蚁=0.. ）
  meta.json      # 样本数 / 各文件 shape / 来源

用法：
  python code/my_ai/entity_transformer/merge_lowq_data.py \
      --src "F:/2026智能体大赛_新2/training_history/lowq_e2" \
      --out "F:/2026智能体大赛_新2/training_history/lowq_e2_merged"
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

from my_ai.entity_transformer.model import TOWER_TYPE_TO_ID  # noqa: E402

MAX_TW = 32
MAX_ANT = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    npzs = sorted(src.glob("*.npz"))
    print(f"[merge] {len(npzs)} npz", flush=True)

    # pass 1：统计总数。**不要用 d["board"].shape** —— 那会触发整块解压
    # （5000 × 14 MB ≈ 70 GB）；直接读 npz(zip) 头里的未压缩字节数即可。
    import zipfile
    _REC = 28 * 19 * 19 * 2  # board 每条字节数（float16）
    total = 0
    for p in npzs:
        with zipfile.ZipFile(p) as zf:
            total += zf.getinfo("board.npy").file_size // _REC
    print(f"[merge] total samples = {total}", flush=True)

    def alloc(name, shape, dtype):
        return np.lib.format.open_memmap(out / name, mode="w+", dtype=dtype, shape=shape)

    board = alloc("board.npy", (total, 28, 19, 19), np.float16)
    stats = alloc("stats.npy", (total, 42), np.float16)
    player = alloc("player.npy", (total,), np.int8)
    vwin = alloc("value_win.npy", (total,), np.float32)
    hp_us = alloc("hp_us.npy", (total,), np.int16)
    hp_opp = alloc("hp_opp.npy", (total,), np.int16)
    rnd = alloc("round.npy", (total,), np.int16)
    tw_type = alloc("tower_type.npy", (total, MAX_TW), np.int16)
    tw_pos = alloc("tower_pos.npy", (total, MAX_TW), np.int16)
    tw_mask = alloc("tower_mask.npy", (total, MAX_TW), np.bool_)
    tw_hp = alloc("tower_hp.npy", (total, MAX_TW), np.float16)
    an_kind = alloc("ant_kind.npy", (total, MAX_ANT), np.int8)
    an_pos = alloc("ant_pos.npy", (total, MAX_ANT), np.int16)
    an_mask = alloc("ant_mask.npy", (total, MAX_ANT), np.bool_)
    an_cont = alloc("ant_cont.npy", (total, MAX_ANT, 4), np.float16)

    off = 0
    starts = [0]
    for i, p in enumerate(npzs):
        ents_path = p.with_suffix(".ents.jsonl")
        with np.load(p, mmap_mode="r") as d:
            n = d["board"].shape[0]
            board[off:off + n] = d["board"]
            stats[off:off + n] = d["stats"]
            player[off:off + n] = d["player"]
            vwin[off:off + n] = d["value_target"]
        rows = []
        if ents_path.exists():
            with open(ents_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        for j in range(n):
            r = rows[j] if j < len(rows) else {}
            hp_us[off + j] = int(r.get("hp_us", 0))
            hp_opp[off + j] = int(r.get("hp_opp", 0))
            rnd[off + j] = int(r.get("round", 0))
            # 我方塔（player 那一方）
            pl = int(r.get("player", 0))
            k = 0
            for t in (r.get("towers") or []):
                tid, owner, x, y, ttype, hp = (int(v) for v in t[:6])
                if owner != pl or k >= MAX_TW:
                    continue
                tw_type[off + j, k] = TOWER_TYPE_TO_ID.get(ttype, 16)
                tw_pos[off + j, k] = x * 19 + y
                tw_mask[off + j, k] = True
                tw_hp[off + j, k] = hp / (10.0 if ttype == 0 else 15.0)
                k += 1
            # 敌方蚁
            k = 0
            for ant in (r.get("ants") or []):
                aid, owner, x, y, hp, max_hp, kind, age, beh = (int(v) for v in ant[:9])
                if owner == pl or k >= MAX_ANT:
                    continue
                an_kind[off + j, k] = min(max(kind, 0), 1)
                an_pos[off + j, k] = x * 19 + y
                an_mask[off + j, k] = True
                an_cont[off + j, k] = (0.0, hp / max(max_hp, 1), kind / 2.0,
                                       age / 64.0)
                k += 1
        off += n
        starts.append(off)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(npzs)}] off={off}", flush=True)
        # 释放页缓存压力
        if (i + 1) % 500 == 0:
            for arr in (board, stats, player, vwin, hp_us, hp_opp, rnd,
                        tw_type, tw_pos, tw_mask, tw_hp, an_kind, an_pos, an_mask, an_cont):
                arr.flush()

    for arr in (board, stats, player, vwin, hp_us, hp_opp, rnd,
                tw_type, tw_pos, tw_mask, tw_hp, an_kind, an_pos, an_mask, an_cont):
        arr.flush()
    meta = {"total": int(total), "src": str(src), "max_towers": MAX_TW, "max_ants": MAX_ANT}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # 每局样本边界（len = 局数+1）：验证集必须**按局切**，否则同局样本泄漏
    np.save(out / "game_offsets.npy", np.asarray(starts, dtype=np.int64))
    print(f"[merge] done -> {out} ({total} samples, {len(starts)-1} games)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
