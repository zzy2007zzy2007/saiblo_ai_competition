#!/usr/bin/env python3
"""从合并数据里**逐局均匀抽帧**，生成紧凑训练集（放到 SSD，可全量进内存）。

为什么抽帧：
  * 标签是**每局一个值**（局末血量差/20）⇒ 独立样本数 = 局数，不是帧数；
  * 每局 856 帧全用是 ~13 倍冗余，而 board 又是磁盘大头；
  * 抽 64 帧/局 ⇒ 5000×64 = 32 万样本、board 仅 6.5 GB，可全量读进内存。

I/O 友好：逐局读 `board[a:b]` 这个 ~1.7 MB 的连续窗口（不是全量 86 GB），
再在窗口内挑帧 ⇒ 总共只读 ~局数×1.7MB（5000 局 ≈ 8.5 GB）。

用法：
  python code/my_ai/entity_transformer/subsample_frames.py \
      --src "F:/2026智能体大赛_新2/training_history/lowq_e2_merged" \
      --out "D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64" \
      --frames-per-game 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-game", type=int, default=64)
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    board = np.load(src / "board.npy", mmap_mode="r")
    offs = np.load(src / "game_offsets.npy")
    stats = np.load(src / "stats.npy", mmap_mode="r")
    vwin = np.load(src / "value_win.npy")
    hp_us = np.load(src / "hp_us.npy")
    hp_opp = np.load(src / "hp_opp.npy")
    n_games = len(offs) - 1
    K = a.frames_per_game
    total = n_games * K
    print(f"[sub] {n_games} games x {K} frames = {total} samples", flush=True)

    def alloc(name, shape, dtype):
        return np.lib.format.open_memmap(out / name, mode="w+", dtype=dtype, shape=shape)

    b_out = alloc("board.npy", (total, 28, 19, 19), np.float16)
    s_out = alloc("stats.npy", (total, 42), np.float16)
    y_out = alloc("label.npy", (total,), np.float32)
    g_out = alloc("game_id.npy", (total,), np.int32)
    hu_out = alloc("hp_us.npy", (total,), np.int16)
    ho_out = alloc("hp_opp.npy", (total,), np.int16)

    rng = np.random.default_rng(12345)
    w = 0
    for g in range(n_games):
        s0, e0 = int(offs[g]), int(offs[g + 1])
        n = e0 - s0
        if n <= K:
            idx = np.arange(s0, e0)
        else:
            # 均匀分段，每段内随机取 1 帧（避免只取到某个回合区间）
            edges = np.linspace(0, n, K + 1).astype(np.int64)
            idx = np.array([s0 + int(rng.integers(edges[k], edges[k + 1])) for k in range(K)])
        m = len(idx)
        li = idx - s0
        # ⚠️ 所有字段都**先整窗口顺序读进内存**，再从内存里挑帧。
        # 若直接 board[idx] / stats[idx] 散读，在机械盘上就是每局几十次寻道。
        win_b = np.asarray(board[s0:e0])
        win_s = np.asarray(stats[s0:e0])
        b_out[w:w + m] = win_b[li]
        s_out[w:w + m] = win_s[li]
        y_out[w:w + m] = vwin[idx]
        g_out[w:w + m] = g
        hu_out[w:w + m] = hp_us[idx]
        ho_out[w:w + m] = hp_opp[idx]
        del win_b, win_s
        w += m
        if (g + 1) % 500 == 0:
            with open(out / "progress.txt", "a", encoding="utf-8") as _pf:
                _pf.write(f"games={g+1}/{n_games} written={w}\n")
            print(f"  [{g+1}/{n_games}] written={w}", flush=True)

    for arr in (b_out, s_out, y_out, g_out, hu_out, ho_out):
        arr.flush()
    (out / "meta.json").write_text(json.dumps(
        {"games": n_games, "frames_per_game": K, "samples": int(w),
         "src": str(src)}, indent=1), encoding="utf-8")
    print(f"[sub] done -> {out} ({w} samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
