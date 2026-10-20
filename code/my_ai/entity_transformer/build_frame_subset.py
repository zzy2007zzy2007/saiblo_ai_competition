#!/usr/bin/env python3
"""从全量合并数据里按"每局 K 帧"抽样，生成子集目录（board/stats/标签/frame_idx）。

为什么要单独写：`subsample_frames.py` 只处理 terminal 标签，而 rel/kgeo 标签需要
按**同一批帧索引**一起拷过来，才能保证 board 与各标签严格对齐。

抽样规则与 `subsample_frames.py` 一致（同 seed、同均匀分段），
所以 K=64 时生成的帧索引与已有 `lowq_f64` 完全相同。

用法：
  python code/my_ai/entity_transformer/build_frame_subset.py \
      --src D:/.../lowq_full --out D:/.../lowq_f256 --frames-per-game 256 \
      --extra-label label_geo_tau10.npy --extra-label label_kgeo_tau50.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="全量合并目录（含 board/stats/game_offsets）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-game", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--extra-label", nargs="*", default=[],
                    help="额外要一并抽取的标签文件名（如 label_geo_tau10.npy）")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    board = np.load(src / "board.npy", mmap_mode="r")
    stats = np.load(src / "stats.npy", mmap_mode="r")
    offs = np.load(src / "game_offsets.npy")
    term = np.load(src / "value_win.npy")
    hp_us = np.load(src / "hp_us.npy")
    hp_opp = np.load(src / "hp_opp.npy")
    extras = {n: np.load(src / n) for n in a.extra_label}
    n_games = len(offs) - 1
    K = a.frames_per_game
    total = n_games * K
    print(f"[sub] {n_games} games x {K} = {total} samples; extras={list(extras)}", flush=True)

    # ① 先算帧索引（与 subsample_frames.py 同规则）
    rng = np.random.default_rng(a.seed)
    fidx = np.empty(total, dtype=np.int64)
    w = 0
    per_game = []
    for g in range(n_games):
        s0, e0 = int(offs[g]), int(offs[g + 1])
        n = e0 - s0
        if n <= K:
            idx = np.arange(s0, e0)
        else:
            edges = np.linspace(0, n, K + 1).astype(np.int64)
            idx = np.array([s0 + int(rng.integers(edges[k], edges[k + 1])) for k in range(K)])
        fidx[w:w + len(idx)] = idx
        per_game.append(len(idx))
        w += len(idx)
    fidx = fidx[:w]
    total = w
    np.save(out / "frame_idx.npy", fidx)
    print(f"[sub] frame_idx built: {total} rows", flush=True)

    def alloc(name, shape, dtype):
        return np.lib.format.open_memmap(out / name, mode="w+", dtype=dtype, shape=shape)

    b_out = alloc("board.npy", (total, 28, 19, 19), np.float16)
    s_out = alloc("stats.npy", (total, 42), np.float16)
    l_out = alloc("label.npy", (total,), np.float32)
    g_out = alloc("game_id.npy", (total,), np.int32)
    hu_out = alloc("hp_us.npy", (total,), np.int16)
    ho_out = alloc("hp_opp.npy", (total,), np.int16)
    ex_out = {n: alloc(n, (total,), np.float32) for n in extras}

    # ② 逐局：整窗口顺序读，再按索引挑（避免 HDD/SSD 上的散读）
    w = 0
    for g in range(n_games):
        s0, e0 = int(offs[g]), int(offs[g + 1])
        m = per_game[g]
        idx = fidx[w:w + m]
        li = idx - s0
        win_b = np.array(board[s0:e0])
        win_s = np.array(stats[s0:e0])
        b_out[w:w + m] = win_b[li]
        s_out[w:w + m] = win_s[li]
        l_out[w:w + m] = term[idx]
        g_out[w:w + m] = g
        hu_out[w:w + m] = hp_us[idx]
        ho_out[w:w + m] = hp_opp[idx]
        for n, arr in extras.items():
            ex_out[n][w:w + m] = arr[idx]
        w += m
        del win_b, win_s
        if (g + 1) % 500 == 0:
            with open(out / "progress.txt", "a", encoding="utf-8") as pf:
                pf.write(f"games={g+1}/{n_games} rows={w}\n")
            print(f"  [{g+1}/{n_games}] rows={w}", flush=True)

    for arr in [b_out, s_out, l_out, g_out, hu_out, ho_out, *ex_out.values()]:
        arr.flush()
    (out / "meta.json").write_text(json.dumps(
        {"games": n_games, "frames_per_game": K, "samples": int(w),
         "extras": list(extras), "src": str(src)}, indent=1), encoding="utf-8")
    print(f"[sub] done -> {out} ({w} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
