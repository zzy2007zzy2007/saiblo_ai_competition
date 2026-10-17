#!/usr/bin/env python3
"""解析桥接的 raw-state dump（`MVS_RAW_STATE_DIR`）→ 亚军价值蒸馏数据集。

记录格式（每条 = 一个玩家在一个回合的决策前状态，定长小端）：
    board: 28*19*19 个 float16
    stats: 42 个 float16
    player: 1 个 uint8
    round: 1 个 int16
    value: 1 个 float32   # = 亚军 evaluate_terminal(state, player)

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/distill_top2/postprocess_bridge_raw.py \
      --raw-dir training_history/distill_top2/ru_selfplay_raw \
      --out training_history/distill_top2/value_ds_runnerup_selfplay.npz
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

BOARD_SHAPE = (28, 19, 19)
BOARD_BYTES = int(np.prod(BOARD_SHAPE)) * 2      # 20216
STATS_BYTES = 42 * 2                              # 84
REC_BYTES = BOARD_BYTES + STATS_BYTES + 1 + 2 + 4  # player u8 + round i16 + value f32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="training_history/distill_top2/value_ds_runnerup_selfplay.npz")
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    if not raw.is_absolute():
        raw = _REPO / raw
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out

    files = sorted(raw.glob("raw_seed*.bin"))
    print(f"[post] {len(files)} 个 .bin，raw_dir={raw}", flush=True)

    boards, stats_list, players, values, seeds, rounds = [], [], [], [], [], []
    for bi, f in enumerate(files):
        seed = int(f.stem.replace("raw_seed", ""))
        n_rec = f.stat().st_size // REC_BYTES
        if n_rec == 0:
            continue
        buf = np.fromfile(f, dtype=np.uint8)[: n_rec * REC_BYTES].reshape(n_rec, REC_BYTES)
        board = buf[:, :BOARD_BYTES].copy().view(np.float16).reshape(n_rec, *BOARD_SHAPE)
        off = BOARD_BYTES
        stats = buf[:, off:off + STATS_BYTES].copy().view(np.float16).reshape(n_rec, 42)
        off += STATS_BYTES
        pl = buf[:, off].astype(np.int64)
        off += 1
        rnd = buf[:, off:off + 2].copy().view(np.int16).reshape(n_rec)
        off += 2
        val = buf[:, off:off + 4].copy().view(np.float32).reshape(n_rec)
        boards.append(board)
        stats_list.append(stats)
        players.append(pl)
        rounds.append(rnd)
        values.append(val)
        seeds.append(np.full(n_rec, seed, dtype=np.int32))
        print(f"  [{bi+1}/{len(files)}] {f.name}: {n_rec} 条", flush=True)

    if not values:
        print("[post] 没有样本！", flush=True)
        return 1

    board_a = np.concatenate(boards).astype(np.float16)
    stats_a = np.concatenate(stats_list).astype(np.float32)
    player_a = np.concatenate(players).astype(np.int8)
    round_a = np.concatenate(rounds).astype(np.int16)
    value_a = np.concatenate(values).astype(np.float32)
    seed_a = np.concatenate(seeds).astype(np.int32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, board=board_a, stats=stats_a, value=value_a,
                        seed=seed_a, round=round_a, player=player_a)
    print(f"[post] saved {len(value_a)} 样本 -> {out} (board {board_a.shape})", flush=True)
    print(f"[post] value mean/std/min/max = {value_a.mean():.1f}/{value_a.std():.1f}/"
          f"{value_a.min():.1f}/{value_a.max():.1f}", flush=True)
    print(f"[post] round mean/max = {round_a.mean():.0f}/{int(round_a.max())} "
          f"games={len(set(seed_a.tolist()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
