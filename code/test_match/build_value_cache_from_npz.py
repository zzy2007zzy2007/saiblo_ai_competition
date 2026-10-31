#!/usr/bin/env python
"""把 `az_selfplay --write-npz` 产出的**每局一个 npz** 目录，压成 `train_value_net.py` 用的紧凑 memmap 缓存。

为什么需要：`train_value_net.py` 的 `--cache` 只要四个 memmap + `meta.npz`（`board/stats/player/value_target`
＋每局行数），而它自带的 `--build-cache-only` 只吃 **pkl**。采集脚本写的是 npz（值网格式，更轻），
所以这里做一次"npz 目录 → cache 目录"的转换，之后训练/评估直接 `--cache <out>`，不用改训练器。

输出（与 `train_value_net.py:97-118` 完全同构）：
  board.npy  (N,28,19,19) float16
  stats.npy  (N,42)       float16
  player.npy (N,)         int8
  value_target.npy (N,)   float32        ← = 采集时的 terminal 标签（局末血差/20，按 player 定符号）
  meta.npz   counts(N_games,) int64 / names(N_games,) str

用法：
  python code/test_match/build_value_cache_from_npz.py \
      --src training_history/collect_v2_s1 --out training_history/value_retrain_v1/cache_all
  # 只取前 N 局（按文件名排序）：
  ... --limit 112
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="npz 目录（每局一个 az_selfplay_seed*.npz）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-keys", default="board,stats,player,value_target")
    a = ap.parse_args()
    src = Path(a.src) if Path(a.src).is_absolute() else REPO / a.src
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    req = [k.strip() for k in a.require_keys.split(",") if k.strip()]

    files = sorted(src.glob("az_selfplay_seed*.npz"))
    if a.limit:
        files = files[:a.limit]
    # 先探测形状 + 过滤坏文件/缺键
    ok, shapes, skipped = [], None, []
    for f in files:
        try:
            z = np.load(f)
            if any(k not in z.files for k in req):
                skipped.append((f.name, "缺键 " + ",".join(k for k in req if k not in z.files)))
                continue
            b = z["board"]
            s = z["stats"]
            if b.ndim != 4 or s.ndim != 2 or b.shape[0] != s.shape[0] or b.shape[0] < 8:
                skipped.append((f.name, f"形状异常 board={b.shape} stats={s.shape}"))
                continue
            if shapes is None:
                shapes = (b.shape[1:], s.shape[1])
            elif (b.shape[1:], s.shape[1]) != shapes:
                skipped.append((f.name, f"形状不一致 {b.shape[1:]} vs {shapes}"))
                continue
            ok.append(f)
        except Exception as exc:                                  # noqa: BLE001
            skipped.append((f.name, f"{type(exc).__name__}: {exc}"))
    if not ok:
        raise SystemExit("没有可用 npz")
    counts = [int(np.load(f)["board"].shape[0]) for f in ok]
    total = int(sum(counts))
    bshape, sdim = shapes
    print(f"[cache] {len(ok)} 局 / {total} 个决策点；board=({total},)+{bshape} stats=({total},{sdim})")
    if skipped:
        print(f"[cache] ⚠️ 跳过 {len(skipped)} 个：{skipped[:5]}")

    C, H, W = bshape
    out.mkdir(parents=True, exist_ok=True)
    mm_b = np.lib.format.open_memmap(out / "board.npy", mode="w+", dtype=np.float16,
                                     shape=(total, C, H, W))
    mm_s = np.lib.format.open_memmap(out / "stats.npy", mode="w+", dtype=np.float16,
                                     shape=(total, sdim))
    mm_p = np.lib.format.open_memmap(out / "player.npy", mode="w+", dtype=np.int8,
                                     shape=(total,))
    mm_v = np.lib.format.open_memmap(out / "value_target.npy", mode="w+", dtype=np.float32,
                                     shape=(total,))
    off = 0
    for i, f in enumerate(ok):
        z = np.load(f)
        n = counts[i]
        mm_b[off:off + n] = np.asarray(z["board"], dtype=np.float16)
        mm_s[off:off + n] = np.asarray(z["stats"], dtype=np.float16)
        mm_p[off:off + n] = np.asarray(z["player"], dtype=np.int8)
        mm_v[off:off + n] = np.asarray(z["value_target"], dtype=np.float32)
        off += n
        if (i + 1) % 25 == 0:
            print(f"  [cache] {i + 1}/{len(ok)}（{off}/{total} 行）", flush=True)
    for m in (mm_b, mm_s, mm_p, mm_v):
        m.flush()
    np.savez(out / "meta.npz",
             counts=np.asarray(counts, dtype=np.int64),
             names=np.asarray([f.stem.replace("az_selfplay_", "") for f in ok]))
    # 顺手记一份来源清单（便于复现/排查）
    (out / "source.json").write_text(json.dumps(
        {"src": str(src), "n_games": len(ok), "n_rows": total,
         "games": [f.name for f in ok], "skipped": skipped}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[cache] done -> {out}（{total} 行，{len(ok)} 局）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
