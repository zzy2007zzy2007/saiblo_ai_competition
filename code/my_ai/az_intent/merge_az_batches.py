"""Merge per-game intent-space npz into ONE ``merged_<group>.npz`` per batch.

Why: a large value pool (thousands of games) must be STREAMED — ``np.load`` over
thousands of tiny files is dominated by open/header overhead.  Following the
design in ``docs/value_data_merge_and_train.md`` we merge **per batch** (keeps
provenance: which round/batch a game came from) rather than by an arbitrary fixed
chunk size; the streaming trainer shuffles the file order every epoch, so batches
still mix during training.

Constraint: a merged file is loaded fully into RAM, and a decompressed game is
~32 MB (board 17 MB + action_map 15 MB).  Keep each batch <= ~50 games
(50 x 32 MB ~ 1.6 GB).  The iterative loop therefore writes 500-game rounds as
10 sub-batches rather than one.

Keys are preserved 1:1 with the per-game schema:
    board, stats, action_map, head_logits, player, value_target

Two source layouts:
  --layout dirs : group by the immediate subdirectory name (batch1/ ... )
  --layout flat : group by the seed prefix ``seed // --seed-div``
                  (az_selfplay seeds are ``base*10000 + game``, so the batch base
                   survives in the leading digits: 400000000 / 420000000 / ...;
                   warm_polonly was converted flat, so its 21 batches are only
                   recoverable this way)

Usage:
    python code/my_ai/az_intent/merge_az_batches.py --src <dir> --out <dir> \
        --layout flat --seed-div 10000000
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_KEYS = ("board", "stats", "action_map", "head_logits", "player", "value_target")
_SEED_RE = re.compile(r"seed(\d+)\.npz$")


def _group_of(path: Path, src: Path, layout: str, seed_div: int) -> str:
    if layout == "dirs":
        return path.parent.name if path.parent != src else "root"
    m = _SEED_RE.search(path.name)
    if not m:
        raise SystemExit(f"--layout flat 需要文件名形如 *_seed<digits>.npz，收到 {path.name}")
    return f"seed{int(m.group(1)) // seed_div:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-game az npz into one file per batch")
    ap.add_argument("--src", required=True, help="directory with per-game .npz (flat or nested)")
    ap.add_argument("--out", default=None, help="output dir (default: <src>/merged)")
    ap.add_argument("--layout", choices=["dirs", "flat"], default="dirs")
    ap.add_argument("--seed-div", type=int, default=10_000_000,
                    help="flat layout: group = seed // this (default 1e7)")
    ap.add_argument("--min-games", type=int, default=1,
                    help="skip groups with fewer games (default 1)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src / "merged"
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(src.rglob("*.npz"))
    paths = [p for p in paths if not p.name.startswith("merged_")]
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[_group_of(p, src, args.layout, args.seed_div)].append(p)

    print(f"[merge] {len(paths)} per-game npz -> {len(groups)} groups -> {out}")
    t0 = time.perf_counter()
    total_games = total_frames = 0
    for gname in sorted(groups):
        gps = sorted(groups[gname])
        if len(gps) < args.min_games:
            print(f"  [skip] {gname}: {len(gps)} games < min {args.min_games}")
            continue
        acc: dict[str, list[np.ndarray]] = {k: [] for k in _KEYS}
        n_frames = 0
        for p in gps:
            d = np.load(p)
            for k in _KEYS:
                if k in d.files:
                    acc[k].append(d[k])
            n_frames += int(d["board"].shape[0])
            d.close()
        out_path = out / f"merged_{gname}.npz"
        np.savez_compressed(out_path, **{k: np.concatenate(v, axis=0) for k, v in acc.items() if v})
        mib = out_path.stat().st_size / 1024 / 1024
        total_games += len(gps)
        total_frames += n_frames
        print(f"  {gname:>10s}: {len(gps):4d} games  {n_frames:6d} frames  {mib:7.0f}MB  "
              f"-> {out_path.name}", flush=True)
    print(f"[merge] done {time.perf_counter()-t0:.0f}s  games={total_games} frames={total_frames}")


if __name__ == "__main__":
    main()
