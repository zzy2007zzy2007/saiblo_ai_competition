#!/usr/bin/env python3
"""自对弈原始 dump → 亚军价值蒸馏数据集。

输入（同一个决策在两处各写一条，**按出现顺序一一对齐**）：
  - `AZAI_DUMP_RAW_DIR` 的 `obs_seedXXXXX_pP.bin`：board(28*19*19 f16)+stats(42 f16)+player(u8)+chosen(3)+cmask(24)；
  - `AZAI_AZ_DUMP` 的 `az_seedXXXXX_pP.jsonl`：每行一条决策，含 `n/seed/player/state`（原始塔/蚁/金币/基地）。

对每条：用 `state` 构 `RawState` → 亚军 `evaluate_terminal(state, player)` 当标签。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/distill_top2/postprocess_selfplay_dump.py \
      --bin-dir training_history/distill_top2/sp_bin --jsonl-dir training_history/distill_top2/sp_jsonl \
      --out training_history/distill_top2/value_ds_selfplay.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code", _REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from SDK.utils.constants import (  # noqa: E402
    STRATEGIC_BUILD_ORDER,
    TOWER_STATS,
    TowerType,
)
from my_ai.distill_top2.lure_value_eval import evaluate_terminal  # noqa: E402

BOARD_SHAPE = (28, 19, 19)
N_CLS = 24
REC_BYTES = int(np.prod(BOARD_SHAPE)) * 2 + 42 * 2 + 1 + 3 + N_CLS


# ── RawState：把原始 dict 包成 evaluate_terminal 能吃的接口 ─────────────────────
class RawBase:
    def __init__(self, hp: int):
        self.hp = int(hp)
        self.generation_level = 0
        self.ant_level = 0


class RawTower:
    def __init__(self, d):
        self.tower_id, self.player, self.x, self.y, self.tower_type, self.hp = (int(v) for v in d)
        self.max_hp = int(TOWER_STATS[TowerType(self.tower_type)].max_hp)
        self.level = 0

    def is_alive(self) -> bool:
        return self.hp > 0


class RawAnt:
    def __init__(self, d):
        self.ant_id, self.player, self.x, self.y, self.hp, self.max_hp, self.kind, self.age, self.behavior = (
            int(v) for v in d)

    def is_alive(self) -> bool:
        return self.hp > 0


class RawState:
    def __init__(self, raw: dict):
        self.round_index = int(raw["round"])
        self.coins = [int(c) for c in raw["coins"]]
        self.bases = [RawBase(raw["bases_hp"][0]), RawBase(raw["bases_hp"][1])]
        self.towers = [RawTower(t) for t in raw["towers"]]
        self.ants = [RawAnt(a) for a in raw["ants"]]

    def towers_of(self, p: int):
        return [t for t in self.towers if t.player == p]

    def tower_count(self, p: int) -> int:
        return len(self.towers_of(p))

    def tower_at(self, x: int, y: int):
        for t in self.towers:
            if t.x == x and t.y == y:
                return t
        return None

    def strategic_slots(self, p: int):
        return STRATEGIC_BUILD_ORDER[p]


# ── 读取 ────────────────────────────────────────────────────────────────────
def load_bin(path: Path):
    n_rec = path.stat().st_size // REC_BYTES
    if n_rec == 0:
        return None
    buf = np.fromfile(path, dtype=np.uint8)[: n_rec * REC_BYTES].reshape(n_rec, REC_BYTES)
    bsz = int(np.prod(BOARD_SHAPE)) * 2
    board = buf[:, :bsz].copy().view(np.float16).reshape(n_rec, *BOARD_SHAPE)
    stats = buf[:, bsz:bsz + 84].copy().view(np.float16).reshape(n_rec, 42)
    pl = buf[:, bsz + 84].astype(np.int64)
    return board, stats, pl


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_seed_player(name: str):
    # obs_seed00100_p0.bin  /  az_seed00100_p0.jsonl
    parts = name.replace(".", "_").split("_")
    seed = int(parts[1].replace("seed", ""))
    player = int(parts[2][1:])
    return seed, player


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--out", default="training_history/distill_top2/value_ds_selfplay.npz")
    ap.add_argument("--limit-files", type=int, default=0, help=">0 时只处理前 N 个 bin（冒烟用）")
    ap.add_argument("--stride", type=int, default=1, help="每 N 条采一条")
    a = ap.parse_args()

    bin_dir = Path(a.bin_dir)
    if not bin_dir.is_absolute():
        bin_dir = _REPO / bin_dir
    jsonl_dir = Path(a.jsonl_dir)
    if not jsonl_dir.is_absolute():
        jsonl_dir = _REPO / jsonl_dir
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out

    bins = sorted(bin_dir.glob("obs_seed*_p*.bin"))
    if a.limit_files > 0:
        bins = bins[: a.limit_files]
    print(f"[post] {len(bins)} 个 .bin，jsonl_dir={jsonl_dir}", flush=True)

    boards, stats_list, players, values, seeds, rounds = [], [], [], [], [], []
    n_pair_mismatch = n_skip = 0
    for bi, bf in enumerate(bins):
        seed, player = _parse_seed_player(bf.name)
        jf = jsonl_dir / f"az_seed{seed:05d}_p{player}.jsonl"
        if not jf.exists():
            print(f"  [skip] 无对应 jsonl: {bf.name}", flush=True)
            continue
        bdata = load_bin(bf)
        jrows = load_jsonl(jf)
        if bdata is None or not jrows:
            continue
        board, stats, pl = bdata
        n = min(len(board), len(jrows))
        if len(board) != len(jrows):
            n_pair_mismatch += 1
        kept = 0
        for i in range(0, n, a.stride):
            row = jrows[i]
            raw = row.get("state")
            if not raw:
                n_skip += 1
                continue
            try:
                val = float(evaluate_terminal(RawState(raw), int(pl[i])))
            except Exception as _e:  # noqa: BLE001
                n_skip += 1
                continue
            boards.append(board[i])
            stats_list.append(stats[i])
            players.append(int(pl[i]))
            values.append(val)
            seeds.append(seed)
            rounds.append(int(raw.get("round", row.get("n", 0))))
            kept += 1
        print(f"  [{bi+1}/{len(bins)}] {bf.name}: bin={len(board)} jsonl={len(jrows)} kept={kept}",
              flush=True)

    if not values:
        print("[post] 没有样本！", flush=True)
        return 1

    board_a = np.stack(boards).astype(np.float16)
    stats_a = np.stack(stats_list).astype(np.float32)
    value_a = np.array(values, dtype=np.float32)
    seeds_a = np.array(seeds, dtype=np.int32)
    rounds_a = np.array(rounds, dtype=np.int16)
    players_a = np.array(players, dtype=np.int8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, board=board_a, stats=stats_a, value=value_a,
                        seed=seeds_a, round=rounds_a, player=players_a)
    print(f"[post] saved {len(value_a)} samples -> {out} (board {board_a.shape})", flush=True)
    print(f"[post] value mean/std/min/max = {value_a.mean():.1f}/{value_a.std():.1f}/"
          f"{value_a.min():.1f}/{value_a.max():.1f}", flush=True)
    print(f"[post] round mean/max = {rounds_a.mean():.0f}/{int(rounds_a.max())} "
          f"games={len(set(seeds_a.tolist()))} pair_mismatch={n_pair_mismatch} skipped={n_skip}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
