#!/usr/bin/env python3
"""生成「亚军 value 函数」蒸馏数据集：每个样本 = (board, stats) + 亚军 evaluate_terminal 标签。

- 价值标签：**亚军** cpp_lure_v4 的 `evaluate_terminal`（`lure_value_eval.py` 逐条复刻）。
- 状态来源：**C++ 引擎**的 `GameStateFacade`（权威；不用官方 Python 引擎）。
- 策略：简单 build/upgrade（只为产生多样的对局状态，不追求强度）。

用法（训练环境 python）:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/distill_top2/gen_value_dataset.py \
      --seeds 100-109 --max-rounds 512 --out training_history/distill_top2/value_ds.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code", _REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.add_dll_directory(str(_REPO / "code" / "cpp_engine"))
os.add_dll_directory(r"C:/mingw64/bin")

from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import (  # noqa: E402
    MAX_ACTIONS,
    MAX_ROUND,
    MAP_SIZE,
    TOWER_UPGRADE_TREE,
    OperationType,
    TowerType,
)
from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.distill_top2.lure_value_eval import evaluate_terminal, eval_breakdown  # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


MODES = [
    ("aggressive", 0.95, 0.45),
    ("moderate", 0.70, 0.30),
    ("passive", 0.35, 0.15),
]


def _mode_params(rng: np.random.Generator) -> tuple[float, float]:
    """随机抽一个策略档位 (build_rate, upgrade_rate) —— 制造局面多样性。"""
    mode = MODES[int(rng.integers(len(MODES)))]
    return (mode[1], mode[2])


def simple_policy(state, player: int, rng: np.random.Generator,
                  build_rate: float = 0.85, upgrade_rate: float = 0.35):
    """简单策略：空的核心槽位上建塔，偶尔升级；非法/买不起的函数会自己拒绝。
    build_rate/upgrade_rate 由调用方按"策略档位"传入（造多样性）。"""
    ops = []
    try:
        if rng.random() < build_rate:
            for (x, y) in state.strategic_slots(player):
                if state.tower_at(x, y) is not None:
                    continue
                op = Operation(OperationType.BUILD_TOWER, x, y)
                if state.can_apply_operation(player, op, ops):
                    ops.append(op)
                    break
        if rng.random() < upgrade_rate and len(ops) < 2:
            # 随机选一个合法升级目标（覆盖 level-2 / level-3，让回收价值项有多样性）
            candidates = [t for t in state.towers_of(player)
                          if len(TOWER_UPGRADE_TREE.get(TowerType(int(t.tower_type)), ())) > 0]
            if candidates:
                tower = candidates[int(rng.integers(len(candidates)))]
                targets = TOWER_UPGRADE_TREE[TowerType(int(tower.tower_type))]
                target = targets[int(rng.integers(len(targets)))]
                op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
                if state.can_apply_operation(player, op, ops):
                    ops.append(op)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] policy error seed={state.round_index} player={player}: {exc}", flush=True)
        return []
    return ops


def _tower_maps(state):
    """把塔的 hp 比例 / 精确类型 dump 成 2x19x19 辅助图（board 里没有这两项）。
    存下来是为了：若蒸馏失败，能离线加特征再测，不用重跑对局。"""
    hp_map = np.zeros((2, MAP_SIZE, MAP_SIZE), dtype=np.float16)
    type_map = np.zeros((2, MAP_SIZE, MAP_SIZE), dtype=np.int16)
    for tower in state.towers:
        p = int(tower.player)
        x, y = int(tower.x), int(tower.y)
        max_hp = max(int(tower.max_hp), 1)
        hp_map[p, x, y] = np.float16(max(int(tower.hp), 0) / max_hp)
        type_map[p, x, y] = np.int16(int(tower.tower_type))
    return hp_map, type_map


def run_game(seed: int, fe: FeatureExtractor, rng: np.random.Generator,
             max_rounds: int, stride: int,
             policies: list[tuple[float, float]]) -> list[dict]:
    state = GameStateFacade.initial(seed=seed)
    zero_mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    rows: list[dict] = []
    while not state.terminal and state.round_index < max_rounds:
        if state.round_index % stride == 0:
            tower_hp, tower_type = _tower_maps(state)
            for p in (0, 1):
                obs = fe.encode_observation(state, p, zero_mask)
                rows.append({
                    "seed": seed,
                    "round": int(state.round_index),
                    "player": p,
                    "board": obs["board"].astype(np.float16),
                    "stats": obs["stats"].astype(np.float32),
                    "tower_hp": tower_hp,
                    "tower_type": tower_type,
                    "value": float(evaluate_terminal(state, p)),
                })
        for p in (0, 1):
            ops = simple_policy(state, p, rng, policies[p][0], policies[p][1])
            if ops:
                state.apply_operation_list(p, ops)
        state.advance_round()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-104", help="种子区间，如 100-109 或 1,3,5-7")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUND)
    ap.add_argument("--stride", type=int, default=1, help="每 N 回合采一次样")
    ap.add_argument("--out", default="training_history/distill_top2/value_ds.npz")
    ap.add_argument("--smoke", action="store_true", help="只跑第一个种子并打印 breakdown")
    ap.add_argument("--max-games", type=int, default=0, help=">0 时只跑前 N 个种子")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.smoke:
        seeds = seeds[:1]
    if args.max_games > 0:
        seeds = seeds[: args.max_games]

    print(f"[gen] seeds={seeds} max_rounds={args.max_rounds} stride={args.stride} out={args.out}",
          flush=True)

    fe = FeatureExtractor(max_actions=MAX_ACTIONS)
    rng = np.random.default_rng(12345)
    all_rows: list[dict] = []
    for i, seed in enumerate(seeds):
        policies = [_mode_params(rng) for _ in range(2)]
        rows = run_game(seed, fe, rng, args.max_rounds, args.stride, policies)
        max_tw = int(max(((r["tower_hp"] > 0).sum()) for r in rows)) if rows else 0
        print(f"[policy] seed={seed} policies={policies} max_towers={max_tw}", flush=True)
        vals = np.array([r["value"] for r in rows], dtype=np.float32) if rows else np.array([], dtype=np.float32)
        print(f"[gen] seed={seed} ({i+1}/{len(seeds)}) samples={len(rows)} "
              f"value min/med/max={vals.min() if len(vals) else float('nan'):.1f}/"
              f"{np.median(vals) if len(vals) else float('nan'):.1f}/"
              f"{vals.max() if len(vals) else float('nan'):.1f}", flush=True)
        all_rows.extend(rows)

    if not all_rows:
        print("[gen] 没有样本！", flush=True)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    board = np.stack([r["board"] for r in all_rows]).astype(np.float16)
    stats = np.stack([r["stats"] for r in all_rows]).astype(np.float32)
    tower_hp = np.stack([r["tower_hp"] for r in all_rows]).astype(np.float16)
    tower_type = np.stack([r["tower_type"] for r in all_rows]).astype(np.int16)
    value = np.array([r["value"] for r in all_rows], dtype=np.float32)
    seed_arr = np.array([r["seed"] for r in all_rows], dtype=np.int32)
    round_arr = np.array([r["round"] for r in all_rows], dtype=np.int16)
    player_arr = np.array([r["player"] for r in all_rows], dtype=np.int8)
    np.savez_compressed(out, board=board, stats=stats, tower_hp=tower_hp,
                        tower_type=tower_type, value=value,
                        seed=seed_arr, round=round_arr, player=player_arr)
    print(f"[gen] saved {len(value)} samples -> {out} (board {board.shape}, stats {stats.shape})",
          flush=True)
    n_tw = (tower_hp > 0).sum(axis=(1, 2, 3))
    print(f"[coverage] towers/sample mean={n_tw.mean():.1f} p50={np.median(n_tw):.0f} "
          f"p95={np.percentile(n_tw, 95):.0f} max={int(n_tw.max())}", flush=True)
    print(f"[coverage] round mean={float(round_arr.mean()):.0f} max={int(round_arr.max())} "
          f"games={len(set(seed_arr.tolist()))}", flush=True)

    if args.smoke:
        resid = value - value.mean()
        print(f"  smoke value: mean={value.mean():.2f} std={value.std():.2f} "
              f"p5={np.percentile(value,5):.1f} p95={np.percentile(value,95):.1f}", flush=True)
        print(f"  smoke 'interesting' residual std={resid.std():.2f} "
              f"(base_hp*200 dominate ~10000)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
