"""构建「状态实体张量 + 计划特征 → 8 回合期望值」数据集。

和 `build_lowq_entity_dataset.py` 的区别：
1. ⚠️ **两套 ants schema 并存**，按错的会被静默读错：
   - `F:\...\lowq_e2\*.ents.jsonl`（5000 局老数据）：**9 项**
     `[id, player, x, y, hp, max_hp, kind, age, behavior]` ⇒ `build_lowq_entity_dataset.py`
     用 `ant[:9]`、index 8 取 behavior，**对它是对的**。
   - `training_history/distill_top2/abl_linear2_raw/ents_*.jsonl`（`cpp_probe_match.dump_state` 产的）：**12 项**
     `[id, player, x, y, hp, max_hp, kind, age, level, behavior, status, frozen]`
     ⇒ 若拿旧脚本读这版，会把 **level 当 behavior**、behavior/status 全丢。
   本脚本按 **12 项** 写（index 6=kind,7=age,8=level,9=behavior），只适用于新 dump。
2. 输入除状态外还有**候选计划特征**（本条数据要预测的是"这个计划执行后 8 回合的期望分数"）。
3. 标签是 q8 数据集里的 `mean`（N 次 rollout 的均值），不是叶子评估。

输出 npz：
    tower_cont (N,48,3) float32 / tower_type (N,48) int16 / tower_pos (N,48) int16 / tower_mask
    ant_cont   (N,96,4) float32 / ant_kind (N,96) int8 / ant_behavior (N,96) int8
    ant_pos    (N,96) int16 / ant_mask
    state_feats (N,S) float32   （聚合状态特征，来自 build_q8_dataset 的 s_*）
    plan_feats  (N,P) float32   （计划特征 p_*）
    value (N,) float32 / std (N,) float32 / n (N,) int16
    heuristic (N,) float32 / is_hold (N,) bool
    group (N,) int64           （seed*1e6+round*10+player，用于按状态切分）

用法：
    python build_q8_entity_dataset.py \
        --q8 training_history/distill_top2/q8/q8_dataset_v2.jsonl \
        --ents-dir training_history/distill_top2/abl_linear2_raw \
        --out training_history/distill_top2/q8/q8_ent.npz
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

MAX_TOWER = 48
MAX_ANT = 96
SKIP = {"seed", "round", "player", "key", "ops", "mean", "std", "n"}


def build_state_tensors(state: dict, player: int):
    tower_cont = np.zeros((MAX_TOWER, 3), np.float32)
    tower_type = np.full((MAX_TOWER,), 16, np.int16)
    tower_pos = np.full((MAX_TOWER,), 361, np.int16)
    tower_mask = np.zeros((MAX_TOWER,), bool)
    ant_cont = np.zeros((MAX_ANT, 4), np.float32)
    ant_kind = np.full((MAX_ANT,), 2, np.int8)
    ant_behavior = np.full((MAX_ANT,), 5, np.int8)
    ant_pos = np.full((MAX_ANT,), 361, np.int16)
    ant_mask = np.zeros((MAX_ANT,), bool)

    slot = 0
    for row in state.get("towers") or []:
        if slot >= MAX_TOWER:
            break
        tower_id, owner, x, y, tower_type_value, hp = (int(v) for v in row[:6])
        tower_cont[slot] = (1.0 if owner == player else 0.0,
                            hp / (10.0 if tower_type_value == 0 else 15.0), 0.0)
        tower_type[slot] = TOWER_TYPE_TO_ID.get(tower_type_value, 16)
        tower_pos[slot] = x * 19 + y
        tower_mask[slot] = True
        slot += 1

    slot = 0
    for row in state.get("ants") or []:
        if slot >= MAX_ANT:
            break
        # [id, player, x, y, hp, max_hp, kind, age, level, behavior, status, frozen]
        _aid, owner, x, y, hp, max_hp = (int(v) for v in row[:6])
        kind, age, level, behavior = (int(v) for v in row[6:10])
        ant_cont[slot] = (1.0 if owner == player else 0.0,
                          hp / max(max_hp, 1), min(level, 2) / 2.0, min(age, 64) / 64.0)
        ant_kind[slot] = min(max(kind, 0), 1)
        ant_behavior[slot] = min(max(behavior, 0), 4)
        ant_pos[slot] = x * 19 + y
        ant_mask[slot] = True
        slot += 1
    return (tower_cont, tower_type, tower_pos, tower_mask,
            ant_cont, ant_kind, ant_behavior, ant_pos, ant_mask)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8", default="training_history/distill_top2/q8/q8_dataset_v2.jsonl")
    parser.add_argument("--ents-dir", default="training_history/distill_top2/abl_linear2_raw")
    parser.add_argument("--out", default="training_history/distill_top2/q8/q8_ent.npz")
    args = parser.parse_args()

    rows: list[dict] = []
    with open(args.q8, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(f"[q8] {len(rows)} 行")
    state_feature_names = [k for k in rows[0] if k.startswith("s_")]
    plan_feature_names = [k for k in rows[0] if k.startswith("p_")]
    print(f"[特征] state {len(state_feature_names)} 维 / plan {len(plan_feature_names)} 维")

    # 每个 (seed, round) 只读一次状态
    cache: dict[tuple[int, int], dict] = {}
    needed = sorted({(row["seed"], row["round"]) for row in rows})
    by_seed: dict[int, list[int]] = {}
    for seed, round_index in needed:
        by_seed.setdefault(seed, []).append(round_index)
    for seed, rounds in sorted(by_seed.items()):
        path = Path(args.ents_dir) / f"ents_seed{seed:05d}.jsonl"
        if not path.exists():
            candidates = list(Path(args.ents_dir).glob(f"ents_*{seed}.jsonl"))
            if not candidates:
                print(f"[warn] 缺少 seed={seed} 的状态文件")
                continue
            path = candidates[0]
        wanted = set(rounds)
        with open(path, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in wanted and line.strip():
                    cache[(seed, index)] = json.loads(line)
        print(f"[状态] seed={seed} 读入 {sum(1 for k in cache if k[0] == seed)} 个回合")

    count = len(rows)
    arrays = {
        "tower_cont": np.zeros((count, MAX_TOWER, 3), np.float32),
        "tower_type": np.full((count, MAX_TOWER), 16, np.int16),
        "tower_pos": np.full((count, MAX_TOWER), 361, np.int16),
        "tower_mask": np.zeros((count, MAX_TOWER), bool),
        "ant_cont": np.zeros((count, MAX_ANT, 4), np.float32),
        "ant_kind": np.full((count, MAX_ANT), 2, np.int8),
        "ant_behavior": np.full((count, MAX_ANT), 5, np.int8),
        "ant_pos": np.full((count, MAX_ANT), 361, np.int16),
        "ant_mask": np.zeros((count, MAX_ANT), bool),
    }
    state_feats = np.zeros((count, len(state_feature_names)), np.float32)
    plan_feats = np.zeros((count, len(plan_feature_names)), np.float32)
    value = np.zeros(count, np.float32)
    std = np.zeros(count, np.float32)
    samples = np.zeros(count, np.int16)
    heuristic = np.zeros(count, np.float32)
    is_hold = np.zeros(count, bool)
    group = np.zeros(count, np.int64)
    # 计划的目标格（361=pad）：建塔用坐标，升级/降级用那座塔的位置。
    # 只给 dx/dy 汇总是不够的——模型需要知道"打在哪个格子"才能结合棋盘判断。
    plan_pos = np.full(count, 361, np.int16)

    missing = 0
    for index, row in enumerate(rows):
        key = (row["seed"], row["round"])
        state = cache.get(key)
        if state is None:
            missing += 1
            continue
        tensors = build_state_tensors(state, row["player"])
        for name, array in zip(("tower_cont", "tower_type", "tower_pos", "tower_mask",
                                "ant_cont", "ant_kind", "ant_behavior", "ant_pos", "ant_mask"),
                               tensors):
            arrays[name][index] = array
        for column, name in enumerate(state_feature_names):
            state_feats[index, column] = float(row.get(name, 0.0))
        for column, name in enumerate(plan_feature_names):
            plan_feats[index, column] = float(row.get(name, 0.0))
        value[index] = row["mean"]
        std[index] = row["std"]
        samples[index] = row["n"]
        heuristic[index] = row["heuristic"]
        is_hold[index] = row["key"] == "hold"
        group[index] = row["seed"] * 1_000_000 + row["round"] * 10 + row["player"]
        towers_by_id = {int(t[0]): t for t in (state.get("towers") or [])}
        for operation in row.get("ops") or []:
            op_type, arg0, arg1 = (int(v) for v in operation[:3])
            if op_type == 11:  # 建塔：arg0/arg1 是坐标
                if 0 <= arg0 < 19 and 0 <= arg1 < 19:
                    plan_pos[index] = arg0 * 19 + arg1
                break
            if op_type in (12, 13, 21):  # 升级/降级/闪电：arg0 是塔 id（闪电是坐标，下面单独处理）
                tower = towers_by_id.get(arg0)
                if tower is not None:
                    plan_pos[index] = int(tower[2]) * 19 + int(tower[3])
                    break
                if op_type == 21 and 0 <= arg0 < 19 and 0 <= arg1 < 19:
                    plan_pos[index] = arg0 * 19 + arg1
                    break
            if op_type == 21 and 0 <= arg0 < 19 and 0 <= arg1 < 19:
                plan_pos[index] = arg0 * 19 + arg1
                break

    keep = np.ones(count, bool)
    if missing:
        # 缺状态的样本直接丢掉
        for index, row in enumerate(rows):
            if (row["seed"], row["round"]) not in cache:
                keep[index] = False
        print(f"[warn] {missing} 行缺状态，已丢弃")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{name: array[keep] for name, array in arrays.items()},
        state_feats=state_feats[keep], plan_feats=plan_feats[keep],
        value=value[keep], std=std[keep], n=samples[keep],
        heuristic=heuristic[keep], is_hold=is_hold[keep], group=group[keep],
        plan_pos=plan_pos[keep],
    )
    print(f"[完成] {int(keep.sum())} 行 -> {out}")
    print(f"[shape] tower_cont {arrays['tower_cont'][keep].shape} "
          f"state_feats {state_feats[keep].shape} plan_feats {plan_feats[keep].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
