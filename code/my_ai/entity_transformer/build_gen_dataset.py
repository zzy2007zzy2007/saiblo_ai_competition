"""为「用神经网络生成候选动作」构建数据集。

任务定义
--------
输入：状态（实体张量 + 聚合特征）
输出：一组候选动作（先做"首个 op + hold"，再扩到多 op 序列）
标签：亚军**真实下发**的那个计划（日志里 `rank==1` 的那条），
      拆成 (动作类, 目标) —— 建塔/闪电的目标是格子，升级/降级的目标是"我方的第几座塔"。

为什么用日志而不是自己 rollout：生成候选只需要"亚军选了什么"，
日志里就是真值，**不需要跑任何 rollout**，所以数据集可以做得很大（64 局 × 512 回合 × 双方 ≈ 65k）。

标签编码
--------
cls:  0=hold, 1=build(11), 2=upgrade(12), 3=downgrade(13), 4=lightning(21)
tgt:  格子 id (x*19+y)（build/lightning）；或"我方塔的下标"（按 tower_id 排序）（upgrade/downgrade）
tgt_kind: 0=格子, 1=塔下标
arg1: 升级目标塔型（upgrade 用）
cls2/tgt2/tgt_kind2/arg1_2: 第二个 op（约 46% 的"有动作"决策是双 op）

用法：
    python build_gen_dataset.py --seeds 732 733 ... --out .../gen_ds.npz
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
from my_ai.distill_top2.build_q8_dataset import state_features  # noqa: E402

MAX_TOWER = 48
MAX_ANT = 96
CLASS_OF_OP = {11: 1, 12: 2, 13: 3, 21: 4}


def parse_key(key: str):
    if key == "hold":
        return []
    ops = []
    for segment in key.split("|"):
        if segment.startswith("F:"):
            continue
        parts = segment.split(":")
        if len(parts) < 2:
            continue
        try:
            values = [int(x) for x in parts[:3]]
        except ValueError:
            continue
        while len(values) < 3:
            values.append(-1)
        ops.append(tuple(values))
    return ops


def load_chosen(log_path: Path) -> dict[tuple[int, int], list]:
    """读日志，取每个 (round, player) 的 rank==1 计划。"""
    chosen = {}
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # 快速预筛：一个决策有 ~150 条 plan 行，只有 rank==1 那条有用。
            # 先做字符串判断可以省掉 99% 的 JSON 解析（97 MB/局的日志）。
            if '"rank":1,' not in line or "base_heuristic" not in line:
                continue
            record = json.loads(line)
            chosen[(record["round"], record["player"])] = parse_key(record["key"])
    return chosen


def build_tensors(state: dict, player: int):
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
        tower_id, owner, x, y, ttype, hp = (int(v) for v in row[:6])
        tower_cont[slot] = (1.0 if owner == player else 0.0,
                            hp / (10.0 if ttype == 0 else 15.0), 0.0)
        tower_type[slot] = TOWER_TYPE_TO_ID.get(ttype, 16)
        tower_pos[slot] = x * 19 + y
        tower_mask[slot] = True
        slot += 1
    slot = 0
    for row in state.get("ants") or []:
        if slot >= MAX_ANT:
            break
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


def encode_op(op, own_tower_ids: dict[int, int]):
    """op -> (cls, tgt, tgt_kind, arg1)。own_tower_ids: tower_id -> 我方塔下标。"""
    op_type, arg0, arg1 = op
    cls = CLASS_OF_OP.get(op_type, 0)
    if cls in (1, 4):          # build / lightning：目标是格子
        return cls, arg0 * 19 + arg1, 0, 0
    if cls in (2, 3):          # upgrade / downgrade：目标是"我方第几座塔"
        return cls, own_tower_ids.get(arg0, 0), 1, (arg1 if cls == 2 else 0)
    return 0, 0, 0, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ents-dir", default="training_history/distill_top2/abl_linear2_raw")
    parser.add_argument("--log-dir", default="match_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[732])
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--out", default="training_history/distill_top2/q8/gen_ds.npz")
    args = parser.parse_args()

    seeds = args.seeds
    rows = {
        "tower_cont": [], "tower_type": [], "tower_pos": [], "tower_mask": [],
        "ant_cont": [], "ant_kind": [], "ant_behavior": [], "ant_pos": [], "ant_mask": [],
    }
    state_feat_rows, labels = [], []
    meta = {"game": [], "round": [], "player": []}
    state_feat_names: list[str] | None = None
    for seed in seeds:
        ents_path = Path(args.ents_dir) / f"ents_seed{seed:05d}.jsonl"
        if not ents_path.exists():
            candidates = list(Path(args.ents_dir).glob(f"ents_*{seed}.jsonl"))
            if not candidates:
                print(f"[skip] seed={seed}")
                continue
            ents_path = candidates[0]
        # 两名玩家的日志各覆盖一个 player
        chosen_by_player: dict[int, dict] = {}
        for player, pattern in ((0, f"ai0_runnerup_seed{seed}.stderr.log"),
                                (1, f"ai1_ai3_linear_seed{seed}.stderr.log")):
            log_path = Path(args.log_dir) / pattern
            if log_path.exists():
                chosen_by_player[player] = load_chosen(log_path)
        if not chosen_by_player:
            print(f"[skip] seed={seed} 没有日志")
            continue

        states = [json.loads(line) for line in ents_path.open(encoding="utf-8") if line.strip()]
        if args.max_rounds:
            states = states[:args.max_rounds]
        kept = 0
        for round_index, state in enumerate(states):
            for player, chosen in chosen_by_player.items():
                ops = chosen.get((round_index, player))
                if ops is None:
                    continue
                tensors = build_tensors(state, player)
                for name, array in zip(rows, tensors):
                    rows[name].append(array)
                features = state_features(state)
                if state_feat_names is None:
                    state_feat_names = list(features)
                state_feat_rows.append([features[name] for name in state_feat_names])

                # ⚠️ 目标必须是**塔在 state["towers"] 里的原始下标**（含敌方塔），
                #    因为 `build_tensors` 就是按这个顺序填槽位的，掩码也用同一套下标。
                #    之前用"我方第几座塔"会和掩码错位 ⇒ 目标落在被 mask 的位置 ⇒ CE 爆到 1e8。
                slot_of_tower = {int(t[0]): index
                                 for index, t in enumerate(state.get("towers") or [])
                                 if index < MAX_TOWER}
                own = [tid for tid, idx in slot_of_tower.items()
                       if int((state["towers"])[idx][1]) == player]
                first = encode_op(ops[0], slot_of_tower) if ops else (0, 0, 0, 0)
                second = encode_op(ops[1], slot_of_tower) if len(ops) > 1 else (0, 0, 0, 0)
                third = encode_op(ops[2], slot_of_tower) if len(ops) > 2 else (0, 0, 0, 0)
                labels.append((len(ops),) + first + second + third + (len(own),))
                meta["game"].append(seed)
                meta["round"].append(round_index)
                meta["player"].append(player)
                kept += 1
        print(f"[seed {seed}] 保留 {kept} 个决策")

    count = len(labels)
    print(f"[总计] {count} 个决策, 状态特征 {len(state_feat_names or [])} 维")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(values) for name, values in rows.items()}
    label_array = np.array(labels, dtype=np.int16)
    np.savez_compressed(
        out,
        **arrays,
        state_feats=np.array(state_feat_rows, np.float32),
        labels=label_array,
        game=np.array(meta["game"], np.int32),
        round=np.array(meta["round"], np.int32),
        player=np.array(meta["player"], np.int8),
        state_feat_names=np.array(state_feat_names or [], dtype=object),
    )
    print(f"[完成] -> {out}")
    print("labels 列含义: n_ops, cls0, tgt0, tgt_kind0, arg1_0, "
          "cls1, tgt1, tgt_kind1, arg1_1, cls2, tgt2, tgt_kind2, arg1_2, n_own_towers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
