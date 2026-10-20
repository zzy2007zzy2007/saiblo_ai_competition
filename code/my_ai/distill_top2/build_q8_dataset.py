"""构建「8 回合期望值」数据集：每个 (状态, 候选计划) 一行，标签是 N 次 rollout 的均值。

为什么需要新目标（见 docs/findings_runnerup_prior_structure.md）：
亚军的决策量是 `mean_terminal_score` = 8 回合随机 rollout 的终止评估均值，
而我们的价值网学的是**叶子评估** `evaluate_terminal(state)`。
实测「叶子评估 + 1 步前瞻」的决策一致率 = 永远 hold 的基线 ⇒ 必须直接学这个期望值。

输出每行：
    seed, round, player, key, heuristic, ops, horizon, followup_len,
    mean, std, n,
    state_*  （状态特征：金币/基地/塔数/蚁数/冷却等）
    plan_*   （计划特征：各类操作数、升级目标塔型、是否 hold 等）

用法：
    python build_q8_dataset.py --seeds 732 733 734 735 --rollouts 64 \
        --out training_history/distill_top2/q8/q8_dataset.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
HEX_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
# 塔的等级：0=基础，1=一级升级(1/2/3/4)，2=二级升级(11/12/13/21/22/23/31/32/33/41/42/43)
# 枚举见 antgame_cpp_sdk/include/antgame_sdk/sdk.hpp:62
TOWER_LEVEL = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1,
               11: 2, 12: 2, 13: 2, 21: 2, 22: 2, 23: 2,
               31: 2, 32: 2, 33: 2, 41: 2, 42: 2, 43: 2}
SNIPER_TYPE = 23


def make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def hex_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """offset-hex 的近似距离：用 cube 距离做尺度参考，只需单调性。"""
    def to_cube(x: int, y: int) -> tuple[int, int, int]:
        cube_x = x
        cube_z = y - (x - (x & 1)) // 2
        cube_y = -cube_x - cube_z
        return cube_x, cube_y, cube_z

    ax, ay, az = to_cube(x1, y1)
    bx, by, bz = to_cube(x2, y2)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def state_features(state: dict) -> dict:
    towers = state.get("towers") or []
    ants = state.get("ants") or []
    bases = state.get("bases") or [[50, 0, 0], [50, 0, 0]]
    coins = state.get("coins") or [0, 0]
    cooldowns = state.get("weapon_cooldowns") or [[0] * 5, [0] * 5]
    usage = state.get("super_weapon_usage") or [0, 0]

    feats: dict[str, float] = {}
    for player in (0, 1):
        mine = [t for t in towers if t[1] == player]
        theirs = [t for t in towers if t[1] == player and t[1] != player]  # 占位，实际不用
        del theirs
        my_ants = [a for a in ants if a[1] == player]
        enemy_ants = [a for a in ants if a[1] != player]
        feats[f"s_coins_{player}"] = coins[player]
        feats[f"s_base_hp_{player}"] = bases[player][0]
        feats[f"s_gen_lv_{player}"] = bases[player][1]
        feats[f"s_ant_lv_{player}"] = bases[player][2]
        feats[f"s_n_towers_{player}"] = len(mine)
        feats[f"s_tower_hp_{player}"] = sum(t[5] for t in mine)
        feats[f"s_tower_cd_{player}"] = sum(t[6] for t in mine)
        for tower_type in (0, 1, 2, 3, 4, 5):
            feats[f"s_tower_t{tower_type}_{player}"] = sum(1 for t in mine if t[4] == tower_type)
        feats[f"s_tower_x_{player}"] = sum(t[2] for t in mine)
        feats[f"s_tower_y_{player}"] = sum(t[3] for t in mine)
        feats[f"s_n_ants_{player}"] = len(my_ants)
        feats[f"s_ant_hp_{player}"] = sum(a[4] for a in my_ants)
        feats[f"s_ant_lvl_{player}"] = sum(a[8] for a in my_ants)
        for kind in (0, 1, 2):
            feats[f"s_ant_kind{kind}_{player}"] = sum(1 for a in my_ants if a[6] == kind)
        feats[f"s_cd_sum_{player}"] = sum(cooldowns[player])
        feats[f"s_cd_min_{player}"] = min(cooldowns[player]) if cooldowns[player] else 0
        for weapon in range(5):
            feats[f"s_cd_w{weapon}_{player}"] = cooldowns[player][weapon]
        feats[f"s_usage_{player}"] = usage[player]
        # 敌人离我基地最近距离（威胁代理量）
        base_x, base_y = (0, 9) if player == 0 else (18, 9)
        if enemy_ants:
            feats[f"s_enemy_min_dist_{player}"] = min(hex_distance(a[2], a[3], base_x, base_y)
                                                      for a in enemy_ants)
            feats[f"s_enemy_close3_{player}"] = sum(
                1 for a in enemy_ants if hex_distance(a[2], a[3], base_x, base_y) <= 3)
        else:
            feats[f"s_enemy_min_dist_{player}"] = 20
            feats[f"s_enemy_close3_{player}"] = 0
    feats["s_n_effects"] = len(state.get("active_effects") or [])
    return feats


def parse_plan_key(key: str) -> tuple[list[tuple[int, int, int]], int]:
    """从 plan key 反解出 ops 与 followup 步数。

    key 形如 `11:4:15|13:14:-1|F:1:14:1:1`：`F:` 开头的是 followup 步骤，
    其余每段是 `type:arg0:arg1`。rollout 模式的输出里不带 ops 字段，所以走 key。
    """
    if key == "hold":
        return [], 0
    ops: list[tuple[int, int, int]] = []
    followup_len = 0
    for segment in key.split("|"):
        if segment.startswith("F:"):
            followup_len += 1
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
        ops.append((values[0], values[1], values[2]))
    return ops, followup_len


def plan_features(plan: dict, state: dict, player: int) -> dict:
    ops, followup_len = parse_plan_key(plan["key"])
    towers_by_id = {t[0]: t for t in (state.get("towers") or [])}
    feats: dict[str, float] = {}
    feats["p_n_ops"] = len(ops)
    feats["p_is_hold"] = 1.0 if plan["key"] == "hold" else 0.0
    feats["p_horizon"] = plan["horizon"]
    feats["p_followup_len"] = followup_len
    feats["p_heuristic"] = plan["heuristic"]
    op_type_counts = {11: 0, 12: 0, 13: 0, 21: 0, 31: 0, 32: 0}
    base_x, base_y = (0, 9) if player == 0 else (18, 9)
    for slot in range(3):
        for key in ("type", "arg0", "arg1"):
            feats[f"p_op{slot}_{key}"] = 0.0
        feats[f"p_op{slot}_tower_type"] = -1.0
        feats[f"p_op{slot}_tower_hp"] = 0.0
        feats[f"p_op{slot}_tower_lv"] = -1.0
        feats[f"p_op{slot}_dist_base"] = -1.0
        feats[f"p_op{slot}_dx_base"] = 0.0
        feats[f"p_op{slot}_dy_base"] = 0.0
        feats[f"p_op{slot}_is_own_tower"] = 0.0
    for index, op in enumerate(ops[:3]):
        op_type, arg0, arg1 = op
        op_type_counts[op_type] = op_type_counts.get(op_type, 0) + 1
        feats[f"p_op{index}_type"] = op_type
        feats[f"p_op{index}_arg1"] = arg1
        tower = towers_by_id.get(arg0)
        if tower is not None:
            feats[f"p_op{index}_tower_type"] = tower[4]
            feats[f"p_op{index}_tower_hp"] = tower[5]
            feats[f"p_op{index}_tower_lv"] = TOWER_LEVEL.get(tower[4], -1)
            feats[f"p_op{index}_is_own_tower"] = 1.0 if tower[1] == player else 0.0
            feats[f"p_op{index}_dist_base"] = hex_distance(tower[2], tower[3], base_x, base_y)
            feats[f"p_op{index}_dx_base"] = tower[2] - base_x
            feats[f"p_op{index}_dy_base"] = tower[3] - base_y
        if op_type == 11:  # 建塔：arg0/arg1 是坐标
            feats[f"p_op{index}_dist_base"] = hex_distance(arg0, arg1, base_x, base_y)
            feats[f"p_op{index}_dx_base"] = arg0 - base_x
            feats[f"p_op{index}_dy_base"] = arg1 - base_y
    for op_type, count in op_type_counts.items():
        feats[f"p_cnt_{op_type}"] = count
    return feats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training_history/distill_top2/abl_linear2_raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[732, 733, 734, 735])
    parser.add_argument("--rounds", default="40,80,120,160,200,240,280,320,360,400,440")
    parser.add_argument("--players", default="0,1")
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=1, help="并行 exe 进程数")
    parser.add_argument("--out", default="training_history/distill_top2/q8/q8_dataset.jsonl")
    args = parser.parse_args()

    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    players = [int(x) for x in args.players.split(",")]
    env = make_env()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 先把所有 (state, player) 任务列出来，再按批并行跑
    tasks: list[tuple] = []
    for seed in args.seeds:
        paths = sorted(glob.glob(str(Path(args.data_dir) / f"ents_*{seed}.jsonl")))
        if not paths:
            print(f"[skip] seed={seed}")
            continue
        with open(paths[0], encoding="utf-8") as handle:
            all_states = [json.loads(line) for line in handle if line.strip()]
        for round_index in rounds:
            if round_index >= len(all_states):
                continue
            for player in players:
                tasks.append((seed, round_index, player, all_states[round_index]))
    print(f"[任务] {len(tasks)} 个 (状态, 玩家) combo，jobs={args.jobs}")

    def run_task(task):
        seed, round_index, player, raw_state = task
        state = dict(raw_state)
        state.update(seed=seed, player=player, mode="rollout",
                     rollouts=args.rollouts, max_plans=0)
        proc = subprocess.run([str(EXE)], input=(json.dumps(state) + "\n").encode(),
                              capture_output=True, env=env)
        return seed, round_index, player, raw_state, proc

    t_start = time.time()
    total_rows = 0
    with out_path.open("w", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for seed, round_index, player, raw_state, proc in pool.map(run_task, tasks):
            if proc.returncode != 0:
                print(f"[fail] seed={seed} r={round_index} p{player}")
                continue
            result = json.loads(proc.stdout.decode("utf-8", "replace").splitlines()[0])
            if result.get("ok") != 1:
                print(f"[warn] {result.get('error')}")
                continue
            shared = state_features(raw_state)
            for plan in result["plans"]:
                ops, followup_len = parse_plan_key(plan["key"])
                row = {
                    "seed": seed, "round": round_index, "player": player,
                    "key": plan["key"], "heuristic": plan["heuristic"],
                    "ops": ops, "horizon": plan["horizon"],
                    "followup_len": followup_len, "mean": plan["mean"],
                    "std": plan["std"], "n": plan["n"],
                }
                row.update(shared)
                row.update(plan_features(plan, raw_state, player))
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
            if total_rows and (len(tasks) % 50 == 0):
                pass
            if total_rows and total_rows % 50000 < 200:
                print(f"[进度] {total_rows} 行  {time.time() - t_start:.0f}s", flush=True)
    elapsed = time.time() - t_start
    print(f"\n[完成] {total_rows} 行 -> {out_path}  用时 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
