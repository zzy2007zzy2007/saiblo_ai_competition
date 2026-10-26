"""最小验证：**一回合的出招必须一次性交给引擎**，否则合法性与官方判题器不一致。

官方 cold 路径（`game.cpp:2627` 起，`m1_support.cpp:171` 逐行复刻）里，
`used_tower`（同一座塔一回合只能操作一次）和 `camp_upgraded_flag`（一回合只能升一次基地）
是**一次 `apply_operation` 调用内部的局部变量**。所以：

  一回合 [op_a, op_b] 一次调用    ⇒ op_b 可能被"这座塔本回合已用过"挡掉（= 官方判题器行为）
  op_a 一次调用、op_b 一次调用    ⇒ 挡板被重置，op_b 照样生效（= 错的，局面从此分叉）

最小例子：`2\\n12 0 1\\n13 0`（先把 0 号塔从 Basic(0) 升成 Heavy(1)，再降级同一座塔）
  * 官方判题器 main.exe   ：`13 0` 被 "Tower has been used" 挡掉 ⇒ 塔还是类型 1、金币 -60
  * facade 一次调用整列表 ：同官方（类型 1、-60）
  * facade 拆成两次调用   ：`13 0` 生效 ⇒ 塔降回类型 0、金币还多出一笔降级返还
（`upgrade_type_check` = `new_type/10 == 当前类型`，所以 Basic 只能升到 1/2/3/4，
  不能直接升 11；之前的 `12 0 11` 会被规则挡掉，演示不出这个问题。）

用法：
  python code/test_match/probe_oplist_batching.py
  python code/test_match/probe_oplist_batching.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PASS_ROUNDS = 40          # 攒钱：初始 50 金币，约 +1.5/回合；建塔 15 + 升级 60 需要 ~75
BUILD, UPGRADE, DOWNGRADE = 11, 12, 13


def _packet(obj) -> bytes:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def plan(seed: int, pass_rounds: int):
    """在 facade 上找**真正能建**的格子（用真路径 trial：clone 后 apply 看塔数）。"""
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    base = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(pass_rounds):
        base.advance_round()
    for x in range(19):
        for y in range(19):
            trial = base.clone()
            trial.apply_operation_list(0, [Operation(OperationType(BUILD), x, y)])
            if len(trial.towers) > len(base.towers):
                return x, y, [int(c) for c in base.coins]
    raise RuntimeError("没找到合法建塔格")


def facade_run(seed: int, whole_list: bool, cell, pass_rounds: int):
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    x, y = cell
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(pass_rounds):
        st.advance_round()
    st.apply_operation_list(0, [Operation(OperationType(BUILD), x, y)])
    after_build = [(int(t.tower_id), int(t.tower_type), int(t.hp)) for t in st.towers]
    st.advance_round()
    ops = [Operation(OperationType(UPGRADE), 0, 1),    # 12 0 1   -> Basic(0) 升 Heavy(1)
           Operation(OperationType(DOWNGRADE), 0)]      # 13 0     -> 降级同一座塔
    if whole_list:
        st.apply_operation_list(0, ops)
    else:
        for o in ops:
            st.apply_operation_list(0, [o])
    after_ops = [(int(t.tower_id), int(t.tower_type), int(t.hp)) for t in st.towers]
    coins_after_ops = [int(c) for c in st.coins]
    st.advance_round()
    return {"after_build": after_build, "after_ops": after_ops,
            "coins_after_ops": coins_after_ops,
            "after_round": [(int(t.tower_id), int(t.tower_type), int(t.hp)) for t in st.towers]}


def judge_run(exe: Path, seed: int, cell, out_json: Path, pass_rounds: int):
    x, y = cell
    if out_json.exists():
        out_json.unlink()
    init = {"player_list": [1, 1], "player_num": 2,
            "config": {"random_seed": seed, "movement_policy": "enhanced",
                       "cold_handle_rule_illegal": True},
            "replay": str(out_json), "time": 0}
    buf = bytearray(_packet(init))
    for _ in range(pass_rounds):
        for p in (0, 1):
            buf += _packet({"player": p, "content": "0\n", "time": 0})
    buf += _packet({"player": 0, "content": f"1\n{BUILD} {x} {y}\n", "time": 0})
    buf += _packet({"player": 1, "content": "0\n", "time": 0})
    buf += _packet({"player": 0,
                    "content": f"2\n{UPGRADE} 0 1\n{DOWNGRADE} 0\n", "time": 0})
    buf += _packet({"player": 1, "content": "0\n", "time": 0})
    buf += _packet({"player": -1, "content": json.dumps({"player": 0, "error": 0}), "time": 0})
    proc = subprocess.run([str(exe)], input=bytes(buf), stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, timeout=600)
    if not out_json.exists():
        raise RuntimeError(f"no replay rc={proc.returncode} "
                           f"stderr={proc.stderr.decode('utf-8', 'replace')[:500]}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    live: dict[int, int] = {}
    timeline = []
    for i, entry in enumerate(data):
        for t in entry["round_state"].get("towers", []):
            if int(t["type"]) == -1:
                live.pop(int(t["id"]), None)
            else:
                live[int(t["id"])] = int(t["type"])
        timeline.append((i + 1, entry.get("op0"), entry.get("op1"), dict(live)))
    return {"rounds": len(data), "timeline": timeline, "live": live}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--pass-rounds", type=int, default=PASS_ROUNDS)
    ap.add_argument("--judge", type=Path,
                    default=REPO / "code/cpp_engine/build_probe/judge_O0.exe")
    ap.add_argument("--replay-dir", type=Path, default=Path("C:/Temp/probe_build/replay"))
    args = ap.parse_args()

    x, y, coins0 = plan(args.seed, args.pass_rounds)
    print(f"seed={args.seed}: 先空过 {args.pass_rounds} 回合（金币 {coins0}），"
          f"再在 ({x},{y}) 建 0 号塔，然后测试回合 p0 = `2\\n{UPGRADE} 0 1\\n{DOWNGRADE} 0`\n")

    whole = facade_run(args.seed, True, (x, y), args.pass_rounds)
    per_op = facade_run(args.seed, False, (x, y), args.pass_rounds)
    print(f"facade 整列表一次调用（正确用法）: 建塔后={whole['after_build']} "
          f"测试回合 apply 后={whole['after_ops']} 金币={whole['coins_after_ops']}")
    print(f"facade 拆成两次调用（错误用法）  : 建塔后={per_op['after_build']} "
          f"测试回合 apply 后={per_op['after_ops']} 金币={per_op['coins_after_ops']}")
    print(f"  -> 两种调用方式{'一致' if whole['after_ops'] == per_op['after_ops'] else '**不一致**'}\n")

    if not args.judge.exists():
        print(f"!! 判题器不存在：{args.judge}")
        return 0
    j = judge_run(args.judge, args.seed, (x, y),
                  args.replay_dir / f"replay_batch_{args.seed}.json", args.pass_rounds)
    print(f"官方判题器 {args.judge.name}: replay {j['rounds']} 条")
    for r, op0, op1, live in j["timeline"][-4:]:
        print(f"  round {r:3d} op0(已接受的 op)={op0} 塔状态(id->type)={live}")
    judge_towers = sorted(j["live"].items())
    facade_towers = sorted((t[0], t[1]) for t in whole["after_round"])
    facade_towers_bad = sorted((t[0], t[1]) for t in per_op["after_round"])
    print(f"\n官方判题器 最终塔(id->type)    : {judge_towers}")
    print(f"facade 整列表 最终塔(id->type) : {facade_towers}   "
          f"{'✅ 与官方一致' if judge_towers == facade_towers else '❌'}")
    print(f"facade 拆两次 最终塔(id->type) : {facade_towers_bad}   "
          f"{'✅ 与官方一致' if judge_towers == facade_towers_bad else '❌ 与官方不一致'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
