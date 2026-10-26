"""探针：复现"重放 base_hp 和日志 RESULT 不一致"，并把它归因到**回放脚本**（不是引擎）。

背景：`code/test_match/recall_on_runnerup.py` 的"重放版"曾报
`重放 base_hp=(4,0) 记录=(0,28)`，据此怀疑 Python 引擎与官方 C++ 引擎不一致。
本探针把同一份日志用 4 种方式重放，隔离出两个**纯粹属于回放脚本**的 bug：

  bug-1 日志解析：日志里出招列表打印成 Python `repr(Operation)`，形如
        `-> [Operation(op_type=<OperationType.BUILD_TOWER: 11>, arg0=8, arg1=16)]`，
        这不是合法 Python 字面量 ⇒ `ast.literal_eval` 抛异常 ⇒ 该侧这一回合的
        出招被**静默丢弃**；如果同一回合两侧都有出招，这一回合连
        `advance_round()` 都不会被调用。
  bug-2 出招施加：官方 cold 路径的 `used_tower` / `camp_upgraded_flag` 是
        **"一次 apply 调用内"**的局部状态。把一回合的出招拆成多次
        `apply_operation_list(p, [op])` 调用，就绕过了"同一座塔一回合只能操作一次"、
        "一回合只能升一次基地"的限制，同一回合的重复操作会被重复执行。

用法：
  python code/test_match/probe_replay_bug_demo.py <log>
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_replay_engines import parse_log, parse_ops_text  # noqa: E402

# recall_on_runnerup.py 里的原版正则（注意 `原文='[^']*'` 只取到 repr 的第一段）
RECALL_LINE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文='[^']*' -> (\[.*\])")


def recall_parse(path: Path):
    """逐字复刻 recall_on_runnerup.py:parse_log 的解析行为。"""
    per: dict[int, dict[str, list]] = {}
    stats = {"printed_lines": 0, "literal_eval_fail": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RECALL_LINE.search(line)
        if not m:
            continue
        stats["printed_lines"] += 1
        rnd, side, ops = int(m.group(1)), m.group(2), m.group(4)
        try:
            lst = ast.literal_eval(ops)
        except Exception:  # noqa: BLE001
            stats["literal_eval_fail"] += 1
            continue
        per.setdefault(rnd, {})[side] = lst
    return per, stats


def replay(per_ops, n_rounds, seed, whole_list: bool):
    """whole_list=True  ⇒ 一回合两侧各一次 apply_operation_list（引擎的正确用法）
       whole_list=False ⇒ 每个 op 单独一次 apply_operation_list（recall 的用法）"""
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    for r in range(n_rounds):
        if st.terminal:
            break
        d = per_ops.get(r, {})
        for p in (0, 1):
            ops = d.get(p) or []
            if not ops:
                continue
            conv = [Operation(OperationType(int(o[0])),
                              int(o[1]) if len(o) > 1 else -1,
                              int(o[2]) if len(o) > 2 else -1) for o in ops]
            if whole_list:
                st.apply_operation_list(p, conv)
            else:
                for o in conv:
                    st.apply_operation_list(p, [o])
        st.advance_round()
    return (int(st.bases[0].hp), int(st.bases[1].hp))


def main() -> int:
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4/10.log")
    info = parse_log(log)
    seed, n_rounds, hp_ref = info["seed"], info["result"]["rounds"], info["result"]["hp"]
    print(f"[log] {log.name}  seed={seed}  RESULT rounds={n_rounds} base_hp={hp_ref}\n")

    # ── bug-1：日志解析 ──
    per_bad, stats = recall_parse(log)
    n_zero_side = sum(1 for r, d in per_bad.items() for s in ("p0", "p1") if not d.get(s))
    print("== bug-1  日志解析 ==")
    print(f"  日志里打印出招的行数            : {stats['printed_lines']}")
    print(f"  ast.literal_eval 失败（被丢弃） : {stats['literal_eval_fail']}")
    print(f"  recall 解析后还剩的回合数       : {len(per_bad)}")
    print(f"  被保留的回合里'某侧没有出招'的次数: {n_zero_side}")

    # ── bug-1 的直接后果：多少回合/多少 op 被丢掉 ──
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    n_ops_true = n_ops_bad = 0
    n_rounds_dropped = 0
    for r in range(n_rounds):
        d = info["ops_text"].get(r, {})
        true_here = 0
        for p in (0, 1):
            true_here += len(parse_ops_text(d.get(p, "0\n"), Operation, OperationType))
        bad_here = sum(len(per_bad.get(r, {}).get(f"p{p}") or []) for p in (0, 1))
        n_ops_true += true_here
        n_ops_bad += bad_here
        if true_here > 0 and bad_here == 0:
            n_rounds_dropped += 1
    print(f"  真实出招总数                    : {n_ops_true}")
    print(f"  recall 解析后剩下的出招总数     : {n_ops_bad}"
          f"  (丢掉 {n_ops_true - n_ops_bad} 条 = "
          f"{100.0 * (n_ops_true - n_ops_bad) / max(n_ops_true, 1):.1f}%)")
    print(f"  整回合被跳过（该回合出招一条都没剩下）: {n_rounds_dropped} 回合\n")

    # ── 4 种重放 ──
    good_ops = {}
    for r in range(n_rounds):
        d = info["ops_text"].get(r, {})
        good_ops[r] = {p: [(int(o.op_type), int(o.arg0), int(o.arg1))
                           for o in parse_ops_text(d.get(p, "0\n"), Operation, OperationType)]
                       for p in (0, 1)}
    bad_ops = {r: {int(s[1]): v for s, v in d.items()} for r, d in per_bad.items()}

    combos = [
        ("V0  recall 原样：错解析 + 逐条 apply", bad_ops, False),
        ("V1  只修解析    ：对解析 + 逐条 apply", good_ops, False),
        ("V2  只修 apply  ：错解析 + 整列表 apply", bad_ops, True),
        ("V3  两处都修    ：对解析 + 整列表 apply", good_ops, True),
    ]
    print("== 4 种重放（都是同一个 native_game.pyd，同一个 seed） ==")
    for name, ops, whole in combos:
        hp = replay(ops, n_rounds, seed, whole)
        print(f"  {name:38s} base_hp={hp}  "
              f"{'✅ 与 RESULT 一致' if hp == hp_ref else '❌ 与 RESULT 不一致'}")
    print("\n结论：hp 偏差完全由回放脚本的解析/施加方式决定；"
          "把出招按'整回合一次 apply + 从原文解析'喂进去就与 RESULT 一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
