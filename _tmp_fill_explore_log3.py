"""回填 repair_cycle_n96 / repair_cycle_n96(grid) 的 result + 记录 net/评估器体检。"""
import io
import re

P = "docs/experiment_log.md"
RES = {
    "repair_cycle_n96":
        "12 臂 × 96 局（含修理循环 rc60/rc40）：**repair_cycle60（打残才升级 + 下一回合降级退回）"
        "margin +5.72 ±1.06、胜率 77.1%、闪电 10.85/局、终局金币 267（对照 damaged60 +5.00/66.7%/191）；"
        "ref（有钱就升）−0.31、upg=never +3.16。** 引擎实测净成本 6 币/次（升级 60 → 降级退 54，血量回满）。"
        "见 `docs/free_exploration_strategy_arena_20261102.md` §5b。",
}


def main():
    t = io.open(P, encoding="utf-8").read()
    lines = t.splitlines()
    cur = None
    changed = 0
    for i, l in enumerate(lines):
        m = re.search(r"— (\S+)$", l)
        if m and m.group(1) in RES:
            cur = m.group(1)
            continue
        if cur and l.strip().startswith("- **result**:"):
            lines[i] = "- **result**: " + RES[cur]
            cur = None
            changed += 1
    io.open(P, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("patched", changed)


if __name__ == "__main__":
    main()
