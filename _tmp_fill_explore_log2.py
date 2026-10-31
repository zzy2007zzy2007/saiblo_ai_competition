"""把本次"修理式升级 / 网络裸策略"两批实验的 result 回填到 docs/experiment_log.md。"""
import io
import re

P = "docs/experiment_log.md"
RES = {
    "repair_mechanic_n96":
        "8 臂 × 96 局 = 768 局 / 17 s。**升级的时机比升级本身重要：`upg=damaged`"
        "（塔血量比 ≤ 阈值才升级 = 用「升级回满血」当修理）damaged40/60/80 = +5.79/+5.00/+5.67"
        "（胜率 67-73%，闪电 10.1-10.6/局）；`upg=never` +3.16；`upg=always`（参照）−0.31；"
        "`build_reserve=300` +7.44。** ⇒ 无条件升级是负的、修理式升级是正的，且不牺牲闪电次数。"
        "见 `docs/free_exploration_strategy_arena_20261102.md` §5b。",
    "arena_net_vs_best":
        "**网络裸策略（3 网 ckpt + decode_network_output argmax，无搜索）在竞技场里 32 局："
        "vs `best` 规则策略 margin −1.53 ±1.84（13W-19L，40.6%），双方终局合计塔数 0.28 ⇒ "
        "网络裸策略基本「不造塔、攒钱放闪电」，只差调优规则策略 1.5 HP。** 结合 "
        "`why_no_upgrade_20261028.md`（类头永远 argmax 到 17）⇒ 现役的「升级吃掉 69% 收入」"
        "不是先验的主意，是评估器改坏的。见 free_exploration 文档 §6.3。",
    "arena_net_vs_rule":
        "作废：bash 把 JSON 策略串的引号吃掉了（`--duel ... {build_reserve: 300, ...}` 不是合法 spec），"
        "进程 17 s 后 exit=1。改用命名预设 `best` 重跑（见 arena_net_vs_best）。",
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
