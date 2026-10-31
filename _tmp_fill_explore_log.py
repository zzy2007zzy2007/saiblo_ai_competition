"""把本次探索登记的 8 条实验 result 回填到 docs/experiment_log.md。"""
import io
import re

P = "docs/experiment_log.md"
RES = {
    "sweep_axes_n192":
        "第一版（旧 ref：3 塔/升 Sniper/不预留）：34 臂 × 192 局 = 8448 局、140 s。reserve=200/100 分别 "
        "+22.3/+19.0（SE~1）。**但发现「upg 先于 build」会把塔数锁死在 1**，臂之间被混淆 ⇒ 重写策略顺序后"
        "重跑（见 sweep_axes2_n192）。",
    "sweep_axes2_n192":
        "**47 臂 × 192 局 = 9792 局 / 134 s。闪电是主效应：weapons=none −21.9 HP、胜率 7.3%；"
        "build_reserve=300 +7.7/68.8%、upg_reserve=120 +5.9/74.0%；upg=never +3.5；三级塔 −5.6~−6.0；"
        "n_towers 0/1/3/4/5 = −0.7/−0.6/−1.5/−1.8/−1.8。** 读法见 "
        "`docs/free_exploration_strategy_arena_20261102.md` §3。",
    "sweep_greedy_cells":
        "第一版候选集错了（idx 0/1 = 基地格与其邻格，不可建 ⇒ 多个臂完全相同）。已修（只留两座位都可建的 "
        "idx 1..33），见 sweep_greedy_cells2。",
    "sweep_greedy_cells2":
        "33 臂 × 96 局 × 3 步 ≈ 9400 局 / 150 s。**首塔 idx=1（(4,9)/(14,9)）最好，任何别的位置 "
        "−2.4~−4.8；第 2 位 idx=2 最好；第 3 步无差别（n_towers=2 只建 2 座塔）。⇒ 默认 "
        "STRATEGIC_BUILD_ORDER（去掉不可建项）= 最优，贪心 3 步没能改进。**",
    "sweep_grid_n192":
        "22 臂 × 192 局：**最优 = build_reserve=300 + upg=always + n_towers=2（+7.72 ±0.99, 68.8%）；"
        "upg=never 家族 +0.4~+2.6 但胜率 55-67%；`br200_upnever_n1` 胜率 66.7% 而 margin 仅 +2.55"
        "（hp 判胜局里胜率与血差不单调）。**",
    "sweep_final_n192":
        "作废：此批 `best_policy.json` 被 grid 阶段的排序 bug 写成了 ref（当时比的是 ref 而不是 best）。"
        "重跑见 sweep_final2_n192 / sweep_final2b_n192。",
    "sweep_final2_n192": "grid 重跑（修 best 选择）：best = br300_upalways_n2，+7.72。",
    "sweep_final2b_n192":
        "**迁移检验，192 局/项：best vs ref +7.72（68.8%）；vs 不放武器 +29.0（100%）；"
        "vs 只放闪电不建塔 +7.55（71.4%）；vs 旧风格（3 塔+猛升级）+22.95（95.3%）；"
        "vs noop +30.7（100%）；vs 自己 −1.22（48.4%，镜像 sanity）。**",
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
