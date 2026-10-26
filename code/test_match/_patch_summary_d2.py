"""用行范围替换 00_summary.md 的 D 小节（上次字符串匹配没命中）。"""
from pathlib import Path

p = Path("docs/rl_survey/00_summary.md")
lines = p.read_text(encoding="utf-8").splitlines()

new_section = [
    "### D. 先手承诺序列博弈（`d_sequential_commitment.md`）",
    "",
    "| 结论 | 等级 |",
    "|---|---|",
    "| 局面全可见 ⇒ 信息集是单点 ⇒ 这是**完全信息扩展式博弈**，解概念是 **SPE（逆向归纳）**，不是 Stackelberg 均衡；P0 的均衡首着 = `argmax_{a0} min_{a1} V` | [正文] Conitzer & Sandholm EC'06 |",
    "| **P0 用混合策略在理论上弱劣势**：P1 看到的是**实现值**而非分布 ⇒ `E[min] ≤ max min`；混合只能当 exploit，**应从主方案降级** | [正文] + [推理] |",
    "| 座位差量级格格不入：棋类锚点 3–5.5pp、项目内 AntWar 后手优势 5.4%；**项目史上同型的 28.1pp 在配对补测里归零**（53.1% vs 53.1%）⇒ 更像\"搜索只在 P1 侧有增益 + 小样本伪影\"，**先配对 + McNemar 重测** | [正文] |",
    "| 落地（10 s 内可行）：只改 P0 侧打分——先验截到 K0=8，每个 a0 取对手 top-3 回应（用我们自己的网当对手模型），softmin_β，预算 **0.64 s/决策**；完整 2-ply + rollout 需 184 s，**不可行** | [推理] 预算推算 |",
    "| **瓶颈不是时间而是候选数**：91.6% 的决策只有 1 个候选、80% 被 SKIP1 短路 | 项目实测 |",
    "| 零成本诊断：\"回应值缺口\"——`argmax` 一致率 <70% ⇒ 方案值得做；>90% ⇒ 查价值网座位校准 | [推理] |",
]

start = next(i for i, l in enumerate(lines) if l.startswith("### D. 先手承诺"))
end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("---"))
out = lines[:start] + new_section + [""] + lines[end:]
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"  替换 {start+1}..{end} 行，新 D 小节 {len(new_section)} 行")
