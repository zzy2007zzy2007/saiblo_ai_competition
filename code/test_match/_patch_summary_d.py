"""修正 00_summary.md 的 D 线小节与座位差口径（27pp → 47pp）。"""
from pathlib import Path

p = Path("docs/rl_survey/00_summary.md")
t = p.read_text(encoding="utf-8")

# ① 口径修正：47pp（72% vs 25%），不是 27pp
t = t.replace("座位差 27pp", "座位差 47pp（P1 72% / P0 25%）")
t = t.replace("27pp 座位差", "47pp 座位差")
t = t.replace("（27pp）", "（47pp）")

# ② D 线小节替换
old_d = """### D. 先手承诺序列博弈（`d_sequential_commitment.md`）——**待补**

（该线仍在调研：P0 先出招、P1 看到 P0 出招后回应、信息完全可见 ⇒ Stackelberg/commitment game；需回答 27pp 座位差如何建模与缓解。）"""
new_d = """### D. 先手承诺序列博弈（`d_sequential_commitment.md`）

| 结论 | 等级 |
|---|---|
| 局面全可见 ⇒ 信息集是单点 ⇒ 这是**完全信息扩展式博弈**，解概念是 **SPE（逆向归纳）**，不是 Stackelberg 均衡；P0 的均衡首着 = `argmax_{a0} min_{a1} V` | [正文] Conitzer & Sandholm EC'06 + [推理] |
| **P0 用混合策略在理论上弱劣势**：P1 看到的是**实现值**而非分布 ⇒ `E[min] ≤ max min`，随机化 ≤ 最好纯动作。混合只能当"打不完美对手"的 exploit，**应从主方案降级** | [正文]（作者主页 PDF）+ [推理] 不等式 |
| 座位差的量级格格不入：棋类锚点 3–5.5pp、项目内 AntWar 后手优势 5.4%；而**项目史上同型的 28.1pp 在配对补测里完全归零**（53.1% vs 53.1%）⇒ 更像"搜索只在 P1 侧产生增益 + 小样本伪影"，**应先配对 + McNemar 重测再改算法** | [正文] + [正文] experiment_log |
| 落地（10 s 内可行）：只改 P0 侧打分——候选先验截到 K0=8，对每个 a0 取对手 top-k=3 回应（用我们自己的网当对手模型），softmin_β，预算 **0.64 s/决策**；完整 2-ply + rollout 需 184 s，**不可行** | [推理] 预算推算 |
| **瓶颈不是时间而是候选数**：91.6% 的决策只有 1 个候选、80% 被 SKIP1 短路 | 项目实测 |
| 零成本诊断："回应值缺口"（`argmax` 一致率 <70% ⇒ 方案值得做；>90% ⇒ 去查价值网的座位校准）| [推理] |"""
t = t.replace(old_d, new_d)
p.write_text(t, encoding="utf-8")
print("  已更新 D 小节与座位差口径")
