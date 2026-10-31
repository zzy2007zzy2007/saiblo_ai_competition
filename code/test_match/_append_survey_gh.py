"""把 G/H 两条线补进 00_summary.md。"""
from pathlib import Path

p = Path("docs/rl_survey/00_summary.md")
t = p.read_text(encoding="utf-8")

add = """

---

## 七、补充线：AlphaZero 家族与 PPO 家族（2026-10-26 追加）

### G. AlphaZero 家族变体（`g_alphazero_variants.md`，14 篇读正文）

| 结论 | 等级 |
|---|---|
| **唯一同时解决"分支多 + 采样带噪 + 预算极薄"的是 Gumbel 系**；唯一解决"数据贵要榨干"的是 KataGo 系；其余（learned model / 连续动作 / 随机环境 / 大规模分布式）对我们负收益，文档列了 14 条"不必看" | [正文] |
| ① **同一组 Gumbel 决定候选集与选中者 + `σ(q)` 缩放**：可消掉 `σ√(2ln k)≈371` 的选择偏差。⚠️ 我们的 q 是"血量差"量纲（几百点），**必须先无量纲化**（建议 /200），否则 σ 会放大噪声 | [正文]+[推理] |
| ② **`π_MPO = softmax(logπ_θ + σ(q̂))` 当策略目标**：我们已有 q̂，零额外计算，且绕开"没有访问计数"的结构问题；风险是 q̂ 噪声大时退化成 one-hot（看目标熵诊断）| [正文] |
| ③ **分层预算**（KataGo playout cap 同构）：91.6% 单候选决策可做 0 次 rollout，省下的灌给 8.4% 关键决策 ⇒ SE 从 29 降到 ~9 | [正文] |
| ④ 值头改 **HL-Gauss** + 结构化辅助头（血量差/塔数/金币）：KataGo 辅助头消融 1.65×。⚠️ **Stop Regressing 实测 Two-Hot 在线 RL 反不如 MSE**，别抄 MuZero 的 two-hot | [正文] |
| ⑤ **离线 Reanalyse**：用当前网重标存量决策（我们数据:算力 1100:1）。优先级用 `\|q̂−z\|` 时噪声可能占比过高，建议改 Brier 或多次均值 | [正文] |
| 一行改动但几乎一定对：**Dirichlet 改总 α 恒定**（`α = 10.83/n_legal`）—— 我们合法候选数在 1~266 剧烈波动，固定 α 在只有 1 个候选时等于注入 100% 噪声 | [正文] KataGo |
| **依赖关系**：① 的效果强依赖"先把 k 缩到 8-16"（log₂266≈8 阶段，每阶段只剩 2 次采样）⇒ **缩小候选集是 Gumbel 方案的前提，不是可选项** | [正文]+[推理] |

**纠正**：`ReZero` 这个方向不存在（arXiv 同名是 LLM 检索方向）；`2103.04931` 是 MCTS 综述（Sampled MuZero 是 `2104.06303`）；Gumbel MuZero 无 arXiv 版（OpenReview 403，公式取 MiniZero + MA-Gumbel 正文复述）。

### H. PPO / actor-critic 家族（`h_actor_critic_family.md`）

见该文档；要点将并入下一版汇总（正在收尾）。
"""

if "## 七、补充线" not in t:
    p.write_text(t.rstrip() + add, encoding="utf-8")
    print("  已补 G/H 小节")
else:
    print("  已存在")
