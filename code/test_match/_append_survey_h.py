"""把 H 线（PPO/actor-critic）结论补进 00_summary.md。"""
from pathlib import Path

p = Path("docs/rl_survey/00_summary.md")
t = p.read_text(encoding="utf-8")

h = """### H. PPO / actor-critic 家族（`h_actor_critic_family.md`，39 篇 PDF）

| 结论 | 等级 |
|---|---|
| **大动作空间：没有一篇大规模实践用 flat softmax**。OpenAI Five 联合空间 30×4×189×81≈183 万维，但拆成 primary + Delay/Unit/Offset 三个参数头，每步实际只有 8.1 个 primary；非法组合靠 mask；**稀有动作单独开头**（否则被淹没）。AlphaStar 联合空间 ~10²⁶，用**自回归逐因子采样**，并明确写"假设各因子独立、分别更新"以缓解 trace cutting | [正文] |
| **纯稀疏奖励能学但要 10⁹ 量级**：OpenAI Five 纯 0/1 奖励消融能到 TrueSkill 155，但原文说"对样本效率是巨大惩罚"（未给倍数）；反例更强：**FTW/CTF 用纯胜负奖励"什么都没学到"** | [正文] |
| **长视野不是"1000 步太长"，而是 γ 设错了**：Five 的视界消融 45→720 s 单调改善；GAE λ=0.95 要求片段 ≥1/(1−γλ)≈20 步，Five 用 256 步。**我们 08 月 γ=0.99 ⇒ 有效视界仅 100 回合 = 512 回合的 1/5** | [正文]+[推算] |
| **"搜索+蒸馏"不比 model-free 差，反而更强**：ExIt vs REINFORCE（同架构）"learns stronger policies faster"；MuZero-Reanalyze 同帧数下中位分 731% vs IMPALA 192%/Rainbow 231% | [正文] |
| **不该把 PPO 当主线**：① 91.6% 决策只有 1 个候选 ⇒ 每局仅 ≈84 个"有梯度"的决策，我们是"榨干三元组"场景，PPO 是"要海量三元组"的算子；② 纯稀疏奖励无小算力先例；③ 我们两次失败根因（搜索/特征坍缩）换算法不解决 | [正文]+[推理] |
| **一个改变优先级的算术**：若 rollout 不用搜索（一次前向 ≈2 ms）⇒ 单局成本降 2-3 个数量级；但**行为策略决定天花板**——搜索当行为 ≈256 h（吃掉整月预算），策略网当行为 ≈30 min 但上限就是"单次前向策略"（08 月对外仅 31.2%）。**没有第三条路** | [推算] |
| 事实核对：**FTW 不是 PPO**（Ape-X DQN + transformed Bellman + 进化内部奖励）；OpenAI Five 论文**无 timeout/draw 规则**（我们 512 回合是真终止，不需要 bootstrap）| [正文] |
| 还有一张"**被大规模实验证伪的常用技巧**"表：value loss clipping 全部有害、Huber 不如 MSE、advantage 归一化影响不大、熵/KL 正则在多数 MuJoCo 任务无收益 | [正文] |
"""

if "### H. PPO / actor-critic 家族（`h_actor_critic_family.md`）" in t and "39 篇 PDF" not in t:
    lines = t.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("### H. PPO / actor-critic 家族"))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ") or lines[i].startswith("---")), len(lines))
    new = lines[:start] + h.splitlines() + lines[end:]
    p.write_text("\n".join(new) + "\n", encoding="utf-8")
    print(f"  已替换 H 小节（{start+1}..{end}）")
else:
    print("  状态未知，未修改")
