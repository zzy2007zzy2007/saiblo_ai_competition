# 计划：实体 token Transformer（先替换 value 通路）—— 三步去风险

> 2026-10。用户提出的激进方向：网络换成 Transformer，**塔/蚁每个实体一个 token**，
> 棋盘按格 token，位置用二维编码，动作头给"动作 token"输出分数。
> 本文把它**拆成三步**，每步单独判据，不做"一次性全换"。

## 0. 动机（为什么要换）

现有 `AntWarNetwork` 是 **CNN + GAP**，有两个已实测的硬伤：
1. **有损压缩**：board 28ch 里**没有塔血、没有蚁种类（worker/combat）**，蚁血按格求和 ⇒
   亚军的 `evaluate_terminal` 里 `salvage`/`threat` 两项**根本算不出来**（实测：棋盘只贡献 +0.016）。
2. **表达不了"先验"**：当前输出是 `(class, cell)` 的 logits，**没有 `(state, plan)→score` 的打分头**；
   而亚军的力量恰好是"对具名多 op 计划打分"（top1−top2 差 72，价值只占 8）。

⇒ 换成"实体 token + 动作打分头"的 Transformer，理论上**同时解决**这两件事。

## 1. 三步（按代价从小到大）

### 第 1 步：实体 token 版 **value**（本次先做）
- **输入**（无损）：
  - **塔 token**：`[x, y, player, tower_type, hp/max_hp, cooldown]`
  - **蚁 token**：`[x, y, player, kind(worker/combat), hp, max_hp, level, age, behavior]`
  - **格 token**（可选，提供地形/信息素/效果）：`[x, y] + 28ch` 里地形与状态那几维，或退化为只喂
    "地形 + 信息素 + 效果" 的稀疏 token
  - **stats token**：42 维 + 一个"全局/CLS" token
- **位置编码**：**学习式二维位置 embedding**（固定 19×19，比 RoPE 简单；RoPE 留作后续可选）
- **编码器**：N 层 Transformer（先 N=3，d_model=128）
- **value 头**：对 `[CLS]` token（或 global mean）→ MLP → 标量
- **判据**：val R² vs 现有 CNN（**0.869**）/ 线性 stats（**0.932**），且补上塔血/蚁种类后
  **应显著超过 0.93**（如果超不过，说明"实体 token"这条路也一般，及时止损）。

### 第 2 步：**plan scorer** 离线验证（价值最大的一步）
- 目标：`(state tokens, plan token) → 分数`，拟合亚军 `debug=plans` 日志里的 `score`（=value+heuristic）。
- plan token = 把该 plan 的若干 op（`(op_type, 类/目标, 格)`）各自 embed，再拼/池化成一向量。
- 判据：在**未参与训练的对局**上，与亚军真实 `score` 的 **Spearman 秩相关 ≥ 0.8**（先拟合"能排对"，
  再谈绝对误差）。这是"架构能不能表达先验"的直接检验。

### 第 3 步：query-based 动作头（最后，且只在 1、2 都过时做）
- 每个候选动作（或候选束）一个 query token → attend state tokens → 输出分数/概率；
- 与现有 `bundle_mcts` 的候选枚举对接（先离线，再接搜索）。

## 2. 数据（第 1 步的前提）

现有 `value_ds_runnerup_selfplay.npz` 只有 `board+stats+value`，**没有实体特征**。
⇒ 需要**扩展桥接的 raw dump**：每回合除 board/stats 外，另存一份
`ents_seed<seed>.jsonl`：
```
{"round": r, "towers": [[id, player, x, y, type, hp], ...],
 "ants": [[id, player, x, y, hp, max_hp, kind, age, behavior], ...],
 "coins": [c0, c1], "bases_hp": [h0, h1]}
```
然后重采 **8–12 局**亚军自对弈（jobs=8，约 5–10 分钟），足够做第 1 步。

## 3. 明确不做 / 风险

- **不动**现有 `AntWarNetwork`（新模型另起 `code/my_ai/entity_transformer/`），不回退任何既有读数。
- **风险**：Transformer 更吃数据、更慢（部署搜索的毫秒预算要重评估）。所以第 1 步只拿来**验证表达力**，
  不承诺能替换部署。
- **数据量**：第 1 步 8–12 局（~8–12k 样本）够做"表达力"对比；真要训到可用再扩。
