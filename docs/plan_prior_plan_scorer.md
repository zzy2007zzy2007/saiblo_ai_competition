# 计划：Plan Scorer（亚军先验蒸馏的第一步 —— 验证表达力）

> 2026-10-18。承接 `docs/entity_transformer_value_results.md`：
> 价值网已经能拟合到 R² 0.9969，但**亚军的力量不止在价值**——它的候选打分 = `value + 手写 heuristic`。
> 本文只做**第一件事**：验证 Transformer 能否学会亚军"对给定计划打分"的函数（`(state, plan) → score`）。
> **不涉及候选集生成**，也**不改我们的搜索**。

## 1. 为什么先做"打分"而不是"生成"

亚军的运作是：`root_plans` 手写模板生成 ~150 个候选计划 → 每个计划用
`value(rollout) + base/lure/lightning heuristic - penalty` 打分 → UCB1 bandit 选。

- **打分（A）**：给定候选，输出分数 —— **直接对口**亚军的先验，训练数据现成（plans 日志）；
- **生成（B）**：自己产生动作 —— 是"策略"，更难，留到 A 通过之后。

离线消融已表明：去掉**价值项**后 41.8% 决策 argmax 改变；去掉**整个价值**（zero）会被 27:5 碾压。
所以价值与 heuristic **都重要**，而 heuristic 那一半还没有任何网络能表达 —— 本文补这一半。

## 2. 数据

**来源（一次采集同时拿两份）**：
1. **状态**：桥接 `MVS_RAW_STATE_DIR` 的 `raw_seed*.bin`（board+stats）+ `ents_seed*.jsonl`（塔/蚁实体）；
2. **计划**：亚军 `MVS_RUNNERUP_DEBUG=plans` 写进 stderr 的 `{"kind":"plan",...}` 行，
   桥接归档为 `match_results/ai{0,1}_<label>_seed<seed>.stderr.log`。

**对齐**：按 `(seed, round, player)` 把"状态"和"该决策的 ~150 个 plan"配对。
一个决策 = 一条 `(state, [plan_1..plan_K, score_1..score_K])`；一局 ~1024 决策、~15 万 (state, plan, score) 对。

**原始 dump 已包含**：`round / player / serial / rank / key / name / first / pretty / second /
base_heuristic / lure_heuristic / lightning_heuristic / operation_penalty / followup_penalty / heuristic /
score_before_heuristic / score / mean_terminal_score` 等。

## 3. 目标（y）

**三层可选，按"先验纯度"从高到低**：

| 目标 | 含义 | 用途 |
|---|---|---|
| `heuristic_net = score - mean_terminal_score` | 纯手写分（去价值） | **主目标**：验证"先验"能否被表达 |
| `score` | 价值 + heuristic 的完整选择准则 | 备选（但会混入价值） |
| `rank` / 访问次数（bandit 最终选择）| 亚军实际选谁 | 下游"选对率"评估 |

> 主目标用 `heuristic_net`，因为它才是"先验"本身；价值我们已经能单独拟合。

## 4. 模型（复用实体 Transformer 的 state 编码）

```
state tokens : 塔 token + 蚁 token + stats token（与 value 网完全相同，权重可 warm-start）
plan token   : plan = [op_1, ..., op_m]（m≤3 + followup）
               每个 op = (op_type, class, cell, 目标塔类型) → 一个 embedding
               把 m 个 op embedding 池化（加 op 位置 embedding）→ 一个 plan 向量
score        : plan token 对 state tokens 做 cross-attention → 一个标量
```

- **多 op / followup**：`first`/`second`/`pretty` 三个字段正好给出 op 序列，按位置编码；
- **可解释性**：attention 权重能看"这个计划在关注哪些塔/蚁"，便于诊断；
- **输入不变**：state 编码复用价值网（可冻结或微调）。

## 5. 判据

- 在**未参与训练的对局**（按 seed 切）上，算"我们分数 vs 亚军 `heuristic_net`"的 **Spearman 秩相关**；
- **通过线**：Spearman **≥ 0.8**（先求"排得对"）；
- 辅助：`argmax 一致率`（我们 top1 与亚军 top1 是否同一个 plan）、top5 召回。

> 若 Spearman 上不去，说明"手写 heuristic 不是轻量 Transformer 能学的"，这本身是对
> "亚军的强来自手写规则"的一个硬证据。

## 6. 明确不做 / 风险

- **不做候选集生成**（`root_plans` 复现留第二步）；
- **不改搜索/部署**；
- **风险**：plans 日志体积大（~15 MB/局），64 局 ~1 GB——可接受，但解析要流式；
- **风险**：`op_type/class/cell/目标塔类型` 的解析要严格对齐协议的 token 格式。

## 7. 步骤

1. 解析 plans 日志 → `(seed, round, player, plan_key, op序列, heuristic_net)` 的 npz；
2. 与实体状态按 `(seed, round, player)` join；
3. 用实体 Transformer 的 state 编码 + 新的 plan 编码 + cross-attention 打分头训练；
4. 看 Spearman / argmax 一致率；
5. 通过后再谈"候选集"与"接进搜索"。
