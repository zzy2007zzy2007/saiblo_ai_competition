# B 线调研：结构化、带约束、可组合动作空间的策略表示与搜索

> **范围**：只做文献调研与落地建议，不改代码、不动正在跑的实验。
> **写作时间**：2026-10-26。
> **项目现状的事实来源**（本文引用的内部数字均出自这几份文档，未重新跑实验核对）：
> - [prior_ceiling_20261026.md](../prior_ceiling_20261026.md)：先验 recall（建塔 @1 6.3% / 随机 3.1%，@24 74.9% / 随机 74.1%；闪电 @1 5.5% / 随机 0.4%；降级支撑集中位 2 个）；非法招判负（`cold_handle_rule_illegal` 缺省 `false`）；离线指标与实战三次反向。
> - [az_action_factorization_idea.md](../az_action_factorization_idea.md)：束中位 1 个 op（均值 1.28、最多 3）；91% 回合 HOLD；class-only / pos-only 消融（增益 100% 在位置维、类维 0）；成本随候选多样性而非"搜哪一维"变化。
> - [pos_topk_candidates_plan.md](../pos_topk_candidates_plan.md)：束 ≤3 op；top-k 按 `action_map[类]` 降序 + 字典序定序。
> - [az_posnet_random_class_plan.md](../az_posnet_random_class_plan.md)、[alphazero_intent_mcts_plan.md](../alphazero_intent_mcts_plan.md)：`action_map` 是**类内** softmax（跨类不可比）；联合 softmax 会因"超武合法落点 ~200、建塔合法格 ~33"而稀释。
> - [model_design.md](../model_design.md)：`action_map` = 24 类 × 19×19，与 3 个类头拼接使用。
>
> **网络访问说明**：本次会话 `arxiv.org` 可正常抓取；`doi.org`、`www.nature.com`、`proceedings.mlr.press`、`iclr.cc`、`export.arxiv.org` 出现 DNS 解析失败（"resolves to a non-public IP address"）。因此有 4 条文献只确认了标题/编号而未读到正文，已在正文逐条标注，并汇总在 §5。

---

## 0. 术语对齐：我们的东西 ↔ 文献的词

| 我们的说法 | 文献里的说法 | 备注 |
|---|---|---|
| 动作束 bundle（≤3 个 op） | macro-action / action sequence / **set-valued action** | 束是"一次决策内多个原子动作"，不是"跨多步的技能" |
| `action_map` 24×19×19 | **factored action head**（per-cell × per-type logits） | 我们已经是一个因子化头 |
| 3 个 `head_logits` × 24 类 | multi-head / **autoregressive factors** | 缺的是因子间**条件化** |
| 金币、冷却、塔位、超武每回合一次 | 可行性约束 / **knapsack budget** / semi-bandit 的 super-arm 可行性 | |
| 非法操作**直接判负** | **hard constraint / one-shot failure**（不是 soft penalty） | 决定了 §2.4 的选型 |
| 266 个候选束 | candidate set / action subset generation | |
| 位置先验 recall@1 6.3% | **proposal distribution** 的质量 | 它是提案，不是策略本身 |

**一句话的现状判断**：我们的表示已经是"因子化动作头"，缺两样东西——**因子间条件化**（头 2 看不到头 1 选了什么）和**跨类可比的原子候选**。§3 会把这句说精确。

---

## 1. 结论摘要

1. **与我们形状匹配的是 factorized / autoregressive action head，不是 HRL。** options / Option-Critic / FeUdal 解决的是"跨时间的技能复用与长程信用分配"；我们的分解发生在**同一个决策时刻内部**（先选格、再选塔型、再决定是否追加），op 之间没有中间奖励，束只有 ≤3 个原子动作。把预算投到 FeUdal 上会错配问题。
2. **因子化必须"顺序化 + 每步重算 mask"，绝不能"并行分支 + 各自贪心"。** BDQ 类做法的独立分解会产出"每个因子单独合法、合起来非法"的组合；而我们非法招是**判负**不是扣分，代价不对称到不能靠概率兜底。正确形态是 AlphaStar 式自回归动作头：先动作类型 → 再位置/参数，每一步都在**已被上一步更新过的**合法集上归一化。
3. **硬约束场景应优先 shielding / masking，而不是 CMDP / Lagrangian 惩罚。** CPO 类方法保证的是"每次迭代近似满足约束"，不保证单步不违规；而我们违规一次就结束。掩码必须是**策略定义的一部分**、进 `logπ`（否则 rollout 分布与训练 ratio 不一致，PPO 会崩），并且要在**搜索的每个子节点**上重算，不能只在根节点掩一次。
4. **"表达不了协同"的确切位置是提案可达性，而不是"策略无法表达协同"。** `action_map` + 3 头的联合分布近似等于各因子边缘的乘积（无条件化），所以"单独看都平庸、一起很强"的 pair 在 266 个候选里**根本不存在**；但项目自己的测量说这个问题的**量级小**（束中位 1 个 op、最多 3）。所以先做**候选层**的补齐（零训练），比改网络划算得多。
5. **搜索预算不该平均分给各因子。** 项目已有的 `--search-mode {joint,class-only,pos-only}` 消融就是文献所说"因子化搜索的预算分配"的直接实验，结论是**增益 100% 落在位置维、类维为 0**；且成本随**候选多样性/树大小**增长（`pos-only(playable)` 比 joint 慢 3 倍）。原则：**把预算投到"有选择空间且增益最大"的那个因子搜索到底，其余因子用 argmax 解码**——多因子同时搜的成本近似相乘，收益却近似相加。

---

## 2. 逐方法

### 2.0 速览表

| 方法 | 解决什么 | 对我们 | 主要成本 |
|---|---|---|---|
| options / Option-Critic / FeUdal | 跨时间技能复用 | ❌ 主结构不匹配 | — |
| BDQ（action branching） | 动作维线性增长 | ⚠️ 只借结构，不借独立贪心 | 低 |
| AlphaStar 自回归动作头 | 因子顺序生成 + 每步掩码 | ✅ 最该抄的形态 | 中 |
| Pointer Networks | 输出字典随输入变 | ⚠️ 语义已有，价值在多步指针 | 低 |
| Order Matters | 集合输出的顺序假设 | ✅ 直接命中我们的"束内顺序"坑 | 低 |
| Wolpertinger | 百万级离散动作 | ❌ 需要动作度量结构 | 中 |
| RL with Combinatorial Actions（MIP） | 组合动作的 argmax | ✅✅ 最贴"金币预算 + 多塔" | 高（每次决策解 MIP） |
| Maskable / invalid action masking | 合法动作集上归一化 | ✅✅ 必做 | 极低 |
| Shielding | 每步安全动作集 + 纠正 | ✅ 与"判负"语义吻合 | 低 |
| CPO | 软约束 | ❌ 不适合一次性致命 | 中 |
| Deep Sets / Set Transformer | 集合编码（协同） | ✅ 条件输入的便宜做法 | 低 |
| CMAB（含 triggered arms） | super arm 的探索预算 | ⚠️ 借语言，不借算法 | — |
| ExIt | 搜索→策略的蒸馏闭环 | ✅ 已在做，要点是"束级"标签 | 低 |
| Sequential Halving | 固定预算下的臂分配 | ⚠️ 只用于候选排序 | 低 |

---

### 2.1 时序抽象三件套（options / Option-Critic / FeUdal）

**A. options 框架** — Sutton, Precup & Singh, *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*, Artificial Intelligence 112 (1999)。
链接：<https://doi.org/10.1016/S0004-3702(99)00052-1>（**本次未打开验证**，见 §5；DOI 号按记忆填写，若要正式引用请先核对）。

- **核心思想**：把"一段策略 + 终止条件"封装成一个时间上延展的动作（option），在 semi-MDP 上做规划与学习；理论保证（SMDP Q-learning 等）在终止时间随机时仍成立。
- **关键假设**：存在可复用的、在多个状态下重复出现的"技能"，且技能内部的动作序列本身不需要被精细优化；奖励可以在 option 结束时结算。
- **对我们**：❌ 不作为主结构。我们的"束"是**同一决策时刻内的动作组合**，不是跨步技能；束内 op 之间没有中间奖励、没有"终止条件可学"的空间（一次 `apply_operation_list` 就结算了）。硬套 options 只会把 266 个候选再套一层无意义的外壳。

**B. Option-Critic** — Bacon, Harb & Precup, AAAI 2017。链接：<https://arxiv.org/abs/1609.05140>（已验证标题/摘要）。
- **核心思想**：不需要额外奖励或子目标，直接用策略梯度定理**同时学** option 的内部策略与终止函数，以及"option 之上的策略"。
- **关键假设**：终止函数可微地参数化；option 的数量先验给定；环境的"技能"结构可通过梯度发现。
- **对我们**：❌ 同上。唯一有意义的借用是**术语**：我们的"束解码器"其实就是一个"终止条件固定（≤3 个 op 或金币耗尽）的 option"。

**C. FeUdal Networks** — Vezhnevets et al., ICML 2017。链接：<https://arxiv.org/abs/1703.01161>（已验证）。
- **核心思想**：Manager 在低时间分辨率上输出**目标向量**（不是动作），Worker 在每步产生原始动作去实现该目标；两者用"目标·状态差分方向"做内在奖励解耦。
- **关键假设**：状态空间有可用的**方向性表示**（state = 位置/特征向量，goal 是"往哪走"）；子目标是连续可插值的。
- **对我们**：❌。我们的状态没有"目标向量可插值"的语义（"把塔建在 (x,y)"不是状态空间里的方向）；且 50HP 的对抗游戏里"跨回合目标"更像**开局 build order**，是策略层面的东西，用 Hierarchical 结构反而把 10 秒决策预算浪费在抽象层。
- **可以留的一个后门**：把"开局 3-5 回合的 build order"当成一种 macro-action 做**跨回合**的层次结构（Manager 选 build order，Worker 选具体格）——但这属于 C 线/长期工作，与本次"束的动作表示"不同题。

> **为什么这一节要写得这么负面**：任务书把 HRL 列为候选代表作，但 options/FeUdal 的收益机制是"时间抽象 + 信用分配"，我们的痛点却是"同一时刻的因子组合 + 硬约束 + 联合提案缺失"。把两者混为一谈会导致改错模块。

---

### 2.2 因子化 / 自回归动作生成（**核心节**）

**A. Action Branching / BDQ** — Tavakoli, Pardo & Kormushev, AAAI 2018。链接：<https://arxiv.org/abs/1711.08946>（已验证）。
- **核心思想**：一个共享决策模块 + 每个动作维一条分支，输出数随自由度**线性**增长；总 Q 由各分支价值聚合（论文用均值）；每维独立贪心选择。
- **关键假设**：**各维可独立分解**（Q ≈ 聚合各分支 Q），维与维之间没有硬耦合；探索可以按维进行。
- **对我们**：⚠️ 半借用。**借**"共享主干 + 每因子一头"的结构（我们已经是 `action_map_conv` + 3 个类头）；**不借**"独立贪心"。理由：我们第二座塔的合法集取决于第一座塔花了多少金币，独立分支会给出"每维都合法、合起来超预算"的束 ⇒ **判负**。这是 BDQ 的假设与我们的约束直接冲突的地方，必须显式说出来。
- **可借的一处细节**：BDQ 论文强调"共享决策模块对协调各分支很重要"——这与"每维一条独立网络"相比更省参数，也是我们现有结构的正当性来源。

**B. AlphaStar 的自回归动作头** — Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575 (2019)。
链接：<https://doi.org/10.1038/s41586-019-1724-z>（**正文未打开**，见 §5）。
配套的动作空间规范（开放、可读）：*StarCraft II: A New Challenge for Reinforcement Learning* (SC2LE)，<https://arxiv.org/abs/1708.04782>（已验证）。
- **核心思想**：动作不是一个"大离散下标"，而是一棵**参数树**：先选动作类型（function），再依次选它的参数（选哪个单位、哪个位置、是否排队……）；策略按固定顺序**自回归**地逐步生成，每步只在**当前仍合法的参数集**上归一化。搜索与蒸馏都建在这套分解上。
- **关键假设**：① 参数顺序固定且可预先定义；② 每一步的合法集能**精确、廉价**地算出来；③ 参数条件独立近似够用（自回归只建模"给定前序参数"的条件，不建模全联合）。
- **对我们**：✅ **这是最该抄的形态**，而且几乎是"改名版"的我们：
  - 我们的 `(类, 格)` 就是 (function, argument)；
  - 我们的 `make_position_masks` 就是"每步合法集"；
  - 我们缺的只是**把 3 个头的独立采样改成顺序自回归**（头 2 的输入包含头 1 的选择），以及**每步重算 mask**。
- **注意**：SC2LE 论文给的是**动作规范（spec）**，不是 AlphaStar 的实现细节。若要把"自回归"作为设计依据，先读 SC2LE 的 action spec 章节（公开、可读），AlphaStar 正文的方法部分作为补充（本次未能读到，标为待核）。

**C. Pointer Networks** — Vinyals, Fortunato & Jaitly, NeurIPS 2015。链接：<https://arxiv.org/abs/1506.03134>（已验证）。
- **核心思想**：输出是"指向输入序列某个元素的指针"，因此**每步的输出字典大小随输入长度变化**；用 attention 当指针，天然泛化到训练时没见过的规模。
- **关键假设**：解可以从输入集合中"选元素"构成（排序、凸包、TSP 这类）。
- **对我们**：⚠️ 我们已经在用它的等价物（`action_map` 的类内 softmax + mask = "在合法格集合上做指针"）。真正的增量价值不在单步指针，而在**多步指针 = 集合生成**（连续指 k 次，指过的加 mask），这正是 §3 草案 B 的形状。另外它的"泛化到更长输出"性质，对"以后塔位变多/空格变多"的鲁棒性是好消息。

**D. Order Matters** — Vinyals, Bengio & Kudlur, ICLR 2016。链接：<https://arxiv.org/abs/1511.06391>（已验证）。
- **核心思想**：当输出本质是**集合**时，"排成什么序列"是模型的一个**人为假设**，会显著影响学习；论文用"训练时搜索顺序（或对顺序取期望）"来消除这个假设带来的偏差。
- **关键假设**：集合元素之间没有天然的先后；任务对顺序不敏感。
- **对我们**：✅ **直接命中一个我们已有但没命名的坑**：引擎对束内 op 的顺序**不可见**（整回合一次 `apply_operation_list`），但自回归解码器会强加一个顺序（比如"先建塔再降级"）。若不处理，模型会学到"顺序偏好"这种纯噪声，还会让 `p(bundle)` 的估计依赖排序约定。
  - 最小处理：束标签按**规范顺序**（如按 `(类, x, y)` 字典序）排序后再训练；更好的是对束的**多种排列**做 loss 平均（Order Matters 的做法）。
  - 这条对我们**立刻可用**，成本是数据管线里加一个排序/多重排列，不改网络。

**E. DouZero（集合动作的工程模板）** — Zha et al., KDD 2021。链接：<https://arxiv.org/abs/2106.06135>（标题/编号已由 ar5iv 页面确认：<https://ar5iv.labs.arxiv.org/html/2106.06135>）。
- **核心思想**：动作是"一手牌"= 一个**集合**；用 4×15 的 one-hot 矩阵编码该集合，LSTM 对该编码打分（而非在组合空间里枚举）；配合动作候选生成（合法牌型）。
- **关键假设**：动作集合可从局面**廉价枚举**（斗地主的合法牌型可由规则生成），且集合的编码是定长的。
- **对我们**：✅ 工程模板价值：**用定长张量编码一个 op 集合，让网络给整束打分**，而不是让网络"生成"整束。这正是我们价值网该做的事，也是 §3 草案 A 里"用价值网重排候选束"的依据。

---

### 2.3 大 / 组合动作空间的候选化（把 argmax 变成优化问题）

**A. Wolpertinger** — Dulac-Arnold et al., 2015（ICML 2016）。链接：<https://arxiv.org/abs/1512.07679>（已验证）。
- **核心思想**：把动作嵌入连续空间，用**近似最近邻**检索候选 → 对候选精确打分 → 输出；复杂度对动作数是**亚线性**（对数级），可处理百万级动作。
- **关键假设**：动作空间有**度量结构**（相似动作的 Q 值相近，即 Q 在嵌入空间 Lipschitz），且能学到一个有用的动作嵌入。
- **对我们**：❌ 作为算法不适用——我们的 (格, 塔型) 没有自然嵌入："在 (3,4) 建 Basic" 与 "在 (3,5) 建 Basic" 可能价值差异巨大（相邻塔的覆盖完全不同），Lipschitz 假设不成立。
  - ✅ 但它的**骨架**（便宜提案 → 精确重排）我们已经在用（266 候选 + 价值网）。这条的主要价值是给我们一个"文献先例"：**候选化不是妥协，是标准做法**。

**B. RL with Combinatorial Actions（MIP-as-action-selection）** — Delarue, Anderson & Tjandraatmadja, NeurIPS 2020。链接：<https://arxiv.org/abs/2010.12001>（已验证）。
- **核心思想**：值函数型 RL 里"给定 Q 求 argmax"在组合动作空间上是 NP-hard 的，于是**显式把动作选择写成混合整数规划（MIP）**，用 OR 求解器（或近似）求解；策略改进 = 用 MIP 解出的动作做近似策略迭代。示例任务：CVRP 里"下一步动作 = 构造一整条路径"。
- **关键假设**：① Q（或价值）可以**分解成可线性化的项**（否则 MIP 没有目标函数）；② 求解 MIP 的时间可接受（他们允许离线/宽松时限）；③ 约束是线性的（容量、覆盖）。
- **对我们**：✅✅ **形态上最贴"金币预算 + 一次多座塔 + 硬约束"的方法**。把"本回合下哪些塔"写成 0-1 规划：

  ```
  max  Σ_i v_i·x_i + Σ_{i<j} w_ij·x_i·x_j      （v = 单塔价值, w = 成对协同）
  s.t. Σ_i cost_i·x_i ≤ coins                   （金币预算）
       每格至多一座：Σ_{i∈cell} x_i ≤ 1         （塔位）
       超武每回合 ≤1：Σ_{i∈superweapons} x_i ≤ 1 （冷却/唯一性）
       x_i ∈ {0,1}，且 x_i 只对合法 (格,型) 存在
  ```
  它把"枚举 266 个候选束"换成"在联合空间里按约束**选集合**"，理论上前者漏掉的"单项平庸、成对很强"的解可以被找到。
  - **风险（必须写清）**：`w_ij`（成对项）我们没有现成的、可信的估计；若只用 `v_i`（单塔打分之和），这个 MIP 就**退化成 unary**，等价于"按单塔分从高到低买"，协同仍然丢失。所以这条的真正难点是**评估**而不是优化——与项目当前"瓶颈在评估"的判断一致（[prior_ceiling_20261026.md](../prior_ceiling_20261026.md) §五）。
  - **便宜版本**：先做"限制在 top-k 原子候选上的 knapsack"（k≈16-32），用**现有价值网**给整束打分做 `v`，只用 MIP 处理预算/互斥，不做成对项。这就是草案 A。

**C. OpenAI Five（Dota 2）** — Berner et al., 2019。链接：<https://arxiv.org/abs/1912.06680>（已验证标题/摘要）。
- **核心思想（与本主题相关部分）**：动作空间是**组合式**的（选目标、选动作、…），且每步可用动作只是全空间的一小部分；策略在可用子集上采样，并配合大规模分布式自对弈。
- **关键假设**：环境能廉价地给出"当前可行动作"。
- **对我们**：✅ 工业先例（"候选 + 掩码"在超大动作空间是可行的工程路线）。**具体数值（每步动作数、掩码实现）本次未核实**，不要把细节数字当引用。

---

### 2.4 动作掩码与硬约束

**A. Invalid Action Masking** — Huang & Ontañón, FLAIRS 2022（arXiv 2020）。链接：<https://arxiv.org/abs/2006.14171>（已验证）。
- **核心思想**：策略梯度算法里"把非法动作 mask 掉、只在合法集上采样"是标准做法，但此前缺少分析。论文做了三件事：① 给出该做法的**理论正当性**；② 实验显示**非法动作占比越大，掩码越重要**；③ 评估"训练时用掩码、之后去掉掩码"的不同 regime，发现**训练后去掉掩码会变差**。
- **关键假设**：掩码只依赖状态（不依赖采样结果）；每个状态至少有一个合法动作。
- **对我们**：✅✅ **必做，且我们已经在做一半**。三条具体含义：
  1. **掩码必须进 `logπ`**：rollout 采样分布 = 掩码后的分布，训练算 `logπ` 就必须用同一分布（我们踩过的"采样侧 ÷100、训练侧不归一化"是同一类不对称错误的另一种表现，见 [az_policy_loss_scale_fix_plan.md](../az_policy_loss_scale_fix_plan.md)）。
  2. **掩码后的策略仍是合法策略**：不必因为"掩码改变了分布"而担心策略梯度失效；但**被掩掉的 logits 拿不到梯度**，长期会让这些输出通道"饿死"（与我们已记录的"超武通道难覆盖"一致，见 [az_posnet_random_class_plan.md](../az_posnet_random_class_plan.md)）。
  3. **全掩码必须显式处理**：某个状态下"三个头都没有合法动作"时，掩码后的 softmax 是 0/0。我们的 `decode_head` 返回 `None` 就是这条路径——它必须被定义成 **HOLD（显式 STOP）**，而不是"解出来是空束"（这两者在训练标签上语义不同，会让 HOLD 头学到噪声）。
- **本次未逐条核读定理部分**（只读了 abs 与二手摘要），若要在文档/答辩里引用定理形式，请先读 §3 的定理陈述。

**B. Shielding** — Alshiekh et al., AAAI 2018。链接：<https://arxiv.org/abs/1708.08611>（已验证）。
- **核心思想**：为"安全规范"（时序逻辑）合成一个**反应式 shield**，有两种插入方式：① 每次决策前给出**安全动作列表**（等价于我们说的 mask）；② 放在策略之后，**只纠正**会导致违规的动作。论文还讨论了 shield 需要满足什么条件才能**保持学习器的收敛保证**。
- **关键假设**：安全规范可形式化、可在线判定；违规是"可判定的坏动作"。
- **对我们**：✅ **这是"判负"场景的正确语义**（比 CPO 贴）。要点：
  - 我们的 spec 就是"本回合结束时金币 ≥0、塔位不冲突、超武不超用、每 op 在当步合法"；
  - **shield 必须对搜索的每个子节点生效**，不能只在根节点掩码（否则树里会长出非法分支，评估被污染）；
  - 论文的"保持收敛性"条件提醒：shield 若**依赖探索历史的隐状态**（比如"这一回合已经用过闪电"），就属于"非马尔可夫的纠正"，会让策略学到"反正会被纠正"的侥幸行为——我们的 `used_tower` / `camp_upgraded_flag` 正好是这种回合内状态，**必须进状态/进 mask，不能靠事后纠正**。

**C. CPO（Constrained Policy Optimization）** — Achiam et al., ICML 2017。链接：<https://arxiv.org/abs/1705.10528>（已验证）。
- **核心思想**：在 CMDP 上做策略搜索，用 trust-region 保证**每次迭代**的约束满足程度"近似不退化"（每步都在约束边界附近，允许小违反）。
- **关键假设**：约束是**期望意义**上的（期望代价 ≤ 阈值），允许偶尔违反；有可估的代价函数。
- **对我们**：❌ 不推荐。理由：① 我们是**一次性致命**约束，不是期望代价，"每次迭代近似满足"没有意义；② 需要额外的代价 critic 与乘子，10 秒决策预算里不划算；③ 惩罚项会让"稍微违规"变得有吸引力（在 PPO 里表现为偶发非法招），而我们**偶发一次就输**。
  - 唯一例外：将来若要表达"用血量换经济"这类**可交易**的软约束，才考虑 CMDP。

---

### 2.5 集合动作（set-valued action）与稀疏奖励

**A. Deep Sets** — Zaheer et al., NeurIPS 2017。链接：<https://arxiv.org/abs/1703.06114>（已验证）。
- **核心思想**：置换不变的集合函数必须具有 Σφ(x) 的分解形式（论文给出充要条件与等变版本）。
- **关键假设**：元素独立同分布地贡献；不做元素间交互建模。
- **对我们**：✅ **最便宜的条件输入**：把"本回合已选 op 的嵌入"做 mean/max pooling，拼进第二段特征。一行代码级别的改动，就能让第二个 op 的预测**知道**第一个 op 是什么（一阶条件化）。局限：pooling 丢掉"谁和谁配对"，只给"已选了什么"的摘要。

**B. Set Transformer** — Lee et al., ICML 2019。链接：<https://arxiv.org/abs/1810.00825>（已验证）。
- **核心思想**：用 attention 建模集合**元素之间**的交互，并用 inducing point 把 self-attention 从二次降到线性。
- **关键假设**：交互可以用少量 inducing points 概括；集合规模中等。
- **对我们**：✅ 若要用"已选集合 + 候选格集合"做**交互建模**（"这个格和我已选的塔相邻吗？"），这是标准的轻量做法。但对我们的规模（≤3 个已选 op、候选 ≤48 个原子）用一层 attention 就够，不需要 inducing point。

**C. 顺序集合生成（sequential set generation）** — Order Matters <https://arxiv.org/abs/1511.06391>、Pointer Networks <https://arxiv.org/abs/1506.03134>（均已验证）。
- **核心思想**：集合不直接一次输出，而是**逐个指/逐个生成**，每次在剩余候选上加 mask；顺序问题由 Order Matters 那套处理。
- **关键假设**：元素数有上界（或学到 STOP）；每步候选集可算。
- **对我们**：✅ 这就是草案 B 的骨架，也是**唯一能真正表达"p(op₂ | op₁)"**的形状。

**D. CMAB（组合多臂老虎机）+ triggered arms** — Chen, Wang, Yuan & Wang, ICML 2013 / JMLR 2017。链接：<https://arxiv.org/abs/1407.8339>（已验证）。
- **核心思想**：把"选一个 base arm 子集（super arm）"作为一步决策；**semi-bandit 反馈**（能看到 super arm 内每个 base arm 的结果）；在存在一个 **(α,β)-近似 oracle**（能对均值给出 α 近似最优的 super arm）的前提下，CUCB 达到 O(log n) 的 (α,β)-近似 regret；并扩展到"被触发的臂"（outcome 会概率性触发更多臂）。
- **关键假设**：有一个离线近似 oracle 能解组合优化；reward 对 base arm outcome 满足单调性/有界性；feedback 是 semi-bandit（能拆开看每个 base arm）。
- **对我们**：⚠️ **借语言，不借算法**。映射很清楚：我们的"束" = super arm，金币 = knapsack 约束，价值网/rollout = oracle，(α,β) = 我们的候选生成器漏掉好解的程度。
  - 它给的最有用的一句话是：**regret 的常数由 oracle 的 α 决定**。换句话说，**候选生成器的覆盖率（α）比探索策略更决定上限**——这与项目实测"先验 @24 建塔 ≈ 随机（74.9% vs 74.1%）"合起来读就是：我们的 α 在"建塔"这一维上几乎等于 1 的平凡值（因为支撑集只有 32 格，取 24 个当然 75%），**真正的 α 是在"多 op 束"这个组合层面上的覆盖率，而我们没测过**。
  - 不直接落地为算法：它假设"重复玩同一批 super arm"，而我们的局面不重复。

**E. 稀疏奖励下的组合动作 → ExIt 式束级监督** — Anthony, Tian & Barber, NeurIPS 2017（*Thinking Fast and Slow with Deep Learning and Tree Search*，即 Expert Iteration）。链接：<https://arxiv.org/abs/1705.08439>（已由 HuggingFace papers 页面确认：<https://huggingface.co/papers/1705.08439>）。
- **核心思想**：专家（树搜索）与学徒（策略网）交替：搜索产出"改进了的策略"，网络去拟合它；网络变强后同样预算的搜索也变强。
- **关键假设**：搜索本身能改善动作选择（哪怕只在少数决策上）；有足够算力做搜索-拟合循环。
- **对我们**：✅ 我们已经在做（搜索标签 + 先验蒸馏）。**要点在"束级"**：稀疏奖励下"束内哪个 op 有功"几乎不可归因（一个束是一起结算的），所以监督信号必须是**整束一个目标**（搜索访问分布 / 价值），不要试图给束内每个 op 单独的 reward——那是组合 credit assignment，文献里没有便宜解法。这一点与项目已有的"过滤只能按轴做、不能整体过滤"的经验是一致的（[az_action_factorization_idea.md](../az_action_factorization_idea.md) §八末）。

---

### 2.6 预算分配与"先验-搜索"耦合

**A. Sequential Halving** — Karnin, Koren & Somekh, AISTATS 2013（*Almost Optimal Exploration in Multi-Armed Bandits*）。
链接：<https://proceedings.mlr.press/v28/karnin13.html>（**本次 DNS 解析失败，未打开**，见 §5）。
- **核心思想**：在**固定总预算**做"简单遗憾"下的最优臂识别时，把预算按轮**减半候选数**，比均匀分配更优（遗憾 O(√(nK log K)) 级别）。
- **关键假设**：各臂评估**独立**、方差同阶；目标是"找出最好的一个"，不是"最大化累积收益"。
- **对我们**：⚠️ 只用于**候选束排序**（266 → 128 → 64 → … → 1），不适合直接替换树搜索：我们的候选共享子树（同一束的前缀会被复用），不满足独立性假设。但"固定预算下别均匀撒"这个教训是通用的，且与 §1-5 的结论同向。

**B. ExIt** — 见 §2.5-E。链接：<https://arxiv.org/abs/1705.08439>。
- **落地要点**：策略越强，同样预算的搜索越强 ⇒ **"先验"和"搜索"不是替代关系而是乘数关系**。这直接支持项目"先修评估、不修先验"的判断（离线本身上限已到顶）。

**C. Gumbel MuZero / "Policy improvement by planning with Gumbel"** — Danihelka, Guez, Schulman & Weber, ICLR 2022。
链接：<https://www.iclr.cc/virtual/2022/poster/6418>（**本次 DNS 解析失败，未打开**，见 §5）。
- **为什么本来想引它**：它正是回答"**极少模拟次数下如何让搜索真的改进策略**"的工作（用 Gumbel 采样 + 单调策略改进保证），对我们"10 秒决策预算 / 每回合只够浅浅搜"的场景高度对口。
- **本次状态**：只确认了标题与发表场所，**没有读到内容** ⇒ 本文不基于它做任何建议；把它列入 §6"该读的下一步"，而不是当作结论依据。

**D. 工程算术（本文的分析，非文献结论）**
因子化后若"每个因子都搜"，成本近似是**各因子预算之积**（深度 × 宽度），而收益是**加性**的（每个因子的可改进量独立叠加）。项目实测：位置维 +43pp、类维 0pp ⇒ 最优分配是把预算压到**单因子**上。判据不是"哪个因子更重要"（先验说不清），而是**"哪个因子有选择空间 + 搜索能改进它"**，这正好是 `--search-mode` 三臂 + `chosen==raw` / `room` 诊断在测的东西。

---

### 2.7 同域直接相关工作

**Tower Defense 的高层策略 RL** — *Reinforcement Learning for High-Level Strategic Control in Tower Defense Games*, 2024。
链接：<https://arxiv.org/abs/2406.07980>（编号来自 NASA ADS 记录 `2024arXiv240607980B` + 标题匹配；**未读全文**，见 §5）。
- **为什么列**：与我们同域（建塔/升级/资源的高层决策），是"动作空间怎么建模"最直接的可对照工作。
- **本次状态**：只确认了标题与编号。**不要**在文档里引用它的具体结论。

---

## 3. 我们的 `action_map` 到底表达什么（表达力诊断）

### 3.1 形式化

把现有解码写成概率形式（3 个头各自采样 `(类, 格)`，再解码成束）：

```
p(bundle) ≈ Σ_{decode} ∏_{i=1..3} p_i(c_i) · p_i(x_i, y_i | c_i) · 1[decode 成功]
```

**已经表达了的**：
- **集合性**：最多 3 个 op ⇒ "一次下多座塔"在**表示层面是有的**（这一点任务书的担心部分不成立）；
- **类内位置条件**：`p_i(x,y | c_i)` 是类内 softmax + mask，位置随塔型而变；
- **HOLD**：通道 23。

**没表达的（三处，按重要性）**：

1. **因子间条件化缺失**（结构性）。头 2 的分布**不依赖**头 1 选了什么：`p_2(c_2, x_2, y_2)` 与 `(c_1, x_1, y_1)` 独立。因此：
   - "如果在这里建 A，那旁边就必须建 B"这种**联合高峰无法用提案表达**；
   - 但**约束**（金币/塔位）是靠 decode+mask 事后补的，所以不是"生成非法束"，而是"**生成的束在分布上不相关**"；
   - 结论：`action_map` 是**提案分布**，它对联合空间的覆盖是 **unary-additive（逐原子可加）**的。**"单独看都平庸、一起很强"的解在 266 个候选里根本不会被提出**。
   - 精确说法：**策略本身（价值网 + 搜索）能表达协同，提案不能**。这就是"表达力不足"的确切位置。
2. **跨类不可比**（读数问题）。`action_map` 是**类内**归一化（每个通道在自己的合法格上 softmax），所以 `map[3,x,y]` 与 `map[17,x,y]` 的数值**不能直接比较**（项目已记录这一点）。后果："哪个格子整体最值得动"无法从 map 直接读出；取 top-k 时必须**先按类分别取再合并**，否则大支撑集的类（超武 271 个合法格 vs 建塔 32 个）会系统性占位。
   - 注意这**不是**"应该改用联合 softmax"——项目已分析过联合 softmax 会按"合法格数量"稀释类别（同上文档）。正确做法是**分层**：先给每个类一个权重（可行性 × 类头概率），再类内 softmax，最后**每类各取 top-1 合并**。
3. **约束不在分布里**（这是设计选择，不是缺陷，但有操作要求）。合法性完全由 mask/decode 承担 ⇒ 要求"**每个因子在生成时用更新后的状态重算 mask**"。现在 3 个头是"并行"采的，mask 只在采样时按当前局面算，头 2 不会看到头 1 已花的金币 ⇒ **只有在 decode 阶段才可能发现超预算**，而 decode 失败就退化成空束（项目已观测到"pos-only 的候选全塌成空 bundle"现象，见 [az_action_factorization_idea.md](../az_action_factorization_idea.md) §八）。

### 3.2 三条修法（成本从低到高）

| 编号 | 改什么 | 解决哪一条 | 要训练吗 | 成本 |
|---|---|---|---|---|
| **M1** | 跨类候选合并：每类各取 map 的 top-1/top-2 → 合并成一个"原子候选池"（≤48）→ 用**现有价值网**统一重排 | 3.1-② 跨类不可比 | 否 | 1 次价值前向/原子候选（可批） |
| **M2** | 配对候选：从 M1 的 top-k 里生成 `(op_i, op_j)` 对，按"金币够 + 塔位不同 + 超武不重复"过滤，用价值网打分 | 3.1-① 的**后果**（候选缺失） | 否 | k²/2 次价值前向（k=16 ⇒ 120 次，需批处理） |
| **M3** | 条件化续加头：第一段出 map → 选中 `(c,x,y)` → 用该 op 的 embedding 做 FiLM/拼接 → **再出一次 map** + 一个 STOP logit | 3.1-① 的**根因** + 3.1-③ | **是** | 2-3× 位置头前向；需束级标签 |

### 3.3 一个反方向的提醒（避免过度设计）

项目数据说：**束中位 1 个 op、均值 1.28、最多 3**（[az_action_factorization_idea.md](../az_action_factorization_idea.md) §三①）。也就是"一次下多座塔"在真实对局里**很少发生**。所以：
- M2/M3 的**收益上限受"多 op 束的发生率"限制**；
- 但**不能因此不做**：发生率低可能是**策略/先验从来没提出过组合**造成的（自证预言），而不是组合本身没用。要分辨，就得先做 M1/M2（零训练）把候选池扩开，再看"组合候选被价值网选中的比例"。

---

## 4. 落地草案

### 草案 A（首推，零训练，1-2 天可出数）：M1 + M2 + 预算内集合选择

**改哪个模块**：`code/my_ai/decoder.py` 的候选生成 + `bundle_mcts` 的候选菜单构造（只在**推理/搜索**侧；不动 `network.py`、不动训练、不动正在跑的实验，因为它是新开关/新函数）。

```python
# 伪代码：原子候选池 + 配对候选 + 预算内选集合（不训练）
def build_atomic_candidates(state, net_out, player, k_per_class=1):
    """返回 ≤48 个 (class, x, y, cost, logp) 原子候选（M1）"""
    cands = []
    for c in range(24):
        mask = make_position_mask(state, c, player)      # 每类自己的合法格（已有）
        if not mask.any():
            continue
        top = topk_masked(net_out.action_map[c], mask, k=k_per_class)   # 类内 top-1
        for (x, y), v in top:
            cands.append(Atom(class_=c, x=x, y=y,
                              cost=op_cost(state, c),      # 金币/冷却/唯一性
                              logp=v - logsumexp_masked(net_out.action_map[c], mask)))
    return merge_and_dedup(cands)          # 去重、按 logp 粗排（跨类用类头概率加权）


def build_pairs(atoms, state, coins, max_pairs=256, radius=2):
    """返回可行的 (op_i, op_j) 对（M2）：解决'单看平庸、成对强'的候选缺失"""
    pairs = []
    for a, b in combinations(atoms, 2):
        if a.cost + b.cost > coins:            continue      # 预算硬约束
        if (a.x, a.y) == (b.x, b.y):           continue      # 塔位互斥
        if is_superweapon(a) and is_superweapon(b): continue # 超武每回合一次
        if hex_dist(a, b) > radius:            continue      # 只留局域组合，控成本
        if not bundle_legal(state, [a, b]):    continue      # 用引擎/现有 guard 二次校验
        pairs.append(Bundle([a, b]))
    return pairs[:max_pairs]


def choose_bundle(state, net_out, player, coins, n_sims):
    atoms = build_atomic_candidates(state, net_out, player)
    cands = [Bundle([a]) for a in atoms] + build_pairs(atoms, state, coins)
    cands = cands + [Bundle([])]                      # 显式 HOLD，别用 decode 失败代替
    # 1) 便宜预筛：单塔分之和（unary）+ 先验 logp 之和
    cands = topk(sorted(cands, key=lambda B: unary_score(B) + prior_score(B)), 64)
    # 2) 贵评估：现有价值网 / rollout 打分（只在 64 个上做）
    vals = batch_value(cands, state, player)          # 一次批量前向，别逐个
    # 3) 预算内选集合：k 很小，直接按"束级价值 - λ·金币占用"贪心/小规模 DP
    return argmax_with_knapsack(cands, vals, coins)
```

**为什么先做这个**：① 它不动网络和训练，**不碰正在跑的实验**；② 它正面回答了"表达力不足"的**可见后果**（候选里没有协同项）；③ 评估预算可以自己定（64 个），与 §2.6 的"别均匀撒预算"一致。

**收益预期（写成可证伪的形式）**：
- 若"组合有用"，则新候选池里**亚军实际束的召回**应当上升（现在建塔 @24 与随机持平 = 74.9% vs 74.1%，说明还有空间）；
- 若"组合没用"，则价值网会**几乎从不选中配对候选**（这是一个干净的零假设，可以直接量）。

**风险**：
1. 成本：配对候选的价值评估次数 ~120-256 次/决策 ⇒ 必须**批处理**，否则 10 秒预算不够（项目已记录单次特征提取 2.8ms 量级，见 [cpp_engine_rewrite_research.md](../cpp_engine_rewrite_research.md)，要按批量重算）。
2. 评估噪声：价值网对"没见过的组合"外推差，可能给出**系统性高估**（OOD 吸引），导致"选了配对但打得更差"。缓解：把配对候选的分数做**保守化**（减一个固定 margin / 用序数而非绝对值），并先跑小样本 A/B。
3. 判负风险：`bundle_legal` 必须复用**引擎同源**的校验（项目已有 `decode` 的 None 路径 + `cold_handle_rule_illegal` 的影子统计需求，见 [prior_ceiling_20261026.md](../prior_ceiling_20261026.md) §六-2）。**宁可多滤，不可漏滤。**

---

### 草案 B（次推，要训练）：条件化续加头（M3）

**改哪个模块**：`code/my_ai/network.py` 加 `cont_head`（STOP logit）+ FiLM 条件化；`decoder.py` 改成循环；`distill`/`ss_train` 的目标加"束级 STOP 标签"与"束内规范排序"。（这条**必须在 A 出数之后**，且需要新采集。）

```python
# 伪代码：自回归束生成（AlphaStar 形状 + Order Matters 的顺序处理）
def ar_bundle_policy(state, net_out, player, max_ops=3):
    feat = backbone(state)                    # 已有
    ops, coins = [], state.coins[player]
    for t in range(max_ops):
        logits = map_head(feat)               # 24 × 19 × 19（复用同一头）
        mask = legal_mask(state_after(ops), coins)   # ★ 每步重算：预算/塔位/唯一性
        if not mask.any():
            break
        c, x, y = sample_or_argmax(logits, mask)
        op = decode(c, x, y, state_after(ops))
        if op is None:  break                 # 解码失败 ⇒ 当作 STOP，不当空束
        ops.append(op); coins -= cost(op)
        feat = film(feat, embed(op))          # ★ 条件化：把 op 注入特征图
        if sigmoid(cont_head(feat)) < 0.5:    # ★ 学出来的"还要不要追加"
            break
    return Bundle(ops) if ops else HOLD       # 显式 HOLD
```

**训练目标**（关键：束级、顺序无关）：
```
L = λ_pos · CE(逐 op 的 (类,格) 软目标)             # 已有 visit/蒸馏目标
  + λ_stop · BCE(cont_head, "该束是否还有第 t+1 个 op")   # 由数据自动生成
  + λ_prior · KL(新 map || 旧 map)                       # anchor，防漂移（项目已有先例）
# 束内 op 顺序：按 (类,x,y) 字典序规范排序后训练；或对 2-3 种排列取 loss 平均（Order Matters）
```

**收益**：真正表达 `p(op₂ | op₁)`；每步重算 mask ⇒ 硬约束在**生成时**就成立（不再依赖 decode 兜底）；STOP 头给 HOLD 一个**显式**语义（修掉"空束 vs HOLD 混淆"）。
**风险**：① **STOP 头坍缩**——项目有过"类头坍缩到闪电 / 91% HOLD"的先例，加一个二分类头很容易学到"总是 STOP"（因为束中位就是 1 个 op ⇒ 数据极不平衡）。必须用正样本重采样 + 分轴统计守门（参考 [plan_oversample_rare_actions.md](../plan_oversample_rare_actions.md) 的做法）。② **数据饥饿**：现有蒸馏数据 80 局 / 3.2 万局面，建塔 @1 只有 6.3%，加二阶条件会进一步稀释；②' 需要**先**有"能分辨 2-3pp"的评估功效（16 局分辨率 ±25pp，见 [prior_ceiling_20261026.md](../prior_ceiling_20261026.md) §八）。
**前置条件**：草案 A 已证明"组合候选有价值网能识别的信号"；否则 B 只是在拟合噪声。

---

### 明确不推荐

| 做法 | 理由 |
|---|---|
| 用 options / FeUdal / Option-Critic 当主结构 | 收益机制是"跨时间抽象"，我们的问题是"同一决策内的因子 + 硬约束"（§2.1） |
| CPO / Lagrangian 惩罚替代掩码 | 期望约束 ≠ 一次性致命；惩罚会让"偶发非法"变得有吸引力（§2.4-C） |
| 改回联合 softmax 解决"跨类不可比" | 会按合法格数量稀释类别（超武 271 格 vs 建塔 32 格），项目已分析过（§3.1-②） |
| 在类维上再花搜索预算 | 已有实测：类维增益 0pp、且 100% 回合"可换类但价值头从不认可"（[az_action_factorization_idea.md](../az_action_factorization_idea.md) §八） |
| 给束内每个 op 单独的奖励 | 组合 credit assignment，束是一次结算的；只能束级监督（§2.5-E） |

### 怎么验证（复用项目已有装置，别发明新指标）

| 判据 | 装置 | 读法 |
|---|---|---|
| 候选池召回 | 复刻 `recall_on_runnerup_v3.py` 的思路，对**束**（而非单 op）算召回 | 现在建塔 @24 与随机持平 ⇒ 有空间；若配对候选把召回抬起来，说明候选缺失是真问题 |
| 搜索是否真在改动作 | `chosen==raw` + `room`（候选数 ≥2 的回合占比） | A 之后 `room` 应上升；若 `chosen==raw` 仍 ~97%，说明价值网不认组合 |
| 强度 | `search-vs-raw`（配对种子） | **注意**：历史上"完全失效"就是 50%，不要把 >50% 当合格线（[az_action_factorization_idea.md](../az_action_factorization_idea.md) §四） |
| 样本量 | 按座位拆分、固定地图/`FLAT_SEED` | 座位效应 27pp > 先验差异（[prior_ceiling_20261026.md](../prior_ceiling_20261026.md) §4.4）；6%→12% 需要 ~350 局/臂 |

---

## 5. 不确定 / 访问受限清单

**A. 网络访问受限（本次会话 DNS 解析失败，未读到正文）**

| 文献 | 链接 | 状态与影响 |
|---|---|---|
| AlphaStar（Nature 2019） | <https://doi.org/10.1038/s41586-019-1724-z> | `doi.org` / `www.nature.com` 解析失败。**"自回归生成动作"这条细节我未在正文核实**，来自二手材料与对该系统的一般理解。若要把"自回归"作为设计依据，请先读 SC2LE(<https://arxiv.org/abs/1708.04782>) 的 action spec（公开可读）+ AlphaStar 方法补充材料 |
| options 原文（Sutton, Precup & Singh 1999） | <https://doi.org/10.1016/S0004-3702(99)00052-1> | 未打开。**DOI 号按记忆填写，请核对后再正式引用**。技术内容我以 Option-Critic（已验证）为锚 |
| Sequential Halving（Karnin et al. 2013） | <https://proceedings.mlr.press/v28/karnin13.html> | PMLR 解析失败。遗憾界只按"O(√(nK log K)) 级别"的常识描述，**未逐条核对** |
| Gumbel MuZero / *Policy improvement by planning with Gumbel* | <https://www.iclr.cc/virtual/2022/poster/6418> | `iclr.cc` 解析失败。**只确认了标题与发表场所**，正文未读 ⇒ 本文**未**基于它给建议，仅列入 §6 |

**B. 只确认了标题/编号、未读全文**
- *Reinforcement Learning for High-Level Strategic Control in Tower Defense Games* (2024)：<https://arxiv.org/abs/2406.07980>。编号来自 NASA ADS 记录 `2024arXiv240607980B` + 标题匹配；具体方法与结论**未核实**，请勿引用其细节。
- OpenAI Five（Dota 2）：<https://arxiv.org/abs/1912.06680>。只读了摘要；"每步动作规模 / 掩码实现细节"未核实，正文中已避免具体数字。
- Huang & Ontañón 的**定理陈述**：只读了 abs + 二手摘要，未逐条核读定理形式与"掩码后策略梯度"的精确条件。

**C. 本文属于我的推理、不是文献结论的部分**
- §1-5 的"成本相乘、收益相加"工程算术；
- §2.6-D 的预算分配结论；
- §3.1 的表达力形式化（把 3 头独立采样写成乘积形式）——这是我的读码/读文档推论，**未与代码逐行核对**（`decoder.py` 里 3 个头是否存在隐式依赖，我只从文档推断为"独立"）；
- §4 两个草案的收益估计。

**D. 未查到的内容（明确写"未找到"）**
- 专门针对"**六边形网格塔防 + 组合建塔动作**"的 RL 论文：**未找到**（塔防相关工作以 §2.7 那一篇为代表，且未读）。
- "**set-valued action + 硬约束 + 稀疏奖励**"三者同时处理的专门方法：**未找到**。文献要么做组合动作（MIP/CMAB），要么做掩码，要么做集合生成（seq2set），三者交叉的工作我没有检索到。这可能就是我们这个题目的空白点。

---

## 6. 建议的下一步（读什么 / 测什么，按性价比排序）

**读（各 30-60 分钟）**
1. SC2LE <https://arxiv.org/abs/1708.04782> 的 action spec 章节 —— 确认"参数树 + 每步合法集"的工程细节；
2. Huang & Ontañón <https://arxiv.org/abs/2006.14171> 的定理部分 —— 我们要在 PPO 里正确接入 mask 的 logπ；
3. Delarue et al. <https://arxiv.org/abs/2010.12001> 的 MIP 表述 —— 若要做草案 A 的"预算内选集合"的严格版；
4. （可选）Gumbel MuZero —— 若"每回合只能浅浅搜"成为主要瓶颈，这是最对口的文献。

**测（都是小实验，不碰主训练）**
1. **束内 op 数分布 + 组合候选的价值分布**：现有 266 候选里，价值网对"2-op 束"的打分与"其两个 1-op 束之和"差多少？差值大 ⇒ 存在协同信号，M2 值得做；差值≈0 ⇒ 协同在本游戏里可加，根本不需要 M2/M3。
2. **M1 的跨类合并**：只改候选生成（每类 top-1 合并），看 `room` 与 `chosen==raw` 怎么变。这是最便宜的一步，且能验证"跨类不可比"是否真的在伤害候选质量。
3. **束级召回**：把 `recall_on_runnerup_v3.py` 从"单 op"扩到"整束"，得到"我们候选池对亚军束的覆盖率 α"——这是 CMAB 视角下真正决定上限的量，而我们现在**没测过**。
