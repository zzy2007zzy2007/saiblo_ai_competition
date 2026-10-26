# D. 先手承诺 / 后手回应：序列博弈文献调研

> 目的：为 Ant-Game（六边形网格塔防对抗）里"同一套 AI 坐 P1 胜率 72%、坐 P0 胜率 25%"这一现象找**正确的建模语言**与**可落地的改法**。
> 约束：只做调研，不写产品代码；所有结论附链接；**证据分级**标在每条后面。
> 证据分级说明：
> - **【正文】** = 我读到了论文/文档正文（PDF 抽文本、HTML 全文或页面抽取）；
> - **【仅摘要/元数据】** = 只读到标题、摘要或 DOI/Crossref 元数据，正文未读；
> - **【二手】** = 只从别的材料转述得知，未直接核到原文；
> - **【项目内】** = 本仓库历史实验数据（附文件:行）。

---

## 0. 先把形式化写对（问题 1 的核心）

### 0.1 这局到底是什么博弈

一回合的协议（`Ant-Game/README.md:359-364`、`Ant-Game/README.md:292-298` 的胜负判定链）是：

```
回合 t：
  1) P0 提交操作列表 a0        （P0 不知道 P1 本回合会做什么）
  2) 平台验证并执行 a0，把 a0 转发给 P1
  3) P1 观测到 (局面 s_t, a0) 后提交 a1
  4) 平台执行 a1，双方操作按序应用，然后整回合结算 → s_{t+1}
```

写成扩展式博弈：

```
G = ⟨N={0,1}, H, Z, A, P, u⟩
- 状态 s_t ∈ S，对双方**完全可见**（没有隐藏信息/暗牌）
- P0 的信息集 I0(s) 是单点；P1 的信息集 I1(s, a0) 也是单点
- 确定性转移 s_{t+1} = f(s_t, a0, a1)（引擎结算；含随机但由 seed 决定）
- 终局 z：u_0(z) ∈ {+1, 0, −1}，等血时按字典序裁决
      血 → 击杀数 → 超级武器使用次数 → AI 用时 → 先手(P0)胜
```

**关键判断：所有信息集都是单点 ⇒ 这是一个"完全信息扩展式博弈"（perfect-information extensive-form game），不是不完全信息博弈。** 所以：

- 解概念是 **子博弈完美均衡（SPE）**，不是"CFR 收敛到的 Nash"；
- 算法是**逆向归纳 = min-max 搜索**；
- 每回合的均衡条件是

```
V(s_t) = max_{a0 ∈ A0(s)} min_{a1 ∈ A1(s,a0)} V(s_{t+1})        （P0 视角；P1 节点取 min）
P0 的均衡首着 a0* = argmax_{a0} min_{a1} V( f(s,a0,a1) )
```

这就是你说的"**我们一步 + 对手一步**"的双层搜索——它不是启发式，它就是 SPE 的一步逆向归纳。**我们现在"rollout 里对手 idle 或按固定概率随机动作"，数学上等于把 min 层换成了一个固定策略，即解的不是 SPE，而是"单人规划问题 + 一个假对手模型"。**

参考（定义层面）：
- 子博弈完美均衡的定义与"在每个子博弈里都是 NE、要求序贯理性"：【正文】<https://plato.stanford.edu/entries/game-theory/>（SEP, *Game Theory*）；【正文】<https://en.wikipedia.org/wiki/Subgame_perfect_equilibrium>
- 完全信息有限博弈必有值、SPE 存在（Zermelo 1913 / Kuhn 1953 / Selten 1965 这条线）：【正文】同上 SEP 条目

### 0.2 那"Stackelberg / 承诺博弈"这一套怎么对应？

**先给标准定义（读到的原话）。** arXiv:2111.02111 摘要 + 引言：

> "Stackelberg equilibrium is a solution concept that describes optimal strategies to commit: Player 1 (the leader) first commits to a strategy that is publicly announced, then Player 2 (the follower) plays a best response to the leader's commitment."
> "In this setting, the leader commits to a **randomized strategy in each possible decision point** that can arise in the game and the follower plays a best response to this commitment."
> —— 【正文】<https://arxiv.org/abs/2111.02111>（*Computing Stackelberg Equilibrium with Memory in Sequential Games*）

这段话里有两个容易漏掉、但对我们至关重要的点：

**(a) 标准 SE 里 leader 承诺的是"完整、公开宣布的策略"，不是"一手"。** 也就是说 leader 承诺的是"在每个可能到达的决策点我会怎么动"的整张表（可随机化）。follower 是对**这张表**做最佳回应。
我们 P0 **没有承诺装置**：它每回合只宣布本回合的操作，下一回合可以完全反悔。所以严格讲：

> **我们的博弈不是 Stackelberg 博弈，而是"完全信息序列博弈 + 无承诺能力"的博弈。** 它的解是 SPE（逆向归纳），不是 SE。

这一点在文献里也有对应说法：SE 会指定**不可信（off-equilibrium-path）**的承诺，所以需要额外的 refinement（quasi-perfect SE）——

> "Akin to usual extensive-form game refinements, Stackelberg equilibrium (SE) refinements should guarantee both the optimality of the commitment off the equilibrium path and some form of robustness against small trembles of the opponent."
> —— 【正文】<https://arxiv.org/abs/1811.03871>（*Quasi-Perfect Stackelberg Equilibrium*）

这告诉我们一件反直觉的事：**"P0 先出招"并不自动等于"P0 有承诺优势"**。承诺优势来自"我能把自己绑住、而且你相信我绑住了"。我们既没有绑定装置，P1 也知道我们下一回合能反悔——所以 P0 在理论上不吃承诺红利，只承担"先暴露"的成本。（这与游戏里 HOLD 的处境一致：HOLD 是唯一"不新增一个可被针对性反制的具体承诺"的动作。）

**(b) 混合策略的价值，恰恰依赖"follower 回应的是分布"。** 标准模型里 follower 是对 leader 宣布的**随机化策略**做最佳回应；而我们的 P1 看到的是**实现值**（P0 这一手实际出了什么）。这两者的差别是决定性的：

```
若 P1 回应"分布 σ"：  leader 收益 = max_σ min_{a1} E_{a0~σ}[ V(a0,a1) ]     ← 混合可能 > 纯动作
若 P1 回应"实现值 a0"：leader 收益 = max_σ E_{a0~σ}[ min_{a1} V(a0,a1) ]
                                  ≤ max_{a0} min_{a1} V(a0,a1)            ← 混合必然 ≤ 最好的纯动作
```

第二行是不等式（不是等式）：对任意混合 σ，被求期望的每一项 `min_{a1} V(a0,a1)` 都不超过 `max_{a0} min_{a1} V(a0,a1)`，所以期望也不超过它。**结论：在我们的协议下，P0 的随机化在理论上弱劣势（weakly dominated）；它只能用来"打不完美的对手"（利用 P1 搜索的盲点、或让 P1 的跨回合模式识别失灵），不能当机制性解药。** 这一点直接回答了你的第 3 个候选方案。

### 0.2 补充：我读到的原文（PDF 抽文本，**已读正文**）

Conitzer & Sandholm, *Computing the Optimal Strategy to Commit To*（EC'06），PDF：<https://users.cs.duke.edu/~conitzer/commitEC06.pdf>【正文】。五段原文（引号内为论文原话；PDF 抽文本后可能有个别字符被替换）：

1. > "In many settings, one player is able to commit to a strategy before the other player makes a decision. Such models are synonymously referred to as **leadership, commitment, or Stackelberg models**..."
   —— 三个词是同一件事，别当成三种不同框架。

2. > "In general, if commitment to mixed strategies is possible, then (under minor assumptions) **it never hurts, and often helps, to commit to a strategy**. **Being forced to commit to a pure strategy sometimes helps, and sometimes hurts**（例如在石头剪刀布里被强制先承诺一个纯动作）"
   —— 这就是"混合承诺有用"的出处。**但注意它的前提是"能承诺混合"**，不是"能承诺纯动作却偷偷随机"。

3. > "Once the code for an agent ... is finalized and the agent is deployed, the agent is committed to playing the (**possibly randomized**) strategy that the code prescribes."
   —— 论文把"承诺"理解为**部署一个程序**：对手面对的是**程序（分布）**，而不是某一次的抽签结果。**这正是我们和标准模型的分歧点**：我们的 P1 看到的是 P0 这一手**实际出了什么**（实现值），不是 P0 的分布。所以第 2 条那半句"never hurts"不适用于我们（见 §0.2(b) 的不等式）。

4. > "...the follower (**who breaks ties to maximize the leader's utility**)..."
   —— 这是 **SSE（Strong Stackelberg Equilibrium）** 的标准 tie-breaking 假设：follower 在同优回应里挑对 leader 最有利的。现实里我们的 P1 会挑对 P0 **最不利**的（即 WSE 那一侧），所以我们做鲁棒打分时应该按 **WSE** 假设。

5. > "（For generic two-player games, the leader's subgame-perfect equilibrium payoff is unique.）"
   —— 论文顺带承认了承诺模型与 **SPE** 的关系：leader 的"承诺收益"和"SPE 收益"在一般两人博弈里是对得上的（除退化情形）。这句话支持我们的判断：**在没有额外承诺装置时，P0 的均衡就是 SPE，而不是某种"更强的 SE"。**

参考（SE / SSE / 混合承诺 / 纯 vs 混合）：
- Conitzer & Sandholm, *Computing the optimal strategy to commit to*（EC'06）：DOI <https://doi.org/10.1145/1134707.1134717>【正文已读，用作者主页 PDF：<https://users.cs.duke.edu/~conitzer/commitEC06.pdf>；ACM 站点本身 403】
- Letchford & Conitzer, *Computing optimal strategies to commit to in extensive-form games*：DOI <https://doi.org/10.1145/1807342.1807354>【仅摘要/元数据】
- Letchford, Conitzer, Munagala, *Learning and approximating optimal strategies to commit to*（课程页 PDF，可下载）：<https://courses.cs.duke.edu/fall18/cps296.2/commitment_full.pdf>【仅确认可访问，PDF 正文未抽】
- von Stengel & Zamir, *Leadership games with convex strategy sets*（SSE/WSE 的经典出处之一）：DOI <https://doi.org/10.1016/j.geb.2009.11.008>【仅摘要/元数据】
- Sinha, Fang, An, Kiekintveld, Tambe, *Stackelberg Security Games: Looking Beyond a Decade of Success*（IJCAI'18 综述，SSE/WSE 与"承诺混合策略"的标准写法）：DOI <https://doi.org/10.24963/ijcai.2018/775>【落地页可读】，PDF：<https://www3.ntu.edu.sg/home/boan/papers/IJCAI18_SG_review.pdf>【仅确认 PDF 存在，未抽正文】
- *Sequential commitment games*（GEB 2017）：DOI <https://doi.org/10.1016/j.geb.2017.08.009>【仅元数据，**未读正文，故全文未引用其任何结论**】

### 0.3 一句话总结形式化

| 问题 | 错误答案（常见的） | 正确答案（本局） |
|---|---|---|
| 博弈类型 | 不完全信息博弈（所以上 CFR） | **完全信息扩展式博弈**（信息集全是单点） |
| 解概念 | Nash / CFR 的平均策略 | **SPE（= 逆向归纳的结果）** |
| P0 的角色 | Stackelberg leader（有承诺力） | **先手方，但无承诺装置** → 只是"先动的一步" |
| 均衡下 P0 该做什么 | 混合策略 | **`argmax_{a0} min_{a1} V`（纯动作 + 最坏回应）**，混合只作 exploit 用 |
| 我们 rollout 里的对手 | "对手会 idle / 随机" | 这是**固定对手模型**，解的不是 SPE |

---

## 1. 结论摘要（5 条）

1. **建模：别用 CFR 的语言，用"完全信息序列博弈 + SPE/逆向归纳"的语言。** 局面全可见 ⇒ 信息集全是单点 ⇒ 均衡是 SPE；每回合 P0 的最优首着就是 `argmax_{a0} min_{a1} V(s,a0,a1)`，也就是你们已经在考虑的双层搜索。伪 Stackelberg 的说法要小心：标准 SE 要求 leader 公开承诺**一整张（可随机化的）策略表**（【正文】arXiv:2111.02111），我们 P0 只能承诺"这一手"，所以严格讲我们**没有承诺红利，只有"先暴露"的成本**。

2. **P0 必须建模 P1 的回应，而且这是唯一有理论保证的改法。** 现在 rollout 里"对手 idle / 固定概率随机"等于把 min 层换成固定策略，优化的是一个**假对手**。改成"对每个候选 a0，取 P1 的前 k 个候选回应，用值网/短 rollout 打分，再取 min（或温度化 softmin）"——这就是 SPE 的一步逆向归纳，也是最省的做法（预算核算见 §3.1，10 秒内可行）。

3. **"P0 应该用混合策略"这条要撤回，至少不能当机制解药。** 那条结论（Conitzer–Sandholm 一系）成立的前提是 **follower 只能对分布做最佳回应**——原文自己把"承诺"类比成"**部署一个程序**，对手面对的是程序（分布）而不是某一次的抽签结果"（§0.2 补充第 3 条，已读正文）。我们 P1 看到的是**实现值**，此时 `E[min] ≤ max min`，混合恒弱劣势（§0.2(b)）。混合只值得当"打不完美对手"的 exploit 手段单独做一个 ablation，**不要**把它写进主方案。这一条我建议明确纠正。

4. **27pp 这个量级，文献与项目史都不支持"结构性解释"，更像"策略/度量伪影"。** 四个参照：
   - 项目内同型现象：64 局测出"执后手 +28.1pp"（46.9% vs 75.0%），**配对补测后完全消失**（53.1% vs 53.1%，`docs/experiment_log.md:1015-1024`）【项目内】；
   - 同批 AntWar 数据里后手优势只有 **5.4%**（`docs/12游戏RL适配性评估.md:130`）【项目内】；
   - 棋类锚点：国际象棋白方总分 **52.2%–55.5%**（含引擎赛 55.4%），围棋贴目 **5–7 目**（见 §2.10）；
   - 综述短文的原话是"most popular strategy games give the first player a **modest** advantage"（§2.10，已读正文）——**modest 是 4–6pp 量级，不是 27pp**。
   → 结论：**27pp 里"结构项"大概是个位数 pp，剩下的主要来自"我们的搜索只在 P1 侧产生增益"**（历史实测：搜索后手 83–88% vs 先手 50%，`docs/az_bundle_mcts_results.md:69-79`；256 迭代时先手增益才出现，`docs/az_training_validation_results.md:78-92`）【项目内】。

5. **近和局要用"字典序 + W/D/L 头"，别用纯 ±1。** 我们的终局是字典序裁决（血 → 击杀 → 超武次数 → 用时 → 先手），纯 ±1 会把"血差"这个次级信号全部抹掉，而血差恰恰是打满 512 回合时唯一还有梯度的地方。学界标准做法是 **W/D/L 三头**（Leela Chess Zero，<https://lczero.org/blog/2020/04/wdl-head/>）+ **平滑 utility** + **规则级裁决**（50 步规则、贴目）。另外别忘了第 4 条 tiebreak 是"**AI 用时少者胜**"——**想得久本身就是一项劣势**，这在接近和局时是实打实的。

---

## 2. 逐方法

> 每条给出：名称 + 链接 + 核心思想 + 关键假设 + 对我们能不能落地。

### 2.1 完全信息序列博弈 / SPE / 逆向归纳

- **链接**：<https://plato.stanford.edu/entries/game-theory/>【正文】；<https://en.wikipedia.org/wiki/Subgame_perfect_equilibrium>【正文】
- **核心思想**：从叶节点往回推，每个节点上"轮到谁，谁就取对自己最优的动作"；这样得到的策略在每个子博弈里都是 NE，因而排除了"空威胁"（不可信的 off-path 行为）。
- **关键假设**：博弈有限、完美回忆、信息集是单点（= 完全信息）。**本局全部满足**（回合上限 512，`Ant-Game/README.md:13`；局面完全可见）。
- **落地**：✅ 直接可用。这就是"我们一步 + 对手一步"。**注意**：我们的 rollout 现状（对手 idle/随机）破坏了这个假设，等于解了别的问题。

### 2.2 Stackelberg / commitment games（SE、SSE、WSE）

- **链接**：<https://arxiv.org/abs/2111.02111>【正文】；<https://arxiv.org/abs/1811.03871>【正文】；<https://doi.org/10.24963/ijcai.2018/775>【落地页】；<https://users.cs.duke.edu/~conitzer/commitEC06.pdf>【正文】
- **核心思想**：leader 先**公开承诺**一个（可随机化的）完整策略，follower 对它做最佳回应；leader 在这一约束下最大化自己的收益。
- **关键假设**：① leader 有**不可反悔的承诺装置**；② 承诺**公开可观测**；③ follower 回应的是**策略/分布**。
- **对我们的判断**：⚠️ **三条里我们只满足第 ② 条**。P1 确实观测到我们的动作（②✓），但我们没有承诺装置（①✗，每回合可反悔），P1 回应的是实现值而不是分布（③✗）。**所以"我们是一个 Stackelberg 博弈"这个说法是半个错**——正确的说法是"每回合内的单步子博弈长得像 Stackelberg，但整局是 SPE；而单步子博弈里 P1 观测实现值这一条，让混合策略失效"。
- **落地**：✅ 只用它的**打分形式**（`max_{a0} min_{a1}`，即 WSE 式的"P1 在同优回应里挑最差的给我们"），不要用它的"混合承诺"结论。

### 2.3 完全信息博弈里"随机化"到底还有没有用

- **链接**：同 2.2；QRE：<https://en.wikipedia.org/wiki/Quantal_response_equilibrium>【正文】
- **核心思想**：混合策略的价值 = 让对手**在不知道实现值**时无法针对性回应（matching pennies 逻辑）。一旦对手看到实现值，混合的最优值退化到 `max_{a0} min_{a1}`。
- **关键假设**：对手的信息 = 分布，而不是实现。
- **落地**：⚠️ **理论上是弱劣势**，但工程上有两个正当用途：(i) 在 P1 是**不完美搜索者**时，随机化能打到它的盲点（这是 exploit，不是均衡）；(ii) 在多个候选**评估值几乎相同**时随机选，能避免被对手跨回合建模（去相关）。→ 建议做成**独立 ablation**（项目里 `TEMP` 开关曾被发现"在本管线里没有实际效果"，`docs/phase0_report_20261017.md:84`），不要混进主改法。

### 2.4 AlphaZero 式 min/max 备份（把对手回应放进树里）

- **链接**：<https://arxiv.org/abs/1712.01815>【摘要页】；MuZero <https://arxiv.org/abs/1911.08265>【摘要页】；KataGo 的 MCTS 说明 <https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md>【正文】
- **核心思想**：树里**交替 max/min**；价值从"当前行动者视角"给出，回传时取负（negamax）。对手的回应因此**自动**进入评估。
- **关键假设**：价值网在叶节点分布上有意义；模拟次数够（我们只有 16 次 rollout/候选，属极低预算）。
- **落地**：✅ **最直接的实现路径**。你们已经有"候选 → 值网打分"的框架，缺的只是"在候选之后插入一层 P1 的回应节点，并把回传从 average 换成 min"。

### 2.5 极低模拟预算下的策略改进（Gumbel AlphaZero）

- **链接**：<https://www.iclr.cc/virtual/2022/poster/6418>（ICLR 2022，*Policy improvement by planning with Gumbel*，Danihelka et al.）【仅摘要页】
- **核心思想**：用 **Gumbel-Top-k + 序贯 halving** 代替 PUCT 的访问计数分配，在**几十次模拟**这种极端预算下明显优于 PUCT，且带策略改进保证。
- **关键假设**：有策略先验（我们有）；动作数不太大（我们 ~24 个候选）。
- **落地**：✅ 很契合"每步 10 秒、每候选 16 回合 rollout"的 regime。**代价**：要改候选分配逻辑（不是改网络），属中等改动；建议在方案 A 之后作为"同预算下比更多候选"的加速器。

### 2.6 深度受限求解 + 价值函数（depth-limited solving）

- **链接**：<https://arxiv.org/abs/1805.08195>【摘要页】；ReBeL <https://arxiv.org/abs/2007.13544>【摘要页】；安全子博弈求解 <https://arxiv.org/abs/1705.02955>【摘要页】
- **核心思想**：真实博弈树太深，就在深度 d 处**用价值函数替代继续搜索**；把"深搜"换成"浅搜 + 好学到的叶值"。
- **关键假设**：价值函数在"被搜索分布"上校准得好（这正是你们反复踩的坑：价值头的符号/视角/校准）。
- **落地**：✅ 我们已经在做（16 回合 rollout 就是粗糙版）。**真正的改法是：把"16 回合 rollout"换成"1 步 + 值网"，把省下的预算花在"多算几个 P1 回应"上。** 这是方案 A 的核心 trade-off。

### 2.7 对手建模 / 最佳回应（不追均衡，追 exploit）

- **链接**：PSRO <https://arxiv.org/abs/1711.00832>【摘要页】；Stackelberg Planning（AAAI'18，leader 用搜索预测 follower 的最优计划）<https://doi.org/10.1609/aaai.v32i1.12090> / <https://ojs.aaai.org/index.php/AAAI/article/view/12090>【落地页】
- **核心思想**：真实对手不是"全知最坏对手"。**对具体对手求最佳回应（BR）**，收益通常高于 minimax；PSRO 用"种群 + BR"迭代逼近。
- **关键假设**：对手策略相对固定、可预测。
- **落地**：✅ **这是我们最该拿的那块**。P1 就是"我们自己的程序"，所以 **P1 的回应模型可以直接用我们自己的策略网 top-k 来近似**（self-play rollout）。成本最低、偏差可控。注意这与"minimax 的 min"不同：min 假设对手完美，BR 假设对手就是它自己——**实践中 BR 更准**，但会被"对手换人"打脸，所以两者要能切换。

### 2.8 软化 min：QRE / 认知层级

- **链接**：QRE <https://en.wikipedia.org/wiki/Quantal_response_equilibrium>【正文】；认知层级 <https://doi.org/10.1016/j.geb.2016.08.007>【仅元数据】
- **核心思想**：真实对手不会精确最小化我们的收益；用 `softmin_β`（或 level-k 的有限层推理）替代硬 min，避免 P0 为"想象中的完美反击"过度保守。
- **关键假设**：β 可估（用历史对局里"P1 实际选择 vs 我们预测的 top-k"的命中率拟合 β）。
- **落地**：✅ 一行代码级别的替换：`score = -1/β · log Σ_k exp(-β·V_k)`；β→∞ 就是硬 min。**建议默认 β 有限**（因为 P1 是启发式搜索，不是完美最小化器）。

### 2.9 CFR 为什么不是首选（把这条写清楚，避免走回头路）

- **链接**：Zinkevich et al., *Regret Minimization in Games with Incomplete Information*（NeurIPS 2007）<https://proceedings.neurips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html>【摘要页】
- **核心思想**：把整体 regret 分解到每个信息集上的反事实 regret，独立最小化即可；两方零和时平均策略收敛到 ε-Nash。
- **关键假设**：**不完全信息**（信息集含多个状态）+ 完美回忆。它的全部威力来自"对手不知道我在哪个状态"。
- **落地**：❌ **不推荐作为主方法**。理由：(i) 我们信息集全是单点，CFR 的机制空转；(ii) 它收敛到 NE，而完全信息博弈里 **NE 未必是 SPE**（空威胁可以是 NE），off-path 行为不可信；(iii) 成本高（要存整棵树的 regret）。**唯一值得借的是它的"平均策略"思想**——但那又回到 2.3 的"混合只作 exploit"。

### 2.10 先手/后手优势：策略窃取、贴目、pie rule、zugzwang

- **链接**：策略窃取 <https://en.wikipedia.org/wiki/Strategy-stealing_argument>【正文】；贴目 <https://en.wikipedia.org/wiki/Komi_(Go)>【正文】；国际象棋先手优势 <https://en.wikipedia.org/wiki/First-move_advantage_in_chess>【正文】；pie rule <https://en.wikipedia.org/wiki/Pie_rule>【正文】；zugzwang <https://en.wikipedia.org/wiki/Zugzwang>【正文】
- **核心思想与量级**：
  - **策略窃取**：在"多走一步永不吃亏"的对称博弈里，**后手不可能必胜**（否则先手偷后手的策略）。→ 先手优势的存在性有理论保证，但没有数值。
  - **国际象棋**：白方总分 52.55%（1851–78）→ 55.47%（1919–32）→ **引擎赛 55.4%**（CEGT 2009）→ 54.95%（Chessgames 2015 库）。**即"巨大的先手优势"在数值上也就 3–5.5pp。**
  - **围棋**：贴目补偿黑先，"黑先优势的公认值约 5–7 目"；现代贴目 6.5/7.5 使胜负接近 50/50。
  - **pie rule / swap rule**：桌游界对"已证实存在先手优势"的标准缓解手段——**开局后让后手选择"接盘或换边"**，直接把优势拍卖掉。
  - **zugzwang**：反例——在某些局面里"必须走棋"本身是劣势。说明"先动"不是单调有利，**先动 = 先暴露承诺**在信息博弈里可以是负资产。
- **量化材料（两条，均已读正文）**：
  - *First-Turn Bias in Turn-Based Games*（3 页综述短文，**非同行评审**）：<https://big-stupid-jellyfish.github.io/GFMath/pages/files/first-turn.pdf>【正文】。量化结论："In chess ... White scores about **54%** on average, roughly a **+5.9% win-probability edge**"；"Go: Black moves first, but standard komi (usually **6.5** points) compensates; **without komi Black would have a strong advantage**"；Connect Four 已被解出为先手必胜，checkers / tic-tac-toe 是和。原文用词是"most popular strategy games give the first player a **modest** advantage"——**modest 这个词，正好和"27pp 不是 modest"对上**。
  - Micklich, *Do First Mover Advantages Exist In Competitive Board Games: The Importance of Zugzwang*（ABSEL 2009，7 页，Connect-Four 实证）：<https://absel-ojs-ttu.tdl.org/absel/article/view/371>【正文】。结论：**先手不是胜负的唯一决定因素**，zugzwang 等非线性因素也参与；但作者自己写明这些因素 **"not statistically significant"**。→ 只能当"方法论旁证"（先手优势≠全部），**不能当量级证据**。
- **关键假设**：策略窃取要求"多走一步不吃亏"（单调性）。**塔防游戏不满足**：出招要花金币、且被对手看见，先动可能是负资产。所以我们的座位差**不被理论保护，也不被理论解释成 27pp**。
- **落地**：✅ 但落地在**评测协议**而不是 AI：
  - 座位置换配对（同 seed 打两局、交换先手）+ McNemar；
  - 报告**分座位**的 CI，不报混合胜率；
  - 训练侧做**座位平衡采样** + 座位条件化（把 `player` 作为特征，已有）。
  - ❌ 不要在比赛里指望 pie rule（规则不是我们能改的）。

### 2.11 近和局 / 平局判定

- **链接**：50 步规则 <https://en.wikipedia.org/wiki/Fifty-move_rule>【正文】；残局库 <https://en.wikipedia.org/wiki/Endgame_tablebase>【正文】；WDL 头 <https://lczero.org/blog/2020/04/wdl-head/>【正文】；多目标/字典序 <https://www.jair.org/index.php/jair/article/view/10836>【落地页】；贴目 <https://en.wikipedia.org/wiki/Komi_(Go)>【正文】
- **核心思想（四种流派）**：
  1. **规则级裁决**：50 步规则 / 三次重复 / 子力不足 → 把"无限接近和局"直接判掉；围棋用贴目把先手优势折成分数。
  2. **W/D/L 三头**：不要把"期望得分"当一个标量——它分不清"和棋"和"胜负各半"。Lc0 从 2019-07 起所有网络都有 WDL 头，原文明确说"从'胜率'变成'期望得分'是一个假设，而我们可以直接预测三种结果"【正文】。
  3. **平滑 utility**：赢 1 目和赢 100 目不该等价，但也不能让"刷分"替代"求胜"。
  4. **字典序 / 多目标**：先保证主目标（胜负），再优化次级目标（血差 / 击杀 / 用时）。多目标序贯决策综述见 Roijers et al.【落地页】。
- **对我们的直接含义（5 条）**：
  - a) 终局回报别用纯 ±1：`u = sign(胜负) + ε·(血差/50)`，或干脆**双头**（胜负头 + 血差头），让近和局有梯度；ε 要小到不改变"求胜"的排序。
  - b) **方差/不确定性也要进打分**：近和局时值网方差大，`score = mean − λ·std` 能避免"赌一把"。
  - c) **打满 512 回合是一个独立分支**，要有专门的训练样本（现在很多局 ~240 回合结束，`docs/alphazero_intent_mcts_plan.md:16`），否则"进入裁决"这条路径根本没被学过。
  - d) **用时是 tiebreak 的第 4 项**（`Ant-Game/README.md:297`）：搜索越久，越可能在"血同、击杀同、超武次数同"时输掉。→ 预算要**自适应**（只在多候回合花满 10 秒，单候选直接短路，见 §3.1），并把"累计用时"作为特征喂给值网。
  - e) 概念提醒：残局库里的 **"cursed win"（赢棋但违反 50 步规则 ⇒ 实际判和）** 是我们情形的精确类比——"按实力赢了，但按规则没赢"。官方页面被 bot 检测拦截，未能读到（§4.1）。

---

## 3. 落地草案

> 前提：只改**打分/评测**，不改动作空间、不惩罚 HOLD、不动正在跑的实验。

### 3.1 方案 A（首选）：P0 侧的"回应感知打分"（1-ply Stackelberg / negamax 打分）

**改哪个模块**
- `code/my_ai/bundle_mcts.py`：根节点候选打分（把叶值 `value` 换成 `score_resp`）。
- 新增 `code/my_ai/opponent_response.py`：给定 `(state, a0)`，用**我们自己的策略/位置先验**在 `s1 = apply(a0)` 上取 P1 的 top-k 回应。
- 只对 **P0 侧**启用。P1 侧不需要——P1 已经看到 P0 的动作，它缺的不是"建模回应"，而是"更准的价值"（另一条线）。

**伪代码**

```python
# ── 现状：对手不回应 ───────────────────────────────
def score_now(state, a0):
    s1 = apply(state, a0)
    return rollout(s1, horizon=16, p1_policy=idle_or_random)   # ~0.32s/候选

# ── 方案 A：一步回应感知（SPE 的一步逆向归纳）─────────
def score_resp(state, a0, k1=3, beta=4.0, lam=0.5):
    s1 = apply_ops(state, a0)                       # 引擎 apply，~20ms
    cands1 = topk(policy_prior(s1, player=1), k1)   # 用我们自己的网当"P1 回应模型"
    vals = []
    for a1 in cands1:
        s2 = resolve_turn(s1, a1)                   # 一次整回合结算，~20ms
        v  = value_net(s2)                          # 一次前向，~5ms（不 rollout）
        vals.append(v)
    m = softmin(vals, beta)                         # beta→∞ 退化为 hard min（WSE 式）
    return m - lam * stdev(vals)                    # 惩罚"回应分歧大"的候选

# ── 根节点 ────────────────────────────────────────
def choose(state, K0=8, K1=3):
    cands0 = topk(candidates(state), K0)            # 先验截断 24 → 8
    scored = [(score_resp(state, a0), prior(a0), a0) for a0 in cands0]
    best = max(scored)[2]
    # 只对冠军做一次完整 16 回合 rollout 复核（可选，+0.32s）
    return best if rollback_check(state, best) else second_best(scored)
```

**10 秒预算核算（用项目实测常数）**

| 项 | 常数 | 来源 |
|---|---|---|
| 引擎整回合结算 | ≈ 20 ms | `docs/az_training_validation_results.md:42`【项目内】 |
| 现状每候选 | 16 回合 rollout ≈ 16×20ms ≈ 0.32 s | 推算 |
| 现状每决策 | 24 候选 × 0.32 s ≈ **7.7 s** | 推算（贴 10 s 上限） |
| 方案 A 每候选 | k1=3 × [apply 0.02 + 前向 0.005] ≈ **0.08 s** | 推算 |
| 方案 A 每决策 | 8 × 0.08 ≈ **0.64 s**（再留一次 16 回合复核 +0.32s） | **≪ 10 s** |
| 全 2-ply + rollout | 24 × 24 × 0.32 ≈ **184 s** | **不可行**（这就是为什么不能直接上完整 minimax） |

**关键洞察：预算根本不是瓶颈，候选数才是。** 项目实测 91.6% 的决策只有 1 个候选、80% 的决策被 `SKIP1` 短路（`docs/goal2_winrate_method_analysis.md:68`）【项目内】——也就是说省下来的时间极其集中。方案 A 可以在**只占 6% 预算**的情况下把 min 层加进去；剩下的预算应该花在"多留几个 P1 回应候选（k1=5~8）"和"P0 侧多留几个 a0（K0=12）"上。

**为什么不做完整 alpha-beta**：分支因子是组合爆炸（bundle 24×24），叶值评估要跑引擎（昂贵），而且我们没有可靠的界做剪枝——alpha-beta 的收益随"评估便宜"而涨，我们的场景恰好相反。**1-ply + 值网叶值**才是正确形态（这也正是 AlphaZero 在极低预算下的做法，§2.4）。

**预注册验收判据（建议）**
1. 机制判据：P0 侧的**动作改变率** ≥ 20%（证明打分确实改决策，而不是装饰）；
2. 性能判据：P0 侧胜率（座位置换配对、n≥64 对）提升 ≥ 5pp 且 CI 下界 > 0；
3. 成本判据：P0 侧平均用时 ≤ 现状 × 1.5（否则会在 tiebreak 上亏回去）；
4. 反例判据：**HOLD 比例不应显著下降**（回应感知打分不应把 HOLD 挤掉；见 §3.3 的解释——理论上它反而可能上升）。

### 3.2 方案 B：座位解耦的评测协议（零算法风险的"先做这个"）

**为什么必须先做**：你们现有的 64 局数字（72% / 25%）如果按座位各 32 局，差的 SE 约 12.5pp，27pp 只有 ~2.2σ；而且项目史上**同型的 28.1pp 在配对补测里直接归零**（`docs/experiment_log.md:1015-1024`）。**在协议没定死之前，任何针对"座位差"的算法改动都可能是在拟合噪声。**

**做法**
1. 同一批地图，每张图打两局（交换先手），得**配对**样本；
2. 主指标用**配对 McNemar**（只统计"一胜一负"的不一致对），并报**分座位 CI**；
3. 样本量：64 局（32 对）只能分辨 ~25pp 的效应；要分辨 10pp 量级需要 ~150+ 对（项目里"64 局分辨不了 60% 的效应"的经验，`docs/experiment_log.md:947`）；
4. 同时报**血量差**（连续量，方差小得多）作为副指标——项目已经在用（`docs/exp_plan_20261024_night.md:4`）。

### 3.3 一个"零成本诊断"（建议先跑，用来决定 A 值不值得做）

**回应值缺口（response gap）测量**：在 N≈200 个真实 P0 决策点上，对每个候选 a0 同时算
- `V_idle(a0)`：现状打分（对手 idle）；
- `V_resp(a0)`：方案 A 的 `min_k V(s(a0,a1_k))`。

然后报三个数：
1. `Δ = mean_a0 [ V_idle(a0) − V_resp(a0) ]`：**忽略回应让我们高估了多少**；
2. `argmax 一致率`：现状选的 a0 在回应感知下还是不是第一（这个数直接换算成"P0 有多少比例的决策是错的"）；
3. `P1 回应命中率`：P1 实际动作落在我们 top-k 预测里的比例 → 用来校准 `beta`。

如果一致率 > 90%，说明"忽略回应"不是主因，27pp 要去查**价值网的座位校准**或**数据分布**；如果一致率 < 70%，方案 A 基本锁定。**这个诊断不需要训练、不需要改对局流程，一天能出。**

**关于 HOLD**：本方案**不改变动作空间、不改先验、不惩罚 HOLD**。而且从机制上讲，HOLD 是 P0 唯一"不新增一个可被针对性反制的具体对象"的动作——它不泄露"我要在哪一格起塔"。所以在回应感知打分下，**HOLD 的相对价值可能上升而不是下降**。验收判据里我把它写成"不应显著下降"，就是为了防止把 HOLD 当成缺陷去压。

---

## 4. 不确定 / 访问受限清单（读不到就写读不到）

### 4.1 工具与访问
- **`web_fetch` 在本环境不可用**：所有外部域名被解析到非公网地址段（如 `arxiv.org → 198.18.3.21`），工具直接拒绝。实际用了三条替代路径：
  1. **Playwright MCP** 打开 `ar5iv.labs.arxiv.org` 的 HTML 全文（用于 arXiv:2111.02111、arXiv:1811.03871）；
  2. **pwsh `Invoke-WebRequest`** 抓标题/元数据/Wikipedia REST API；
  3. **PDF 下载 + PyMuPDF 抽文本**（用于 Conitzer–Sandholm、Micklich、First-Turn Bias 三篇）。
  → 所以"【正文】"的读法有差异：**PDF 抽文本最可靠；HTML 抽取次之**（可能丢公式/表格，也可能被页面重定向打断——读 arXiv:1811.03871 时页面曾被重定向到另一篇论文，我回到正文重取）。
- **403 / 付费墙，正文未读**：`dl.acm.org`（Conitzer–Sandholm 的 ACM 页 403——但作者主页 PDF 拿到了，已读正文）、`sciencedirect.com`（GEB 系列全部无法直连）、`openreview.net`（"Verifying your browser" 反爬，Gumbel 那篇只读到 ICLR 的 poster 摘要页）。
- **bot 检测拦截**：`syzygy-tables.info` 返回 "Making sure you're not a bot!"，**cursed win 的权威说明没读到**。
- **官方规则页不可达**：`handbook.fide.com/chapter/E012023`（FIDE 国际象棋规则，50 步规则的官方出处）连接失败，**只核到 Wikipedia 版本**。
- **下载失败**：Bošanský 的 habilitation 论文（`dspace.cvut.cz`，里面有 Stackelberg 在扩展式博弈里的完整定义章节）下载卡死、0 字节，**未读**；改用 arXiv:2111.02111 的正文替代。

### 4.2 证据清单（谁读到了正文、谁只是二手）

| 文献 | 状态 | 我怎么读的 |
|---|---|---|
| Conitzer & Sandholm, EC'06 | **正文** | `users.cs.duke.edu/~conitzer/commitEC06.pdf` + PyMuPDF |
| arXiv:2111.02111（SE with memory） | **正文**（摘要+引言+定义段） | Playwright / ar5iv HTML |
| arXiv:1811.03871（Quasi-Perfect SE） | **正文**（摘要+引言） | Playwright / ar5iv HTML |
| First-Turn Bias（3 页短文，非同行评审） | **正文** | PDF + PyMuPDF |
| Micklich, ABSEL 2009（Connect-Four） | **正文** | PDF + PyMuPDF |
| Wikipedia：SPE / Komi / 先手优势 / 50 步 / pie rule / zugzwang / 策略窃取 / tablebase / MCTS / QRE | **正文** | REST API 摘要 + 页面正文 |
| SEP *Game Theory* | **正文**（相关段落） | HTML 抽取 |
| Zinkevich et al., NeurIPS 2007（CFR） | 仅摘要页 | — |
| AlphaZero 1712.01815 / MuZero / depth-limited / ReBeL / safe subgame / PSRO | **仅摘要页** | — |
| ICLR 2022 Gumbel（poster 页） | 仅摘要 | OpenReview 被反爬 |
| Leela WDL 博客 / KataGo GraphSearch.md | **正文**（相关段落） | HTML 抽取 |
| IJCAI'18 SSG 综述 | 仅 DOI 落地页 + PDF 存在性 | 正文未抽 |
| Letchford & Conitzer（ACM TEAC） | 仅 DOI | — |
| Letchford/Conitzer/Munagala（Duke 课程 PDF） | 仅确认可下载 | 正文未抽 |
| von Stengel & Zamir（GEB 2010） | 仅 DOI | — |
| *Sequential commitment games*（GEB 2017） | 仅 DOI，**未读，故全文未引用其结论** | — |
| Bošanský habilitation（CVUT） | **下载失败，未读** | — |

### 4.3 没找到的（明确不编）
- **没找到**"塔防 / 六边形网格 / RTS 类游戏里后手信息优势的实测量级"的直接研究。arXiv 上按 `"turn order" AND "board game"`、`"first-player advantage"`、`"second-mover advantage"` 检索均无相关实测论文。
  **退而求其次找到的是棋类/桌游锚点**（已读正文）：国际象棋约 **54%**、围棋贴目 **6.5**、Connect Four 先手必胜、checkers/tic-tac-toe 和；以及一篇**非同行评审**的 3 页综述。**这些只能给"量级感"，不能给"我们这局应该是多少"。**
- **没找到**"CFR 在完全信息博弈里收敛到 SPE"的正面结论。我只确认了 CFR 的设计目标是**不完全信息**两方零和，"完全信息博弈是它的特例"在形式上成立。**"CFR 的 off-path 行为是否可信"我没有证据，故按"未证实"处理。**
- **没核到**"leader 在完全信息 + follower 观测实现值的设定下，混合策略严格劣于最优纯动作"的**正式定理编号**。§0.2(b) 的不等式是我自己写的初等推导（`E[min] ≤ max min`）；Conitzer–Sandholm 原文只给了"能承诺混合时混合不亏"这一侧（§0.2 补充第 2 条），**另一侧（观测实现值时混合不赚）是我补的推理，没有文献定理编号**。

### 4.4 数字口径需要你确认
- 你给的 "P1 72% / P0 25%" 与"座位差约 27pp"**对不上**：72 − 25 = **47pp**。可能的解释：(i) 27pp 是"相对 50% 的平均偏移"（22 + 25)/2 ≈ 23.5pp，仍对不上；(ii) 两个数字来自不同子集/不同样本量；(iii) 27pp 是另一个实验。**建议先把这两个数字的分母、子集、是否配对写清楚**，否则后面的显著性都无从谈起。
- 项目内还有个方向相反的数字：`docs/experiment_log.md:1312` 记录 "P1 91 胜 / P0 69 胜 = 57%"，而 `docs/phase0_report_20261017.md:105` 记录"禁闪电后先手反而变劣势（~18pp 的角色不对称）"。**不同子博弈、不同对手，座位效应符号都会变**——这本身又是"座位差主要是策略/对手相对强度，而不是游戏常数"的证据。

---

## 附：一句话回答你的四个问题

1. **标准建模**：完全信息序列博弈 → **SPE / 逆向归纳**；"Stackelberg"只应作为**单回合单步子博弈**的类比（leader=P0，follower=P1），且必须声明"我们没有承诺装置、P1 观测实现值"这两个偏差。**均衡概念是 SPE，不是 SE。**
2. **对搜索的含义**：必须把 min 层加进去。三种做法的代价：self-play rollout（最便宜，近似 BR）、1-ply minimax（最正确，8 候选 × 3 回应 ≈ 0.64 s，**可行**）、完整 2-ply rollout（184 s，**不可行**）。**混合策略按 §0.2(b) 应降级为 exploit 手段，不是主方案。**
3. **27pp 是否与机制一致**：**方向一致（P1 有信息优势），量级不一致**。棋类锚点 3–5.5pp、项目内 AntWar 数据 5.4pp、项目史上同型 28pp 在配对补测里归零。**建议先按 §3.2 重测，再决定是否为它改算法。**
4. **近和局**：规则级裁决（50 步/贴目/cursed win）+ WDL 三头 + 字典序效用；我们是"血 → 击杀 → 超武次数 → 用时 → 先手"的字典序，**别用纯 ±1**，并且记住"想得久"本身是第 4 项 tiebreak 的劣势。
