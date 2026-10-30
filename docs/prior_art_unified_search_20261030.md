# 统一候选生成 + 两阶段评估：先行工作调研（2026-10-30）

> 调研对象：`docs/unified_cand_design_20261029.md`（设计）与 `code/my_ai/az_intent/unified_cand.py`（实现）。
> 性质：**只读 + 联网检索**，本文不跑代码、不跑对局、不改实现。
> 检索渠道：arXiv / DBLP / ACM DL / Science / AAAI / ICLR / chessprogramming wiki。部分站点（nature.com 正文、Semantic Scholar API、DBLP API、export.arxiv.org）在检索时被 429 或跨域重定向挡住，因此个别条目给的是 **DOI 链接** 或 **DBLP/Scholar-Archive 记录**，条目上已标注。
> 约定：**凡我没查到可靠出处的，明确写"未找到"，不编。**

---

## 0. 三句话结论（展开见 §4–§6）

1. **最接近哪个已知工作**：没有一个"同构"工作。整体上最像 **AlphaStar 的"动作类型 → 参数"分解式动作空间**（Vinyals et al. 2019）**叠上 progressive widening 的"按先验排名逐格展开"**（Couëtoux et al. 2011 / Chaslot et al. 2008），而 `Q[c]` 那个利用项在数学上**精确等于 mellowmax(β=3)**（Asadi & Littman 2017）——这是本文最硬的一条对照，见 §3.3。
2. **最值得借鉴的 2–3 条**：(a) **β 必须随 n 放大**——mellowmax 相对 max 的折扣上界是 `log(n)/β`，β=3、n=10 时最多比 max 低 0.77（我们的分数值域才 [−1,1]），而"只有一个神手"的类恰好吃到这个最大折扣，与设计意图相反；(b) **两阶段按 successive halving 的 η 设比例**（阶段 1 的 48 条确定性 rollout 总成本 < 阶段 2 单条候选的成本，所以"先筛"几乎免费；真正的旋钮是阶段 2 取 `k₂ = K/η`，η∈[3,6]）；(c) **多动作束的分数不要原样回写 `Q[c]`**——组合 bandit 的信用分配是已知问题（CUCB 明确要求 semi-bandit 反馈，即能观测到每个组成臂的贡献），我们是 full-bandit 反馈。
3. **文献里最该警惕的坑**：**确定性 idle rollout 是 null-move 剪枝的同款近似，而 null-move 的经典失效模式就是 zugzwang**——在"必须回应"的局面里，"双方都不动"的 rollout 完全看不见代价，会系统性高估进攻、低估防守；再叠上 progressive widening 的**早熟收敛**（宽度增长慢于价值增长 ⇒ 好动作永远没被展开）和 **max 型索引的 winner's curse**（Extreme bandits），三者会合流成"类被一次偶然高分锁死"。

---

## 1. 我们的机制（照实现抄，供逐条对照）

参数名取自 `unified_cand.py`：`AZAI_CAND_BETA=3.0`、`AZAI_CAND_UCBC=1.0`、`tau`（`_sample_ucb` 的温度）、`K=48`、阶段 2 沿用现役 flat（`random` + opponent-only + horizon16 + N=64）。

| # | 机制 | 实现细节 |
|---|---|---|
| M1 | **类级 UCB 生成候选**（不再先钉一个类） | 每类 `Q[c]`（利用项）+ `c·√(log N / n)`（探索项，c=1.0）；未试类 `= +inf` 优先 |
| M2 | **利用项 = LogSumExp 软最大** | `Q[c] = (1/β)·log((1/n)Σ exp(β·xᵢ))`，β=3；对负数成立，β→0 退化均值、β→∞ 退化 max |
| M3 | **位置游标**（用静态位置图按排名逐格推进） | `cursor[形状] = {t, pos[]}`，第 t 步各槽名次按**混合进制**展开（槽 0 变化最快），只 `argpartition` 取第 k 名、不做全排序 |
| M4 | **游标按"带重数的规范多重集"索引** | 去 HOLD、排序、**保留重复**；`[build] ≠ [build,build]`，`[build,lightning] == [lightning,build]`；每个形状独立推进 |
| M5 | **束内类选择**：头 0 = `UCB argmax`，头 1..n−1 = `softmax(UCB/τ)` 采样（允许重复类） | `_pick_group` |
| M6 | **动作别名去重缓存** | 缓存键 = 规范化 bundle `(op_type,arg0,arg1)` 序列（去 HOLD），**跨形状共享**；命中 ⇒ 复用分数 + 计入 `Q[c]` + 游标推进一格（**不跳过**） |
| M7 | **两阶段预算** | 阶段 1 = **确定性 idle rollout ×1**（双方不出招，游戏确定性 ⇒ 1 次即精确）→ 只用于 **top-k 选择**；阶段 2 = 现役随机化 rollout 全评估（不动） |
| M8 | **选谁进阶段 2** | 简单版 = 阶段 1 分数 top-k；加权版 = `rollout + λ·位置网分`（起步 λ=0，位置网分只在游标提议顺序里起作用） |
| M9 | **意图→动作多对一** | 通道 0–15 在空地**都**解码成 `BUILD Basic`；已占格解码成**升级**；21/22 基地升级、23 HOLD 是**单位置类** |

---

## 2. 对照表：我们的机制 ↔ 先行工作 ↔ 能借鉴什么

| 我们的机制 | 对应先行工作（名称 / 年份） | 一句话做法 | 我们能借鉴什么 |
|---|---|---|---|
| **M1** 类级 UCB | UCB1（Auer, Cesa-Bianchi & Fischer, 2002）；PUCT（Rosin, 2011）；HOO（Bubeck, Munos, Stoltz & Coquelin, 2011） | 索引 = 经验值 + 探索奖励；PUCT 把探索项乘上策略先验 `P(s,a)`；HOO 在层级划分的 cell 上用"cell 内最大值的置信上界"当索引 | PUCT 的 `P/(1+N)` 形式告诉我们"先验应该**乘**在探索项上、而不是当成初始值"；HOO 说明"用 **max 型**索引给一个**集合**打分"是标准做法，不是野路子 |
| **M2** LogSumExp 利用项 | **mellowmax**（Asadi & Littman, 2017）；Extreme bandits（Carpentier & Valko, 2014） | mellowmax = `(1/β)log((1/n)Σe^{βxᵢ})`，落在 **mean 与 max 之间**，是一个非扩张算子；Extreme bandits 研究"奖励是**样本最大值**"时 bandit 的行为 | **我们的公式和 mellowmax 逐字等价**（实现里 `m + log(mean(exp(b(x-m))))/b` 化简后就是它）。可直接继承它的温度分析：`max − Q ≤ log(n)/β`。β 不该是常数，见 §5-1 |
| **M3** 位置游标 | Progressive Widening（Chaslot et al., 2008）；Double Progressive Widening, DPW（Couëtoux, Hoock, Sokolovska, Teytaud & Bonnard, 2011）；KataGo 的 policy-prior 宽度控制（Wu, 2019） | 子节点数按 `⌈C·N^α⌉` 增长（α≈0.5），而不是一上来就全展开；按先验排序取前几个 | `α` 是**唯一**该调的旋钮，且"按先验排名展开"= 我们的游标；DPW 还强调**动作空间宽度与机会空间宽度要分开加宽**（对应我们"类预算"与"位置预算"应分开） |
| **M4/M5** 规范多重集 + 束内采样 | Gumbel AlphaZero 的 **Gumbel top-k** 不放回采样（Danihelka et al., 2022）；组合半 bandit（Chen, Wang & Yuan, 2013） | Gumbel top-k：用 `gumbel(logits)` 取前 k 个**互异**动作，等价于"按先验不放回采样"；CUCB：组合臂的奖励可分解到组成臂 | 我们现在的"头 1..n−1 用 `softmax(UCB/τ)` 独立采样"是**有放回**采样 ⇒ 会产生 `[build,build]` 这类重复束（设计上要，但会挤掉多样性）。可借用 Gumbel top-k 的"有放回/不放回"两种模式显式切换 |
| **M6** 别名去重缓存 | **transposition table**（棋类引擎）；**UCD：DAG 上的 UCB**（Knowledge-Based Systems, 2012）；action translation（Ganzfried & Sandholm, 2013） | 同一局面经不同着法序到达时只存一次；UCD 指出把树当 DAG 直接套 UCB 会导致**备份重复计数、访问数虚高** | 我们的"命中 ⇒ 复用分数 + 计入 `Q[c]`"正是 DAG 备份的重复计数模式。**`n` 应记"有效独立样本数"，命中次数单独记**，否则探索项被压 |
| **M7/M8** 两阶段（便宜筛 → 贵评） | **null-move pruning** + **late move reductions**（棋类引擎）；**successive halving**（Karnin, Koren & Somekh, 2013）、**Hyperband**（Li et al., 2018）；Pandora's rule（Weitzman, 1979） | null-move：先"白送对手一步"做廉价搜索，若仍好则直接剪枝；LMR：可疑着法先浅搜、好再深搜；SH/Hyperband：每轮保留 1/η、预算翻 η 倍 | null-move 给的不是"低预算 rollout"，而是**"改变了模型的近似"**——这正是我们的 idle rollout；LMR 给的是"**浅→深**"的预算阶梯，可用来把我们的"1 次 → 64 次"改成"1 → 8 → 64"三级 |
| **M9** 意图→动作多对一 | Action translation（Ganzfried & Sandholm, 2013）；Libratus 的动作抽象（Brown & Sandholm, 2018）；invalid action masking（Huang & Ontañón, 2020）；options / macro-action（Sutton, Precup & Singh, 1999） | 抽象动作到具体动作的**硬翻译**会引入系统性偏差，软翻译（按距离加权）更稳；掩码后在合法集上重归一化会改变策略分布 | "硬翻译"的教训：把多个意图折成同一个具体动作后，**意图的多样性信息消失**，但计数还在 ⇒ 类级统计会"过度自信"。见 §5-5 |
| **M7** 确定性 = 1 次采样 | 静态评估函数 / one-step lookahead（棋类引擎传统）；"An Analysis of Monte Carlo Tree Search"（James, Konidaris & Rosman, 2017） | 方差为 0 时 1 个样本就是精确值；但"精确"只对**它自己的模型**精确 | 关键区分：**估计量无偏 ≠ 模型正确**。idle rollout 的方差是 0，但它估的是"对手不动"这个**错误模型**下的值 |

---

## 3. 分方向综述

### 3.1 A. 巨大组合动作空间的搜索

**AlphaStar（Vinyals et al., 2019, Nature）** 把每步决策拆成**自回归的三段**：先选动作类型（`action_type`），再选参与的单位（`selected_units`），再选目标（`target`）；三段共享一个 transformer，指针网络只在合法目标上归一化。这跟我们"动作类 → 位置/参数"的分解**结构同源**，区别是它把这个分解**内嵌进策略网络**（一次前向就是一次决策），而我们是把分解**当成搜索算子**（在分解出的空间里做 bandit + 游标）。链接：[doi:10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z)（Nature 正文站点在检索环境里被 idp 重定向挡住，故给 DOI）。

**OpenAI Five（Berner et al., 2019）** 的动作空间是**多头**的：主动作类型一个 softmax 头，之后若干"参数头"（目标槽位、坐标）各自 softmax，每个头有各自合法掩码。这跟我们的"24 个类 + 位置图"几乎一一对应（类 = 主头，位置图 = 参数头）。它踩过的坑写得很清楚：**多头独立 softmax 会让"类型合法但参数非法"的组合出现**，所以必须逐头掩码并在采样后做合法性校验——我们的 `can_apply` 逐条过滤是同一件事。链接：[arXiv:1912.06680](https://arxiv.org/abs/1912.06680)。

**StarCraft II 学习环境（Vinyals et al., 2017）** 是把 RTS 动作空间**显式定义为 `function + args`** 的论文（`no_op / select_point / move / attack` 等 + 坐标），这是"类 × 参数"形式化最早、最常被引的一处；它同时说明为什么不能枚举笛卡尔积（空间大小 ~10^26）。链接：[arXiv:1708.04782](https://arxiv.org/abs/1708.04782)。

**Factored / branched action space 在 RL 里**：BDQ（Tavakoli, Pardo & Kormushev, 2018）把 Q 网络按动作维度分叉，每个分支输出一个子动作的 Q，用**各分支的 argmax 拼出联合动作**；它在 Atari 的 18 键组合动作空间上有明显收益，但论文自己也指出：分支之间**不建模交互**（联合 Q = 各分支 Q 的聚合），所以"两键同时按"这类**协同动作**学不到。这条对我们很关键——我们的束是"2–3 个动作的协同"，恰恰落在 factored 假设失效的地方。链接：[arXiv:1711.08946](https://arxiv.org/abs/1711.08946)。

**线性 bandit 视角**（值得一提但没深挖）：把"类 one-hot + 位置特征"拼成一个特征向量，用 LinUCB 之类的方法可以在**不枚举笛卡尔积**的前提下给任意"类×位置"打分；Bubeck & Cesa-Bianchi 的综述第 2 章覆盖了这条线（[arXiv:1204.5721](https://arxiv.org/abs/1204.5721)）。我们的"类分 + 位置图分"其实是它的手工分解版。

### 3.2 B. 候选生成 + 评估的两段式

**棋类引擎的"走法生成 + 搜索"**（alpha-beta，Knuth & Moore, 1975；工程细节见 [chessprogramming wiki: Alpha-Beta](https://www.chessprogramming.org/Alpha-Beta)）在语义上跟我们最像：先生成**合法走法集合**（便宜、确定、可完全枚举），再做**昂贵搜索**。但注意一个关键差别：棋类的走法生成是**无损**的（不丢好走法），我们的候选生成是**有损**的（K=48 截断），所以棋类那边"生成"从不背锅，而我们的"生成"正是当年 0/42 的锅（见 `docs/prior_candidate_fix_research_20261029.md`）。这条差别值得在设计里写死：**截断即近似，必须把截断率当指标监控**。

**Progressive widening** 两条源头：Chaslot, Winands & van den Herik（2008）提出 **progressive widening / progressive unpruning**——不要一次展开全部子节点，而是先按先验取前几个，随访问数增加再"解冻"更多（[Scholar-Archive 记录](https://scholar.archive.org/work/73t2pv5xajdqlg5lfhq3jnn7pm)）。Couëtoux et al.（2011）把它形式化成 **DPW**：子节点数 `k(n) = ⌈C·n^α⌉`，并强调**动作空间与机会（outcome）空间要各自加宽**（[doi:10.1007/978-3-642-25566-3_32](https://doi.org/10.1007/978-3-642-25566-3_32)）。我们的"位置游标第 t 步取第 t+1 名"**就是 α=1 的 DPW**（宽度线性于访问数）——这个对照很重要，因为 α=1 在预算小时意味着"每个形状只摊到 1 格"，正是我们文档 §78 记的"塌成 7 条"那种现象。

**"先用便宜启发式筛、再用贵 rollout 评估"** 在搜索里有两个成熟形态：一是**棋类的剪枝阶梯**——null-move pruning（先"白送对手一步"做浅搜，若仍够好就整枝剪掉，[wiki](https://www.chessprogramming.org/Null_Move_Pruning)）、late move reductions（可疑着法先浅搜、分数好才加深，[wiki](https://www.chessprogramming.org/Late_Move_Reductions)）、futility/razoring（"这点分差不可能翻盘 ⇒ 不搜了"）；二是 **coarse-to-fine / retrieve-and-rerank** 式的两段式（信息检索与 NLP 里的标准范式）。前者的价值在于它明确记录了自己的**代价**：剪枝会**漏掉少量好着**，且漏检率随评估函数与实际分布的**失配**上升——这直接对应我们"阶段 1 分数与阶段 2 分数可能反相关"的风险。

**综述可查**：Świechowski et al.（2023）的 MCTS 综述系统整理了 widening / 先验引导 / rollout 替换这一族变体，是查"某个技巧有没有人做过"最快的入口（[arXiv:2103.04931](https://arxiv.org/abs/2103.04931)）。

### 3.3 C. UCB / bandit 作用在"动作子空间"上

**基础与 PUCT**：UCB1（Auer, Cesa-Bianchi & Fischer, 2002，[doi:10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352)）给出 `均值 + c√(log N/n)` 与"先各试一次"的初始化。PUCT（Rosin, 2011，[doi:10.1007/s10472-011-9258-6](https://dl.acm.org/doi/abs/10.1007/s10472-011-9258-6)）把它改成 `Q + c·P(s,a)·√N/(1+N(s,a))`：**先验乘在探索项上、并让先验的影响随访问数衰减**。AlphaZero（Silver et al., 2018，[arXiv:1812.03828](https://arxiv.org/abs/1812.03828)）就是用这个。**对我们的直接含义**：如果以后启用类网先验，正确做法是 `c·P[c]·√(log N)/n` 之类（先验缩放探索），而不是把 `Q[c]` 初始化成先验值——后者会跟利用项混在一起、无法区分"先验说好"和"实测好"。

**层级 bandit / 在结构空间上的 bandit**：HOO / X-armed bandits（Bubeck, Munos, Stoltz & Coquelin, 2011，[arXiv:1001.4475](https://arxiv.org/abs/1001.4475)）把动作空间做**层级划分**，每个 cell 的索引是"**cell 内最大值的置信上界**"，靠"把 cell 越切越细"来逼近全局最优。这是"bandit 作用在动作子空间上"最正统的模板，也解释了为什么我们的 `Q[c]` 用 max 型聚合是合理的：它本来就是在给一个**集合**（类）算"里面的最好能有多好"。

**组合 / factored / semi-bandit**：CUCB（Chen, Wang & Yuan, 2013，[ACM](https://dl.acm.org/doi/10.5555/3042817.3042836)；JMLR 扩展版 [ACM](https://dl.acm.org/doi/10.5555/2946645.2946695)）是"组合臂"的标准框架，但它的 regret 保证建立在 **semi-bandit 反馈**上（拉一个组合后能看到**每个组成臂**的贡献）。我们只看到**束的整体 rollout 分**，是 **full-bandit** 反馈 ⇒ 把束分平摊给每个类是有偏的（§6-2）。

**"利用项用 max / 软 max 而非均值"有没有对应？** 有，而且比想象的更直接：
- **mellowmax**（Asadi & Littman, 2017，[arXiv:1612.05628](https://arxiv.org/abs/1612.05628)）就是 `(1/β)log((1/n)Σe^{βxᵢ})`，论文把它当 softmax 的替代算子，证明了它落在 **mean 与 max 之间**、是**非扩张**算子、β 是温度旋钮（β→0 得 mean、β→∞ 得 max）。**我们的 `Q[c]` 与它逐字相同**，所以"利用项用软最大"这件事在算子层面**不是新的**，出处应该引 mellowmax。
- **Extreme bandits**（Carpentier & Valko, 2014，[DBLP](https://dblp.org/rec/conf/nips/CarpentierV14.html)）研究的是"奖励 = 样本的**最大值**"这类目标，给出了"max 型目标需要**更多样本**才分得清"的具体代价与重尾情形下的失效条件。
- **Best-arm identification**（LUCB，Kalyanakrishnan et al., 2012，[ACM](https://dl.acm.org/doi/abs/10.5555/3042573.3042606)）里"**经验最优臂**"天然就是 max 聚合，它的采样规则（"对当前冠军的挑战者加采样"）某种意义上正是"关心类里有没有好动作"的算法化版本。
- **明确"未找到"**：我没有找到任何论文把 LogSumExp / softmax **作为 UCB 的利用项**写进索引公式并做理论分析。也就是说，"mellowmax + 标准 UCB 探索项"这个**拼法**（而非算子本身）在文献里没有现成名字——这既是唯一勉强算"新"的地方，也意味着**没有任何经验常数可以抄**。

### 3.4 D. 预算分配

**Successive halving**（Karnin, Koren & Somekh, 2013，[PMLR v28](http://proceedings.mlr.press/v28/karnin13.html)）：每轮把预算翻 η 倍、候选砍到 1/η，直到只剩一个；论文证明这个"粗暴"策略在"只有一个好臂"的场景下**几乎最优**（只差 log 因子）。**Hyperband**（Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar, 2018，[arXiv:1603.06560](https://arxiv.org/abs/1603.06560)）把它包一层"外层扫 (n, r) 组合"，防止"η 选错导致全灭"，工程上的默认就是 **η=3**。这两篇就是我们"阶段 1 小预算筛、阶段 2 大预算评"的最直接对应物，而且给了一个**可以直接用的比例**：保留 1/3。

**Best-arm identification 的固定预算策略**：Bubeck, Munos & Stoltz（2009）的 "Pure exploration in multi-armed bandits"（[DBLP](https://dblp.org/rec/conf/alt/BubeckMS09.html)）给出固定预算下的淘汰式算法；LUCB（上引）是"每轮给冠军和挑战者各加采样"的自适应版。它们共同的教训是：**固定预算下，"每个臂先均匀试一次"是必要的（否则方差估计不可用）**，这解释了为什么标准 UCB1 要"未试过的臂优先"——我们的实现里 `_ucb` 返回 `+inf` 正是这条。

**一个更贴合我们的框架：Pandora's rule**（Weitzman, 1979，[Econometric Society](https://jstor.econometricsociety.org/publications/econometrica/1979/05/01/optimal-search-best-alternative)）。"开箱要花钱、箱子里有个价值的分布"——最优策略是给每个箱子算一个**保留值（reservation value）**，按保留值降序开箱，开到"下一箱的保留值 < 当前最好值"就停。我们的"阶段 1 便宜探测 / 阶段 2 昂贵精评"就是**带异质检查成本**的 Pandora 问题，它的结论可以直接翻译成一句话：**按"上限"排序而不是按"当前估计"排序，并且用"探测成本 / 精评成本"的比值决定要不要多探一个**。这条对我们"λ=0 还是 λ>0"的争论提供了一个更干净的替代问法。

### 3.5 E. 动作别名 / 去重

**棋牌游戏的"动作抽象 + 翻译"**是这个问题最成熟的战场。Ganzfried & Sandholm（2013）的 "Action translation in extensive-form games with large action spaces"（[ACM](https://dl.acm.org/doi/10.5555/2540128.2540148)）明确区分**硬翻译**（把抽象动作映到最近的合法具体动作 ⇒ 多个抽象动作映到**同一个**具体动作）与**软翻译**（按距离把概率/价值**分摊**给邻近具体动作），并证明硬翻译在抽象桶很粗时会产生系统性可利用（exploitable）偏差。Libratus（Brown & Sandholm, 2018，[Science](https://www.science.org/doi/10.1126/science.aao1733)）工程上也强调抽象后的"off-tree 动作"必须妥善翻译，否则会被对手针对。

**"掩码"这条支线**：invalid action masking（Huang & Ontañón, 2020，[arXiv:2006.14171](https://arxiv.org/abs/2006.14171)）指出：把非法动作的概率置零再重归一化，会**改变策略分布**并引入梯度偏差（它给的是"策略梯度"语境下的修正，但不影响这条教训的普适性）。对我们的含义：`can_apply` 过滤 + 去重之后，**"每类被采样的概率"和"每类实际贡献的候选数"已经不是一回事**，`N[c]` 的语义必须重新定义。

**macro-action / options**：Sutton, Precup & Singh（1999，[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370299000521)）的 options 框架是"把一串原始动作打包成一个动作"的正统理论；它的经典警告是**macro-action 之间会重叠/竞争**（同一个原始动作序列能被多个 option 产生），做统计时必须去重，否则会重复计数——这和我们 M6 的缓存问题是同一类。

**目标/记忆**：我**未找到** "intent-to-action 多对一映射在 search 里如何处理"的专门论文；最接近的就是上面的 action translation + transposition table（UCD，见下）**两头夹**。

### 3.6 F. 确定性 rollout 作为廉价评估

**静态评估函数**（棋类引擎的 eval）是"不 rollout、只看局面"的最极端版本，它的已知局限（评估函数与真实胜负的失配会**主导**搜索质量；深度换广度的边界）在棋类工程里被反复记录。**one-step lookahead** 是它加一层：走一步、评一次、取最好——这跟我们的"确定性 rollout ×1"在成本上是同一档。

**null-move pruning** 是"故意用一个**残缺模型**做廉价评估"的祖宗：搜的时候让对手**不走**，如果这样还够好，就认为"这步本来就够好"、整枝剪掉（[wiki](https://www.chessprogramming.org/Null_Move_Pruning)）。它的经典失效模式写在所有引擎教材里：**zugzwang**（"必须走棋反而更糟"的局面）——此时"白送一步"会得到**方向完全相反**的结论。**我们的 idle rollout（双方都不动）比 null-move 还激进**（null-move 只省掉对手的着法，我们是双方都省），所以 zugzwang 类错误只会更重。

**"一次采样够不够"**：如果模拟器**完全确定**、且被评估的量就是"这条轨迹的终局分"，那么 1 次 = 精确，方差为 0，多采是纯浪费——这一点我们的设计是对的。但要小心把这句话用错地方：我们**真正关心**的量是"在随机对手下的期望终局分"，idle rollout 估的是**另一个量**。James, Konidaris & Rosman（2017）在 "An Analysis of Monte Carlo Tree Search"（[AAAI](https://aaai.org/papers/11028-aaai-31-2017/)）里系统分析了 rollout 的偏差/方差权衡，核心结论之一是 **rollout 策略的系统偏差会盖过它的低方差优势**。

**"用学到的东西替换随机 rollout"是主流方向**：KataGo（Wu, 2019，[arXiv:1902.10565](https://arxiv.org/abs/1902.10565)）直接用策略/价值网替代随机 rollout，并给出"rollout 换成网络后强度大涨"的实测。这条对我们有直接启示：**阶段 2 的 64 次随机 rollout 本身也是"廉价近似"**，如果将来值网够强，"阶段 1=1 次确定性、阶段 2=64 次随机"这个阶梯可以整体换成"1 次 → 8 次 → 值网精评"。

**DAG / 缓存的理论侧**：把搜索树当 DAG（共享子树）会破坏 UCT 的备份假设。**UCD**（Upper Confidence bound for rooted Directed acyclic graphs，Knowledge-Based Systems, 2012，[ACM](https://doi.org/10.1016/j.knosys.2011.11.014)）专门处理"同一节点经多路径到达 ⇒ 统计量被重复备份"的问题；这条线近年演进为 **Monte-Carlo Graph Search**（例：ANN-CMCGS, AAMAS 2026 的 [PDF](https://ifmas.csc.liv.ac.uk/Proceedings/aamas2026/pdfs/KAUA8981.pdf)，该链接来自检索结果、我未逐页核对结论）。我们的"缓存命中 ⇒ 复用分数 + 计入 `Q[c]` + 推进游标"在结构上就是 DAG 备份的一个变体，**应当引这条线来解释为什么需要单独记"命中次数 vs 独立样本数"**。

---

## 4. 这套方案有多大程度是新的？

**标准做法（可以放心用、不用论证）**：

| 部件 | 判决 |
|---|---|
| 类级 UCB 选臂、`c√(log N/n)`、未试臂 +∞ | 教科书（UCB1, PUCT） |
| 用"静态先验排名 + 逐格推进"限制宽度 | progressive widening / DPW / KataGo-prior，标准 |
| 动作空间按"类型 → 参数"分解 | AlphaStar / OpenAI Five / SC2LE，标准 |
| 两阶段"小预算筛选 → 大预算评估" | successive halving / Hyperband，标准 |
| 同一具体动作只评一次（缓存） | transposition table，标准（但有 DAG 记账的坑） |
| 廉价 rollout 初筛（含"双方都不动"） | null-move pruning 的近亲，标准（但有 zugzwang 的坑） |

**有新意（或至少我没找到同构工作）的部分**，三处：

1. **`Q[c]` = 类内 rollout 分数的 mellowmax，再叠**标准**UCB 探索项**。算子是老的（mellowmax），拼法是新的。理论地位也很清楚：它是 HOO"cell 内最大值置信上界"的一个**平滑化 + 非置信界**版本——所以它**没有** HOO 的保证，别指望。
2. **"位置游标"按"动作类的带重数规范多重集"索引，每个形状独立推进**。这把"束的协同结构"和"位置枚举顺序"绑在一个键上，是相当工程化的设计。文献里最接近的是"macro-action 去重"和"Gumbel top-k 不放回采样"，都不是同一件事。
3. **"解码缓存命中 ⇒ 复用分数 + 计入 `Q[c]` + 游标推进"这条三元耦合**：缓存命中既当"证据"又当"进度"，这是**没见过的**。它的合理性论证（"LSE 对重复同值不敏感 ⇒ 重复计分不会撑虚 `Q[c]`"）**只在 `Q[c]` 这一侧成立**，在 `N[c]` 那一侧不成立（§6-3）。这是本文最想指出的一个**逻辑漏洞**。

**总结判断**：这是一套 **"已知部件的非标准拼装"**，学术新颖性低、工程针对性强。真正稀缺的不是想法，而是**耦合处的常数**（β、c、τ、α、η、k₂）——而这些常数**没有任何文献给过**，只能自己扫。反过来说，这也意味着：**如果实验结果好，别归因于"新方法"，要归因于"这几个常数在这个游戏上对了"**；如果不好，优先怀疑常数与耦合，而不是想法。

---

## 5. 可操作的借鉴（7 条，都能直接改）

**① β 不该是常数 3：`β ≳ log(n)/Δ`。**
mellowmax 的性质是 `max − log(n)/β ≤ Q ≤ max`（实现里 `_lse` 化简后就是 mellowmax）。也就是说 **β=3、n=10 时 `Q[c]` 最多可比类内最好的那个动作低 0.77**——而我们的分数值域只有 [−1, 1]，这个折扣相当于"把 max 拉回到中位偏上"。更糟的是：**折扣最大恰恰发生在"类里只有一个神手、其余都烂"的情形**（数学上就是那个极值），而这正是设计文档 §2 里说"我们关心的是类里**有没有**好动作"的那个情形 —— **实现与该意图方向相反**。
建议：把 β 设成随该类样本数增长，`β = max(β₀, log(n)/Δ)`，取 `Δ∈[0.1, 0.2]`（即"最多允许比 max 低 0.1–0.2"）：n=5 → β≈8–16，n=10 → β≈12–23，n=20 → β≈15–30。也可以就用 **top-m 均值（m=1 即 max）** 做对照组——设计文档 §2 已把它列为备选，从文献看**m=2~3 的 top-m 均值是更常见的做法**（对 max 的高方差做了一点平滑，又不像 mellowmax 那样在 n 大时失控）。**扫参建议：β ∈ {global 3, log(n)/0.2, top-3 均值} 三档对照，别再扫连续的 β。**

**② UCB 的 c 和 β 耦合在同一量纲上，先把 Q 的极差量出来。**
c=1.0 时，探索项 `c√(log N/n)` 在 `N_total≈144`（48 束 × ~3 头）下：n=1 → 2.23，n=3 → 1.29，n=10 → 0.70，n=30 → 0.41。而 `Q[c]` 的类间极差大概 ≤ 1（分数在 [−1,1]）。**这意味着在 n ≤ 10 的阶段，探索项完全主导，bandit 的行为约等于"轮转 + 极小偏好"**。建议：先打一行日志把 `max_c Q[c] − min_c Q[c]` 与 `c√(log N/n)` 同时记下来，让 `c` 定在"探索项 ≈ 极差的 10–20%"，即 `c ≈ 0.1~0.2 × range / √(log N/n)`。这比盲扫 [1.0, 2.0] 有信息量。

**③ 三个温度别一起扫：`β`、`c`、`τ` 会互相冒充。**
`_sample_ucb` 用 `softmax(UCB/τ)` 采样头 1..n−1，等于把"探索"又做了一遍（UCB 已经含探索，再 softmax 又是一种 Boltzmann 探索）。三个旋钮都能产生"多试几个类"的效果 ⇒ 三维扫参会跑出大量等价点、白烧算力。**建议固定 τ=1、只调 c；或者固定 c=0（纯贪心索引）只调 τ。** 文献里 Boltzmann 探索与 UCB 探索是**互斥的两套**（PUCT 只在 UCB 一侧加先验，不再在采样上加温度），混用没有理论指导。

**④ 两阶段比例照 successive halving：阶段 2 取 `K/η`，η=3（Hyperband 默认），最多到 6。**
算术上：阶段 1 的 48 条 × 1 次确定性 rollout ≈ **0.75 个**阶段 2 候选的成本（阶段 2 一条 = 64 次随机 rollout），所以"用阶段 1 筛"这件事几乎免费，**真正的预算旋钮是阶段 2 送几条**。K=48、η=3 ⇒ k₂=16；η=6 ⇒ k₂=8。若担心"筛错"，正确做法不是把 k₂ 提到 48（那等于不筛），而是**留 1–2 个"阶段 1 落选但类先验/位置图高分"的探索席位**（= Pandora 保留值的思想），并**按类分层取 top-k**（设计文档的方案 B）而不是全局 top-k——全局 top-k 在 idle 分数下会系统性地偏向"贪心建塔"这一类。

**⑤ 加一级预算阶梯：1 → 8 → 64，别从 1 直接跳到 64。**
LMR 的思路是"可疑着法先浅搜、好再深搜"，对应到我们就是把阶段 2 拆成"8 次随机 rollout 粗排 → 只剩 k₃ 条做满 64 次"。在**阶段 1 与阶段 2 分数可能反相关**的情形（§6-1），中间这一级能显著降低"筛错就全错"的代价：8 次的估计已经能看到对手随机性，成本只有满额的 1/8。

**⑥ 把游标的 α 从 1 降下来。**
现在的实现是"第 t 步取第 t+1 名"，即 `k(n)=n`（α=1，线性加宽）。DPW 的经验是 **α≈0.5**，即"访问父节点 4 次才多开 1 个位置"。α=1 在小预算下的表现就是"每个形状只摊 1 格"（我们文档 §78 记的塌缩现象与此一致）。**具体改法**：`k₂(n) = ⌈C·n^0.5⌉`，等价于"先让少数形状推进 2–3 格，而不是所有形状都推 1 格"；这跟"按形状分数优先推进"是同一件事的两种写法。**建议至少加一个 `AZAI_CAND_CURSOR_ALPHA` 开关做 A/B。**

**⑦ 去重的两层键继续用，但把 `N[c]` 换成"有效样本数"。**
现在的 `_pick_group` 对每次被选中的类都 `N[c] += 1`，**包括缓存命中的情况**。按 UCD/transposition 的教训，这会让 `N[c]` 虚高 ⇒ `c√(log N/n)` 偏小 ⇒ 该类**过早被判定为"已经充分探索"**。改法很便宜：维护 `N_seen[c]`（被选中次数）与 `N_eff[c]`（**解出新 bundle 的次数**，即真正做过 rollout 的次数），探索项用 `N_eff`，`N_seen` 只用于诊断。这同时能直接暴露"某类被记了很多次但只观测到 1 个不同动作"的别名塌缩（M9 的坑）。

---

## 6. 已知的坑（文献记录的失败模式 → 对我们的具体后果）

**① 确定性 idle rollout = null-move 的近亲，会踩 zugzwang。**
null-move pruning 的教科书级反例：**在必须回应的局面（zugzwang），"白送一步"的评估方向是反的**。我们的 idle rollout **双方都不动**，比 null-move 更激进：对手的威胁（正在推过来的塔、下一回合的反打）、"再不降级/再不闪电就要崩"这类**时效性代价，在 idle 世界里完全不存在**。后果是**系统性高估"进攻/建塔"、低估"防守/降级/闪电"**——而这两类恰好是设计文档里说"候选缺 HOLD、缺升级"的历史痛点区。
**可检验的做法**：把阶段 1（idle）分数与阶段 2（random，64 次）分数的 **Spearman 相关按局面分层**报告：`有敌方威胁` vs `无威胁`。如果前者的相关明显更低甚至为负，就说明阶段 1 在关键局面上是**有害**的，应该按威胁程度给阶段 1 降权、或对有威胁的局面直接跳过阶段 1。

**② 组合 bandit 的信用分配：我们是 full-bandit，不是 semi-bandit。**
CUCB 的全部保证建立在 semi-bandit 反馈上（能看到每个组成臂的贡献）。我们把"束的整体 rollout 分"回写给**每个**成员类（`_pick_group` → `_update`），意味着 `Q[build]` 里混进了"那一次搭档的 lightning 放得好不好"。多动作束（2–3 个动作）里，**搭档质量的相关性会污染类级统计**，而我们恰恰要靠类级统计决定"下一步该多试哪个类"。
**可检验的做法**：(a) 只让**单动作束**（含 HOLD）回写 `Q[c]`，2–3 动作束的分数只用于**排序候选**、不回写；(b) 或按 `1/|束|` 折扣回写；(c) 对比两种口径的 `Q[c]` 排序一致性。这在文献里没有现成答案，但**"full-bandit 的组合反馈有偏"是明确记录过的**。

**③ 缓存重复计数（DAG 备份）：`LSE` 免疫 ≠ `N` 免疫。**
设计文档 §2 的论证是"LogSumExp 对重复同值不敏感 ⇒ 重复计分不会撑虚 `Q[c]`"——**这一半是对的**（`LSE([x,x])=x`），但**另一半不成立**：`N[c]` 会涨。`N[c]` 涨 ⇒ 探索项 `c√(log N/n)` 缩 ⇒ 该类**被认为"已经探够了"**，于是 bandit 会去别处。也就是说，**别名塌缩会伪装成"这个类已经充分探索"**。UCD 那条线（DAG 上的 UCB）就是为了修这个，结论是**必须把"路径数"和"独立样本数"分开**。

**④ Progressive widening 的早熟收敛（宽度增长慢于价值增长）。**
Chaslot 的 widening 与 Świechowski 的综述都记录了同一个失效模式：**如果加宽太慢，真正的好动作会落在"还没解冻"的尾部，永远等不到访问**。我们的 α=1 + K=48 + 24 个类 ⇒ 平均每类只有 2 个观测，**大部分类的位置游标只走到第 1–2 名**。"位置图头部弱（建塔 @1 ≈ 1.2× 随机）"（设计文档 §2 自己写的）⇒ **游标约等于在随机前几名里打转**，宽度再大也没用。这条是"上游先验弱 ⇒ 下游搜索放大器放大噪声"的经典链条。

**⑤ max 型索引的乐观偏差 / winner's curse。**
Extreme bandits 的结论：**以"样本最大值"为目标时，需要更多样本才能分辨**，且重尾下 UCB 类方法会失效。我们是 mellowmax（比 max 温和），但 n 小时 `Q[c] ≈ x₁`（第一个样本）——**一次偶然的高分就能把一个类锁在 `argmax` 上很久**（因为探索项随 n 涨、`Q` 也不降）。**"类级 winner's curse"是这套方案最现实的失败模式**，且它与 ④ 会互相强化。

**⑥ 阶段 1 与阶段 2 的目标不一致（"脆"动作被系统性送进决赛）。**
设计文档 §2 提的 λ=0 决策，前提是"阶段 1 分数是阶段 2 分数的好代理"。但 idle 分数奖赏的是"对手不还手时收益最大"，随机化分数奖赏的是"对手还手时期望收益最大"——**这两者的最优点常常不同**（把金币 all-in 在一座塔上：idle 下极强，随机对手下很脆）。棋类剪枝（null-move / LMR / futility）的代价就是**漏掉少量好着**，且漏检率随评估函数失配上升。**这不是"多一点 k₂"能修的，是"筛选口径"问题。**

**⑦ 掩码 / 别名带来的统计口径混乱（M9）。**
通道 0–15 在空地**全部**解码成 `BUILD Basic`：这是最典型的"多对一"。后果有三：(a) 缓存去重后，**16 个塔通道只贡献 1 个候选**，`N[c]` 涨了 16 次但只观测到 1 个动作；(b) 塔通道之间的"偏好差异"（本可以表达"我更喜欢 Pulse 而不是 Cannon"）在**空地这个分支上信息被丢掉**；(c) 位置图的通道内量纲不同（设计文档 §2 提到"通道内 z-score 归一化"），做 top-k 时若跨通道比较原始分数，等于把不同量纲混比。action translation 的"硬翻译"教训正是 (a)+(b)：**折叠是必要的，但要显式记账**。

**⑧ 一个我推断的坑（非文献，标注为推断）：位置游标的"混合进制全枚举"在 19×19 下会退化成"top-1 排序"。**
混合进制按名次全扫，位置空间 361 格 ⇒ 预算内只能走前几个名次，**位置多样性 = 位置图 argmax 的稳定性**。如果位置图的最优点随局面抖动（弱先验的典型表现），游标就是在噪声上散步。**这解释了为什么"游标 + 缓存"可能反而比"每类取 argmax + 少数随机扰动"更差**——后者至少有显式的随机性。建议加一个"位置扰动"对照组。

---

## 7. 未找到 / 检索局限（诚实清单）

- **未找到**任何论文把 LogSumExp / softmax 作为 **UCB 的利用项**并做理论分析。"mellowmax 算子 + 标准 UCB 探索项"这个**拼法**没有现成名字与经验常数（算子本身有出处，见 §3.3）。
- **未找到**"意图 → 具体动作多对一解码 + rollout 缓存去重"在**搜索**文献里的对应工作。最接近是扑克的动作抽象/翻译（§3.5）与 transposition table（§3.6），但都不是同一问题。
- **未找到** AntWar2 这个游戏的任何公开文献/先行 AI 工作（它是赛题，不是研究对象）。
- **未找到** "用带重数的规范多重集索引搜索游标"的任何工作。
- **检索受限**：nature.com 正文（AlphaStar）、Semantic Scholar API、DBLP API、`export.arxiv.org` 在检索时段返回 429/跨域重定向，因此 AlphaStar 给的是 **DOI** 链接、Chaslot 2008 给的是 **Scholar-Archive** 记录；这两个条目的**标题、作者、年份**我有把握，但**发表版本的页码**未逐条核对。
- 本文所有"数值估算"（探索项大小、`log(n)/β` 折扣、成本比 0.75）都是**我按公式手算的**，标注了假设（`N_total≈144`、分数值域 [−1,1]、阶段 2 = 64 次），**未经实测**。

---

## 8. 参考文献（全部为本文引用过的条目）

**A. 组合动作空间 / 动作分解**
1. Vinyals, O. et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature 575, 350–354. [doi:10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z)（站点重定向，给 DOI；新闻报道可核：[nature.com/d41586-019-03298-6](https://www.nature.com/articles/d41586-019-03298-6)）
2. Berner, C. et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* [arXiv:1912.06680](https://arxiv.org/abs/1912.06680)
3. Vinyals, O. et al. (2017). *StarCraft II: A New Challenge for Reinforcement Learning.* [arXiv:1708.04782](https://arxiv.org/abs/1708.04782)
4. Tavakoli, A., Pardo, F., Kormushev, P. (2018). *Action Branching Architectures for Deep Reinforcement Learning.* [arXiv:1711.08946](https://arxiv.org/abs/1711.08946)

**B. 两段式 / 宽度控制**
5. Chaslot, G., Winands, M., van den Herik, J. (2008). *Progressive Strategies for Monte-Carlo Tree Search.* New Mathematics and Natural Computation. [Scholar-Archive 记录](https://scholar.archive.org/work/73t2pv5xajdqlg5lfhq3jnn7pm)
6. Couëtoux, A., Hoock, J.-B., Sokolovska, N., Teytaud, O., Bonnard, C. (2011). *Continuous Upper Confidence Trees*（DPW）. [doi:10.1007/978-3-642-25566-3_32](https://doi.org/10.1007/978-3-642-25566-3_32)
7. Świechowski, M. et al. (2023). *Monte Carlo Tree Search: A Review of Recent Modifications and Applications.* [arXiv:2103.04931](https://arxiv.org/abs/2103.04931)
8. Knuth, D., Moore, R. (1975). *An analysis of alpha-beta pruning.* Artificial Intelligence 6(4). 工程细节：[chessprogramming: Alpha-Beta](https://www.chessprogramming.org/Alpha-Beta)
9. Danihelka, I. et al. (2022). *Policy improvement by planning with Gumbel*（Gumbel top-k 不放回采样）. [ICLR 2022](https://www.iclr.cc/virtual/2022/poster/6418)

**C. Bandit / UCB / 子空间**
10. Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem.* [doi:10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352)
11. Rosin, C. (2011). *Multi-armed bandits with episode context*（PUCT）. [doi:10.1007/s10472-011-9258-6](https://dl.acm.org/doi/abs/10.1007/s10472-011-9258-6)
12. Silver, D. et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go*（AlphaZero）. [arXiv:1812.03828](https://arxiv.org/abs/1812.03828)
13. Bubeck, S., Munos, R., Stoltz, G., Coquelin, P. (2011). *X-Armed Bandits*（HOO）. [arXiv:1001.4475](https://arxiv.org/abs/1001.4475)
14. Chen, W., Wang, Y., Yuan, Y. (2013). *Combinatorial Multi-Armed Bandit: General Framework and Applications*（CUCB）. [ACM](https://dl.acm.org/doi/10.5555/3042817.3042836)；JMLR 扩展版 [ACM](https://dl.acm.org/doi/10.5555/2946645.2946695)
15. Carpentier, A., Valko, M. (2014). *Extreme Bandits.* [DBLP 记录](https://dblp.org/rec/conf/nips/CarpentierV14.html)
16. Kalyanakrishnan, S. et al. (2012). *PAC Subset Selection in Stochastic Multi-armed Bandits*（LUCB）. [ACM](https://dl.acm.org/doi/abs/10.5555/3042573.3042606)
17. Asadi, K., Littman, M. (2017). *An Alternative Softmax Operator for Reinforcement Learning*（mellowmax）. [arXiv:1612.05628](https://arxiv.org/abs/1612.05628)
18. Bubeck, S., Cesa-Bianchi, N. (2012). *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems.* [arXiv:1204.5721](https://arxiv.org/abs/1204.5721)

**D. 预算分配**
19. Karnin, Z., Koren, T., Somekh, O. (2013). *Almost Optimal Exploration in Multi-Armed Bandits*（successive halving）. [PMLR v28](http://proceedings.mlr.press/v28/karnin13.html)
20. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. (2018). *Hyperband.* [arXiv:1603.06560](https://arxiv.org/abs/1603.06560)
21. Bubeck, S., Munos, R., Stoltz, G. (2009). *Pure Exploration in Multi-armed Bandits Problems.* [DBLP 记录](https://dblp.org/rec/conf/alt/BubeckMS09.html)
22. Weitzman, M. (1979). *Optimal Search for the Best Alternative*（Pandora's rule）. [Econometric Society](https://jstor.econometricsociety.org/publications/econometrica/1979/05/01/optimal-search-best-alternative)

**E. 动作别名 / 抽象 / 掩码**
23. Ganzfried, S., Sandholm, T. (2013). *Action Translation in Extensive-Form Games with Large Action Spaces.* [ACM](https://dl.acm.org/doi/10.5555/2540128.2540148)
24. Brown, N., Sandholm, T. (2018). *Superhuman AI for heads-up no-limit poker: Libratus.* Science. [science.org](https://www.science.org/doi/10.1126/science.aao1733)
25. Huang, S., Ontañón, S. (2020). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* [arXiv:2006.14171](https://arxiv.org/abs/2006.14171)
26. Sutton, R., Precup, D., Singh, S. (1999). *Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in RL*（options / macro-actions）. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370299000521)

**F. 廉价评估 / 确定性 rollout / 缓存**
27. chessprogramming wiki: [Null Move Pruning](https://www.chessprogramming.org/Null_Move_Pruning)、[Late Move Reductions](https://www.chessprogramming.org/Late_Move_Reductions)
28. James, S., Konidaris, G., Rosman, B. (2017). *An Analysis of Monte Carlo Tree Search.* [AAAI](https://aaai.org/papers/11028-aaai-31-2017/)
29. Wu, D. (2019). *Accelerating Self-Play Learning in Go*（KataGo）. [arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
30. *UCD: Upper Confidence Bound for Rooted Directed Acyclic Graphs.* Knowledge-Based Systems (2012). [ACM](https://doi.org/10.1016/j.knosys.2011.11.014)
31. *ANN-CMCGS: Generalizing Continuous Monte-Carlo Graph Search with Approximate Nearest Neighbors.* AAMAS 2026. [PDF](https://ifmas.csc.liv.ac.uk/Proceedings/aamas2026/pdfs/KAUA8981.pdf)（链接来自检索结果，结论未逐页核对）

---

## 9. 落到我们自己文档上的三个具体改动建议（按性价比排序）

1. **改 `_lse` 的 β**：`AZAI_CAND_BETA` 从常数 3.0 改成 `log(n)/Δ`（Δ=0.15），或加一档 `top-m 均值`。理由：§5-①，且这是**唯一有明确数学上界**的一处。
2. **`N[c]` 拆成 `N_seen` / `N_eff`**：探索项用 `N_eff`。理由：§6-③（DAG 重复计数）+ §5-⑦。改动量最小、直接消除一个逻辑漏洞。
3. **加一个"有威胁局面跳过阶段 1"或"按威胁分层"的开关**：理由：§6-①（null-move 的 zugzwang）+ §6-⑥（脆动作被送进决赛）。可以先只做**诊断**（分层报告 Spearman 相关），不改行为——这样零风险、直接回答"阶段 1 到底有没有害"。
