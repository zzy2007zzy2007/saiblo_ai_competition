# 资源/经济取舍的价值函数建模 —— RL 与相邻领域文献调研

> 日期：2026-11-02
> 主题：怎么让价值函数正确判断"现在花掉资源（买持久资产）还是攒着（换一次性高价值动作）"。
> 我们的具体病：一个 42→64→64→1 的 MLP 价值网（叶子评估器，用"终局血量差"当标签做回归）判不好金币该花还是该攒——读不出"攒 90 币放闪电"这个跨时间的经济结构（阈值价值、时间价值、流动性）。

---

## §0 先把问题翻译成三个"经济学结构"（便于后面对齐术语）

我们的病可以精确拆成三个经典结构，这决定了该去哪个领域找关键词：

1. **阈值价值（threshold / strike）**：金币在 90 处有一个"折点"——<90 放不了闪电，≥90 能放。价值网对金币是弱线性（实测 `∂V/∂(金币/300) ≈ +0.027`，约 `+0.037/币`），没有 90 这个折点。这对应金融里**期权执行价（strike price）**与**实值/虚值（moneyness）**：90 币就是"闪电看涨期权"的执行价，<90 是虚值（out-of-the-money）。

2. **时间价值（time value）**：越接近"能放闪电"，金币越值钱；价值网无视冷却。这对应**货币时间价值（time value of money）**与**期权的剩余期限（time-to-expiry）**。

3. **流动性 vs 持久资产（liquidity vs illiquid）**：金币是**流动性资产**（随时能变成任意塔/武器），塔是**持久/半不可逆资产**（造了只能按 90%×残血比例返还，且塔价 `15×2^i` 指数增长）。价值网在短 horizon 叶子只认"持久资产+省下的钱"，不认"一次性伤害"收益（叶子落地时收益已结算）。这对应**流动性溢价（liquidity premium）/ 非流动性折价（illiquidity discount）**、**投资的不可逆性 + 沉没成本（sunk cost）**，以及**预防性储蓄（precautionary savings）**：持有现金 = 保留"未来某个时点买闪电"的选择权。

一句话：**这是一个"静态叶子评估器读不出跨时间经济结构"的问题，本质上 = 延迟奖励的时间信用分配（RL）+ 持有流动性的期权价值（金融/实物期权）。**

---

## §1 术语 / 关键词表

下表第一列是我们的叫法，第二列是对应英文术语，第三列是领域，第四列是它和我们的对应关系。

| 我们的叫法 | 英文术语 | 领域 | 与我们的对应 |
|---|---|---|---|
| 金币该花还是该攒 | consumption–savings / spend-vs-save | 经济学（跨期选择） | 攒=储蓄，买塔=消费/投资 |
| 攒钱换一次性大招 | **real option / option value of waiting**（实物期权 / 等待的期权价值） | 金融 / 实物期权 | 持有金币=持有一个"买闪电"的看涨期权 |
| 90 币这个折点 | **strike price / moneyness**（执行价 / 实值虚值）；threshold policy | 期权 / 最优停时 | 90=执行价；<90 虚值 |
| 越接近能放越值钱 | **time value of money / time-to-expiry**（货币时间价值 / 到期时间） | 金融 | 冷却倒数=期权剩余期限 |
| 金币=流动性、塔=持久 | **liquidity premium / illiquidity discount**；sunk cost & irreversibility | 金融 / 实物期权 | 现金保留灵活性，塔半不可逆 |
| 攒钱防风险 | **precautionary savings / buffer stock**（预防性储蓄 / 缓冲库存） | 经济学 | 保留现金作为保险（已有 `safe_coin` 雏形） |
| 一次性收益在叶子已结算 | **delayed reward / temporal credit assignment (CAP)**（延迟奖励 / 时间信用分配） | 强化学习 | 终局收益要回溯分配给"攒钱"那个决策 |
| 高层"攒钱 vs 花钱"+低层执行 | **hierarchical RL / options / macro action**（分层 RL / 选项 / 宏动作） | RL | 时间抽象 |
| 资源/预算约束 | **budgeted RL / constrained MDP (CMDP)** | RL / 运筹学 | 金币=预算约束 |
| 何时触发一次性动作 | **optimal stopping**（最优停时） | 运筹学 / 随机控制 | 何时放闪电 |
| 预测未来资源 | **cash-flow / resource trajectory prediction**（现金流预测） | RL 辅助任务 / 金融 | 预测未来金币轨迹 |
| 游戏里"经济/宏观" | **economy / macro / resource management**；**spending quotient** | 游戏 AI | StarCraft/Dota 的"花 vs 攒"效率 |

**实际有效的搜索词**（这些组合直接命中了相关文献，建议复用时照抄）：

- `reinforcement learning option value real options value of waiting` → 实物期权 + RL 的交叉文献
- `reinforcement learning temporal credit assignment delayed reward` → 命中关键综述 [2312.01072]
- `RUDDER return decomposition delayed reward` → 回报分解
- `Temporal Value Transport credit assignment Hung 2019` → 时间价值传输
- `potential-based reward shaping Ng Harada Russell 1999` → 奖励塑形
- `auxiliary tasks reinforcement learning reward prediction UNREAL` → 辅助任务
- `constrained Markov decision process budget reinforcement learning Lagrangian` → 约束/预算 RL
- `hierarchical reinforcement learning options framework Sutton Precup Singh temporal abstraction` → options/分层 RL
- `real options investment under uncertainty Dixit Pindyck value of waiting` → 实物期权经济学
- `precautionary savings buffer stock liquidity constraint` → 预防性储蓄
- `AlphaStar build order z statistic macro` → 游戏 AI 经济
- `Hindsight Experience Replay HER` → 后见之明经验回放

---

## §2 已知方法综述

### A. 时间信用分配 / 延迟奖励（最直接对口的一族）

- **TD(λ) / eligibility traces（资格迹，Sutton & Barto）**：把单步 TD 与蒙特卡洛回报线性插值（λ-return），让"远处的收益"通过资格迹回流到"早先的决策"。思想是对付延迟奖励的经典工具；我们的"攒钱→几十回合后放闪电→赢"正是需要把终局收益回流到攒钱状态，但**前提是价值网用 TD 自举训练**——我们现在是回归到终局标签，直接套 TD(λ) 要先改成 TD 目标。
- **RUDDER: Return Decomposition for Delayed Rewards**（Arjona-Medina et al., NeurIPS 2019，[arXiv:1806.07857]）：用 LSTM + 贡献分析把"只有回合末才出现"的终局回报**分解并前移（redistribute）**成每一步的密集奖励，从而把信用分配问题转成有监督的回报分解问题。和我们的病几乎同构：终局血量差是唯一信号，"攒钱"状态拿不到自己的那份 credit。
- **Temporal Value Transport (TVT)**（Hung et al., Nature Communications 2019，[arXiv:1810.06721]）：用**注意力机制把"未来才兑现的价值"沿时间轴搬运回早期状态**，显式建模长程依赖，专门解决"价值跨长时间尺度优化"的问题。名字本身就是我们问题的答案方向。
- **Hindsight Credit Assignment (HCA)**（Harutyunyan et al., 2019）：用**反事实**"如果我当时没做这个动作，结局会不会不同"来给动作分配信用。适合"一次性大招该不该放"这种离散关键决策的反事实评估。
- **Hindsight Experience Replay (HER)**（Andrychowicz et al., 2017，[arXiv:1707.01495]）：把"没达成的目标"事后改写为"已达成"，把稀疏奖励变稠密。思路可迁移到"把没放成的闪电事后当作目标"来制造正样本，但更偏目标条件 RL，和我们的回归式叶子评估器不是完全同构。
- **A Survey of Temporal Credit Assignment in Deep RL**（Pignatelli et al., [arXiv:2312.01072]）：把 CAP 统一成"学习动作对结果的**影响（influence）**"，并系统梳理延迟、转置、无影响等挑战与方法。是这一族的总纲，强烈建议作为主参考文献。

### B. 奖励塑形 / 奖励分解

- **Potential-based reward shaping**（Ng, Harada & Russell, ICML 1999）：若塑形奖励是**势函数之差** `F = γφ(s') − φ(s)`，则不改变最优策略（policy invariance）。可以用 `φ(s) = f(距阈值距离, 冷却倒数)` 把"接近能放闪电"的结构灌进奖励——**但注意**：这只在"用奖励信号训练策略/价值"的 RL 回路里有效；我们现在是**有监督回归到终局标签**，塑形没有可塑的对象，除非切换到 RL 训练。
- **Reward decomposition（奖励分解）**：把总奖励拆成可解释子项（如"金币收入项 + 伤害项 + 资源库存项"），分别学各自的价值再合成。优点是让网络显式看到"金币项"与"伤害项"是两回事，避免被平均掉。

### C. 辅助预测任务（把结构灌进表征）

- **UNREAL / Reinforcement Learning with Unsupervised Auxiliary Tasks**（Jaderberg et al., 2017，[arXiv:1611.05397]）：在共享表征上加**无监督辅助目标**（reward prediction、value replay、pixel control），让表征学会"只靠主奖励学不到"的规律。**这是对我们病最省力的一招**：把辅助目标换成"预测未来金币轨迹 / 未来能否放闪电"，就能把"攒钱→未来可放闪电"这条跨时间结构直接写进 64 维表征，而不依赖终局标签。

### D. 时间抽象 / 分层 RL

- **Options framework**（Sutton, Precup & Singh, AIJ 1999）：把"一段时间内的策略"封装成**选项（option）**，提供时间抽象。可定义高层选项"攒钱攒到 90 再放闪电"与低层"造塔/升塔"。
- **Option-Critic**（Bacon, Harb & Precup, AAAI 2017，[arXiv:1609.05140]）：端到端学习 options 及何时终止。适合"高层攒钱 vs 花钱 + 低层执行"，但**它修的是策略/结构层，不是叶子评估器**——价值网该瞎还是瞎，除非配合价值函数一起学。

### E. 约束 / 预算 RL

- **Constrained MDP (CMDP) / Lagrangian / CPO / Augmented Lagrangian**：把"资源≤预算"写成约束，用拉格朗日乘子或增广拉格朗日求解。与"预算/资源受限"直接相关（金币=预算），但我们的病**不是"如何不超支"，而是"如何给'以后再花'这个选项定价"**，所以是相邻但不对症的一族。预算约束与流动性的区别：约束是"上限"，流动性是"选择权"。

### F. 经济学 / 运筹学（给出"等待=期权"的定价语言）

- **Real options theory**（Dixit & Pindyck, *Investment under Uncertainty*, 1994）：当投资**不可逆**且未来**不确定**时，"等待再投资"有正的期权价值；净现值（NPV）法则会系统性低估等待的价值。**这正好解释我们的病**：塔是半不可逆投资，金币是"买闪电"的期权，叶子评估器只看 NPV（当前持久资产+剩余现金），漏掉了"等待的期权价值"。交叉方向也有工作把最优停时/实物期权写成 RL（如 [ETH 论文]、[Hierarchical DRL for asset management under budget constraints]）。
- **Precautionary savings / buffer stock**（Carroll 等）：收入不确定时，理性主体会持有一个**预防性现金缓冲**，缓冲的价值是"避免未来被迫在坏时机变卖/放弃机会"。与"攒 90 币保闪电"是同一逻辑的连续版本。
- **Optimal stopping（最优停时）**：一次性动作何时触发的最优规则是**阈值策略**（threshold policy）——状态越过某个临界值才触发。90 币 + 冷却好 = 触发闪电的阈值。

### G. 游戏 AI 实际例子

- **AlphaStar**（Vinyals et al., *Nature* 2019，[s41586-019-1724-z]）：没有显式的"经济价值函数"，而是用**build order 统计量 z**（把"开局造了什么/升了什么"编码成条件变量）让策略在"宏观 build order"之间切换，配合自对弈联赛 + 仅回合末胜负奖励学习。启示：经济/宏观被当作**高层条件变量**显式暴露给策略，而不是指望一个标量价值网从原始状态里自己悟出来——与我们"给价值网显式喂经济结构"的思路一致。
- **OpenAI Five**（Berner et al., 2019，[arXiv:1912.06680]）：Dota 经济（补刀/装备/买活）靠团队奖励 + 大规模自对弈隐式学习，未单独建模"流动性/攒钱"。
- **StarCraft 宏观术语**：社区用 **macro / resource management**，并用 **Spending Quotient (SQ)**（实际花费/总收入的比值，衡量"花 vs 攒"效率，过度攒钱会被 SQ 惩罚）作为评估指标——"该花还是该攒"在游戏 AI 里已有可度量指标，说明它是被长期关注的问题。

---

## §3 最对症的 1–3 个方法 + 落到我们设置的思路

先说诊断，再给处方。我们的价值网**不是**在用 TD 自举，而是**有监督回归到终局血量差**；它读不出经济结构，根因有两层：

- **表征层**：42 维输入里金币是 `coins/300`（无阈值参照）、冷却全是 `cd/时长` 比值（闪电冷却 35），**没有**"距 90 的距离"、没有"买得起 且 冷却好"的合取、没有"距能买还差几回合"、没有"流动性盈余"。让 64 个隐单元去学 `(coins/300 ≥ 0.3) ∧ (cd/35 == 0)` 这种又稀又尖的非线性交互，从单一终局标量里几乎学不出来——于是金币只能退化成弱线性 `+0.037/币`。
- **标签层**：终局血量差是回合末才兑现的单点信号，"攒钱→放闪电→赢"的收益离攒钱决策几十回合远，回归标签把稀有的闪电 payoff 平均/模糊掉了。

三个处方按"性价比 + 对症度"排序：

### 处方 1（最便宜、最对症）：特征工程 + 实物期权归纳偏置

把"等待的期权价值"直接写成输入特征，让阈值/时间/流动性从**要学出来的非线性交互**变成**线性可见的输入**。在 `Ant-Game/SDK/utils/features.py` 的 `encode_stats` 里，给每个超级武器（至少闪电 90、EMP 135）追加：

- `moneyness_w = clip((coins − cost_w) / cost_w, −1, 1)` —— 距执行价的距离（<0 虚值，>0 实值）
- `fireable_w = 1[coins ≥ cost_w] · 1[cd_w == 0]` —— "买得起 且 冷却好"的合取（把那个稀有的乘积直接算好喂进去）
- `time_to_afford_w = max(0, (cost_w − coins) / income_rate)` —— 货币时间价值（"还差几回合能买"）
- `liquidity_surplus = coins − max(reserve_for_lightning, next_tower_cost)` —— 流动性盈余（攒的钱扣掉"预留闪电"和"下一座塔 15·2^i"后还剩多少）

为什么对症：这四条分别精准对应诊断里的阈值、时间、流动性。它不是给网络"更多信息"，而是把网络**本来就在瞎猜的那个折点**变成一阶输入。代价极小：只改特征向量（42→46~50），数据、标签、网络架构都不动，用同一份终局标签重训即可。验证方式已有现成探针（`code/test_match/probe_value_coin_curve.py`）：重训后应看到 `∂V/∂金币` 在 90 处出现台阶，而不是一条平缓直线。

### 处方 2：UNREAL 式辅助预测头（把跨时间经济结构灌进 64 维表征）

在价值网 64 维 stats embedding 上并接 2–3 个辅助头，和主头（终局血量差）**共享躯干联合训练**：

- `coins_{t+k}` 回归（预测 k 回合后的金币），或
- `fireable_lightning_{t+k}` 分类（k 回合后"能否放闪电"），或
- `time_until_lightning_ready` 回归（离"冷却好且买得起"还有几回合）。

为什么对症：主标签（终局血量差）不包含"攒钱→未来可放闪电"这个方向的信息，但**回放里每一回合的金币、冷却、是否放了闪电都完整记录了**（`replay.json` / self-play dump 里都有），所以辅助目标有充足、免费的监督。它强制共享表征编码"现在攒钱 = 未来获得闪电选择权"，正好补上主标签漏掉的时间价值与阈值结构。这是 UNREAL "reward prediction" 思想对**资源轨迹**而非奖励的移植。成本低：只加头、不改数据、不动躯干。

### 处方 3：时间信用分配 / 价值传输（RUDDER / TVT 或 λ-return 思想，修标签）

把"终局血量差"这个单点标签换成**把收益回溯分配后的稠密 credit**：

- 若走**回归路线**：用 RUDDER 式回报分解（小 LSTM/注意力网络沿"金币/血量/冷却轨迹"把终局值重分配到每个回合），或用 TVT 式注意力做时间价值搬运，得到"每个状态其实值多少"的稠密标签，再回归价值网——让"攒钱"状态直接拿到"未来能放闪电"那份 credit。
- 若改走 **TD 路线**：把训练目标从"回归终局值"换成 `V(s) ← r + γ V(s')` 的 λ-return / n-step 自举，让收益通过资格迹回流。

为什么对症：这是对"延迟奖励"的正统解法，直击标签层根因。之所以排第三：它要动**目标计算管线**（当前是单点终局回归），改动最大；且如果处方 1+2 已经把结构显式写进输入和表征，处方 3 的边际收益会被部分吃掉。**建议先做 1+2，若价值网对"攒钱"状态仍低估，再上 3。**

### 明确"不太对症 / 慎用"的两类（避免走弯路）

- **Potential-based reward shaping**：我们当前是回归式叶子评估器，没有可塑的奖励信号，塑形无处着力；只有将来切成 RL 训练回路才有意义（那时可用 `φ = f(moneyness, cooldown)`）。
- **Options / 分层 RL、constrained/budgeted RL**：前者修的是**策略结构**而非**叶子评估器**（价值网该瞎还是瞎）；后者解决"如何不超支"而非"如何给'以后再花'定价"。可作为后续的补充，但不是当前病的解药。

---

## §4 参考链接

**时间信用分配 / 延迟奖励**

- [A Survey of Temporal Credit Assignment in Deep Reinforcement Learning](https://ar5iv.labs.arxiv.org/html/2312.01072)（arXiv:2312.01072）
- [RUDDER: Return Decomposition for Delayed Rewards — arXiv:1806.07857](https://ar5iv.labs.arxiv.org/html/1806.07857)
- [RUDDER — NeurIPS 2019 proceedings](https://proceedings.nips.cc/paper_files/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf)
- [Optimizing Agent Behavior over Long Time Scales by Transporting Value (TVT) — arXiv:1810.06721](https://ar5iv.labs.arxiv.org/html/1810.06721)
- [TVT — Nature Communications 2019](https://www.nature.com/articles/s41467-019-13073-w)
- [Hindsight Experience Replay — arXiv:1707.01495](https://arxiv.org/abs/1707.01495)

**辅助任务 / 表征**

- [Reinforcement Learning with Unsupervised Auxiliary Tasks (UNREAL) — arXiv:1611.05397](https://ar5iv.labs.arxiv.org/html/1611.05397)

**时间抽象 / 分层 RL / 约束 RL**

- [The Option-Critic Architecture — arXiv:1609.05140](https://arxiv.org/abs/1609.05140)
- [Policy invariance under reward transformations (potential-based shaping), Ng–Harada–Russell 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

**经济学 / 实物期权 / 最优停时**

- [Hierarchical Deep Reinforcement Learning for multi-year asset management under budget constraints](https://www.tandfonline.com/doi/full/10.1080/15732479.2026.2704183)
- [Casting optimal stopping into a unifying reinforcement learning framework (ETH thesis)](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/675732/Thesis_JohnEry.pdf)
- [Extended Abstract: Using Reinforcement Learning in Applied Real Options Modelling](https://realoptions.org/programAbstracts2023/1/39.pdf)
- Dixit & Pindyck, *Investment under Uncertainty* (1994), Princeton University Press（实物期权经典教材）

**游戏 AI 实例**

- [AlphaStar — Nature 2019, s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z)
- [OpenAI Five (Dota 2 with Large Scale Deep RL) — arXiv:1912.06680](https://arxiv.org/abs/1912.06680)

**我们仓库内的相关诊断/探针（落地时直接复用）**

- [Ant-Game/SDK/utils/features.py](../Ant-Game/SDK/utils/features.py) —— 42 维特征构造（处方 1 的改点）
- [code/test_match/probe_value_coin_curve.py](../code/test_match/probe_value_coin_curve.py) —— 金币价值曲线探针（处方 1 的验证）
- [code/test_match/probe_value_grad.py](../code/test_match/probe_value_grad.py) —— `∂V/∂stats` 梯度探针

---

## 最终三句话

**① 这个问题学界/业界叫什么、有哪些有效关键词**：在金融/经济学里叫 **real options（实物期权）/ option value of waiting（等待的期权价值）/ time value of money（货币时间价值）/ liquidity premium（流动性溢价）/ precautionary savings（预防性储蓄）**，把"攒 90 币放闪电"精确建模成"持有一个执行价=90、剩余期限=冷却倒数的看涨期权"；在 RL 里叫 **temporal credit assignment / delayed reward（时间信用分配/延迟奖励）**，配套 return decomposition、value transport；在游戏 AI 里叫 **economy / macro / resource management**（附 spending quotient 指标）。有效搜索词见 §1。

**② RL 里最成熟的几个相关方法**：TD(λ)/eligibility traces、**RUDDER（回报分解）**、**TVT（时间价值传输）**、HCA（后见之明信用分配）、HER、potential-based reward shaping、**UNREAL 辅助预测任务**、options/分层 RL、constrained/budgeted RL（CMDP/Lagrangian）。

**③ 最对症我们病情的 1–3 个方法 + 落地思路**：
1. **特征工程 + 实物期权归纳偏置**——在 `encode_stats` 里给价值网直接喂 `moneyness`（coins−90 归一）、`fireable`（买得起∧冷却好）、`time_to_afford`（还差几回合）、`liquidity_surplus`（预留闪电后的流动性盈余），把阈值/时间/流动性从"要学出的非线性交互"变成线性输入，重训同一终局标签，用 `probe_value_coin_curve.py` 验证 90 处出现台阶（最便宜、最对症）。
2. **UNREAL 式辅助预测头**——在 64 维 stats embedding 上并接"预测未来金币 / 未来能否放闪电"的头联合训练，把"攒钱→未来可放闪电"灌进共享表征（免费监督来自 replay 里已记录的金币/冷却轨迹）。
3. **RUDDER/TVT 或 λ-return 式信用分配**——把单点终局标签换成回溯分配后的稠密 credit，让"攒钱"状态拿到"未来放闪电"那份信用（最彻底，但要动目标管线，建议 1+2 不够再上）。
