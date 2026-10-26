# 「动作多到不能全枚举」时，AlphaZero 式搜索该怎么做？

> 调研对象：Ant-Game 六边形塔防对抗（我们自己的 AI：位置先验网 → 266 个候选动作束 → 每候选 16 回合 rollout → 值网打分 → 取 top1）
> 本文只关心一个问题：**在候选多、预算有限、单次评估带噪的情况下，「选哪个候选」这一步的正确做法**。
> 调研时间：2026-10-26。所有结论都标注了「我读到了什么」与「哪些是二手转述」，第 5 节是完整的阅读台账。
> 本文不改任何代码，也不涉及正在跑的实验。

---

## 1. 结论摘要（先给答案）

1. **我们现在的做法在文献里对应「固定预算、均匀分配、一次采样定胜负」的最坏情况**：`flat_rollout_greedy.py` 里
   `per_candidate_rounds = max_total_rounds // (n_cand * horizon)`，即把 39000 回合预算**平均**摊给所有候选，
   然后 `np.nanargmax(scores)` 硬取 top1。文献里这条路线叫 "fixed-allocation top-k selection"，
   而在**固定总预算**下被研究得最透、最有理论保证的做法是 **Sequential Halving（SH）**：
   同样一份预算，SH 是「每轮把所有存活候选都摸一遍 → 砍掉一半 → 每轮每人样本数翻倍」，
   **样本集中在最后几个幸存者身上**，而不是所有人一样多。同一份 39000 回合预算下，我们现在的名额分配是 `160 个 × 15 次`；
   SH 可以是 `266 个 × 1 次 → 133 × 2 → … → 冠军 ≈338 次`。**这是本文性价比最高的一条改动**。

2. **数字上我们现在就在「选择基本靠噪声」的区间**，这也解释了「k=160 掉、k=48 好、换先验就 28–44%」：
   你们自己的实测（`code/my_ai/distill_top2/fit_q8_baseline.py:175` 的注释）是**单次 rollout σ≈111、top1-top2 分差中位 ≈71**。
   266 个候选取 1 次采样做 max，噪声的期望最大值约 `σ·√(2 ln k) ≈ 3.34σ ≈ 371`，是真实分差（71）的 **5 倍**；
   k=48、n=50 时噪声降到 `111/√50·2.78 ≈ 44`，才勉强低于 71。
   **候选数 k 与每候选样本数 n 是同一个预算的两种花法，我们现在把预算几乎全花在"广度"上，而"深度"才是决定 top1 对不对的东西。**

3. **Gumbel AlphaZero / Gumbel MuZero 可以替换我们的「独立打分取 top1」，但不是直接换，是换掉选择层**：
   它的核心不是"加噪声探索"，而是**用同一组 Gumbel 噪声同时完成「采样候选集」和「在候选集内选动作」**，
   从而在**只评估 k 个动作**的前提下仍然满足 `E[q(A)] ≥ E_{a~π}[q(a)]`（相对先验策略的策略改进）。
   我们这种「根节点一次性评估、没有树」的结构恰好只需要它**根节点**那一半（Sequential Halving + Gumbel-top-k + completed-Q），
   不需要 MuZero 的树/模型部分，落地成本比想象的低。
   链接：[ICLR 2022 官方页（摘要已读）](https://www.iclr.cc/virtual/2022/poster/6418)、
   [OpenReview](https://openreview.net/forum?id=bERaNdoegnO)（**PDF 我这边 403，见 §5**）、
   [作者官方 slides（已读全文）](https://www.iclr.cc/media/iclr-2022/Slides/6418.pdf)。

4. **「评估贵」这一类确实有专门方法，而且就是我们缺的那一环**：固定预算 best-arm identification（BAI）/
   successive halving / multi-fidelity（Hyperband、BOHB）。
   它们的共同前提是「**每次拉臂都要花钱，所以要把钱花在还有希望的臂上**」——这正是"每个候选要 16 回合 rollout + 一次值网"的处境。
   反过来，UCT/UCB 优化的是 **cumulative regret**（过程中攒奖励），我们要的是 **simple regret**（最后选中谁），
   **目标函数不匹配**，所以 UCT 式预算分配不该直接搬。
   Gumbel 论文的根节点做法就是这么选的（讲义第 35 页转述原文 "Planning on Stochastic Bandit" 一节：
   根节点目标是 `E[q(A_{n+1})]` 即 simple regret，UCB 的最小化累计后悔不是这个目标；
   作者比较了几种 simple-regret 算法后选 SH，理由是**参数无关、更好调**）。
   [讲义 PDF](https://ufal.mff.cuni.cz/~straka/courses/npfl139/2324/slides.pdf/npfl139-2324-12.pdf)。

5. **必须先分清「选错」和「评估本身有偏」**：SH / BAI 只能消**方差**，消不掉**偏差**。
   我们的 rollout 里对手 idle / 随机，这个近似是系统性偏差；即使把冠军采样 1000 次，也只是把
   `q̂(对手idle)` 估得很准，不是把真实胜率估准。
   所以正确的组合是：**SH 管"谁被采样多少次"（方差）+ Gumbel/completed-Q 的 softmax 收缩管"先验与估计的加权"（偏差与过拟合）**，
   两者缺一不可；只做前者会出现"更自信地选错"。

---

## 2. 逐方法：核心思想、关键假设、能不能用

格式：**方法** → 链接 → 一句话核心 → 关键假设 → 对我们的结论（能 / 不能 / 需改造）+ 理由。

### 2.1 Gumbel AlphaZero & Gumbel MuZero（Danihelka, Guez, Schrittwieser, Silver, ICLR 2022）

- 入口：[ICLR 2022 官方页（摘要已读）](https://www.iclr.cc/virtual/2022/poster/6418) ·
  [OpenReview forum](https://openreview.net/forum?id=bERaNdoegnO) ·
  [作者 slides（ICLR 官网 PDF，已读）](https://www.iclr.cc/media/iclr-2022/Slides/6418.pdf) ·
  [mctx 官方实现 README（已读）](https://github.com/google-deepmind/mctx)（本机 DNS 过滤打不开 github 页面，
  我是通过 [raw README](https://raw.githubusercontent.com/google-deepmind/mctx/main/README.md) 读到的；
  README 明确写 "We recommend to use the `gumbel_muzero_policy`" 并给出
  [`examples/policy_improvement_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py) 这个可跑的改进性验证脚本）。
  ⚠️ **这篇论文不在 arXiv 上**（我用 arXiv 官方搜索页确认：搜标题只返回 1 条无关结果 GAZ‑PTP；见 §5），所以不要去找 arXiv ID。
- **一句话核心**：AlphaZero 在根节点**没访问完所有动作**时，用访问计数当策略目标是**不能保证策略改进**的；
  改成「按先验 logits 做 **Gumbel‑Top‑k 无放回采样**得到 k 个候选 → 用 **Sequential Halving** 在这 k 个里花掉 m 次仿真 →
  用 **completed Q**（未访问的动作填 `v_π(s)`）构造 `π' = softmax(logits + σ(completedQ))`」，
  就恢复了「**在期望上不劣于先验策略**」的改进性质，而且在 m 很小（作者 slides 第 7 页演示 **n=2 次仿真**）时仍然工作。
- **机制细节（讲义转述 + 作者 slides，原文 Algorithm 1/2）**：
  1. `g(a) ~ Gumbel(0,1)` 一次性采 K 维；`A_1..A_n = argtop_a (g(a) + logits(a))`（Gumbel‑Top‑k 无放回采样）。
  2. 在候选集内选：`A_{n+1} = argmax_{a∈{A_1..A_n}} ( g(a) + logits(a) + σ(q(a)) )`，σ 单调递增。
  3. 改进性证明的关键：**对固定的 g，全局 argmax 一定在候选集里**，而加单调 σ(q) 只会让选中的 q 更大 ⇒
     `q(A_{n+1}) ≥ q(全局 argmax)`，再对 g 取期望 + Gumbel‑Max 恒等式 ⇒ `E[q(A_{n+1})] ≥ E_{a~π}[q(a)]`。
  4. 根节点用 **Sequential Halving** 花仿真预算：仿真数取 `min(n, 16)`（n = 被考虑的动作数），
     并保证每个被考虑的动作至少被访问一次，预算用完即停（以上三点来自讲义对原文 Algorithm 2 的转述）。
  5. `completedQ(a) = q(a) if N(a)>0 else v_π`；`π' = softmax(logits + σ(completedQ))`；
     `σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂`，论文默认 `c_visit=50, c_scale=1.0`。
  6. 训练目标是 `KL(π_θ ‖ π')`（不是只用 one‑hot 的 `-log π_θ(A_{n+1})`）。
- **关键假设**：
  - **(A1) 被考虑的动作上有可用的 q 估计**——不要求"精确"，但要求它和真值同序；噪声会让"改进性"退化为"对 q̂ 的改进性"。
  - **(A2) 采样集与选择集用同一组 Gumbel 噪声**。这条最容易实现错：**先按 logits 取 top‑k，再对 k 个独立加噪声，是没有这个保证的**。
  - **(A3) 先验 π 本身"覆盖"了最优动作**（候选集由先验决定），否则改进性是相对"被截断的动作空间"成立的近似。
- **对我们是：需改造，可以替换"选择层"，不必上树**。
  我们本来就没有树（根节点枚举 + rollout），而 Gumbel 的改进性本来就是**根节点**性质，
  `π'` 也可以直接喂给蒸馏/自博弈当软标签。需要改造的三个点：
  ① `σ` 的强度要按噪声重新标定（我们 q̂ 的 σ≈111、真实分差 71，SNR<1，`c_visit=50` 这种"重度信任 Q"的设置会过拟合噪声）；
  ② 候选集必须由 `g + logits` 决定（否则丢保证）；
  ③ 我们有**层级解码**（类头 × 位置头 × 多座塔），要把"266 个 bundle"当成一个**扁平 categorical**，
     其 logits 用 `log P(bundle) = log P(class) + log P(pos|class) + …` 拼出来（近似即可，但要**自洽**）。

### 2.2 Sampled MuZero（Hubert et al., 2021, ICML）

- 链接：[arXiv:2104.06303（abs）](https://arxiv.org/abs/2104.06303) ·
  [HTML 全文（已读 §4 定理与实验）](https://arxiv.org/html/2104.06303v1)
- **一句话核心**：把 policy improvement / evaluation 都写成「对**完全改进策略** `Iπ` 的期望」，
  然后用从 proposal `β` 采的 k 个动作去**估计**这个期望：`Î_β π(a|s) = (β̂/β)(a|s) · f(s,a,Ẑ_β(s))`；
  搜索时把 PUCT 的先验换成 `π̂_β ∝ (β̂/β)·π` 以抵消采样偏差，最后照常用访问计数当（样本化的）改进策略。
- **理论保证与限制（我读到的原文表述）**：
  - 定理：样本化改进算子**依分布收敛**到真改进算子，且当 `K→∞` 时近似**正态**地围绕它 ⇒ 误差是 `O(1/√K)` 量级（不是精确等式）。
  - 保证的对象是**采样支撑集上的策略**，正确性依赖 `β̂/β` 这个重要性权重；`β` 必须有正概率覆盖到好动作。
  - 限制：**没有**有限样本、非平稳、噪声 q̂ 下的 regret/正确率界；它给的是"无偏/收敛"性质，不是"一定选得对"。
- **实验数字（对我们很有用）**：19×19 Go 动作空间 362，`K∈{15,25,50,100}`，**K=50 已接近全量搜索**；
  humanoid 连续控制里 **K=3 就够**。⇒ "候选集不必等于全空间，几十个量级就够"是可复现的经验结论，
  但前提是 proposal 与策略同源（他们用 `β=π`）。
- **对我们是：理论解释可用，算法主体不适用（不能直接用）**。
  我们没有可学习的隐模型/树，`β` 也不是归一化密度而是一段**解码过程**（先验网 → top‑k 束），
  重要性权重 `β̂/β` 算不出来。但它给了我们两个正确判断：
  ① 我们现在「266 截断 + 均匀打分」= **没有做重要性修正的样本化策略改进**，所以截断会引入偏差，
     这个偏差只能靠"把候选生成器训好"（先验/候选质量）来解决；
  ② 「k=48 vs k=160」不是玄学：样本化方法的误差随 K 增大而减小（K=50≈全量），
     **但如果把 K 增大导致每候选样本数下降，误差项反而变大**——两者竞争，存在最优点，这正是我们该扫的超参。

### 2.3 MCTS as Regularized Policy Optimization（Grill et al., NeurIPS 2020）

- 链接：[arXiv:2007.12509](https://arxiv.org/abs/2007.12509)
- **一句话核心**：AlphaZero 的 PUCT + 访问计数可以看成是「`argmax_y qᵀy − λ·KL(π‖y)`」的正则化策略优化，
  其解析解形如 `softmax((α − q)/λ)`；访问计数的目标在仿真数少时只是这个解的粗糙近似。
- **关键假设**：需要"策略改进 = 对 q 取 softmax 再投影"的视角；λ 随仿真数递减。
- **对我们是：能（作为"把 hard top1 换成带温度的 softmax"的理论依据）**。
  它解释了为什么"用访问计数/硬 argmax 当目标是低仿真下失效的"，并给出我们想要的形状：
  `π'(a) ∝ π_prior(a) · exp(q̂(a)/τ)`（或在 logits 上等价地 `softmax(logits + q̂/τ)`）。
  Gumbel 论文里的 `softmax(logits + σ(completedQ))` 就是它的 completed‑Q 版本。

### 2.4 Sequential Halving（Karnin, Koren, Somekh, ICML/COLT 2013）

- 链接：[PMLR 页面（摘要已读）](https://proceedings.mlr.press/v28/karnin13.html) ·
  [PDF（已下载并读了 Algorithm 2 与 Theorem 4.1）](http://proceedings.mlr.press/v28/karnin13.pdf)
- **一句话核心**：**固定总预算 T** 的 best-arm identification：把预算**均分到 log₂n 轮**，
  每轮对存活臂均匀各拉 `t_r = ⌊T/(|S_r|·⌈log₂n⌉)⌋` 次，砍掉经验均值差的一半；**参数无关**（无需知道 Δ）。
- **原文的界（Theorem 4.1，我读到的）**：正确识别最优臂的概率 ≥ `1 − 3·log₂n·exp(−T/(8·H₂·log₂n))`，
  其中 `H₂ = Σ_i 1/Δ_i²`（难度量）。等价地，达到 `1−δ` 需要 `T = O(H₂·log n·log(log n/δ))`。
- **关键假设**：① 臂之间**独立**、噪声**平稳**；② 有固定预算 T；③ **T 至少够每臂第一轮摸一次**
  （`t_0 = ⌊T/(n·⌈log₂n⌉)⌋ ≥ 1`，即 `T ≥ n·⌈log₂n⌉`，否则 t_0=0，算法退化）。
- **⚠️ 对我们的一个尖锐警告**：`H₂ = Σ 1/Δ_i²`。我们 266 个候选里大量是「只差一个格子/一个类」的**近重复**，
  它们的 Δ_i≈0 ⇒ `H₂` 爆炸 ⇒ **最坏情况保证等于没有**。所以：
  **不要指望"证明能选对 266 个里的最优"，要把目标改成「(a) 别选出明显差的 + (b) 把候选集先去重/聚类到几十个有区分度的束」**。
- **对我们是：能，且是最高优先级**。见 §3 草案 A。

### 2.5 Successive Halving / Hyperband / BOHB（多保真度）

- 链接：[Jamieson & Talwalkar, "Non-stochastic Best Arm Identification and Hyperparameter Optimization"（AISTATS 2016）arXiv:1502.07943](https://arxiv.org/abs/1502.07943) ·
  [Hyperband, arXiv:1603.06560](https://arxiv.org/abs/1603.06560) ·
  [BOHB, arXiv:1807.01774](https://arxiv.org/abs/1807.01774)
- **一句话核心**：SH 可以被推广成「**保真度（fidelity）阶梯**」——先用便宜的近似评估筛，再逐级加保真度，
  把同样的预算换成"更多候选 × 更粗评估 → 更少候选 × 更准评估"。Hyperband = 对 SH 的最优初始候选数做网格搜索；
  BOHB = Hyperband + 代理模型（把"先验"显式化）。
- **关键假设**：**低保真度评估与高保真度评估同序**（"successive halving 的正确性依赖 fidelity 单调性/相关性"）。这是最强的假设，也是最该先验证的。
- **对我们是：能（需先验证一条单调性假设）**。
  我们的保真度天然有三档：**rollout 回合数 h（4/8/16）**、**每候选采样数 n**、**对手模型（idle→随机→混合）**。
  需要先做的一个廉价实验：对同一批候选，比较 `h=4` 排序与 `h=16, n=64` 排序的 Spearman 相关。若 ρ>0.7，多保真度可行且省一半以上预算。

### 2.6 UCT / UCB / 纯探索（pure exploration）

- 链接：[Kocsis & Szepesvári, "Bandit based Monte-Carlo Planning"（UCT，已确认页面可达）](https://link.springer.com/chapter/10.1007/11871842_29) ·
  [Slivkins, "Introduction to Multi-Armed Bandits"（已确认标题）arXiv:1904.07272](https://arxiv.org/abs/1904.07272) ·
  [Bubeck & Cesa-Bianchi, "Regret Analysis of Stochastic and Nonstochastic MAB" arXiv:1204.5721](https://arxiv.org/abs/1204.5721)
- **一句话核心**：UCT/UCB 在树节点上做**累计后悔最小化**的探索；pure exploration（BAI）则只关心**最后选谁**（simple regret）。
- **关键假设**：UCT 的收敛保证针对**累计**目标；用在"只选一次"的根节点上会让"已经很好的臂反复被采"（浪费），
  而"暂时看起来差但没摸够"的臂被过早放弃。
- **对我们是：不能（作为根节点分配规则）；但"淘汰"思想可用**。
  Gumbel 论文明确说了这一点（讲义第 35 页转述：根节点目标是 `E[q(A_{n+1})]`，UCB 的最小化累计后悔不合适，
  作者实测后选了 SH，因为**参数无关、好调**）。
  可借用的是 **action elimination / early stopping**：
  [Even-Dar, Mannor, Mansour, JMLR 2006（页面已确认）](https://www.jmlr.org/papers/v7/evendar06a.html) ——
  用置信区间把"明显不可能赢"的候选提前剔掉，省下的样本给头部候选。

### 2.7 "大动作空间 + 先验筛候选"的同类工作

- 链接：[Wolpertinger, "Deep RL in Large Discrete Action Spaces" arXiv:1512.07679](https://arxiv.org/abs/1512.07679)
- **一句话核心**：用策略网络给出**连续原型**，再用最近邻把动作映射回离散动作集（先验 + 局部候选）。
- **关键假设**：动作空间有度量结构（我们：六边形网格 + 塔型，确实有），且候选映射足够快。
- **对我们是：能（作为架构类比，不是方法）**。我们的「位置先验网 → 266 束」与它同构；
  但**它没有策略改进保证**——这恰好说明：**候选生成（先验）和候选评估（搜索）必须各自有独立的质量度量**，
  否则"搜索变强"会被"候选集漏掉好动作"抵消（这正是我们"换先验就掉到 28–44%"的一种可能解释，见 §3 的评估方案）。

### 2.8 两个近期实证（用来校准"该不该做"）

- **MiniZero**（[arXiv:2310.11305](https://arxiv.org/abs/2310.11305)，摘要已读）：实现了 AZ / MuZero / Gumbel‑AZ / Gumbel‑MuZero 四者的公平对比。
  结论：棋类上**仿真越多越好**；Atari 上**MuZero 与 Gumbel‑MuZero 都值得一试**；
  并提出 training 期「progressive simulation」（训练时逐步提高仿真预算）能显著提升。→ 对我们的启示：**部署预算与训练预算不必相同**，
  自博弈采集可以用小预算，蒸馏/评估用大预算。
- **ReSCALE**（[arXiv:2603.21162](https://arxiv.org/abs/2603.21162)，摘要已读）：把 Gumbel‑AZ 的「Dirichlet 噪声 + PUCT」换成
  「Gumbel 采样 + Sequential Halving」，在 LLM 树搜索里让**精度随预算单调上升**（baseline 反而随预算下降），
  且消融显示 **Sequential Halving 是主要收益来源**。→ 直接支持 §3 草案 A 的核心判断。

---

## 3. 落地草案（两个改动，按性价比排序）

### 3.0 先算清楚我们到底有多少预算（关系到一切）

从代码读到的口径（`code/my_ai/az_intent/flat_rollout_greedy.py`）：
`DEFAULT_MAX_TOTAL_ROUNDS = 39_000`，`per_candidate_rounds = max(1, max_total_rounds // (n_cand * horizon))`，
`n_rollouts = min(rollouts_per_candidate, per_candidate_rounds)`，随后 `best = nanargmax(scores)`。

由此，**一次决策能做的 rollout 次数** `T = 39000/h`：

| 配置 | h | k（候选） | 每候选样本 n | T（评估次数） |
|---|---|---|---|---|
| 当前部署 | 16 | 160 | 15 | 2437 |
| 当前实验 | 16 | 48 | 50（cap 64） | 2437 |
| 现在这版 | 16 | 266 | 9 | 2437 |

而在 §1 第 2 条里我们已经算过：单次 σ≈111、真实分差≈71 时，**每候选 n 少于 ~20 次，top1 基本靠运气**。
所以问题不是"要不要上 Gumbel"，而是「**同一份 2437 次评估，怎么分配**」。

### 3.1 草案 A（主推）：把「均匀分配 + 硬 argmax」换成「Gumbel‑Top‑k + Sequential Halving + completed‑Q 软选择」

**改哪个模块**：`code/my_ai/az_intent/flat_rollout_greedy.py::FlatRolloutGreedy.search()`
（以及 `bundle_mcts.py::_topk_candidates` 的**候选保序**：候选列表要连同 logits 一起交出来，
不要让下游只能看到"已经截断的 k 个"）。**不动网络结构、不动 rollout 引擎**。

**参照实现**：[mctx `gumbel_muzero_policy`](https://github.com/google-deepmind/mctx)
（README 已读，原话：`policy_output.action_weights` contain targets usable to train the policy ——
即它输出的改进策略可直接当训练目标，正好对上我们"搜索 → 蒸馏"的链路）。

**伪代码（可直接对着写）**：

```python
# 每次决策调用一次。所有量都是"这一回合"的。
def gumbel_sh_decide(state, cand_logits, value_fn, rollout_fn,
                     T=2437, h=16, m_max=266,
                     c_visit=50.0, c_scale=1.0, kappa=1.0):
    """
    cand_logits: list[float]，266 个候选 bundle 的 log 先验（类头 + 位置头联合概率的 log，未归一化也行）
    T:           本次决策允许的 rollout 次数（= 39000 // h）
    rollout_fn:  (bundle, rng) -> float，跑 h 回合 + 叶值网，返回该候选的分数
    value_fn:    (state) -> float，值网对当前局面的评估 v(s)（= completed Q 的填充值）
    """
    K = len(cand_logits)

    # ① 共享的一组 Gumbel 噪声（关键：采样与选择用同一组 g）
    g = -log(-log(uniform(rng, size=K)))            # g ~ Gumbel(0,1)
    noisy = cand_logits + g

    # ② Gumbel-Top-m：这一步就是"按先验无放回采样"，不是"按先验 top-m 再独立加噪"
    #    m 由预算决定：保证每轮至少能摸 1~2 次
    m = min(K, m_max, int(max(2, T // max(1, ceil(log2(max(2, K)))))))  # 经验：让 t_0 >= 1~2
    S = argsort_desc(noisy)[:m]

    # ③ Sequential Halving：预算均分到 ceil(log2 m) 轮，每轮均匀摸一遍，砍半
    t = {a: 0 for a in S}          # 已采样次数（跨轮累积，不丢样本）
    sum_q = {a: 0.0 for a in S}
    n_rounds = ceil(log2(m))
    for r in range(n_rounds):
        t_r = max(1, T // (len(S) * n_rounds))       # ← Karnin Alg.2 的 t_r
        for a in S:                                   # 均匀：本轮每个存活候选都摸 t_r 次
            for _ in range(t_r):
                sum_q[a] += rollout_fn(a, rng)        # roll_fn 内部已含 h 回合 + 叶值网
                t[a] += 1
        S = top_half(S, key=lambda a: mean(sum_q[a], t[a]))   # 砍掉经验均值差的一半
        if len(S) == 1: break

    # ④ completed Q：被访问过的用经验均值，没访问过的填 v(s)（这就是"截断不再等于丢弃"）
    #    注意：这里的 S0 是第 ②步被考虑的那 m 个（不是全部 266 个）——
    #    Gumbel 的改进性只对"被考虑集"成立，集外候选相当于概率 0（这就是截断的近似代价）
    S0 = set(order[:m])
    v = value_fn(state)
    q_hat = {a: (mean(sum_q[a], t[a]) if t.get(a, 0) > 0 else v) for a in S0}
    max_N = max(t.values(), default=0)

    # ⑤ 改进策略：softmax(logits + sigma * q_hat_norm)，q_hat 先按本次候选间的 std 归一
    scale = std([q_hat[a] for a in S0]) + 1e-8
    sigma = (c_visit + max_N) * c_scale
    pi_impr = softmax({a: (cand_logits[a] + sigma * kappa * (q_hat[a] - v) / scale)
                       for a in S0})

    a_act = argmax(pi_impr)          # 部署：确定性 argmax；自博弈/采集：按 pi_impr 采样（温度另调）
    return a_act, pi_impr            # pi_impr 同时作为策略蒸馏 / 自博弈策略目标
```

**几个必须照抄的细节（做不到就等于没做）**：

1. **候选集由 `logits + g` 决定**（第 ②步）。如果保留"按 logits 取 top‑k"，那第 ① 步的 g 就只在下游起作用，
   `E[q(A)] ≥ E_{a~π}[q(a)]` 的证明链断了（证明用的正是"全局 argmax ∈ 候选集"）。
2. **未评估的候选填 `v(s)`**（completed Q），而不是填 0、填 -inf 或丢掉。它们仍有先验概率质量，
   这样"预算不够"不会等价于"这个动作不存在"——这也是我们 `scores=0 → 落到先验 argmax` 那种兜底的正确版本。
3. **σ 的强度要按我们的噪声重标**。论文用 `c_visit=50`（重度信任 Q）。我们 SNR<1（σ≈111 vs 分差 71），
   直接抄会让 `pi_impr` 变成"对噪声的 argmax"。建议扫 `kappa ∈ {0.1, 0.3, 1.0}` 与 `c_visit ∈ {0, 1, 5, 50}`，
   用 §3.3 的离线指标选，不要用对局胜率选（方差太大）。
4. **训练目标换成 `KL(π_θ ‖ pi_impr)`**（软标签），不要再用 top1 one‑hot。

**预期收益（保守，可验伪）**：

- **同一份 2437 次评估**，从 `266×(9 次)` 变成 `266×1 → 133×2 → 67×4 → 34×7 → 17×15 → 9×30 → 5×54 → 3×90 → 2×135`，
  冠军累计样本 ≈ **338 次**，其标准误 `SE ≈ 111/√338 ≈ 6`，与真实分差 71 相比 `z ≈ 12`。
  也就是说：**"冠军一定不是明显差的"这件事第一次变得可判定**。
- 直接可测的代理指标（见 §3.3）：**选中的候选在"参考评估"（n=128）下落入前 5% 的比例**。
  现在这个数在 k=160/n=15 下应该很接近随机（≈20% 才对），改完应该显著上升。
- 间接收益：`pi_impr` 让"换先验就掉到 28–44%"的**部分**原因（硬 argmax 放大了先验噪声）变成"软加权"，鲁棒性更好。

**风险 / 代价（写清楚，别指望它包治百病）**：

- **偏差不解决**：SH 只降方差。rollout 里对手 idle 的系统偏差，采样再多也还是那个偏差。
  ⇒ SH 之后大概率会出现"内部指标变好但对真实对手不变"的情况；必须同时做 §3.2 或换更强对手模型。
- **`H₂` 爆炸**：266 个近重复候选 ⇒ 理论界失效。**必须配一个候选去重/聚类**（例如按"位置 ±1 格 / 同塔型"合并，
  或按先验概率做 nucleus 截断），把有效 `Δ_i` 拉开。这是草案 A 的**前置条件**，不是可选项。
- **10 秒硬时限**：SH 天然 anytime（每轮结束都有当前最优），但要加"剩余时间 < 一轮成本就停"的守卫；
  另外 `t_r` 是**预先算好的**，实际执行会因"候选非法/引擎早停"而偏差，需要按实际完成数动态重算。
- **实现复杂度**：比"平均分预算"高一个档次；还有**采样复用/确定性**问题（自博弈 replay 需要固定种子策略）。
  建议：先在**离线数据集**上做（§3.3），再进对局。

### 3.2 草案 B（次推，低成本高杠杆）：共享随机数 + 多保真度阶梯

**B1. 共享随机数（CRN, common random numbers）——一行级别的改动**
现在 `_tree_rollout_value(child, player, rngs)` 里每个候选用自己的 `rng`（甚至每次 rollout 重新抽）。
如果让**所有候选共用同一批 rollout 种子**（同样的随机出招序列/同样的对手动作），
那么候选间的**差值**方差会显著下降（配对比较），而均值不变。
这是仿真实验里的标准方差缩减技巧，收益在"候选间差异"这个量上，正好是我们唯一关心的量。
**注意**：这不是论文结论，是工程实践（若写进实验日志请标"经验性改动"）。

**B2. 多保真度阶梯（先粗筛再精评）**
把 `h` 当成保真度：`h=4 → 8 → 16`，每档之间做一次 halving。参考
[Hyperband](https://arxiv.org/abs/1603.06560) / [Jamieson & Talwalkar](https://arxiv.org/abs/1502.07943) / [BOHB](https://arxiv.org/abs/1807.01774)。
**前置验证（半天就能做完，且不占对局）**：取同一批候选，比较 `h=4` 与 `h=16 × 64 次` 的排序相关性。
- ρ(Spearman) > 0.7 ⇒ 可以直接上三级阶梯，**同预算下再把冠军样本数翻倍**；
- ρ < 0.5 ⇒ h=4 的评估基本是另一个量，多保真度不成立，放弃 B2，只做 A + B1。

### 3.3 配套：把"指标与强度关系不稳"这个问题本身修掉

现在的问题之一是**用对局胜率当调参指标**（噪声大、成本高、且随对手池漂移）。建议加一个**零对局的离线指标**：

> 对 `M` 个真实决策状态（可从 `data/seedstate_dump` 或自博弈日志抽）：
> 1. 用"参考评估"给每个候选打分：`n=128` 次 rollout，取均值（当"真值"，成本一次性）；
> 2. 对比各配置选出的 top1：计算
>    - **Top‑1 命中率**：选中的候选是否真的在参考评估里排第一（这个数现在多半很低，别慌）；
>    - **Regret**：`参考评估(最优) − 参考评估(选中)`，报告均值与 90 分位；
>    - **Rank 分位**：选中者在 266 个里的名次分位（最实用，抗"参考评估也不完美"）。
> 3. 只在这个指标上扫 `(k, m, n, SH 与否, kappa, c_visit)`，**选出的配置再上对局验证**。

这个指标的好处：**它直接测量"选择质量"**，而不是"胜率"；它能让你在 1 小时内判断草案 A 是否成立，
而不是跑 200 局对局再猜。文献里这属于"offline evaluation of a selection rule"，没有专门论文，
但 BAI 的"simple regret"就是它的理论对应物（见 [Slivkins 综述](https://arxiv.org/abs/1904.07272)）。

---

## 4. 我看不到的 / 不确定的（不要当成已核实）

1. **Gumbel 论文正文**：OpenReview 的 `https://openreview.net/pdf?id=bERaNdoegnO` 在我这里是 **HTTP 403**
   （`https://openreview.net/forum?id=bERaNdoegnO` 被浏览器挑战页拦住，Playwright 也过不去）。
   ⇒ 我读到的是：① ICLR 官方**摘要**；② 作者**官方 slides**（ICLR 官网 PDF，8 页，已读）；③ Charles Univ 的
   [NPFL139 Lecture 12 讲义](https://ufal.mff.cuni.cz/~straka/courses/npfl139/2324/slides.pdf/npfl139-2324-12.pdf)（已读，含 Algorithm 1/2 的公式与 σ/c_visit 默认值）。
   **§2.1 里的公式级细节（σ 形式、c_visit=50、m=min(n,16)、非根节点选择公式）来自这份讲义对原文的转述，我没在原始 PDF 上逐字核对**。
   另外：**这篇论文没有 arXiv 版本**（arXiv 官方搜索结果只有 1 条无关结果，已附图证据在工具调用里）。
2. **Sequential Halving 的定理我读的是原文 PDF**（Algorithm 2 + Theorem 4.1 已确认）。但我**没有**核对
   Karnin 论文里 "固定置信度"那半边的 Exponential‑Gap Elimination（Algorithm 1），本文不依赖它。
3. **Audibert & Bubeck (COLT 2010) "Best Arm Identification in Multi-Armed Bandits"（UCB‑E / Successive Rejects）**：
   我试了 `https://proceedings.mlr.press/v23/audibert12a.html` → **404**，没找到稳定可验证的公开链接。
   本文对 UCB‑E 的提及是通过 Karnin 论文的引用（原文 p2 提到 UCB‑E 与 Successive Rejects）间接得到，**不作为独立引用**。
4. **"评价贵 + 动作多"是否真有比 SH 更好的专门方法**：在我检索到的范围里，主流就是 SH / Hyperband / 多保真度 / 阈值淘汰
   （`Even-Dar` 的 action elimination 是同类思想）。**没有找到**声称"针对单次评估极贵且候选极多"能显著优于 SH 的量身方法。
   **parent 提到的 `TEDS` 我没有核实到**——搜不到明确对应文献，**不敢引用**（若你指的是某篇具体论文，给我 ID 我再补）。
5. **Racing（Hoeffding races, Maron & Moore 1993）**：概念上等价于"用置信区间边跑边淘汰"，与 Even‑Dar 的 elimination 同类。
   我**没找到**一个我敢引用的稳定公开链接，故未列条目。
6. **Progressive widening（Couëtoux et al. 2011 等）**：适用于"连续/超大动作空间的树"，与我们"根节点扁平枚举"不匹配，且我没验证到稳定链接，**未纳入**。
7. 你们的 `39000` 预算是从代码常量 `DEFAULT_MAX_TOTAL_ROUNDS` 与 `per_candidate_rounds` 的除法关系反推的
   （与题干给的 `N = 39000/(k×16)` 完全吻合）。若部署时另有硬件时限覆盖它，**§3.1 的 `T=2437` 要按实际值重算**。
8. 我们自己的 `σ=111 / 分差=71` 取自**代码注释里的既有实测**（`code/my_ai/distill_top2/fit_q8_baseline.py:175`），
   我没有重跑实验验证。**本文所有"选择基本靠噪声"的结论都依赖这两个数**，建议先花 10 分钟重测一次确认。

---

## 5. 阅读台账（读了什么、读到哪一层）

| # | 文献 | 入口链接 | 我读到的层级 | 状态 |
|---|---|---|---|---|
| 1 | Policy improvement by planning with Gumbel (ICLR 2022) | [ICLR 页](https://www.iclr.cc/virtual/2022/poster/6418) | 官方摘要 | **已读** |
| 1b | 同上，作者 slides | [slides PDF](https://www.iclr.cc/media/iclr-2022/Slides/6418.pdf) | 8 页全文 | **已读**（下载 + 抽文本） |
| 1c | 同上，OpenReview PDF | [forum](https://openreview.net/forum?id=bERaNdoegnO) | — | **403 / 挑战页，读不到** |
| 1d | 同上，公式级转述 | [NPFL139 L12 讲义](https://ufal.mff.cuni.cz/~straka/courses/npfl139/2324/slides.pdf/npfl139-2324-12.pdf) | 第 30–42 页全文 | **已读**（二手，已标注） |
| 1e | 官方实现 | [mctx](https://github.com/google-deepmind/mctx) / [README(raw)](https://raw.githubusercontent.com/google-deepmind/mctx/main/README.md) | README 全文 | **已读**（github 页面被本机 DNS 过滤，未直接打开） |
| 2 | Learning and Planning in Complex Action Spaces (Sampled MuZero) | [arXiv:2104.06303](https://arxiv.org/abs/2104.06303) · [HTML](https://arxiv.org/html/2104.06303v1) | 摘要 + §4 定理 + Go/humanoid 实验数字 | **已读** |
| 3 | MCTS as Regularized Policy Optimization | [arXiv:2007.12509](https://arxiv.org/abs/2007.12509) | 标题与摘要层（经讲义第 16–25 页的系统转述） | **部分已读**（未逐字读原文） |
| 4 | Almost Optimal Exploration in MAB (Sequential Halving) | [PMLR](https://proceedings.mlr.press/v28/karnin13.html) · [PDF](http://proceedings.mlr.press/v28/karnin13.pdf) | 摘要 + §4 + Algorithm 2 + Theorem 4.1 | **已读原文 PDF** |
| 5 | Non-stochastic BAI and Hyperparameter Optimization | [arXiv:1502.07943](https://arxiv.org/abs/1502.07943) | 标题层 | **未读正文**（仅标题/定位） |
| 6 | Hyperband | [arXiv:1603.06560](https://arxiv.org/abs/1603.06560) | 摘要层（方法熟知） | **未逐字读** |
| 7 | BOHB | [arXiv:1807.01774](https://arxiv.org/abs/1807.01774) | 标题层 | **未读正文** |
| 8 | Action Elimination and Stopping Conditions (Even-Dar et al., JMLR 2006) | [JMLR](https://www.jmlr.org/papers/v7/evendar06a.html) | 标题 + 页面元数据 | **部分已读** |
| 9 | Introduction to Multi-Armed Bandits (Slivkins) | [arXiv:1904.07272](https://arxiv.org/abs/1904.07272) | 标题层（作为 BAI/regret 的教科书入口） | **未读正文** |
| 10 | Regret Analysis of Stochastic and Nonstochastic MAB | [arXiv:1204.5721](https://arxiv.org/abs/1204.5721) | 标题层 | **未读正文** |
| 11 | Bandit based Monte-Carlo Planning (UCT) | [Springer](https://link.springer.com/chapter/10.1007/11871842_29) | 仅确认链接可达 | **未读正文**（内容为通用知识） |
| 12 | AlphaZero | [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) | 摘要层 | **未读正文** |
| 13 | MuZero | [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | 摘要层 | **未读正文** |
| 14 | MiniZero（AZ/MuZero/Gumbel 公平对比 + progressive simulation） | [arXiv:2310.11305](https://arxiv.org/abs/2310.11305) | 摘要（含结论句） | **已读摘要** |
| 15 | ReSCALE（Gumbel + SH 让预算可伸缩） | [arXiv:2603.21162](https://arxiv.org/abs/2603.21162) | 摘要（含消融结论） | **已读摘要** |
| 16 | Deep RL in Large Discrete Action Spaces (Wolpertinger) | [arXiv:1512.07679](https://arxiv.org/abs/1512.07679) | 标题层 | **未读正文** |

**工具限制说明**：本环境里 `web_fetch` 对部分域名会报 "resolves to a non-public IP address"
（`github.com`、`proceedings.mlr.press`、`export.arxiv.org`、`discovery.ucl.ac.uk` 都遇到过），
但用 `pwsh + Invoke-WebRequest` 直连可以下载（PMLR PDF、ICLR slides、讲义 PDF 都是这样拿到的），
PDF 文本用 `PyMuPDF(fitz)` 抽。**OpenReview 是唯一真正打不开的（403 / JS 挑战）**。
