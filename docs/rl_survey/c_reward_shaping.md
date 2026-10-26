# 奖励塑形与长视野稀疏奖励（调研 C）

> 2026-10-26 写。面向 Ant-Game（六边网格塔防对抗，双方 50HP，512 回合上限，单局约 1000 个决策）的
> **奖励设计 / 信用分配** 文献调研。**不写产品代码，只出文档。**
>
> 结构：§1 结论摘要 → §2 逐方法 → §3 落地草案 → §4 已读/未读清单 → §5/§6 是追加的两问（可单独看）。
> 只看结论：§1 的第 1–4 条 + §5.3 + §6.D 的表。

## 0. 调研方法与"读到哪一步"

- 本机 `web_search` / `web_fetch` 均不可用（搜索端点 fetch failed；`web_fetch` 把所有域名解析成非公网 IP）。
- 实际路径：`curl` / `Invoke-WebRequest` 直连下载 PDF → **PyMuPDF (`fitz`)** 抽文本。
- **Ng, Harada & Russell 1999 的 PDF 字体是 Type1 子集、无 ToUnicode**，`pypdf` / `fitz` / 自写 zlib
  抽取全部失败（出来是乱码）。最终用 **PyMuPDF 把指定页渲染成 190dpi PNG，再逐页读图**，
  所以 §2.1 里的**定理、推论、注释都是从论文原页读到的原文**（只做了排版归一化，词句未改）。
- 所有引用 URL 都做过 HTTP 状态校验；读不到的东西全部列在 §4 并标明"仍未读到"。

---

## 1. 结论摘要

**一句话前提（全篇最重要）**：potential-based shaping 只能把**你已经知道的信号换个时间位置放**，
**不能注入你不知道的信号**。这不是经验之谈，是原论文 Remark 2 的直接结论：
若奖励本身就是 `R = γΦ(s')−Φ(s)` 的形式，那么**任何策略都是最优的**——塑形函数本身对策略是"无差别"的。
所以"金币没有定价"这个病，**用 shaping 治不好**，能治的只有值函数/标签本身。

1. **不要做"金币奖励"，无论系数多少。** 回合制里"按金币给中间奖励"形式上就是取 `Φ = 金币` 的 potential。
   按 Ng 1999 定理，合法的写法是 `F = γΦ(s') − Φ(s)` **且 `Φ(终局)=0`**；此时整局望远镜求和剩下
   `−Φ(初始)`（常数），**你攒的每一分钱都会在终局被没收**。反过来，如果终局不归零，
   整局就多出 `ρ·(终局金币−初始金币)`，等于把目标改写成 `margin + ρ·终局金币`：
   `ρ>0` 必然攒钱、`ρ<0` 必然花光、**`ρ=0` 才中性**。
   ⇒ 你们看到的"调高→AI 攒金币不做事、调低→完全没效果"**不是调参问题是形状问题**；
   而且它和 Ng 1999 引言里的两个经典 bug 是**同一个 bug**（§2.1）。
2. **要做 shaping 就只做一种：`Φ = V`（值网），且只用在搜索的评估函数上**，不要用在奖励里。
   理由：(a) 在带 critic 的 actor-critic 里，`Φ=V` 的塑形对优势函数**恒等**（`A'=A`，§2.1 有推导）
   ⇒"拿它训练奖励来教金币"理论上什么都不会发生；(b) 它真正有效的地方是**叶值不可靠/被截断的搜索评估**
   （= 你们 prereg `docs/prereg_20261008_mc_value_delta.md` 的 `delta` 臂）。
3. **"1000 步才见终局"的正解是缩短信用链，不是设计中间奖励。** 算术：`γ=0.99` 时
   `0.99^1000 ≈ 4.3e-5`，终局奖励折扣后**等于不存在**（有效视野 `1/(1−γ)=100` 步）。
   要么 `γ ≥ 0.999`（`0.999^1000 ≈ 0.37`）配 n-step 自举，要么 `γ=1` + **在截断/终局处 bootstrap `V`**。
   MuZero 的 value target 就是 `z_t = Σ_{i=1..n} γ^{i−1} u_{t+i} + γ^n ν_{t+n}`（§6.B 有原文）；
   棋类里 `n` 直接到终局、Atari 用 `n=10`。
4. **你们其实没有面临"稀疏奖励"**：蒸馏时**每一帧都有标签**（未来 HP 差），一局 1000 个决策就是 1000 个标签；
   稀疏只是"model-free RL 视角"下才成立的问题。所以别为了"稀疏"去加 shaping。
   你们的真问题是**标签噪声 / 代理指标与棋力脱钩**（`docs/test_match/_fill_prior_summary.py` 反复记录：
   Spearman 与棋力三次脱钩、值网 R²≈0.47）——这属于"评估器质量"问题，不是"奖励密度"问题。
5. **如果真要走 model-free RL**：优先级 ① AZ/MuZero 式（`z` + 搜索里的 n-step backup + 分布化 value head）
   ② 课程/对手池（便宜、不可被刷）③ RUDDER（可行，但代价在"回报预测器"上）④ 中间奖励（**最后**，
   且必须证明它与真值单调）。reward machines 不适用，HER 收益小。

> 对原问题 (a)(b) 的直接回答：
> **(a)** 想不被刷分，shaping 只能写成 `F=γΦ(s')−Φ(s)`、`Φ` 只用状态、`Φ(终局)=0`。
> 满足这三条就"刷不动"——代价是它也**改不动最优策略**（安全与无用是同一件事的两面，原文 Remark 2）。
> **(b)** 长视野稀疏奖励该用的是：n-step/GAE 自举（缩短信用链）+ 搜索做时序归因（AZ/MuZero）+ 值网把
> 终局信号摊到每步（你们已经在做），**而不是另外设计一份奖励**。

---

## 2. 逐方法

### 2.0 一张总表

| # | 方法 | 能不能改最优策略 | 能不能被刷 | 对我们 | 落地位置 |
|---|---|---|---|---|---|
| 2.1 | Potential-based shaping（PBRS） | 不能（`Φ(s0)=0` 时） | 不能 | **能，但只有一种用法** | 搜索评估函数（§3.1） |
| 2.2 | RUDDER / Align-RUDDER | 不能（return-equivalent） | 不能 | 中等（贵在回报预测器） | 你们已有轻量版（`rel` 标签） |
| 2.3 | Reward machines | 能（它就是在定义任务） | 能（写错就会被刷） | **不适用** | — |
| 2.4 | HER | 不改目标、只改采样 | 不适用 | 收益小 | 可借"目标条件化"做课程 |
| 2.5 | 课程 / 自对弈 / 联盟 | 能（改数据分布） | 不适用 | **最实用** | 对手池、前缀分层、horizon |
| 2.6 | n-step / GAE / eligibility trace | 不能（同一目标的估计器） | 不能 | **最该优先** | 值标签与搜索 backup |
| 2.7 | reward hacking 形式化 | — | 定义工具 | 用来写验收判据 | §3.5 |

### 2.1 Potential-based reward shaping（Ng, Harada & Russell 1999）

- 论文：*Policy invariance under reward transformations: Theory and application to reward shaping*, ICML 1999
  - PDF（可点，200）：<https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf>
  - 镜像（未逐字比对）：<https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf>

#### 2.1.1 定理原文（我读的是第 4 页，逐字）

> **Theorem 1.** *Let any S, A, γ, and any shaping reward function `F : S × A × S → ℝ` be given.
> We say F is a **potential-based** shaping function if there exists a real-valued function `Φ : S → ℝ`
> such that for all `s ∈ S − {s0}`, `a ∈ A`, `s' ∈ S`,*
>
> `F(s, a, s') = γΦ(s') − Φ(s)`,  **(2)**
>
> *(where `S − {s0} = S` if `γ < 1`). Then, that F is a potential-based shaping function is a
> **necessary and sufficient** condition for it to guarantee consistency with the optimal policy
> (when learning from `M' = (S,A,T,γ,R+F)` rather than from `M = (S,A,T,γ,R)`), in the following sense:*
> - *(Sufficiency) If F is a potential-based shaping function, then **every** optimal policy in M' will also
>   be an optimal policy in M (and vice versa).*
> - *(Necessity) If F is not a potential-based shaping function …, then there exist (proper) transition
>   functions T and a reward function R …, such that **no** optimal policy in M' is optimal in M.*

注意三处容易被二手转述丢掉的关键细节：

1. **定理是"充要"的**：在"对 `T,R` 无先验知识"的意义下，potential 形式是**唯一**能保证策略一致性的形式。
   原文：*"without more prior knowledge about T and R, such potential-based shaping functions F are the only F
   that will guarantee consistency with the optimal policy in M."*
2. **`s ∈ S − {s0}`**：`s0` 是**吸收态**。也就是说，**合法写法里终局的那一步必须用 `Φ(s0)` 来结算，
   而 `Φ(s0)` 必须是 0**（见下面的 Corollary 2）。这不是我加的实现约定，**它就在定理里**。
3. `γ=1` 时需要额外的**正则性条件**（原文 §2.1）：S 里有吸收态 `s0`，"MDP 进入 `s0` 后停止、此后不再有奖励"，
   且所有策略是 **proper** 的（从任何状态出发以概率 1 最终进入 `s0`）。
   ⇒ 我们的 512 回合上限正好提供了一个"终止"，但要注意**被截断的这局才是被保证的那个 MDP**
   （截断游戏的规则就是真实规则，所以没问题）。

**Corollary 2（原文，第 5 页）**：

> *Under the conditions of Theorem 1, suppose that F does indeed take the form `F(s,a,s') = γΦ(s') − Φ(s)`.
> Suppose further that `Φ(s0) = 0` if `γ = 1`. Then for all `s ∈ S, a ∈ A,`*
> `Q*_{M'}(s,a) = Q*_M(s,a) − Φ(s)`，`V*_{M'}(s) = V*_M(s) − Φ(s)`.

**Remark 1（robustness）**：原文说这两条恒等式**对任意策略**也成立（`V^π_{M'} = V^π_M − Φ`），
于是"近似最优也被保留"：若在 `M'` 里学到 `|V^π_{M'} − V*_{M'}| < ε`，则在 `M` 里同样 `< ε`。
**这正是"刷不动"的严格版本**：你不能靠 shaping 拿到一个"在真目标上差很多"的策略。

**Remark 2（对策略无差别）**：*"if we have an MDP M that has a potential-based reinforcement function
`R(s,a,s') = γΦ(s') − Φ(s)`, then any policy is optimal in M. Thus, potential-based shaping functions are
indifferent to policies…"*
**这就是"安全 = 无用"的原文出处**：塑形项本身不表达任何偏好。

#### 2.1.2 它为什么能防刷：环路论证（原文引言 / §3）

原文举了两个真实 bug（这**正是**你们"攒金币"的同款）：

- Randløv & Alstrøm 1998 的**骑自行车**任务：为了加速学习，对"朝目标前进"给正奖励，结果
  *"The agent learned to ride in tiny circles near the start state because no penalty was incurred for riding
  away from the goal."*（**在起点附近绕小圈**）
- 一个足球机器人：为"碰到球"给奖励，结果 *"it remained next to the ball and 'vibrated,' touching the ball
  as frequently as possible."*（**贴着球抖**）

原文的诊断：

> *"if there is some sequence of states `s1, s2, …, sn` such that the agent can travel through them in a cycle
> … and gain net positive shaping-reward by doing so … then it seems that the agent may be 'distracted' from
> whatever it really should be trying to do … and instead try to repeatedly go round this cycle."*
> *"To address this difficulty with cycles, a form for F that immediately comes to mind is to let F be a
> **difference of potentials** … we have eliminated the problem of cycles that 'distract' the agent."*

**"攒金币不做事"就是这条环路**：收入→持有→收入，每一圈都拿到正 shaping 奖励，而这条环路跟胜负无关。
`F = γΦ(s')−Φ(s)` 的望远镜性质让**任何环路的总收益恒为 0**，所以它被结构性地堵死了。

#### 2.1.3 我们能怎么用（四种 potential 的合法性判定）

| 候选 `Φ` | 是否合法 | 终局 `Φ(s0)` | 后果 |
|---|---|---|---|
| `Φ = 金币` | 形式合法 | 若强制 0 | 中性（每分钱终局被没收）；但每一步"花钱"=−ρ、"攒钱"=+ρ，**梯度方向是反的** |
| `Φ = 金币` | 形式合法 | 不归零 | **破坏 invariant** ⇒ 目标变成 `margin+ρ·gold_T` ⇒ **必被刷（攒钱）** |
| `Φ = HP 差/20` | 形式合法 | `= margin_T`（≠0） | 回报被**单调放大 2 倍**，排序不变（安全）；但零新信息（§5） |
| `Φ = V`（值网） | 形式合法 | 建议强制 `=Φ(s_0)`/裁剪 | 唯一能表达"金币/塔位值多少"的载体 |

最后一行值得强调：**"给金币定价"这件事只能在 `V` 内部完成**，因为 `V` 是唯一一个
"输入里含金币、输出是未来 HP 差"的函数；`Φ` 的其他选择都只是在**陈述**一个价格，而不是**估计**它。

#### 2.1.4 两个实现陷阱

- **陷阱 1（双计）**：若 `Φ = V_θ` 又同时当 bootstrap 值，TD 目标变成
  `r + F + γV(s') = r + 2γV(s') − V(s)` ⇒ 值函数系统性有偏。要么冻结 `Φ`（旧网/目标网），要么别放进 TD。
- **陷阱 2（AC 里恒等）**：若把 baseline 也平移 `V' = V − Φ`，则
  `A' = r + F + γV'(s') − V'(s) = r + γV(s') − V(s) = A`
  ⇒ **优势逐点不变，策略梯度逐点不变**。这就是"PPO 里加 shaping 常常毫无效果"的解释，
  也说明你们 `docs/ppo_experiment_plan.md` 里"只用血量变化、避免人工 shaping"的直觉是对的。

#### 2.1.5 与 Wiewiora 2003 的关系

Wiewiora 证明 *potential-based shaping 与 Q 值初始化等价*（JAIR 19:205–208，
<https://arxiv.org/abs/1106.5267>、<https://www.jair.org/index.php/jair/article/view/10338>）：
shaping 等价于"把值函数从一个更好的起点开始"。**它省的是收敛步数，不是解。**
（推论：如果 `Φ` 每轮都在变，等价于每轮**重置**你的值函数初始化 ⇒ 可能反而更慢。）

### 2.2 RUDDER（延迟奖励的回报分解）

- 论文：**Arjona-Medina, Gillhofer, Widrich, Unterthiner, Brandstetter, Hochreiter**, NeurIPS 2019
  <https://arxiv.org/abs/1806.07857>
- **作者更正**：任务里写的 "Patel et al. 2019" 不对，RUDDER 是 **Arjona-Medina 等（JKU Linz）**；
  "Patil/Patel" 那批人是后续的 **Align-RUDDER**（<https://arxiv.org/abs/2009.14108>）。

**目标（原文摘要）**：让"期望未来回报 = 0"，于是 Q 值估计退化成"即时奖励的均值"。两个概念：
(i) **reward redistribution** → **return-equivalent** 决策过程（*characterized by having the same optimal policies*）；
(ii) **return decomposition via contribution analysis**。

**关键理论（我读到了正文式 (1) 与 Theorem 3）**：

```text
最优 reward redistribution 的充要刻画（式 1）：
（I）κ(T−t−1, t) = 0   （即 redistribution 最优）
（II）E[R_{t+1} | s_{t−1},a_{t−1},s_t,a_t] = q̃π(s_t,a_t) − q̃π(s_{t−1},a_{t−1})
     —— 期望的新奖励 = 相邻 Q 值之差

Theorem 3：若 redistribution 最优，则 SDP 的 Q 值为
qπ(s_t,a_t) = r(s_t,a_t) = q̃π(s_t,a_t) − ψπ(s_t)
且 **SDP 与原 MDP 有相同的 advantage function**。
```

**贡献分析怎么算（原文 §"Contribution analysis"）**：让 `g` 在**每一步**都预测"整条序列的回报"，
则**重分配后的奖励 = 相邻两次预测之差**。原文的关键性质：

> *"Consecutive predictions share the same past and the same future contributions except for two immediate
> state-action pairs. Thus, in the difference of consecutive predictions contributions cancel except for the
> two immediate state-action pairs. **Even for imprecise predictions** of future contributions to the return,
> contribution analysis is more precise, since **prediction errors cancel out**."*

另两种实现（integrated gradients / LRP）依赖序列**后面**的信息 ⇒ 会引入非马尔可夫奖励、方差更大；
RUDDER 明确**偏好"相邻预测之差"**这一种。还有 "explaining away problem"（奖励引起的动作不一定拿到奖励）。

**"1000 步太长"这个担心成立吗？——不成立。** 原文实验就是**按延迟长度 `n` 同时放大网格和回合长度**
（"For a delay of n, a (3n/4)×(3n/4) grid is used with an episode length of n"），
所以 RUDDER 本来就是**冲长延迟设计的**，且"误差相消"意味着预测器不必很准。

**对我们可行吗：中等**。真正的代价在两处：

1. **回报预测器要覆盖 1000 步**（双向 LSTM），而监督信号是**每局一个标量**
   ⇒ 独立样本数 = 局数，不是步数——你们自己的文档里已经写过同一句话
   （`code/my_ai/entity_transformer/subsample_frames.py` 开头）。
2. 我们的终局信号是**连续** HP 差、且**已经被 `rel` 标签摊销过**（见下），RUDDER 的边际收益被削弱。

**但你们已经有一个"轻量 RUDDER"**：`build_rel_labels.py` / `az_train.py::add_weighted_labels` 的 `rel` 模式
= "指数加权未来血量差 − 当前血量差"，`tau ∈ {5,10,50}` 就是"摊多宽"的旋钮。
差别只在于：它是**固定的几何权重**，而 RUDDER 的权重是**学出来的**（能做到"期望未来回报=0"）。
要不要升级成学出来的版本，取决于你们愿不愿意为"每局一个标量"的监督再付一个 LSTM 的成本。

### 2.3 Reward machines

- Toro Icarte, Klassen, Valenzano, McIlraith, *Reward Machines*, JMLR 2022 <https://arxiv.org/abs/2010.03950>
- 核心：把奖励写成一个**有限状态自动机**（"到 A 再到 B 才给奖励"），让智能体看到奖励函数的**结构**，
  从而做自动塑形 / 任务分解 / 反事实推理。表达力 = 正则语言。
- **关键假设**：奖励天然可分解成**子任务/时序条件**。我们的奖励只有一个终局标量，
  唯一可能的划分是"开局/中期/残局"——那是**状态特征**，塞进网络输入即可，用不着自动机。
- **结论：不适用。** 别为了用它去发明子任务，那会引入新的可刷分面（而且 reward machine 本身
  *就是*在定义目标，写错必然被刷）。

### 2.4 Hindsight Experience Replay（HER）

- Andrychowicz et al., NeurIPS 2017 <https://arxiv.org/abs/1707.01495>
- 核心：奖励**稀疏二值**且有 goal 时，把失败轨迹的 goal **重标**成实际达成的 goal ⇒ 失败样本变成功样本。
- **关键假设**：① 有 goal 且可重标；② 奖励稀疏二值。
- 我们的终局奖励是**连续 HP 差**，本身就"稠密"，HER 的主要收益（0→1）不存在。
- **结论：收益小。** 可借的想法是**目标条件化**（把"要求净胜 m 血"当条件输入，做 1 血→5 血→15 血的课程），
  但那属于 §2.5 的课程，不需要 HER 的重标机制。

### 2.5 课程 / 自对弈 / 联盟（最便宜、不可被刷）

- 综述：Narvekar et al., JMLR 2020 <https://arxiv.org/abs/2003.04960>
- AlphaZero（自对弈 + MCTS）<https://arxiv.org/abs/1712.01815>
- 联盟 / 优先虚构自对弈（PFSP）：AlphaStar, Nature 2019 <https://doi.org/10.1038/s41586-019-1724-z>
  （**正文付费墙，我只核到 DOI 可解析**）；一般化的"按对手优先采样"框架见 PSRO <https://arxiv.org/abs/1711.00832>
- **OpenAI Five 的奖励退火**（最值得抄的工程经验）<https://arxiv.org/abs/1912.06680>：
  其超参表里 `Team Spirit 0.3 → 0.8`、`GAE Horizon 180s → 360s` 都是"训练中**平滑单调**过渡"，
  表注原文：`x → y` 表示 "a smooth monotonic transition (usually a linear change over one to three days)"。
  ⇒ **辅助信号可以先大后小，最后让主目标主导。**
- 为什么这条对我们最实用：课程/对手池改的是**数据分布**，不是**奖励函数**，所以不存在"被刷"的问题；
  而且你们已有对手池/固定段前缀的基建。
- 三条可落地：① 对手池按强弱分层采样（PFSP）；② **前缀/局面分层**——把"1–3 血差距"的平局与"差距明显"的局
  **分开报**（前者信噪比低，你们已知 32 局分辨率 ±17pp / ±5.5hp）；③ horizon 课程
  （先 `MC_HORIZON=128` 校准评估器，再推到 256/512）。

### 2.6 n-step / GAE / eligibility trace（真正的"长视野"工具）

- 书（官方免费）：Sutton & Barto, 2nd ed.，第 12 章 <http://incompleteideas.net/book/the-book-2nd.html>
- GAE：Schulman et al., ICLR 2016 <https://arxiv.org/abs/1506.02438>
- `δ_t = r_t + γV(s_{t+1}) − V(s_t)`；`A_t = Σ (γλ)^k δ_{t+k}`。
  `λ=0` → 纯 TD（低方差、高偏差、信用链 1 步）；`λ=1` → 纯 MC（无偏、高方差、等终局）。
  **`n`/`λ` 就是在"等多久"上做插值**，用的是**同一个终局信号**，所以不存在刷分问题。
- 必须记住的算术（`γ^T` 决定终局信号还剩多少）：

| γ | 有效视野 `1/(1−γ)` | `γ^1000` | 1000 步后终局信号还剩 |
|---|---|---|---|
| 0.99 | 100 | 4.3e-5 | **等于没有** |
| 0.995 | 200 | 6.7e-3 | 0.7% |
| 0.999 | 1000 | 0.368 | 37% |
| 0.9995 | 2000 | 0.606 | 61% |

⇒ **在 400–512 回合、约 1000 决策的局里，`γ=0.99` 是错误的默认值**（它等价于"只看未来 100 步"）。
要么 `γ≈0.999` 接受高方差，要么 **`γ=1` + 在截断/终局处 bootstrap `V`**（MuZero 在棋类上的做法）。

### 2.7 "刷分"的形式化（用来写验收判据）

- Skalse, Howe, Krasheninnikov, Krueger, *Defining and Characterizing Reward Hacking*, NeurIPS 2022
  <https://arxiv.org/abs/2209.13085>
- 定义（原文摘要）：代理 `R̂` **unhackable** ⟺ "提高期望代理回报**永远不会降低**期望真回报"。
  关键结论：奖励对"状态-动作访问计数"是线性的，这让 unhackable 成为**非常强的条件**；
  在**全体随机策略**上，"两个奖励函数 unhackable"**只有当其中一个是常数时才可能**。
- 用法：给出一条**判据模板**——"我的代理（含 shaping）与真值（HP 差/胜负）是否同序？"
  PBRS 之所以真安全，是因为 `V^π_{M'} = V^π_M − Φ` 对**所有**策略成立（Remark 1）——它安全和无用是一体两面。
- 背景：Amodei et al., *Concrete Problems in AI Safety*（reward hacking 一节）<https://arxiv.org/abs/1606.06565>

---

## 3. 落地草案：potential 函数的具体设计

### 3.0 四条设计原则

1. **只奖励 payoff 的估计，不奖励"状态量"本身**（金币、塔数、HP 都是状态量）。
   判据：任何新信号先问——**它是不是某个"只看状态"的函数的差分？** 是 → 可能是合法 potential；不是 → 必被刷。
2. **`Φ(终局) = 0`**（Ng 1999 定理的 `s ∈ S − {s0}`）。这是硬要求，不是可选约定。
3. **能不加系数就不加系数**（量纲对齐成 1:1）。需要手调的自由系数通常说明形状还没想清楚。
4. **探索靠熵/温度，不要靠奖励项**。用奖励买探索是典型的刷分入口。

### 3.1 主推：`Φ = V`，只加在搜索的评估函数上

```text
# 对每个候选动作 a（当前状态 s，执行后 s'）
score(a) = R̂_quiet(s')                     # 现状：安静 rollout 的终局血量差
         + [ γ·Φ(s') − Φ(s) ]              # 新增：potential 项，Φ = V_θ，γ=1
         = R̂_quiet(s') + Φ(s') − Φ(s)
```

- 在当前状态 `s`，`−Φ(s)` 对所有候选动作是同一常数，**不改变 argmax**；
  所以单步决策里它的作用等价于"给叶值加上 `V(s')`"。
- `Φ` 不可替代的地方在**多步/不同深度**：加了 `−Φ(s)` 后，`score` 变成"相对当前势的增量"，
  不同深度、不同截断点的分数才可比（望远镜性质）。
- **量纲**：`R̂` 与 `V` 同为 `HP 差 / HP_SCALE`（±1 量级）⇒ **1:1 相加，无自由系数**。
  这正是 `docs/prereg_20261008_mc_value_delta.md` 的 D 臂，**形态是正确的**。
- **实现要点**：`rel` 模式训练出的值网输出 = `未来加权血差 − 当前血差`，**它本身就是一个"Φ − 基线"**。所以
  - 放进**差分项** `Φ(s')−Φ(s)` ⇒ 不需要 `AZAI_REL2ABS`；
  - 当**绝对叶值** ⇒ 必须 `AZAI_REL2ABS=1.0` 还原，否则少了一个 `−Φ(s)` 偏移——
    这正是 `az_train.py` 注释里写的 "the search reads value as an absolute evaluation → mismatch"。
    用 §2.1 的话说：**`rel` 值网天生适合当 potential，不适合当绝对叶值。**

### 3.2 备选：`Φ_hp`（纯血差潜力），只当兜底/基线

```text
Φ_hp(s) = clip( (hp_us − hp_opp)/20, −1, +1 )   # 玩家视角；初始 50/50 ⇒ Φ(s_start)=0
F = Φ_hp(s') − Φ_hp(s)                          # 恰好是"每回合血量变化"
```

严格按定理，终局那一步应当用 `Φ(s0)=0` 结算；你们若用 `Φ(s_T)=margin_T`，得到的是
`Σ F = margin_T − 0` ⇒ 塑形回报 = `2·margin_T`，**仍是单调正缩放、排序不变**（安全，但零新信息）。
（一般规则：`Φ(终局)` 可以等于 payoff 的单调函数；**一旦它依赖 payoff 之外的东西（金币/塔数），invariant 就破了**。）

### 3.3 资源项：**不建议叠加**；要定价就"续盘校准"

若硬要 `Φ = Φ_hp + ρ·Φ_gold`（`Φ_gold = 金币差/20`），按定理只有 `Φ_gold(终局)=0` 才合法：

```text
合法形态（终局归零）：Σ F = −ρ·gold_0 = 常数        ⇒ 数学上中性
  但每步 F = ρ·(收入 − 支出)：花钱 = −ρ，攒钱 = +ρ，终局一次性没收
不合法形态（终局不归零）：Σ F = ρ·(gold_T − gold_0)
  ⇒ 目标变成 margin + ρ·gold_T ⇒ ρ>0 就是在奖励攒钱（这正是你们看到的现象）
```

**所以：安全 ≠ 有用。** 即使写成合法形态，这个 potential 的**每步梯度方向也是反的**
（它惩罚"花钱"这个你唯一想鼓励的动作），只靠终局没收来"对冲"——在有限样本/函数近似下，
"先奖后罚"的时序错配就是"攒钱不做事"的温床。而且它跟 `Φ_hp` 会**互相拆台**：
`Φ_hp` 说"别掉血"、`Φ_gold` 说"别花钱"，而花钱正是唯一能减少未来掉血的动作。

**有原则的做法：不叠项，去校准"金币的 HP 汇率"**（用来改**标签**，不改奖励）：

```text
# 金币定价实验（离线，不改对局）
1. 取截断局面 s_T（打满 512 回合的局，尤其 1–3 血差附近）；
2. 用当前最强策略让双方从 s_T 继续跑 N=256 回合，得 margin_cont；
3. 回归：margin_cont ~ a + b·margin_T + c·gold_diff + d·tower_val_diff + e·ant_val_diff
4. 若 c 的 95%CI 不含 0 ⇒ 金币在截断点确实有价，把 c 当汇率写进标签；
   否则 ⇒ 数据说金币在终局无价（那就别给它定价——这正是你们 P3 想知道的答案）。
```

价值：把"金币值多少"从**手调系数**变成**可测参数**，且**不受 reward hacking 影响**（只改标签/评估，不引入新优化项）。
成本：引擎要能从中局续跑（你们有 `clone`）。

### 3.4 系数怎么定（汇总）

| 方案 | 系数 | 怎么定 | 是否被刷 |
|---|---|---|---|
| `Φ=V` + 1:1 | 无 | 量纲相同，直接相加 | 否 |
| `Φ_hp` | 无 | 同上（`/20`） | 否 |
| `Φ_hp + ρΦ_gold` | ρ | **只能靠续盘回归估**，不能手调 | 终局不归零 ⇒ **是** |
| 直接给"金币奖励" | ρ | — | **是（必然）** |

### 3.5 验证"没被刷分"的四关

> ⚠️ **背景修正（重要）**：在这个游戏里 **HOLD（不动）常常就是正确决策**——乱建塔再拆、空放超武都白烧金币。
> 所以**"建塔数下降"本身不是刷分信号**，也不能当"给金币定价生效"的证据。
> 判据必须落在**结果**上（HP margin / 胜率），再加一个**专属性指纹**（终局金币）。

1. **结构断言（unit test）**：随机状态 + 所有终局状态上断言 `Φ` 不依赖动作/回合数；`Φ(终局)=0`
   （或严格等于 payoff 的单调函数）。5 分钟能写完，能挡掉 80% 的实现错误。
2. **缩放不变性（最有力的自检）**：把 `Φ → 2Φ`（或 `Φ + 常数`），在**同一批状态**上重算 `argmax_a score(a)`。
   **正确实现必须给出完全相同的 argmax**。若变了，你写的不是 PBRS
   （最常见原因：终局没归零 / `Φ` 用了动作或 `t` / 差分写反成 `Φ(s) − γΦ(s')`）。
3. **机制探针**（你们已有 8 个读数）：建塔数、闪电、出招回合、对局回合、`dec_max/dec_slow`，
   再加两个**为刷分专设的**：**终局金币**、**终局塔数/塔价值**。
   判据：**若 HP margin 不涨而"终局金币"显著涨 ⇒ 刷分**（攒钱的直接指纹）；
   若 HOLD 比例上升但 margin 也涨 ⇒ **不是刷分，是变强了**（见上面的背景修正）。
4. **端到端 A/B（写死判据）**：主指标 HP margin（32 局分辨率 ±5.5hp），副指标胜率（±17pp）；
   必须配对、固定前缀、两臂都报。

### 3.6 与现有实验的接口

- `AZAI_MC_SCORE=delta`（Φ=V 差分）**就是** §3.1，形态正确，值得继续；
- `AZAI_MC_SCORE=value_only` 相当于"把 `R̂` 权重设为 0"——它**不再是 PBRS**（少了基座那一项），
  是"纯值网评估"，**没有 policy invariance 保护**，只能靠端到端验证。建议当"快"的对照，不当"真"。
- 若 `delta` 无效：按 §3.3 做续盘校准，改**标签**（把 `c` 加进 `rel` 标签目标），而不是改奖励。

---

## 4. 已读 / 未读清单（诚实版）

| 文献 | 状态 | 说明 |
|---|---|---|
| Ng, Harada & Russell 1999 | ✅ **已读原文（第 2–5 页）** | PDF 字体无 ToUnicode，`pypdf`/`fitz` 抽出来是乱码；改为**渲染 PNG 逐页读图**。§2.1 的定理/推论/注释均为原文词句 |
| RUDDER (NeurIPS 2019) | ✅ **已读原文** | `fitz` 抽出 102 页全文（含附录）；引用了式 (1)、Theorem 3、contribution analysis 段 |
| MuZero (arXiv 1911.08265) | ✅ **已读原文（方法节 + 附录 G）** | `fitz` 抽出 21 页；value target 公式、`n=10`、`K=5` 均为原文 |
| Skalse 2022（reward hacking） | ✅ 已读原文 | 摘要 + §3 的势函数引理 |
| Reward Machines / HER / GAE / 课程综述 / AlphaZero / OpenAI Five | ✅ 已读摘要 + 部分正文 | OpenAI Five 的"退火"只读到超参表与表注，**没有读 Team Spirit 的定义** |
| Wiewiora 2003 | 🟡 **二手**（摘要级） | 只读了 arXiv 摘要 + JAIR 页面；"shaping ≡ Q 初始化"的证明没读 |
| AlphaStar 的 PFSP 定义 | 🔴 **未读到** | Nature 付费墙；只核到 DOI 可解析。§2.5 里"按对手强弱优先采样"的表述依据 PSRO 那篇 |
| Grzes 2017（episodic shaping 的正环问题） | 🔴 **未读到** | 猜的 IFAAMAS 链接 404；**故全文未引用**。它是"`γ<1` 时 PBRS 可能变差"这条论证的关键文献，属于本次缺口 |
| MuZero 的 two-hot / 分布化 value 表示 | 🔴 **未读到** | arXiv 版 PDF grep `two-hot` 0 命中（可能在 Nature 版/SI），**故 §6 不引用** |
| 2024–2025 的新工作（LLM 辅助 reward design 等） | 🔴 **未覆盖** | 搜索工具不可用，只靠 arXiv API 按标题精确检索 + 已知经典文献，**可能整片漏掉** |

其他限制：

- `web_search` 报 `DeepSeek search request failed: TypeError: fetch failed`；`web_fetch` 因 fake-IP DNS
  （`arxiv.org → 198.18.3.21`）被判"非公网 IP"拒绝。所有抓取走 `curl` / `Invoke-WebRequest` + 系统代理。
- 我读到的可能不是你打开的版本（arXiv 有 v1/v2/v3；RUDDER 我读的是 v3, 2019-09-10）。
- **OpenReview 的 PDF 是 403 + JS 挑战**，本次没用到，但记录在此以便后续。

---

## 5. 追加问答①：把"每回合血量变化"当 potential

> 补充问题：取 `Φ(s) = 我方血量 − 对方血量`，则 `F = γΦ(s') − Φ(s)` **就是**血量变化。
> 它满足 Ng 1999 吗？失效模式是什么？能不能再叠一个"金币/塔数"的 potential？

### 5.1 满足吗？——**满足**，而且是"最干净"的那一类

- 形式合法：`Φ` 是**纯状态函数**（只含双方 HP），`F = Φ(s')−Φ(s)`，`γ=1`，初态 50/50 ⇒ `Φ(s_start)=0`。
- 按 Corollary 2：`V^π_{M'}(s) = V^π_M(s) − Φ(s)` ⇒ 塑形回报 = 真回报 `+ 常数`，**所有策略同序**。
- 望远镜：`Σ F = Φ(s_T) − Φ(s_start) = margin_T − 0`，而真回报也是 `margin_T` ⇒ 塑形回报 = `2·margin_T`
  （单调正缩放，排序不变）。
- ⚠️ 两个必须钉住的前提：
  1. **`γ` 必须是 1**。若 `γ<1`，`Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0) ≈ 0`（`0.99^1000≈4.3e-5`）
     ⇒ 终局项被抹掉，目标退化成"**贴现的短期掉血**"，**不再是原目标**。
     这是"用血量变化当奖励"最容易踩的坑：`γ=0.99` + 1000 步 ⇒ 你在优化一个**短视**目标。
  2. **终局 payoff 的定义要和 `Φ(s_T)` 一致**。你们打满 512 的局，payoff 就是那一刻的 HP 差 ⇒ 一致。
     但若 payoff 改成"胜负 ±0.5"（`train_value_stream --label win`），
     则 `Φ(s_T)=margin_T` 与 payoff **不同序**（1 血胜 = 20 血胜，但 potential 觉得差 20 倍）⇒ 排序会变。
- 更关键的一点：**这条 potential 提供的是 0 比特新信息**（它只是把同一个 margin 摊成 1000 份）。
  它的真收益是**把信用链从 1000 步压到 1 步**（方差骤降），不是"教会 AI 新东西"。

### 5.2 失效模式

| 失效模式 | 机制 | 可测症状 |
|---|---|---|
| **苟血 / 不投资** | `Φ` 只含 HP。塔/蚂蚁是"用金币换未来减伤"的投资，收益要 N 回合后才体现在 HP 上；截断 + 函数近似下投资"先亏后赚"，而 shaping 把"先亏"立刻记账 | margin 不涨；且**不能**用"建塔数下降"当证据（HOLD 本来就常是最优，见 §3.5） |
| **归因滞后** | 本回合掉的血是 N 回合前放进去的蚂蚁造成的 ⇒ `F_t` 与该步动作相关性弱 | 同一状态下不同动作的 `F` 方差极小（可直接测） |
| **正交盲区** | `Φ` 不含金币/塔位/经济 ⇒ 对"该不该花钱"**零信息**。这和你们"安静 rollout 里金币没用"是同一个病，只换了位置 | 花钱相关机制读数不随它变化 |
| **截断漏** | 若在打满 512 的局上额外把 `Φ` 归零，等于**没收**了真实的终局血差 | 平局局分数被系统性抬高、margin 分布变差 |
| **`γ<1` 短视** | 见 5.1 | 长局（400+ 回合）表现更差 |
| **`clip(±1)` 丢序** | `|margin|>20` 时不严格单调 | 大优势局之间无法区分（训练影响小，但会掩盖"赢 20 血 vs 赢 45 血"） |

### 5.3 能不能叠第二个 potential（金币/塔数）？

1. **数学上不冲突**：`Φ = Φ_hp + ρΦ_gold` 仍是状态函数，仍是合法 potential（"两个 potential 之和"= 一个 potential）。
2. **但终局项决定一切**（唯一真正重要的地方）：

   ```text
   合法（Φ_gold(终局)=0）：Σ F = −ρ·gold_0 = 常数        ⇒ 中性，但每步"花钱=−ρ、攒钱=+ρ"
   不合法（终局不归零）：Σ F = ρ·(gold_T − gold_0)
     ⇒ 目标变成 margin + ρ·gold_T ⇒ ρ>0 必攒钱、ρ<0 必花光、ρ=0 才中性
   ```

   **换个写法不换命**：把"金币奖励项"换成"金币 potential 项"，只要终局不归零，就是同一个 bug。
3. **语义上互相拆台**：`Φ_hp` 要"别掉血"、`Φ_gold` 要"别花钱"，而花钱是唯一能减少未来掉血的动作。
   两者的和让"花钱买防御"在记账上**先净亏**，只有在 `V` 或后续 `Φ_hp` 里才补回来。
   `ρ` 偏大 ⇒ 过度省钱；`ρ` 偏小 ⇒ 完全没效果。**中间没有一个"刚好"的 `ρ`，因为问题不在幅度而在符号。**

**建议**：

- **不要在任何 potential 里出现第二个状态量。** 把金币/塔位放进**值网输入**（你们 `stats` 里已有），
  让唯一的 `Φ=V` 去表达它们的价格——这是唯一不会自相矛盾的写法。
- 若确实需要显式汇率，用 §3.3 的**续盘回归**估 `c`，把它加进**标签**而不是 potential。
  区别：标签错只是"学得偏"，potential 错是"目标错"。

---

## 6. 追加问答②：我们该走哪条范式（蒸馏 vs AlphaZero）

### 6.A 保持"搜索蒸馏"：reward shaping 还有意义吗？

**有，但要改名。** 蒸馏范式里没有"奖励"，只有**评估函数**。shaping 搬到评估函数后，三条正当用法：

| 用法 | 公式 | 解决什么 | 理论保护 |
|---|---|---|---|
| ① 截断偏差修正 | `score(a) = R̂(s') + V(s')` | rollout 被 `MC_HORIZON` 截断时，补上"剩下那段"的价值 | 有（`Φ=V`，`Φ(终局)=0`） |
| ② 方差缩减 / 基线 | `score(a) = R̂(s') + [V(s') − V(s)]` | `−V(s)` 是常数不改 argmax，但让**不同深度**的分数可比 | 有（同上） |
| ③ 差值形式匹配 | 用 `rel` 训练的值网做 `Φ(s')−Φ(s)` | `rel` 标签天生是 `V−基线`，放进差分项才对 | 有（§3.1） |

**不能做的**：指望 `F` 给评估函数注入值网不知道的信息。你们实测过两次——
`docs/test_match/_fill_prior_round12.py`："值网 R²=0.47（标签=终局血量差）⇒ 它的排序只是与'赢'弱相关的代理"；
`docs/test_match/_fill_prior_summary.py`："Spearman 三次与棋力脱钩（含'越准越差'）"。
**瓶颈在评估器的信息量/校准，不在信号密度。**

**一句话**："血量变化"应该拿来改**搜索的评估函数**（差分、深度一致性、兜底），**不要**当训练奖励；
"金币的价值"要靠**改标签/续盘校准**（§3.3）解决，不是靠任何 shaping。

### 6.B 改成 AlphaZero/MuZero：先纠正一处理解

你的理解**一半对**，要分开两件事（都有一手原文）。

**(1) value target 用的什么？**

- **AlphaGo Zero / AlphaZero**：value target = 终局 `z`（胜负），**无 shaping**。✓ 你的理解正确。
  <https://arxiv.org/abs/1712.01815>
- **MuZero**（我读的是方法节原文，`fitz` 抽取，逐字）：

  > "Also like AlphaZero, the improved value targets are generated by playing the game or MDP. However,
  > unlike AlphaZero, we allow for long episodes with discounting and intermediate rewards by bootstrapping
  > n steps into the future from the search value,
  > **z_t = u_{t+1} + γu_{t+2} + … + γ^{n−1}u_{t+n} + γ^n ν_{t+n}**.
  > Final outcomes {lose, draw, win} in board games are treated as rewards `u_t ∈ {−1, 0, +1}`
  > occurring at the final step of the episode."

  再对照 "…where `z_{t+k}` is a sample return: **either the final reward (board games) or n-step return (Atari)**"，
  以及附录 G："**For board games, we bootstrap directly to the end of the game, equivalent to predicting the
  final outcome; for Atari we bootstrap for n = 10 steps into the future.**"

  ⇒ **棋类（无中间奖励）的 value target 就是终局 `z`**；只有**有中间奖励的域（Atari）**才 `n=10` 混合。

**(2) 具体怎么混合、参数多少？**

```text
z_t = Σ_{i=1..n} γ^{i−1} · u_{t+i}  +  γ^n · ν_{t+n}
       └── 真实观测到的 n 步奖励 ──┘     └─ 第 t+n 步的【搜索值 ν】，不是网络输出 v ─┘
```

- `n = 10`（Atari）；棋类无中间奖励 ⇒ `n` 直接到终局，`z` 就是 `±1/0`。
- 训练时模型 unroll `K = 5` 步，每一步都有 value/policy/reward 三项损失（式 1）。
- **混合的是"真实 n 步奖励"与"搜索值 ν"**（一个更可靠的 bootstrap），**不是"z 与网络 V 的 50/50"**。
- 附带一条对你们有用的细节：MuZero 说两人零和局里 value 假设在 `[0,1]`，但"many environments the value is
  unbounded"，于是它对 Q 做**基于搜索树 min-max 的归一化**——这正是你们 `HP_SCALE=20` + `clip(±1)` 想解决的问题。

**(3) 对我们的映射**：我们的局**更像棋类**（一个终局标量 HP 差），**不像 Atari**（有中途 kill/金币等真奖励）。
所以该抄的是 **AZ 的 `z` + MCTS backup**，**不是**"设计中间奖励"。
你们现在的 `ITERS=256 DEPTH=4` + 叶值网络，**已经在做 MuZero 的 Eq.(3)**（`l−k = 4`）：

```text
MuZero Eq.(3)（搜索 backup，所有域都用）：
G_k = Σ_{τ=0}^{l−k−1} γ^τ · r_{k+1+τ} + γ^{l−k} · v_l
```

所谓"长视野"，在这个范式里是**搜索深度 + 值网质量**的问题，不是奖励问题。

### 6.C "n 步自举"要不要优先于"设计中间奖励"？

**要，而且优先级高得多。** 四条理由：

1. **不可被刷**：n 步自举用的是**同一个终局信号**，只是把"等 1000 步"改成"每 n 步用 V 猜一次"，
   不引入新目标 ⇒ 没有新的可刷分面（§2.1 Remark 2）。"设计中间奖励"引入新目标 ⇒ 必须过 §3.5 全套。
2. **它解决的是真问题**：`γ=0.99 / T=1000` 让终局信号只剩 `4.3e-5`（§2.6 表）；`γ=1` + n 步自举把
   有效信用链压到 `n`。这是"长视野"的**定义级**解法。
3. **它就是你们已经在做的事**：`tau ∈ {5,10,50}` 的指数加权未来血差 = TD(λ) 式的多步回报；
   `MC_HORIZON=256` 的安静 rollout = "很长的 n 步"。升级路线是**把 n 步做得更规范**
   （调 `tau`/`n`、用 `V` bootstrap 替代 256 步 rollout、在截断处显式 bootstrap），而不是加一个新奖励。
4. **中间奖励在零和自对弈里更危险**：任何非 potential 的中间奖励都会改变最优策略（Ng 逆定理），
   而对手会**学会诱发你的奖励项**——这不是理论担忧，是自对弈的常态。

### 6.D 优先级建议（按性价比）

| 优先级 | 动作 | 成本 | 依据 | 风险 |
|---|---|---|---|---|
| **P0** | 把值标签的 `tau`/`n` 当"信用链长度"来扫（现成旋钮） | 低 | §2.6 | 无 |
| **P0** | `delta` 臂（`Φ=V` 差分）继续跑，按 §3.5 四关验收 | 低 | §2.1/§3.1 | 低 |
| **P1** | 截断局"续盘校准"求金币汇率，写进**标签**（§3.3） | 中 | §3.3 | 中（引擎要能续跑） |
| **P1** | 分布化/分桶 value head（对抗 1–3 血的分辨率问题） | 中 | 你们已测 `std 0.024` vs `0.062` | 低 |
| **P2** | 对手池 PFSP + 平局/接近局分层报告 | 中 | §2.5 | 低 |
| **P3** | 真 RL（AZ 式：`z` + MCTS backup），**不带任何中间奖励** | 高 | §6.B | 高（算力/稳定性） |
| **P4** | RUDDER / reward machines / HER | — | §2.2–2.4 | 不推荐 |

---

## 附：与本项目已有文档的接口

- `docs/ppo_experiment_plan.md`：只用 `ΔHP` 当奖励、"避免人工 shaping"——**方向正确**；
  但方案里的 `γ=0.99` 对 1000 步是错的（§2.6 表）。
- `docs/prereg_20261008_mc_value_delta.md`：D 臂 = `Φ=V` 的 PBRS，**形态正确**；
  其 P3（"若 V 对金币不敏感 ⇒ 要改训练目标而不是换评分"）**与本文 §1 第 1 条/§2.1 Remark 2 完全一致**，
  可以视为理论上的预先支持。
- `code/my_ai/entity_transformer/build_rel_labels.py`：`rel` 标签 = "未来加权血差 − 当前血差"
  = 一个"value − 基线"的量，**天生适合当 potential 差分**（§3.1）。
- `code/my_ai/az_intent/mcts.py`：`HP_SCALE=20`、`clip(±1)`——量纲对齐良好；
  注意 clip 在 `|margin|>20` 时丢序（§5.2 最后一行）。
