# E 线调研：强度评估协议，与 KataGo / Leela Zero 可抄的评估配方

> **范围**：只做文献调研 + 落地草案，不改代码、不动正在跑的实验。
> **写作时间**：2026-10-26。
> **本文要回答的一句话**：在"每局很贵（几分钟到几十分钟）、座位差 27pp、离线指标与实战多次脱钩"的条件下，用**多少局、怎么配对、用什么统计量、阈值取多少**，才能可信地说"A 比 B 强 / 弱 / 还没证据"。

> **网络访问说明（重要，影响可信度分级）**
> 本会话 `web_fetch` 对**所有**外部域名返回 "resolves to a non-public IP address"（与 B 线那次会话不同，那次 arXiv 可直连）。实际采用的抓取路径是：`pwsh` 里 `Invoke-WebRequest` 直连 + `PyMuPDF` 抽 PDF 文本；Playwright 作备用。
> 因此本文对每条引用都标了 **【正文】**（我读到了原文，可能只是其中若干页）或 **【二手】**（只确认了标题/被转述）。汇总见 §5。

---

## 0. 术语对齐：我们的话 ↔ 文献的话

| 我们的说法 | 文献里的说法 | 备注 |
|---|---|---|
| "16 局 ±25pp" | 二项比例的 **95% 置信区间半宽** = 1.96·√(p(1−p)/n) | 纯抽样噪声，跟 AI 好坏无关 |
| "做 16 回合 rollout、值网打分" | **playout / visit cap** 的搜索（不是 MCTS 树） | 文献里的 cap 指"每次决策花多少 playout"，与我们同义 |
| 每候选 16 rollout | fixed-budget **one-step lookahead with rollouts** | |
| 座位 P0 / P1 | **color / first-player advantage**；BayesElo 叫 `eloAdvantage`，Ordo 叫 `W_adv` | 关键：它是 **logit 空间的一个常数偏移**，不是"噪声" |
| 同一张图两局、交换座位 | **paired games with reversed colors** / **pentanomial model** | Stockfish 全家桶的标准做法 |
| "更强 / 更弱 / 无差别" | **three-way decision**；SPRT 原生只有两态 | 见 §3.3 的坑 |
| 离线 Spearman / recall@k | **proxy metric / offline metric** | 文献里几乎没有"它就是强度"的正面论证 |
| 先验网 | **policy prior**（KataGo 叫 raw policy） | |
| 值网 | **value head**，KataGo 还有 ownership / score 辅助头 | |
| HOLD（不动） | 合法动作；在 KataGo 里对应 **pass** | 评估协议里不该把它当异常，见 §4.1 |

---

## 1. 结论摘要（5 条，直接回答"我们该用什么评估协议"）

1. **不要再做"固定 16 局看胜率"这种事，改用序贯检验 + 配对 + 座位协变量这三件套。** 固定样本量设计在效应大时浪费、效应小时不够；SPRT 在同样的 α/β 下**期望局数只有固定样本设计的一半**，而且给出的是一个停止规则而不是一个拍脑袋的 N。配对（同图两局、交换座位）把"这张图/这个座位好不好"这一层方差从比较里扣掉；座位协变量把 27pp 的座位主效应显式建模，而不是赌随机化能把它平衡掉。

2. **"多少局"这件事有闭式答案，可以直接算。** fishtest 的统计文档（van den Bergh，`sprta.pdf`）给出：对 α=β=0.05 的 SPRT，
   - 真值正好等于 H1 时：`E[N] = 5.30 · σ² / Δ²`
   - 真值正好卡在 H0、H1 中点上（**最坏情况**）：`E[N] = 8.67 · σ² / Δ²`
   - 其中 Δ = 得分率差、σ² = 每局得分的方差（无平局时 σ²=0.25）。
   换算成我们的语言：**判定"A 对 B 得分率 52.5%（强 5pp）"最坏需要约 3470 局；判定 55%（强 10pp / 约 35 Elo）最坏约 870 局。** 配对如果能把方差压掉 30%（κ=0.3，见 §3.5），这两个数变成 2430 / 610。**在"每局 5 分钟、16 workers"的价位上，这就是 12.6 小时 vs 3.2 小时的区别——所以配对不是可选项，而"5pp 值多少小时"必须提前算清楚。**

3. **"座位差 27pp"不是要绕开的噪声，是一个必须写进模型的协变量；它同时是一个 bug 线索。** 72% 对 25% 在 logit 尺度上是 **355 Elo** 的先手优势（§3.6 有换算）。这种量级下，"胜率"这个绝对量本身没有意义（取决于座位配比），但**A 与 B 之间的差**在座位平衡时是干净的。正确做法不是二选一（配对 vs 回归），而是**配对设计 + Bradley–Terry 里加一个 seat 项**：Ordo 的 `W_adv` 和 BayesElo 的 `eloAdvantage` 就是现成实现（BayesElo 在 29610 局上估出国际象棋先手 = 32.8±4 Elo）。另外，355 Elo 的先手优势在我们这个游戏里大得反常，**第一件该查的是镜像等变性/位置编码是否有座位相关的不对称**，而不是接受它。

4. **离线指标（Spearman、recall@k）不能拿来判强度，只能拿来当冒烟测试。** 这条在 Leela Zero 官方 FAQ 里被直说：`"Self-Elo" is not a true indicator of net strength, but just a general parameter of the "health" of the training run`。你们"离线 +4.7pp、实战 −20pp"不是奇怪现象，是**代理指标与目标指标脱钩**的教科书案例。可行的补救不是"找更好的离线指标"，而是：把离线指标降级为回归测试，并且**用历史 checkpoint 拟合一条"离线分 → 实战 Elo"的校准曲线**，只有当校准 R² 足够高时才允许用它做门禁。

5. **"便宜"的四个杠杆，按性价比排序**：(a) 序贯停止（省 ~50%）；(b) 配对/同种子对开（省 κ 比例的方差，fishtest 实测 15–18%，我们的座位/地图方差应该远大于此）；(c) 用**得分差/HP 差**而不只是胜负二值（每局的信息量从 1 bit 变成一个连续量）；(d) 用**全局 Bradley–Terry 拟合所有历史对局**，而不是每次只做 A/B 两两对比（KataGo 评估自己时是用 ~50000 局拟合 relative Elo，并且**按预测方差 p(1−p) 采样对手**，把算力花在信息量最大的对局上）。

---

## 2. 速览表

| # | 方法 | 一句话 | 我们要不要抄 | 关键链接 |
|---|---|---|---|---|
| 3.1 | 二项样本量公式 | 16 局 ±25pp 是数学必然 | 只用来**反推**"这个局数能分辨多少" | [wiki: statistical power](https://en.wikipedia.org/wiki/Power_of_a_test) |
| 3.2 | **SPRT**（Wald） | 累积 LLR 走出 `[ln(β/(1−α)), ln((1−β)/α)]` 就停 | **抄**，核心 | [wiki SPRT](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test) |
| 3.3 | **GSPRT**（fishtest 实现） | 把未知参数换成 MLE，LLR 用 (2.1) 近似 | **抄**，实现只有几十行 | [gsprt.pdf](https://www.cantate.be/Fishtest/GSPRT_approximation.pdf) |
| 3.4 | **Normalized Elo** | 让"多少局"与和棋率无关的归一化 | **抄**（我们是无平局，退化成正数倍） | [normelo.pdf](https://www.cantate.be/Fishtest/normalized_elo_practical.pdf) |
| 3.5 | **Pentanonial / 配对** | 同开局两局交换颜色，方差能小多少有恒等式 | **抄**，但要先测 κ | [acct.pdf](https://www.cantate.be/Fishtest/accounting_identity.pdf) · [pentanomial](https://github.com/vdbergh/pentanomial) |
| 3.6 | Elo / BayesElo / Ordo | 把一堆对局压成一个数 | 用其**模型**，不用其**排名** | [BayesElo](https://www.remi-coulom.fr/Bayesian-Elo/) · [Ordo](https://github.com/michiguel/Ordo) · [LCZ FAQ](https://lczero.org/dev/wiki/faq/) |
| 3.7 | 座位/地图当协变量 | logit 里加 seat 项 | **抄**，直接解决 27pp | [BradleyTerry2](https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html) |
| 3.8 | 离线 vs 实战 | 代理指标脱钩 | 降级为冒烟测试 | [LCZ FAQ](https://lczero.org/dev/wiki/faq/) |
| 3.9 | Best-arm ID | "少量样本分两臂"的理论下界 | 只用来**定预算下限** | [Kaufmann JMLR 2016](https://jmlr.csail.mit.edu/papers/volume17/kaufman16a/kaufman16a.pdf) |
| 3.10 | 截断 / 裁判 | 提前结束对局换方差 | 谨慎，需双报告 | [KataGo §6.2.1](https://arxiv.org/abs/1902.10565) |
| 3.11 | KataGo 配方 | playout cap oscillation、辅助头、gate | **抄** | [KataGo 论文](https://arxiv.org/abs/1902.10565) |
| 3.12 | LZ / AZ gate | 400 局 55% | 抄它的**廉价**思路 | [LZ #1054](https://github.com/leela-zero/leela-zero/issues/1054) |

---

## 3. 逐方法

### 3.1 先把"分辨率"钉死：二项比例的样本量

**【正文】** 标准两比例/单比例样本量公式（Power of a test 是标准内容，我引 Wikipedia 页面作可点击出处）：

```
单比例 95% CI 半宽:  h = 1.96·√(p(1−p)/n)
两比例检验(α=0.05 双侧, power=80%): n_每臂 = (1.96+0.84)²·[p₁(1−p₁)+p₂(1−p₂)] / Δ²
```

**核对你们的三个数字**（全部自洽，说明"分辨率差"是抽样噪声的必然，不是 bug）：

| 你们的观测 | 公式值 |
|---|---|
| 16 局 ±25pp | 1.96·0.5/√16 = **±24.5pp** ✓ |
| 32 局 ±17pp | 1.96·0.5/√32 = **±17.3pp** ✓ |
| 6%→12% 需 350 局/臂 | (1.96+0.84)²·(0.0564+0.1056)/0.06² = **353 局/臂** ✓ |

**关键假设**：每局独立同分布、同一座位配比。**对我们的意义**：这条公式同时说明——**在 n<400 的规模上，任何 <5pp 的差异都不该被当真**；想靠"多跑几十局"解决问题是不可能的，必须换设计。

---

### 3.2 SPRT（Wald 序贯概率比检验）【正文】

- **链接**：[Wikipedia: Sequential probability ratio test](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test)
- **核心思想**：不要预先定 N。每来一个观测就往累积对数似然比上加一项，一旦 LLR 跑出区间就停并判决。Wald 证明它在同样的 α、β 下**期望样本量最小**。
- **边界（这是要抄的第一条公式）**：

```
LLR_t = Σ_{i=1..t} log[ f(x_i | θ₁) / f(x_i | θ₀) ]

a = ln( β / (1−α) )        < 0
b = ln( (1−β) / α )        > 0

LLR_t ≤ a  →  接受 H0（"没有 H1 说的那么强"）
LLR_t ≥ b  →  接受 H1
否则继续
```

  α=β=0.05 时 `b = −a = ln 19 = 2.9444`。

- **期望局数（闭式，`sprta.pdf` Theorem 1.1）**——**这是本节最有用的东西**：

```
h_µ = [ 2µ − (µ₀+µ₁) ] / (µ₁−µ₀)          # 真值在 [µ₀,µ₁] 里的相对位置，h=−1 对应 µ₀，h=+1 对应 µ₁
w   = (µ₁ − µ₀) / σ                        # σ = 每观测标准差
P(h) = (1 − e^{−h a}) / (e^{−h b} − e^{−h a})            # 通过概率
T(h) = (2/h) · [ b·P(h) + a·(1−P(h)) ]                   # 无量纲时长
E[N] = T(h_µ) / w²
```

  α=β 时化简为 `P(h)=1/(1+e^{−hb})`、`T(h)=(2b/h)·(1−e^{−hb})/(1+e^{−hb})`，于是：

| 真值位置 | T(h) | E[N] |
|---|---|---|
| 正好 = H1（h=1） | 5.300 | `5.30·σ²/Δ²` |
| 正好 = H0（h=−1） | 5.300 | `5.30·σ²/Δ²` |
| **卡在中间**（h=0，最坏） | `−ab = b² = 8.670` | `8.67·σ²/Δ²` |

  **对照固定样本**：α=0.05 单侧、power=95% 的固定样本检验需要 `(1.645+1.645)²·σ²/Δ² = 10.8·σ²/Δ²`。所以 **SPRT 期望局数 ≈ 固定样本的一半**（5.30 vs 10.8）。这就是"省 50%"的确切来源。

- **关键假设**：观测 i.i.d.；H1 是**点假设**；离散时间下 LLR 会"冲过头"（overshoot），严格处理要加修正。fishtest 的实践结论值得抄：**理论正确的 overshoot 修正实测没有任何可检测的改善，于是他们保留了朴素做法**（见 [fishtest issue #348 评论](https://github.com/official-stockfish/fishtest/issues/348#issuecomment-457519686)）。
- **对我们**：**必抄**。实现量约 60 行。唯一要注意的是它只给两态（见 3.3）。

---

### 3.3 GSPRT：fishtest 真正跑的那个版本【正文】

- **链接**：[A practical introduction to the GSPRT](https://www.cantate.be/Fishtest/GSPRT_approximation.pdf) · [Fishtest Mathematics 文档](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
- **核心思想**：真实问题里 H0/H1 里还有讨厌参数（和棋率、开局偏置），所以不用简单 SPRT，而用**广义**似然比：`LLR = L(θ̂₁,x) − L(θ̂₀,x)`，其中 θ̂ᵢ 是"在约束 φ(θ)=φᵢ 下的 MLE"。文档给出一个**不用解约束优化**的近似（公式 2.1）：

```
LLR ≈ (1/2) · (φ₁ − φ₀) · (2φ̂ − φ₀ − φ₁) / V(φ̂)
```

  其中 φ̂ 是无约束 MLE，V(φ̂) 是它的方差估计。**多叉分布的例子（直接可用来算我们的 pentanomial）**：

```
φ = Σ aᵢpᵢ ,  φ̂ = Σ aᵢp̂ᵢ ,  V(φ̂) = (1/N)·( −φ̂² + Σ aᵢ²p̂ᵢ )
三项式（胜负平 W/D/L）:       a = (1, 1/2, 0)
五项式（配对交换颜色）:  (p₁..p₅)=(p₂, p₃/2, p₁, p₁/2, p₀),  a = (1, 3/4, 1/2, 1/4, 0)
                        ——注意五项式里 N 是"局数/2"（一个 trial = 一对局）
```

- **还有一条更省事的现成 LLR（`normelo.pdf` 式 4.14，带正则化）**：

```
LLR ≈ (N/2)·log[ (1 + (t̂ₙ − tₙ,₀)²) / (1 + (t̂ₙ − tₙ,₁)²) ]
其中 tₙ 是 normalized t-value（见 3.4），σ 从测试自身估计；
正则化：LLR/N 截断到 [−(tₙ,₁−tₙ,₀)/2, +(tₙ,₁−tₙ,₀)/2]，否则前几对局同结果时 σ̂→0 会炸。
```

- **关键假设**：α、β 是**渐近**保证；早期样本少时 t̂ 会虚高，必须正则化。
- **对我们**：这是"能直接抄成代码"的版本。**注意 fishtest 自己承认他们跑的是 GSPRT 而不是教科书 SPRT**，所以如果他们写"SPRT"，我们该理解为"GSPRT + 近似 LLR"。

**⚠️ 三态判决的坑（必须点明）**：SPRT 只有两个吸收态。你要"更强 / 更弱 / 无差别"三态，有三条路：
1. **两条单侧 SPRT 同时跑**（一条测 H0: Δ≤0 vs H1: Δ≥+δ；一条测 H0: Δ≥0 vs H1: Δ≤−δ），先被触发的赢；
2. 把"无差别"做进 H0：`H0: |Δ| ≤ δ` vs `H1: |Δ| ≥ δ`（连续化后就是 TOST 的序贯版）；
3. 改用**序贯 Bayes 因子**（§3.9），它天然是 `BF>3 / BF<1/3 / 继续` 三态，而且对"反复偷看"免疫。

fishtest 的做法是最省事的第 1 种的变体：`SPRT(elo0, elo1)` 里 `elo0` 通常取 0（"没变强"），`elo1` 取"值得合并的最小增益"，**它没有"更弱"这个独立结论——弱就是 fail**。

---

### 3.4 Normalized Elo：让"多少局"和和棋率解耦【正文】

- **链接**：[Comments on normalized Elo](https://www.cantate.be/Fishtest/normalized_elo_practical.pdf)
- **核心思想**：定义 `tₙ = (µ − 1/2)/σ_pg`（µ = 期望得分，σ_pg = 每局得分标准差），再乘一个常数换成 Elo 量纲：`eₙ = C·tₙ`，`C = 800/ln10 = 347.44`。对"完美平衡的开局"有

```
σ_pg = (1/2)·√(1−d)        d = 和棋率
eₙ ≈ el / √(1−d)           el = 普通 logistic Elo
最坏情况期望局数:  T = D / (eₙ,₁ − eₙ,₀)² ,  D = C²·ln(19)² ≈ 1,046,535
（在 H0/H1 端点上：D' = 9C²ln19/5 ≈ 639,770）
```

  原文表（α=β=0.05，最坏情况）：

| 归一化 Elo 差 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 期望局数 | 1,046,535 | 261,634 | 116,282 | 65,408 | 41,861 | 29,070 |

- **关键假设**：Elo 加性、开局偏置与 Elo 差可分离。
- **对我们**：我们是**无平局**（d=0），所以 `σ_pg = 0.5`、`eₙ = el` 直接可用。**这条最有用的地方是它给了我们一个"期望局数只取决于归一化 Elo 差"的换算器**——不用每次重推。它也顺带说明了一件事：**和棋率高的游戏（国际象棋 d≈0.67）想分辨同样的 Elo 差，需要的局数只有我们这种无平局游戏的 1/3**。我们的"16 局不够"很大一部分是这个原因。

---

### 3.5 配对设计 / Pentanomial：能省多少局，有恒等式【正文】

- **链接**：[The accounting identity](https://www.cantate.be/Fishtest/accounting_identity.pdf) · [vdbergh/pentanomial](https://github.com/vdbergh/pentanomial) · [fishtest issue #348](https://github.com/official-stockfish/fishtest/issues/348)
- **核心思想**：开一对局（同开局、交换颜色），把**这一对**当成一个观测。设 `Xᵢ` 是引擎在开局 i 的第一局得分、`Yᵢ` 是换色后的得分，`Zᵢ=(Xᵢ+Yᵢ)/2`。三项式模型错误地把 2n 局当 i.i.d.；五项式模型用 n 个 `Zᵢ`。
- **量化（这是文献里少见的硬结果）**：

```
Theorem（accounting identity）:  V₃ − V₅ = E(b²) + O(n^{−1/2})

其中 b 是"开局的偏置"（这个局面本身对先手/后手有多有利），
V₃ = 三项式模型给出的每局方差，V₅ = 五项式模型的每局方差。
√E(b²) 就叫开局书的 RMS bias。
```

  换句话说：**配对省下来的方差，正好等于"开局偏置的方差"。** 如果开局没有偏置，配对一点不省；偏置越大，省得越多。

- **实测数字**：
  - `pentanomial` README 的模拟（elo1=10、真值 10、draw_elo=327、book bias=[−90,200]）：三项式 `E[N]=2191`，五项式 `E[N]=1809` → **省 17%**；而且三项式的实际通过率是 0.968 而不是 0.950（"名义 95%，实际 96.8%"，即**高估了方差**）。原文结论：`"The trinomial implementation of the SPRT overshoots the target of 95% pass probability and hence takes 20% longer to complete."`
  - fishtest issue #348 里 van den Bergh 给的解析比值 `(σ²−σ²_book)/(σ²+μ²_book)`，LTC 上 d=0.67、book bias σ_book=0.103、μ_book=0.041 时算出 **0.852**，与实测 LLR 高 18% 吻合。
- **关键假设**：（i）真的一对一配对（同开局/同种子、严格换座位）；（ii）Elo 可加（式 0.1：`µᵢ = s + bᵢ`, `µᵢ° = s − bᵢ`）。
- **对我们**：**必须做**。但要诚实：**κ（方差压掉的比例）必须自己测**，不能照搬 17%。我们的"地图"和"座位"比国际象棋的开局偏置大得多——尤其 27pp 的座位差若随地图变化，κ 会远高于 0.15。测法见 §4.3。
- **⚠️ 一个常见误解**：配对**不能**消掉"座位主效应"。如果座位优势是一个**常数**（每张图上先手都强 355 Elo），那么"交换座位"只是让它在对内抵消，而**两个 arm 都同时吃到两种座位**，所以差的方差里本来就没有座位主效应——这跟配对无关，是因为座位平衡。配对真正消掉的是**「这张图 × 座位」的交互 / 这张图的偏置 bᵢ**。所以：**座位平衡是必需的（design），配对是加分项（variance），两者不要混为一谈。**

---

### 3.6 Elo / BayesElo / Ordo：用模型，别用排名【正文】

- **链接**：[BayesElo（Coulom）](https://www.remi-coulom.fr/Bayesian-Elo/) · [Ordo 手册](https://github.com/michiguel/Ordo) · [Whole-History Rating](https://www.remi-coulom.fr/WHR/WHR.pdf)
- **核心思想**：把"谁赢谁"写成 `P(A wins) = f(elo_B − elo_A − eloAdvantage + eloDraw)` 这样的似然，然后做 MLE（Ordo）或带先验的 MAP（BayesElo）。BayesElo 用 minorization-maximization 迭代。
- **两个模型细节直接对我们有用**：
  - **BayesElo 显式估计先手优势**：`P(WhiteWins) = f(eloBlack − eloWhite − eloAdvantage + eloDraw)`；在 Leo Dijksman 的 WBEC **29610 局**上估出 **`eloAdvantage = 32.8 ± 4`**、`eloDraw = 97.3 ± 2`。→ **这就是"座位当协变量"的现成实现**。
  - **Ordo 把先手当成一个全局参数 W_adv 迭代求解**：`W_e = Σᵢ Expectancy(RWᵢ + W_adv, RBᵢ)`，用"最小化实际总得分与期望总得分的平方误差"来解 `W_adv`，然后再重估 rating（手册 §"Theorem"/式 (9)–(11)）。
- **坑（每条都有出处）**：
  1. **自对弈 Elo 膨胀**。LCZero 官方 FAQ 原文：*"The chart is not calibrated to CCRL or any other common list. It sets 'the first net' to Elo 0, so it is not comparable, **even between different training runs**. ... **Self-play tends to exaggerate gains in Elo** compared to gains when playing other chess engines."* → 我们的"自对弈胜率涨了 X"不能跨 run 比较、也不能外推到比赛对手。
  2. **先验会收缩（这是 feature 不是 bug）**。BayesElo 文档演示：A 对 B 一比零时，Elo 只给 A +41（而不是 Elostat 的 +300）；十比零才给 +169。原文：*"a high rating difference has to be deserved, and requires many more games than with Elostat."* → **少量对局时任何 Elo 数字都应被理解为"被先验拉回 0 的估计"**，别拿它当强度。
  3. **非传递性**。LCZero FAQ 原文：*"One net could be better at short time control and another at long time controls. One could be better at drawing against Stockfish but have too many draws against lower-rated engines, while another may trash lower rated engines and be trashed by Stockfish. One could be worse against other engines, but **beat every other net in head to head**."* 另有专门做非传递性 benchmark 的工作：[Population-dependent agent performance in non-transitive games（Scientific Reports）](https://www.nature.com/articles/s41598-026-55417-9)（**【二手】**：只确认标题与摘要级信息）。→ **单一 Elo 数字不足以描述我们的 AI**，评估至少要分座位、分地图报告。
  4. **Elo 是 logit 尺度**，所以"座位 = 常数偏移"是合法的建模；但**幅度大到 355 Elo 时，"胜率"这个绝对量已经失去意义**（取决于座位配比）。见 §3.7。
- **对我们**：Elo 不是"评估协议"，它是"把一批对局压成一个可比较数"的模型。**我们真正需要的是 A−B 这个系数 + 它的区间**，不是全表排名。

---

### 3.7 把座位（和地图）当协变量：这是 27pp 问题的正解【正文】

- **链接**：[BradleyTerry2 vignette（Turner & Firth）](https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html)
- **核心思想**：标准 Bradley–Terry 是 `logit P(i beats j) = λᵢ − λⱼ`。加协变量就是加项。vignette 里"主场优势"那一节的模型是：

```r
BTm(cbind(home.wins, away.wins), home.team, away.team, data = baseball,
    formula = ~ team + at.home)     # at.home 是 0/1 的"是否主场"
```

  换成我们的语言：`formula = ~ version + seat`，其中 `seat ∈ {0,1}`。**`seat` 的系数就是那 355 Elo**，`version` 的系数就是我们要的 A−B 差，而且它是在**扣掉座位效应之后**估的。

- **等价的三条实现路线**（按实现成本排）：
  1. **McNemar / 条件 logistic**：只用配对的 2 局，看"不一致对"（A 赢 B 输 / B 赢 A 输）的计数，二项检验。零依赖，20 行。
  2. **CMH（Cochran–Mantel–Haenszel）分层**：按地图分层，对每层算 2×2 表再合并。适合"地图只有几张"的情况。
  3. **logistic 回归 / 混合效应**：`logit P = β₀ + β₁·arm + β₂·seat (+ 地图随机截距)`。最灵活，能同时估 seat、map 方差。
- **关键假设**：seat 效应在 logit 尺度上可加（Logistic 回归的标准假设）；配对时层内独立。
- **对我们**：**这是比"配对"更本质的答案**。配对解决的是"怎么把设计做干净"，协变量解决的是"怎么把已知的已知量用上"。**两者叠加**：配对设计（保证 seat 平衡、去掉 map 偏置）+ 回归里加 seat 项（吸收残余不平衡、并给出诊断量）。
- **顺手的一个诊断**：如果 seat 系数真的接近 355 Elo，先做一次**镜像等变性测试**——把局面做座位镜像后喂给网络，看 policy/value 是否等变。不等变就说明是**表示/编码 bug**（项目里已有 `az_t_pos_default_fix.md` 这类线索），而不是游戏本身不平衡。这比跑 3000 局便宜几个数量级。

---

### 3.8 离线指标与实战脱钩：文献怎么说【混合】

- **直接证据不多，但方向很清楚**：
  - **【正文】** LCZero FAQ：`"Self-Elo" is not a true indicator of net strength, but just a general parameter of the "health" of the training run`；以及 `"Substantial amount of testing is required to assess superiority outside of error bars (small sample size problem)"`。
  - **【正文】** KataGo 论文的消融：去掉 ownership/score 辅助头会**明显降低学习效率**（原文 `"removing auxiliary training targets and score maximization resulted in a noticeable drop"`）。但注意——**这是"训练效率"指标，不是"强度预测"指标**；论文没有声称辅助头 loss 能预测 Elo。反而它给的机制解释是：value 头"每局只拿到一个样本"，所以需要辅助头**制造更多样本**（原文：`"The game outcome value target is relatively data-starved ... since all of these targets receive only one new unique sample per entire game"`）。→ **这条对我们极其相关**：如果我们的值网也是"每局一个标签"，那么它的 loss 与强度的关系必然弱。
  - **【二手】** Wang & Emmerich, *Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play*（[Semantic Scholar](https://www.semanticscholar.org/paper/Policy-or-Value-Loss-Function-and-Playing-Strength-Wang-Emmerich/b125c8933d0264b9a103cb8fa80f226f8c9c3cdc)）。标题直指这个问题，但我**没拿到全文**，因此不引其结论。
- **我的判断（明确标为推断，非文献结论）**：离线指标测的是"与 teacher/标签的一致程度"，实战测的是"与对手的对抗结果"。两者只在"teacher 本身是强对抗解、且被测维度就是 teacher 的强项"时才单调相关。你们的先验是**蒸馏**来的，teacher 的上限就是离线指标的上限——所以"离线涨 4.7pp、实战掉 20pp"完全可能，而且**这不是异常，是蒸馏目标与胜率目标不同**。
- **可落地的补救**（不是找更好的离线指标）：
  1. 离线指标**降级为冒烟/回归测试**（查崩坏：NaN、分布塌缩、非法动作率、HOLD 率突变），不参与强度判定；
  2. 用历史 checkpoint 拟合**校准曲线**：x = 离线分，y = 该 checkpoint 的配对实战 Elo，看 R² 与残差。R² < 0.5 就永久禁止用离线分做门禁；
  3. 如果一定要一个便宜的代理，**用"对固定锚点集的 1-ply/短 rollout 胜率"**，而不是先验的 Spearman——前者至少是同一因果链上的量。

---

### 3.9 "用少量对局做可信比较"：理论下界 + 三个可用工具【正文/二手混合】

- **Best-arm identification 的理论下界（【正文】，我读了 JMLR 原文并核对了定理陈述）**：
  [Kaufmann, Cappé, Garivier, *On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models*, JMLR 17 (2016)](https://jmlr.csail.mit.edu/papers/volume17/kaufman16a/kaufman16a.pdf)
  原文（§1）：两臂高斯模型下，δ-PAC 策略的样本复杂度满足 `E_ν[τ] ≃ 2σ²/(µ₁−µ₂)² · log(1/δ)`；LUCB 的上界是 `8σ²/(µ₁−µ₂)² · log(1/δ)`。
  **对我们的意义**：这是"不管你怎么聪明，都至少要这么多局"的硬下界。代入 σ²=0.25、Δ=0.05、δ=0.05：`2·0.25/0.0025·3.0 = 600 局`。**SPRT 的 530 局（在 H1 上）已经贴着这个下界了**——所以"再聪明一点就能用 50 局判定 5pp"是做不到的，别在这上面花时间。
- **序贯 Bayes 因子（【二手】）**：[Schönbrodt, Wagenmakers, Zehetleitner, Perugini (2017), *Sequential hypothesis testing with Bayes factors*, Psychological Methods 22(2):322-339](https://pubmed.ncbi.nlm.nih.gov/26651986/)。核心卖点：三态输出（`BF₁₀ > 3` / `< 1/3` / 继续），且**对"随时停下看结果"免疫**（Bayes 因子是 anytime-valid 的，不需要 α-spending）。**我只读到 PubMed 摘要，未读全文**，所以只推荐其**思想**，不引其具体数值。
- **Thompson sampling / 多臂**：适合"候选很多、要边打边选"。**两臂二值时不如 SPRT**（它是 regret 最优，不是 sample 最优）。我们的场景是两臂，**不要用**。
- **Bayesian A/B（Beta-Binomial）**：把 `P(A>B)` 直接算出来，好处是"结论可读"（"A 优于 B 的概率 93%"），坏处是**它不告诉你该停**。可以和 SPRT 并联：SPRT 管停，Beta 后验管汇报。

**结论**：两臂 → **SPRT / GSPRT**；要三态且想随时停 → **序贯 Bayes 因子**；多候选 → **全局 Bradley–Terry + 序贯**。

---

### 3.10 截断 / 提前裁判（truncated / adjudicated games）

- **【正文】** KataGo 的做法值得抄：它**在自对弈里不用 resignation**，改成"当败方连续 5 手的 MCTS 胜率估计都 <5% 时，把 visits 从 N 降到 `λn+(1−λ)N`"，并且给这些训练样本降权到 `0.1+0.9λ`。理由原文：`"Playing out the full game allows the final determination of the ownership of all areas and the final score difference ... avoiding resignation also reduces the number of positions that are incorrectly recorded"`。→ **截断省算力，但代价是"错误标注"**。
- **【正文】** 我们有一个 KataGo 没有的优势：**终局 HP 差是一个连续量**。用"打到 T 回合、用 HP 差当结果"可以把每局的 1 bit 变成连续观测，方差大幅下降（二项 σ²=0.25 vs 标准化连续量的 σ² 可能只有 0.02–0.1）。**但**：HP 差与胜率**不单调等价**（能赢但赢得难看 vs 冒险赢得漂亮），所以**必须双报告**：主判决用胜率/得分，辅助报告 HP 差与其 CI。
- **【未找到】** 我**没有找到**"截断对胜率估计偏差"的量化文献（国际象棋界的 adjudication 讨论基本都在论坛里，没有可引的论文）。这一条在 §5 里明确标为未找到，不要当成有文献背书。

---

### 3.11 KataGo 的配方（可以直接抄的清单）【正文】

来源：[Accelerating Self-Play Learning in Go (arXiv:1902.10565)](https://arxiv.org/abs/1902.10565)（我读的是 PDF 全文）+ [SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) + [gatekeeper1.cfg](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/gatekeeper1.cfg)

| 配方 | 具体数字（原文） | 我们能抄吗 |
|---|---|---|
| **Playout/visit cap oscillation** | 只用 `p=0.25` 的回合跑满 `N` 次 playout，其余回合跑 `n≪N`；只有跑满 N 的回合进训练。`(N,n)` 从 `(600,100)` 退火到 `(900,150)`、`(1200,200)` | **能**，但我们不是训练而是**评估**：评估时应该**固定 cap**，不要振荡（振荡是为了造数据） |
| **辅助头：ownership（每点归属）+ score（目差）** | "greatly improve the learning efficiency" | **能**，而且是"值网样本饥饿"的正解 |
| **辅助头：opponent policy（对手策略预测）** | 有独立增益 | 可选 |
| **Forced playouts + target pruning** | `n_forced(c) = sqrt(2·P(c)·ΣN(c'))` | 训练用 |
| **Global pooling bias** | 全局池化后回注到每个空间位置 | 能 |
| **Progressive net sizing** | 6b96 → 10b128 → 15b192，不迁移参数，直接并行训 | 能 |
| **Gatekeeper 极松** | `numGamesPerGating = 200`；**"candidate nets must win at least 100 out of 200"**（就是 ≥50%）；gating 用 `maxVisits=150/200`（比自对弈低） | **能，而且这条最反直觉也最值钱**：只用来**挡灾难性退化**，不用来确认进步 |
| **评估时用全局 Bayesian MLE Elo** | `"Maximum-likelihood relative Elo ratings were computed using the final set of approximately 50000 games"` | **能**：把所有历史对局一起拟合，而不是每次只做 A/B |
| **对手按 p(1−p) 采样** | `"choosing them with probability roughly proportional to the predicted variance p(1−p) of the game result"` | **能，且这是我们最该抄的一条**：把算力花在"胜负最不确定"的对局上 |

**为什么"gate 只要求 100/200"是对的**：200 局 50% 门槛下，一个真实弱 5pp 的网仍有约 20% 概率通过，一个真实强 5pp 的网也有约 20% 概率被拒。但只要**假阴性的代价只是"晚一代"**（下一代会再被测试），而假阳性的代价是"用坏网产一代数据"，这个不平衡的损失矩阵下**松门禁是最优的**。反过来，**如果我们用 gate 来做"哪个版本更强"的学术结论，200 局就远远不够**——这两件事必须分开。

---

### 3.12 Leela Zero / AlphaGo Zero 的 gate 配方【正文 / 二手】

- **【正文】** [leela-zero#1054 "RFC: Promotion Strategy"](https://github.com/leela-zero/leela-zero/issues/1054) 里，作者把 LZ 当时的做法写得很清楚：

  > *"We want 95% confidence that the new network is better, so that's **55% at 400 games**. But we don't promote based on that directly. We promote based on SPRT pass ..."*
  > *"... in order to reach 95% confidence on a winrate of 51% (or worse 49%!) you need about **9000 games**. So... that's probably too many."*

  **核对**：`n = (1.96·0.5/0.01)² = 9604` ✓（单臂胜率 ±1pp 的 95% 区间）。这条换算极其有用：**分辨 1pp 要 ~9600 局，5pp 大约 9600/25 ≈ 384 局**（1/Δ² 关系）——与我们 §4 的 SPRT 数字同量级。
- **【正文】** 同一 issue 里 gcp 的回复点出了 SPRT 的真实含义：*"it's about being more likely that it's 35 Elo better than that it's worse"*（即 elo0/elo1 是"两个假设"，不是"置信度"）。
- **【正文】** [leela-zero#676 "Win rate for new networks between 50% and 55%"](https://github.com/leela-zero/leela-zero/issues/676)（22 条评论，我读了正文+前 8 条）里有三个可直接抄的东西：
  1. **等价换算**：`"55% at 400 games makes sure that the probability is larger than 95% that the true win probability is >50%"`；扩展版是 `"at 1000 games, a win rate of 53.2% would also ensure 95% confidence"`。→ **`(局数, 阈值)` 是一对可以互换的参数，不是一个固定答案**。我按 `n=(z·0.5/δ)²` 复核：1000 局、95% 单侧对应 ≈52.6%，他们说 53.2%（更保守），量级一致。
  2. **分辨率的价格必须显式拒绝**：gcp 原话 *"we can go on for a long time playing **250k games each time to find 1 Elo improvement**. I do not think this is a good use of resources."* → **"能分辨"不等于"值得分辨"；先定"最小值得为之改代码的效应"，再算局数，而不是反过来。**
  3. **⚠️ 多重比较 / winner's curse**：`"if on average you had 20 networks being tested consecutively all more or less equal with the current best, even 95% confidence is not really enough"`。→ **如果你一天测 20 个 checkpoint，按 95% 阈值挑"最好的那个"，你挑到的多半是噪声。** 这对我们尤其致命：离线指标 + 少量对局的组合筛选，正是 winner's curse 的标准配方。
- **【二手】** AlphaGo Zero 原文的 "400 局 / 55% 门槛" 是社区广泛引用的，但**我这次没能访问 nature.com / arXiv 原文**（见 §5），所以不引它作为一手证据；上一条 LZ 的 400/55% 是**一手**。

---

## 4. 落地草案：一个具体、可执行的评估协议

### 4.1 先定估计量（这一步比统计方法重要）

**主估计量**：`Δ := p − 0.5`，其中 `p` = 版本 A 对版本 B 在**评估赛制**下的得分率（我们无平局，所以就是胜率）。
**报告形式**：`Δ̂` + 95% CI，以及 SPRT 的判决（更强 / 无证据 / 更弱）。

三个必须写死的口径：
1. **座位配比**：主估计量定义在"座位 50/50"上。如果比赛赛制不是 50/50，必须**另报一个按赛制配比加权**的版本。否则"胜率 72%"这种数字没有意义。
2. **地图配比**：同上，按比赛地图池的分布加权；同时**分地图报告**（这是非传递性的最低成本监控）。
3. **HOLD 不是异常**：HOLD（不动）是合法且经常正确的决策，**不能**把它计入"非法/退化"指标，也不能用"HOLD 率下降"当变强的证据。评估协议里对 HOLD 的唯一用法是**监控分布漂移**（HOLD 率突变 → 查 bug），不是判强度。

### 4.2 赛制（配对设计）

```
每个 (地图 m, 对手 o) 组合跑 2 局：
   局 1：A 坐 P1，B 坐 P0
   局 2：A 坐 P0，B 坐 P1
   两局使用同一个 seed / 同一个初始局面（共同随机数 CRN）

座位由 pair 内的序号决定（不是随机分配）→ 保证每个 pair 内严格平衡
地图按赛制分布抽样，但**成对使用**（同一 pair 内地图相同）
```

**为什么这样能省方差**：`Var(配对差) = Var(A) + Var(B) − 2Cov(A,B)`（[Wikipedia: Variance reduction / Common Random Numbers](https://en.wikipedia.org/wiki/Variance_reduction) 有这条公式的推导）。Cov 来自"同一张图/同一个 seed 使两局共享同一个偏置 b"，而 §3.5 的 accounting identity 说 `V₃−V₅ = E(b²)`——**配对省下的正好是偏置的方差**。

**⚠️ 确定性 AI 的陷阱**：如果我们的 AI 在固定 seed 下是确定性的，那么"同 seed 换座位"这两局可能高度相关（好事，方差小）**也可能高度反相关或完全决定性**（坏事：这对局只贡献 1 bit 甚至 0 bit）。**这必须实测**，见 §4.3 的 κ 估计。若发现"同一 pair 两局结果几乎总是相同"，说明 seed 粒度太粗，应改用"多 seed × 每 seed 少局"。

### 4.3 统计量（三个，都要算）

**(1) 主判决：GSPRT on paired outcomes**

把每个 pair 的结果记成五项式的一档（我们用"该 pair 内 A 的得分/2"，取值 0, 1/4, 2/4, 3/4, 1）。按 §3.3 的公式：

```
φ = Σ aᵢp̂ᵢ ,  a = (1, 3/4, 1/2, 1/4, 0)
V(φ̂) = (1/N)·(−φ̂² + Σ aᵢ²p̂ᵢ) ,   N = pair 数
LLR ≈ (1/2)·(φ₁−φ₀)·(2φ̂−φ₀−φ₁) / V(φ̂) ，并截断 |LLR/N| ≤ |φ₁−φ₀|/2
每完成一个 pair 更新一次；LLR ≥ ln19 → 停，判"更强"；LLR ≤ −ln19 → 停，判"更弱"
```

**(2) 主报告：logistic 回归**

```
logit P(A 赢这一局) = β₀ + β₁·arm + β₂·seat + β₃·map   (map 用固定效应或随机截距)
报告 β₁（= 扣掉座位/地图后的 A−B 差）+ 95% CI
```

`β₂` 就是座位效应，**每次都要打印**——它是免费的诊断量。

**(3) 必须记录的三个数（用来给下一次定预算）**

```
κ̂ = 1 − Var(paired diff) / Var(unpaired diff)     # 配对效率，决定下一次乘 (1−κ)
ρ̂ = 座位效应（logit 系数，或直接换成 Elo）
Var(map 效应)                                      # 地图间胜率的方差
```

拿到 κ̂ 之后，把 §4.5 表里的局数乘 `(1−κ̂)`。

### 4.4 阈值与停止规则

| 参数 | 建议值 | 理由 |
|---|---|---|
| α, β | 0.05, 0.05 | fishtest 默认；`ln19 = 2.944` |
| H0 | Δ = 0（"没变强"） | |
| H1 | Δ = 0.025（"A 对 B 得分率 52.5%，即强 5pp"） | 这是"值得为之改代码"的最小效应；按 §4.5 的预算调整 |
| N_max（强制截断） | 由 4 小时 / 10 小时预算反算，**必须先定** | 防止"无差别"的对局无限跑 |
| 截断后的报告 | 用固定样本 CI，并**明说 α 已膨胀** | 序贯停止后不能用固定样本 CI 当 95% 区间；fishtest 用 Brownian 近似重估（见 [fishtest Mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html) 指向的 `brownian_approximation.pdf`） |
| 判决之外 | **永远同时报 `Δ̂` 和 CI** | "没证据" ≠ "没差别" |

**三态实现建议**（最省事）：跑**一个双边界 SPRT**，`LLR ≥ +2.944 → 更强`，`LLR ≤ −2.944 → 更弱`，`到 N_max 还没出界 → 无证据`。这就是三态，且 α/β 各 5% 只在两端成立（中间那档是"未决"，不是"等价"，这点要在文档里写清楚）。

### 4.5 "4 小时内判定 5pp 需要多少局" —— 量化例子

先给**术语陷阱**的明确定义，否则差 4 倍：

> **Δ := p − 0.5**，`p` = A 对 B 的得分率（无平局时 = 胜率）。
> "A 对 B 胜率 52.5% / 47.5%" → **Δ = 0.025**（≈17 Elo）
> "A 对 B 胜率 55% / 45%" → **Δ = 0.05**（≈35 Elo）
> 换算：`Elo = 400·log₁₀(p/(1−p))`；在 p≈0.5 附近 **1pp ≈ 3.5 Elo**。

**SPRT 期望局数（α=β=0.05，无平局 σ²=0.25）**：

| p（A 得分率） | Δ | ≈Elo | 真值=H1 时 E[N] | **最坏情况 E[N]** | 配对 κ=0.3 后（最坏） |
|---|---|---|---|---|---|
| 0.510 | 0.010 | 6.9 | 13,250 | 21,674 | 15,172 |
| 0.520 | 0.020 | 13.9 | 3,312 | 5,419 | 3,793 |
| **0.525** | **0.025** | **17.4** | **2,120** | **3,468** | **2,428** |
| 0.530 | 0.030 | 20.9 | 1,472 | 2,408 | 1,686 |
| **0.550** | **0.050** | **34.9** | **530** | **867** | **607** |
| 0.600 | 0.100 | 70.4 | 132 | 217 | 152 |

**4 小时内能跑多少局**（16 workers、单局 T 秒 → `N = 16·4·3600/T`）：

| 单局耗时 T | 4h 可跑局数 | 能判定的 Δ（最坏情况，κ=0.3） | 折合 Elo |
|---|---|---|---|
| 30 s | 7,680 | 0.014 | ~10 |
| 1 min | 3,840 | 0.020 | ~14 |
| **2 min** | **1,920** | **0.028** | **~20** |
| **5 min** | **768** | **0.044** | **~31** |
| 10 min | 384 | 0.063 | ~44 |
| 20 min | 192 | 0.089 | ~62 |

**要判定 Δ=0.025（5pp），需要多少小时**（κ=0.3，最坏情况 2428 局）：

| 单局耗时 T | 需要时长（16 workers） |
|---|---|
| 1 min | 2.5 h |
| **2 min** | **5.1 h** |
| 5 min | 12.6 h |
| 10 min | 25.3 h |
| 20 min | 50.6 h |

**读这张表的三条结论**：
1. 你们说的"每局几分钟到几十分钟" ⇒ **4 小时只能判 20–60 Elo 的差异**（Δ≈0.03–0.09）。想判 5pp（Δ=0.025 / 17 Elo）在 5 min/局 的价位上需要 **12 小时以上**，而且这还是**配对已生效**的乐观情形。
2. **所以"4 小时内判 5pp"在当前对局成本下基本不成立**，除非：(a) 用更便宜的对局（更小地图/更少回合/更浅搜索）做**筛选**，只在最后阶段用完整赛制确认；或 (b) 接受"只筛掉 >20 Elo 的回归"，把小效应交给**多次实验的累积证据**（见下条）。
3. **务实的做法是"分层预算"**：
   - **冒烟层（~15 min）**：离线指标 + 24–48 局配对 → 只能查**灾难性崩坏**（24 局 ≈ 只能分辨 Δ>25pp，48 局 ≈ Δ>18pp）；
   - **筛选层（~2–4 h）**：300–800 局配对 SPRT（H1 设在 Δ=0.05）→ 判"强 10pp / 弱 / 未决"；
   - **确认层（1–2 夜）**：2000–3000 局 → 判 Δ=0.025。
   只有"要写进结论/要上线"的版本才进确认层。

### 4.6 三天内能做的改造（按性价比排序）

| # | 改造 | 成本 | 收益 |
|---|---|---|---|
| 1 | **座位在 pair 内严格交替**（不是随机分配） | 20 行 | 立刻消掉小样本下的座位不平衡（n=16 时随机分配可能有 6:10 的偏斜 = 7pp 假信号） |
| 2 | **同图两局、交换座位**（配对） | 30 行 | 消掉 `E(b²)` 那部分方差；κ 待测 |
| 3 | **换 SPRT 停止规则**（§4.3 式） | 60 行 | 期望省 ~50% 局数 |
| 4 | **回归里加 `seat` 项并每次打印** | 20 行 | 免费拿到座位诊断；不顺眼就去查镜像等变性 bug |
| 5 | **记录 HP 差**作为第二估计量 | 10 行 | 方差更低，且能看出"赢了但变差了" |
| 6 | **对手按 p(1−p) 采样**（KataGo 做法） | 30 行 | 同样的局数下信息量更高 |
| 7 | **把所有历史对局丢进一个 Bradley–Terry 拟合** | 半天 | 每个候选都能复用历史局，而不是每次从零开始 |

### 4.7 不要做的事（每条都有出处）

1. **不要用离线指标 gate 强度**（§3.8）。它只能查崩坏。
2. **不要用自对弈 Elo 跨 run 比较**（LCZero FAQ 原文："not comparable, even between different training runs"）。
3. **不要在 n<400 时相信任何 <5pp 的差异**（§3.1 的公式）。
4. **不要在 SPRT 停止后报朴素 95% CI**（序贯偏差）。
5. **不要用"某一个 Elo 数字"描述我们的 AI**（非传递性；LCZero FAQ 原文："One could be worse against other engines, but beat every other net in head to head"）。
6. **不要为了让 SPRT 早点结束而临时改 elo1**（这是 p-hacking 的序贯版本）。
7. **不要把 HOLD 率高当成退化**（§4.1）。
8. **不要"测 20 个 checkpoint 然后挑胜率最高的那个"**（winner's curse，见 §3.12；Leela Zero 的开发者明确点过这个坑）。要么**预注册单一候选**，要么对"挑选"本身做多重比较校正。

---

## 5. 读到正文 vs 仍是二手 vs 未找到

### 5.1 读到正文（我实际解析了原文内容）

| 来源 | 读到的内容 |
|---|---|
| [sprta.pdf](https://www.cantate.be/Fishtest/sprta.pdf) | Theorem 1.1 全部公式、对称情形化简、h=±1 与 h=0 的 T 值 |
| [GSPRT_approximation.pdf](https://www.cantate.be/Fishtest/GSPRT_approximation.pdf) | GSPRT 定义、LLR 近似式 (2.1)、多叉分布例子（含五项式系数 a=(1,¾,½,¼,0)） |
| [normalized_elo_practical.pdf](https://www.cantate.be/Fishtest/normalized_elo_practical.pdf) | σ_pg=(½)√(1−d)、C=800/ln10、D=1,046,535、期望局数表、LLR 式 (4.14)(4.15) 与正则化规则 |
| [accounting_identity.pdf](https://www.cantate.be/Fishtest/accounting_identity.pdf) | Theorem 0.1：V₃−V₅ = E(b²) 的完整证明 |
| [fishtest Mathematics 文档](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html) | GSPRT / 五项式 / normalized Elo / SPSA / χ² 质量控制的描述 |
| [fishtest issue #348](https://github.com/official-stockfish/fishtest/issues/348) | 全部评论（经 GitHub API 取 JSON）：18% LLR、0.852 比值、overshoot 修正实测无改善 |
| [vdbergh/pentanomial](https://github.com/vdbergh/pentanomial) | README 全文：五/三项式模拟对比 2191 vs 1809、"20% longer"、真实 probs5 |
| [KataGo 论文 PDF (1902.10565)](https://arxiv.org/abs/1902.10565) | 全文 38 页：§4 辅助头、§5.3 forced playouts、§5.5 tree reuse、§6.1 playout cap oscillation、§6.2.1 不 resign、§6.2.4 gating、§7.1 评估方法 |
| [KataGo SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md) | 全文（5 步闭环、gating 可选） |
| [KataGo gatekeeper1.cfg / 2a.cfg](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/gatekeeper1.cfg) | `numGamesPerGating = 200`、`maxVisits = 150/200`、resign 阈值 |
| [BayesElo 官网](https://www.remi-coulom.fr/Bayesian-Elo/) | 模型公式、eloAdvantage=32.8±4 / eloDraw=97.3±2（29610 局）、"10-0 ≠ 1-0" 的演示 |
| [Ordo MANUAL.pdf](https://github.com/michiguel/Ordo) | 式 (9)(10)(11)：W_adv 迭代估计 |
| [LZ issue #1054](https://github.com/leela-zero/leela-zero/issues/1054) | issue 正文 + 全部 3 条评论 |
| [LZ issue #676](https://github.com/leela-zero/leela-zero/issues/676) | issue 正文 + 前 8 条评论（400局55% / 1000局53.2% / "250k games for 1 Elo" / 多重比较） |
| [LCZero FAQ](https://lczero.org/dev/wiki/faq/) | "Self-Elo is not a true indicator"、"not comparable even between different runs"、"beat every other net in head to head" |
| [Wikipedia: SPRT](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test) | 边界 a≈log(β/(1−α))、b≈log((1−β)/α)、LLR 累积定义 |
| [Wikipedia: Variance reduction](https://en.wikipedia.org/wiki/Variance_reduction) | CRN 一节，Var[Z(n)] = (Var₁+Var₂−2Cov)/n |
| [BradleyTerry2 vignette](https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html) | `~ team + at.home` 的主场优势模型 |
| [Kaufmann JMLR 2016](https://jmlr.csail.mit.edu/papers/volume17/kaufman16a/kaufman16a.pdf) | §1 的 2σ²/(µ₁−µ₂)²·log(1/δ) 与 LUCB 的 8σ²/(µ₁−µ₂)²·log(1/δ) |

### 5.2 仍是二手（只确认了标题/编号/摘要，未读正文）

| 来源 | 状态 |
|---|---|
| AlphaGo Zero（Nature 2017）的 "400 局 / 55%" gate | **未读到原文**；nature.com 抓取返回 425 字节（拦截页）。同一条实践在 LZ issue #1054 里有**一手**表述，我引的是后者 |
| Wang & Emmerich, *Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play* | Semantic Scholar 429（限流），未取到 PDF；Semantic Scholar API 也 429。**不引其结论** |
| Schönbrodt et al. 2017（序贯 Bayes 因子） | PubMed 只给摘要（"Cookies must be enabled"）；JMLR/arXiv 无此篇。**只推荐思想，不引数值** |
| Nature Sci Rep: *Population-dependent agent performance in non-transitive games* | 页面返回 303 字节（拦截）。只确认标题与期刊 |
| Wald (1945) 原始 SPRT 论文 | 未获取。本文所有公式均引 van den Bergh 的整理版（`sprta.pdf`），那是 fishtest 的实际实现依据 |
| TrueSkill（NIPS 2006）PDF | 我下载了 PDF 但**没有细读**，故正文未引用其结论（只在一句话里提到"Bayesian skill rating"这一族） |
| Glicko-2（Glickman）PDF | 同上，下载未细读，正文未引用 |
| Coulom, *Whole-History Rating*（WHR） | §3.6 把它列在"延伸阅读"里，但**我这次没有读它**；正文中关于 BayesElo/Ordo 的所有断言都来自 §5.1 列出的两个来源，与 WHR 无关 |

### 5.3 明确"未找到"

1. **"截断/提前裁判对胜率估计的偏差有多大"** —— 没有找到可引的量化文献。国际象棋界的 adjudication 讨论集中在论坛（talkchess / Cute Chess 文档），没有论文级证据。**§3.10 里我明确标了这是"未找到"，不是"有文献支持"。**
2. **"离线指标与实战强度的相关性"的系统性研究** —— 只找到 Wang & Emmerich 的标题，没有读到正文；也没找到综述。**§3.8 里"我的判断"一段是推断，不是文献结论，我已标注。**
3. **塔防 / 非对称六边形网格游戏的强度评估文献** —— 完全没有。所有可抄的配方都来自国际象棋（fishtest）、围棋（KataGo/LZ）、以及通用序贯分析。
4. **fishtest 的"默认 SPRT 边界"具体值** —— `Creating my first test` 页面只说"when in doubt, stick with the standard test parameters"，没有在该页给出数值；`sprt_calc` 页面（`https://tests.stockfishchess.org/sprt_calc`）在本会话 DNS 不可达，未能验证。**所以本文所有 α=β=0.05 的例子都是按 `ln19` 自算的，不是抄默认值。**

### 5.4 访问受限的域名（本会话）

`arxiv.org`、`export.arxiv.org`、`ar5iv.labs.arxiv.org`、`chessprogramming.org`、`tests.stockfishchess.org`、`pubmed.ncbi.nlm.nih.gov`、`nature.com`、`jstatsoft.org`、`api.semanticscholar.org`（429）、`rlg.mlanctot.info`、`deepwiki.com`、`jmlr.csail.mit.edu`（web_fetch 失败，但 **pwsh 直连成功**）。
→ 结论：**`web_fetch` 全灭，`Invoke-WebRequest` 能拿到绝大多数**（含 arXiv PDF、fishtest PDF、GitHub API、Wikipedia、CRAN）。下次调研直接走 pwsh + PyMuPDF。

---

## 6. 引用清单（全部为可点击 URL）

**序贯检验 / 统计**
1. https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
2. https://www.cantate.be/Fishtest/sprta.pdf
3. https://www.cantate.be/Fishtest/GSPRT_approximation.pdf
4. https://www.cantate.be/Fishtest/normalized_elo_practical.pdf
5. https://www.cantate.be/Fishtest/accounting_identity.pdf
6. https://www.cantate.be/Fishtest/brownian_approximation.pdf
7. https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html
8. https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html
9. https://github.com/official-stockfish/fishtest/issues/348
10. https://github.com/vdbergh/pentanomial
11. https://en.wikipedia.org/wiki/Variance_reduction
12. https://en.wikipedia.org/wiki/Power_of_a_test
13. https://en.wikipedia.org/wiki/Cochran%E2%80%93Mantel%E2%80%93Haenszel_statistics
14. https://jmlr.csail.mit.edu/papers/volume17/kaufman16a/kaufman16a.pdf
15. https://pubmed.ncbi.nlm.nih.gov/26651986/

**评分系统 / 协变量**
16. https://www.remi-coulom.fr/Bayesian-Elo/
17. https://www.remi-coulom.fr/WHR/WHR.pdf
18. https://github.com/michiguel/Ordo
19. https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html
20. https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

**围棋 / 国际象棋 AI 的评估实践**
21. https://arxiv.org/abs/1902.10565 （KataGo, *Accelerating Self-Play Learning in Go*）
22. https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md
23. https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/gatekeeper1.cfg
24. https://github.com/leela-zero/leela-zero/issues/1054
25. https://github.com/leela-zero/leela-zero/issues/676
26. https://lczero.org/dev/wiki/faq/

**其他**
27. https://www.semanticscholar.org/paper/Policy-or-Value-Loss-Function-and-Playing-Strength-Wang-Emmerich/b125c8933d0264b9a103cb8fa80f226f8c9c3cdc （仅标题）
28. https://www.nature.com/articles/s41598-026-55417-9 （仅标题）

---

## 附：本文所有数字的计算方式（可复现）

```
Elo(p)            = 400*log10(p/(1-p))
Δ                 = p - 0.5
SPRT E[N] @H1     = 5.300 * 0.25 / Δ²          (σ²=0.25, α=β=0.05)
SPRT E[N] worst   = 8.670 * 0.25 / Δ²
固定样本 (α=.05双侧, power 80%)  = 7.849 * 0.25 / Δ²
固定样本 (α=.05单侧, power 95%)  = 10.82 * 0.25 / Δ²
配对效率          = 乘以 (1-κ)
4h 局数           = 16 workers * 14400 s / T
可分辨 Δ (给定 N)  = sqrt(8.670*0.25*(1-κ)/N)
```
其中 `5.300 = T(h=1) = 2b(1−e^{−b})/(1+e^{−b})`、`8.670 = T(h=0) = b²`、`b = ln19`，均来自 `sprta.pdf` Theorem 1.1。
