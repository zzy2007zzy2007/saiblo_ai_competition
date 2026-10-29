# 先验/候选动作问题：根因与修复方案（只读研究，2026-10-29）

> 工作目录 `D:\2026智能体大赛_新2_distill`。**只读研究**：没有跑对局、没有训练、
> 没有 kill 进程、没有改 `code/my_ai/**`。全部结论来自现有文档 + 代码逐行核对 + 历史实验日志。
>
> 上游事实（直接引用，不重新推导）：
> * `docs/why_effective_candidates_collapse_20261028.md`（1518 个真实决策点：78.0% 被钉成 HOLD；
>   `n_cand==1` 82.3%；26.8% 的回合候选集根本没有 HOLD；升级从不出现在候选）
> * `docs/always_hold_ab_20261029.md`、`docs/nonhold_inject_ab_20261029.md`（补 HOLD +12.5pp；
>   注入非 HOLD 外部动作 12.5%→45.8%，配对 p=0.039 ⇒ **评估器能用好候选，是候选生成不行**）

---

## 0. 一句话结论

1. **A（位置网"效果一般"的根因）**：不是"网络/数据彻底不行"，而是**下游管线把它饿死了**——
   78% 的决策被 `mc_quiet` 仲裁钉成 HOLD（类 23），位置网只在剩下 ~22% 的回合被真正用到；
   而那 22% 里，建塔通道**本来就没有可学信号**（`Spearman +0.046 ≈ 随机`，只有闪电通道有真信号 +0.738）。
   所以"离线 recall 再高也带不到实战"。
2. **B（统一搜索当时差在哪）**：原设计是**跨类的 UCB bandit（Q[c] + 探索项 + 游标扫位置）**，
   实现把它退化成了**"类网 softmax 一次性分预算 + 每个原子动作 1 次 rollout + 复评 top-8"**——
   bandit 的 `Q[c]` 更新、探索项、游标全部没做；而且它**换掉了强评估器**（flat 的 N×h rollout）
   换成"1 样本挑候选"的弱评估。实测 **0W/42L** 输给 flat+cells4。
3. **C（统一候选生成能否根除缺 HOLD/缺升级）**：**能**。HOLD 是 action_map 的第 23 通道，天然存在；
   升级（`OperationType(12)`）不是 action_map 的通道，是"塔意图通道解码到已占格"的产物，
   所以必须**显式枚举塔×升级树**（`unified_action_search.enumerate_by_channel` 已经这么写了），
   不能靠"跨类 argmax"自然产出。最小改法是：**复用 `enumerate_by_channel` 当候选生成器，
   喂给现有的 flat 强评估器**，默认关、零影响。

---

## A. 位置网重训"效果一般"到底卡在哪

### A.1 当时的训练目标 / 数据 / 评估指标（`docs/prior_repair_20261023.md`、`docs/day_summary_20261027.md`）

| 项 | 值 | 出处 |
|---|---|---|
| 训练目标 | ① softmax CE（T=0.5，软标签）② z-score（Pearson）相关损失 ③ 混合 CE+z（λ=1/3）④ 硬 one-hot（教师 argmax） | `train_prior_v2.py:109-129`（CE）、`train_prior_z.py:66-93`（z/CE/硬 one-hot） |
| 教师 | 值网 rollout 排序（`g*.npz`）、MCTS 访问平均（`m*.npz`） | `prior_repair_20261023.md:34-36`、`:173-176` |
| 数据量 | 40 → 80 局（≈3.2 万局面） | `prior_ceiling_20261026.md:108` |
| 评估指标 | recall@1/@4/@24（按动作类型分项：建塔/闪电/降级…） | `train_prior_z.py:96-114`、`day_summary_20261027.md:79-88` |
| 训练范围 | 只动 `action_map_conv`（可训练 1,560）或解冻主干（544,992） | `train_prior_z.py:151-153`、`prior_repair_20261023.md:379-383` |

**关键核对（用户要求的数字）**：`docs/day_summary_20261027.md:86-90`：
`posnet_A` 总体 @24 = 81.3%、建塔 @1 = **14.0%**；`hard_ce_w3` 总体 @24 = **86.2%**、建塔 @1 = **10.5%**。
⇒ **"训练提升了整体排序（@24 81→86）但把建塔 @1 从 14% 打到 10.5%"**——确认成立（81.3→86.2、14.0→10.5）。

### A.2 为什么"离线变好、实战不涨"（证据链）

1. **代理指标三次与棋力脱钩**（`prior_repair_20261023.md:16-18`、`:212-216`）：Spearman 从 −0.11 → +0.354 → +0.428，
   但 k=24 棋力始终在 44–53% 噪声带；`prior_ceiling_20261026.md:235-248` 更是**反向**：
   离线 recall 越高（hybrid 最好）实战越差（28.1%）。
2. **建塔通道本来就没有信号**（`findings_flat_rollout_vs_tree.md:140-147`）：修正 `[ch,y,x]→[ch,x,y]` 转置 bug 后，
   建塔位置图 vs 1-ply 优势的 Spearman 只有 **+0.046**（≈ 随机），闪电通道 **+0.738**（唯一强信号）。
3. **位置先验本身"够覆盖"**（`prior_ceiling_20261026.md:8-27` 修正版）：亚军的建塔 @24 = **74.9% ≈ 随机 75.0%**、
   闪电 @24 = **67.3% vs 随机 8.9%**；降级支撑集只有 2 个、无法评价。⇒ 建塔"能学到的就这么多"。

### A.3 关键判断：是"网络/数据不行"还是"下游管线没给空间"？

**主因是下游管线（pin 塌缩）没给位置网发挥空间；网络/数据是次因。**

- **主因（结构性）**：`why_effective_candidates_collapse_20261028.md:0` 实测 1518 个真实决策点，
  `mc_quiet` 仲裁把 **78.0%** 钉成 HOLD（类 23）、`n_cand==1` 占 82.3%。
  位置网只在建塔（6.7%）/闪电（3.8%）/降级（11.5%）这些**非 HOLD**回合被真正用于抽位置。
  ⇒ 即便离线 recall=100%，它对那 78% 的 HOLD 决策也**零影响**。
  （旁证：`always_hold_ab_20261029.md:178` 实测 `added = 2089/7800 = 26.8%` 的回合候选里根本没有 HOLD。）
- **次因（信号层面）**：非 HOLD 回合里，位置网唯一能发挥的建塔通道**没有可学信号**
  （Spearman +0.046 ≈ 随机），只有闪电（3.8% 的回合）有真信号。
- **结论**：位置网"效果一般"的**根因是"78% HOLD 钉死"让它大部分时间根本没上场**；
  "重训不涨"的直接原因是**教师（值网排序）与"赢"不对齐 + 建塔通道无信号**。
  这两条叠加，解释了"离线 recall 再好也带不到实战"（`prior_ceiling_20261026.md:145-150` 的因果链）。

---

## B. 统一搜索：原始设计 vs 当时实现，差在哪

### B.1 原始设计的关键点（`docs/idea_unified_bandit_search.md`）

| 关键点 | 内容 | 出处 |
|---|---|---|
| 类 bandit | 维护 `Q[c]`（类的 rollout 分数）+ `cursor[c]`（该类位置榜已用位数）；每轮按 `Q[c]` 采类 → 取该类下一个位置 → rollout → 更新 `Q[c]` | `:21-29` |
| 探索项 | `Q[c] + c·√(log N / n_c)`，`c ≈ 1σ`（纯 Q 采样会过早收敛） | `:87-90` |
| 两阶段预算 | 生成阶段每个候选 1 次 rollout；复评阶段 top-k 补到 ≥30 样本（把选错率从 ~42% 降到 ~5%） | `:66-68`、`:62-64` |
| 多动作组 | 分数按动作类统计，"已用过"键 = (类1,类2,类3)；位置先验 = 三通道图相乘；三头**独立采样**（能表达"降级+建塔"） | `:31-35`、`:94-97` |
| 明确的价值区间 | 单动作 + 小 h(≤8)：空间小到**直接穷举**，bandit 没必要；h≥32 或多 op 才需要 bandit | `:128-143` |

### B.2 实现的偏离（`code/my_ai/az_intent/unified_action_search.py`）

| 设计点 | 原设计 | 实现 | 判定 |
|---|---|---|---|
| 类选择 | UCB bandit（`Q[c]` + 探索，随 rollout 更新） | **一次性**类网 softmax 分预算（`allocate()` `:155-192`），无 Q 更新、无探索项 | **bandit 被整个删掉** |
| 游标 | 按位置榜游标 `cursor[c]` 逐个扫 | 通道内按位置图排序取 top-N（`:147-151`），等价"位置图先验一次性截断" | 保留排序、丢掉游标语义 |
| 评估器 | 复用强评估（复评 ≥30 样本/候选） | 阶段一每候选 **1 次** rollout（`:259`），阶段二只复评 top-8（`:263-266`） | **换成了弱评估**（1 样本挑 150+ 候选） |
| 多动作组 | 三头独立采样（能表达"降级+建塔"） | **只有单原子动作**（`enumerate_by_channel` 每条 `ops` 都是单 op） | **组合表达没做** |
| 升级 | 无（设计未展开） | 显式枚举（`:104-118`，这是实现**对**的地方） | 实现领先 |
| HOLD | 无（设计未展开） | 通道 23 无条件加入（`:142-143`） | 实现领先 |

**更根本的语义偏离（比 `[ch,y,x]→[ch,x,y]` 转置更重）**：
原设计是**"用 rollout 动态更新 Q[c] 的 bandit 搜索"**，实现是**"用类网（一个已知不可靠的 frozen 先验）
一次性分配预算 + 静态枚举"**。类网本身 class recall@1 只有 59.7%、建塔类 @1 = 0.0%
（`prior_repair_20261023.md:342-344`），用它当跨通道预算分配器 ≈ 把预算撒给"类网以为重要的类"，
而这正是原设计想用 bandit **绕开**的东西。

### B.3 当时 0/42 的可能原因（`_ab_unified.sh` → 历史日志）

实测结果（`training_history/runs/20261021_235737_ab_unified_v2/output.log:268`）：
```
=== flat vs mcts：42 局  flat 胜率 0.0%  (0W/0D/42L)  ±15.4%(2SE)  wall=504.7 min ===
```
⚠️ 标签陷阱：`_ab_unified.sh:34` 把 A=统一搜索 传给 `--flat-env`、B=flat+cells4 传给 `--mcts-env`，
所以汇总行的 "flat 胜率 0.0%" **= 统一搜索 0 胜**（`ab_search_modes.py:598-599` 用 `flat` 代称 `--flat-env`）。
逐局行（`:225-266`）全部 `flat=P0/P1 score=0.0 hp=(0, N)` ⇒ 统一搜索每局基地被打到 0。

可能原因（按可信度排序，**均为推断，未跑对局复核**）：

1. **评估器被降级**（最可能）：阶段一 150+ 个原子动作每个只 1 次 h=32 安静 rollout + 值网叶子（σ≈110），
   用 1 个样本从 ~150 个候选里挑 top-8，选错率按设计文档自己算的公式约 42%（`idea_unified_bandit_search.md:62-64`）；
   而对手 flat 是"48 候选 × 64 次 × 16 回合"的强评估。**同预算下评估噪声被碾压**。
2. **类网预算分配把预算撒错**：跨通道预算用类网 softmax（建塔类 @1=0.0% 的类网），
   `min_per_class=2`（`:74`、`:172`）还把预算硬性平摊到 ~16 个通道，进一步稀释。
3. **候选结构不对（单原子）**：只枚举单 op，表达不了"降级+建塔"（真实 2-op 模板 93%），
   而 flat 侧能产出多 op 束。
4. **参数没调**：h=32 违背设计文档自己建议的"单动作 + 小 h(≤8) 先穷举"（`:139-143`）；
   `candidate_cap=256` 在真实原子空间（~37-45，`why_effective_candidates_collapse_20261028.md:218`）下等于全枚举。
5. **有 bug 但非主因**：v2 日志里大量 `ValueError: invalid literal for int() with base 10: 'None'`
   （`winner_side=None`）是 `ab_search_modes.py` 结果解析的崩溃，不影响"统一搜索被打爆"的结论，
   但会让胜率统计的样本变少（42 局而非 64 局）。

⇒ **当时实现输在"把强评估器换成了弱评估 + 把 bandit 换成类网先验"，不是"统一候选"这个方向本身错了。**
证据：同样的 flat 评估器，一旦候选集里有真动作（非 HOLD 注入），就能 12.5%→45.8%
（`nonhold_inject_ab_20261029.md:121`）。

---

## C. 统一的"类×位置"候选生成能否根除"缺 HOLD / 缺升级"

### C.1 通道语义（先厘清，避免把 op 码当通道）

`OperationType`（`Ant-Game/SDK/utils/constants.py:121-130`）：`11=BUILD_TOWER / 12=UPGRADE_TOWER /
13=DOWNGRADE_TOWER / 21-24=超武 / 31-32=基地升级`。
网络的 `NUM_CLASSES=24` 是**意图类**（`network.py:130`），不是 op 码：
`0-15=塔意图(目标塔型)、16=降级、17-20=超武、21-22=基地升级、23=HOLD`（`decoder.py:37-72`）。

⇒ **"升级"不是 action_map 的一个通道**。它由 `_decode_tower_action` 在"塔意图通道 × 已占格"上产出：
`decoder.py:451-456`（friendly tower → `upgrade_step` → `Operation(UPGRADE_TOWER, id, step)`）。

### C.2 回答：跨类 argmax 能否自然产出 HOLD / 升级

| 动作 | "跨所有类×所有位置 argmax" 能否自然进候选 | 依据 |
|---|---|---|
| **HOLD（类 23）** | ✅ **能**（只要生成器显式把通道 23 当候选；它无位置、`class_mask[23]=True` 恒成立） | `decoder.py:160-161`、`unified_action_search.py:142-143` |
| **升级（op 12）** | ⚠️ **不能只靠 argmax**。argmax 只有在"塔意图通道(1-15)恰好落在已占格"时才通过 `_decode_tower_action` 变成升级；空高地远多于已占塔（31 vs 1-3），概率极低。**必须显式枚举"塔 × 升级树"** | `decoder.py:433-456`、`make_position_masks` `:196-214` |

⇒ 结论：**HOLD 天然进候选（通道 23 存在）；升级必须显式枚举，不能靠 argmax。**
`unified_action_search.enumerate_by_channel` **已经**把升级显式枚举对了（`:104-118`），
只是被"弱评估 + 类网分预算"拖死，而不是候选枚举错。

### C.3 最小可行设计：候选生成器（只换"构造"，不换"评估"）

**核心洞察**：`nonhold_inject_ab` / `always_hold_ab` 已证明**评估器（flat N×h rollout + 值网）没问题**，
问题只在**候选生成**。所以正解不是重写搜索，而是**把统一枚举当候选生成器，喂给现有 flat 评估器**。

候选生成函数签名（概念）：
```
generate_candidates(action_map, head_logits, state, player, k=48) -> list[bundle]
```
步骤（复用 `unified_action_search.enumerate_by_channel` 的现成逻辑）：
1. 用 `decoder.make_class_mask / make_position_masks` 拿合法类×合法格（`decoder.py:119-228`）；
2. **显式枚举**：建塔（空格）/ 升级（每塔×升级树，`enumerate_by_channel:104-118`）/ 降级（每塔）/
   超武（合法格）/ 基地升级 / **HOLD（通道 23，无条件）**；
3. 通道内按该类位置图排序，去重（`(op_type, arg0, arg1)` 键），截断到 k=48；
4. 交给 **`FlatRolloutGreedy.search`** 现有的 `_collect_extras` 通道（`flat_rollout_greedy.py:445-510`）
   或直接把枚举结果当 `root.bundles` 喂评估器。

**需要的改动（默认关、零影响）**：

| 文件 | 改动 | 规模 |
|---|---|---|
| `code/my_ai/az_intent/flat_rollout_greedy.py` | 新增一个 `cand_mode="unified"` 分支：`_expand` 之后不采束，改调 `enumerate_by_channel` 产出 bundles | +~30 行 |
| `code/my_ai/az_intent/unified_action_search.py` | 把 `enumerate_by_channel` 提取为可复用的模块级函数（或直接 import） | +~0 行（已是方法） |
| `code/test_match/az_bridge_ai.py` | 加 `AZAI_CAND_MODE=unified` env 读取（默认 `""`） | +1 行 |

零影响证明思路（与 `AZAI_PIN_K` 同款，`why_effective_candidates_collapse_20261028.md:342-349`）：
默认 `AZAI_CAND_MODE=""` ⇒ 走原有 `_expand` 采样，逐位不变；冒烟对拍同一批 1518 决策点的 JSON 应 0 处不同。

### C.4 与 `AZAI_PIN_K` / `AZAI_ALWAYS_HOLD` 的关系：谁是补丁、谁是正解

| 开关 | 性质 | 治什么 | 局限性 |
|---|---|---|---|
| `AZAI_ALWAYS_HOLD=1`（`az_bridge_ai.py:2422-2429`） | **补丁**（症状级） | 候选里强制补一个 `()` | 只治"缺 HOLD"；治不了"缺升级/缺建塔/降级 churn"；评估器仍可能在 48 条空束里被迫选 HOLD |
| `AZAI_PIN_K=K`（`bundle_mcts.py:1010-1027`） | **补丁**（结构级，但仍在类轴内） | 把 720 次采样分给 K 个类，让建塔/降级/HOLD 共存 | 仍走"钉类→类内抽位置"，**升级仍产不出**（类轴不产 op 12）；K 类仍由 `_econ_rank`/菜单驱动 |
| **统一候选生成（C.3）** | **正解** | 候选 = 显式枚举"类×位置"（含 HOLD、含升级、含建塔） | 需要正确评估器兜底（这正是 flat 已有的） |

一句话：`ALWAYS_HOLD` 和 `PIN_K` 是**补丁**（分别补"缺 HOLD"和"类被钉死"两个症状），
**统一候选生成才是正解**——它从根上让"类×位置"的所有原子动作（含 HOLD、含升级）同时进候选，
再把选择权交给已被证明可靠的 flat 评估器。

---

## 对照表：当前候选管线 vs 统一候选管线

| 维度 | 当前管线（现役） | 统一候选管线（C.3 设计） |
|---|---|---|
| 类选择 | `mc_quiet` 仲裁选**1 个类**，`pinned=[_ccls]*3`（`bundle_mcts.py:884`） | 不选类，**全类枚举** |
| 78% HOLD | HOLD 由廉价仲裁（4×16 回合）独占决定 | HOLD 只是候选之一，由 flat 长 rollout 定去留 |
| 候选构造 | 钉死类的 720 次采样去重取 top-48（`bundle_mcts.py:1005/1084/1097`） | `enumerate_by_channel` 显式枚举 + 去重截断 |
| HOLD 在不在 | **26.8% 的回合不在**（`always_hold_ab_20261029.md:178`） | **恒在**（通道 23 无条件枚举） |
| 升级在不在 | **从不在**（类轴不产 op 12；唯一构造点 `mc_upg` 默认 0） | **恒在**（塔×升级树显式枚举，`enumerate_by_channel:104-118`） |
| 建塔/降级/超武 | 只在非 HOLD 回合、且单一类内 | 全部并列候选 |
| 评估器 | flat N×h rollout + 值网（强，已被 A/B 证明可靠） | **同一个 flat 评估器（不动）** |
| 先验用途 | 类轴先验（不可靠）+ 位置先验（建塔无信号） | 位置先验只用于**通道内排序**，不决定"类去留" |

---

## 分阶段最小可行修复方案（附风险）

### 阶段 1（补丁，立即可用，默认关）
1. `AZAI_ALWAYS_HOLD=1` —— 已实现（`az_bridge_ai.py:2422-2429`），候选恒有 HOLD。收益 +4~12.5pp（不显著但无害）。
2. `AZAI_PIN_K=2` —— 已实现（`bundle_mcts.py:1010-1027`），`n_cand==1` 82.3%→0.7%（`why_effective_candidates_collapse_20261028.md:262`）。
3. 升级候选：把 `mc_upg` 的显式升级构造（`bundle_mcts.py:798-834`）**从"绕过评估器直接发"改为"喂进 extra_bundles 让 flat 评估"**（默认关）。

**风险**：p95/max 决策耗时可能越界（`n_cand` 放大 14 倍，`why_effective_candidates_collapse_20261028.md:364-369`）；
`PIN_K` 仍是类轴内补丁，升级/组合表达仍受限。

### 阶段 2（正解，默认关，独立开关）
统一候选生成（C.3）：候选 = 显式枚举的原子动作（含 HOLD、含升级），交 flat 评估器。
**风险**：
1. 成本：候选数 ~37-45（`why_effective_candidates_collapse_20261028.md:218`），flat 预算公式
   `n_rollouts = min(64, max_total_rounds // (n_cand × h))`（`flat_rollout_greedy.py:549-552`）会自动压采样数；
   需实测 p50/p95/max。
2. 单原子动作表达不了多 op 组合（93% 的 2-op 模板是"降级+建塔"）——阶段 2 先单原子，组合留到阶段 3。
3. 与注入通道（`_collect_extras`，`flat_rollout_greedy.py:445-468`）的去重/追加语义要核对，避免把 HOLD 重复计。

---

## 诚实清单

1. **0/42 的归因是推断**：约束禁止跑对局，无法复核"统一搜索输在哪一个具体机制"。
   上述 B.3 五条是从代码 + 设计文档 + 历史日志推的，未做消融。
2. **`network.py` 文档字符串与代码不一致**：文件头/`forward` 注释写 `action_map(23,19,19)`，
   但 `NUM_CLASSES=24`（`network.py:130`）、`action_map_conv=Conv2d(64,24,1)`（`:172`）⇒ 实际是 **24 通道**，
   通道 23 = HOLD。本文以代码为准。
3. **"升级从不出现在候选"的精确表述**：类轴**在塔意图通道(1-15)落在已占格时理论能产 op 12**
   （`decoder.py:451-456`），但现役 mc_quiet 钉的类实测只有 {0,16,17,23}（`why_effective_candidates_collapse_20261028.md:97`），
   这四类都不落到"可升级格"，所以**实测 0 次**；显式构造点只有 `mc_upg`（默认 0）。表述成立。
4. **未找到**：`docs/experiment_log.md:11706-11715` 的 `ab_unified_v2` 结果字段仍是 `_待填_`；
   本文的 0/42 直接取自 `training_history/runs/20261021_235737_ab_unified_v2/output.log:268`。
5. `AZAI_MC_UPG` 默认 0 见 `az_bridge_ai.py:1265`；`AZAI_PIN_K` 默认 0 见 `:1282`。
