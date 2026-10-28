# 为什么我们自己的候选生成器几乎从不产生 `UPGRADE_TOWER`（只读诊断，2026-10-28）

> 工作目录 `D:\2026智能体大赛_新2_distill`。**只读诊断**：没有改 `code/my_ai/**`、没有跑对局、
> 没有 kill 任何进程。全部证据来自**已落盘的日志** + **对已落盘对局的离线重放**（引擎只读接口）。
>
> 上游：`docs/inject_vs_runnerup_20261028.md`（注入 A/B 对外标尺，commit `89607af`）§3.7 报出
> "我方 op 里 `UPGRADE_TOWER` = **0.0%**，对手 **9.3%**，注入后 4.3%"。本文解释那个 0.0%。
>
> 探针（本次新增，只读）：
> * `code/test_match/probe_why_no_upgrade.py` —— 出手构成 / 金币 / 塔等级 / 升级机会（纯日志）
> * `code/test_match/probe_upgrade_class_axis.py` —— 重放真实对局 + 跑现役网，算**类轴**能不能表达升级

---

## 0. 结论（一句话）

**是"根本没进候选"，而且不是"偶尔没抽到"，是"支撑集为空"**：
现役配置下整束 48 个候选**全部被钉在同一个类**上（`bundle_mcts.py:870` 的 `pinned=[_ccls]*3`），
而那个类在 **7875 个真实决策点里 100% 落在 {0, 16}**（85.7% / 14.3%，落在 1-15 的 **0 次**）；
类 0（目标 = BASIC 本身）在解码器里**数学上不可能**产出升级（`decoder.py:452-456`），
类 16 走的是降级分支（`decoder.py:417-428`）。唯一能显式构造 `UPGRADE_TOWER` 候选的代码块
（`bundle_mcts.py:781-820`）被开关 `AZAI_MC_UPG` 门控，**默认 0 且现役配置没有设它**
（`az_bridge_ai.py:816`；`_inject_vs_runnerup.sh:45` 的 BASE 里逐字核对过）。

⇒ 三句话：**① 原因不是"金币不够/位置不可行"**（42.3% 的回合有引擎认定合法且买得起的单步升级，
累计 22368 个合法升级 op，一次都没发）；**② 不是"进了候选但评估器没选"**（`UPGRADE_TOWER` 在候选
束的**可达集合**里出现次数 = 0/7850 ⇒ 压根没有可选的升级；反过来，只要塞给它一个候选，现役 `net`
评分器在 5/30 = 17% 的机会局面里会选中，§6.4）；**③ 是"类轴 + 一个默认关的开关"共同
造成的结构性问题**——类头永远 argmax 到 17（闪电）、经济类仲裁永远挑 0/16，升级类从来没被钉过。

---

## 1. 先厘清一个容易搞错的映射（任务书要求核对）

**"类索引" ≠ "op 类型"，而且 `UPGRADE_TOWER` 没有自己的类。**

| 项 | 值 | 出处 |
|---|---|---|
| `OperationType` | `BUILD_TOWER=11` / **`UPGRADE_TOWER=12`** / `DOWNGRADE_TOWER=13` / 闪电 21-24 / 基地升级 31-32 | `Ant-Game/SDK/utils/constants.py` |
| `decoder.NUM_CLASSES` | **24** = 0-22 动作类 + 23 HOLD | `code/my_ai/decoder.py:72` |
| 类 0-15 | **不是 16 种动作**，而是"**目标塔类型**"（16 个通道 → 目标 `TowerType`） | `code/my_ai/decoder.py:37-54` |
| 类 16 | **降级**（op 13），不是升级 | `decoder.py:417-428` |
| 类 17-20 | 四种超级武器（op 21-24） | `decoder.py:58-70` |
| 类 21-22 | 基地升级（op 31/32） | `decoder.py:369-373` |
| 类 23 | HOLD | `decoder.py:160-161,366-367` |
| **升级怎么表达** | 格子**已有我方塔**、且该通道的**目标类型**能从当前类型一步走到 ⇒ `UPGRADE_TOWER(tower_id, int(step))` | `decoder.py:433-456` |

⚠️ 两个坑（本文全部数字都按这张表核对过）：

1. **`TowerType` 的枚举值不是 0-15 连续**：`BASIC=0 / HEAVY=1 / QUICK=2 / MORTAR=3 / PRODUCER=4 /
   HEAVY_PLUS=11 / ICE=12 / BEWITCH=13 / QUICK_PLUS=21 / DOUBLE=22 / SNIPER=23 / MORTAR_PLUS=31 /
   PULSE=32 / MISSILE=33 / PRODUCER_FAST=41 / PRODUCER_SIEGE=42 / PRODUCER_MEDIC=43`。
   ⇒ 日志里 `towers=[(id, player, x, y, tower_type, hp)]` 的那个 `tower_type` 是**真枚举值**，
   **不是**类号（`match_via_sdk.py:770-771`）。
2. **升到三级的费用不是 60 而是 200**：`upgrade_tower_cost(target)` 按**目标**取值
   （`game_state_facade.py:304-310`：`target.value < 10` ⇒ 60，否则 200）。而
   `decoder.py:209-213` 的**位置掩码**按**当前塔型**选常数（当前是 HEAVY 时也取 60）⇒
   对"二级塔升三级"这一步掩码**乐观 140 金**（会放进来一个其实买不起的候选，
   随后被 `sample_bundle` 的 `can_apply_operation` 丢掉，`bundle_mcts.py:309`）。
   这是**另一个**小 bug，不是本文的主因（我们连二级塔都没有）。

---

## 2. 候选束是怎么生成的（现役配置的完整路径）

现役配置（`_inject_vs_runnerup.sh:45` 的 `BASE`，逐字）：
`AZAI_MODE=pos-only; AZAI_POSPIN=mc_quiet; AZAI_MC_EVERY=1; AZAI_MC_ROOTONLY=1; AZAI_MC_CELLS=4;
AZAI_SKIP1=1; AZAI_TCLASS=0.5; AZAI_TPOS=1.0; AZAI_POSPRIOR=off; AZAI_SEARCH=flat; AZAI_K=48; …
AZAI_MC_SCORE=net; AZAI_MC_NET_SAMPLES=4; AZAI_MC_HORIZON=16`
（**没有** `AZAI_MC_UPG`、**没有** `AZAI_MC_UPG_TARGETS`、**没有** `AZAI_MC_UPG_BONUS`、
**没有** `AZAI_MC_TOPK`）

一根束的生成只有 6 步，升级在每一步的位置如下：

| 步 | 做什么 | 代码 | 对"升级"的影响 |
|---|---|---|---|
| 1 | 建掩码（`intent_decoding=True`） | `bundle_mcts.py:658-661` → `decoder.py:119-228` | 类 1-15 在"我方 BASIC 塔格"上**是合法**的 ⇒ **掩码不排除升级** |
| 2 | `pos-only` ⇒ **类被钉住** | `bundle_mcts.py:677` | 整束**只可能**是这一个类 |
| 3 | 钉哪个类：`pos_pin=mc_quiet` 的 MC 仲裁 | `bundle_mcts.py:695-912` | 见 §3，**这里是断点** |
| 4 | 3 个 head 同一类 + 位置抽样 | `bundle_mcts.py:870`（`pinned=[_ccls]*3`）、`:991`、`:1040-1047`、`decoder.py:231-274` | 位置在**该类全部合法格**上按 z-score softmax 抽样（`t_pos=1.0`）⇒ 只要类对，塔格会被抽到 |
| 5 | 每束逐条 `can_apply` 过滤 | `bundle_mcts.py:309` | 合法才留下 |
| 6 | flat rollout 给 48 束打分取 argmax | `flat_rollout_greedy.py:456-617` | 评估器在这一步**才有机会**看到升级 |

**MC 菜单里能出现的候选（`mc_cells=4` 分支）** = `{闪电?}` ∪ `{HOLD}` ∪ `{(经济类, 最优 4 个空格)}`
∪ `{(经济类, 我方每一座塔的格子)}`（`bundle_mcts.py:721-748`）。
⚠️ 注意菜单里的"选格"候选**只带类号**：`mc_quiet_choose` 返回 `(类, x, y)`，调用方取 `_ccls = int(_c[0])`
（`bundle_mcts.py:858-870`）⇒ **那个 `(x, y)` 被丢掉了**，位置随后重新抽（第 4 步）。
所以"选类×选格"实际退化成"选类"，塔格信息不起作用。

**唯一能显式塞进 `Operation(12, …)` 的地方**（`bundle_mcts.py:781-820`）：

```python
# bundle_mcts.py:784
if self.mc_upg > 0:                                   # ← 默认 0（az_bridge_ai.py:816）
    for _tw in _tw_list:                              # 前 mc_upg 座塔
        for _tgt_enum in list(_tgts)[: mc_upg_targets]:  # 默认只取第一个（= HEAVY）
            _cost = int(node.state.upgrade_tower_cost(_tgt_enum))
            if int(node.state.coins[node.player]) < _cost: continue   # ← 金币闸门在这
            _cands.append(('op', 12, int(_tw.tower_id), int(_tgt_enum)))
```
被选中后**不经过 48 束的评估器**，直接在 `decide()` 里发出（`az_bridge_ai.py:1381-1384`）：
```python
_expl = getattr(mcts, 'mc_last_explicit', None)
if _expl is not None:
    return [Operation(OperationType(int(_expl[0])), int(_expl[1]), int(_expl[2]))]
```
⇒ **这条路径的语义是"让 MC 仲裁直接接管这一手"，不是"让升级成为候选"**。
它和注入通道（`set_extra_bundles`）是两件不同的事，历史 A/B 的差异主要来自这里（§6.3）。

---

## 3. 根因：类轴（class axis）把升级压成 0

### 3.1 机制：钉住的类 ∈ {17, 23, 经济类}

`bundle_mcts.py:700-712` 的经济类排序（**逐字复刻**进探针）：

```python
_economy = argmax_{c ∈ 0..16, 掩码允许}  max_{(x,y) ∈ 该类合法格} action_map[c][x,y]
```
注释自己就写了这件事（`bundle_mcts.py:751-752`）：
> "现有菜单的经济候选实际上永远是『类 0 或类 16』（**跨类 map 最大值不可比**），
>  导致我们只会『铺一级塔 + 反复降级』、**从不升级**（对手 9.4 次升级/局）。"

而 `pinned=[_ccls]*3` 之后，**类 0 的解码结果只能是 `BUILD_TOWER`**：
`_decode_tower_action(BASIC 目标)` 在我们自己的 BASIC 塔格上 `upgrade_step(BASIC, BASIC) → None`
⇒ 返回 `None`（`decoder.py:452-454`）；`make_position_masks` 对类 0 只在**空格**上置真
（`decoder.py:205-214`：`step is None` ⇒ 掩码保持 False）。
⇒ **类 0 的整束候选在数学上不可能含 `UPGRADE_TOWER`**。

### 3.2 实测（重放 B 臂 16 局，7875 个我方决策点，全部用现役 ckpt）

探针 `probe_upgrade_class_axis.py`（重放保真度：**塔态逐回合与对局日志完全一致，
16 局共 7875 回合，不符 0 次**）：

| 量 | 值 | 说明 |
|---|---|---|
| 经济类（= 整束被钉住的类）分布 | **类 16 = 6728（85.7%）**，**类 0 = 1122（14.3%）** | 另有 25 个决策点连一个合法经济类都没有 |
| 落在类 1-15（唯一能表达升级的区间） | **0 / 7850 = 0.00%** | |
| 3 个 head 的类头 argmax | **类 17（闪电）= 7875/7875 = 100%**（三个 head 全是） | 与"闪电永远 argmax"的历史一致 |
| 候选束**可达 op 集合**里含 `UPGRADE_TOWER` 的决策点 | **0 / 7850 = 0.00%** | 把钉住类的每个合法格按解码器口径逐个解一遍（支撑集分析） |
| 对照：若整束被钉在类 1..15 | 抽样 983 个决策点 × 15 个类 = 14745 个 (类,决策点) 组合里 **6435（43.6%）有 ≥1 个能解成升级的格** | 与 §4 的"有升级机会"同量级 ⇒ **位置轴完全没问题，问题只在类** |
| 每一类通道的可用性 | 类 1-15 **每个**都在 3329 个状态里有 ≥1 个"能解成升级的格" | 类 0 永远 0（目标就是 BASIC 本身） |

### 3.3 为什么几个"看起来相关"的开关救不了它

| 开关 | 现役值 | 实际生效点 | 对升级的效果 |
|---|---|---|---|
| `AZAI_MC_CELLS` | **4** | `bundle_mcts.py:729-748` | 让菜单能表达"类×格" ⇒ 但`(x,y)` 被丢弃（§2），且 `_c0` 仍是 0/16 ⇒ **无效** |
| `AZAI_MC_TOPK` | 未设（0） | `bundle_mcts.py:749-761` | 这个分支是 `if mc_cells>0: … **elif** mc_topk>0:` ⇒ **`MC_CELLS>0` 时 `MC_TOPK` 完全死掉**（历史宽菜单实验都是 `MC_CELLS=0` 的配置） |
| `AZAI_MC_UPG` | **未设（0）** | `bundle_mcts.py:784-820`、`az_bridge_ai.py:816` | **唯一的显式升级候选入口，默认关** |
| `AZAI_MC_UPG_TARGETS` | 未设（1） | `bundle_mcts.py:803` | 只取升级树第一个目标（= HEAVY）；历史发现"统一升成最差的那个"（`bundle_mcts.py:799-802`） |
| `AZAI_MC_UPG_BONUS` | 未设（0） | `mc_quiet.py:250-258`、`az_bridge_ai.py:827` | 给 `('op',12,…)` 候选加常数分（= 亚军 `c1_root_bonus` 的类比）。**2026-10-22 加了代码，只在离线探针 `code/my_ai/az_intent/probe_upgrade_selectability.py:65-72` 里扫过 0.05/0.10/0.20；`grep` 全部 `code/test_match/*.sh` **无命中** ⇒ 没有任何真实对局臂用过它** |
| `AZAI_MC_EVERY` | 1 | `bundle_mcts.py:719-720` | =1 ⇒ **每回合**都仲裁 ⇒ 上面的钉类每回合都覆盖类头 |
| `AZAI_TCLASS` | 0.5 | `az_bridge_ai.py:787` | **`pos-only` 下完全不生效**：`class_ids` 是钉死的，`decode_head` 的 `class_id` 分支优先（`decoder.py:344-345`）⇒ 类采样/温度整段不执行 |
| `AZAI_TPOS` | 1.0 | `az_bridge_ai.py:788` → `bundle_mcts.py:1031` | 位置**抽样**（不是 argmax）⇒ 类对的话塔格会被抽到（这是 §3.2 对照行能成立的原因） |
| `AZAI_SKIP1` | 1 | `bundle_mcts.py:1127` | **`AZAI_SEARCH=flat` 下不生效**（`flat_rollout_greedy` 不引用它）；即便生效也只是"单候选时跳过搜索"，与升级无关 |
| `AZAI_POSPRIOR` | off | `bundle_mcts.py:1025-1029` | off ⇒ 回调为 `None`。即便开 `value/uniform` 也只改**被钉类内部**的位置分布（`_prior_classes = pinned`，`:1023-1024`）⇒ **改不了类** |
| `AZAI_BAN_CLASSES` | 未设 | `decoder.py:163-184` | 掩码层禁类（历史只用于禁 17/闪电），**从未禁过 1-15 或 12** |
| `AZAI_BAN_OPS` | 未设 | `az_bridge_ai.py:1436-1444` | 执行层禁 op，**从未设成 12** |
| `AZAI_NO_DOWNGRADE` | 未设 | `az_bridge_ai.py:1396-1409` | 只删降级（还给钱的 op）⇒ 少了退款，**金币更少**，不会带来升级 |

---

## 4. 不是"位置/金币/塔等级不满足"

数据：`INJECT_VS_RU_B`（**不注入**，即现役 BASE 配置）× 16 局 = **7875 个我方决策回合**。
金币取**决策时刻**（本回合操作执行前，`match_via_sdk.py:715-727` / `progress.py:101-117`）；
塔态取同一时刻的 `[round N] towers=[…]` 行（`match_via_sdk.py:770-771`）。

### 4.1 我方金币分布（够不够升级？够）

| 量 | 值 |
|---|---|
| 金币 p10 / p25 / **p50** / p75 / p90 / max / mean | 21 / 39 / **64** / 97 / 143 / 352 / **75.6** |
| `P(金币 ≥ 60)`（二级升级价） | **54.2%**（4271/7875） |
| `P(金币 ≥ 200)`（三级升级价） | 3.8%（299/7875） |
| **局末金币**（16 局，`RESULT` 行） | 我方 mean **93** / p50 84 / min 11 / max 234；**对手 mean 372 / p50 328** |
| 历史文档"局末金币闲置 ~500" | 那是**更早的配置**；本批我方局末只有 84（p50）⇒ 但**过程**中 54% 的回合 ≥ 60 ⇒ **不是"没钱"** |

### 4.2 我方塔等级（有没有可升的塔？有，而且全是可升的一级塔）

| 量 | 我方 | 对手（亚军） |
|---|---|---|
| 塔·回合数（n） | 12468 | 13410 |
| **一级（BASIC）** | **12468 = 100.0%** | 6827（50.9%） |
| 二级 | **0 = 0.0%** | 1906（14.2%） |
| 三级 | **0 = 0.0%** | **4677（34.9%）** |
| 有 ≥1 座一级塔的回合 | 6728/7875 = **85.4%** | 73.8% |
| 一座塔都没有的回合 | 1147/7875 = 14.6% | — |

⇒ 我们的塔**永远停在 1 级**（历史文档"我们 10.8 次降级/局、从不升级"在这里复现：
降级占我方 op 46.0%，而 `DOWNGRADE` 打在 BASIC 塔上 = **拆塔退钱**，所以"建塔→拆塔"的 churn
把塔数/塔级都锁死在一级）。

### 4.3 "合法升级机会"（用两种独立方法算，结果一致）

口径：`合法单步升级 = 塔在升级树上某一目标 + 金币 ≥ 该目标费用`（费用同 `game_state_facade.py:304-310`）。
方法 A（纯日志离线复算，`probe_why_no_upgrade.py`）；方法 B（重放状态 + 引擎 `can_apply_dry`
逐条问，`probe_upgrade_class_axis.py`）——**两个方法给出完全相同的数字**：

| 量 | 值 |
|---|---|
| 有 ≥1 个"买得起的单步升级"的回合（方法 A） | **3329 / 7875 = 42.3%** |
| 有 ≥1 个"引擎认定合法"的升级 op 的决策点（方法 B） | **3329 / 7875 = 42.3%**（逐局：269/402/169/184/…） |
| 累计"买得起的单步升级"组合（塔×目标） | **22368** |
| 累计"引擎认定合法"的升级 op | **22368**（同值 ⇒ 金币够 + 树上有目标 ⇒ 引擎就认） |
| **实际发出的 `UPGRADE_TOWER`** | **0**（机会转化率 **0.0%**） |

⇒ **Q3 的答案是"条件满足但从不发生"**：既不是没钱，也不是没塔，更不是非法。

### 4.4 逐局明细（我方 / 对手的 `UPGRADE_TOWER` 次数）

| token | 座位 | 回合 | 我方 UPG | 对手 UPG | 有合法升级机会的回合 |
|---|---|---|---|---|---|
| 7 | P0 | 497 | **0** | 17 | 269 |
| 7r | P1 | 512 | **0** | 17 | 215 |
| 8 | P0 | 456 | **0** | 15 | 141 |
| 8r | P1 | 512 | **0** | 14 | 147 |
| 9 | P0 | 369 | **0** | 7 | 79 |
| 9r | P1 | 512 | **0** | 18 | 169 |
| 10 | P0 | 512 | **0** | 12 | 272 |
| 10r | P1 | 436 | **0** | 13 | 122 |
| 11 | P0 | 490 | **0** | 16 | 228 |
| 11r | P1 | 512 | **0** | 16 | 247 |
| 12 | P0 | 512 | **0** | 16 | 402 |
| 12r | P1 | 512 | **0** | 19 | 291 |
| 13 | P0 | 507 | **0** | 19 | 137 |
| 13r | P1 | 512 | **0** | 22 | 201 |
| 14 | P0 | 512 | **0** | 17 | 225 |
| 14r | P1 | 512 | **0** | 14 | 184 |

（16 局 **0**；对手 252 次。没有任何一局、任何一个座位出现 1 次升级。）

---

## 5. 证据表：出手构成（我方 / 对手 / 注入臂）

用 `probe_why_no_upgrade.py` 从**落盘 trace** 重算（不是抄文档），三方对比：

| op | B 臂（不注入）我方 | B 臂对手（亚军） | A 臂（注入）我方 |
|---|---|---|---|
| `BUILD_TOWER(11)` | 1544（**48.6%**） | 1119（41.5%） | 830（42.4%） |
| **`UPGRADE_TOWER(12)`** | **0（0.0%）** | **252（9.3%）** | **85（4.3%）** |
| `DOWNGRADE_TOWER(13)` | 1461（**46.0%**） | 1250（46.3%） | 840（42.9%） |
| `USE_LIGHTNING_STORM(21)` | 170（5.4%） | 61（2.3%） | 203（10.4%） |
| `USE_EMERGENCY_EVASION(24)` | 0 | 15（0.6%） | 0 |
| 合计 op | 3175 | 2697 | 1959 |
| **有出手的决策数 / 全部决策回合** | **1935 / 7875** | — | **1303 / 8117** |
| op / 有出手的决策 | **1.64**（3175/1935） | — | **1.50**（1959/1303） |
| 我方塔·等级构成 | **100% 一级** | 50.9/14.2/34.9% | 76.8/15.3/7.8% |
| 金币 p50（决策时刻） | **64** | — | 108 |

两点读数：

1. **A 臂那 4.3% 的升级只能来自注入通道**：现役配置的候选束支撑集里升级出现次数 = 0/7850（§3.2）
   ⇒ 我们自己的生成器一手都给不出，那些升级是亚军那一手经 `set_extra_bundles` 进来、被 flat rollout
   argmax 选中的。**这反过来证明评估器认得出升级**（有候选时被选中 1244/2721 = 45.7%，
   见上游报告 §3.5）。
2. **A 臂的金币 p50 从 64 抬到 108、塔等级从 100% 一级变成 76.8/15.3/7.8** ⇒ 升级一旦出现，
   金币结构跟着变（不再"建塔→拆塔"净流失 10%）。

---

## 6. 历史实验：开关确实能打开升级，但打开后**没赢更多**

### 6.1 "开关一开，升级就出现"（同一套日志口径）

| tag | 关键配置 | 我方 op 总数 | 我方 `UPGRADE(12)` | 我方塔等级 |
|---|---|---|---|---|
| `VS_RUNNERUP_cells4`（10-21，32 局） | `MC_CELLS=4`，**`MC_UPG` 未设** | 5173 | **0（0.0%）** | 100% 一级 |
| `VS_RUNNERUP_cells4_upg1`（10-21，15 局） | `MC_UPG=1`（**在 `int('op')` bug 修复之前**） | 1411 | **0（0.0%）** | 100% 一级 |
| `VS_RUNNERUP_upg_net`（10-22 08:47，8 局） | `MC_UPG=1` + `MC_SCORE=net`（**同样在 bug 修复之前**） | 755 | **0（0.0%）** | 100% 一级 |
| **`VS_RUNNERUP_upg1fix3`**（10-22 09:24，22 局） | `MC_UPG=1`（**bug 修复之后**，quiet 评分） | 1275 | **115（9.0%）** | 78.5% 一级 / **21.5% 二级** |
| `INJECT_VS_RU_B`（10-28，现役 BASE） | **`MC_UPG` 未设** | 3175 | **0** | 100% 一级 |
| `INJECT_VS_RU_A`（10-28，注入） | + `AZAI_ORACLE_INJECT` | 1959 | 85（4.3%） | 76.8/15.3/7.8 |

⚠️ **必须看的一列**：那两个 `MC_UPG=1` 却 0 升级的臂，是踩了 `int('op')` 的 bug
（`bundle_mcts.py:853-857` 的注释 + commit `184d7b6` 的提交信息）：候选进了菜单、但取类时
`int('op')` 抛 `ValueError`、被宽 `except` 吞掉 ⇒ `mc_last_explicit` 永远 `None`
⇒ **升级动作从落地以来一次都没被执行过**。commit `184d7b6`（2026-10-22 08:58）修的就是这个
（提交信息："30 局面实测 0/30 → 12/30"）。
**所以"配了 MC_UPG=1 也没有升级"这个观察在 10-22 08:58 之前全部无效**，不能用来说明
"评估器不选升级"。

### 6.2 `upg1fix3` 证明了什么（关键对照）

同一个评估器、同一批种子、唯一差别是 `MC_UPG=1` ⇒ **升级 9.0%（115 次），塔等级出现 21.5% 二级**。
⇒ **"候选里一旦有升级，就会被选中/被执行"**。这独立复现了注入臂的结论：
**瓶颈在候选生成，不在评估器。**

### 6.3 但"打开升级"在自对弈里**变弱**了（三次，全部 A/B 单变量）

| 运行目录 | 配置（A = 带升级，B = 不带升级） | n | **A 胜率** | 备注 |
|---|---|---|---|---|
| `runs/20261022_100436_ab_upg1_vs_none` | quiet 评分 + `MC_UPG=1` | 64 | **23.4%**（15W-0D-49L） | `experiment_log` 里 result 字段是 `_待填_` |
| `runs/20261022_112332_ab_upg_alltgts` | quiet + `MC_UPG=1` + `UPG_TARGETS=4` | 32 | **18.8%**（6W-0D-26L） | 同上 |
| `runs/20261022_221757_ab_upg_under_net` | **net 评分** + `MC_UPG=1` + `UPG_TARGETS=4` | 32 | **31.2%**（10W-0D-22L） | **与现役评分方式一致的那一个**，`experiment_log` result 仍是 `_待填_` |

⇒ 这三组都是"**加升级候选 ⇒ 自对弈更差**"。这就是 BASE 里不设 `AZAI_MC_UPG` 的真实原因
（上游报告 §3.7 那句"历史上三次验证升级有害"指的就是它们），**不是漏配**。

对亚军本体的那几批（`cells4_upg1` / `upg1fix3`）局数不全、没有完整 `RESULT`，不能用来下结论：

| tag | 有 RESULT 的局 | 战绩 | 血量差 |
|---|---|---|---|
| `VS_RUNNERUP_cells4_upg1` | 7 | 0W-7L | −25.14 |
| `VS_RUNNERUP_upg1fix3` | 6 | 0W-6L | −29.50 |
| `VS_RUNNERUP_cells4`（对照，完整 32 局） | 32 | 1W-31L | −27.00 |

### 6.4 已有的离线探针：**升级候选在仲裁里是能被选中的**（含现役 net 评分器）

`code/my_ai/az_intent/probe_upgrade_selectability.py`（2026-10-22 写的，输入是落盘的
97 MB / 94 MB 对局 stderr，`seed=732`）本次**重跑了**（只读，~3 min），
30 个"双方都有塔"的真实中局 × 6 种仲裁评估器：

| 仲裁模式 | 菜单里含升级候选 | **升级被选中** | 平局占比 | 升级候选的中位排名 |
|---|---|---|---|---|
| `quiet`（h=256，历史默认） | 26/30 | **11/30（37%）** | 27% | 2.0 |
| **`net`（h=32，≈ 现役 `MC_SCORE=net`）** | 26/30 | **5/30（17%）** | **0%** | **3.0** |
| `quiet` + `UPG_BONUS=0.05` | 26/30 | 13/30（43%） | 23% | 1.5 |
| `quiet` + `UPG_BONUS=0.10` | 26/30 | 18/30（60%） | 19% | 1.0 |
| `quiet` + `UPG_BONUS=0.20` | 26/30 | 21/30（70%） | 4% | 1.0 |
| **`net` + `UPG_BONUS=0.10`** | 26/30 | **17/30（57%）** | **0%** | 1.0 |

三点读数：

1. **"评估器不认升级"这个解释是错的**：现役 net 评分器在有候选时**会**选中它（5/30 = 17%），
   加上 `UPG_BONUS=0.10` 就到 57%。⇒ 0.0% 的唯一原因是**没有候选**（§3.2）。
2. 但它**不偏好**升级：net 下升级候选的中位排名是 **3.0**（quiet 下 2.0），
   要过线得靠一个常数奖励（`mc_quiet.py:250-258` 就是为此写的）。
3. 这一格与 §6.3 的 31.2% 并不矛盾：**"能被选中" ≠ "选了对赢有利"**。
   这正是本文对"值不值得做"的回答：机制上做得到，收益上没证据（且现有三次 A/B 都是负的）。

---

## 7. Q4：有没有哪个开关**无意中禁用**了升级？——**没有**

`grep` 了 `code/test_match/*.sh`、`code/test_match/*.py`、`code/my_ai/az_intent/*.py` 全部脚本：

* **`AZAI_BAN_OPS=12`：未找到**
* **`AZAI_NO_UPGRADE`（及任何同义开关）：未找到**（代码里根本没有这个 env）
* `AZAI_BAN_CLASSES` 历史只出现过 `17`（禁闪电类）、`AZAI_BAN_OPS` 只出现过 `21`（禁闪电 op），
  见 `code/my_ai/az_intent/diag_*.py`、`probe_*.py` 里的 `os.environ.setdefault`。
* `AZAI_BAN_CLASSES` 的**通道号映射**上，"12" 是**类 12 = MISSILE（导弹塔）**，不是 `UPGRADE_TOWER`
  （§1 那张表）——**即便有人写过 `AZAI_BAN_CLASSES=12`，禁的也是导弹塔，不是升级**。
  这是个容易造成误判的坑，记在这里。

⇒ 升级不是被"禁"掉的，是**从来没被"允许生成"过**。

---

## 8. 最小修复方案（只提方案，未改代码）

### 8.1 方案 A（零代码，改环境变量）——**能打开升级，但已知更弱**

```
AZAI_MC_UPG=1        # bundle_mcts.py:784 → 把 UPGRADE_TOWER 显式放进 MC 菜单
AZAI_MC_UPG_TARGETS=4 # bundle_mcts.py:803 → 枚举 4 个升级目标（否则永远只升 HEAVY）
```
* 唯一入口：`az_bridge_ai.py:816`（默认 `"0"`），改环境变量即可，**一行代码都不用动**。
* 效果（实测）：`upg1fix3` 里升级 **9.0%** 的 op、二级塔 21.5%。
  离线探针（§6.4）显示：`net` 评分下这个候选被选中 5/30（17%），加 `AZAI_MC_UPG_BONUS=0.10`
  抬到 17/30（57%）——**机制上随时可调**。
* **代价/风险（已测）**：
  1. **自对弈更弱**：quiet 23.4% / 18.8%，**net 31.2%**（§6.3）⇒ 现在做等于把一个已知负收益的
     开关打开。
  2. 语义不是"成为候选"：MC 仲裁选中后**直接发这一手**（`az_bridge_ai.py:1381-1384`），
     **绕过 48 束的 flat rollout 比较**；而且 `mcts.search()` 已经跑完、结果被丢掉（白烧算力）。
  3. 只在我们**前 `mc_upg` 座塔**上生成（`_tw_list` 按 `-tower_type, x, y` 排序，即塔型高的优先），
     默认只取升级树第一个目标。
  4. 每回合额外 1-4 个候选进 MC 菜单 ⇒ 仲裁成本 +1-4 次 rollout（`MC_HORIZON=16` 很短，影响小）。

### 8.2 方案 B（**推荐**，~15 行代码）：把升级走**已有的额外候选通道**，让评估器去比

复用**注入通道**（`flat_rollout_greedy.py:417-453` 的 `set_extra_bundles` / `_collect_extras`，
现在只有 oracle / placebo 在用的那条），在 `decide()` 里 `mcts.search(...)` **之前**塞
最多 1-4 条 `((12, tower_id, target),)` 单 op 候选：

* 插入点：`code/test_match/az_bridge_ai.py:1376-1378`
  （`with _T("search"): res = mcts.search(facade, player, temperature=temperature)` 之前，
  与 `_oracle_step()` 同一位置纪律：**每回合重设**，失败 `set_extra_bundles(None)`）；
* 候选怎么挑（**确定性、可复现**，别用 RNG）：按 `(coins ≥ cost)` 过滤后，取
  `mc_upg` 座塔 × `mc_upg_targets` 个目标；`_placebo_bundle`（`az_bridge_ai.py:431-438`）
  已经写好了同样的"塔×目标"枚举，可直接照抄；
* 合法性：逐条 `facade.can_apply_operation(player, op, pending=accepted)`（**同一口径**：
  `_oracle_bundle`，`az_bridge_ai.py:1011-1036`）⇒ 有非法就丢整条，绝不发半块；
* 预算影响可算：`n_cand 48 → 49..52` ⇒ `n_rollouts = min(64, 49152//(n_cand×16))` 仍被
  `FLAT_ROLLOUTS=64` 封顶（`flat_rollout_greedy.py:520-523`），**49 候选时 49152//784=62**，
  即采样数 **−3.1%**（与上游报告 §4.1 对注入候选的算例同式）。

**为什么推荐 B 而不是 A**：A 是"MC 仲裁直接接管"，B 是"进候选、由同一个 flat rollout 评估器
和另外 48 条同台比较"——**B 才对应上游报告测出来的那个机制**
（A 臂的 4.3% 升级就是这么进来的，选中率 45.7%）。
B 还能顺带回答上游报告 §5.2 留下的替代解释：**升级这一"类"提高的到底是什么**。

### 8.3 风险清单（务必和方案一起读）

| 风险 | 说明 | 现有护栏 |
|---|---|---|
| **半块非法**（`NO_DOWNGRADE` 那一类） | 升级会**花掉金币**，同束里后续的建塔可能因此钱不够 ⇒ 整块非法 | ① 构候选时逐条 `can_apply(pending=…)`（`_oracle_bundle` 口径）；② `sample_bundle` 自带 `can_apply` 过滤（`bundle_mcts.py:309`）；③ **`_block_legal_gate`（`az_bridge_ai.py:327-356`）可直接复用**，但它现在只在"事后删 op"时才跑（`:1449`），**注入/额外候选这条路不在它的覆盖范围内** ⇒ 要么显式调用它，要么依赖 ① |
| 预算稀释 | 多 1-4 个候选 ⇒ 每候选 rollout 数 −0%~−8% | 已量化（§8.2）；`FLAT_ROUNDS` 是按回合数计的硬预算 |
| 与注入组合 | A 臂已经在用 `set_extra_bundles`，**同一条通道**里要**合并**两份（`_collect_extras` 会去重），不能覆盖 | `_collect_extras(extra_bundles, existing)` 支持多份，但调用点得改成"合并"而不是"set" |
| 收益不确定 | 三次自对弈 A/B 都是负的；对亚军本体的升级臂局数不全 | **这正是"值不值得做"的答案：值不值得"先验证一个更便宜的代理指标"** |
| 二级塔的掩码费用 bug | `decoder.py:209-213` 对"二级→三级"用 60 而非 200 ⇒ 会放进买不起的候选（随后被丢掉，不会非法） | 修它就是一行（按 `step` 而不是按当前塔型取常数）；与本文主因无关，可另开 |

### 8.4 我的建议（一句话）

**先把方案 B 当"诊断臂"跑 16 局**（不是当改进）：它把 `UPGRADE_TOWER` 从 0.0% 抬到
"由评估器裁决"的位置，且能直接读出"升级候选进束后被选中比例"——
这个数才回答上游报告 §5.2 那个没排除的替代解释（"注入的收益里有多少只是类型偏好"）。
**在拿到这个读数之前，不建议把 `AZAI_MC_UPG` 打开**（三次 A/B 都说它更弱）。

---

## 9. 复现命令（全部只读，单进程）

```bash
PY=D:/anaconda3/envs/pytorch-gpu/python.exe

# ① 出手构成 / 金币 / 塔等级 / 升级机会（纯日志，秒级）
$PY -u code/test_match/probe_why_no_upgrade.py --tag INJECT_VS_RU_B \
    --json match_results/inject_vs_ru/probe_why_no_upgrade_B.json
$PY -u code/test_match/probe_why_no_upgrade.py --tag INJECT_VS_RU_A      # 注入臂对照
$PY -u code/test_match/probe_why_no_upgrade.py --tag VS_RUNNERUP_upg1fix3  # 开了 MC_UPG 的臂
$PY -u code/test_match/probe_why_no_upgrade.py --tag VS_RUNNERUP_cells4    # 同配置历史批次

# ② 类轴（重放 16 局 + 现役 ckpt 前向；CPU 约 46 s）
$PY -u code/test_match/probe_upgrade_class_axis.py --tag INJECT_VS_RU_B \
    --games 7,7r,8,8r,9,9r,10,10r,11,11r,12,12r,13,13r,14,14r --every 1 \
    --json match_results/inject_vs_ru/probe_class_axis_B_all16.json

# ③ 已有的离线探针（升级候选在仲裁里能不能被选中，6 种评估器；~3 min）
$PY -u code/my_ai/az_intent/probe_upgrade_selectability.py
```

落盘工件：`match_results/inject_vs_ru/probe_why_no_upgrade_{A,B}.json`、
`match_results/inject_vs_ru/probe_class_axis_B_all16.json`、
`match_results/inject_vs_ru/probe_logs/{phase1_A,phase1_B,class_axis_B,upgrade_selectability}.log`。

---

## 10. 没查到 / 没覆盖的（诚实清单）

1. **`VS_RUNNERUP_cells4_upg1` / `VS_RUNNERUP_upg_net` 的确切 env 未找到**：产生它们的脚本版本
   已被覆盖（`_vs_runnerup_upg1.sh` 现在打的是 `upg1fix3`），`run_logged` 的 `meta.txt` 只有
   commit/命令，不含 env 清单。配置是**从 tag 名 + 运行时间 + commit `121b088` 的提交信息**推定的
   （"UPG 臂改用 mc_score=net"）。这两批的"0 升级"已由 `int('op')` bug 解释，不影响主结论。
2. **`ab_upg_under_net`（net + 升级）的 31.2% 只在原始 `output.log` 里**，
   `docs/experiment_log.md` 的 `result` 字段仍是 `_待填_` ⇒ 我没有把它当"已确立结论"，
   只当一条**待复现的读数**。
3. **没有跑任何新对局**（约束要求）：所以"方案 B 会不会变强"本文**没有**答案，只有设计。
4. **只统计了 `INJECT_VS_RU_{A,B}` 的逐决策金币**（只有这两个 tag 有
   `training_history/progress/<tag>/` 的逐回合 trace）；其余 tag 的金币只能用
   ladder 日志里的 `towers=` 行推塔态、拿不到逐回合金币。
5. **"类 1-15 若被钉住会怎样"是支撑集分析（可达性），不是实跑**：我验证的是"那 15 个类的
   合法格里确实有能解成升级的格子"（43.6% 的 (类,决策点) 组合），
   没有验证"抽到它的概率"（z-score softmax 的实际质量分布，`t_pos=1.0` 下 `action_map`
   在塔格上的值可能很低）。这一条是方案 B 要测的第一件事。
6. **§6.4 的 30 个局面来自单个 seed（732）的一局复盘**，只回答"能不能被选中"，
   不代表对局分布；而且 `probe_upgrade_selectability.py` 用 `iterations=1,k=1` 只跑
   `_expand`，与真实对局（K=48 + flat rollout）不是同一条完整路径。
