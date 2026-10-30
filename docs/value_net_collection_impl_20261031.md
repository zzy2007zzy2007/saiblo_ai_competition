# 值网重训数据采集：实现 + 冒烟实测（2026-10-31）

> 对应计划 `docs/value_net_retrain_collection_plan_20261031.md`（commit `bda2f02`）。
> 本文只记**实现**与**冒烟的真实数字**（不训练、不做正式采集）。
>
> 阶段调整（用户定夺）：**第一阶段 = 原版统一搜索自对弈、不注入**；`--inject-coin-rule` 接口**实现但默认关**，
> 留到第二阶段补 90 币折点。所以下面把"主线成本"和"注入接口单测"分开报。
>
> 机器：本机 CPU，单进程（`--workers 1`），C++ 引擎（`--native-engine`），`AZAI_CAND_MODE=unified`。
> 原始日志：`training_history/collect_smoke_coinrule/*/run.log`（`training_history/` 在 `.gitignore:6`，不入库）。

---

## 0. 一句话

原版统一搜索自对弈实测 **6.5 min / 200 回合（200 回合截断，400 个决策）⇒ 每决策中位 621 ms、p95 1.86 s，
外推 512 回合约 16.6 min/局**；这 1 局里 **`coins≥90` 只占 0.75% 的决策（3/400）**、**闪电 2 次**。
规则注入接口已验证可用（命中率 **50–52% ≈ p=0.5**、`inject_src` 正确落盘、**默认关时逐字节零影响**），
但**不要和 `--skip-hold-search` 混用**（见 §4）。

---

## 1. 代码改了什么（`code/my_ai/az_intent/az_selfplay.py`）

| # | 改动 | 说明 |
|---|---|---|
| 1 | `--inject-coin-rule`（**默认关**）、`--inject-coin-rule-p`（0.5）、`--inject-coin-rule-cd`（5）、`--inject-coin-rule-hold-lo`（60） | 「金币×冷却」规则注入；关着时**整段不执行**（不读 state、不建 ActionCatalog、不抽随机数） |
| 2 | `_coin_rule_decision()` / `_lightning_ops()`（模块级，纯函数、可单测） | 规则与落点选择。闪电落点**优先用 `ActionCatalog._superweapon_candidates`**（它已含"敌方蚂蚁/塔为圆心 + `_storm_value` 打分 + `can_apply_operation` 校验"，`SDK/utils/actions.py:158-176`）；被它的 `value>1.5` 门槛挡掉时，用"半径内敌方单位最多"兜底，**仍过 `can_apply_operation`** ⇒ 绝不发非法 op |
| 3 | 注入点在**动作选择之后、`apply_operation_list` 之前**，且**覆盖 skip 分支** | 我们 AI 在"钱够 90 且 cd 已好"时大概率本来就选 HOLD；若只在 search 分支注入，最该注入的那类决策反而注入不到。顺序：search → example/random 注入 → 规则注入（最后执行，覆盖前面所有结果） |
| 4 | 独立随机流 `coin_rule_rng = default_rng(seed ^ 0x5EEDC01E)` | **不消耗 `mcts.rng`** ⇒ 开关关着时对搜索随机序列零影响；开着时注入率也不被搜索抽随机干扰 |
| 5 | 样本加 `inject_src`（0=AI / 1=规则注入） | 写进 **npz**（`inject_src` int8 数组）+ **ents.jsonl**（每行一个字段） |
| 6 | 逐局统计：`[cost]`（总墙钟 / 每决策中位·p95·均值）+ `[inject]`（可注入→命中、闪电 op 数、`coins≥90` 决策数） | 采集规模的**唯一成本口径**；`collect_games_parallel` 末尾再打一行跨局汇总 |
| 7 | `--search {bundle,flat,unified}` + `--search-total-rounds` | 默认 `bundle`（旧行为逐位不变）。`unified` 实例化 `UnifiedActionSearch`（`BundleMCTS` 的子类，`search()` 签名/返回结构相同），**另外单独 load 一次 value 网**（它的 `_leaf_values` 要真 torch 模块，而 `ThreeNetPolicy` 门面没有 `.parameters()`）。<br>⚠️ 口径澄清：`UnifiedActionSearch.search()` 本身就是统一候选算法（`enumerate_by_channel` → 预算分配 → 逐候选 rollout → top-k 复评，`unified_action_search.py:195-244`），**不依赖** `AZAI_CAND_MODE`；那个 env 是 **bridge** 在 `FlatRolloutGreedy.search` 里切 pin/unified 时用的。命令里照样设 `AZAI_CAND_MODE=unified` 只是**与部署脚本保持一致**，无害也不必要。 |

**没改**：搜索算法、网络、标签生成（kgeo/abs 仍走 `build_rel_labels.py` / `az_train.py:119-129`）。

用法（**第一阶段主线**，无注入）：

```powershell
$env:AZAI_CAND_MODE="unified"; $env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe -u code/my_ai/az_intent/az_selfplay.py `
  --checkpoint checkpoints/posnet_A_k5_m32.pt --games 2000 --workers 16 --max-rounds 512 `
  --iterations 256 --max-depth-rounds 4 --k 24 --native-engine --skip-single-candidate `
  --search unified --write-npz --out-dir training_history/collect_v2_s1
```

（第二阶段再叠 `--inject-coin-rule --inject-coin-rule-p 0.5`。）

---

## 2. 冒烟 A：原版统一搜索（**第一阶段主线**）

配置：`--search unified --skip-single-candidate`、**不** skip-hold、seed 8301、`--max-rounds 200`（截断）、1 局。

| 读数 | 值 |
|---|---|
| 墙钟 | **6.5 min**（200 回合 = 400 个决策） |
| 每决策 | **中位 621 ms / p95 1.86 s / 均值 969 ms** |
| **`coins≥90` 决策** | **3 / 400 = 0.75%** |
| **闪电 op** | **2 次**（这局 200 回合） |
| 动作构成（双方合计） | BUILD 127 / DOWNGRADE 114 / UPGRADE 5 / DEFLECTOR 3 / **LIGHTNING 2**（共 251 个 op） |
| 外推 512 回合 | **≈ 16.6 min/局**（400→1024 决策线性外推） |

**读法**：
1. 统一搜索的动作构成是**重度 churn**（127 建 / 114 拆），跟 §13 的诊断一致 —— 钱在"建了又拆"里烧掉，
   所以 `coins≥90` 只有 0.75%（比 match trace 上按回合算的 3.0% 更低，因为这里按**决策**算、且两侧都是我们自己）。
2. **闪电只有 2 次**，而 200 回合里理论上限是 `2 方 × floor(200/35) ≈ 11 次`（闪电冷却 35 回合）
   ⇒ **我们只用了 ~18% 的可用次数**，头寸很大（见 §3 的"上限"一节）。
3. 每决策 p95/中位 ≈ **3.0×**，说明成本不是均匀的（候选多的局面贵）；`--search-total-rounds` 是唯一的总预算旋钮。

---

## 3. 冒烟 B：同一局 + `--skip-hold-search`（**不推荐与 unified 混用**）

同 seed、同预算，只加 `--skip-hold-search`：

| 读数 | A（原版） | B（+skip-hold） |
|---|---|---|
| 墙钟 | 6.5 min | **2.7 min**（2.4× 快） |
| 每决策 中位 / p95 | 621 ms / 1.86 s | **6 ms** / 1.73 s |
| `coins≥90` 决策 | 0.75% | **51.2%** |
| 闪电 op | 2 | 8 |
| 动作总数（200 回合、双方） | **251** | **93**（BUILD 43 / DOWN 39） |

⚠️ **`--skip-hold-search` 不能和 `--search unified` 混用**：它的等价性证明是**在 bundle 搜索上**做的
（`az_selfplay.py:403-407`：*"the search can only pick HOLD there too (measured agreement: 2 wrong skips in 10,002)"*）——
前提是"候选集由类头 argmax 决定"。**统一搜索的候选集不是这么来的**（它按通道枚举、用 rollout 值排序），
所以"heads 全投 HOLD"时它照样可能选建塔/降级。实测证据就是上表：动作数 251 → 93、`coins≥90` 0.75% → 51.2%
⇒ **skip 让 AI 被动得多，分布已经不是"原版统一搜索"了**。

⇒ **第一阶段如果要"原版统一搜索的分布"，就别加 `--skip-hold-search`**（代价是贵 2.4×）。
反过来，如果目标是"多拿 `coins≥90` 样本"，B 的 51% 很有诱惑力，但那是"AI 被钉在 HOLD"的伪分布
（第二阶段用规则注入拿同样的样本、且分布是**有意图**的，更干净）。

---

## 4. 冒烟 C/D：`--inject-coin-rule` 接口单测（**默认关、第一阶段不用**）

### 4.1 规则命中率 ≈ p（两条独立实验）

| 实验 | 配置 | 攒钱(HOLD)：可注入 → 命中 | 放闪电：可注入 → 命中 |
|---|---|---|---|
| **D**（接口主证据） | bundle 搜索、2 局 × 200 回合、seed 8101、**p=0.5** | 74 → **38（51.4%）** | 21 → **11（52.4%）** |
| **C** | 统一搜索 + skip-hold、1 局 × 200 回合、seed 8301、p=0.5 | 58 → **29（50.0%）** | 6 → 1（16.7%，n=6 太小） |

⇒ 规则按 `p` 生效（**50–52% vs p=0.5**），两个分支都验证过。

### 4.2 `inject_src` 落盘正确（C 的产物）

```
npz:  inject_src 非零 30/400  {0:370, 1:30}       # 29 次攒钱 + 1 次放闪电，与日志一致
注入样本 coins(×300) = 60,63,65,66,67,70,72,73,75,77,79,80,82,83,85,86,88,193
注入样本 cd(×35)     = 0                            # 放闪电那条；攒钱要求 cd≤5，实际都落在 0
ents.jsonl: 每行都有 inject_src 字段
```
无注入的运行（`--inject-coin-rule` 不设）：`inject_src` **全 0**（400/400、240/240 均如此）。

### 4.3 ⚠️ "闪电事件数"有**结构性上限**：冷却 35 回合

闪电 `cooldown=35`（`Ant-Game/SDK/utils/constants.py:194`）⇒ **每方每 35 回合最多 1 次**。
所以"闪电事件/局"的硬上限 ≈ `2 × rounds / 35`：**200 回合 ≈ 11 次，512 回合 ≈ 29 次**。

这解释了为什么 D 里"注入 11 次闪电"却没有把总数推高：**注入臂与基线臂都撞在 10–11 次的天花板上**
（两局都是 10、11）。**真正能涨的空间在基线**：原版统一搜索只用了 **2/11 ≈ 18%** 的可用次数。

⇒ 对 KPI "闪电 ≥1 万次"的含义：**上限 ≈ 29 次/局**（512 回合），所以
"1 万次"需要 **≥350 局（打满上限）到 5000 局（按基线 2 次/局）**；
按注入后现实的 ~12–15 次/局，**约 700–850 局**就能到 1 万次。

---

## 5. 零影响证明（`--inject-coin-rule` 不设）

做法：同 seed / 同配置（bundle 搜索、2 局 × 120 回合、seed 8201），
先用 **`git stash` 回到改动前**跑一遍（`zi_old/`），再 `git stash pop` 用新代码跑一遍（`zi_new/`），逐字段比：

```
seed 82010000: common keys=6 all-bit-identical=True  new-only keys=['inject_src']
          ents rows 240 vs 240 | 去掉 inject_src 后逐条相同 = True | inject_src 全 0 = True
seed 82010001: 同上
ZERO-IMPACT: PASS
```

- npz 的 6 个公共键（`board / stats / action_map / head_logits / player / value_target`）**逐位相同**；
- `ents.jsonl` 去掉新增的 `inject_src` 后**逐条相同**；
- 结构上也成立：注入块被 `if inject_coin_rule:` 整段包住，**关着时不抽任何随机数**（独立的 `coin_rule_rng` 也不创建）。

---

## 6. 规模推算（给"第一阶段采多少局"用）

单局 = 512 回合 ≈ 1024 个决策（按 §2 的线性外推）。

| 配置 | 每局墙钟 | 2000 局（16 workers） | `coins≥90` 样本 | 闪电事件 |
|---|---|---|---|---|
| **A 原版统一（主线）** | **≈16.6 min** | **≈ 34.6 h** | 0.75% × 1024 ≈ **7.7/局** ⇒ 2000 局 ≈ **1.5 万** | 2/200 回合 ⇒ **≈5/局** ⇒ 2000 局 ≈ 1 万 |
| B 统一 + skip-hold | ≈7.2 min | ≈ 15.0 h | 51.2% ⇒ 524/局（**伪分布**） | ≈20/局 |
| C 统一 + skip-hold + 注入 | ≈11.5 min | ≈ 24 h | 29.5% ⇒ 302/局 | ≈15/局 |

**建议（按"`coins≥90` 够用 + 耗时可控"）**：
- 若坚持 A（分布最干净）：**先采 600–800 局**（**≈10–14 h**，16 workers；每 worker ≈3.6 局/小时）。
  这已能给 **≈4.6–6.2 千个 `coins≥90` 决策点** + **≈3–4 千次闪电**，足够跑 P1 四臂对照。
- 若允许 B/C（分布被动）：12–24 h 能拿到 10 万级 `coins≥90` 点，但**训练分布与部署不一致**的风险要写进结论。
- 正式采集前请先按 §7 的"两个坑"过一遍。

---

## 7. 本次得到的两个坑（写进结论）

1. **`--skip-hold-search` × `--search unified` 不兼容**（§3）：skip 的等价性只在 bundle 搜索上验证过；
   混用会把 AI 变成"大量 HOLD"，动作数 251 → 93、`coins≥90` 0.75% → 51.2%。
   ⇒ 第一阶段若用 unified，**不要**加 `--skip-hold-search`。
2. **闪电受 35 回合冷却硬上限**（§4.3）：注入不能突破上限，只能把"没用掉的额度"用起来。
   所以"闪电事件数"不是"注入越猛越多"，而是由**局数 × 回合数**决定的天花板。

---

## 8. 局限 / 未做

- 只跑了 1 局 × 200 回合（截断）做成本测量；**没有跑满 512 回合的整局**，16.6 min/局 是线性外推
  （风险：回合越靠后棋盘越复杂，每决策可能更贵 ⇒ 外推可能偏乐观）。
- `--workers 1`：**多进程并行的实际加速比未测**（16 workers 会争内存带宽/CPU 频率；34.6 h 这个数要打折扣）。
- 注入臂只做了 2–3 局，只能验接口，不能验"对训练的作用"。
- 分类标签（kgeo/abs）**未在本次落盘**：采集只存 `board/stats/player/value_target`（+`inject_src`），
  逐帧标签仍按计划用 `build_rel_labels.py` 从 `stats[:,1]` 现算（`--save-full`），不改实现。

---

## 9. 复跑命令（本文所有数字）

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:AZAI_CAND_MODE="unified"
$PY="D:\anaconda3\envs\pytorch-gpu\python.exe"; $CK="checkpoints/posnet_A_k5_m32.pt"

# A 原版统一搜索（主线成本）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8301 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --search unified --write-npz --out-dir training_history/collect_smoke_coinrule/unified_1g

# B 加 skip-hold（对照，别当主线）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8301 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --skip-hold-search --search unified --write-npz --out-dir training_history/collect_smoke_coinrule/unified_skiphold

# C 规则注入接口单测（统一搜索）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8301 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --skip-hold-search --search unified --inject-coin-rule --inject-coin-rule-p 0.5 `
  --write-npz --out-dir training_history/collect_smoke_coinrule/unified_inject

# D 注入命中率（bundle 搜索，事件更多、更划算的接口单测）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 2 --workers 1 --seed 8101 `
  --max-rounds 200 --iterations 64 --max-depth-rounds 3 --k 16 --native-engine `
  --skip-hold-search --skip-single-candidate --write-npz --inject-coin-rule --inject-coin-rule-p 0.5 `
  --out-dir training_history/collect_smoke_coinrule
```
