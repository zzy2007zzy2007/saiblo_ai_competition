# 值网重训数据采集：实现 + 冒烟实测（2026-10-31 起，2026-11-01 修正搜索器口径）

> 对应计划 `docs/value_net_retrain_collection_plan_20261031.md`（commit `bda2f02`）。
> 本文只记**实现**与**冒烟的真实数字**（不训练、不做正式采集）。
>
> **两阶段**（用户定夺）：第一阶段 = **原版统一搜索自对弈、不注入**；`--inject-coin-rule` 接口**实现但默认关**，
> 留到第二阶段补 90 币折点。
>
> ⚠️ **2026-11-01 重要修正**：`--search unified` 之前接错了——接的是**老版** `unified_action_search.py`
> 的 `UnifiedActionSearch`（那个 0/42 的错误版）。现在已改为接**新版** `unified_cand.make_unified_searcher_class()`
> （`UnifiedCandFlatSearch`，= `az_bridge_ai.py` 里 `AZAI_CAND_MODE=unified` 用的**同一个入口**）。
> **本文所有成本/占比数字都以新版为准**；老版数字保留在 §3 只为对照，**不能用于决策**。
>
> 机器：本机 CPU，单进程（`--workers 1`），C++ 引擎（`--native-engine`）。
> 原始日志：`training_history/collect_smoke_coinrule/*/run.log`（`training_history/` 在 `.gitignore:6`，不入库）。

---

## 0. 一句话

新版统一候选（`UnifiedCandFlatSearch`）自对弈实测：**200 回合（截断）/ 400 决策 = 22.7 min**，
每决策 **中位 2.75 s / p95 9.09 s / 均值 3.40 s**；这局 `coins≥90` 占 **1.75%**（7/400）、**闪电 2 次**。
这局在第 194 回合双方血量已 16/11，按失血速率真实局长约 **235–255 回合** ⇒ **实际每局 ≈25–30 min**，
**第一阶段 512 局 ≈13–16 h（16 workers 名义）/ ≈17–21 h（按 12 路有效并行）**。
规则注入接口在新版上同样可用（命中率 **54.0% / 50.0% ≈ p=0.5**、`inject_src` 正确落盘、默认关时**逐位零影响**）。

---

## 1. 代码改了什么（`code/my_ai/az_intent/az_selfplay.py`）

### 1.1 搜索器选择（本次修正的核心）

| `--search` | 实例化 | 说明 |
|---|---|---|
| `bundle`（**默认**） | `BundleMCTS` | 旧行为，**逐位不变** |
| `unified` | **`unified_cand.make_unified_searcher_class()` → `UnifiedCandFlatSearch`** | **部署用的新版统一候选**：候选 = 类 UCB bandit + 位置游标（`UnifiedCandidateGen`），评估 = 父类 `FlatRolloutGreedy` 的 flat rollout。**与 `az_bridge_ai.py:1359-1361` 同一个入口** |
| `flat` | `FlatRolloutGreedy`（pin 候选） | 对照用 |
| `unified-legacy` | `UnifiedActionSearch`（`unified_action_search.py`） | **老版**，2026-10-31 之前那个（0/42 的错误版）；只为复现旧读数保留 |

flat/unified 家族参数**照抄 bridge 部署取值**：`--cand-k 48`（= `AZAI_K`，**候选条数**，不是 bundle 束长）、
`--flat-rollouts 16`、`--flat-horizon 64`、`--flat-batch 32`、`--search-total-rounds 39000`（= `AZAI_FLAT_ROUNDS`）、
`rollout_mode=idle` / `rollout_random_side=opponent` / `prob_mode=const` / `decay_c=0.6` / `decay_k=3.0`。
另外**单独 load 一次 value 网**（`load_three_models(ckpt)[2]`）：flat 家族的 `_leaf_values` 会调 `.parameters()`，
而 `make_net_fn_from_ckpt` 返回的 `ThreeNetPolicy` 门面没有这个属性（踩过、已修）。

**stderr 证据（新版确实被实例化）**：

```
[search] seed=83010000 cls=UnifiedCandFlatSearch(unified_cand) k=48 rollouts=16 horizon=64 batch=32
         rounds=39000 cand_mode_env='unified' cand_verbose='0'
```
`AZAI_CAND_VERBOSE=1` 时每次决策还会打（证明 `UnifiedCandidateGen` 在跑）：
```
[cand] mode=unified n_cand=13 含HOLD=1 含升级=0 decode=legacy 池=13 缓存命中=641 步数=653 形状=60(穷尽0) ...
[candq] 阶段1结束 β=3.0 c=1.0 N_total=181 合规类=17 | Q/N（按 Q 降序）: ...
```

### 1.2 「金币 × 冷却」规则注入（默认关，第一阶段不用）

| # | 改动 | 说明 |
|---|---|---|
| 1 | `--inject-coin-rule`（**默认关**）+ `-p`(0.5) / `-cd`(5) / `-hold-lo`(60) | 关着时**整段不执行**（不读 state、不建 `ActionCatalog`、不抽随机数） |
| 2 | `_coin_rule_decision()` / `_lightning_ops()`（模块级纯函数，可单测） | `coins≥90 & cd==0` ⇒ 强制闪电；`coins∈[60,90) & cd≤5` ⇒ 强制 HOLD。落点**先用 `ActionCatalog._superweapon_candidates`**（已含打分 + `can_apply_operation` 校验，`SDK/utils/actions.py:158-176`），被它的 `value>1.5` 门槛挡掉时用"半径内敌方单位最多"兜底、**仍过合法性校验** ⇒ 绝不发非法 op |
| 3 | 注入点在**动作选择之后、`apply_operation_list` 之前**，且**覆盖 skip 分支** | 顺序 search → example/random → 规则（最后执行）。对**任何**搜索器都有效（不依赖搜索器类型） |
| 4 | 独立随机流 `default_rng(seed ^ 0x5EEDC01E)` | 不消耗 `mcts.rng` ⇒ 关着时对搜索随机序列零影响 |
| 5 | `inject_src`（0=AI / 1=规则注入） | 写进 **npz**（int8 数组）+ **ents.jsonl** |
| 6 | 逐局 `[cost]` / `[inject]` 统计 | 总墙钟、每决策中位/p95/均值、可注入→命中、闪电 op、`coins≥90` 决策数；跨局再打一行汇总 |

**没改**：搜索算法、网络、标签生成（kgeo/abs 仍走 `build_rel_labels.py` / `az_train.py:119-129`）。

### 1.3 第一阶段主线命令（512 局）

```powershell
$env:AZAI_CAND_MODE="unified"; $env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe -u code/my_ai/az_intent/az_selfplay.py `
  --checkpoint checkpoints/posnet_A_k5_m32.pt --games 512 --workers 16 --max-rounds 512 `
  --iterations 256 --max-depth-rounds 4 --k 24 --native-engine --skip-single-candidate `
  --search unified --write-npz --out-dir training_history/collect_v2_s1
```
（`--search unified` 自己就实例化新版类；`AZAI_CAND_MODE=unified` 只是与部署脚本对齐，无副作用。）

---

## 2. 冒烟 A：**新版** UnifiedCandFlatSearch（第一阶段主线）

配置：`--search unified --skip-single-candidate`、**不** skip-hold、seed 8301、`--max-rounds 200`（截断）、1 局、
预算 = 部署默认（`rounds=39000`、rollouts=16、horizon=64）。

| 读数 | 值 |
|---|---|
| 墙钟 | **22.7 min**（200 回合 = 400 决策） |
| 每决策 | **中位 2753 ms / p95 9088 ms / 均值 3398 ms** |
| **`coins≥90` 决策** | **7 / 400 = 1.75%** |
| **闪电 op** | **2 次**（200 回合，双方合计） |
| 动作构成（200 回合、双方、共 445 op） | **BUILD 223 / DOWNGRADE 213 / UPGRADE 7 / LIGHTNING 2** |
| 真实局长估计 | 第 194 回合 HP=16/11，末期失血 ≈0.28 HP/回合 ⇒ **≈235–255 回合** |
| **实际每局耗时** | **≈25–30 min**（200 回合实测 22.7 min × 245/200 ≈ 27.8 min） |
| 512 局 | 单 worker **≈239 h**；16 workers 名义 **≈15 h**；按 12 路有效并行 **≈20 h** |

**读法**：
1. **比老版慢 3.5×**（老版 22.7 vs 6.5 min / 200 回合）：新版每次决策要跑 bandit 枚举 + 每候选 1 次 idle rollout +
   阶段 2 复评（`k=48` 候选），所以"每决策"从 0.97 s 涨到 3.40 s（均值）。
2. **p95/中位 = 3.3×**：候选多的局面（穷尽枚举）明显更贵，这是 `--search-total-rounds` / `--cand-k` 之外的主要方差源。
3. `coins≥90` 从老版的 0.75% 涨到 **1.75%**；闪电仍只有 **2 次**（200 回合的理论上限 ≈11 次，见 §6.2）。
4. 动作构成仍是**重度 churn**（223 建 / 213 拆 / 7 升级）——和 §13 诊断（钱在"建了又拆"里烧掉）一致。

---

## 3. 冒烟 B：老版 `UnifiedActionSearch`（**作废，只作对照**）

同 seed / 同预算（200 回合截断）：

| 读数 | 新版（有效） | 老版（作废） |
|---|---|---|
| 墙钟 | 22.7 min | 6.5 min |
| 每决策 中位 / p95 / 均值 | **2753 / 9088 / 3398 ms** | 621 / 1862 / 969 ms |
| `coins≥90` 决策 | **1.75%** | 0.75% |
| 闪电 op | 2 | 2 |
| 动作构成（共 N op / 200 回合） | 445：BUILD 223 / DOWN 213 / UP 7 / LIGHT 2 | 251：BUILD 127 / DOWN 114 / UP 5 / DEFLECTOR 3 / LIGHT 2 |

⇒ **老版的成本被低估 3.5×**，所以之前"512 局 ≈13 h"的说法**不成立**；以 §2 的 25–30 min/局 为准。

---

## 4. 规模推算（给"第一阶段 512 局"用）

| 项 | 值（新版，512 局） |
|---|---|
| 单局 | ≈25–30 min（≈490 决策） |
| 512 局 / 16 workers（名义） | **≈13–16 h** |
| 512 局 / 12 路有效并行（打 75 折） | **≈17–21 h** |
| 决策点总数 | 512 × 490 ≈ **25 万** |
| **`coins≥90` 决策点** | 1.75% ⇒ **≈4.4 千**（每局 ≈8.6） |
| **闪电事件** | ≈2.4/局 ⇒ **≈1.2 千次** |
| 数据量（实测） | 单局 npz **4.8 MB**（400 帧，`np.savez_compressed`）+ ents.jsonl 0.30 MB ⇒ **≈12 KB/帧**；512 局 × ≈490 帧 ≈ **3 GB**（压缩后，很轻） |

**KPI 对照（计划 §5）**：计划的"闪电 ≥1 万次"是**第二阶段（注入）**的目标；
第一阶段（无注入）预期只有 ≈1.2 千次闪电，但 `coins≥90` 决策点 ≈4.4 千 + 大量"钱不够/在建塔"的对照，
足以跑 P1 四臂（terminal / abs+kgeo / rel+kgeo / abs+geo-tau10）。

---

## 5. 冒烟 C：`--inject-coin-rule` 接口（默认关，第一阶段不用）

### 5.1 命中率 ≈ p（新版搜索上实测）

| 实验 | 配置 | 攒钱(HOLD)：可注入 → 命中 | 放闪电：可注入 → 命中 |
|---|---|---|---|
| **C1（新版，主证据）** | `--search unified`、`--search-total-rounds 8000`（**缩预算只为测接口**）、200 回合、p=0.5 | 50 → **27（54.0%）** | 6 → **3（50.0%）** |
| C2（新版，短） | 同上、60 回合、p=0.5 | 6 → 4（66.7%，n=6） | 0（60 回合内没出现 ≥90&cd==0） |
| D（bundle，大样本） | bundle 搜索、2 局 × 200 回合、p=0.5 | **74 → 38（51.4%）** | **21 → 11（52.4%）** |

⇒ 规则按 `p` 生效；**接口对任何搜索器都有效**（注入发生在动作选择之后，与搜索器类型无关）。

### 5.2 `inject_src` 落盘（C1 的产物）

```
npz: inject_src 非零 30/400 {0:370, 1:30}          # 27 次攒钱 + 3 次放闪电，与日志一致
注入样本 coins(×300) ∈ [60.0, 130.0]                # 攒钱 60–89、放闪电 ≥90
注入样本 cd(×35) ∈ {0,2,3,4,5}                      # 攒钱允许 cd≤5；放闪电要求 cd==0
ents.jsonl: 每行都有 inject_src
```
无注入运行（不设 `--inject-coin-rule`）：`inject_src` **全 0**。

### 5.3 零影响（`--inject-coin-rule` 不设 + `--search bundle` 默认路径）

做法：同 seed / 同配置（bundle、2 局 × 120 回合、seed 8201），
先 `git stash` 回**改动前**（commit `b558840`）跑一遍（`zi_old2/`），再 `git stash pop` 用当前代码跑（`zi_new2/`），逐字段比：

```
az_selfplay_seed82010000.npz: npz 公共键 7 个（含 inject_src）逐位相同=True | ents 240/240 逐条相同=True
az_selfplay_seed82010001.npz: 同上
ZERO-IMPACT: PASS
```

> 注意：上面这条只覆盖 `--search bundle`（默认路径）。`--search unified` 是**新代码路径**（新版类），
> 无"改动前"可对照；它的正确性由 §2 的 `[search] cls=...` + `[cand] mode=unified` 日志保证。

---

## 6. 两个坑（写进结论）

### 6.1 `--skip-hold-search` 不能和统一搜索混用

它的等价性只在 **bundle 搜索**上验证过（`az_selfplay.py:403-407`：*"the search can only pick HOLD there too
(measured agreement: 2 wrong skips in 10,002)"*），前提是"候选集由类头 argmax 决定"。
**统一候选不是这么来的**（bandit 按类采样 + 位置游标枚举）⇒ heads 全投 HOLD 时它照样可能选建塔/降级。

**老版上实测**（同 seed 8301、200 回合）：动作数 **251 → 93**、`coins≥90` **0.75% → 51.2%**、
闪电 2 → 8、墙钟 6.5 → **2.7 min**（快 2.4×，但分布已不是统一搜索）。
新版机制相同（候选由 `UnifiedCandidateGen` 生成，不由类头 argmax 决定），**未实测**，但结论方向一致：
⇒ **第一阶段不要加 `--skip-hold-search`**。

### 6.2 闪电有 35 回合冷却硬上限

闪电 `cooldown=35`（`Ant-Game/SDK/utils/constants.py:194`）⇒ **每方每 35 回合最多 1 次**。
"闪电事件/局"的硬上限 ≈ `2 × rounds / 35`：200 回合 ≈11 次、512 回合 ≈29 次、**真实局长 245 回合 ≈14 次**。
实测新版只用了 **2 次 / 200 回合（≈18% 的可用额度）** ⇒ 注入能涨的是"没用掉的额度"，不能突破上限。

---

## 7. 局限 / 未做

1. **只跑了 1 局 × 200 回合（截断）**：真实局长（≈245 回合）与"跑满 512 回合"都没跑过；
   25–30 min/局 是**用 200 回合的实测 + 失血速率推的**（不是直接测的整局）。
2. **多进程加速比未测**：16 workers 的实际加速会低于 16×（内存带宽/CPU 频率）；本文给了 75% 折的区间。
3. **数据体量**：按 1 局 4.8 MB（400 帧、压缩后）外推，512 局 ≈ **3 GB**（很轻；实测单局已确认，总量未实测）。
4. 注入臂只做了 3 局（1 长 2 短），只能验接口，不能验"对训练的作用"。
5. **旧读数作废**：任何引用"6.5 min/200 回合、coins≥90 0.75%"的地方都要改成 §2 的新版数字。

---

## 8. 复跑命令（本文所有数字）

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:AZAI_CAND_MODE="unified"
$PY="D:\anaconda3\envs\pytorch-gpu\python.exe"; $CK="checkpoints/posnet_A_k5_m32.pt"

# A 新版统一候选（主线成本；~23 min）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8301 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --search unified --write-npz --out-dir training_history/collect_smoke_coinrule/ucand_1g

# A' 证明是哪个类 + 每次决策的候选读数
$env:AZAI_CAND_VERBOSE="1"
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --max-rounds 4 --native-engine `
  --skip-single-candidate --search unified --out-dir training_history/collect_smoke_coinrule/unified_cand_probe

# B 老版（作废对照）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8301 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --search unified-legacy --write-npz --out-dir training_history/collect_smoke_coinrule/unified_1g

# C1 注入接口（新版、缩预算只为测命中率）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8403 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --search unified --search-total-rounds 8000 `
  --inject-coin-rule --inject-coin-rule-p 0.5 --write-npz `
  --out-dir training_history/collect_smoke_coinrule/ucand_inject200
```
