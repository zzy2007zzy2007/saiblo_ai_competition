# 值网重训数据采集：实现 + 冒烟实测（2026-10-31 起；2026-11-01 修正搜索器/配置/落盘）

> 对应计划 `docs/value_net_retrain_collection_plan_20261031.md`（commit `bda2f02`）。
> 本文只记**实现**与**冒烟的真实数字**（不训练、不做正式采集）。
>
> 三条修正线（都发生在这两天）：
> 1. **搜索器**：`--search unified` 之前接成了**老版** `unified_action_search.UnifiedActionSearch`（0/42 那个）；
>    现在接的是**新版** `unified_cand.make_unified_searcher_class()` → `UnifiedCandFlatSearch`
>    （= `az_bridge_ai.py:1359-1361` 里 `AZAI_CAND_MODE=unified` 用的**同一个入口**）。老版降级为 `--search unified-legacy`。
> 2. **阶段 2 参数**：之前用的是"轻配置"（rollouts=16 / horizon=64 / rounds=39000 / **idle**），
>    与部署 BASE **不一致**；现在逐项对齐 `_us_vs_external.sh:68` 的 `BASE`
>    （rollouts=64 / horizon=16 / rounds=49152 / **random + opponent + decay(1.0,3.0)** / share_prefix=1 / seed=53 / batch=32 / k=48）。
> 3. **落盘**：除了 `board+stats`，现在还存**每决策的候选 + 两阶段打分**（`.cand.jsonl` + npz 紧凑字段）
>    和**官方 replay**（`seed + 每回合双方完整 ops`，可逐位重放）。
>
> 机器：本机 CPU，单进程（`--workers 1`），C++ 引擎（`--native-engine`），`AZAI_CAND_MODE=unified`。
> 原始日志：`training_history/collect_smoke_coinrule/*/run.log`（`training_history/` 在 `.gitignore:6`，不入库）。

---

## 0. 一句话

**部署 BASE 全配置**（rollouts=64 / horizon=16 / rounds=49152 / random+opponent+decay(1,3) / share_prefix=1 / seed=53）下，
新版统一候选自对弈 1 局 × 200 回合（**400 决策**）实测 **26.3 min**，每决策 **中位 2.61 s / p95 10.7 s / 均值 3.95 s**；
这局 `coins≥90` 占 **2.2%（9/400）**、闪电 **2 次**、动作 BUILD 69 / DOWNGRADE 69 / UPGRADE 14 / LIGHTNING 2。
**每决策的候选列表 + 阶段1/阶段2 分数 + Q/N 已 100% 落盘**；**官方 replay 用引擎重放 400/400 个决策逐位一致（0 mismatch）**；
**被选得多的形状，游标 t 中位=12（66% 卡满 `steps_per_shape`）**，且 27% 的决策把 `steps_total=1024` 用满。

---

## 1. 代码改了什么（`az_selfplay.py` / `unified_cand.py` / 新 `verify_replay.py`）

### 1.1 搜索器与参数：逐项对齐部署 BASE

`_us_vs_external.sh:68` 的 `BASE`（原文）：

```
AZAI_SEARCH=flat; AZAI_K=48; AZAI_FLAT_BATCH=32; AZAI_FLAT_HORIZON=16;
AZAI_FLAT_SHARE_PREFIX=1; AZAI_FLAT_ROLLOUTS=64; AZAI_FLAT_ROLLOUT_MODE=random;
AZAI_FLAT_RANDOM_SIDE=opponent; AZAI_FLAT_PROB_MODE=decay;
AZAI_FLAT_DECAY_C=1.0; AZAI_FLAT_DECAY_K=3.0; AZAI_FLAT_ROUNDS=49152; AZAI_FLAT_SEED=53
```

对应到采集脚本（**默认值现在就是 BASE**，另有同名 flag 可覆盖）：

| BASE 环境变量 | 采集侧 | 默认值 | 说明 |
|---|---|---|---|
| `AZAI_SEARCH=flat` + `AZAI_CAND_MODE=unified` | `--search unified` | `unified`（本次） | 实例化 `UnifiedCandFlatSearch` |
| `AZAI_K=48` | `--cand-k` | **48** | **候选条数**（不是 bundle 束长 `--k`） |
| `AZAI_FLAT_ROLLOUTS=64` | `--flat-rollouts` | **64** | 每候选 rollout 次数 |
| `AZAI_FLAT_HORIZON=16` | `--flat-horizon` | **16** | rollout 视界 |
| `AZAI_FLAT_BATCH=32` | `--flat-batch` | **32** | 叶值网批大小 |
| `AZAI_FLAT_ROLLOUT_MODE=random` | `--flat-mode` | **random** | ⚠️ 阶段 2 **必须 random**；`idle` 只是 unified_cand **阶段 1** 的候选打分（写死在 `cand_gen.rollout_fn`） |
| `AZAI_FLAT_RANDOM_SIDE=opponent` | `--flat-random-side` | **opponent** | 只对手随机出招 |
| `AZAI_FLAT_PROB_MODE=decay` | `--flat-prob-mode` | **decay** | p(r)=C·exp(-r/k) |
| `AZAI_FLAT_DECAY_C/K=1.0/3.0` | `--flat-decay-c/-k` | **1.0 / 3.0** | |
| `AZAI_FLAT_SHARE_PREFIX=1` | `--flat-share-prefix` | **1** | rollout 前缀树共享 |
| `AZAI_FLAT_ROUNDS=49152` | `--search-total-rounds` | **49152**（0 ⇒ 用这个） | 每决策 rollout 总回合预算 |
| `AZAI_FLAT_SEED=53` | `--flat-seed` | **53** | rollout 随机种子 |

**stderr 证据（本次冒烟原样）**：

```
[search] seed=85010000 cls=UnifiedCandFlatSearch(unified_cand) k=48 rollouts=64 horizon=16 batch=32
         rounds=49152 mode=random side=opponent prob=decay(1,3) seed=53 share_prefix=1
         cand_mode_env='unified' cand_verbose='1'
```

`AZAI_CAND_VERBOSE=1` 时另有每次决策的 `[cand] mode=unified n_cand=…` / `[candq] 阶段1结束 …` / `[candcursor] …` 三行 ⇒ 确认是 `UnifiedCandidateGen` 在跑。

### 1.2 A：每决策存「候选 + 两阶段打分」

变长数据放不进 npz 的矩形数组，所以拆成两层（都与 npz 行 **1:1 同序**）：

- **`az_selfplay_seedNNNNN.cand.jsonl`**（每决策一行）：
  `round / player / skipped / inject_src / n_cand / cand / s1 / s2 / chosen_idx / chosen / root_value / Q / N / N_total / has_hold / n_upg`
  · `cand` = 进阶段 2 的 top-k 候选（每条是 `[op_type, arg0, arg1]` 的列表；HOLD = `[]`）
  · `s1` = 阶段 1（确定性 idle rollout）分，与 `cand` 逐条对齐
  · `s2` = 阶段 2（flat 随机化 rollout）分，与 `cand` 逐条对齐（长度对不上时写 `null`，宁缺勿错）
  · `Q`/`N` = 阶段 1 结束时每个**合规类**的分数与计数（`N_total` 同）
- **npz 紧凑字段**（便于快速过滤，不用解 JSON）：`n_cand`(i16) / `chosen_idx`(i16) /
  `chosen_s1`(f32) / `chosen_s2`(f32) / `root_value`(f32) / `search_skipped`(i8) / `inject_src`(i8)

开关：`--dump-cand 1`（默认开）。

### 1.3 B：官方 replay（`seed + 每回合双方完整 ops`）

`az_selfplay_seedNNNNN.replay.json`：

```json
{"format": "antwar2-replay/1", "engine": "cpp_native", "collector": "az_selfplay",
 "seed": 85010000,
 "config": {"search": "unified", "cand_k": 48, "flat_rollouts": 64, "flat_horizon": 16, "...": "..."},
 "steps": [{"round": 0, "op0": [[11,4,2],[11,5,11]], "op1": []}, "..."],
 "result": {"winner": null, "hp": [25,25], "rounds": 200, "terminal": false, "value_target_p0": 0.0}}
```

- 语义照 `其他版本ai/AAArena规则/antwar2/replay_format.md`：**每轮 `op0/op1` 是该角色同轮被接受的有序原子操作**
  （`11 建 / 12 升 / 13 降 / 21–24 超武 / 31–32 基地升级`；**空列表 = HOLD**），公开状态对称可见。
- 写的是**实际被执行**的 ops（含 `--inject-coin-rule`/随机注入的结果）⇒ 重放 == 这一局。
- ⚠️ 与"progress trace 重放"的区别：trace 是**有损**的（只记 HP/金币/op 名字，缺精确塔/蚁状态），
  历史实测 144 局只有 2 局能复现终局血；replay 记的是**完整 op 参数 + seed**，引擎确定性 ⇒ 可精确重建。
- 校验脚本：`code/test_match/verify_replay.py`（新增），逐决策比对 `coins / hp_us / hp_opp`，
  并核对终局 `winner/hp/rounds` 与 npz `value_target`。

开关：`--dump-replay 1`（默认开）。

### 1.4 C：游标推进深度诊断（`AZAI_CAND_VERBOSE=1` 才打，默认零影响）

- `unified_cand.py` 的 `UnifiedCandFlatSearch._emit_candcursor()`：每决策一行
  `[candcursor] shapes=N 本决策 t 合计=… 前8=<形状>:t=…`；同时把 `(形状 → 每决策 t)` 累计进 `cursor_agg`。
- `az_selfplay.py` 收局时打 `[candcursor-agg]` 汇总表（top-20 形状：累计步 / 决策数 / t 中位 / t 最大 / t_p90 / 卡满占比）。
- `t` 的定义：该形状在**本决策**被推进的步数 = `Σ over 每次被选中`（每次最多 `steps_per_shape=12`）。

---

## 2. 冒烟 A：部署 BASE 全配置（1 局 × 200 回合，截断）

seed 8501，`--search unified`（默认即 BASE），`--skip-single-candidate`，`--max-rounds 200`，1 worker。

| 读数 | 值 |
|---|---|
| 墙钟 | **26.3 min**（200 回合 = 400 决策） |
| 每决策 | **中位 2606 ms / p95 10683 ms / 均值 3950 ms** |
| `coins≥90` 决策 | **9 / 400 = 2.2%** |
| 闪电 op | **2 次**（200 回合、双方合计） |
| 动作构成（共 154 op） | **BUILD 69 / DOWNGRADE 69 / UPGRADE 14 / LIGHTNING 2** |
| 落盘 | npz 13 键（含 `n_cand/chosen_*/root_value`）、`.cand.jsonl` 400 行、`.ents.jsonl` 400 行、`.replay.json` 200 步 |
| 数据量 | ≈**17 MB/局**（npz ≈14 MB + cand.jsonl ≈3 MB + ents 0.3 MB + replay 0.1 MB） |

**与"轻配置"（rollouts=16/horizon=64/rounds=39000/idle）对比**：轻配置 22.7 min、中位 2.75 s ——
也就是说**改成部署 BASE 只慢约 16%**（`rollouts×horizon` 的乘积都是 1024，成本近似不变），
**没有理由再用轻配置**。

**每局耗时外推**（本局 200 回合时双方 HP 还是 25/25 ⇒ 会打满 512，所以按局长给区间）：

| 假设局长 | 每局耗时（按均值 3.95 s/决策） |
|---|---|
| 245 回合 | **≈32 min** |
| 350 回合 | **≈46 min** |
| 512 回合（打满） | **≈67 min** |

**256 局 @ 12 workers**（名义，未计并行效率）：

| 平均局长 | 总时长（12 路名义） | 打 75 折（≈9 路有效） |
|---|---|---|
| 245 回合（32 min/局） | **≈11.4 h** | ≈15 h |
| 350 回合（46 min/局） | **≈16.4 h** | ≈22 h |
| 512 回合（67 min/局） | **≈23.8 h** | ≈32 h |

> ⚠️ 26.3 min 这个测量是**开着 `AZAI_CAND_VERBOSE=1`** 的（每决策 3 行日志 + `[candq]` 排序）；
> 正式采集建议 `AZAI_CAND_VERBOSE=0`（会快一点，未单独测；估计 <5%）。

---

## 3. A 落盘验证（候选 + 两阶段打分）

`.cand.jsonl`（400 行）实测：

```
样例 round=0 player=0 n_cand=15
  cand[:2] = [[[11,4,2],[11,5,11]], [[11,4,3],[11,5,11]]]
  s1[:4]   = [0.0272, 0.0256, 0.0210, 0.0208]
  s2[:4]   = [0.0088, 0.0026, 0.0016, 0.0008]
  chosen_idx=0 chosen=[[11,4,2],[11,5,11]] root_value=0.0088
  Q 条目=17  N 条目=17  N_total=280
有 cand 的行 = 100% ; 有 s1 = 100% ; 有 s2 = 100%
```
npz 紧凑字段：`n_cand` min=1 max=48、`chosen_s1` ∈ [−0.243, 0.700]、`chosen_s2` ∈ [−0.053, 0.721]、
`root_value` 与 `chosen_s2` 一致（`root_value` = 选中项的阶段 2 分）。

⇒ **"评估器当时看到了什么、怎么打的分"已可离线复现，不用重跑搜索。**

---

## 4. B 落盘验证（官方 replay 逐位重放）

```
python code/test_match/verify_replay.py <...>/az_selfplay_seed85010000.replay.json
→ steps=200, checked_decisions=400, mismatches=0, first_mismatch=null
  final_hp=[25,25] final_winner=null final_rounds=200 terminal=false
  expected={winner:null, hp:[25,25], rounds:200, value_target_p0:0.0}
  value_target_npz_p0=0.0  value_target_replayed_p0=0.0
  ok=true   [verify] 1/1 通过
```

**400/400 个决策的 `coins/HP` 全部逐位一致，终局 HP/winner/rounds 与 npz `value_target` 全部一致** ⇒
replay 可靠（seed + 完整 ops 足以精确重建任意时刻）。

---

## 5. C 游标推进深度（"是不是太浅"的第一组数字）

### 5.1 每决策的游标推进总量

| 指标 | 值 |
|---|---|
| 每决策 t 合计（400 个决策） | **中位 22** / p90 **1024** / p99 1024 / max 1024 |
| **用满 `steps_total=1024` 的决策** | **27.2%** |
| 只推进 ≤48 步（≈`dry_limit=max(32,k)=48` 量级）的决策 | **72.0%** |

⇒ **两种局面**：
- **27% 的决策把 1024 步全烧完**（候选空间大、钱够）⇒ 这些是**被总预算卡住**的；
- **72% 的决策只走 ~22 步就停**（位置空间小 / `dry` 早停）⇒ 这些**不是预算问题**，是可达空间本来就小。

### 5.2 被选得多的形状，游标推到几格？（只看 N 大的，不混入 t=0 的形状）

聚合表（`steps_per_shape=12`，`形状数=1478`，**被选 ≥5 步的形状=1439**；下表按累计步数降序 top-20）：

| 形状 | 累计步 | 决策数 | t 中位 | t 最大 | t_p90 | 卡满占比 |
|---|---|---|---|---|---|---|
| 降级+降级+降级 | 2820 | 316 | 8 | **64** | 27 | 22% |
| 降级+降级 | 2167 | 310 | 4 | 16 | 16 | 14% |
| 降级 | 801 | 302 | 3 | 12 | 4 | 1% |
| 空（HOLD） | 294 | 294 | 1 | 1 | 1 | 0% |
| c1+c6+降级 | 255 | 26 | **12** | 14 | 12 | 58% |
| c6+降级 | 221 | 22 | **12** | 15 | 12 | 59% |
| c2+降级 | 218 | 26 | 11 | 12 | 12 | 38% |
| c9+升级+降级 | 209 | 23 | **12** | 12 | 12 | **61%** |
| c1+c11 | 209 | 19 | **12** | 12 | 12 | **68%** |
| c3+c5 | 198 | 18 | **12** | 15 | 14 | **72%** |
| c1+c10 | 196 | 17 | **12** | 17 | 16 | **82%** |
| c11+c14+c15 | 195 | 17 | **12** | 12 | 12 | **82%** |
| c9+c15 | 193 | 16 | **12** | 19 | 12 | **88%** |

**结论：被选得多的形状（除"降级组合"外）t 中位就是 12，66%（样本加权）卡满 `steps_per_shape`。**
- **卡在 12，不是更浅** ⇒ 加深 `AZAI_CAND_STEPS` 对这批形状**直接有空间**（38%–88% 的实例会把新预算吃掉）。
- 但 **27% 的决策已被 `steps_total=1024` 卡住** ⇒ 光抬 `steps_per_shape` 没用，**要同时抬 `AZAI_CAND_STEPS_TOTAL`**（如 2048/4096）。
- ⚠️ **更关键的发现**：累计步数 top3 全是**降级组合**（2820+2167+801 = 5788 步，占绝对多数），
  而"建塔/升级"类形状每个只有 ~190–255 步。**步数预算被"降级 churn"吃掉了**。
  ⇒ 单纯加步数会**先加深降级**（churn 更多），想让"建塔/升级"变深必须改**配额/采样**（算法改动，本次不做）。

---

## 6. 注入接口（`--inject-coin-rule`，默认关，第一阶段不用）

- 新版搜索上实测（p=0.5）：攒钱 **50 → 27（54.0%）**、放闪电 **6 → 3（50.0%）**；短跑 60 回合 6→4；
  bundle 大样本 74→38（51.4%）、21→11（52.4%）。
- `inject_src` 落盘：30/400（27 攒钱 + 3 放闪电）；注入样本 coins∈[60,130]、cd∈{0,2,3,4,5}。
- 零影响：同 seed/同配置 vs 改动前（commit `b558840`），npz 7 键逐位相同、ents jsonl 逐条相同（PASS）。
- ⚠️ 两个坑：`--skip-hold-search` **不要**与统一搜索混用（等价性只在 bundle 上验证过）；
  闪电受 **35 回合冷却**硬上限（每方每 35 回合 1 次）。

---

## 7. 精确 relaunch 命令（256 局 @ 12 workers，含 A/B 落盘）

```powershell
cd D:\2026智能体大赛_新2_distill
$env:PYTHONIOENCODING="utf-8"
$env:AZAI_CAND_MODE="unified"      # 与部署 BASE 一致（--search unified 自己就实例化新版类）
$env:AZAI_CAND_VERBOSE="0"         # 正式采集关掉诊断日志（默认就是 0，显式写一次防环境残留）

D:\anaconda3\envs\pytorch-gpu\python.exe -u code/my_ai/az_intent/az_selfplay.py `
  --checkpoint checkpoints/posnet_A_k5_m32.pt `
  --games 256 --workers 12 --seed 900 `
  --max-rounds 512 --iterations 256 --max-depth-rounds 4 --k 24 `
  --native-engine --skip-single-candidate --search unified --write-npz `
  --cand-k 48 --flat-rollouts 64 --flat-horizon 16 --flat-batch 32 `
  --flat-mode random --flat-random-side opponent --flat-prob-mode decay `
  --flat-decay-c 1.0 --flat-decay-k 3.0 --flat-share-prefix 1 --flat-seed 53 `
  --search-total-rounds 49152 `
  --dump-cand 1 --dump-replay 1 `
  --out-dir training_history/collect_v2_s1
```

- 上面每一条 `--flat-*` / `--cand-k` / `--search-total-rounds` 都与 BASE 相同（**其实都是默认值**，
  写全是为了"命令行自带证据"，不让默认值漂移影响复现）。
- `--seed 900` ⇒ 实际 seeds = `9000000 … 9000255`（脚本用 `seed*10000 + i`）。
- 建议先 `--games 4 --workers 4` 跑一小批，确认**每局耗时/落盘体量**再放大。
- 落盘体量：≈17 MB/局 ⇒ **256 局 ≈ 4.4 GB**。
- 采完后用 `python code/test_match/verify_replay.py --dir training_history/collect_v2_s1 --quiet`
  抽查 replay（每局几秒），再跑 `build_rel_labels.py` 算 kgeo/abs 标签。

---

## 8. 局限 / 未做

1. **只跑了 1 局 × 200 回合（截断）**：本局在 200 回合时双方 HP 25/25 ⇒ 真实会打满 512 回合，所以
   "每局 32/46/67 min" 是按 3.95 s/决策 × 2×回合数外推的**区间**，不是整局实测。
2. **多进程加速比未测**（12 workers 的实际效率）。
3. **注入只验接口**（默认关），不参与第一阶段。
4. 游标诊断需要 `AZAI_CAND_VERBOSE=1`（约 3 行/决策）；正式采集关掉即可，不影响落盘的候选/分数。
5. 旧读数作废：任何"6.5 min/200 回合"（老版搜索）或"22.7 min"（轻配置）都不能用于决策；
   决策用 §2 的 **26.3 min / 200 回合（BASE 全配置）**。

---

## 9. 复跑命令（本文所有数字）

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:AZAI_CAND_MODE="unified"; $env:AZAI_CAND_VERBOSE="1"
$PY="D:\anaconda3\envs\pytorch-gpu\python.exe"; $CK="checkpoints/posnet_A_k5_m32.pt"

# BASE 全配置冒烟（~26 min，含 cand/replay/cursor 三份诊断）
& $PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint $CK --games 1 --workers 1 --seed 8501 `
  --max-rounds 200 --iterations 256 --max-depth-rounds 4 --k 24 --native-engine `
  --skip-single-candidate --search unified --write-npz `
  --out-dir training_history/collect_smoke_coinrule/ucand_deploy1g

# replay 逐位校验
& $PY -u code/test_match/verify_replay.py `
  training_history/collect_smoke_coinrule/ucand_deploy1g/az_selfplay_seed85010000.replay.json
```
