# 叶子金币项 α 扫描：给 flat rollout 叶子分数加"己方金币"权重（2026-11-02）

> 状态：**实现 + 离线预检已完成；对局扫描进行中**（结果见 §4，跑完回填）。
> 一句话：`AZAI_LEAF_COIN_ALPHA=α` 把 flat rollout 叶子分数从 `V(leaf)` 改成
> `V(leaf) + α·(my_coins/300)`，看"让金币在叶子评估里更值钱"能不能让统一搜索
> 更爱攒钱/放闪电、从而在打 pin 时更接近 pin 的强度。

## 0. 动机

值网对金币是**弱线性**读数：`∂V/∂(coins/300) ≈ +0.037`（`docs/value_net_attention_20261031.md`
里 `coins[me]/300` 的实测梯度；也写作 "+0.037/币"）。这导致统一搜索（unified）在
阶段 2 的叶子评估里**读不懂"塔的长期价值"**，把"降级退款/空过攒钱"判得比"建塔"更值，
叠上"建塔→变穷→合法动作坍缩成 {降级, HOLD}"的贫困陷阱，形成建塔↔降级的烧钱循环
（`docs/unified_rootcause_deepdive_20261101.md`）。

本实验做最直接的干预：**在叶子评估处人为加大金币权重**，看能不能让搜索"更爱攒钱、
（进而）攒到 90 放闪电"，从而打 pin 更强。α 为正值 = 让金币更值钱。

## 1. 实现（默认 0 = 零影响）

- `code/my_ai/az_intent/flat_rollout_greedy.py`：
  - `FlatRolloutGreedy.__init__` 新增 `leaf_coin_alpha: float = None`（None ⇒ 读
    `AZAI_LEAF_COIN_ALPHA`，不设 = `"0"`）。
  - `_leaf_values` 末尾：`if self.leaf_coin_alpha != 0.0:` 对**每个叶子**（含终局叶子）加
    `α · (state.coins[root_player] / 300.0)`。
  - `my_coins/300` 与值网输入 stats[22]（`Ant-Game/SDK/utils/features.py:227`
    `state.coins[player]/300.0`）**同尺度**，故 α 的量纲 = 每单位 coins/300 的 V 增量
    （V 单位 = 1，×20 = HP）。
  - `_tree_rollout_value` 在 `_dump_tree=True` 时额外 dump `leaf_coins`（每条 rollout 叶子
    的己方金币，只读诊断，供离线预检用；默认路径零影响）。
- `code/my_ai/az_intent/unified_cand.py`：`[unified] engine=flat ...` 日志行加
  `leaf_coin_alpha={self.leaf_coin_alpha}`（证明 α 真的传进去了）。
- **默认 α=0 ⇒ 与现役逐字一致，不影响现役**（`if leaf_coin_alpha != 0.0` 整段跳过）。

## 2. 离线预检（跑对局扫描之前先做的便宜实验）

**做法**（`code/test_match/offline_leaf_coin_precheck.py`）：从 `training_history/collect_v2_s1/*.cand.jsonl`
抽三类"纯"局面各 20 个（上一决策 chosen = 纯升级 op12 / 纯建塔 op11 / 纯闪电 op21，共 60 个），
复用 `dump_rollout_tree.py` 的 `replay_state` 重放重建决策点，再用
`FlatRolloutGreedy._tree_rollout_value(_dump_tree=True)` 按采集同 rng 重跑每个候选的 rollout，
读 `leaf_coins` 取 `mean(coins)/300`。值网叶子分**不重算**（α=0 基线直接取 cand.jsonl 记录的 s2），
因为 `s2'(α) = s2 + α·mean(coins)/300` 线性可加 ⇒ 对 α 网格做**解析扫描**（等价逐 α 重跑，但不浪费）。

**结果**（top-1 ≠ chosen 的占比；"翻转 α" = top-1 首次偏离当时选择的 α）：

| α | 升级局面 | 建塔局面 | 闪电局面 |
|---|---|---|---|
| 0.037（=值网自带 1×） | 15% | 25% | 0% |
| 0.10 | 50% | 40% | 5% |
| 0.30 | 85% | 80% | 20% |
| 0.50 | 95% | 100% | 40% |
| 1.0 | 100% | 100% | 75% |

翻转 α（中位）：升级 **0.125**、建塔 **0.175**、闪电 **1.0**（闪电明显更"抗压"）。
翻转去向：升级→HOLD 12/20（其余降级3/建塔2/其它3）；建塔→HOLD 11/20 + 建塔+降级 7/20；
闪电→HOLD 12/20（其余降级/降级+闪电/升级+降级）。

**离线预检结论**：
1. **方向部分对**：+α 确实让搜索停止无脑升级/建塔（→ HOLD 攒钱），且闪电比升级/建塔
   "抗压"得多（中位翻转 α = 1.0 vs 0.13/0.18）⇒ 中等 α（≈0.07~0.15）能压掉大半升级/建塔、
   只伤 ~5-10% 闪电，这是"甜区"。
2. **关键张力坐实**：翻转去向是 **HOLD/降级（囤币/退款），不是"放更多闪电"**；α≥0.5 起
   闪电被明显压掉（40%+）。所以 +α 会让搜索"更攒钱/更囤币"，但**不会让它放更多闪电**，
   反而在高 α 抑制闪电。
3. **α 选择依据**：值网自带 +0.037 只有 15-25% 翻转 ⇒ 太小；甜区 0.07~0.15；α=0.3 是
   "开始误伤闪电"的上界。⇒ 对局扫描维持 6 点 `{0, 0.02, 0.037, 0.07, 0.15, 0.30}`（覆盖
   "太小 / 甜区 / 上界"三段），不必加 0.5+。

## 3. 对局扫描设计

| 项 | 值 |
|---|---|
| A 臂 | `AZAI_CAND_MODE=unified` + `AZAI_LEAF_COIN_ALPHA=<α>` |
| B 臂 | `AZAI_CAND_MODE=pin`（现役 pin+flat，BASE 自带 random rollout） |
| 两臂共同 | 部署 BASE（`_us_vs_external.sh:68` 逐字）：ckpt=posnet_A_k5_m32.pt、AZAI_SEARCH=flat、K=48、rollouts=64、horizon=16、batch=32、random+opponent+decay(1,3)、rounds=49152、seed=53、share_prefix=1、eval=net |
| α | `{0, 0.02, 0.037, 0.07, 0.15, 0.30}`（6 个） |
| 对局 | 每 α 8 局 = seeds 100..103 × 2 座位（镜像配对，抵消先手偏） |
| 并行 | `--jobs 6`（α 之间串行） |
| 驱动 | `code/test_match/_leaf_coin_alpha_scan.sh` → `ab_search_modes.py`（ai0/ai1 都指 az_bridge_ai.py，各自 env） |
| 记录 | `bash code/run_logged.sh leaf_coin_alpha_scan bash code/test_match/_leaf_coin_alpha_scan.sh`（WMI detached 启动） |
| 有效性 | 只算 `verdict=engine` |

## 4. 结果（**待回填**，扫描完成后由 `analyze_leaf_coin_alpha.py` 产出）

### 4.1 α 扫描表（胜率 / 血差）

| α | n | 胜率（W-L） | Wilson 95% | Δhp 配对 mean [t95] | 攒90% A | 闪电 A | HOLD% A | 建塔 A | 降级 A |
|---|---|---|---|---|---|---|---|---|---|
| 0 | | | | | | | | | |
| 0.02 | | | | | | | | | |
| 0.037 | | | | | | | | | |
| 0.07 | | | | | | | | | |
| 0.15 | | | | | | | | | |
| 0.30 | | | | | | | | | |

（A=unified+α；B=pin 的攒90%/闪电/HOLD% 作为对照一并出，见分析输出。）

### 4.2 行为构成（关键：加大 α 是更爱攒钱、放闪电多了、还是纯囤币不放）

（待填：A 臂每个 α 的"攒到 90 决策占比 / 放闪电 op21 次数 / HOLD 占比"，对照 pin。）

## 5. 结论（**待回填**）

- 有没有某个 α 让 unified 打 pin 变好？
- 最优 α 大概是多少、是值网自带 +0.037 的几倍？
- 是不是甜区 0.07~0.15 真的赢更多？

## 6. commit

- `26bdbaa` flat rollout 叶子评估加可调己方金币项（`AZAI_LEAF_COIN_ALPHA`，默认 0 零影响）+ α 扫描驱动/分析器
- `bf5e9a3` `_tree_rollout_value` 加 `leaf_coins` dump（只读诊断）+ 离线 α 解析扫描脚本
- `f3adf68` α 扫描默认 8 局/α（seeds 100..103 × 2 座位，JOBS=6）
- （结果 commit 待补）
