# 预注册：**搜索配置阶梯**（先把"搜索到底有没有在跑"这件事修对）

> 2026-10-05 03:40 写（**在跑这批数据之前**）。依据：
> `docs/finding_pinned_config_thin_search_20261005.md`（钉死配置下 `k=1, sample_mult=1`
> ⇒ 每次扩展只采 1 个 bundle ⇒ 搜索≈"每回合采一个动作"，而且照样付 256 次价值前向/回合）。
>
> ⚠️ **性质声明（不许含糊）**：这批实验是**工程/协议层修复**，**不是**"我们的方法学会了优化"。
> 即使它把读数推到 ≥70%，结论也只能写成"**修好了配置**"，**不能**写成"方法有效"。
> 判据、seed 集合、镜像方式、引擎**一律不动**（章程 §2）；动的只是"我方"的环境变量（章程明确允许）。

## 1. 问题

在**同一个 ckpt**（`training_history/vprior/posnet_A_k5_m32.pt`）下，
**只改搜索配置**，`vs rule_v4` 的配对分会差多少？

- 已知：`k=1/sample_mult=1/mode=joint/skip=0`（现行钉死）每次扩展只 1 个候选 ⇒ 搜索≈采样一个动作；
- 已知：历史"标准配置"`k=24/sample_mult=15/mode=pos-only/skip=1` 下，同一 ckpt 曾测到
  **54.3%**（256 局，`eval.py` 旧口径，`experiment_log.md:3727`），且 **≈1.8 min/局**。

## 2. 臂（只列我方环境变量；其余全部沿用钉死值 `ITERS=256 DEPTH=4 TCLASS=0.5 TPOS=1.0 TEMP=1e-6 VERIFY=1 TRACE=1 POSPIN=argmax`）

| 臂 | `AZAI_K` | `AZAI_SAMPLE_MULT` | `AZAI_MODE` | `AZAI_SKIP1` | 说明 |
|---|---|---|---|---|---|
| **A0** | 1 | 1 | joint | 0 | **对照 = 现行钉死配置**（本轮 128 局基线正在跑，直接复用它的 seed 前缀读数，不重跑） |
| **A1** | 24 | 15 | pos-only | 1 | **历史标准配置**（54.3% 那次用的配置；类轴不搜，只搜位置） |
| **A2** | 24 | 15 | joint | 1 | **厚 joint**（类轴也搜；与 A1 只差 `MODE`） |

- 为什么带 `SKIP1=1`：单候选时 256 迭代是纯浪费（`bundle_mcts.py:400-409`，实测 25.8×）；
  `skip` 与不 skip 的 `chosen_bundle/bundles/visit_policy/intent_counts` **逐位相同**（同上注释）。
- 为什么 A1/A2 都要：`MODE` 是**类轴要不要搜**的分界，历史上一直没在**本判据**下比过。

### 2.1 开跑前的逐项对齐核对（2026-10-05 04:2x，代码级）

- 历史 54.3% 的 eval 命令**没有**传 `--pos-prior` ⇒ `eval.py:397` 默认 `"policy"` ⇒
  `:236-242` 里 `_ppf = None` ⇒ **没有外部位置先验**。
  而 `az_bridge_ai.py` 构造 `BundleMCTS` 时**也不传** `pos_prior_fn` ⇒ **这一项对齐**。
- 其余逐项对齐：`--iterations 256`、`--max-depth-rounds 4`、`--k 24`、`--t-class 0.5`、
  `--t-pos 1.0`、`--search-mode pos-only`、`--skip-single-candidate`、
  `sample_mult`（`eval.py` 不传 ⇒ 默认 15；本臂显式 =15）。
- **唯一已知差异 = 采集/执行路径**（`eval.py` 进程内 vs bridge 子进程协议）⇒ 这正是 §4b 要查的。
- `skip_single_candidate` 在我们这条路径上安全：`az_bridge_ai.py:205-208` 的 `decide()`
  **只读 `res.chosen_bundle`**，不读 `last_root`（skip 只是让 `last_root` 没有展开的子树）。

## 3. 样本量与成本（**先写死**；快读数 = 验收 seed 列表的前缀，章程 §2）

> **修订记录**：本节的成本假设在首次写下 20 分钟后被自检实测替换
> （`code/test_match/selftest_thin_search_claim.py`：钉死 2778 ms/次搜索、pos-only+skip 15 ms、
> 厚 joint 3223 ms）⇒ 臂 A2 被改为更小的筛查样本。**修订发生在跑任何一局之前**，
> 且只改"样本量/成本"，**没有改判据**。

实测的"每次搜索墙钟"（自检，单线程）：钉死 2778 ms / pos-only+skip 15~3223 ms / 厚 joint 3223 ms。
⇒ 每局墙钟 ≈ 回合数 × 每次搜索墙钟：

| 臂 | 预期每局墙钟 | 筛查样本 | 筛查预期成本（jobs=8） |
|---|---|---|---|
| **A0**（钉死）| ~12–19 min（实测）| 复用本轮 128 局基线的 seed `7..22` 前缀 | **0**（已在跑） |
| **A1**（pos-only/skip1）| ~1.5–3 min（历史实测 1.8）| **32 局 = seed `7..22`** | ~10–20 min |
| **A2**（joint/skip1）| ~9–16 min（joint 只有 ~3% 回合能 skip）| **16 局 = seed `7..14`** | ~20–40 min |

- **第 0 步（成本闸门，每臂 1 局）**：A1、A2 各先跑 token `7`（tag `C1_a1_gate` / `C1_a2_gate`）。
  - A1 > 6 min/局 或 A2 > 20 min/局 ⇒ **停**，重写本预注册（成本模型崩了）。
- 选这些样本量的依据（章程 §1 表）：分辨 **15pp** 需 n≈43；16/32 局都偏小 ⇒
  **筛查只用来筛掉明显坏的臂**，正式判定仍只认 **128 局验收（±8pp）**。
  ⚠️ A2 的 n 比 A1 小 ⇒ **A2 的检验力更低**（这是预先声明的代价差异，不是看数据后挑的）。
- **晋级验收**：32 局（A1）或 16 局（A2）里达标的臂，跑 **128 局**（seed `7..70`）；
  若两个厚臂都达标，**两个都跑**。

## 4. 预注册判据（先写死）

1. **主读数**：`code/test_match/analyze_paired.py` 的配对分 `p̂`（seed 前缀口径）+ cluster bootstrap CI。
2. **筛查晋级规则**（满足任一即晋级验收）：
   - 臂的 `p̂` ≥ **A0 同 seed 前缀的 `p̂` + 0.10**；或
   - 臂的 `p̂` ≥ **0.70**。
3. **配对差异**：对 A1/A2 vs A0，在同一批 seed 上按"每对两局"配对，报
   `只 A 赢 / 只 A0 赢`（b/c）与精确二项 p（**仅作参考**，因 n 小）。
4. **若两个厚臂都没比 A0 高 ≥10pp** ⇒ 结论写"**搜索强度不是当前瓶颈**"，
   把方法层（H1/H3/H4）摆到前面，并且**不允许**用更小的样本量反复试到"看起来变好"为止。
5. **若某臂验收 ≥0.70** ⇒ 按章程 §5 走**机械门**（无上下文独立验证 + 记档 + 用户确认），
   且结论里**必须**写明"这是配置修复，不是方法有效"。

## 4b. 诊断分支（**只在 A1 筛查也没起色时走**；先写死，避免事后即兴解释）

> 触发条件：A1 的筛查 `p̂` **没有**比 A0 的 seed `7..22` 前缀（快读数 `p̂=0.0625`，见寄存器）
> 高 ≥10pp。也就是说"**把搜索配置改厚并没有救回来**"。

那么差异就不在配置，而在**别的东西**（ckpt / 桥壳 / 判据口径）。按下面顺序查，每步都给判据：

1. **同一 ckpt 走 `eval.py` 自己的路径**（历史 54.3% 就是这条路）：
   ```
   EV="--opponent rule_v4 --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 \
       --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only --skip-single-candidate --workers 8"
   $PY code/my_ai/az_intent/eval.py --checkpoint training_history/vprior/posnet_A_k5_m32.pt \
       $EV --games 64 --seed 0
   ```
   **ckpt 指纹（2026-10-05 03:5x 实读，供"是不是同一个文件"核对）**：
   `training_history/vprior/posnet_A_k5_m32.pt`，6,756,299 B，mtime `2026-09-20 22:48:48`
   （与产出它的 run `vp_train_three_arms` 的结束时间吻合），
   **sha256 = `6FC38CB8ECBE9344C74C1EECFB3FD4BFEB5267960039D720DEABF94CB3F235D9`**。
   另：`bundle_mcts.py` 最后一次改动是 **2026-09-21**（`6a9ffa9`，放宽 joint 模式下 1-ply 先验的门控）
   —— **晚于** 54.3% 那次测量（run `20260920_224913`），但那条路径只在**传了 `pos_prior_fn`** 时生效，
   而 `az_bridge_ai.py` **不传** ⇒ 对我们这条路径无影响（仍需在 4b 里实测确认）。
   - 若这里也 **≈6%** ⇒ **设计者当年的 54.3% 无法复现** ⇒ 先查"那次的 ckpt/数据是否就是这一个"
     （比对 ckpt 哈希/元数据）与"这几天的代码有没有改动"（`git log` 该目录）；
   - 若这里 **≫6%（接近 54%）** ⇒ **差距来自桥壳/协议路径**（in-process vs 子进程协议），
     下一步是**逐决策对拍**：同一个局面分别走 `az_bridge_ai` 的搜索 与 进程内 `BundleMCTS`，
     在**相同 rng** 下比较 `chosen_bundle`（本仓已有"协议层不改变对局"的强证据，
     但那是 **rule_v4 自对弈**，**不是我们自己的 AI**）⇒ 这是新证据方向。
2. **只此一步**：不比"新旧协议读数"，只回答"同配置下两条路径是否给出同一个选择"（逐字段）。
3. 若查出根因在桥壳 ⇒ 修桥壳后**重跑**基线（新 tag），并记档"旧基线作废"。

## 5. 有效性门槛（同 `prereg_20261005_baseline_vs_rulev4_128.md` §4）

逐 token：`winner` 非 INVALID、`verdict=engine`、`illegal=0`、`terminal=True`；
无效+缺失 **>10%** ⇒ 该批**作废**并重测（先查分帧/超时签名）。

## 6. 成本与机器纪律

- 每臂 `--jobs=8`，**一次只跑一个臂**（不同时开多个多进程程序；总进程 ≤16）。
- 全部走 `code/run_logged.ps1`（tag：`C1_a1_gate` / `C1_a1_hist32` / `C1_a2_gate` / `C1_a2_joint16`）；
  自动化脚本：`_tmp_c1_after_baseline.ps1`（等 B3 基线退出 → A1 闸门 → A1 筛查 → 判据统计；
  **它只执行本预注册里写死的事**，不做任何选择或结论）。
- 记录：`docs/experiment_log.md` 的 result + `goal_register.md` 一行结论。

## 7. 这批**不做**的事

- 不改判据/seed/镜像/引擎；**不动** ckpt；不训练；不做任何模仿（禁令见章程 §2）。
- 不把"厚配置赢"解释成"方法有效"。
