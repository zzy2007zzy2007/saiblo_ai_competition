# 价值网训练标签管线调研（2026-10-31，只读）

> 目的：为重训值网、把「金币 × 冷却」的定价教进去，先摸清仓库里**已有哪些标签类型 / 数据 / 管线**，
> 以及"从 terminal 切到逐帧"到底要动几处。
>
> **只读**：没训练、没跑对局、没跑值网探针。为盘点数据集覆盖度，跑了 5 段**纯 numpy 数据统计**
> （只读 `stats.npy`/`label*.npy`/`board.npy` 的 mmap + 3 个 pkl + `entity_ds_128.npz`），命令见 §11，秒级、无 GPU。
> 本文所有结论都带 `文件:行号` 或文档出处；查不到的写「未找到」。
> 时间敏感性：本文引的 `value_net_pipeline_survey.md`（10-27）与探针文档（10-30）是当前最权威的口径，
> 更早的实验文档（09 月那批 336 局时代的读数）只当历史先验用。

---

## 0. 一页结论

| 问题 | 答案 |
|---|---|
| 现役值网用什么标签？ | **terminal**（整局终局 HP 差 /20，clip ±1，按 player 定符号）。同一局同一玩家的**所有决策点同值**（实测局内 std 恒为 0）。`train_value_net.py` 的注释自己写明了后果（`code/my_ai/az_intent/train_value_net.py:265-267`）。 |
| 仓库里有哪些标签类型？ | **两组**：① **HP 差族**（`terminal` / `abs` / `rel` / `mix` × 权重 `geo` / `kgeo` × `tau`），逐帧，**已实现**（`az_train.add_weighted_labels`、`value_warmup._exp_weighted_labels`）；② **非 HP 族**（亚军手写每帧值 `evaluate_terminal`、搜索回传 `root_value`、`quiet_value` rollout 值、`adv`）——都是逐帧，但都是**别人的/别的口径**，现成数据规模小或退化。 |
| 哪个能给「逐帧 + 金币×冷却」监督？ | **只有 HP 差族的 `abs`/`rel`（配 `kgeo`）**能提供**逐帧**监督；但它是**间接**的：标签只由"未来 HP 轨迹"算出来，**没有任何一项显式含冷却**（10 个标签公式里 0 处 cd），金币也只通过"实际花没花、花得值不值"进到未来 HP 里。实测帧内与 coins/cd 的偏相关只有 **+0.09 / +0.03**（§4.3）⇒ 能学，但不要指望它自动学会 90 币折点。 |
| 从 terminal 切到逐帧要动几处？ | **标签生成不用改代码**（`--label-mode/--label-weight/--tau` 已经在 CLI 里，`train_value_net.py:208-213`）；**网络/损失/数据加载都不用动**。要动的只有 4 件事：`--label-scale`（幅度）+ `--freeze-bn`（默认已开）+ **评估加金币/冷却分层**（现在只按有塔/无塔分，`:166,179`）+ 部署侧开关（只有 `rel` 需要 `AZAI_VALUE_TANH=0 + AZAI_REL2ABS`）。 |
| 现成数据集哪个能用？ | 逐帧必须用**全帧**数据：`inject_ex02`（部署血统，400 局 / 369,476 帧 / 38.9% 有塔，**在 F: 备份**）或 `lowq_f256`（5000 局 / 128 万帧 / 36.8% 有塔 / 48% 帧 ≥90 币 / 12.8% 帧 cd==0，**本机 24.7 GB**）。`lowq_f64` 每局只 64 帧，**只用掉 7.5% 的逐帧信号**，不适合当逐帧标签的主训练集。 |
| 最大先验风险 | "换标签"**不是处女地**，而且**历史上是负的**：`kgeo50` 128 局验收 **−7.0pp**、`rel geo_tau50` **−13pp**（`docs/experiment_log.md:11391-11395`、`docs/state_of_play_20261006.md:86`）。本轮**不能以胜率为第一判据**，应以 P0-2/P0-3 的**函数形状**为判据，胜率只当回归门。 |
| **数据来源该怎么换？** | **换掉 lowq/M6c，改用"当前 ckpt 自对弈 + 亚军 oracle 条件化注入"现场采集**（§8）。理由：三份现成数据的动作分布**两两不同**（§8.1），而且"放超武"的频率差一个量级（`lowq` 12.4 次/局/方 vs 我们自己的对局里闪电占 0.6% 决策）。bootstrap 破法 = **把"动作质量"外部化给 oracle**（注入），标签仍由未来 HP 轨迹算——**不需要先有一个会定价的 V**。 |
| **现成逐帧标签含 90 币阈值吗？** | **不含**（§12 实测 1.28M 帧 lowq）：`kgeo50` 与金币的斜率在 pooled ≈ **+0.0012/100 币（R²≈0）**、within **−0.0154**；`≥90` 桶比 `60-89` 桶**低 9.2% 个标签 std**，扣掉回合趋势后只高 **+0.9%**，90 处**无台阶**（`1[coins≥90]` 系数 −0.0065，t=−9.2）。⇒ 折点只能靠**数据**（放/攒对照）造出来，不是靠换标签。 |
| **但我们自己的对局里，根本不到 90 币** | **`coins≥90` 只占 3.0% 的回合**（36 局 / 9,046 回合实测，中位金币 **29**；冒烟那局 **3.35%**，与流传的 3.4% 一致），而 lowq 是 48.2%、同一批对局里对手（亚军）是 **69.1%**（§13）。现算的 `kgeo50` 在 `≥90` 桶比 `60-89` **高 +0.088（≈0.56 std，按局 bootstrap 90%CI 排除 0）** ⇒ **方向是正的不是反的**，但 275 个事件只够当提示。⇒ 瓶颈是"这个状态不存在"，不是"标签没信号"。 |

**一句话**：现役是 terminal（局内同值 ⇒ 对"花 45 币 vs 攒着"零信号）；逐帧可选项就是 `abs/rel × geo/kgeo`
那一族（`rel` 唯一去掉了"抄当前 HP 差"的捷径），但**金币×冷却只能靠它间接学**——所以重训的正确顺序是
「先做数据侧可识别性检查 → **换数据来源（注入采集）** → 换标签重训 → 用 P0-2/P0-3 探针验收 → 才谈胜率」。

---

## 1. 现役 terminal 标签管线

### 1.1 一句话链路

`az_selfplay.py`（Bundle-MCTS 自对弈 + 采样）→ 每局一个 `az_selfplay_seed*.pkl`
（样本含 `board/stats/player/...`）→ 局末统一回填 `value_target`（终局 HP 差）→
`train_value_net.py --label-mode terminal` 把这些值直接当标签 → 在 `value_state` 上做 MSE 微调 →
写回三网 ckpt（`class_state`/`pos_state` 逐位照抄）。

### 1.2 训练数据从哪来

| 项 | 值 | 出处 |
|---|---|---|
| 现役值网 ckpt | `checkpoints/posnet_A_k5_m32.pt` 的 `value_state`，**逐位 = `training_history/inject_ex02/valnet_A2.pt`**（94/94 张量相同） | `docs/next_step_survey_20261005.md:53-59`、`docs/checkpoints.md:20` |
| 采集脚本 | `code/my_ai/az_intent/az_selfplay.py`（`--games 400 --workers 16 --iterations 256 --max-depth-rounds 4 --k 24 --search-mode pos-only --skip-single-candidate --native-engine --inject-example-prob 0.02`） | `docs/experiment_log.md:3054` |
| 采集用的策略 ckpt | `training_history/vprior/posnet_r1.pt` | 同上 |
| 数据产物 | `training_history/inject_ex02/data/az_selfplay_seed*.pkl`，**400 局 / 17 GB** | `docs/experiment_log.md:3058` |
| 训练用的紧凑缓存 | `training_history/inject_ex02/value_cache`（board/stats/player/value_target 四个 memmap，369,476 行） | `code/my_ai/az_intent/train_value_net.py:65-129`、`docs/next_step_survey_20261005.md:36-38` |
| 训练配方 | `--label-mode terminal --freeze-bn --epochs 3 --lr 3e-4` | `docs/experiment_log.md:3119` |
| **数据现在在哪** | `training_history/inject_ex02/` 本机已不在；**`F:\2026智能体大赛_新2\training_history\inject_ex02\` 在**（本会话 `Test-Path` 确认；`data/` 400 个 pkl 共 17.3 GB，`valnet_{A,A2,B,C}.pt` 也在）。**`value_cache/` 不在备份里**（只有 `data/` + 4 个 pt）⇒ 切标签前要先按 `--build-cache-only` 重建（`train_value_net.py:231-235`） | 本会话实测；`docs/value_net_coin_pricing_hypothesis_20261030.md:504,682` |

> ⚠️ **纠正一个流传的说法**：**"训练分布里有塔局面 0%"指的是注入前的祖先数据，不是 `valnet_A2` 的训练集。**
> `inject_ex02` 之所以存在，就是为了把"场上有塔"的决策点从 **0.0% → 38.9%**（`docs/experiment_log.md:3064-3068`、
> `docs/value_net_coin_pricing_hypothesis_20261030.md:116-132,487-493`）。本会话抽 3 局实测 **42.4%** 帧有己方塔，与 38.9% 同量级。
> 所以"有塔局面没见过"已经不是当前的主要病灶；**主要病灶是"一局一个标量"这条标签本身**。

### 1.3 terminal 标签怎么算

1. 局末：`diff = state.bases[0].hp - state.bases[1].hp`
2. `v_p0 = clip(diff / HP_SCALE, -1, 1)`，`HP_SCALE = 20`（`code/my_ai/az_intent/mcts.py:46`）
3. 平局但有 winner：`v_p0 = ±0.1`
4. 按样本的 `player` 定符号：`value_target = v_p0 if player==0 else -v_p0`

代码：`code/my_ai/az_intent/az_selfplay.py:484-489`；同一逻辑的另两份实现：
`value_warmup.py:81-86`（暖启动采集）、`selfplay.py:37-47`（`terminal_value_target`）。

⇒ **是胜负/HP 差，不是二值胜负**：它是"终局血量差 /20"这个**连续标量**，同局同玩家恒定。
**本会话实测**（`lowq_f64`，320k 帧 / 5000 局）：`label.npy` 每局只有 **2 个**不同取值（= ±|v|，按 player 分），
按 (局, player) 分组后**局内 std = 0.000000**。这就是"零信号"的精确形态。

### 1.4 网络怎么读特征、损失是什么

| 环节 | 事实 | 出处 |
|---|---|---|
| 输入 | `board (B,28,19,19) f16` + `stats (B,42)`；`stats` = 22 个 named 特征 + 20 个 extras | `Ant-Game/SDK/utils/features.py:134-157, 225-257` |
| 价值头输入 | `board_emb = spatial.mean([2,3])` (64) ⊕ `stats_emb` (64) → 128 维 → `Linear(128,64)+ReLU+Linear(64,1)`，**裸线性输出、无 tanh** | `code/my_ai/network.py:184-186, 210-214, 236, 241, 272, 283` |
| 损失 | **MSE**：`loss = F.mse_loss(out, vv)`（`out = value_model(bb,ss)["value"].squeeze(-1)`） | `code/my_ai/az_intent/train_value_net.py:331-332` |
| 评估 | 整体 + **按"该帧场上有没有己方塔"分组**的 MSE / Pearson r（`board[:,4]`） | `train_value_net.py:162-185`（分组在 `:166,179`） |
| 其他训练开关 | `--freeze-backbone`（冻骨干）、`--freeze-bn`（**默认开**：源值网 BN 统计量从未训过，train 模式会猛冲）、`--anchor-lambda`（参数 L2 锚回初值） | `train_value_net.py:214-225, 293-299, 306-311` |

**没有的东西**：没有 z-score 归一化（用的是原始标签尺度 + `--label-scale`）、没有金币/冷却分层评估、
没有按"标签价值"加权采样。

---

## 2. 标签类型对照表

### 2.1 HP 差族（现成、逐帧、可直接用）

记 `d_t` = 第 t 帧的**统一 P0 视角**瞬时血量差（`= ±(hp_us-hp_opp)`，读自 `stats[1]`），
`γ = exp(-1/tau)`，`k` 为未来偏移。

| 标签 | 目标值公式 | 逐帧？ | 0 中心的含义 |
|---|---|---|---|
| `terminal` | 整局终局 `clip(margin/20,±1)`，同局同玩家恒定 | ❌ **局内常数** | "这局最终赢多少" |
| `abs` | `raw_t = Σ_{k≥0} w_k d_{t+k} / Σ w_k`（指数加权未来血量差均值） | ✅ | "从这个位置起，未来的血差水平" |
| `rel` | `label_t = raw_t − d_t` | ✅ | "从这出发未来优势**会变多少**"（设计推荐） |
| `mix` | `label_t = α·d_t + (1−α)·raw_t`（`α=--label-mix-alpha`，默认 0.5） | ✅ | 当前与未来的混合 |
| `geo`（权重） | `w_k = γ^k`，峰值在 k=0（**含当前帧**） | — | 与 `stats[1]` 相关 **0.937**（tau50）⇒ 可"抄输入" |
| `kgeo`（权重） | `w_k = k·γ^k`，**w_0 = 0，排除当前帧**，峰值移到 k≈tau | — | 与 `stats[1]` 相关 **0.863**、R² 0.723（tau50）⇒ 捷径被削但未消灭 |

- 公共尾巴：`label = clip(lab / HP_SCALE, ±1) * label_scale`，再按 `player` 翻符号（`az_train.py:107-117`、`value_warmup.py:182-190`）。
- 递推（O(n) 反向扫描）：`geo` `az_train.py:131-135`；`kgeo` `az_train.py:119-129`（`value_warmup.py:193-211` 是同一实现的第二份）。
- **捷径的量化**（重要）：`abs` 类标签 75–90% 的方差可被"把 `stats[1]` 乘个数"解释；`rel` 是唯一把 `d_t` 结构性减掉的
  （与当前血差相关 **−0.24**）——见 `docs/az_pool_tau_label_plan.md:89-111, 113-158` 与
  `docs/next_step_survey_20261005.md:29`（本缓存实测 R²：tau50 geo **0.898** / kgeo **0.777**；tau200 geo 0.748 / kgeo 0.573）。

**生成代码位置**

| 用途 | 文件:行号 |
|---|---|
| 训练器（pkl/memmap 缓存路径，逐局流式重算） | `code/my_ai/az_intent/train_value_net.py:261-283`（`--label-mode != terminal` 分支，`:269-279`） |
| 同一份实现的「核心」 | `code/my_ai/az_intent/az_train.py:56-135` |
| 第二份实现（npz 路径，逐位等价，已验证） | `code/my_ai/az_intent/value_warmup.py:143-212`（`_load_chunk` 在 `:215-242` 按 `label_cfg` 切换） |
| AZ 训练器（pkl，`--value-only`） | `code/my_ai/az_intent/az_train.py:828-840` |
| 给低质大数据**离线**算全量逐帧标签 | `code/my_ai/entity_transformer/build_rel_labels.py:56-99`（复用 `value_warmup._exp_weighted_labels`，`:27,66-68`；`--save-full` 另存全量，`:96`） |

### 2.2 非 HP 差族（逐帧，但口径不同 / 数据退化）

| 标签 | 定义 | 生成代码 | 现成数据 | 能用吗 |
|---|---|---|---|---|
| **亚军手写每帧值** `evaluate_terminal` | `base_hp*200 + tower_value + 等价金币(≤400 部分×10、超出×8) + 闪电 bonus(+100/−200/伤害×1.5) − threat`；**显式含金币，无冷却项**；量纲 ~1e4 | 由亚军 C++ 在 dump 时一并写出（`raw_seed*.bin` 的 `value` 字段）→ `entity_transformer/build_dataset.py:18`、`build_rollout_dataset.py` | `training_history/distill_top2/entity_ds_128.npz`（127,578 帧 / 128 局）、`value_ds_{12g,24g}.npz`、`value_ds_ent8.npz`、`rollout_ds_test.npz`（184,368 帧但**仅 2 局、stats 退化**） | ⚠️ 换口径风险已实测：用它替换后 128 局只 **0.3906**（基线 0.7344，`docs/experiment_log.md:11391-11395`、`docs/entity_transformer_value_results.md:33-37`）。**只能当"容量/表达力"实验，不能当部署标签** |
| **搜索回传 `root_value`** | 每次决策的搜索根值（稠密、平滑、已被前瞻改进） | `code/my_ai/az_intent/azjsonl_to_value_npz.py:1-12,69-71`（目标 = `root_value`） | `training_history/distill_top2/ru_ent_raw/*.bin`、`code/distill/convert_az_dump.py` 产出的 jsonl；ckpt `checkpoints/grafted_sdk_A_vsearch.pt`、`value_search_net.pt` | ✅ 逐帧、便宜，但**是另一个价值函数的输出**（自举/蒸馏），会继承它的缺陷，且不是"金币×冷却"的监督 |
| **`quiet_value`（安静 rollout 值）** | MC rollout 到终局的值 | `code/my_ai/az_intent/mc_quiet.py:46,124,150`；dump 开关 `AZAI_DUMP_QUIET=1`（`code/test_match/az_bridge_ai.py:277-286`） | 只是**采集时可选写入 jsonl**，仓库里**未找到已落盘的 quiet_value 数据集** | ⚠️ 需要跑对局才能采（本轮只读不做）；历史上作为"裁判/补尾"评估过，不在值网标签线上 |
| **`adv`（R − V 相对优势）** | 1-ply 贪婪 rollout 的回报减去 `valnet_A2` 的估值，逐格 | `code/my_ai/az_intent/collect_value_prior.py` → `vp_*.npz` 的 `adv` | `training_history/vprior/vp_A_k5_m32.npz` | ❌ 用途是**训练位置先验图**，不是标量值 |

**另外**：`q8/*`（计划价值）、`prior_labels/*`（类先验）都**不是**局面值标签，别喂值网（`docs/value_net_pipeline_survey.md:86-97`）。

### 2.3 一张图看懂：谁有"逐帧"谁没有

| 标签 | 同局同玩家内的逐帧 std | 与当前 HP 差的相关 | 显式含金币 | 显式含冷却 |
|---|---|---|---|---|
| `terminal` | **0.000000**（本会话实测） | —（局内常数） | ❌ | ❌ |
| `geo_tau50`（abs） | 0.0722（`value_net_pipeline_survey.md:388-397`） | **0.937** | ❌ | ❌ |
| `kgeo_tau50`（abs） | **0.1064**（本会话实测；文档口径 0.1042） | 0.863 | ❌ | ❌ |
| `rel_tau50` | 0.1468（同上文档） | **−0.241** | ❌ | ❌ |
| 亚军 `evaluate_terminal` | 逐帧（每帧独立算） | — | ✅（×10 分段） | ❌ |

---

## 3. 现成脚本/管线清单（哪些已经存在、只是没用起来）

**结论：从 terminal 切到逐帧，**仓库里已经有全套现成实现**，包括"逐帧标签 → 三网 `value_state` → grafted 可部署 ckpt"这条完整链路。**

| 环节 | 现成的东西 | 状态 |
|---|---|---|
| 换标签训练（三网 value_state） | `train_value_net.py --label-mode {terminal,rel,abs,mix} --label-weight {geo,kgeo} --tau --label-scale` | ✅ 已上线，默认 `terminal` |
| 换标签训练（独立 `ValueNetwork`） | `entity_transformer/train_value_stream.py --label-file ...`（z-score MSE，按局切 val） | ✅ 已上线，且已加 `--width/--blocks`（`:69-71`） |
| 全量逐帧标签落盘 | `build_rel_labels.py --save-full`（10 个变体一次算完） | ✅ 有，但**全量标签目录 `distill_top2/lowq_full` 已被清理**（本会话 `Test-Path` = False），要重算 |
| 子集连标签一起抽 | `build_frame_subset.py --extra-label ...`（与 `subsample_frames.py` 同 seed/同逻辑） | ✅ 有，`lowq_f256` 就是它产的（含 10 个标签） |
| 把独立值网 graft 进三网 ckpt | `entity_transformer/graft_value_ckpt.py`（并把 `y_mean/y_std` 折进 `value_head` 最后一层，`:68-76`） | ✅ 有，产出了 `checkpoints/posnet_A_k5_m32_{lowq,relgeo50,relkgeo30,relkgeo50}.pt` |
| 部署侧换独立叶子值网 | `AZAI_VALUE_CKPT` + `value_override.py:13-31` | ✅ 有（⚠️ **只换叶子**，仲裁用的 `net_fn` 里那个 value 不动 ⇒ 金币定价要换的是三网里的 `value_state`） |
| 逐帧标签的历史臂 | `checkpoints/rel_{geo50,kgeo30,kgeo50,kgeo100,kgeo200}/lowq_value_seed0.pt` + 日志 `training_history/distill_top2/train_f256_*.log` | ✅ **已训过**（离线 val R²：geo50 0.5979、kgeo30 0.5580、kgeo50 0.4799、kgeo100 0.3590、kgeo200 0.2825；⚠️ 各臂标签不同，**R² 不可横向比较**） |
| 金币/冷却分层评估 | ❌ **未找到** | 要新写（现成评估只按有塔/无塔分） |
| 显式含金币或冷却的标签 | ❌ **未找到**（除了亚军手写值含金币、无冷却） | 要新设计 |

---

## 4. 明确回答：哪个标签能提供「逐帧 + 金币×冷却」监督

### 4.1 金币和冷却在**输入**里都有

- `stats[22] = coins[player]/300`、`stats[23] = coins[enemy]/300`（`features.py:227-228`）
- `stats[26] = 我方闪电剩余冷却 / 35`、`stats[30] = 敌方闪电冷却 / 35`（`features.py:231-232, 239-240`；
  闪电 `cooldown=35, cost=90`，`Ant-Game/SDK/utils/constants.py:194`）
- 另有 `stats[2]=coin_ratio`、`stats[3]=safe_coin`、`stats[34]=超武使用次数`

⇒ **值网"看不见金币/冷却"不是输入问题，是监督问题**（探针文档已有同样判词：`value_net_coin_probe_20261030.md:302`）。

### 4.2 逐帧标签里，"cd 从 20 落到 0"时标签值会不会变？

**会变，但原因是"时间在走"，不是"cd 这个变量"。** 逐帧标签是 `d_{t..T}` 的加权平均，
时间推进 ⇒ 加权窗口里换了一批 `d`，标签自然变。实测（`lowq_f64`，按 (局, player) 去均值后）：

| 量 | 局内 std | 说明 |
|---|---|---|
| `kgeo_tau50` 标签 | **0.1064** | 总 std 0.1143 ⇒ **93% 的方差在局内**（逐帧信号真实存在） |
| `terminal` 标签 | **0.000000** | 局内零信号（这就是病根） |
| 局内 corr(标签, `hp_delta`) | **−0.573** | kgeo 仍残留"当前血差"信息，但已远低于 geo 的 +0.9 |
| 局内 corr(标签, `coins`) | **−0.069** | 几乎没有同期相关 |
| 局内 corr(标签, `cd_light`) | **+0.063** | 同上 |
| 偏相关(coins, 标签 \| round,hp,cd) | **+0.087** | 控制回合/血差/冷却后仍只有 0.09 |
| 偏相关(cd, 标签 \| round,hp,coins) | **+0.028** | 更弱 |

**怎么读**：逐帧标签**确实**给了"同一局内不同决策点不同目标"，因此它**具备了**教"这一手花钱值不值"的
**结构性条件**（terminal 连这个条件都没有）；但它与金币/冷却只有**很弱的同期关联**，
金币的价值必须通过"花了钱 → 后来 HP 变好/变差"这条**多回合链路**才能进到标签里。
这不是标签的 bug，而是这个游戏里"投资→收益"本来就滞后（`docs/rl_survey/c_reward_shaping.md:446-448` 讲的是同一件事）。

### 4.3 结论（明确回答）

1. **能给"逐帧"监督的**：`abs` / `rel` / `mix`（×`geo`/`kgeo`）——**现成、零新代码**。
   其中 **`rel`** 唯一把"抄当前 HP 差"的捷径（R² 0.87–0.90）结构性去掉；**`abs+kgeo`** 折中（R² 0.72，但**部署零协议变更**）。
2. **能给"金币×冷却"监督的**：**只有它们的间接版本**。没有任何现成标签显式含冷却；
   显式含金币的只有亚军手写值（且含金币的权重是**分段线性 +10/币**、**完全不含冷却**，用它做标签等于把"金币线性定价"直接焊死，
   正是我们要修的那个病）。
3. **所以重训的可行性边界是**：
   - ✅ 能修"一局同值 ⇒ 零信号"（这是 terminal 独有的病）；
   - ⚠️ 能不能修出 **90 币折点**和 **cd 条件斜率 ≥2×**，取决于**数据里"花 vs 攒"的对照够不够强**——
     这一条**必须先量**（§7.1 的 P0），不能直接开训赌。
   - ❌ 若数据侧不可识别，换任何标签都没用；那时唯一的出路是**续盘回归估汇率 c 写进标签**
     （`docs/rl_survey/c_reward_shaping.md:347-359`，需要跑续盘对局）或**新采集带金币/冷却分层的局面**。

---

## 5. 数据集盘点

位置约定：`$D = training_history/distill_top2`（下同）。"覆盖度"一列是**本会话实测**（20,000 行随机抽样，或全量标量列）。

### 5.1 逐帧/大数据

| 路径 | 规模 | 标签 | 有塔帧 | ≥90 币帧 | cd==0 帧 | 备注 |
|---|---|---|---|---|---|---|
| `$D/lowq_f64/` | **320,000 帧 / 5000 局**（64 帧/局），board 6.5 GB | `label.npy`（**terminal**）+ **9 个** rel/kgeo（geo tau5/10/50/100/200、kgeo tau5/10/30/50/100/200） | **36.8%** | 48.3% | 12.8% | 抽样种子硬编码 `default_rng(12345)`（`subsample_frames.py:58`）；**没有 player.npy**（用 `stats[36]/[37]` 还原） |
| `$D/lowq_f256/` | **1,280,000 帧 / 5000 局**（256 帧/局），board 24.7 GB | terminal + **10 个**标签（含 `kgeo_tau30/50/100/200`） | 36.8% | 48.2% | 12.8% | **本地唯一"逐帧够用"的大集**；`meta.json` 的 `src` 指向已删除的 `lowq_full` |
| `F:\...\lowq_e2_merged/` | **4,280,296 帧**（全帧），board 86.5 GB | 只有 `value_win.npy`（terminal）；**10 个 rel/kgeo 标签已被清理** | 同源 | 同源 | 同源 | 标签可用 `build_rel_labels.py --save-full` 从 `stats.npy`(360 MB)+`player.npy` **重算（分钟级，无对局）**；⚠️ 该脚本写死 `out.parent/lowq_full`（`:96`）⇒ 目标目录要先建 |
| `F:\...\inject_ex02\data/` | **400 局 / 369,476 帧 / 17.3 GB pkl** | `value_target`（terminal）；`stats` 里含金币/冷却 ⇒ 可重算逐帧标签 | **42.4%**（3 局抽样；文档口径 38.9%） | 57.2% | 11.0%（`cd>0` 占 89.0%） | **部署血统**（`valnet_A2` 就是它训的）⇒ 做"只换标签"的单变量对照最好；⚠️ `value_cache/` 不在了，要重建 |

> 口径说明：`stats[22]`/`stats[26]` 是 `float16` 存储，本表按 `float16` 读；`cd==0` 的比例在两份 lowq 上都是 12.8%
> （`lowq_f64` 与 `lowq_f256` 抽的是同一批帧的子集，比例一致是自洽性检验）。
> `lowq_f64` 与 `lowq_f256` 的"有塔帧 36.8%"与 `inject_ex02` 的 42.4% 同量级 ⇒ **"有塔局面 0%"确实已是历史**。

### 5.2 其它值网数据（口径不同，慎用）

| 路径 | 规模 | 标签 | 结论 |
|---|---|---|---|
| `$D/entity_ds_128.npz` + `entity_ds_128_board.board.npy` | 127,578 帧 / 128 局 | 亚军 `evaluate_terminal`（~1e4 量纲） | 表达力实验专用（实体 Transformer R²=0.9969），**不是部署口径** |
| `$D/value_ds_{12g,24g,ent8}.npz` / `value_ds_runnerup_selfplay.npz` | 5,942 / 12,356 / 8,192 / 32,768 帧 | 同上 | 局数太少（8–32 局），只能当冒烟/容量测试 |
| `$D/rollout_ds_test.npz` | 184,368 帧 / **2 局** | 亚军 rollout 值 | ⚠️ `stats` 退化（前 4 维有效），跨局 val R² 只 0.624 ⇒ **不能当验证集** |
| `$D/lowq_f64_ent.npz` / `lowq_f256_ent.npz` | 320k / 1.28M 帧 | terminal（实体形式） | 实体 Transformer 线用 |
| `training_history/az_dump_ru/`、`az_dump_bc/` | 10.8 MB / 338 MB jsonl | 每决策 `root_value`（+ `chosen`/`visits`） | 逐帧搜索值标签的原料；要配 `obs_*.bin` 才能转 npz（`azjsonl_to_value_npz.py:56-58`） |
| `training_history/bc_dataset/seed*.npz` | 7.1 MB | — | BC 蒸馏数据，非值标签 |

### 5.3 lowq_f64 到底是什么（回答提问）

- **是什么**：5000 局**低质量自对弈**（M6c 裸策略 + ε=2% 随机动作，C++ 引擎）的**每局均匀抽 64 帧**的子集，
  由 `subsample_frames.py` 从 `lowq_e2_merged`（91 GB，现在在 F:）生成（`meta.json` + `docs/value_net_pipeline_survey.md:61-84`）。
- **标签是什么类型**：`label.npy` = **terminal**（局末血差/20，**按 player 定符号**——本会话确认每局恰好 2 个取值 ±|v|）；
  另有 9 个 `label_{geo,kgeo}_tau*.npy` 是 **rel 族逐帧标签**（`build_rel_labels.py --mode rel`）。
- **有多少样本**：320,000（5000×64）。
- **覆盖**：有塔帧 36.8%、≥90 币帧 48.3%、cd==0 帧 12.8%；金币中位 88 币（`stats[22]` p50=0.293）。
- **能不能当逐帧主训练集**：**不建议**。每局只保留 64 帧 ⇒ 逐帧标签只用掉 **7.5% 的局内信号**
  （`docs/value_net_pipeline_survey.md:386-397` 的局内/局间方差分解 + 历史结论"rel 必须全量帧"）。

---

## 6. 从 terminal 切到逐帧：最小改动清单

### 6.1 不用动的（重要，能省掉大量工作）

| 环节 | 为什么不用动 |
|---|---|
| **网络结构** | 输入没变（`board`+`stats(42)`，金币/冷却本来就在 `stats[22]/[26]`）；价值头还是那个 128→64→1 |
| **损失** | 还是 MSE（`train_value_net.py:332`）。历史换过 MSE / `MSE(tanh,·)` / CE，**没有一条有证据收益**（`docs/value_net_pipeline_survey.md:264-267`） |
| **数据加载** | pkl/缓存路径里 `stats` 已经存了；逐帧标签**按局流式重算**（`train_value_net.py:272-279`），不用改 schema。缓存重建只是"把 pkl 再压一遍"（17 GB → memmap，16 workers） |
| **标签生成的"算法"** | `add_weighted_labels` 已实现 rel/abs/mix × geo/kgeo，且验证过两份实现逐位一致（geo 差 6e-8，见 `docs/az_pool_tau_label_plan.md:163-168`） |

### 6.2 要动的（4 项）

1. **一个 CLI 开关**：`--label-mode abs|rel|mix`、`--label-weight kgeo`、`--tau 50`（`train_value_net.py:208-213`）。
   *这条严格说是 0 行代码。*
2. **`--label-scale`（必调，不是可选）**：`train_value_net.py` 不做 z-score，直接在原始标签上算 MSE。
   `kgeo50` 标签 std ≈ 0.11、`rel50` ≈ 0.15，而 terminal ≈ 0.386 ⇒ 不放大到同量级，值网输出会**小 2.6–3.4 倍**，
   PUCT 探索项会盖过价值项（历史事故就是这么来的：`docs/experiment_log.md:11399-11401`、`docs/value_net_pipeline_survey.md:344-349`）。
   参考标定值（按幅度匹配 terminal 基线 std 0.2945）：**`kgeo50+abs` scale≈1.5**、**`rel50` scale≈4.0**（`docs/az_pool_tau_label_plan.md:169-173`）。
3. **评估要加金币/冷却分层**（新代码，~40 行）：现在 `evaluate()` 只按"有己方塔"分组（`train_value_net.py:166,179`）。
   至少要加：`coins` 分桶（<45 / 45–69 / 70–89 / ≥90）× `cd` 分桶（==0 / >0.3）× 有塔，报 MSE/r。
   **否则训练日志无法回答"金币项有没有学到"**。
4. **部署侧开关（只有 `rel` 需要）**：`AZAI_VALUE_TANH=0` + `AZAI_REL2ABS=<scale>`，
   实现是 `value = value/rel2abs + stats[1]/HP_SCALE`（`az_selfplay.py:187-188`，环境变量入口 `az_bridge_ai.py:1192`）。
   **`abs`/`mix` 不需要任何协议变更** ⇒ 这是首推 `abs+kgeo` 的工程理由。

### 6.3 数据侧的两个"隐藏改动"

- **必须用全帧数据**：逐帧标签的方差 93–96% 在局内，抽 64 帧/局只剩 7.5%（§5.3）。
  ⇒ `inject_ex02`（本来就有全部决策点，369,476 帧）或 `lowq_f256` 可用；`lowq_f64` 不可用。
- **标签的 `player` 视角**：`add_weighted_labels` 假设"同一局内 P0/P1 决策点交错存储、并按 player 统一到 P0 视角
  再回签"（`az_train.py:97-105,114-117`）。`inject_ex02`/`lowq` 都是这种交错存储（本会话核实 5000 局每局 64 帧、
  `stats[36]` 的 player==0 标记恰好在帧间交替）⇒ **不要自己重排样本顺序**，否则符号会错。

### 6.4 一个"要不要动标签语义"的判断题

如果 P0 的数据侧检查说"金币在数据里可识别"，那**只换标签就够**；
如果不可识别，最小改动升级为：**给标签加一个显式的金币项**（`label_t += c·(coins_t/300)` 或
`+ c·1[coins≥90 且 cd==0]`），`c` 由**续盘回归**估（`docs/rl_survey/c_reward_shaping.md:347-359`）。
⚠️ 这条路必须按该文档 §3.5 的四关验收（终局金币指纹等），否则会退化成"奖励攒钱"
（`c_reward_shaping.md:22-36, 147-155, 465-474`）。

---

## 7. 值网重训计划草案

### 7.0 目标与非目标

- **目标**：让 `value_state` 的 V(coins) 在 **90 币处出现折点**、`∂V/∂coins` 在 **cd==0** 时显著大于 **cd 大**时
  （条件斜率比 ≥2）。即"教会它金币×冷却的定价"。
- **非目标**：本轮不以胜率为第一判据。历史告诉我们"离线更好 ⇒ 棋力更差"出现过多次
  （`docs/dl_feasibility_20261026.md:49-55`），而换标签的**胜率读数历史上是负的**（kgeo50 −7.0pp、rel −13pp）。
- **纪律**：一次只动一个变量；每条臂跑完立刻把 P0-2/P0-3 探针跑一遍（**20 秒、纯 CPU**，`value_net_coin_probe_20261030.md:37-49`）。

### 7.1 P0（0.5 天，零训练）——先证明"数据里有没有这个信号"

按 `value_net_coin_pricing_hypothesis_20261030.md:654-664` 定的 P2，在**两份**数据上跑（纯 numpy，秒级）：

1. 局内 `coins ~ round` 的 R²（金币是不是"回合的确定性函数"⇒ 若是，则与 cd 完全共线，条件斜率不可识别）。
   **本会话在 `lowq_f64` 上的读数：R²=0.136** ⇒ 金币不是回合的从属变量，**这一条不构成障碍**（好消息）。
2. 控制 `round`、`hp_delta`、`cd` 后，`coins` 与**逐帧标签**的偏相关。**本会话读数：kgeo50 上 +0.087**（弱）。
3. **新增（关键）**：按 `cd` 分层，比较 `coins` 对**未来 k 帧 HP 变化**（`d_{t+k} − d_t`）的解释力。
   若"cd==0 时金币更能预测未来 10–20 帧的 HP 改善"⇒ 数据可识别，换标签值得做；
   若两层的系数无差别 ⇒ **换标签大概率无效**，直接跳到 §7.4 的 B 计划（省掉一轮训练）。

> **§12 已经把这条做了一半（在标签侧）**：`kgeo50` 标签在 `cd==0` 时的金币斜率是 `+0.0127/100 币`、`cd>0` 时 `−0.0032/100 币`
> （比 3.96×，方向对），但 `≥90` 处**没有台阶**、水平效应只有 0.9% 个 std ⇒
> **"cd 条件信号"在标签里存在但极弱，"90 币阈值"在标签里不存在**。所以 P1 的胜负手不在标签，而在 §8 的数据。

> 这一步的意义：把"值网学没学会"和"数据里有没有"分开。现在的探针只能测**网络里有没有这个函数**，
> 测不出"数据支不支持这个函数"。

### 7.2 P1（半天）——单变量换标签，4 条臂

数据固定 **`inject_ex02`（部署血统，369,476 帧）**；起点 ckpt 固定 `three_mix_r10p_vw_pol_frozen.pt`；
`--freeze-bn`（默认开）、`--epochs 3`、`--lr 3e-4`（与 `valnet_A2` 完全同配方 ⇒ **唯一变量 = 标签**）。

| 臂 | 标签 | 命令要点 | 部署协议变更 | 预期 |
|---|---|---|---|---|
| **A0 对照** | `terminal`（重训一遍） | `--label-mode terminal` | 无 | 应≈复现 `valnet_A2`（否则说明数据/缓存重建有问题） |
| **A1 首推** | `abs + kgeo + tau50` | `--label-mode abs --label-weight kgeo --tau 50 --label-scale 1.5` | **无** | 逐帧信号有、捷径被削（R² 0.72） |
| **A2** | `rel + kgeo + tau50` | `--label-mode rel --label-weight kgeo --tau 50 --label-scale 4.0` | `AZAI_VALUE_TANH=0 AZAI_REL2ABS=4.0` | 唯一无捷径（corr −0.24），但要改协议 |
| **A3** | `abs + geo + tau10` | `--label-mode abs --label-weight geo --tau 10 --label-scale 1.0` | 无 | 短视野对照（tau10 R² 0.988 ⇒ 捷径最重） |

成本：缓存重建（400 局 pkl → memmap，16 workers）+ **104 s/epoch × 3** × 4 臂 ≈ **25 分钟**（`docs/next_step_survey_20261005.md:29`）。
每条臂训完立刻跑探针（`code/test_match/probe_value_coin_curve.py` + `probe_value_grad.py`）。

**第二条数据（可选，若 P0 显示 lowq 的对照更强）**：`lowq_f256` 用 `train_value_stream.py --label-file label_kgeo_tau50.npy`
（已有 ckpt：`checkpoints/rel_kgeo50/*.pt`）→ `graft_value_ckpt.py` 合进三网 ckpt。成本高（10–70 min/臂），
但**局数是 5000 而不是 400**，对"条件斜率"这种需要大量对照的量更可靠。

### 7.3 验收判据（照抄探针文档，先写死再跑）

**硬判据（两条必须同时成立）**

| # | 判据 | 阈值 | 出处/现状 |
|---|---|---|---|
| 1 | **90 币折点**：`coins=0.30` 左右斜率比 & "最大斜率突变 / 平均斜率" | 斜率比落在 **[0.98,1.02] 之外** 且最大突变 **≥20%**（且突变点应在 90 附近，而不是 8/45/53/75/82/128/188/240 那种散点） | `value_net_coin_probe_20261030.md:304-313`；**现状 0.98–1.02、最大突变 3.0% ⇒ 完全没折点** |
| 2 | **条件斜率比** `∂V/∂coins|_{cd=0}` / `∂V/∂coins|_{cd≥0.6}` | **≥2.0**（探针原定 1.5，本轮按用户要求抬到 2） | `value_net_coin_probe_20261030.md:292-302`；**现状 0.99×** |

**回归门（任一条破就否决该臂）**

| # | 门 | 阈值 | 出处 |
|---|---|---|---|
| 3 | 有塔样本 MSE 不劣化 | ≤ 现状 0.05794（或同数据重算的 A0 值） | `docs/value_net_pipeline_survey.md:178` |
| 4 | val 相关系数不劣化 | 整体 Pearson r 不低于 A0 − 0.02 | `train_value_net.py:176-185` |
| 5 | 128 局配对胜率不劣化 | ≥ 基线 −3pp（CI 覆盖即算过） | 历史判据至少 128 局（`docs/value_net_pipeline_survey.md:381-385`） |

**⚠️ 判据 1/2 的已知局限（必须一起写进结论）**：探针的 cd 维度**大部分是外推**
（有塔组 76/103 帧 cd==0，`value_net_coin_probe_20261030.md:326-327,411-412`）。所以
① 验收报告必须**同时**给出"真实 cd>0 帧上的读数"；② 若某臂只在 cd==0 的帧上有改善，**不能宣称学会了时机**。

### 7.4 若 P1 全败（B 计划，按成本排序）

1. **先确认是"学不到"还是"数据没有"**：把 P0 的第 3 步在更细的分层上重做（coins 桶 × cd 桶 × 有塔，各 >500 帧才报）。
2. **换数据来源（§8）**：用"当前 ckpt 自对弈 + 亚军 oracle 条件化注入"现场采集 2000 局，重跑同样的 4 条臂。
   这是**最可能真正解决问题**的一条（我们自己的对局里闪电只占 0.6% 的决策，见 §8.1）。成本：10–25 h 采 + 25 min 训。
3. **改标签（显式金币项）**：续盘回归估 `c`，写成 `label_t += c·coins_t/300` 或 hinge 项；
   终局金币必须归零/中性化，否则会奖励攒钱（`c_reward_shaping.md:147-155, 455-474`）。**需要跑对局（续盘）**。
4. **改数据（有对照的采集）**：新采一批**刻意的"花 vs 攒"对照**局面（同局面 clone、一边 build 一边 hold，
   各续跑 N 回合看 HP），直接把"金币的边际 HP"当标签。这是最贵但最"对症"的一条，且需要预注册。
5. **不要做的事**：加大数据量/帧数（已三面撞墙：R² 上限 0.47）、加参数、把亚军手写值当标签（0.3906）。

### 7.5 风险与先验（写在最前面，避免自我欺骗）

| 风险 | 证据 |
|---|---|
| **换标签这条轴历史上是负的** | `kgeo50` 128 局 **0.4609（−7.0pp）**、`rel geo_tau50` **0.6016（−13pp）**、与旧线"rel −7.0pp"方向一致（`docs/experiment_log.md:11384-11395`、`docs/state_of_play_20261006.md:86`、`docs/goal2_value_label_prior_art.md:30-39`） |
| **48 局/32 局的快读数会骗人** | `kgeo50` 32 局 **+9.4pp** → 128 局 **−7.0pp**；16 局 0.625 两次一致但仍不代表（`docs/state_of_play_20261006.md:86-88`） |
| **离线指标与棋力脱钩** | 值网 R² 0.45–0.47 与胜负只是弱相关（`docs/rl_survey/c_reward_shaping.md:490-493` 引文）；"拟合更准 ⇒ 棋力更差"出现过 5 次 |
| **一局一个标量 ⇒ 有效样本量≈局数** | `docs/az_iterative_value_loop_plan.md:19-24`；336 局时代 4 个种子聚在 25–30%（`docs/az_value_data_scale_hypothesis.md`） |
| **value_state 的 BN 从未训过** | 必须 `--freeze-bn`，否则"变好"里混进 BN 适配（`train_value_net.py:216-221`、`docs/value_net_pipeline_survey.md:351-356`） |
| **按帧切 val 会泄漏 10×** | 一律按局切（`docs/value_net_pipeline_survey.md:364-369`） |

---

## 8. 数据来源：该怎么换、bootstrap 怎么破

> 本节是 2026-10-31 追加的重点。结论先行：**标签换不换是第二位的，"这批对局是谁打的"是第一位的。**
> 现成三份数据的**动作分布两两不同**，而且"放不放超武/闪电"的频率在 `lowq`（12.4 次/局/方）与我们自己的对局（闪电占 0.6% 决策）之间差一个量级；
> 只要采集策略还是"我们现在的 AI"，那么无论换什么标签，学到的都是"在当前这个不会攒钱/不放闪电的策略下，
> 什么局面更有利"——**它永远见不到"该攒钱的那个分叉"**。破法只有一个方向：**把"动作质量"外部化**（注入/规则/或别的 AI），
> 让对局里**先有**这个分叉，再用逐帧标签去学它。

### 8.1 三份现成数据的动作分布（这是"分布错位"的硬证据）

| 数据 | 策略 | 有塔帧 | `cd==0` 帧 | **超武使用次数**（`stats[34]` 上升次数，含闪电/EMP/偏转/闪避） | 备注 |
|---|---|---|---|---|---|
| `inject_ex02`（`valnet_A2` 的训练集） | `posnet_r1` + ExampleAI 动作注入 2% | **38.9%**（文档；本会话 3 局实测 42.4%） | 11.0% | 未逐局统计（`stats[34]` 93.6% 的帧 >0 ⇒ 用过） | 注入前是 **0.0% 有塔**（`experiment_log.md:3064-3068`） |
| `lowq_f64/f256`（本机大数据） | **M6c 裸策略 + ε=2% 随机动作** | 36.8% | **12.8%** | **12.4 次 / 局 / 方**（max 18；10000/10000 个 (局,方) 都 ≥1 次；按每局每方 usage 终值差算） | 本会话实测 |
| 亚军自对弈 `entity_ds_128` | 亚军本体（手写值含 等价金币×10） | —（实体特征，无 board 通道可数） | 35.9% | **6.2 次 / 局 / 方**（256/256 都 ≥1） | 本会话实测；128 局 / 127,578 帧 |
| **当前部署策略**（unified cand + pin） | 我们现在的搜索 | 建塔 pin **6.7%** / 降级 **11.5%** / 闪电 **3.8%** / HOLD **78.0%** | — | **闪电只占 0.6% 的决策**（2/358，**单局读数**） | `docs/always_hold_ab_20261029.md:55`、`docs/unified_cand_fix_20261029.md:637` |

**读法**：
1. 用户说的"老版本只放闪电、从不建塔"**对 `inject_ex02` 的祖先成立**（0.0% 塔），但 `inject_ex02` 本身已经修到 38.9%，
   `lowq` 也有 36.8% ⇒ **"没有塔"已经不是主要矛盾**。
2. 真正的错位在**超武/闪电**：`lowq`（M6c）每局每方用 **12.4 次**，亚军 **6.2 次**，而**我们现在的搜索里闪电只占 0.6% 的决策**
   （单局读数；但方向与"92.9% 把钱花在塔上"一致）⇒ "该攒 90 币等 cd 落地"这个分叉，**在我们自己的对局里几乎不存在**
   （放得少 ⇒ 也就没有"放了之后 HP 怎么变"的样本）。
   ⚠️ 注意 `stats[34]` 是**全部超武**的使用次数，不是只有闪电；但闪电是唯一 90 币、35 回合冷却的那个，其余三个的价格/冷却见 `constants.py`。
3. 反过来，若直接用 `lowq` 训，学到的是"**M6c 那样频繁放超武**的后果"——与部署策略的后果不同。
   ⇒ **两头都不对**：老数据缺塔/口径旧、lowq 是弱策略、当前策略不放闪电。这就是"换数据来源"必须做的原因。

### 8.2 bootstrap 死循环的精确表述

> 要采到"建塔/攒钱做得好"的对局，需要一个会正确定价的值网；但现在的值网定价不准 ⇒ 用它自对弈采出来的还是烂数据。

这个循环是**真的**，但它的前提是"采集动作必须由我们的策略+值网决定"。**只要把动作来源换掉，循环就断了**——
因为循环里缺的是"动作质量"，不是"值网"。

### 8.3 五条可选路径（按可行性排序，全部有仓库证据）

| # | 路径 | 仓库里有没有 | 成本 | 判断 |
|---|---|---|---|---|
| **A** | **把亚军 oracle 的"非 HOLD 动作"注入 `az_selfplay` 采集**（动作级注入，条件化触发） | **半现成**：`--inject-example-prob` + `_example_legal_bundle` 已经是同构机制（`az_selfplay.py:426-431,354`）；oracle 客户端在 `test_match/az_bridge_ai.py:1538-1600` 一带（`RunnerUpOracle`，`_us_vs_external.sh:72-73` 明确说它"只是按官方协议喂局面/收操作的通用外壳，不绑定亚军"） | 新代码 ~200–300 行；采集 2.2–12 min/局 | ✅ **首推** |
| **B** | **候选级注入 + bridge 自对弈 dump**（成品，直接能跑） | **全现成**：`AZAI_ORACLE_INJECT=<亚军exe>` + `AZAI_ALWAYS_HOLD=1` + `AZAI_ORACLE_SKIP_HOLD=1` + `AZAI_AZ_DUMP` + `AZAI_DUMP_RAW_DIR`，驱动 `_us_vs_external.sh` / `match_via_sdk.py`（E3 就是这么跑的，`docs/nonhold_inject_ab_20261029.md:15-31`）；BC 已用同一条路采过 32 局（`training_history/bc_dataset/seed*.npz` 内含 `board(325,28,19,19)+stats(325,42)`） | **43.3 min/局中位**（`nonhold_inject_ab:178`） | ✅ 可行但**只能精采几十~几百局** |
| **C** | **用规则/样例 AI 整局驱动采集**（不是注入） | `ExampleAI`（`--inject-example-prob` 用的就是它）+ `rv4_pkg`（rule_v4）都在 `code/test_match/`；raw 自对弈 **7.7 s/局**（`docs/az_iterative_value_loop_plan.md:26-34`） | 很便宜 | ⚠️ 动作质量中等；策略分布离部署更远，只适合当"多样性补充" |
| **D** | **复用现成 replay / 对局日志** | ❌ **不可行**：`match_results/ladder_logs/` 有 **6,141 个日志 / 6.6 GB**，但日志重放实测 **144 局只有 2 局**能复现终局血量（`code/test_match/_fill_doc_bc.py:19-20`）；能用的只有 `AZAI_DUMP_RAW_DIR` 的 obs `.bin`（权威快照），现存 **22 个文件**（`training_history/distill_top2/sp_bin*/`）⇒ 量级只有几十局 | — | ❌ 不能用 |
| **E** | **亚军自对弈的 128 局**（`entity_ds_128`，board+stats+seed+player+round） | 数据现成；**逐帧 HP 标签可直接从 `stats[1]` 重算，终局标签可从每局最后一帧的 `stats[24]/[25]` 还原** ⇒ **零对局成本** | 0（只算标签） | ✅ 唯一"不跑对局就能加"的新分布数据，但只有 128 局 |

> ⚠️ 冠军 oracle 有已知 bug：`OnRoundState` 影子失配 ⇒ `force_pass_=true` ⇒ **99.93% 的回合返回 `()`**
> （`docs/always_hold_ab_20261029.md:53`、`docs/e4_only_extra_20261029.md:270`）。
> ⇒ **注入只能用亚军**，别用冠军。

### 8.4 明确回答：bootstrap 用什么破

**用"亚军 oracle 的非 HOLD 动作，按经济状态条件化注入"来采（路径 A）。** 三条理由：

1. **它把动作质量外部化** ⇒ 采集时不需要一个会定价的 V，循环断在这里。
2. **有正面证据**：E3（`AZAI_ALWAYS_HOLD=1` + 只注入亚军非 HOLD 动作）24 局 **45.8%**，基准臂 **16.7%**（+29.2pp，
   `docs/nonhold_inject_ab_20261029.md:111-120`）；更早的"注入亚军当候选"16 局 **12.5% → 50.0%**（`docs/inject_vs_runnerup_20261028.md` 一节的读数）。
   注入内容 84% 是 HOLD ⇒ **必须配 `AZAI_ORACLE_SKIP_HOLD`（或我们的条件化触发）**，否则只是把 HOLD 又喂一遍。
3. **它不改我们的执行链**：`inject_example_prob` 的语义是"**搜完之后把 chosen 换掉**"（`az_selfplay.py:426-431`），
   所以采集出的对局是"我们的搜索 + oracle 的若干手"——对**值**训练完全够用（我们只要 `board/stats/player` + 结局），
   对**策略**训练会不一致（visit 目标是我们的搜索给的）⇒ **本批数据只用于值网**，别拿去训策略。

**代价（必须写进结论）**：亚军自己的定价也不对——它的手写值把金币按 **≤400 部分×10、超出×8** 线性计价，
**完全不含冷却**（`docs/rivals_analysis.md:333-357`）。所以注入采出的分叉是"**亚军的定价**"，
学到的上限是亚军；这比"零监督"好，但不是"正确汇率"。要得到正确汇率仍需 §7.4-2 的续盘回归。

### 8.5 采集脚本要改哪几处（路径 A 的施工清单）

| # | 改动 | 位置 | 规模 |
|---|---|---|---|
| 1 | `--search {bundle,flat,unified}`：把 `BundleMCTS(...)` 换成 `UnifiedActionSearch(...)` / `FlatRolloutGreedy(...)`（**两者都是 `BundleMCTS` 的子类，`search()` 签名与返回结构相同**：`flat_rollout_greedy.py:55,522`、`unified_action_search.py:50,195`） | `az_selfplay.py:517-525` | ~15 行 |
| 2 | 抽一个 oracle 客户端（会话生命周期/超时/重建/非法丢弃） | 从 `test_match/az_bridge_ai.py:1538-1600` 搬；协议见 `docs/oracle_inject_design.md:76-131` | ~150–250 行 |
| 3 | `_oracle_legal_bundle(state, player)` + `--oracle-exe/--inject-oracle-prob/--oracle-skip-hold` | 照 `_example_legal_bundle`（`az_selfplay.py:426-431` 调用点） | ~40 行 + argparse |
| 4 | **条件化触发**（关键）：只在"经济相关决策"上问 oracle（`coins ∈ [cost−ε, cost+ε]` 或 `cd==0 且 coins≥90`），命中时 p=0.3–0.5，其余 p=0 | 同上分支 | ~15 行 |
| 5 | 样本里记 `inject_src` / `chosen_class`（便于事后数"花 vs 攒"的对照数） | `az_selfplay.py:432-461` | ~5 行 |

**为什么第 4 条是关键**：E3 实测注入内容 **84.5% 是 HOLD**（8117 次注入 / 6857 次 `()`，`docs/nonhold_inject_ab_20261029.md:61`）。
如果只在随机 2% 决策上问 oracle，命中"经济分叉"的次数会低到没统计意义；条件化触发能把 oracle 的查询预算
集中到我们真正缺的那类决策上（这也顺带把 oracle 查询开销从"每局 900 次"降到几十次）。

### 8.6 采集量级估算

| 参数 | 取值 | 出处/依据 |
|---|---|---|
| 决策点 / 局 | 325（bridge+HOLD skip）~ 924（`az_selfplay` 全量） | `bc_dataset` 325；`inject_ex02` 369,476/400=924 |
| 每条臂的墙钟 | `az_selfplay`+`--skip-hold-search`：**2.2 min/局**（16 workers）；bundle 全强度 30 min/局；bridge 自对弈 **43.3 min/局**；raw 策略 7.7 s/局 | `docs/az_iterative_value_loop_plan.md:26-53`、`nonhold_inject_ab:178` |
| 目标量 | **2000 局**（≈ 65 万~180 万决策点） | 336 局时代"重尾随机"（种子间 25–30%），5000 局才稳定（`docs/value_net_pipeline_survey.md:381-385`） |
| 墙钟（路径 A，in-process unified + 5% oracle） | **≈ 10–25 h**（16 workers）。若用 bundle+skip-hold 则 ≈ 5 h | 由上面两个锚点外推 |
| 关键 KPI（不是局数） | **闪电事件数**：lowq 12.4/局/方、亚军 6.2；目标 **≥1 万次**"cd==0 且放/不放"的对照 | 本会话实测 + §8.1 |

> 结论：**这是"过夜级 / 十小时级"的采集**，符合项目一贯的时间预算（CLAUDE.md：小时级到十小时级可接受），
> 但**不能在一小时内做完**。所以 §7.2 的"用现成数据换标签"那 4 条臂仍值得先跑（25 分钟），
> 它决定的是"标签形式对不对"，与"数据够不够"是两个独立问题。

### 8.7 两条必须一起写进结论的反例

1. **注入改变分布**：注入率高 ⇒ 非 HOLD 更多、局更长、塔更多；训出的 V 在"纯自对弈分布"上可能 OOD。
   缓解：注入率 ≤10–20%，**保留一半纯自对弈数据**混训（并分别报有塔/无塔 + coins/cd 分层指标）。
2. **注入采出的"好动作"是亚军的动作**：若亚军在某些机制上系统性偏弱（它是单边 rollout + 手写值，`docs/oracle_inject_design.md:256-263`），
   这套 V 会把亚军的偏差学进去。**"V 学会了亚军的定价"与"V 学会了正确汇率"是两件事**，验收必须用 §7.3 的形状判据，而不是"胜率涨了"。

---

## 9. 未找到 / 无法确认（诚实清单）

1. **任何显式含"冷却"的现成标签** —— 未找到（10 个 HP 差族标签全部只读 `stats[1]`；亚军值只含金币/威胁/闪电 bonus，不含 cd 剩余时间）。
2. **"值网 R(coins)"以外的金币读数** —— 未找到（只有 `∂V/∂x=+0.0274` 那一个数，且是无塔轨迹，`docs/value_net_coin_pricing_hypothesis_20261030.md:298-322`）。
3. **`inject_ex02` 训练局面的金币/冷却分布** —— 本文之前未找到；**本会话补测了 3 局（2,614 帧）**：coins p50=0.343、≥90 币 57.2%、cd>0 89.0%、usage>0 93.6%。
   **全 400 局的比例未测**（要读 17 GB pkl，本轮没做）。
4. **`lowq_full` 的全量逐帧标签（10 个 .npy）** —— **本机已被清理**（`Test-Path` 失败），只能用 F: 的 `lowq_e2_merged` 重算。
5. **`quiet_value` 的已落盘数据集** —— 未找到（只有 dump 开关 `AZAI_DUMP_QUIET=1` 与消费者代码）。
6. **"coins×cd 交互在数据里可识别"的任何实测** —— 未找到；§7.1 第 3 步是本文新提的检查，**尚未跑**。
7. **当前 `lowq_f64` 的 `label.npy` 是否为 player-signed** —— 文档只写"局末血量差/20"
   （`docs/value_net_pipeline_survey.md:78`），**本会话实测确认每局恰有 2 个取值 ±|v| ⇒ 已按 player 定符号**。
   这与 `subsample_frames.py:77` 直接拷 `value_win.npy` 一致，说明符号在 `merge_lowq_data.py` 那一步就已经写好。

---

## 10. 复现本文的读数（全部只读、无 GPU）

```powershell
$py = "D:\anaconda3\envs\pytorch-gpu\python.exe"   # 本会话实测用的是 numpy 2.3 的独立解释器，等价
$env:PYTHONIOENCODING="utf-8"
```

**① 数据集规模/覆盖（§5）**：mmap `lowq_f64/{stats,board}.npy`，取
`stats[:,22]`(coins/300)、`stats[:,26]`(我方闪电 cd/35)、`board[idx,4]`(己方塔) 的分布；
`lowq_f256` 同法；`inject_ex02` 读 3 个 pkl 的 `samples[*]["stats"]`。

**② 标签是不是局内常数（§1.3, §4.2）**：按 `game_id.npy`（或 `game_id*2 + [stats[36]>0.5]`）分组，
算组内 std / 组内去均值后的相关。**关键读数**：`terminal` 组内 std = **0.000000**；`kgeo_tau50` = **0.1064**。

**③ P2 可识别性（§7.1）**：局内去均值后 `coins ~ round+round²` 的 R² = **0.136**；
`kgeo50` 与 `coins` 的偏相关（控制 round/hp/cd）= **+0.087**；与 `cd` 的偏相关 = **+0.028**。

**④ 历史逐帧臂的离线读数**：`Get-Content training_history/distill_top2/train_f256_*.log -Tail 6`
（geo50 0.5979 / kgeo30 0.5580 / kgeo50 0.4799 / kgeo100 0.3590 / kgeo200 0.2825；不同标签 ⇒ 不可横向比）。

**⑤ §12 的标签阈值诊断（一键复跑，秒级）**：
```powershell
$env:PYTHONIOENCODING="utf-8"
& D:\anaconda3\envs\pytorch-gpu\python.exe -u code/test_match/probe_kgeo_coin_threshold.py --tag f256 > match_results\probe_kgeo_coin_threshold_f256.txt 2>&1
& D:\anaconda3\envs\pytorch-gpu\python.exe -u code/test_match/probe_kgeo_coin_threshold.py --tag f64 --data training_history/distill_top2/lowq_f64
```
输出含：pooled/within 两组回归（coins、+round、+round+hp）、断点回归（90）、hinge+step 阈值检验（带 t）、
4 个金币桶的 pooled/within 均值、90 附近 ±10 币局部、cd 分层条件斜率、双向固定效应（(局,player)+回合十分位）。

**⑥ §13 的"统一搜索自己下的对局"诊断（一键复跑，秒级）**：
```powershell
& D:\anaconda3\envs\pytorch-gpu\python.exe -u code/test_match/probe_unified_kgeo_coin.py --all-unified > match_results\probe_unified_kgeo_coin.txt 2>&1
```
输出含：我方 `coins≥90` 占比（逐 tag + 加总）、4 个金币桶的 pooled/within 均值、`Δ(≥90−60-89)`、
`coins`/`+round`/`+round+hp` 三档斜率、按局 bootstrap 的 90%CI、双向固定效应（局+回合十分位）、
以及**对手视角（亚军）的对照**。（`--dirs` 可指定 tag；`--tau` 可换时间常数。）

---

## 11. 引用清单（按文件）

- `code/my_ai/az_intent/train_value_net.py`：`:13-15`（标签说明）、`:52-62`（缓存字段）、`:65-129`（缓存构建/memmap）、
  `:162-185`（评估，分组在 `:166,179`）、`:208-213`（`--label-mode/--tau/--label-scale/--label-weight`）、
  `:216-225`（`--freeze-bn` 默认开）、`:261-283`（terminal vs 逐帧分支；**零信号注释在 `:265-267`**）、`:331-332`（MSE）
- `code/my_ai/az_intent/az_train.py`：`:56-135`（`add_weighted_labels` 全实现）、`:828-840`（训练器里的同一开关）
- `code/my_ai/az_intent/value_warmup.py`：`:81-86`（terminal 标签）、`:143-212`（`_exp_weighted_labels`）、`:215-242`（`_load_chunk`）
- `code/my_ai/az_intent/az_selfplay.py`：`:187-188`（`REL2ABS` 还原）、`:484-489`（terminal 标签）、`:545-556`（npz + ents jsonl 字段，含 `coins/weapon_cd`）
- `code/my_ai/az_intent/selfplay.py`：`:37-47`（`terminal_value_target`）
- `code/my_ai/az_intent/mcts.py`：`:46`（`HP_SCALE=20`）
- `code/my_ai/az_intent/value_batch.py`：`:22-65`（叶子值批处理、`tanh`）、`code/my_ai/az_intent/value_override.py:13-31`（`AZAI_VALUE_CKPT` 只换叶子）
- `code/my_ai/network.py`：`:184-214`（价值头裸线性、`value_pool`）、`:236,241,272,283`（GAP ⊕ stats → value）
- `code/my_ai/entity_transformer/build_rel_labels.py`：`:5-10`（语义 + 部署开关）、`:27,66-68`（复用 `_exp_weighted_labels`）、`:56-99`（全量标签 `:60-70`、落盘 `:94-96`）
- `code/my_ai/entity_transformer/subsample_frames.py`：`:40,77`（`label.npy` = `value_win.npy`）、`:58`（抽样 seed）
- `code/my_ai/entity_transformer/build_frame_subset.py`：`:39,74,91,95-96`（terminal + extra 标签一起抽）
- `code/my_ai/entity_transformer/train_value_stream.py`：`:41`（`ValueNetwork`）、`:55-56`（`--label-file`）、`:83-90`（标签来源）、`:112-118`（terminal 口径警告）、`:137-140`（z-score MSE）、`:69-71`（`--width/--blocks`）
- `code/my_ai/entity_transformer/graft_value_ckpt.py`：`:68-76`（`y_mean/y_std` 折进 `value_head`）
- `code/my_ai/az_intent/azjsonl_to_value_npz.py`：`:1-12,69-71`（`root_value` 标签）
- `Ant-Game/SDK/utils/features.py`：`:134-157`（22 维 named）、`:225-257`（extras；金币 `:227-228`、闪电 cd `:231-232`）
- `Ant-Game/SDK/utils/constants.py`：`:194`（闪电 `cooldown=35, cost=90`）
- `docs/value_net_pipeline_survey.md`：`:59-84`（lowq 全表）、`:104-115`（训练脚本表）、`:178`（valnet_A2 战绩）、`:228-239`（数据选型）、
  `:264-267`（损失）、`:344-369`（归一化/BN/按帧切泄漏）、`:381-397`（局内方差与帧数）
- `docs/az_pool_tau_label_plan.md`：`:89-111`（`abs` 捷径 R²=0.87）、`:113-158`（tau 扫描 + `kgeo` 取舍曲线）、`:163-180`（scale 标定）、`:182-192`（局内逐帧 std 表）
- `docs/value_label_future_weighted.md`：`:13-44`（公式与递推）
- `docs/az_iterative_value_loop_plan.md`：`:19-24`（有效样本量≈局数）
- `docs/next_step_survey_20261005.md`：`:8-23`（血统）、`:29`（H2 与逐帧标签的已知反证）、`:36-38`（缓存 + 标签重算 11.4 s）、`:53-59`（94/94 逐位比对）
- `docs/experiment_log.md`：`:3048-3090`（inject_ex02 采集）、`:3119`（`valnet_A2` 配方）、`:3064-3068`（塔覆盖 0→38.9%）、
  `:11365-11419`（四轴扫完：R² 上限 0.47、rel −13pp）、`:11399-11401`（反归一化事故）
- `docs/value_net_coin_probe_20261030.md`：`:15-27`（一页结论）、`:86-126`（V(coins) 形状）、`:248-337`（P0-3 曲面与三条判据）、`:397-418`（局限）
- `docs/value_net_coin_pricing_hypothesis_20261030.md`：`:382-397`（B.5 精度上限）、`:416-445`（冷却盲区）、`:467-493`（训练配方）、`:654-664`（P2 数据侧检查）
- `docs/rl_survey/c_reward_shaping.md`：`:22-36`（金币 shaping 必然被刷）、`:147-155`（四种 potential 的合法性）、`:347-359`（续盘回归估汇率）、`:465-474`（把金币放进 `V`，别放进奖励）
- `docs/state_of_play_20261006.md`：`:86-88`（kgeo50 128 局 −7.0pp、快读数骗人）
- `docs/goal2_value_label_prior_art.md`：`:30-39`（标签轴前史：小、噪声、被执先/执后污染）
- `docs/checkpoints.md`：`:18-20`（钉死 ckpt 与 `valnet_A2`）、`:258-259`（lowq 值与 graft）、`:282-310`（32/3 与 folded 部署版）
- `docs/value_net_pipeline_survey.md:86-97`（C 族不要用）、`docs/entity_transformer_value_results.md:15-37`（亚军值表达力与消融）
- **数据来源 / 注入（§8）**：
  `code/my_ai/az_intent/az_selfplay.py:354,426-431`（`inject_example_prob` 的**动作级注入**：搜完替换 chosen）、
  `:432-461`（样本字段，可加 `inject_src`）、`:517-525`（`BundleMCTS` 实例化处 ⇒ 可换成 unified/flat）、
  `code/my_ai/az_intent/flat_rollout_greedy.py:55,522`（`class FlatRolloutGreedy(BundleMCTS)` + `search()`）、
  `code/my_ai/az_intent/unified_action_search.py:50,195`（`class UnifiedActionSearch(FlatRolloutGreedy)`）、
  `code/test_match/az_bridge_ai.py:1538-1600`（oracle 会话/超时/重建）、`:1455`（`AZAI_DUMP_RAW_DIR`）、`:277-286`（`AZAI_DUMP_QUIET`）、
  `code/test_match/_us_vs_external.sh:72-87`（E1–E4 的注入环境）、`:104-117`（`SKIP_HOLD`/重建开关）、
  `code/test_match/probe_value_coin_curve.py:62-153`（从 `[round N] pX(...)` 日志复现局面的解析器）、
  `code/test_match/_fill_doc_bc.py:19-24`（**日志重放不可行** + BC 采集链路）、
  `training_history/bc_dataset/seed*.npz`（32 局 × 325 帧，含 `board+stats`）
- `docs/oracle_inject_design.md`：`:10-18`（候选注入语义）、`:19-45`（环境变量）、`:76-131`（桥接侧生命周期/合法性/兜底）、`:256-263`（已知限制：单边 rollout）
- `docs/nonhold_inject_ab_20261029.md`：`:15-31`（E3 两臂定义）、`:111-120`（45.8% vs 16.7%）、`:61`（注入内容 84.5% 是 HOLD）、`:178-179`（43.3 min/局、决策 p50 2.10 s）
- `docs/inject_vs_runnerup_20261028.md`（注入亚军 12.5%→50.0%）、`docs/always_hold_ab_20261029.md:53-55`（冠军 oracle 恒空过 / pin 分布）、`:150-159`（HOLD 率 75.7%→87.5%）
- `docs/e4_only_extra_20261029.md:270`（冠军 oracle `force_pass` 的机制）、`docs/unified_cand_fix_20261029.md:637`（闪电 2/358 = 0.6%）
- `docs/rivals_analysis.md:333-357`（亚军 `evaluate_terminal` 全权重：金币×10 分段、无冷却）、`docs/az_iterative_value_loop_plan.md:26-53`（采集速度锚点）
- **§12 的标签诊断**：`code/test_match/probe_kgeo_coin_threshold.py`（本文新增，纯 numpy）；
  读出字段 `Ant-Game/SDK/utils/features.py:227-228`（coins）、`:231-232`（我方闪电 cd）、`:134-136`（hp_delta/round）；
  标签公式 `code/my_ai/az_intent/az_train.py:119-129`（`kgeo` 递推）、`value_warmup.py:193-211`；
  原始输出 `match_results/probe_kgeo_coin_threshold_{f256,f64}.txt`（`.gitignore:30`，不入库）
- **§13 的统一搜索对局诊断**：`code/test_match/probe_unified_kgeo_coin.py`（本文新增，纯 numpy）；
  数据 `training_history/progress/<TAG>/az_progress_seed*.txt`（写入器 `code/test_match/progress.py:98-126`，
  格式与"HP/G = 决策时刻"的说明在 `:5-14`）；座位约定 `code/test_match/_us_vs_external.sh:223-231`、
  `code/test_match/_pin_k2_vs_runnerup.sh:161-169`；驱动 `code/test_match/_unified_cand_ab.sh:145`（`--ai0=az_bridge_ai.py` ⇒ 无 `r` 时我们坐 P0）；
  原始输出 `match_results/probe_unified_kgeo_coin.txt`（gitignore）

---

## 12. 追加诊断：`kgeo tau=50` 标签里有没有"金币≥90"这个阈值？（2026-10-31 追加②，只读）

> 动机：§4 的结论是"逐帧标签只由未来 HP 轨迹算出来，没有任何一项显式含金币/冷却"。
> 但"没显式含"不等于"没编码"——金币多了，未来 HP 轨迹本来就可能更好。
> 所以直接问一句**能改变决策的话**：
> **`kgeo50` 标签里，"金币 ≥ 90（= 闪电价格）"有没有带来更高的标签值？**
> - **有阈值** ⇒ 标签已经含了这个结构，那我们要做的是**多采"卡在 90 币"的样本**（数据量问题）；
> - **只单调 / 完全无关** ⇒ 光换标签不够，**必须换数据或给标签加显式金币项**。
>
> 脚本：`code/test_match/probe_kgeo_coin_threshold.py`（纯 numpy，秒级，不训练、不跑对局）。
> 原始输出：`match_results/probe_kgeo_coin_threshold_f256.txt` / `..._f64.txt`（`match_results/` 在 `.gitignore:30`，不入库）。
> 数据：`lowq_f256`（**1,280,000 帧 / 5000 局**，10 个标签齐）+ `lowq_f64`（320k 帧）复核。
> 字段：金币 `= stats[22]×300`、冷却 `= stats[26]×35`（0 = 已好）、血差 `= stats[1]`、回合 `= stats[0]×512`、视角 `= stats[36]`。

### 12.1 口径（两个都报，别只看一个）

- **pooled**：所有帧混一起回归 —— 会被"局级混杂"污染（赢的局可能同时金币多、血差好）。
- **within**：按 **(局, player)** 去均值（固定效应）后再回归 —— 回答"**同一局同一方**，多一币时未来优势是否更好"，
  这才是"标签有没有教金币价值"的正确问法。
- 再加一档 **within + 回合十分位**（双向固定效应，`2d`）：把"局内时间趋势"也扣掉——因为金币是随回合累积的慢变量，
  不扣回合很容易把"时间趋势"读成"金币效应"。

### 12.2 回归：kgeo50 ~ coins（`f256`，n=1.28M；`f64` 复核在括号里）

| 模型 | slope/1 币 | **slope/100 币** | R² |
|---|---|---|---|
| pooled：`coins` | +0.000012（+0.000018） | **+0.0012**（+0.0018） | 0.0000（0.0001） |
| pooled：`coins + round` | +0.000013（+0.000019） | +0.0013（+0.0019） | 0.0000（0.0001） |
| pooled：`coins + round + hp` | +0.000096（+0.000102） | +0.0096（+0.0102） | 0.0200（0.0203） |
| within(局,player)：`coins` | −0.000154（−0.000151） | **−0.0154**（−0.0151） | 0.0050（0.0047） |
| within：`coins + round` | −0.000174（−0.000170） | −0.0174（−0.0170） | 0.0056（0.0053） |
| within：`coins + round + hp` | +0.000117（+0.000121） | +0.0117（+0.0121） | **0.3309**（0.3308） |

**读法（重要）**：斜率**连符号都不稳**（pooled ≈ 0；within 单独看是**负**的；一控 hp 就变正，R² 跳到 0.33——那一跳几乎全部来自 `hp`）。
⇒ **kgeo50 与金币的线性关系 ≈ 0**，任何"金币每多 100 值多少"的点估计都在 ±0.017 之间摆动，量级只有标签 std（0.107）的 **±16%**。

**阈值检验（within，控制 coins/round）**：`y ~ coins + round + (coins−90)⁺ + 1[coins≥90]`

| 项 | 系数 | se | t |
|---|---|---|---|
| `coins` | −0.000145 | 0.000002 | −61.6 |
| `round` | +0.000023 | 0.000001 | +28.7 |
| `(coins−90)⁺` | −0.000238 | 0.000020 | −12.0 |
| **`1[coins≥90]`** | **−0.006529** | 0.000713 | **−9.2** |

⇒ 90 币处的"台阶"**是负的且显著**（−0.0065），完全不是"跨过 90 就值更多"。
断点回归同结论：`slope(<90) = −0.000125/币`、`jump@90 = −0.0065`、`Δslope = −0.00024`。

### 12.3 分桶均值（核心判据）

**within (局, player)**，桶按金币数（`f256` / `f64`）：

| 桶 | 帧数（占比） | pooled 均值 | pooled std | **within 均值** | se | f64 within |
|---|---|---|---|---|---|---|
| `<45` | 186,270（14.6%） | −0.00633 | 0.1211 | **+0.00116** | 0.00026 | +0.00167 |
| `45-59` | 155,001（12.1%） | −0.00041 | 0.1148 | **+0.00473** | 0.00027 | +0.00426 |
| `60-89` | 321,612（25.1%） | +0.00143 | 0.1152 | **+0.00550** | 0.00019 | +0.00472 |
| `≥90` | 617,117（48.2%） | +0.00118 | 0.1116 | **−0.00440** | 0.00013 | −0.00402 |

- **`Δ(≥90 − 60-89) = −0.0099` = −9.2% 个标签 std**（f64：−0.0087 = −8.2%）⇒ **符号是反的**。
- 90 币附近 ±10 币的局部均值（within）：`[80,90) +0.00725` → `[90,100) +0.00655` → `[100,110) +0.00513`
  ⇒ **平滑下降，没有任何跳变**。
- 扣掉"回合十分位"后（双向 FE，`2d`）：`<45 −0.00794` → `45-59 −0.00172` → `60-89 +0.00119` → **`≥90 +0.00221`**
  ⇒ 变成**单调递增**了，但 `Δ(≥90 − 60-89)` 只有 **+0.0010（= 0.9% 个 std）**。
  **也就是说：单向前提下的 −9.9% 主要是局内时间趋势的伪影；扣掉趋势后剩一个"几乎为零的单调上升"，仍然没有台阶。**

**参考标签对照（同一批帧、同一分桶）**

| 标签 | `<45` | `45-59` | `60-89` | `≥90` | 说明 |
|---|---|---|---|---|---|
| `terminal`（pooled） | −0.0731 | −0.0492 | −0.0385 | **+0.0545** | "钱多 ⇒ 最终赢"在数据里**确实有**（跨度 0.128 = 0.33 std） |
| `terminal`（within） | 0 | 0 | 0 | 0 | 局内常数（= 本文 §1.3 的病根） |
| `geo_tau50`（within） | +0.0011 | +0.0026 | +0.0025 | −0.0023 | 与 kgeo 同形（都没阈值） |
| `kgeo50`（within） | +0.0012 | +0.0047 | +0.0055 | −0.0044 | 同左 |

⇒ 关键对照：**"金币多 ⇒ 最终赢"这件事只存在于 terminal（水平量）里；一旦换成差分标签（kgeo/geo），这层信息就被"减掉当前血差"的操作消掉了**，
剩下的只有"未来 50 帧相对现在改善多少"，而它与金币几乎无关。

### 12.4 冷却维度：`cd==0` 时金币才"稍微更值钱"（方向对、但很弱）

因为"金币×冷却"才是目标，所以在 ≥90 桶里再按冷却分，并做条件斜率（`f256`）：

| 口径 | `cd==0` | `cd>0` | 比值 |
|---|---|---|---|
| within + round，slope/100 币 | **+0.0220**（n=163,721） | **−0.0211**（n=1,116,279） | 符号相反 |
| **within + 回合十分位（双向 FE）**，slope/100 币 | **+0.0127** | **−0.0032** | **3.96×（符号 +/−）** |

`≥90` 桶内（within）：`cd==0` 均值 **+0.00495**（n=34,683）vs `cd>0` **−0.00496**（n=582,434）⇒ 差 **+0.0099（9.2% std）**。

**怎么读**：
- **方向是对的**：冷却已好（`cd==0`）时，金币与"未来改善"正相关（+0.0127/100 币）；冷却没好时是 0 甚至负。
  这与"cd 好了钱才值钱"的人工模型一致，也比"完全无关"要好。
- **但它是"斜率"不是"台阶"**：`cd==0` 组内的分桶（双向 FE）是 `<45 −0.0408 → 45-59 −0.0105 → 60-89 −0.0090 → ≥90 +0.0020`，
  单调爬升、**90 币处仍然没有跳变**（60-89 到 ≥90 只 +0.011，而 <45 到 45-59 就有 +0.030）。
- **量级太小**：`cd==0 & coins≥90` 的帧全库只有 **34,683 个**（= 3.5 次/局/方），
  而单帧标签的噪声 std 是 0.107 ⇒ 要靠这个弱信号学出可用的定价，信噪比约 **1:10**。
- **共线提醒**：`cd==0` 与"早局 / 刚放完超武"高度共线。本文只用了线性 round + 回合十分位两级控制，
  **没有**做"同一回合内、同一金币水平、cd 冲 0 前后"的严格配对，所以这一条只能当**提示**，不是结论。

### 12.5 一句明确回答

**`kgeo50` 标签里没有"金币 ≥ 90 ⇒ 值更高"这个阈值。** 它只有（a）一个**几乎为零的单调上升**
（扣掉时间趋势后 `≥90` 比 `60-89` 高 **0.0010 ≈ 0.9% 个标签 std**）、以及（b）一条**方向正确但很弱**的
"`cd==0` 时金币斜率 +0.0127/100 币、`cd>0` 时 −0.0032/100 币（比 3.96×）"。
**既没有台阶，也没有"crossing 90 的收益"**；单向前提下 `≥90` 桶甚至比 `60-89` **低 9.2% 个 std**（那是时间趋势伪影，不是反向信号）。

> ⚠️ **本节结论只对 `lowq`（M6c 老数据）成立。** 换成**统一搜索自己下的对局**（§13）后，
> `≥90` 桶反而**高 +0.088（≈0.56 个标签 std）**——两份数据的差别说明这是**分布问题**、不是标签问题。
> 但**统一搜索自己几乎不到 90 币（3.0% 的回合）**，所以两边合起来的行动结论不变：**先换数据**。

### 12.6 对"换标签 vs 多采数据"的启示

| 判断 | 依据 |
|---|---|
| **"只换标签"不够** | 标签里没有阈值结构。把 `terminal` 换成 `kgeo50` 只是把"局内同值"换成"局内变化"，**不会凭空长出 90 币折点**——折点必须来自数据（"跨 90 币后行为/结果变了"）。这也解释了历史：换标签（rel/kgeo）在 128 局验收上 **−7 ~ −13pp**（§7.5）。 |
| **"只多采数据"也不够（如果数据分布不变）** | 在 M6c 的分布里，`cd==0 & coins≥90` 只有 3.5 次/局/方；把同样的分布采 4 倍，只是把这个 1:10 的弱信号测得更准，**仍然是平滑的**。 |
| **正确做法：换数据（制造对照）+ 保留逐帧标签** | 需要把 `cd==0 & coins 跨越 90` 的**决策**大量制造出来，并且这些决策**真的改变后续 HP**。这正是 §8 注入采集要解决的问题：用亚军 oracle 在"经济相关决策"上条件化注入，让"放闪电 vs 攒着"两种动作都出现。 |
| **仍不行 ⇒ 加显式金币项** | 若注入采集后折点仍不出现，说明"HP 轨迹"这条通道不够，只能把金币直接写进标签（hinge at 90 或 `c·coins/300`，`c` 由续盘回归估），§6.4 与 `docs/rl_survey/c_reward_shaping.md:347-359`。 |

**一个额外的定量提醒**：本诊断里"标签对金币"的**水平效应**上限只有 **0.9–1% 个 std**（双向 FE），
而 P0-2 要求的是"90 币处**斜率突变 ≥20%**"。（严格说两者量纲不同：一个是水平差、一个是斜率比；
但把 1% 的水平差按 90 币的窗口折成斜率，仍然只有百分级。）⇒
**指望"现有 lowq/老数据 + 任何标签"训出折点，在数量级上就很不乐观**。这条是本轮最硬的结论，
也是"必须先换数据"的直接证据。

---

## 13. 追加诊断②：**统一搜索自己下的对局**里，`kgeo50` 有没有"金币≥90"信号？（2026-10-31 追加③，只读）

> §12 跑的是 `lowq_f256`（**M6c 老数据**），不是"我们现在的统一候选搜索"。这一节换成
> **统一搜索（unified cand）自己打的 match 对局**，用它们的 progress trace **现算** `kgeo50` 标签，再问同一个问题。
>
> 脚本：`code/test_match/probe_unified_kgeo_coin.py`；原始输出 `match_results/probe_unified_kgeo_coin.txt`（gitignore）。
>
> **数据**：`training_history/progress/<TAG>/az_progress_seed*.txt`
> （`code/test_match/progress.py:5-14`：`round=r HP0= HP1= G0= G1= P0=[...] P1=[...]`，
> **HP/G 是本回合操作前的值 = 决策时刻**，正好是要的"局面"）。
> **座位**：`seed` ⇒ 我们坐 P0、`seedr` ⇒ 我们坐 P1（`_us_vs_external.sh:223-231`、`_pin_k2_vs_runnerup.sh:161-169`；
> 已用 `UNIFIED_AB_A` 的 seed7 / seed7r 实测核对：无 `r` 那局 P0 第 0 回合就建 2 座塔，带 `r` 那局换成 P1 建）。
> **标签**：这些对局**没有现成标签**，按 `az_train.add_weighted_labels`（`az_train.py:119-129`，`kgeo`+`rel`）
> 与 `value_warmup._exp_weighted_labels`（`value_warmup.py:193-211`）逐步现算：
> `d_t = HP0−HP1`（统一 P0 视角）→ `raw_t = Σ k·γ^k·d_{t+k} / Σ k·γ^k`（`γ=e^{-1/τ}`，末帧退化回 `d_t`）
> → `label_t = clip((raw_t−d_t)/20, ±1)`（再按 player 翻符号）。
> ⚠️ **时间刻度**：trace **每回合一个样本**，训练数据是**每回合 2 个决策点** ⇒ 这里的 `τ=50` 回合 ≈ 训练数据的 `τ=100` 帧。
> 所以下面同时给 **τ=50（照题目）** 和 **τ=25（≈训练数据的 τ=50 帧，可比口径）**。

### 13.1 样本量（只有这些）

| tag | 局数（≥50 回合） | 回合数 | 说明 |
|---|---|---|---|
| `UNIFIED_SPLIT_VERBOSE` | 1 | 358 | 用户点名的"split+verbose 冒烟"（`az_progress_seed8_UCV.txt`） |
| `UNIFIED_AB_A` | 10 | 1,238 | legacy A/B，seeds 7,7r,8,8r,9,9r,10,10r,11,11r（配对座位） |
| `UNIFIED_AB_A_20261030_154735` | 19 | 5,443 | 同驱动的第二批 |
| `UNIFIED_CAND_{BOLTZ,BOLTZ2,BOLTZ3,FIX,FULL,SPLIT}` | 各 1 | 2,007 | 单局冒烟 |
| **合计（加总，鲁棒性）** | **36** | **9,046** | 全部是本机 `progress/UNIFIED_*` 里 ≥50 回合的对局 |

### 13.2 回答 1：我们自己的对局里，`coins≥90` 的回合占多少？

| 数据 | 我方 `coins≥90` 占比 | 我方金币 p50 | 我方金币均值 | **对手金币均值** |
|---|---|---|---|---|
| `UNIFIED_SPLIT_VERBOSE`（1 局） | **3.35%**（12/358） | 28.5 | 33.1 | 207.2 |
| `UNIFIED_AB_A`（10 局） | **0.89%**（11/1,238） | 27 | 29.8 | 116.8 |
| 加总 36 局 | **3.04%**（275/9,046） | 29 | 32.9 | 193.5 |
| 对照：`lowq_f256`（M6c，§12） | **48.2%** | 87 | 98 | — |

⇒ **完全复现父任务提到的"冒烟 3.4%"（实测 3.352%）**；而 lowq 是 48.2%、同一批对局里**对手（亚军）是 69.1%**。
**我们自己的搜索几乎从不把金币攒到 90**：中位数只有 **27–29 币**，p95 才 65–75 币。
换算成事件数：**每局只有 1–3 个回合**处于"钱够买闪电"的状态。

### 13.3 回答 2：现算的 `kgeo50`，`≥90` 桶 vs `60-89` 桶（within 局）

| 数据（τ=50 回合） | n(≥90) | `≥90` 均值 | `60-89` 均值 | **Δ(≥90−60-89)** | 占 within 标签 std |
|---|---|---|---|---|---|
| `UNIFIED_SPLIT_VERBOSE` | 12 | +0.0472 | +0.0159 | **+0.0312** | +19.9% |
| `UNIFIED_AB_A` | 11 | +0.0027 | −0.0131 | **+0.0158** | +22.5% |
| **加总 36 局** | **275** | — | — | **+0.0883** | **+55.8%** |
| 加总（τ=25，帧刻度对齐） | 275 | — | — | **+0.0290** | +23.4% |
| 加总（双向 FE：局 + 回合十分位） | 275 | +0.0893 | −0.0029 | **+0.0921** | +63.1% |
| 参考：`lowq_f256`（§12，within） | 617,117 | −0.0044 | +0.0055 | **−0.0099** | −9.2% |

**按局 bootstrap（2000 次重采样对局）**：加总 `Δ` 的 90%CI = **[+0.0405, +0.1198]**（τ=25：**[+0.0030, +0.0470]**）；
单局/10 局那种小样本的 CI 是 `[−0.021, +0.058]` 甚至退化 —— **明确是"仅提示"**。

**方向：正，不是反的。** 也就是说**不是"我们在 90 币时乱花钱 ⇒ 后续 HP 变差 ⇒ 标签更低"**，
而是在我们自己的对局里，"手里攒着钱"的回合**后面 HP 反而变好**（+0.09 ≈ 0.56 个标签 std）。

### 13.4 回答 3：回归斜率（within 局，τ=50）

| 数据 | `coins` | `+round` | `+round+hp` | R²(全控) |
|---|---|---|---|---|
| `UNIFIED_SPLIT_VERBOSE` | +0.0077/100 币 | −0.0416 | +0.0285 | 0.91（几乎全是 hp） |
| `UNIFIED_AB_A` | +0.0125/100 币 | +0.0112 | +0.0101 | 0.004 |
| **加总 36 局** | **+0.0618/100 币** | +0.0491 | **+0.0514** | 0.041 |
| 加总（τ=25） | +0.0318/100 币 | +0.0334 | +0.0348 | 0.017 |
| 加总（双向 FE） | +0.347/100 币 | — | — | 0.056 |

加总 `slope` 的按局 bootstrap 90%CI = **[+0.0287, +0.0911]/100 币**（τ=25：[+0.0149, +0.0464]）。
**控制"回合"和"血差"之后斜率仍是正的** ⇒ 不是"时间趋势"或"我领先所以有钱"的伪影。

### 13.5 视角对照：同一个 Δ，在"穷的一方"才出现

| 视角 | `coins≥90` 占比 | 金币 p50 | Δ(≥90−60-89) | slope/100 币 | n(≥90) |
|---|---|---|---|---|---|
| **我方**（统一搜索） | **3.04%** | 29 | **+0.0883** | **+0.0618** | 275 |
| **对手**（亚军，同一批对局） | **69.07%** | 151 | +0.0148 | +0.0197 | 6,248 |

⇒ 亚军那边 **69% 的回合都 ≥90 币**（它把金币囤到 400），"钱多"在它那儿是常态 ⇒ 与未来改善几乎无关（+0.015）。
**我们这边"钱多"是稀有事件**，稀有事件与"后面变好"相关（+0.088）。
这条很可能就是机制：**我们的花钱本身是净亏的**（建了又降、降了又建），所以"没花出去"的那几帧对应更好的后续。
⚠️ 这是**相关**不是因果：没有"同一局面一边花一边攒"的对照，分不清"钱的价值"和"安全/没被压"。

### 13.6 一句话结论

**统一搜索自己的对局里，`coins≥90` 只占 3.0% 的回合（每局 1–3 次，中位金币 29），
现算的 `kgeo50` 在 `≥90` 桶比 `60-89` 桶高 +0.088（≈0.56 个标签 std，按局 bootstrap 90%CI 排除 0）——
方向是"攒着钱 ⇒ 后续 HP 更好"，不是"乱花钱 ⇒ 标签更低"；但 275 个事件只够当提示，
而且它更像"我们花钱净亏"的侧写，不是"90 币是门槛"的证据。**

### 13.7 这条对计划意味着什么（修正 §12 的措辞）

| 原判断（§12，基于 lowq） | 修正（本节，基于我们自己的对局） |
|---|---|
| `≥90` 桶比 `60-89` **低** 9.2% std ⇒ "标签里没有钱的价值" | **在自己的对局里是正的**（+56% std）。两份数据差得远，**说明这是"分布"问题不是"标签"问题** |
| "光多采同分布数据不够" | **仍然成立，但理由变了**：不是标签没信号，而是**这个状态在我们对局里根本不存在**（3.0% vs lowq 48.2%）⇒ 采同分布数据，等于把 3% 的稀有事件再采一遍，学到的是"别乱花钱"而不是"90 币的门槛价值" |
| "必须换数据（§8 注入采集）" | **更成立**：注入采集要做的正是把 `cd==0 & coins 跨越 90` 的**状态**从 1–3 次/局提高到 10+ 次/局，并让"放 vs 不放"两种动作都出现 |

**一句话**：`kgeo` 标签**能承载**"钱的价值"（在自己对局里方向是正的），**缺的是"钱够买闪电"这个状态本身**；
所以 §8 的注入采集（并且要**故意让 oracle 在 60–90 币时"攒"、在 ≥90 且 cd==0 时"放"**）比任何标签改动都优先。

### 13.8 局限（必须一起读）

1. **样本小**：加总只有 36 局 / 9,046 个回合 / **275 个 `≥90` 回合**；单看用户点名的两处（1 局 + 10 局）分别只有 12 和 11 个 `≥90` 回合 ⇒ 只有加总那条能报 CI。
2. **标签是现算的**，不是采集时存的；`τ` 的口径有两种（回合 vs 帧），本文两种都报，方向一致但幅度差 3×。
3. **不是因果**：没有"同一局面花 vs 攒"的对照（这正是 §8 要补的）。
4. **混杂仍在**：`≥90` 的帧极可能集中在"被压着打/对面没进攻/卖塔退款后"这几类局面；控制了局、回合十分位、血差之后仍为正，但控制得再细也代替不了对照实验。
5. **对局来源混了 4 种配置**（`A/B`、`BOLTZ*`、`FIX`、`SPLIT`），加起来只为提高 N；单看不构成新结论。
6. 这些对局**都没有存 `board`**（progress trace 只有 HP/金币/操作），所以**不能直接当训练数据**——只能当"诊断"。要用它们训练，得回到 §8 的采集（那里才有 `board+stats`）。

---

## 14. 三句话（给主 agent 的摘要）

1. **现役标签是 terminal**：`value_target = clip((终局我方hp−敌方hp)/20, ±1)`、按 player 定符号
   （`az_selfplay.py:484-489`），**同一局同一玩家的所有决策点同值**（实测局内 std = 0.000000）⇒ 对"花 45 币 vs 攒着"零信号；
   损失是 MSE（`train_value_net.py:331-332`），评估只按"有塔/无塔"分组（`:166,179`）。
2. **rel/abs/kgeo 是"指数加权未来血量差"族的三种形态**（`abs`=未来加权均值、`rel`=`abs − 当前血差`、
   `kgeo`=`w_k=k·γ^k` 把当前帧权重置零以削掉"抄输入"捷径），**三者都是逐帧**；能给"金币×冷却"逐帧监督的**只有它们**，
   但只是**间接**（标签只由未来 HP 轨迹算、**没有任何标签显式含 cd**，帧内与 coins/cd 的偏相关仅 +0.09/+0.03）。
3. **从 terminal 切到逐帧，代码上只需一个 CLI 开关 + `--label-scale`**（网络/损失/数据加载都不动，评估要加金币/冷却分层，
   `rel` 才需要 `AZAI_REL2ABS`）；重训计划 =「P0 数据侧可识别性 → **换数据来源：当前 ckpt 自对弈 + 亚军 oracle 条件化注入采 2000 局**
   → 4 条标签臂（terminal 对照 / abs+kgeo / rel+kgeo / abs+geo-tau10）→ 用 P0-2（90 币折点）与 P0-3（`∂V/∂coins|cd=0 ÷ |cd≥0.6 ≥ 2`）验收」，
   bootstrap 的破法是**把动作质量外部化给 oracle**（E3 实测 +29.2pp），从而不需要先有一个会定价的值网。
4. **但现成的逐帧标签不含"90 币阈值"**（§12 实测，1.28M 帧 lowq）：`kgeo50 ~ coins` 的斜率 pooled ≈ **+0.0012/100 币（R²≈0）**、
   within 是 **−0.0154/100 币**、控 hp 后 **+0.0117/100 币**（符号都不稳）；分桶上 `≥90` 桶比 `60-89` 桶**低 9.2% 个标签 std**，
   扣掉回合趋势后只高 **+0.9%**，90 币处**没有台阶**（`1[coins≥90]` 系数 −0.0065，t=−9.2）。
5. **但我们自己的对局里根本不到 90 币**（§13，36 局 / 9,046 回合）：`coins≥90` 只占 **3.0%**（冒烟那局 3.35%，
   中位金币 **29**），而 lowq 是 48.2%、同一批对局里对手（亚军）是 **69.1%**；现算的 `kgeo50` 在 `≥90` 桶比 `60-89`
   **高 +0.088（≈0.56 std，按局 bootstrap 90%CI [+0.040,+0.120]）**——**方向是正的不是反的**（不是"乱花钱⇒标签更低"），
   但 275 个事件只够当提示，而且它更像"我们花钱净亏"的侧写。
   ⇒ **合起来的结论**：`kgeo` 标签**能承载**钱的价值，**缺的是"钱够买闪电"这个状态**；
   **"光换标签"不会长出折点、"只多采同分布数据"也只是把那 3% 再采一遍** ⇒ 必须先按 §8 换数据（注入采集制造"放 vs 攒"对照），
   再不行才加显式金币项（hinge at 90 / 续盘回归估汇率）。
