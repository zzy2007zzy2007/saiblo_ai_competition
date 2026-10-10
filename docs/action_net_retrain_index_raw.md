# 动作网（类网 / 位置网 / 策略网）重训事件总索引 — 只读扫描

仓库 `D:\2026智能体大赛_新2`。扫描日期：会话时间。**未修改任何文件、未跑训练/对局。**
证据来源（下文出处用短名）：`LOG` = `docs/experiment_log.md`（行号）；`REG` = `docs/goal_register.md`（行号）；
`CKPT` = `docs/checkpoints.md`；`DIR` = `training_history/` 目录与文件 mtime；`GIT` = `git log`；
`HANDOFF`/`MEM`/`MSG`/`CONV` = 根目录四份早期会话总结；`DOC` = `docs/*.md`。

**口径**：一行一事件；同一批实验合并时注明合并了哪些 RUN 名。
"结论"列只写语料里明确给出的定性（采纳/否决/不显著/未评估/中止/仅机制），不对结果做我的再解释。

---

## 1. 总索引表（按时间排序）

### 1.0 血统起点（不是重训事件，但所有事件都从它出发）

| 时间 | 事件 | 头 | 数据/监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| ~2026-06-29/30 | 官方 `ExampleAI` 行为蒸馏 → 初始化 ckpt（**全项目唯一一次"祖先蒸馏"，弱老师**） | 类+位置+价值（同源三头） | ExampleAI 对局（BC） | `code/distill/*` | `distill_data_v2/model_all_bal.pt` | 实战 3 个不同类、vs ExampleAI **50%** | 采纳（仅作初始化） | `DOC/蒸馏训练实验记录.md` §11；`REG:870` |

### 1.1 阶段 A：ES / 精英 BC / 多专家（2026-06-29 → 2026-07-07）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督信号 | 脚本 | 产物 ckpt | 评估读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-06-30 | 统一对手池 + HOLD 类 + BC 训练落地 | 策略网 | 对局轨迹 + 精英筛选 | `code/my_ai/es_train.py`、`code/bc/train_bc.py` | `bc_checkpoint.pt`、`bc_data_2026*.npz`（根目录） | 未见评估 | 采纳（管线） | `GIT` 5b2e924；`DIR` |
| 2026-07-02 | Elite Behavior Cloning（CEM 式） | 策略网 | top-K 对局数据 BC | `code/my_ai/elite_bc.py` | — | 见 `docs/elite_bc_implementation.md` | 采纳 | `GIT` e1741c1；`DOC` |
| 2026-07-02 | 新增 `ss_train.py`（策略空间进化） | 策略网（多头） | 标签突变 + BC 监督 | `code/my_ai/ss_train.py` | — | 设计见 `docs/strategy_space_trainer_design.md` | 采纳 | `GIT` 74c91ca |
| 2026-07-03→07-04 | 行为空间变异替代参数采样 + 头交换变异 | 策略网（多头） | BC（含 HOLD 降采样） | `code/my_ai/ss_train.py` | — | — | 采纳 | `GIT` 954d68c / ccf0654 |
| 2026-07-06 01:26→03:18 | ES → **BC-only** 过夜跑（pop16/games4，200 代） | 策略网 | 精英 BC，**无 ES 梯度**（`es_train.py:359`）；`--load-bc distill_data_v2/model_all_bal.pt` | `es_train.py` | `training_history/20260706_012607/`（**已不存在**） | BC loss 10.64→0.78，avg_fitness 卡在 0.031；"**BC 降损但不赢**" | 弱/否决（BC-only 不足） | `MEM:22-43,45-49` |
| 2026-07-06 ~12:21→20:57 | BC-only 第二跑（pop64/games16，跑到 gen 56 被停） | 策略网 | 同上，bc_k=3 | `es_train.py` | `training_history/20260706_122129/`（**已不存在**）；末 ckpt `gen_0057.pt` | gen27 **26.7%** / gen39 **63.3%** / gen51 **0.0%**（vs ExampleAI，30 局）⇒ **单动作坍缩（100% 闪电 17）** | **否决**（one-trick） | `MEM:51-131`；`CONV`（CP936 乱码，须 `-Encoding` 转码读） |
| 2026-07-06→07 | 多专家架构 + 门控网 + 混合对手 | 策略网（4 专家 + 门控） | ES+BC；`--allowed-classes` / `--opp-probs` | `code/my_ai/es_train.py`、`gating_network.py`、`mixed_opponent.py` | — | 13/13 验证通过 | 采纳（架构） | `GIT` 4ccb58d；`MSG:9-98`；`DOC/multi_expert_plan.md` |
| 2026-07-07 02:04 起 | **专家 A（Thunder + 基地升级）**训练 | 策略网（受限类集） | ES+BC，allowed=[17,21,22,23] | `es_train.py` | `training_history/expert_a_v2/gen_0010.pt` 等（**目录已不存在**） | 10/50 代，BC loss 9.4→3.7 | 未评估（会话中断） | `HANDOFF:11-49` |
| 2026-07-07 | 专家 B/C/D（塔攻防 / 辅助武器 / 产兵）| 策略网（受限类集） | ES+BC | `es_train.py` | `training_history/expert_b|c|d`（**均不存在**） | **无任何读数** | 未执行/未评估 | `HANDOFF:57-59`；`MSG:36-41` |
| 2026-07-09→07-11 | `ss_20260709_191050`（gen_0030） | 策略网 | action_dropout=0.1, lb_inject=3, ES+BC | `ss_train.py` | `training_history/ss_20260709_191050/gen_0030.pt` | vs ExampleAI **90.6%**(58/64) / dropout0.15 **95.3%**；**vs rule_v4 1.6%**(1/64) | 采纳（首个破 50%），但策略窄 | `DIR`；`DOC/notable_checkpoints.md:3-14` |
| 2026-07-10→07-11 | `ss_20260710_155327`（gen_0101） | 策略网 | oversample_alpha=0.6, dropout=0.1 | `ss_train.py` | `…/gen_0101.pt` | vs ExampleAI **67.2%**(43/64)；train eval best 0.667 | 采纳（纯塔防路线） | `DIR`；`DOC/notable_checkpoints.md:16-27` |

### 1.2 阶段 B：GA+SS 混合 / 分数回归蒸馏（2026-07-12 → 2026-07-22）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督信号 | 脚本 | 产物 ckpt | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-07-12 | GA+SS 混合框架（函数空间交叉） | 策略网 | GA 交叉 + BC | `ga_ss_train_draft.py` | — | — | 采纳（框架） | `GIT` f129583 |
| 2026-07-12 | 头交换变异 + 榜单注入 + 类别重平衡 | 策略网 | 无标签进化 | `ss_train.py` | — | — | 采纳 | `GIT` ccf0654 |
| 2026-07-14 | **分数回归蒸馏管线**（`ActionCatalog` 逐格评分作软目标） | 类+位置（仿策略头） | SDK 启发式评分（逐格回归） | `code/score_distill/{collect,train}.py` | `training_history/score_distill_20260714_191151/{epoch_0010..0050,final}.pt` | `docs/distill_rule_v4.md` 里 12%/35%/58%/62% 是 python 注释，**非实测** | 采纳（管线）；数字**存疑** | `GIT` b0ce50a；`LOG:1758-1770`（明确撤回） |
| 2026-07-14 | BN 折叠 / `--no-bn` 等价转换（影响所有后续 GA 训练） | —（推理/训练等价性） | — | `ss_train.py`/`ga_ss_train.py` | — | — | 采纳 | `GIT` 3aa5826 / ccc3cec / 782a8ec |
| 2026-07-14 | 软标签变异 + 蒸馏 BC + 梯度方向变异 | 策略网 | soft label（`head_logits`）| `ss_train.py` | — | — | 采纳 | `GIT` 7ae63b3 / dd0c951；`DOC/soft_label_mutation.md` |
| 2026-07-16→07-17 | **价值网重训线**（`train_value_net.py` 首版） | **价值** | 逐帧终局 HP 差（`ss_train` 写 npz 带 `[hp_us,hp_opp]`）| `code/my_ai/train_value_net.py`、`collect_value_*.py`、`merge_value_npz.py` | `training_history/value_20260711_222304` … `value_20260717_001750`（8 个运行目录） | 见 `docs/value_label_future_weighted.md` / `value_data_merge_and_train.md` | 采纳（价值线独立） | `DIR`；`DOC`；`GIT` 8754f48 |
| 2026-07-19→07-22 | `ga_ss_train.py` 通用化 + 自适应榜单采样 + 温度采样评估 | 策略网 | GA + BC + 榜单对手 | `ga_ss_train.py`、`leaderboard.py` | — | 见 `docs/adaptive_lb_sampling.md`、`temperature_sampling_eval.md` | 采纳 | `DIR`；`GIT` da652a7 / 5ff38bd |
| 2026-07-28→07-30 | dropout 只采单操作 bundle + mutation 预采样 | 策略网 | 无标签进化 | `ss_train.py`/`ga_ss_train.py` | — | — | 采纳（工程） | `GIT` a46285f / 8c62e35 |
| 2026-07-29→08-02 | `ga_ss_20260730_093908`：**GA+SS 长跑（权威血统起点）** | 策略网（三头同源） | GA + 精英 BC | `code/my_ai/ga_ss_train.py` | `training_history/ga_ss_20260730_093908/gen_0001..gen_0120.pt`（~140 个，每个 ~404MB）→ 抽出 `checkpoints/gen_0120_model.pt` | 未见本扫描内的胜率读数；被后续所有 az 线当 hotstart | **采纳（血统）** | `CKPT:25,74-104`；`LOG:37` |
| 2026-07-29 | **意图解码（intent decoding）**：闪电自动降塔筹钱 | 类→位置（解码器规则） | 规则 | `decoder.py` | — | — | 采纳（默认开） | `GIT` e910d37 / 1348205 |

### 1.3 阶段 C：PPO 线（2026-08-02 → 2026-08-03，失败）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-08-02 | PPO 冒烟：`ppo_smoke_test` | 策略网（on-policy） | PPO(z-score logits + 无界价值头 + GAE) | `code/my_ai/ppo_train.py` | `training_history/ppo_smoke_test/ppo_iter_0000..0004.pt` | — | 冒烟通过 | `DIR`；`GIT` 4554ea3 / 5cae986 |
| 2026-08-02 | 自适应榜单对手池进 PPO | 策略网 | PPO + LB challenge | `ppo_train.py` | — | — | 采纳（工程） | `GIT` c274dd2 / 9b3622a |
| 2026-08-03 | 位置通道进策略优化：aux `-logp` → trust-region MSE | 策略网（位置） | PPO + 位置 MSE 信赖域 | `ppo_train.py` | — | 记录为**负结果** | 部分否决 → 改 MSE 后合入 | `GIT` 87c4b3b / 21d910c / 4640a0b / 68ac699；`DOC/ppo_position_*.md` |
| 2026-08-04 | PPO 线**判死刑** | 策略网 | — | `ppo_train.py` | — | 胜率退化 | **否决** | `DOC/cpp_engine_rewrite_research.md:46`（"PPO 是失败实验线，已判死刑"） |

### 1.4 阶段 D：az_intent AlphaZero 自对弈（2026-08-04 → 2026-09-07）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04 | `az_official_val`（官方数据验证自对弈管线） | 类+位置+价值（同网） | 官方 self-play 数据 | `az_intent/az_train.py` | `training_history/az_official_val/{run.log,az_val.npz}` | — | 采纳（对齐） | `DIR`；`GIT` 0d1f167 |
| 2026-08-04 | `az_full`：捆绑 MCTS AZ 训练冒烟 | 同上 | 自对弈 visit 分布 | `az_train.py` | `az_full/az_full.npz` + `training_history/az_full/20260804-135752/train.log`(518KB) | `eval_batch2_20g.log` / `eval_batch5_20g.log` | 采纳 | `DIR` |
| 2026-08-05 | `az_intent`：意图解码 AZ + 大量 eval（约 40 个 eval 日志） | 同上 | 自对弈 + `--value-only` | `az_train.py`、`selfplay.py`、`eval.py` | `az_intent/az_az0..7.pt`、`az_intent_batch2..12.pt` | `eval_*` 系列日志（`DIR`） | 采纳 | `DIR`；`DOC/alphazero_intent_mcts_plan.md` |
| 2026-08-06→08-07 | 第 1–3 轮 AZ 训练改造：双数据池 → **策略/价值拆分成两个独立网** | 策略网 + 价值网（拆分） | 加权未来 HP 差标签（**有符号**）| `az_train.py --split`、`run_az_batches.sh` | — | "从有害到打平" | 采纳 | `GIT` a24bb9c / 2f20ef2 / e7f3416；`DOC/az_weighted_value_retraining_plan.md`、`az_split_policy_value_plan.md` |
| 2026-08-07 | 迭代拆分训练**突破 66.7%** | 策略+价值 | 拆分 + label-scale 6 | `run_az_batches.sh` | — | 66.7% | 采纳 | `GIT` 4646629 |
| 2026-08-07→08-09 | `az_cpp`：C++ 引擎 + 批次循环（约 15 轮） | 策略+价值 | 自对弈，`az_train.py` GPU | `run_az_batches.sh`、`az_selfplay.py` | `az_cpp/az_r101..`、`az_mix_r50p_r62v.pt` 等（多组 p/v 混合） | `az_r50`（21 batch）**68.8%** | 采纳 | `DIR`；`GIT` 6e62e57 |
| 2026-08-11 | τ 20→50 + `--value-only`；**价值头训练步数解耦**（`--value-passes`） | 价值（主）/策略 | 加权标签 τ=50 | `az_train.py`、`run_az_batches.sh` | — | az_r69 弱因**价值欠训**：25%→**46.9%** | 采纳 | `GIT` 3cade21 / e822222；`DOC/az_value_step_decouple_plan.md` |
| 2026-08-11 21:06 | **tau50 + label-scale 6 配方实测 vs rule_v4 43–47%**；rule_v4 突破 **31.2%** | 策略+价值 | 同上 | `az_train.py` | — | 43–47%（另一处记 31.2%） | 采纳 | `DOC/az_collapsed_vs_healthy_paradox.md:5,34`；`GIT` fce8603 |
| 2026-08-12 | 策略损失显示被 value passes 稀释 33× → 修 | 策略（显示口径） | — | `az_train.py` | — | — | 采纳（bug 修） | `GIT` 453cb57；`DOC/az_policy_loss_scale_fix_plan.md` |
| 2026-08-12 | **2×2 策略-价值交叉消融**（强耦合、迭代未提升上限） | 策略×价值 | 4 臂 | `az_train.py` | — | — | 不显著 | `GIT` 15bdcc8 |
| 2026-08-13 | `--pos-single` 位置目标塌到 argmax；MCTS 搜索坍缩到闪电（99.3%） | 位置+类 | visit 分布 | `az_train.py` | — | 定位为"坍缩" | 否决（该做法） | `GIT` 74d04cf / b916877 / 58a85f4；`DOC/az_search_diagnosis.md` |
| 2026-08-14 | `--label-mode rel/abs/mix` 价值标签语义 | **价值** | rel/abs/mix + τ | `az_train.py` | — | "建完又拆"根因 | 采纳（语义修正） | `GIT` f77d1a0 / 59d478f；`DOC/az_value_step_decouple_plan.md` |
| 2026-08-18 | **最终根因：az 训练把特征提取器训坍缩，棋盘信息全丢** | 全部（骨干） | — | — | — | 解释 8 月全部异常 | **否决（旧 az 配方）** | `GIT` eeebd19 |

### 1.5 阶段 E：三网拆分 / 冻结骨干 / 位置轴（2026-09-07 → 2026-09-21）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-09-08 00:06→11:15 | `az_fixed` 批次 1：`az_r1..az_r10`（BN 迁移 + 特征坍缩修复） | 策略+价值 | 自对弈逐批 | `run_az_batches.sh`、`az_train.py` | `az_fixed/az_r1..r10.pt`、`gen0120_bn_init.pt` | 见 `DOC/az_fix_feature_collapse_bn_migration.md` | 采纳 | `DIR` |
| 2026-09-08 13:11→15:08 | **只重训价值头（3 版）**：`az_r10_v_ls1` / `_v_terminal` / `_v_term_anchor` | **价值** | 标签平滑 / terminal / terminal+anchor | `az_train.py --value-only` | `az_fixed/az_r10_v_*.pt` | 输出 std 0.039 / 0.018（**价值饿死**） | 否决 | `LOG:246-266` |
| 2026-09-08 17:14→22:26 | `v_warm` / `mix_r10p_*`：价值 warmup + 策略/价值交叉嫁接 | 策略↔价值 | 混合 | `make_mix_checkpoint.py`、`value_warmup.py` | `az_fixed/mix_r10p_*.pt`、`gen0120_*_warm*.pt` | — | 采纳（工具） | `DIR`；`GIT` d0ece40 |
| 2026-09-09 17:07→19:35 | `vw_wcpp2_gpu` / `vw_cpu_faithful` / `vw_gpu_no_tf32` / `vw_cpu_8threads`（**合并**） | **价值**（+骨干） | warm_data_cpp 200 局 | `value_warmup.py` | `az_fixed/gen0120_warm_cpp*.pt` | GPU 同一配置**逐位相同**；CPU 与历史好头 `w_corr 1.00000`；GPU+noTF32 corr 0.7156；**CPU 8 线程 vs 32 线程骨干 corr 0.568** | 采纳（"训练对计算路径敏感"） | `LOG:99-178` |
| 2026-09-09 19:56→21:47 | **冻结骨干消融**：`vw_frozen_{cpu,gpu}_bb` / `vw_azdata_frozen_*` / `az_frozen_*` / `az_frozen_terminal_gpu`（**合并 6 臂**） | **价值头**（骨干冻结） | gen 数据（200 局）vs az 数据（160 局） | `value_warmup.py` / `az_train.py` | `az_fixed/az_frozen_*.pt` | 冻结 CPU 骨干 **25.0%**、冻结 GPU 骨干 **34.4%**（=历史最好）；az 数据 **3.1%**（1/32） | **gen 数据有效、az 数据无效** | `LOG:202-379` |
| 2026-09-09 23:08→23:55 | `collect_p1_batch1` + `train_p1_batch1` | 策略（仅策略） | 自对弈，**策略-only** | `az_train.py`（价值冻结） | `az_fixed/az_p1.pt` 起 | policy loss 0.5103→0.3711，**value loss 恒 0** | 采纳 | `LOG:381-401` |
| 2026-09-10→09-11 | `polonly_resume` + `polonly_to21_rand002`（策略-only 循环到 p21） | 策略 | 自对弈 + **2% 随机动作注入** | `az_train.py --policy-only`、`resume_polonly.sh` | `az_fixed/az_p2..az_p21.pt` | `eval_azp6_vs_rulev4` / `eval_azp21_vs_rulev4` / `svs_azp6_vs_init` / `svs_azp21_vs_init` / `eval_azp6_rand005` / `eval_azp6_rand002` | 采纳（策略-only 到顶） | `LOG:403-579`；`DIR` |
| 2026-09-11 09:51→12:14 | `train3_vheads_newdata` / `train2_tau_newdata` / `c2_aztrain_nonbn`（**合并 3 训练 + 3 评估**） | **价值头**（3 臂 + τ/BN 臂） | 新数据；A(vw+冻结)/B(vw 不冻结)/C(az_train terminal,BN init)；D/E τ scale2/6 | `az_train.py`/`value_warmup.py` | `az_fixed/az_combined_nonbn.pt` 等 | A 0.0526 / B 0.0481；eval：**53.1%**（34W/30L）⇒ 自对弈训的头与 gen 原版头**等价** | 不显著（等价） | `LOG:579-643` |
| 2026-09-11 16:49→19:36 | `combined_data_2arms` / `gn_2arms` / `tau_nonbn_2arms` / `tau_nonbn_s4`（**合并 4 臂**） | **价值头** | 合并 536 局（gen200+自对弈336）；GroupNorm 转换 | `az_train.py`、`convert_no_bn_to_gn.py` | `mix_r10p_*_nonbn.pt` | F 从零+冻结完成；**H(GN+解冻骨干) 0.0478 / I(GN+冻结) 0.0575**；J/K/L 见下 | 采纳（解冻更好，但见下） | `LOG:643-697` |
| 2026-09-11 19:12→21:32 | `tau4_L`/`eval_tau4_L`/`eval_ijk`（**合并**） | **价值头** | τ-abs scale2/1/4 | `az_train.py` | `mix_r10p_tau{1,2,4}_nonbn.pt` | **tau4=12.5%、tau2=21.9%、tau1=9.4%**（vs rule_v4，32 局） | **否决**（全部 ≤ 对照） | `LOG:691-733` |
| 2026-09-11 21:53→23:45 | `pool_arms_5*` / `pool64`（**合并**） | **价值头** | 空间池化 5 臂 | `az_train.py` | `vw_pool_{gapmask,gapmax,region,grid,attn}.pt` | grid 末 epoch loss=0.0614 | 不显著（次日判定） | `LOG:733-766`；`DOC/az_value_head_spatial_pooling_plan.md` |
| 2026-09-12 01:55 | `validate_gapmax` | **价值头** | 336 局、冻结骨干、no_bn、terminal 标签 | `az_train.py` | — | 策略**全部逐位相同**（= az_r10） | 不显著 | `LOG:812-935` |
| 2026-09-12 09:45→17:54 | `pooltau_6arms` / `pooltau100_6arms`（**合并**） | **价值头** | 池化 × τ 6 臂 | `az_train.py` | — | `LOG:935` 起 | 未评估/不显著（同轴） | `DIR`；`DOC/az_pool_tau_label_plan.md` |
| 2026-09-12 17:55→21:41 | `poolweight_kgeo_rel` + `value_stats_only` | **价值头** | kgeo 相对标签 / 仅统计量输入 | `az_train.py` | — | `DOC/az_value_stats_only_plan.md` | 不显著 | `DIR` |
| 2026-09-12 | **价值头空间池化消融：无效（架构杠杆试完）**；**价值标签"两个 bug"复核推翻 §22** | 价值 | — | — | — | `stats[24]/[25]` 是我方/敌方 HP/50；`53ffca6` 是等价变换 | **否决**（该轴关闭） | `LOG:32-70` |
| 2026-09-13 00:05→09:23 | `gen120rep_kgeo` / `refeval_bn0v_64` / `refeval_r69_history` / `svs_kgeo_flip_paired`（**合并 4**） | 策略+价值（重测既有 ckpt） | — | 评估/重训 | `gen_0120.pt` 系列 | — | 参考读数 | `DIR` |
| 2026-09-13 10:08→20:28 | `pv_2x2` / `v50pool_test` / `v50pool_npz` / `v50label_2x2` / `polpool_kgeo_arm`（**合并 5**） | 价值（4 臂）+策略（1 臂） | v50 池化 / 标签 | `az_train.py` | — | `DOC/az_pool_tau_label_plan.md` | 不显著 | `DIR` |
| 2026-09-13 22:13→09-14 04:31 | `v50pool336_kgeo` / `seed_rep_kgeo` / `noise_floor` / `noise_floor_extra`（**合并 4**） | 价值 | 336 局池化 + 噪声地板 | `az_train.py` | — | 噪声地板（判据分辨力标定） | 采纳（方法论） | `DIR`；`_tmp_noise_floor.sh` |
| 2026-09-14 12:46→19:27 | **`value_loop`：价值头迭代循环首跑** | **价值** | 迭代自举 | `run_value_loop.sh`、`az_train.py` | `az_loop/vw_round1..3.pt`、`cur_bn.pt` | **堆量反而削弱价值头**（配置不一致假说）；`svs_loop_degrade` 确认为真 | **否决** | `LOG:1246` 起；`DOC/az_iterative_value_loop_plan.md`；`DIR` |
| 2026-09-15 00:21→08:22 | `az_bn_kgeo`（5 batch） | 策略+价值 | 原始批循环 + BN + kgeo + 2% 注入 | `run_az_bn_kgeo.sh` | `az_bn_kgeo/az_b1..b5.pt` | — | 采纳（配方） | `DIR` |
| 2026-09-15 | 机制诊断：价值头"用了多少棋盘信息"（stats-probe R²）| **价值** | — | `_tmp_stats_probe.py` | — | **首次出现真正用棋盘的头** | 采纳（诊断） | `LOG` 2026-09-15 段 |
| 2026-09-15 09:47→09-16 08:47 | `kgeorel_strength` / `search_axis_ablation` / `search_speed_bench` / `forced_move_skip_bench` / `collection_cost_bench` / `tclass_preflight` | 价值（kgeo 强度）+ 配置轴 | 搜索轴消融（非重训） | `_tmp_*.sh` | — | `DOC/az_action_factorization_idea.md` | 不显著 | `DIR` |
| **2026-09-16 08:15** | **三网拆分落地产物**：`make_three_net_ckpt.py` 从 `mix_r10p_vw_pol_frozen.pt` 复制"三网同源" | 类/位置/价值（三网） | — | `az_intent/make_three_net_ckpt.py` | `training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt` → `checkpoints/` | 子网指纹：A `class=777fbd73 pos=1cae1a6d value=2b6b00a3` | **采纳（三网从此分立）** | `CKPT:19,32-36`；`DOC/az_three_net_split_plan.md` |
| 2026-09-16 08:47→09:51 | `phase2_posnet` / `phase2_posnet_b`（**合并**） | **位置**（类/价值冻结） | pos-only + 位置 CE + 位置 anchor | `az_train.py --pos-only-net`、`_tmp_phase2_posnet*.sh` | `posnet_p2/pos_l30.pt`、`pos_l150.pt`；`_smoke_posnet/pos_{c2,contrib,trained}.pt` | 见 `DIR` | 采纳（三网拆分第一期） | `DIR`；`LOG:1806-1827` |
| 2026-09-16 10:55→15:38 | **`posnet_loop`：位置网循环** | **位置** | 自对弈 visit 分布（位置先验）| `code/run_posnet_loop.sh` | `posnet_loop/pos_r1..r5.pt` | `posnet_loop_svs` 评估 | 采纳（位置轴开线） | `LOG:1828-1847`；`DIR` |
| 2026-09-16 20:07 | `posnet_convergence` | **位置** | pos-only，30 epoch × 2 轮 | `_tmp_posnet_convergence.sh` | `_posnet_conv/{e8,e30,e80,e30lr3}.pt` | **pos_ce 7.117→6.290（均匀基线 ln361=5.894）⇒ 位置网基本没学到**；argmax 一致率 raw 0.63% / e8 15.62% / e30 28.11% / e80 28.04% | **不显著/否决（该配方）** | `LOG:1866-1876`；`DOC/report_posnet_diag.py` |
| 2026-09-17 | `depth0_vs_raw` / `pos_greedy_vs_raw` / `search_consistency_*` / `map_quality_effect` / `zscore_*` / `tpos_*` | 位置（诊断） | — | `_tmp_*.py/.sh` | — | `tpos_rule_v4` / `tpos_rule_v4_mild` 两臂 | 不显著 | `DIR` |
| **2026-09-18 09:05** | **`distill_prior_rule_v4`：把 SDK 启发式蒸馏版位置图当先验** | **位置**（换 `pos_state`） | `distill_data/model.pt`(V1)、`distill_data_v2/model.pt`(V2) | `_tmp_distill_prior_rule_v4.sh`、`_tmp_make_swapped_pos_ckpt.py` | 换 `pos_state` 的 ckpt（94/94 键对齐） | t_pos=1.0：我们 **20.3%**(13W/51L) vs V1 **9.4%** vs V2 **1.6%**；McNemar 我们 vs V2 **翻转 12:0, p=0.0005** | **否决**（蒸馏位置图更差） | `LOG:2250-2276` |
| 2026-09-18 13:50→15:26 | `value_prior_ab` / `value_prior_rule_v4` / `value_prior_verify`（**合并**） | **价值** | 价值先验（A/B） | `_tmp_value_prior_*.sh` | — | `DOC/az_value_prior_plan.md` | 见下 | `DIR` |
| 2026-09-18 18:34 | **`vprior_train_r1`：位置先验第一次可复现提升** | **位置**（类/价值冻结） | `vp_400.npz` + `--tau 1.0 --t-pos 1.0`，60 epoch | `code/my_ai/az_intent/train_value_prior.py` | `training_history/vprior/posnet_r1.pt` → `checkpoints/posnet_r1.pt` | **打 rule_v4 21.1% → 35.2%**（=复现 oracle 收益的 86%）；样本外 seed128..255 训前 15.6% → 训后 **49.2%** | **采纳（位置轴最强信号）** | `LOG:2458-2563`；`CKPT:28` |
| 2026-09-18 19:32→23:46 | `vprior_baseline_fresh` / `tpos_curve_priors` / `vprior_tau_probe` / `vprior_tau025_tp2`（**合并 4**） | 位置 | τ=0.25/0.5 | `train_value_prior.py` | `vprior/posnet_tau0.25.pt`、`posnet_tau0.5.pt` | `posnet_tau0.25` vs rule_v4：36→17.2%、85→**42.2%**、176→32.8% | 采纳（τ 与 t_pos 同轴） | `LOG:2613-2651`；`DIR` |
| 2026-09-19 10:31→12:00 | `vprior_round2` → `posnet_r2.pt` | **位置** | `vp_r2.npz`（第二轮 400 局） | `train_value_prior.py` | `vprior/posnet_r2.pt` | 与 r1 同价值网、pos_state 不同 | 采纳 | `DIR`；`CKPT:112` |
| 2026-09-19 14:47→15:25 | `cand_pick` / `cand_pick_first` / `cand_pick_k96` / `cand_pick_hl_save`（**合并 4**） | 位置/类（候选选择诊断） | — | `_tmp_cand_pick.sh` | — | — | 诊断 | `DIR` |
| 2026-09-19 17:46→18:54 | **`inject_ex02_400g`：`--inject-example-prob` 注入钩子 + 类轴第一步** | **类**（注入教师动作） | ExampleAI 动作注入 | `_tmp_inject_data_stats.py` | `inject_ex02/valnet_{A,A2,B,C}.pt` | `LOG` 2026-09-19 段；`valnet_A2` 被 `collect_value_prior.py` 用来算 adv | 采纳（类轴开线） | `LOG` 2026-09-19 段；`CKPT:20` |
| 2026-09-19 19:00→19:59 | `valnet_train_arms` / `valnet_train_clean` / `valnet_criterion`（**合并**） | **价值**（valnet） | 干净数据 / 判据 | `train_value_net.py` | `inject_ex02/valnet_*.pt` | — | 采纳 | `DIR` |
| 2026-09-19 19:54→23:35 | `k24_menu_guard` / `menu_lightning_arms` / `random_menu24_arms` / `search_vs_first_arms` / `search256_arms`(中止) | 配置/菜单（非重训） | — | `heuristic_candidates_match.py` | — | `search256_arms` **中止**（非结果） | 不显著 | `DIR` |
| 2026-09-20 00:21→08:01 | `overnight_rv4_then_256` | 配置（过夜） | — | — | — | — | 未评估 | `DIR` |
| 2026-09-20 19:17→22:48 | `vp_arms_km1_multi`（采集）→ **`vp_train_three_arms`** | **位置**（3 臂单变量） | `vp_B_km1.npz` / `vp_A1_km1_m32.npz` / `vp_A_k5_m32.npz` | `train_value_prior.py`、`collect_value_prior.py` | `vprior/posnet_{B_km1, A1_km1_m32, A_k5_m32}.pt`（→`checkpoints/`） | 最佳 val CE：B 5.1542@ep60（**超出均匀 0.448 nats**，top1 37.6%）/ A1 3.0601@ep48（0.406，51.6%）/ A 3.0414@ep60（**0.424，top1 75.5%**，其他 19 通道 hit 80–82%） | **采纳（A 为主臂）** | `LOG:3647-3664`；`CKPT:18,26-27` |
| 2026-09-20 22:49→09-21 00:38 | `vp_net_criteria` | 位置 | 判据 | — | — | — | 未评估 | `DIR` |
| 2026-09-21 07:53 / 18:53 | `svs_joint_ban17_A_vs_A1` / `svs_valueprior_vs_A1map` | 位置（对照） | — | — | A / A1 / B 三臂 | 三臂单变量对照 | 不显著 | `DIR`；`checkpoints.md:26-27`（"当年判据待跑"） |

### 1.6 阶段 F：身份/协议冻结（2026-09-22 → 2026-09-25，无动作网重训）

| 时间 | 事件 | 头 | 说明 | 出处 |
|---|---|---|---|---|
| 2026-09-22→09-23 | `champ_vs_runnerup_*` / `DET*` / `ladder_*` / `B0/B1/B2` | — | **只做对手确定性与天梯评估，无任何训练**（`cpp_ai_matches/ai0.log` 等均为 0 字节） | `DIR`；`DOC/champ_ladder_plan.md`、`protocol_ai_spec.md` |
| 2026-09-24→09-25 | `ab_supervisor_plan.md` / `git_remote_tsinghua.md` | — | 计划 + 远端迁移；A/B 两体架构**暂停** | `DOC`；`REG:23-24` |

### 1.7 阶段 G：value 头阶梯 + 类轴/位置轴/蒸馏法（2026-10-05 → 2026-10-07）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-05 05:47 | `C1_a1_gate` / `C1_a1_hist32` / `C1_a1_128` / `C1_a2_gate` / `C1_a2_screen7_14` | —（基线与配置） | — | `_tmp_c1_arm.ps1` | — | 钉死协议基线 `p̂ = 0.0391`；A1（k24/sm15/pos-only/skip1）**0.5312** | 采纳（A1 成操作配置） | `REG:37,56-57`；`LOG:4528-4610` |
| 2026-10-05 06:55→07:41 | **`C2_class_prior_train` → `posnet_C_class.pt`（类头条件蒸馏）** | **类** | `vp_A_k5_m32.npz`，条件 CE 到搜索/访问分布，30 epoch/309.8s | `code/my_ai/az_intent/train_class_prior.py` | `vprior/posnet_C_class.pt` → `checkpoints/`；`C2_class32` 评估 | 训练：条件 CE 1.3448→1.2916、**`class_state` 42/94 张量变了**、被评估类内 argmax 49.75% 变化；**部署 `p̂ = 0.5000`（=对照），32 局日志与对照逐字节相同** | **否决**（改动进不了部署路径） | `LOG:4611-4620`；`REG:885` |
| 2026-10-05 08:21→09:46 | `D1_playable16` / `D1_playable32` | —（零训练，只改推理 `pos_pin=playable`） | — | — | — | **`p̂ = 0.0000`**（16 局全败）；建塔 151.9/局、闪电 1.9 | **否决**（人工门控） | `REG:886` |
| 2026-10-05 09:50→10:57 | `V1_val20_train` / `V2_collect100` / `V2_valtrain`（**合并，价值头阶梯 V**） | **价值** | 价值标签（20/100 局） | `train_value_net.py` | `vprior/posnet_V1_val20.pt`、`posnet_v2_valfresh.pt` | **V2 早停在 epoch -1**（最佳 val MSE 0.03430） | **否决/关闭** | `LOG:4713-4762`；`REG:62-63` |
| 2026-10-05 11:01→12:28 | `P1_value32` / `P2_uniform32` / `S1_tpos03_32` / `S2_iters1024_32` / `S3_depth8_32|128` | —（先验/搜索配置轴） | — | `_tmp_ladder.py` | — | P1/P2 位置先验 0.41/0.31；S3 depth8 验收 0.5625（+3.1pp, CI 重叠） | 不显著 | `REG:58-61` |
| 2026-10-05 12:41→14:04 | `PIN002_collect100` / `Q1_cache_build` / **`Q1_valtrain`** | **价值** | 判据分布数据（PIN002） | `train_value_net.py` | `vprior/posnet_Q1_valpin.pt` | **"按构造为 null"：最佳 val MSE 0.09964 @epoch -1** | **否决** | `LOG:4919-4928`；`REG:62` |
| 2026-10-05 14:09→14:37 | `Q1b_lr1e4` / `Q1b_lr3e5` / `Q1b_lr1e4_bn` | **价值** | 学习率扫描 | `train_value_net.py` | `posnet_Q1b_lr*.pt` | 最好只能把 val MSE 改善 **0.3%**，一个 epoch 就反弹 | **否决（贴天花板）** | `LOG:4984`；`REG:62-63` |
| 2026-10-05 14:17 | `S5_k48_32` | —（配置） | — | — | — | K48 +2.3pp | 不显著 | `REG` |
| 2026-10-05 15:2x | **候选 R：用对局结果训类头（AWR / 结果加权 CE）最小先行版** | **类** | 按头类探索 p=0.15 的 `rule_v4` 对局 → 16 局 5,488 条，AWR 只训 `class_state` 12 epoch | `az_selfplay.py` + AWR 训练 | `vprior/posnet_R_class16.pt` | **loss 5.49→4.95（能降），但跨局面 logit std 只从 0.024 …** ⇒ 机制不成立 | **否决** | `REG:875`；`DIR` |
| 2026-10-05 15:20→16:37 | `E_reserve90_32` / `E_reserve180_32` / `E_reserve180_128`（**合并**） | —（手写启发式 `pos_pin=reserve`，零训练） | — | `_tmp_e3_run.ps1` | — | E2 **0.6641** vs 关规则 A1 **0.5312** ⇒ 增益随规则消失 | **诊断臂（不算方法侧）** | `REG:70-73` |
| 2026-10-05 16:40→18:53 | `R120_s71_102` / `R180_s71_102` / `R240_s71_102` / `FRESH_A1_128` / `FRESH_R180_128` | —（新鲜 seed 重测） | — | `_tmp_fresh_ab.ps1` | — | — | 参考 | `DIR` |
| 2026-10-05 18:54→19:18 | `M1_collect60_reserve` → **`M1_distill_train`** | **类**（把储备规则蒸馏进类头） | 60 局自对弈（`data_sp_reserve60_cls`），**纯蒸馏 CE（终局/结果标签）**，20 epoch/331s，`train_class_outcome.py` | `code/my_ai/az_intent/train_class_outcome.py` | `vprior/posnet_M1_distill.pt` → `checkpoints/` | 训练：val_loss 0.49（仍在降）、**类头跨局面 logit std 0.024→0.178（7.4×）**；部署：`p̂ = 0.5000`（=A1），建塔 0 / 闪电 11.2。根因：训练数据被选类 **99.3% 是闪电 17**（花钱 head 决策仅 0.74%） | **训练采纳 / 部署否决** | `LOG:5210-5240`；`CKPT:23`；`REG:848-849` |
| 2026-10-05 19:58→20:47 | `M1b_train`+`M1b_masked_32` / `M1c_train`+`M1c_masked_32` / `M1d_train`+`M1d_masked_16`（**合并 3 训练 + 3 部署臂**） | **类** | executed 标签 + `masked` + 反频率权重 power=0.5/1.0 | `train_class_outcome.py` | `posnet_M1b_masked.pt` / `M1c.pt` / `M1d.pt` | M1b **`p̂=0.0000`**（16 对全 0-2，**是我实现 bug**：类权重被 `cw.sum(dim=1)` 约掉 ⇒ 学成 HOLD）；M1c **0.5000**（logit std→0.206，但 argmax 仍只选闪电/HOLD）；M1d **0.3750**（建塔 26.3/降级 13.3/闪电 8.33，过度花钱） | **否决**（"把稀有分支 BC 进 argmax"走不通） | `LOG:5253-5340`；`REG:847-848` |
| 2026-10-05 21:05→21:22 | **`M2_qhead_train` → `qhead_M2.pt`（Q 头 / 类条件价值）** | **类**（Q 型目标） | 冻结主干 `state_emb` + 类 one-hot → 64 → 1；173,514 个 (局面,类) 对，MSE 0.084→0.074 | `code/my_ai/az_intent/train_q_head.py` | `vprior/qhead_M2.pt`（41KB）→ `checkpoints/`；`M2_q_16` 部署臂 | **P-M2a 弱通过**（花钱样本 Q(选中类)−Q(HOLD) 均值 +0.025、85% 为正）；但 `pos_pin=q` 部署**成本闸门超标**（`M2_q_16b` 中止）；机制指向 D 型 churn | **训练成功 / 部署中止** | `LOG:5340-5400`；`CKPT:24`；`REG:88,846` |
| 2026-10-05 21:41→23:23 | `M3_collect100_exp` / `M3_ingest` / `M3_qhead_train` / `M3b_collect60_exp50` / `M3b_ingest` / `M3b_qhead_train`（**合并**） | **类**（Q 头） | 高比例类探索（15%/50%）治识别性 | `train_q_head.py`、`ingest_selfplay_classes.py` | `qhead_M3.pt` / `qhead_M3b.pt` | `_m3_ident.txt` / `_m3b_ident.txt` / `_m3b_probe.txt` 存在 | 不显著（"差一点"） | `LOG:5408-5488`；`REG:614` |
| 2026-10-05 23:44→10-06 02:14 | `M5_menu3_8` / `M5_rel5_4` / `M5_k50_4` | —（把类交回搜索，零训练） | — | `menu3_candidates.py` | — | — | 诊断 | `DIR` |
| 2026-10-06 00:27→03:00 | **价值标签视野阶梯 V**：`VT_rel*` / `VT_k*` / `VR*` / `VK*`（**合并 15 个 run**） | **价值** | `--label-mode rel/abs/mix` × τ × kgeo | `train_value_net.py` | `vprior/posnet_VRrel{5,10,30}.pt`、`Vrel{50,100,200}.pt`、`Vk{35,50,70,100}.pt` | 32 局最强信号 = **rel+kgeo tau50（+9.4pp）**；**但 128 局验收 `VK50b p̂ = 0.4609` vs A1 0.5312（−7.0pp）**；A1 对照 0.5312 | **快读数乐观、验收翻车（第 3 次）→ 否决** | `LOG:5659-5793`；`REG:78-82` |
| 2026-10-06 01:2x | ⚠️ 修正：`rel` 标签头部署必须**还原绝对价值**（`AZAI_VALUE_TANH=0 + AZAI_REL2VAL=1.0`） | **价值** | — | — | — | 旧快筛读数作废 | 修正 | `LOG:5720` |
| 2026-10-06 03:45→13:47 | `M6_mc_128` / `M6_retry8` / `M6b_check8*` / `M6b_mcq_128` / `M6bND_128` / `M6b_re128` / `A1_head_128`(×2) / `M6c_OBS8` / `M6cM4_128` / `M6cROOT_128` / `M6cROOT_fresh128` / `A1_fresh128` / `M6b_fresh128`（**合并 14 个 run**） | —（MC 类决策机制 + 对照，**零训练**） | — | `mc_class.py`、`mc_quiet.py`、`bundle_mcts.py` | — | M6b **0.6641**（+13.3pp）；M6bND 0.4297；**M6c-ROOT 0.7344（首过 0.70）**；样本外 0.6875、合并 0.7109 | **采纳（机制）**；ROOT 相对 M6b +4.3pp 不显著 | `REG:10-11`；`DOC/milestone_20261006_70.md` |
| 2026-10-06 17:08→19:2x | `ENG_mc_light_s11` / `ENG_mem8` / `SMOKE_diag3*` / `SMOKE_archive` | —（引擎/超时诊断） | — | — | — | 定位"原生崩溃"= bridge 300s 读超时 | 采纳（诊断） | `DIR`；`DOC/engine_crash_probe.md` |
| 2026-10-06 21:58→10-07 00:26 | `M6cROOT_rand128` / `R1_bidir8_64` / `R2_bidir32_64` | —（机制臂） | `AZAI_MC_BIDIR=K` | `bundle_mcts.py` | — | 随机段基线 **0.7031** | 不显著 | `REG:14-16`；`DIR` |
| 2026-10-07 第 5 轮 | **闭环重训的两条前置检查** | — | — | — | — | 见下两行 | — | `LOG:6991-7077` |
| 2026-10-07 第 5 轮 | **价值头在新分布上重训（pilot）** | **价值** | 新分布自对弈 | `train_value_net.py` | — | 同一 20 局留出 split：val MSE **0.10145→0.09763（−3.8%）**、r 0.576→0.597（确实更准）；**部署 `p̂ = 0.6406`（−1.6pp，n.s.）** | **不显著**（离线更准≠棋力） | `LOG:7065`；`REG:723-724` |
| 2026-10-07 第 5 轮 | **类头蒸馏：识别性失败 ⇒ 关闭** | **类** | 部署分布自对弈数据 | `az_train.py` 蒸馏 | — | loss **7.46→1.88**（学到边际分布），但 >**23.6% 完全平局** ⇒ 无信号 | **否决（关闭）** | `REG:728-730` |
| 2026-10-07 01:38→03:32 | `G_margin01_64` / `F_upg_64` / `D_nodown_64` | —（MC 机制臂，零训练） | — | — | — | 平局率是三条阴性的根因；升级轴查明是 op 编码问题 | 采纳（诊断） | `DIR`；`LOG` 2026-10-07 第 2/3 轮 |
| 2026-10-07 03:32→09:45 | `loop_r4_collect300` / `E2_density_64` / `V_r5val_64` / `L180_reserve_64` / `B90_buildgate_64` / `B60_buildgate_64` / `CF_gate_S1|S2` / `loop_scale_collect800`（**合并 9 个 run**） | **价值**（闭环重训，R5） | 采集 300/800 局 → 价值重训 | `_tmp_*.ps1`、`train_value_net.py` | `vprior/posnet_R5_val.pt`、`_r4_*.pt` | — | 第 7 轮"正信号"后来被否（见下） | `DIR`；`LOG:7077-7234` |
| 2026-10-07 09:16→12:52 | `C156_clshead_64` / `S_k{1,4,48,96}_64`（**合并 5**） | **类**（156 局蒸馏类头）+ 搜索轴 | 156 局蒸馏 | `az_train.py --policy-only --freeze-backbone` | `vprior/_id156_clshead.pt` | **C156 `p̂ = 0.6562`，与对照 13/16/3 逐对完全相同、0/32 对翻转**；机制：建塔 24.8→23.5、闪电 10.5→10.6 | **否决**（"那个头在部署路径里只被用 argmax"） | `REG:669-675` |
| 2026-10-07 19:02→19:53 | `collect_k48_80` / `collect_explore_80` / `collect_explore_60` | —（采集） | 类探索 | `_tmp_collect_*.ps1` | — | — | — | `DIR` |
| 2026-10-07 20:44→21:30 | `REPRO_m6c_128` | —（复现） | — | `_tmp_repro_m6c.ps1` | — | 独立验证 0.7109 逐位复现 | 采纳（验证） | `DIR` |

### 1.8 阶段 H：S1b SDK 蒸馏 + 头消融（2026-10-08 → 2026-10-09）

| 时间 | 事件 / RUN 名 | 头 | 数据与监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-08 11:09→12:29 | `P0_A1_mirror` / `P0_A2_ex_light` / `P0_A3_ex_nolight` / `P0_A4_rv4_nolight` / `P0_E1_mc1` / `P0_E2_wide` / `P0_E3_playable` / `P0_A5_stoch_mirror`（**合并 8 arm，零训练**） | — | 对手/机制 arm | `_tmp_phase0*.ps1` | — | 见 `DOC/phase0_report_20261017.md` | 未评估/诊断 | `DIR` |
| 2026-10-08 12:34→14:19 | `NLR_fixed128` / `NLR_rand128` / `NLRND_fixed128` | 位置（**无闪电**配置） | — | `_tmp_nlr_nd.ps1` | — | — | 不显著 | `DIR` |
| 2026-10-08 15:46→19:13 | `SG_S0_quiet` / `SG_S1_spend1` / `SG_S2_spend2` / `SG_S4_spend4` / `SG_M25_sp4` / `SG_M50_sp4` / `SG_D2_bidir2` / `SG_D4_bidir4`（**合并 8 个 run**） | —（子博弈评分器臂，零训练） | — | `_tmp_subgame_scorer.ps1` | — | — | 未评估 | `DIR` |
| 2026-10-08 20:42→21:34 | `PS_C0_narrow` / `PS_P1_masked` / `PS_P2_argmax` / `PS_P3_masksamp` / `PS2_C0r_quiet_narrow`（**合并 5**） | —（位置 pin 臂，零训练） | — | `_tmp_subgame_pospin*.ps1` | — | — | 未评估 | `DIR` |
| **2026-10-08 23:33** | **S1b：SDK 评分反投影蒸馏 → `distill_sdk_A.pt`** | **类 + 位置**（策略网整体） | `distill_data/` 400 局 / **106,694 决策**；监督 = SDK 评分反投影到 (class, position)；`--epochs 30 --lambda-soft 0`（**不用软标签**：npz 的 `head_logits` 是旧模型自己输出＝自指闭环） | `code/distill/train.py`、`code/tools/graft_policy.py` | `checkpoints/distill_sdk_A.pt` → 嫁接 `checkpoints/grafted_sdk_A.pt` | 终值 epoch30 **`cls=0.6715 map=0.1491 tot=0.8239`**（epoch1 为 1.0039/1.0137/2.0169）；**94/94 键与部署 ckpt 一致**；嫁接后 `value_state` **保留原 `posnet_A_k5_m32.pt`** | **采纳** | `CKPT:128-137`；`LOG:7981-7996` |
| 2026-10-08 23:36→10-09 10:04 | `S1C_G1_posonly_wide` / `S1C_G2_posonly_masked`(38s 中止) / `S1C_G1_rand128` | —（部署消融，零训练） | — | — | — | S1c 冒烟 2 局：我方 **15.0 建 / 6.5 降 / 7.5 升**，有操作回合 22.0 ⇒ **升级 2.9→7.5/局** | 采纳（机制可见） | `LOG:7994-8018`；`REG:412` |
| 2026-10-09 09:33→11:05 | `HEUR_H0_argmax` / `HEUR_H1_skiptop` / `H2H_new_vs_old` | —（启发式对照，零训练） | — | `code/test_match/heur_pkg/main.py` | — | 蒸馏网+MC+搜索比教师本身 **+24.7pp** | 采纳（管线加价值） | `REG:476-499` |
| **2026-10-09 13:41** | **头消融 A：`distill_maponly`**（`--lambda-class 0`，只训动作图） | **位置/动作图** | 同一 `distill_data/`，只给图头损失 | `code/distill/train.py` | `checkpoints/distill_maponly.pt` → `grafted_maponly.pt` | **`map 0.1308`**（比完整蒸馏 0.1491 更低，容量全给图）；**`cls 42.3`**（类头未约束、漂移）；部署：K=24 **0.6016** / K=1 **0.4375**；机制 churn 型（53 回合 /33 建/17 降） | **采纳（消融臂）**；解读口径：λ=0 那头**不受约束会漂移** | `CKPT:139-146`；`REG:463-465` |
| **2026-10-09 14:14** | **头消融 B：`distill_clsonly`**（`--lambda-map 0`，只训类头） | **类** | 同上 | `code/distill/train.py` | `checkpoints/distill_clsonly.pt` → `grafted_clsonly.pt` | **`cls 0.7023`**；`map 24.9`（未约束）；部署 K=24 **0.5781** / K=1 **0.4297**；机制温和型（30 回合） | 采纳（消融臂） | 同上 |
| **2026-10-09 14:22** | **头消融 C：`distill_phold10`**（`--p-hold 0.1`，HOLD 降采样 10×） | **类 + 位置** | HOLD 降采样 10× 重训，目的是"让**类头分布可用**，为 `joint`（类交给学习）铺路" | `code/distill/train.py` | `checkpoints/distill_phold10.pt` → `grafted_phold10.pt` | 训练中（`checkpoints.md` 成文时）；部署 K=24 **0.7288** / K=1 **0.6250**；基线 0.8016/0.6875 ⇒ 排序与 K=24 一致 | 采纳（铺路） | `CKPT:144`；`REG:427-431` |
| 2026-10-09 第 9–10 轮 | 头消融结论汇总 | 类 / 位置 | — | — | — | 完整蒸馏 **0.8016**；*只训动作图* **0.6016**（52% 份额）；*只训类头* **0.5781**（47%）⇒ **52%+47%≈99%，接近可加、无强协同** ⇒ 修正"类头在 pos-only 下没用" | **采纳** | `REG:461-472` |
| 2026-10-09 14:33–16:48 | `AB_A_maponly` / `AB_B_clsonly` / `AB_SF_ex` / `AB_K1_{maponly,clsonly,phold10}` / `AB_F_masked_phold10` / `AB_C_phold10`（**合并 7 个评估臂**） | —（部署消融评估，零训练） | — | `_tmp_ablation_eval.ps1` | — | 基线 K=24 **0.8016** / K=1 **0.6875**；**F（`masked`+phold10 vs 基线）= `p̂ = 0.0938` ⇒ 惨败**（第 4 条被否的手写门控） | 采纳（否掉 masked） | `REG:412,427-439` |
| 2026-10-09 17:08→18:19 | `AB_I512_screen` / `AB_EV2_screen` / `AB_K1_S0` / `AB_MC1_screen` | —（机制臂） | — | — | — | 见 `DOC/next_step_ideas.md` | 不显著 | `DIR` |
| 2026-10-09 17:59→18:10 | `AZTEST` / `AZTEST2` / `AZTEST3` | **AZ 环数据链路** | 自对弈 dump → 转换器 → 训练器 | `azjsonl_to_value_npz.py`、`convert_az_dump.py` | — | 链路打通 | 采纳（工具） | `LOG:8332`；`REG` |

### 1.9 阶段 I：价值网 S2 / 搜索值 / 路 C（2026-10-09 → 2026-10-10）

> ⚠️ **这一阶段"动作网"的改动只有 `graft_value.py` 的嫁接（只换 `value_state`）与刷屏重训；类/位置头的 last 变更来自 10-08/10-09 的蒸馏线。**

| 时间 | 事件 / RUN 名 | 头 | 数据与监督 | 脚本 | 产物 | 读数 | 结论 | 出处 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-09 23:06→10-10 05:40 | `COLLECT_S2` / `ITER2_COLLECT` | **价值**（采集） | 训练专用 seed 段 `10001..10030`（第一轮）/`10031..10060`（第二轮），双 dump | `_tmp_collect_az.ps1` | — | 60 局 → **29,285 条带标签样本** | 采纳 | `LOG:8689`；`CKPT:166` |
| 2026-10-10 00:29→00:51 | `value_s2_train` / `_stream` / `_stream2`（**合并 3**） | **价值** | **该局终局胜负**（terminal） | `train_value_net.py --stream` | `checkpoints/value_s2.pt` | 按局 `val_mae` **0.3453**；独立小网**不能**被 `load_three_models` 直接加载（键交集 0） | 采纳（仅实验记录） | `CKPT:161,165` |
| 2026-10-10 01:19 | 嫁接 1：`grafted_sdk_A_vs2.pt` | **价值** | 同上（终局目标） | `code/tools/graft_value.py` | `checkpoints/grafted_sdk_A_vs2.pt`（4.29MB / `71616f81b4e2ceef`） | `class_state`/`pos_state` 与 `grafted_sdk_A.pt` **逐位相同 ✓**；通道验证：旧网 `value=-0.168159` vs 新网 `+0.439332`（差 0.607）；"MC 评分器"角色读数 **0.4531**（vs example、32 对） | 采纳（嫁接）；棋力**未单跑判据** | `CKPT:154,334-336` |
| 2026-10-10 01:22→01:36 | `value_search_s2` → `value_search_net.pt` | **价值** | **搜索回传的 `root_value`** | `train_value_antwar.py --save-net` | `checkpoints/value_search_net.pt` | 按局 `val_mae` **0.0718**（比终局目标 0.3453 **低 4.8 倍**）| **采纳（最好学的目标）** | `CKPT:155,162`；`REG:323-328` |
| 2026-10-10 01:36 | 嫁接 2：`grafted_sdk_A_vsearch.pt` | **价值** | 搜索值 | `graft_value.py` | `checkpoints/grafted_sdk_A_vsearch.pt`（4.29MB / `03e5f83a2a8bf52e`） | **0.7188**（当前最好；MC1+旧网 0.6797）；class/pos 逐位相同、value 94/94 全不同 | **采纳** | `CKPT:155,225` |
| 2026-10-10 02:01→06:36 | `AB_VSEARCH_dec` / `AB_MC1_K48_dec` / `AB_VSEARCH_gate` / `AB_VK48_dec`（**合并 4**） | —（评估） | — | `_tmp_ladder.py` | — | 四方闸门齐 ⇒ 价值网在"MC 评分器"角色上**第三次被否**；但重训把该角色从 0.3438 提到 **0.4531（+11pp）** | **否决（角色）** | `REG:311-314`；`LOG:8799` |
| 2026-10-10 03:11→08:46 | `AB_MC1_K48_dec` / `AB_VSEARCH2_diag` / `AB_VSEARCH_unseen`（**合并**） | —（评估） | 未见 seed 501..564 | — | — | MC1+搜索值网 = **0.7109**（33/25/6, CI[0.625,0.789], 无效 0%）vs 固定段 0.7188 ⇒ **未见 seed 确认通过** | 采纳（排除过拟合） | `LOG` 第 57 轮（续） |
| 2026-10-10 05:44 | 嫁接 3：`grafted_sdk_A_vsearch2.pt`（第二轮网） | **价值** | 搜索值（第二轮） | `train_value_antwar.py` + `graft_value.py` | `checkpoints/value_search_net2.pt`、`grafted_sdk_A_vsearch2.pt` | 对应 val_mae **0.0973**；⚠️ `_vs2` **≠** `_vsearch2`（命名提醒） | 采纳 | `CKPT:231`；`DIR` |
| 2026-10-10 08:29 | 嫁接 4：`grafted_phold10_vsearch.pt`（**路 C 混合体**） | **类+位置** ←`grafted_phold10.pt`；**价值** ←`grafted_sdk_A_vsearch.pt` | — | `code/tools/graft_value.py` | `checkpoints/grafted_phold10_vsearch.pt` | C1 首次冒烟（用 `grafted_phold10.pt` 陈旧旧网当叶值）出现**建塔刷屏/僵持**：seed 7 到**第 163 回合、塔 id 104（≈100 座塔）**；换叶值重试 C1b 到**第 105 回合、塔 id 72（≈70 座塔）** ⇒ **同样不过机制闸门** | **否决（路 C 关闭）** | `CKPT:172`；`REG:236-238` |
| 2026-10-10 10:04 / 08:50 | `MC1_rand_ctrl` / `VSEARCH_rand_diag` | —（控制臂，零训练） | 随机段 | `_tmp_rand_seg.ps1` | — | MC1 随机段 **0.7812**；控制臂证明"换价值网 +3.9pp"**证伪**（随机段 −7.8pp） | **否决** | `REG:161`；`LOG:9041` |
| 2026-10-10 11:48 / 14:15 | `AB_DELTA_vsearch_vs_baseline`（MC_SCORE=delta） | —（评估+重训） | — | `_tmp_delta_arm.ps1` | — | 判据 0.6797 = MC1 本身，**低于最好 0.7188**；机制反向（建塔 29→37）；⇒ "价值网替代/混合 rollout 当裁判"**共五次否证彻底关闭** | **否决** | `REG:161`；`LOG` 第 65 轮；`GIT` 53bb62a |
| 2026-10-10 13:45→14:22 | **`AB_C1SPAM_collect` → 刷屏数据重训价值网 → `grafted_phold10_vspam.pt`** | **价值** | 用 C1 路 C 配置（`MODE=joint`/`SKIP1=0`/`CLASS_FEAS_MASK=1`）采集**刷屏局面 + 结局**；300 局（?）→ ingest 终局标签 → `train_value_antwar.py` → `graft_value.py` 嫁接 | `_tmp_spamfix.ps1`、`_tmp_collect_az.ps1`、`graft_value.py` | `checkpoints/value_c1spam.pt`、`grafted_phold10_vspam.pt`（10-10 14:15） | 采集中刷屏方 **30 对全 0-2（p̂=0.0000，一局不赢）**；重训后**按局 val_mae 0.2533 < 终局目标版 0.3453**⇒ 确实学到信号；但同配置只换叶值：建塔 **124.3→111.5**、降级 **115.3→106.5**、**仍 4 局全负** | **否决**（价值网判断不是刷屏驱动 ⇒ 路 C 关闭） | `LOG:9135`；`REG:161`；`GIT` db21bf4 / b3970bd |
| 2026-10-10 14:26→18:47 | `AB_TAIL_*`（T256/T128/T64/T32/T64V/T32V/T64M，**7 臂**） | —（MC_TAIL 实现 + 筛查，零训练） | — | `_tmp_tail_screen.ps1` | — | T256 **0.9219** / T128 0.9062 / T64 0.9062 / T32 0.8125 / T64V 0.9219 / T32V 0.8906 / **T64M 0.8594（混合反而最差）** | 机制成立但**无处可用**（削上限不省时、补尾更慢） | `REG:123-128`；`LOG:9021-9144` |

### 1.10 阶段 J：最终报告（2026-10-10，无新训练）

| 时间 | 事件 | 头 | 出处 |
|---|---|---|---|
| 2026-10-10 | `final_report_20261010.md`：判据未达标（最好 0.7188 固定段 / 0.7109 未见段 = **+21pp**，差 8–9pp）；全部已知杠杆结局清单；`goal2_winrate_method_analysis.md` 定性 **MC 仲裁覆盖 20%→100%（`MC_EVERY` 5→1）是唯一强效应** | — | `DOC`；`GIT` e82ec78 / 64f6742 / 1cfa7c5 |

---

## 2. 按"监督信号"分类的统计

| # | 监督信号 | 动作网重训次数（含同批合并计数） | 价值网重训次数 | 代表性读数 | 结论趋势 |
|---|---|---|---|---|---|
| 1 | **对局结果 / 终局胜负（含 AWR 结果加权、outcome CE）** | **约 10**：ES+BC 精英 BC(~6，06-30~07-11)、BC-only 过夜跑 2、专家 A–D 1 批、候选 R（AWR 只训类头）1、M1/M1b/M1c/M1d 4、C1SPAM 价值重训 1 | 2（`value_s2.pt` 终局目标、`grafted_sdk_A_vs2.pt`） | 候选 R：loss 5.49→4.95 但 **跨局面 logit std 仍 ≈0.024（常数）**；M1c `p̂=0.5000`（=A1，逐字相同）| **几乎全否**：能降 loss，但改不了部署 argmax |
| 2 | **搜索访问分布 / visit 分布（含 `class_prior` 条件 CE、`pos_distill`）** | **约 7**：`train_class_prior`(C2)、`posnet_loop`、`posnet_convergence`、`value_prior`(vprior r1/r2)、`distill_prior_rule_v4`(换 pos)、第 14 轮"K48 教师蒸馏"、"类访问蒸馏" | 1（`value_search_net.pt` / `grafted_sdk_A_vsearch.pt` 用 `root_value`） | 位置轴 **r1：21.1%→35.2%**（采纳）；类轴 C2：**`p̂=0.5000` 与对照逐字节相同**（否决）；"类访问蒸馏"**目标退化，P1 不过，关闭** | **位置轴采纳、类轴否决**（类头只被 argmax 用） |
| 3 | **SDK 评分反投影（`ActionCatalog`/SDK 启发式 → 逐格软目标）** | **3**：score_distill（2026-07-14，12 方案线）、`distill/train.py` S1b（10-08）、`distill_prior_rule_v4`（10-09，用旧 V1/V2 蒸馏位置图当先验） | 0 | score_distill 的 12%/35%/58%/62% **是注释不是实测**（已撤回）；S1b 终值 `cls=0.6715 map=0.1491`、升级 2.9→7.5/局（**采纳**）；用 V1/V2 位置图当先验 20.3% vs 9.4%/1.6%（**否决**） | **S1b 采纳；"旧蒸馏位置图当先验"否决** |
| 4 | **Q 值 / 类条件价值 `Q(s,c)`** | **3**：`train_q_head` M2 + M3 + M3b（严格说训的是 Q 头，不是类/位置头本体） | 0 | M2：173,514 对 MSE 0.084→**0.074**、P-M2a 弱通过；但 `pos_pin=q` 部署**成本闸门超标中止**；M3b 50% 类探索"差一点" | **训练成功、部署中止**；REG 定性"参数化/信号/信用分配三个层面全封" |
| 5 | **GA / ES 无标签（fitness 只做选择，不用标签）** | **约 8**：ES(06-29/30)、ES+BC(07-01~07-07)、`ss_train`(07-02~07-11 两长跑)、GA+SS(07-13/07-14/07-19)、`ga_ss_train` 长跑(07-30~08-02)、专家 A–D(07-07) | 0 | `ss_20260709_191050/gen_0030`：vs ExampleAI **90.6%** 但 vs rule_v4 **1.6%**；`ss_20260710_155327/gen_0101`：vs ExampleAI **67.2%** | **对弱对手采纳、对 rule_v4 全废** |
| 6 | **手工规则 / 启发式蒸馏（储备规则、`pos_pin=reserve`、PD 规则）** | **2 训练**：`M1_distill`（把储备规则蒸进类头）、`distill_prior_rule_v4`；另有 `--pos-pin reserve` 零训练臂 | 0 | M1 训练侧 **logit std 0.024→0.178（状态依赖成立了）**、部署 `p̂=0.5000` | **否决为交付物**（只能当诊断臂，见 `REG:38-39,67-73`） |
| 7 | **价值标签（HP 差 / rel / abs / mix / τ / kgeo / 逐帧/逐局）** | 0（这些标签只训价值头） | **约 40**（`value_*` 8 目录 + `az_fixed` ~10 臂 + `az_bn_kgeo` 5 + `VT/VR/VK` 15 + `value_loop` 3 + `V1/V2/Q1/Q1b` 6 + R5 + S2/搜索值） | 最强 = **rel+kgeo tau50（32 局 +9.4pp）→ 128 局验收 0.4609（−7.0pp）**；`az_frozen` 冻结骨干 **34.4%**；"离线更准不换棋力"val MSE −3.8% 而 `p̂=0.6406`（−1.6pp n.s.） | **全轴关闭**（V1/V2/Q1/Q1b 共 6 次同型；最好只改善 val MSE 0.3%） |
| 8 | **头消融 / 降采样重训（把某个头单独训 / HOLD 降采样）** | **3**：`distill_maponly`、`distill_clsonly`、`distill_phold10` | 0 | `map 0.1308` / `cls 0.7023`；部署 K=24：maponly 0.6016、clsonly 0.5781、phold10 0.7288、完整 0.8016 ⇒ 份额 52%/47% | **采纳**（"接近可加、无强协同"） |
| 9 | **PPO / RL（on-policy）** | **4 步**：08-02 冒烟 → 08-02 LB 池 → 08-03 位置 aux `-logp` → 08-03 trust-region MSE；08-04 判死刑 | 0 | "胜率退化"；`cpp_engine_rewrite_research.md:46` 明确"已判死刑" | **否决** |
| 10 | **嫁接（换掉某个 ckpt 的 `class_state`/`pos_state`）** | **约 5**：`_tmp_make_swapped_pos_ckpt.py`（换 `pos_state` → 蒸馏版）、`graft_policy.py`（→`grafted_sdk_A.pt`）、`graft_value.py`（4 个 `grafted_*`，**只换 value**）、`make_three_net_ckpt.py`（三网同源复制） | — | 换 pos 蒸馏版：**20.3%→9.4%/1.6%（否决）**；换 SDK 蒸馏 class+pos：**采纳**（升级 2.9→7.5） | **换位置=否决；换类+位置=采纳** |
| 11 | **只训价值网（不可混入动作网事件）** | 0 | 见第 7 行 | — | — |

**一句话统计**：动作网重训事件约 **55–60 次**（其中约 35 次可归到"某批实验的一行"），只训价值网的事件约 **45–50 次**，零训练的推理/菜单/机制评估 run 约 **150+ 个**。

---

## 3. 时间轴（粗粒度阶段）

| 阶段 | 时间边界 | 一句话特征 | 支撑出处 |
|---|---|---|---|
| **A. ES / 精英 BC / 多专家** | 2026-06-29 → 07-11 | 用 ES（+BC 替代 ES 梯度）和多专家限类集训练**多头动作网**；对 ExampleAI 能到 90%+，对 rule_v4 ≈1.6%；出现"单动作坍缩（100% 闪电）" | `GIT` 47911ee→cc17cdf；`MEM`；`MSG`；`HANDOFF`；`DOC/notable_checkpoints.md` |
| **B. 策略空间进化 / GA+SS + 分数回归蒸馏** | 2026-07-12 → 08-02 | `ss_train.py`→`ga_ss_train.py`，行为空间变异 + 榜单对手；同时开出"SDK 评分 → 逐格回归蒸馏"支线 | `GIT` f129583→8c62e35；`training_history/ga_ss_20260730_093908/`；`training_history/score_distill_20260714_191151/` |
| **C. PPO 线（短命）** | 2026-08-02 → 08-04 | on-policy PPO + 位置通道进策略优化，**08-04 判死刑** | `GIT` 4554ea3→b2140a9；`DOC/cpp_engine_rewrite_research.md:46`；`training_history/ppo_smoke_test/` |
| **D. az_intent AlphaZero 自对弈** | 2026-08-04 → 09-07 | 从官方数据对齐 → 自对弈 AZ → 策略/价值**拆分** → C++ 引擎提速 → 08-18 定位"**特征提取器被训坍缩、棋盘信息全丢**"这一根因 | `DIR`（az_official_val/az_full/az_intent/az_cpp）；`GIT` 0d1f167→eeebd19 |
| **E. 三网拆分 / 冻结骨干 / 位置轴** | 2026-09-07 → 09-21 | 价值头在冻结骨干上重训（25%/34.4%）→ 三网同源 ckpt → **位置网单独重训** → `vprior_r1` 打 rule_v4 **21.1%→35.2%**（位置轴唯一可复现提升） | `LOG:99-379`；`CKPT:18-19,28`；`DOC/az_three_net_split_plan.md`；`LOG:2458-2563` |
| **F. 协议/身份冻结** | 2026-09-22 → 09-25 | **完全无训练**：对手确定性、天梯、协议规格、远端迁移；A/B 两体架构暂停 | `DIR`（runs 全是 `ladder_*`/`DET*`/`B0-B2`）；`DOC/champ_ladder_plan.md`、`protocol_ai_spec.md`、`ab_supervisor_plan.md` |
| **G. 头消融前夜：类轴/位置轴/蒸馏法（M1–M6, C, V, Q 系列）** | 2026-10-05 → 10-07 | 一天内把类轴（M1 BC→M2 Q头→M3 探索）、价值阶梯（V/Q/VT/VR/VK）、位置轴（C2/vprior）**全部试完并逐条否决**；同时确立 A1 操作配置与 M6b/M6c-ROOT 机制 | `LOG:4528-7490`；`REG:56-95,311-330`；`DOC/az_class_axis_plan.md` |
| **H. S1b SDK 蒸馏 + 头消融** | 2026-10-08 → 10-09 | 用 400 局 SDK 评分反投影重训**整个策略网**（`cls 1.0039→0.6715`、`map 1.0137→0.1491`），嫁接进部署 ckpt；再做 `maponly`/`clsonly`/`phold10` **头消融**（52%/47%） | `CKPT:128-146`；`LOG:7981-8198`；`REG:412,461-472` |
| **I. 价值网 S2 / 搜索值 / 路 C** | 2026-10-09 → 10-10 | **只重训价值网**：终局目标（`val_mae 0.3453`）→ 搜索值（`0.0718`，好 4.8×）→ 刷屏数据（`0.2533`）；路 C（joint+类搜索）**因机制闸门不过关闭** | `CKPT:148-174`；`LOG:8689-9144`；`REG:161,236-238` |
| **J. 报告与归档** | 2026-10-10 | 判据未达标（+21pp，差 8–9pp）；唯一强效应 = **MC 仲裁覆盖 20%→100%** | `DOC/final_report_20261010.md`、`goal2_winrate_method_analysis.md` |

---

## 4. 缺口清单

### 4.1 文档时间缺口（**没有实验记录的时间段**）

1. **2026-06-29 → 2026-09-08：`docs/experiment_log.md` 完全空白。** 日志第一条 `## ` 时间戳是 `2026-09-09 17:07:52`（`LOG:1-32`），且 `run_logged.sh` 本身是 2026-09 后才有的（`LOG:23-29`）。**7 月与 8 月的全部训练事件（约 20–25 次重训）只能从 git commit message、目录名、以及 `MEM`/`MSG`/`HANDOFF`/`CONV` 四份早期总结重建。**
2. **2026-07-12 → 2026-07-13 与 2026-08-04 → 2026-09-06：** git 历史里有提交但**无任何 run 目录明细之外的记录**；8 月的 AZ 训练读数只在 `docs/az_*.md`（计划/预注册）里零散出现，其中 **`az_full`（8/4）的 518KB `train.log` 与 `az_official_val` 的 49KB `run.log` 是唯一的一手日志**，未见整理成实验记录。
3. **2026-09-10 与 2026-09-17：** `docs/experiment_log.md` 这两天的日期标题数为 0（9-10 唯一条目是 `polonly_resume`/`eval_azp6_*`，9-10 只有 1 个 `## 2026-09-10`）；9-17 只有 run 目录名（`depth0_vs_raw` 等），多数没有对应 `## 2026-09-17` 条目细节。
4. **2026-10-05 一天内 100+ 条目但大量 `result: _待填_`**：M1b/M1c/M1d train、M1_distill_32、M1b_masked_32 等的 `result` 字段空着（`LOG:5251,5262,5291,5321` 等），结论只能从后续"手工补记"段落取。
5. **9-24 → 10-04 十天完全无记录**（`ab_supervisor_plan.md` 9-24、`git_remote_tsinghua.md` 9-25，下一条是 10-05）：git 日期直方图确认 2026-09-26 至 2026-10-04 **零 commit**（`GIT` 日期直方图）。
6. **没有"官方天梯成绩"记录**：全是自建 rule_v4/example 读数；`goal2_winrate_method_analysis.md` 自述"该配置对真实天梯的价值未测量"。

### 4.2 RUN 目录存在但找不到对应文档条目

（下列目录/文件在磁盘上有产物，但 `docs/experiment_log.md` 无同名 `## ` 条目，或条目 `result` 为空）

| 目录 / 文件 | 情况 |
|---|---|
| `training_history/value_20260711_222304`、`value_20260711_235320`、`value_20260712_054444`、`value_20260716_1039/1044/1048/1049/1207/1344/2210`、`value_20260717_001750` | **8 个价值网训练目录**，日志期（7 月）日志文件不存在，文档只有 `docs/value_*.md` 计划；**无任何 result** |
| `training_history/ss_20260709_191050/train.log`（3.9KB）、`ss_20260710_155327/train.log`（96KB） | 一手训练日志在，但**没有对应的 experiment_log 条目**；读数只在 `notable_checkpoints.md` |
| `training_history/ga_ss_20260730_093908/train.log`（491KB）+ `config.txt` | **无 experiment_log 条目**；`config.txt` 未被任何文档引用 |
| `training_history/score_distill_20260714_191151/{epoch_0010..0050,final}.pt` | **无条目**；`docs/distill_rule_v4.md` 的数字被 `LOG:1758-1770` 明确撤回为"非实测" |
| `training_history/az_official_val/run.log`、`az_full/{run.log,train.log}` | 4 份大日志（合计 ~1MB）**无条目**；只有 `DIR` 与 eval 日志 |
| `training_history/az_intent/`（~40 个 `eval_*.log`、`az_az0..7.pt`、`az_intent_batch*.pt`、`native_crash_diag*.log`） | **整个 8 月 AZ 训练批次无条目**；`R0/run.log`、`diagnose_*.log` 也未见文档化 |
| `training_history/az_cpp/`（`az_r101..`、`az_mix_*.pt` 等多组 p/v 混合） | **无条目**；8-10 的 `az_r50 68.8%` 只见于 git commit message（`GIT` 6e62e57） |
| `training_history/az_bn_kgeo/az_b1..b5.pt` | 有 `## 2026-09-15 00:21:17 — az_bn_kgeo` 条目，但 `result` 空 |
| `training_history/vprior/posnet_V1_val20.pt`、`posnet_v2_valfresh.pt`、`posnet_Q1_valpin.pt`、`posnet_Q1b_*.pt`、`posnet_R_class16.pt` | 有 run 条目，但结论只在 `REG:62-63,875` 里；**`checkpoints.md:114` 明确把它们列为"未入库、要用再拷"** |
| `training_history/vprior/posnet_VRrel*/Vrel*/Vk*/_r4_*.pt`（约 15 个） | `LOG:5659-5793` 是"手工补记"段（聚合），**逐 run 无独立条目** |
| `training_history/_smoke_posnet/`、`posnet_p2/`、`posnet_loop/`、`_posnet_conv/` | 有 `##` 条目，但 `pos_l30/pos_l150/pos_r1..r5/e8/e30/e80` 的**最终读数未逐 ckpt 记录** |
| `training_history/inject_ex02/valnet_{A,A2,B,C}.pt` | `## 2026-09-19 17:46:49 — inject_ex02_400g` 有，但四个 valnet 的**各自读数未记录** |
| `training_history/runs/20260917_*`（6 个）、`runs/20260922_*`（12 个）、`runs/20260923_*`（4 个） | 大部分在 `experiment_log` 有同名条目，但**9-22/9-23 的 16 个跑批全部是评估/确定性实验，无训练** |
| `training_history/runs/20261007_125245_S_k96_64/output.log`（922B，写于 16:20）、`runs/20261007_193251_collect_explore_80`（143B）、`runs/20261008_120559_P0_E3_playable`（631B） | **日志极短（<1KB）⇒ 极可能异常退出/被守卫拦下**，但 `result` 未记录原因 |
| `training_history/ppo_smoke_test/`（5 个 `ppo_iter_*.pt` + `train.log` 2KB） | **无 `##` 条目**；PPO 线只有计划文档 + 一句"判死刑" |
| 根目录 `training_history/az_loop/{vw_round1..3.pt, cur_bn.pt}`、`az_fixed/` 的 ~45 个 ckpt | 部分有条目（`az_frozen_*`、`mix_r10p_*`），**大部分只有目录没有逐项记录**；`checkpoints.md:115` 自述"`az_fixed/` 里的一些 ckpt 未入库" |
| `training_history/runs/20260909_194407_eval_hyb_gpub_cpuh/output.log`、`…_eval_hyb_cpub_gpuh/output.log` | **0 字节**（中途停止，`LOG:189,200` 有解释） |
| `training_history/cpp_ai_matches/ai0.log`、`ai1.log`、`game.log`、`*.stderr.log` | **全部 0 字节**（8/9 与 9/22 两批）⇒ 对局输出未落盘 |
| `training_history/_smoke_forced_skip/`、`_bench_collect/`、`_smoke_force/`、`_tmp_init_test.log`（68B） | 空目录/空日志 |

### 4.3 已消失的目录（文档提到、磁盘上不存在 ⇒ **无法复现**）

- `training_history/expert_a_v2/`（含 `gen_0010.pt`，9MB）— `HANDOFF:18,48`
- `training_history/expert_b/`、`expert_c/`、`expert_d/` — `HANDOFF:57-59`、`MSG:36-41`（推测：**从未真正启动过**，`HANDOFF` 把它们列为"待完成任务"）
- `training_history/20260706_012607/`、`training_history/20260706_122129/`（BC-only 两跑，含 `gen_0057.pt`）— `MEM:23,102`
- 根目录 `es_small_train.log`（2862B，7/6）是一手日志但**无对应目录**

### 4.4 主题缺口（**语料里没有的东西**）

1. **没有任何"真实天梯/SDK 官方对局"的胜率记录**——全部是自建 `rule_v4` / `example` / 自对弈读数。
2. **没有任何一次动作网重训跑通到"采纳并成为默认配置"的类头/位置头**：唯一的动作网采纳是 10-08 的 S1b SDK 蒸馏（`grafted_sdk_A.pt` 系），而它在 10-10 的最终报告里最好只到 **0.7188/0.7109**（差判据 8–9pp）。
3. **GA/ES 线在 7-30 之后没有任何后续**（`ga_ss_train.py` mtime 2026-07-29，之后未再训练）——文档里也没有"为什么停用 GA/ES"的显式结论。
4. **`rule_v4` 从未被蒸馏过**（`LOG:1758-1770` 明确撤回此前说法）——这是文档里明确标注的一条"**从未发生**"。
5. **多专家 + 门控网络从未训练过门控网**（`multi_expert_plan.md:279-313` 有方案，`MSG`/`HANDOFF` 只到专家 A）——**无任何门控网训练记录**。
6. **9 月中（9-22 → 10-04）与 8 月中下旬（8-19 → 9-06）都没有动作网重训**，只有诊断/协议/迁移工作。

---

## 5. 一句话结论

语料里动作网（类/位置/策略）**被重新训练**的事件可以清楚分成三条血统：**ES/精英BC/多专家（6/29–7/11）→ 策略空间进化 SS/GA+SS（7/12–8/02）→ AZ 自对弈拆分三网（8/04–9-21）→ SDK 蒸馏 + 头消融（10-08/10-09）**；
**唯一被采纳、且在机制层可见效果的动作网重训是 10-08 的 S1b SDK 评分反投影蒸馏**（`cls 1.0039→0.6715`、`map 1.0137→0.1491`、升级 2.9→7.5/局），
而 **位置轴的 `vprior_r1`（21.1%→35.2%）是唯一"单变量、可复现"的动作网提升**；类头方向（M1/M1b/M1c/M1d、C2、C156、候选 R、AWR、类访问蒸馏）**共有 6+ 次独立否决**，
核心原因在所有记录里高度一致：**类头在部署路径里只被 `argmax` 使用，而它的 argmax 恒为闪电/HOLD**。
