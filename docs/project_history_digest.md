# 项目历史浓缩摘要（给上下文被压缩过的新 agent）

> 只读梳理，写于 2026-10-28。工作目录 `D:\2026智能体大赛_新2_distill`，分支 `feat/rollout_search_20261021`，HEAD `768c223`，
> 1,250 commits（2026-06-29 → 2026-10-28）。**本文不改 `code/**`、不跑对局、不 kill 进程。**
>
> **先读这三份，本文只是它们的"目录 + 名词表 + 踩坑索引"，不重复内容：**
> - `docs/project_history.md`（567 行，2026-06-29→10-26 全史，含证据分级/血统/战绩史）← 细节主要在这儿
> - `docs/project_state_20261028.md`（271 行，10-28 当时状态 + 未解 + 矛盾清单）
> - `docs/experiment_log.md`（1.0 MB，实验索引；⚠️ 10-09 之后大量 `_待填_`）

---

## 1. 项目阶段表

| 时间 | 做了什么 | 结论 / 转向原因 |
|---|---|---|
| 2026-06-29~07-15 | 起步即"神经网络+ES"（非规则 AI 打底）；`ss_train`/精英 BC/GA-SS/分数回归蒸馏多线并进 | 管线通但强度≈vs ExampleAI 50%；**553K 维 ES 收敛测试失败**（`讨论记录.md:73-77`）⇒ 弃 ES。产物 `gen_0120` 成后续一切模型血统起点 |
| 2026-07-20~07-29 | 自适应对手池、温度采样、多轮锦标赛、action dropout、intent decoding | **纯工程期，零强度产出**，但后来 3 个月所有 A/B 都靠这套工具 |
| 2026-08-02~08-18 | PPO（崩）+ **Bundle-MCTS AlphaZero 自对弈** + C++ 引擎 pybind11（8.7×） | 自对弈相对指标 50%→68.8%，但 vs rule_v4 只 31.2%，根因"**搜索坍缩到闪电 99.3% ⇒ 特征提取器训坍缩**"⇒ 端到端 RL 关闭 |
| 2026-09-07~09-25 | BN 迁移修坍缩；**三网拆分**（类/位置/价值）；价值网当位置先验 | **历史上唯一一批过统计门槛的 DL 正收益**（+14~24pp vs rule_v4），但都在 09-18~09-20 三天内，且是"换个用法"不是新范式 |
| 2026-10-05~10-10 | rule_v4 攻坚：17 条轴逐条预注册 + 128 局验收 | 0.0391→0.5312 靠**工程修复（薄搜索）**，0.7344 靠 `mc_quiet` 仲裁；**两段 90% 目标未达成**（0.7344/0.7031） |
| 2026-10-10~10-18 | 动作网重训 8 条路线全量落地并判据；Phase 0 禁闪电子博弈 | **8 条全负或中性** ⇒ "重训动作网"这条路被量尺关掉；关键机制：同一张网有仲裁 0.6797 / 无仲裁 0.0312 |
| 2026-10-18~10-22 | 实体 Transformer 价值网、5000 局低质量值网、**扁平 rollout 取代树搜索** | **最大单点跃迁之一**：同候选同网只换评估器 → 内部 86%、vs rule_v4 0.984，且**更便宜**（2.45s→1.01~2.0s） |
| 2026-10-22~10-26 | 先验修复三路线全跑完→全败；先验天花板测量；随机化 rollout；内部循环赛；RL 六线调研 | 内部显著（+5.08hp）**对外 0 传导**（vs 亚军 0W-15L）；固定种子后"hybrid 胜 f10 62.5%"**不复现**（48.4%）⇒ 转"测量学优先" |
| 2026-10-26~10-28 | 引擎 RNG 消费 A/B、非法操作严格模式、**亚军候选注入 A/B**、小值网 32/3 接入 | 注入 **13W-3L=81.2%（+12.38hp）** 但归因有三条替代解释；RNG 消费 **−18.8pp**；历史"48 局 0 非法"是**结构性 0** |

---

## 2. 名词表（最重要的一节）

### 2.1 ⚠️ 三个先钉死的命名陷阱（不钉死，后面所有数字都会读错）

| 名词 | **实际含义** | 定义处 |
|---|---|---|
| **"冠军"**（10-20 之后的对局里） | **我们自己上一版最强配置**（k=160+确定性 N=1），**不是平台冠军** | `code/test_match/_ab_k48_n64_vs_champ.sh:3`；`project_history.md:42` |
| **"亚军"** | **真对手**：大赛第 2 名的 AI，本地重建可跑 | `docs/runnerup_build_note.md`；`phase0_report_20261017.md:11` |
| **`rule_v4`** | **作者本人**参赛 AI，最终 **16/50**；**不是**官方示例基线 | `docs/findings_flat_rollout_vs_tree.md:112-127` |
| **官方 `example`** | 真官方样例（很弱，几乎不放闪电） | `rivals_analysis.md:66-82` |

⇒ 本文分别记作「rule_v4」「亚军」「内部冠军」「example」。**"vs rule_v4 = 100%"与"对外 0%"可以同时成立。**

### 2.2 搜索 / 候选结构

| 术语 | 含义 | 定义处 |
|---|---|---|
| `pos-only` | 搜索模式：**类钉死、只搜同类内的位置**。三选一 `joint\|class-only\|pos-only` | `bundle_mcts.py:419`；`az_selfplay.py:647` |
| `POSPIN` | pos-only 下"钉哪个类"：`argmax\|playable\|reserve\|reserve_up\|masked\|q\|mc_light\|mc_quiet` | `bundle_mcts.py:421` |
| **`mc_quiet`** | **方法侧 M6b＝最强组件**：把候选那一手打下去，之后**双方都不出招**推到终局，按终局 HP 差选类。有偏（不再是"双方正常发挥"）但低方差、便宜（~0.3-0.5s/候选） | `mc_quiet.py:1-11`；分支在 `bundle_mcts.py:696,713` |
| `AZAI_K` | 候选 bundle 数 k（canonical=24）。从 `k×SAMPLE_MULT` 次采样里按计数取 top-k | `bundle_mcts.py:434,668` |
| `AZAI_SAMPLE_MULT` | 采样倍数（canonical=15）⇒ 每决策采样 k×15 次 | 同上 |
| `SKIP1` | 单候选时跳过迭代；pos-only 下 ~97% 回合命中，**便宜 ~26×** | `bundle_mcts.py:1111,1127` |
| `MC_EVERY=5` / `MC_ROOTONLY=1` | 每 5 回合**只在根节点**仲裁一次（全节点仲裁 ~20 min/局） | `az_bridge_ai.py:671,677`；`az_selfplay.py:522,582` |
| `MC_HORIZON=256` | `mc_quiet` 安静 rollout 的回合上限 | `az_bridge_ai.py:668`；`mc_quiet.py:46` |
| `MC_CELLS` | **同类内**多放几个格进 MC 菜单（0=只 1 个）；`MC_CELLS=4` 曾读 75% | `az_bridge_ai.py:683`；`findings_flat_rollout_vs_tree.md:153` |
| `MC_TOPK=3` | 按 `v_c=max AM[c]` 的**跨类经济排名**取前 k 个类进菜单 | `az_bridge_ai.py:681`；`train_pos_class_rank.py:4` |
| `MC_UPG` | 给前 n 座塔放**显式升级候选** `Operation(12,x,y)` | `az_bridge_ai.py:685`；`bundle_mcts.py:475` |
| `TCLASS` / `TPOS` / `TEMP` | 类采样温度 / 位置采样温度 / 最终决策温度（canonical 0.5 / 1.0 / **1e-6≈argmax**） | `az_bridge_ai.py:656-657`；`mcts.py` |
| `POSPRIOR` | 根节点位置先验替代网络图：`off\|value\|uniform`（canonical=off） | `az_bridge_ai.py:629-636` |
| `canonical`（部署口径） | `DEFAULT_CANON_ENV` 那一整串：`K=24 ITERS=256 DEPTH=4 SAMPLE_MULT=15 MODE=pos-only POSPIN=mc_quiet MC_EVERY=5 MC_HORIZON=256 MC_ROOTONLY=1 SKIP1=1 TCLASS=0.5 TPOS=1.0 TEMP=1e-6 POSPRIOR=off` + `posnet_A_k5_m32.pt` | `ab_search_modes.py:64-69` |

### 2.3 扁平 rollout 与随机化

| 术语 | 含义 | 定义处 |
|---|---|---|
| `SEARCH=flat` | 用 `FlatRolloutGreedy` 取代树搜索：根节点一次扩出 k 个候选，每候选跑 N 次 h 回合**真实引擎** rollout，叶子用值网打分，取均值 | `flat_rollout_greedy.py:1-29` |
| `FLAT_ROLLOUTS` | 每候选的 rollout 次数 N（canonical=16；"贵配置"=64） | `az_bridge_ai.py:737` |
| `FLAT_HORIZON` | 每次 rollout 的回合数 h（canonical=64） | 同上 |
| **`FLAT_ROUNDS`** | **每决策的总引擎回合预算**（canonical=39000 ≈ 2.45 s，按 62.7 µs/回合折算）。候选多时自动压小 N ⇒ 等时间对照的实现 | `flat_rollout_greedy.py:20-22,52` |
| **`N_eff = FLAT_ROUNDS/(k×16)`** | ⚠️**事故公式**：k=160 时 N_eff 只有 **15** ⇒ 所有"k=160 下 N=32/64/128/256"的大采样实验**实际都被截断到 15** | `exp_plan_20261024_night.md:10-13` |
| **`FLAT_SEED`** | rollout 随机流种子。**不在 canonical 串里**（代码默认 0），各脚本自己加 53/61/1000+seed。实测它决定"对手性格/轨迹集合"（53 vs 61 轨迹 Jaccard 仅 **8.1%**），能挪 10-30pp **读数** | `az_bridge_ai.py:749`；`day_summary_20261027.md:60` |
| `decay(C,k)` | 随机化 rollout 的出招概率衰减 `p(r)=C·exp(-r/k)`；`const\|decay\|first` 三模式 | `flat_rollout_greedy.py:70-82` |
| `FLAT_SHARE_PREFIX` | 前缀树记忆化：相同前缀轨迹只算一次（共享率 const 27% / decay 43% / first 94%） | `az_bridge_ai.py:754`；`flat_rollout_greedy.py:76-78` |
| `k48/N64` | k=48 + `FLAT_ROLLOUTS=64` + `FLAT_ROUNDS=49152/49920` 的**贵配置**。内部冠军 #9 = 64.1% 的出处；注入主结果也用它 | `prior_repair_20261023.md:488,523` |
| `叶子` / `去重叶子` | N 次 rollout 产生的不同轨迹数；基线 decay(.3,1)/h16 只有 **17/64**，强随机 decay(1,3)/h8 有 **62/64** | `day_summary_20261027.md:9-10` |

### 2.4 checkpoint / 先验命名

| 术语 | 含义 | 定义处 |
|---|---|---|
| **三网** | 一个 ckpt 里三个独立 state：`class_state` / `pos_state` / `value_state`，各 94 张量、**互不共享**、训练时互不影响 | `checkpoints.md:19,32-36`；`az_selfplay.py:112-117` |
| **`posnet_A_k5_m32`** | **钉死 ckpt**（6.5 MB）：10-05 只训位置网（价值先验 multi-**k5** / **max-cells 32** / 400 局 / 60 epoch）。M6c-ROOT 的 0.7344 就是它。血统：class 继承 `three_mix`、value=`valnet_A2` | `checkpoints.md:18,28,33`；`next_step_survey_20261005.md §1` |
| **`f10`** | `prior_distill_f10.pt`：`--frac 1.0`（全量 16 局）用**值网 h=16 rollout 排序**当标签、listwise 软标签、只解冻 `action_map_conv`。**现役先验、最耐打** | `prior_repair_20261023.md:202,461`；`project_history.md:286` |
| `f025` / `f05` | 同上的 `--frac 0.25/0.5` 学习曲线版（只用于诊断欠拟合，不作晋级依据） | `prior_repair:202,232` |
| **`prior_z`** | z-score/Pearson 损失重训的先验（`prior_z_80g/prior_z_best/...`）。**离线更好、实战更差**（80.4%→85.1% 离线，64.1%→43.8% 实战） | `prior_ceiling_20261026.md:111-119` |
| **`hybrid`** | `prior_z_80g` 全套 + **只把闪电通道(17)换回 f10** 的单变量消融臂。"hybrid 更强"已被推翻 | `commit d35ec19`；`prior_ceiling:235-249` |
| `posnet_*` / `bc_prior` / `prior_distill_mcts16` | 位置网历代变体 / 搜索选中动作当标签的 BC 先验（中性）/ MCTS 访问教师（中性） | `checkpoints.md:18-28`；`prior_repair:368-403` |
| `gen_0120_model.pt` | **血统起点**（原文件 404 MB 里只有 ~1% 是模型，其余是 GA 种群） | `checkpoints.md:25,76-101` |
| **"折叠" ckpt** | 独立 `ValueNetwork` 折成三网形态（`v_32_3_folded.pt`），等价误差 1.24e-07 | `code/test_match/fold_small_value_ckpt.py:241` |

### 2.5 开关 / 测量学 / 工具

| 术语 | 含义 | 定义处 |
|---|---|---|
| `AZAI_SEED_STATE=1` | 确定性播种：候选集与访问分布变成状态的确定性函数 | `az_bridge_ai.py:653`；`bundle_mcts.py:379` |
| `AZAI_VALUE_CKPT` | 叶子值网替换开关；**空=关**（原样返回同一对象，行为逐位不变） | `value_override.py:13,38,168` |
| `AZAI_ORACLE_INJECT` | **亚军候选注入**开关（问亚军的最终 action 当额外候选）；空=零开销 no-op | `az_bridge_ai.py:877-920` |
| `AZAI_ORACLE_TIMEOUT` | 亚军查询超时，默认 **10 s**（旧默认 2.5 s，2.5 s 下 5/8 局丢注入） | `commit ede6400`；`project_state_20261028.md:28` |
| `cold_handle_rule_illegal` | 默认 **False = 对齐官方**；`--cold-illegal` 回旧行为 | `docs/illegal_op_strict_mode_20261028.md:14` |
| **`verdict=engine`** | **唯一有效局判据**；其余（timeout/aborted/illegal_strict/hp_tiebreak/无 RESULT）判无效并重跑 | `ab_search_modes.py:75-77` |
| `hit` / `hit_sel` | `hit_sel`=旧口径"注入被搜索选中"；新 `hit`=**动作一致率**（含双方都 HOLD）。⚠️ 注入主结果全是老口径 | `oracle_inject_ab_20261028.md:445-458` |
| **"结构性 0"** | 检查代码永不成立导致的假 0 次（`apply_operation_list` 返回 `None` ⇒ 非法检查永不触发） | `illegal_op_strict_mode_20261028.md:88` |
| 座位 `P0/P1` | 先手/后手。**座位效应 27~47pp（后手优势 38-44pp）**，不分座位会掩盖一切小效应 | `prior_ceiling:130,157`；`rl_survey:19` |
| `支撑集` | 某类动作的合法格集合。降级只 **2 格**（中位）、建塔 32 格、闪电 271 格 ⇒ recall 口径必须看支撑集 | `prior_ceiling:8-27` |
| **`canonical`** | 见 §2.2 最后一行（部署基线串） | `ab_search_modes.py:64-72` |
| **`ab_search_modes.py`** | **主 A/B 驱动**：flat vs mcts 两臂、`--paired-sides` 座位配对、`--resume`/`--retry-invalid`/`--game-timeout-s` | `code/test_match/ab_search_modes.py` |
| **`_tmp_ladder.py`** | 阶梯并行运行器（跑 `match_via_sdk.py`）。纪律：**胜负只读 bridge 的 `RESULT` 行，绝不重推**；必须同时报"双方真的出招了" | `_tmp_ladder.py:1-18` |
| **`probe_*`** | 单用途只读诊断脚本（`probe_menu_rank` / `probe_mc_quiet_candidate_count` / `probe_menu_width` / `probe_upgrade_selectability` …），多在 `code/my_ai/az_intent/` 与 `code/test_match/` |

---

## 3. 重大结论变迁（只列"曾经以为 X，后来发现 Y"）

| 旧结论 | 新事实 | 证据 |
|---|---|---|
| 回放能忠实还原对局 | 一次"引擎不等价"其实是**回放脚本 bug**（`(4,0)` vs `(0,28)`）；16 局重放后 **16/16 忠实** | `commit ec7dd8b`；`prior_ceiling:8` |
| "48 局 0 非法 ⇒ 本地是安全的" | **结构性 0**（检查永不成立）。真测 120 局/50480 op 也是 0，但 `ORACLE_INJECT+NO_DOWNGRADE` **2/2 局真的非法判负** | `illegal_op_strict_mode_20261028.md:19,88,196` |
| "hybrid 比 f10 强 20pp"（59.4%/65.6%，n=64 两次） | 固定种子后 **45.3%，血量差 +0.00** ⇒ 是随机流假象 | `prior_ceiling:280`；`commit 3a8c401` |
| "A4 = 1.000（64-0）打赢 rule_v4" | 内部对打 `MC_EVERY=1` vs `5` 只有 **53.1%（不显著）** ⇒ 天花板假象 | `findings_flat_rollout_vs_tree.md:98-108` |
| "先验覆盖 92.8% ⇒ 先验不是瓶颈" | 被**降级**稀释（占 74% 决策、支撑集中位仅 2 格）；拆开看**建塔 @24 = 74.9% ≈ 随机 75.0%**，只有闪电真有信息（67.3% vs 8.9%，仅占 4% 决策） | `prior_ceiling:8-27` |
| "离线指标好 ⇒ 实战会强" | **五次独立脱钩**：recall@24 93.9→94.4、Spearman −0.11→+0.428、留出@24 80.4→85.1%、R² 0.9969、R² 0.4708 ⇒ **实战全部不动或更差** | `prior_ceiling:111-119`；`checkpoints.md:258` |
| "k=160 下 N=32/64/128/256 的大采样实验有效" | `N_eff=FLAT_ROUNDS/(k×16)`，k=160 时 N_eff 只有 15 ⇒ **那批"大采样"实际全被截断到 15，口径全错** | `exp_plan_20261024_night.md:10-13` |
| "值网更准 ⇒ 更强" | n=128 配对：换更准 terminal 网 0.7344→0.7734（CI 重叠）；换**亚军手写价值** 0.7344→**0.3906（−34pp）**；同一代理的"动作"却 **+31pp** | `experiment_log.md:11386-11392`；`project_state_20261028.md:93-108` |
| "随机化程度不够导致评估噪声大" | 有效叶子 17→62（3.6×），32 局血量差 **−0.78** ⇒ 零效应；消费引擎 RNG 更是 **−18.8pp（−5.31hp）** | `day_summary_20261027.md:13`；`runs/20261028_085257_rng_idle_clean` |
| "rule_v4 蒸馏到 58-62%" | **该数字是设计文档的示例数字，从未实跑** ⇒ 正式撤回 | `commit 1f82201` |
| 类维搜索增益 0、位置维 100% | 该结论在**旧制度**下测（类头已坍缩、91% 回合 HOLD）⇒ **不可迁移**到现在 | `commit f0bbb41`；`project_history.md:103` |
| `AZAI_MC_CELLS` 是"选格更准"的杠杆 | 增益更可能来自"类比较样本变多"而非选格更准 | `findings:153` |
| "先验真的有用（k=24 先验 vs 随机 65.6%）" | 成立但**只值 +5.5hp**，且不能缩 k（"缩 k 免费"被推翻） | `commit 6f88414` |

---

## 4. 死胡同清单（不要再走）

| 方向 | 一句证据 |
|---|---|
| **端到端 AlphaZero / 自对弈闭环** | 搜索坍缩到闪电 99.3% ⇒ 骨干被训坍缩；追平 KataGo 主 run 需 ~146 个月（`eeebd19`；`rl_survey:73`） |
| **把"搜索的输出"当监督目标重训策略网（自蒸馏）** | 蒸分布 **−43.0pp**、蒸 rank-1 −18.0pp、topk −10.2pp、TEMP=1.0 −20.3pp（`cda8fc8`；`baff93d`） |
| **用近视叶评估当教师**（安静 rollout / 1-ply 价值） | A② **−15.6pp**、r1 **−7.8pp**，两次都是"学生比教师更忠实反而更弱"（`889e73b`；`9dc30b2`） |
| **用官方 SDK 蒸馏网替换部署先验** | 拼接蒸馏网 k=24 = **12.5%（−11.6hp）**，两轮都败（`prior_repair:105-141`） |
| **把价值网当"裁判"替代 rollout 打分** | rollout 0.9375 vs 搜索值网 **0.2031**（`097ded8`） |
| **靠加数据 / 加参数解决信用分配** | 数据量曲线**平坦**（14 局 +0.036 / 28 局 +0.040 / 39 局 +0.045）；555k→2.6M 参数 @24 只 +0.2pp（`338853b`；`rollout_noise:62-78`） |
| **手写规则 / 门槛类**（强制升级、抑制降级、闪电储备阈值、建塔花钱门槛） | 全负：0.0000 / 0.4531 / 0.2031 / 合并 −8.9pp（`final_report:44-56`） |
| **Phase 0 禁闪电机理搬到真实对局** | `MC_NO_LIGHT+MC_UPG+MC_TOPK` = **0.6094（−12.5pp）**（`final_report:121-125`） |
| **reward shaping（攒金币）** | potential-based shaping "安全与无用是一体两面"；`Φ_gold(终局)≠0` 会让攒钱成为最优（`rl_survey:52-54`） |
| **靠消费引擎 RNG 换多样性** | 离线划算（68.4% 墙钟拿满 64 叶）但对局 **31.2%（−5.31hp）** |
| **加宽候选菜单 / 显式升级候选 / 平局门控** | −14.1pp / 0.5156 / 三个臂"机制根本没激活"（`final_report:42-46`；`46985ee`） |
| **predictive world model** | **计划存在、代码从未存在**（全仓库无 `world_model`/`WorldModel`） |
| **多专家 + 门控网络** | 代码存在（`gating_network.py`）、被 `agent.py` 引用，但**零实验记录**，不在部署链上 |

---

## 5. 关键资产索引

### 5.1 checkpoint（→ 权威表 `docs/checkpoints.md`）
- **在用/最佳**：`checkpoints/posnet_A_k5_m32.pt`（sha256 `6fc38cb8…`，钉死）——`checkpoints.md:18,44`
- **现役先验**：`checkpoints/prior_distill_f10.pt`（f10）——`prior_repair:9-11`
- **现役值网**：`value_state = valnet_A2.pt`（terminal 标签，2026-09-19，**BN 从未训过**）——`checkpoints.md:20,46`
- **血统起点**：`gen_0120_model.pt`（`ea9d1964…`）——`checkpoints.md:25,51`
- **已废弃**：`docs/notable_checkpoints.md`（`checkpoints.md:104` 明确"作废，保留作考古"）
- **规矩**：拷入 → sha256 → 登记 → **连二进制一起 commit**（只提交文档忘 `git add` 是这套做法最常见的错，`checkpoints.md:120-126`）

### 5.2 数据集（→ 权威表 `docs/value_net_pipeline_survey.md`，含逐键 shape/来源/时间）
- 价值标签族：`entity_ds_128.npz`（127,578 行，亚军**手写价值** ~1e4 量纲，R² 可达 0.9969）、`value_ds_{12g,24g,ent8,runnerup_selfplay}.npz`
- 先验标签族：`training_history/prior_labels/`（`g*`=rollout 教师）、`prior_labels_mcts/`（`m*`=MCTS 访问教师）；**两类标签按文件名前缀分开，混训会错**（`commit 030bea1`）
- 位置先验源：`training_history/vprior/vp_A_k5_m32.npz`（400 局）
- 大件已归档：`distill_top2/lowq_full`（85.1 GB）移去 `F:\2026智能体大赛_新2\training_history\distill_top2\lowq_full`（`commit 7943fc3`）
- ⚠️ 陷阱：`rollout_ds_test.npz` 的 `stats` **退化**（38 维全 0）⇒ 别当留出集；`entity_ds_128_board` 等**无生成脚本**

### 5.3 脚本代表作（对局 / 采集 / 训练 / 诊断 各一）
| 用途 | 文件 |
|---|---|
| 对局 A/B | `code/test_match/ab_search_modes.py`（+ `_tmp_ladder.py` 阶梯 / `match_via_sdk.py` 单局） |
| 采集 | `code/test_match/collect_prior_labels.py`（`--teacher rollout\|mcts`） |
| 训练先验 | `code/my_ai/train_prior_distill.py`（`--pattern/--frac`）；z-score 版 `code/my_ai/train_prior_z.py` |
| 训练值网 | `code/my_ai/az_intent/train_value_net.py`；实体 Transformer `code/my_ai/entity_transformer/train.py` |
| 诊断 | `code/my_ai/az_intent/diagnose_search.py`；`diag_*` 分散在 `code/my_ai/az_intent/` 与 `code/test_match/`（如 `diag_tree_leaves`/`diag_seed_similarity`/`diag_rank_vs_outcome`/`diag_visit_sharpness`） |

### 5.4 关键诊断工具 / 探针
- **噪声**：`docs/rollout_noise_20261026.md`（N=1 跨 7.70 名 / N=16 1.92 / N=64 0.96，R²=0.916）
- **回放审计**：`code/test_match/recall_on_runnerup_v3.py` + `code/test_match/eval_recall_report.py`（`ab_search_modes` 的 `_tmp_ladder.read_game` 是唯一权威解析器，**别重写**，`ab_search_modes.py:93-95`）
- **严格模式探针**：`--cold-illegal` + 逐条影子校验 + `illegal-summary`（`illegal_op_strict_mode_20261028.md`）
- **离线召回**：`eval_recall_report.md`（按动作类型拆分；⚠️ min-cells 三套口径并存）
- **RNG**：`code/my_ai/az_intent/rng_engine.py`（`AZAI_FLAT_RNG_PERTURB`）+ `diag_rng_rank_agreement.py`

### 5.5 必读参考文档
`docs/rivals_analysis.md`（冠亚军源码逆向，全部带 `文件:行号`）· `docs/rl_survey/00_summary.md`（六线调研：不上完整 AZ、该做 Sequential Halving + SPRT）· `docs/dl_feasibility_20261026.md`（DL 可行边界 + 8 条不该再走的路）· `docs/engine_rng_survey.md` / `engine_rng_impl.md`（48-bit LCG，逐位确定）

---

## 6. 未解之谜（最多 5 条）

1. **注入 +31pp（81.2%）到底归因于什么？** 有三条替代解释：①换的是"另一个完整代理的决策（含它自己的搜索）"不是静态先验；②可能是绕过我方值网错误（同一代理的价值函数接进来是 −34pp）；③前缀树让新候选拿到预算优待。**缺安慰剂臂**（注入同数量随机合法候选）—— 这是全项目最便宜、最能一次排除三个解释的实验（`project_state_20261028.md:129-165`）。
2. **小值网 32/3 的 128 局强度 A/B 从未跑。** 工程全部就绪（1.52× 提速、8/8 冒烟过），判据（A≥41.2%）和命令都写好了，10-28 全天**未找到该 run**。这是"评估器质量→棋力"这一环**唯一还空着**的证据（`project_state_20261028.md:229`）。
3. **"1.000 vs 0.984" 的 0.016 未裁决。** 同一配置 `findings:82` 写 63-1-0→1.000，同文件 `:46` 与 commit `ceeae5d` 写 64-0；`match_results/ladder_logs/A4_flat_every1` 目录在但**从未逐局重算**（`project_history.md:561`）。
4. **部署口径（k=24 / N=16）的噪声与有效叶子从未测。** 现有噪声矩阵和 17/64 都在 k=48/N=64 上做的 ⇒ 所有"降噪无用"的结论其实**不是在部署配置下得出的**（`project_state_20261028.md:207`）。
5. **CRN（配对 RNG 偏移）是"降噪"方向上唯一没被证伪的杠杆，但从未进过对局 A/B。** 离线实测候选差 sd 降 13%-56%、成本≈0（`engine_rng_survey.md:14`）。另一条并列未解：**座位差 38-47pp 从未排查**（P1 能看到 P0 出招），它足以吞掉任何 <10pp 的效应。

---

## 附：读历史数字的三条纪律（都是踩坑换来的）

1. **按座位拆分**（座位效应 27~47pp ≈ 355 Elo）；2. **vs rule_v4 的读数在 0.89 以上要打折**，判别必须用内部对打（`findings:110`）；3. **16 局分辨率只有 ±25pp**，"要分辨 6%→12% 需 ~350 局/臂"（`prior_ceiling:182`）。
外加一条元规则：**看 commit 日期与文档内的"结论/判定"节，不要看标题** —— `final_report_*` 这类"结论味"标题过期最快，而 `plan_*`/`next_step_*` 大多**早已被执行**（`project_history.md:553`）。
