# 下一轮候选干预的调研（2026-10-05，B3 基线跑的同时做的**只读**调研）

> 目的：为「第一里程碑（vs rule_v4 ≥70%）」挑**下一轮**要做的干预，先把"可行路径 + 成本 + 已知反证"
> 落到文件里（**文件才是记忆**）。本文**全是只读调研的产物**（两个无上下文 subagent + 引用到的历史文档/日志），
> **不是实验结论**，强度一律按"**弱**（引用+机制读数，未做新对局）"读。
> ⚠️ 本文**不是预注册**：真要动手，必须另写预注册（含样本量）并走机械门。

## 1. 当前起点的血统（钉死 ckpt = `training_history/vprior/posnet_A_k5_m32.pt`）

- 血统：`gen0120`（官方 `ExampleAI` 蒸馏/初始化）→ `az_fixed/mix_r10p_vw_pol_frozen.pt`
  → `make_three_net_ckpt.py`（三个 state 同源拷贝 = 三网 ckpt `three_mix_r10p_vw_pol_frozen.pt`）
  → `collect_value_prior.py --multi-class-k 5 --max-cells 32 --row-weight-lambda 0.2`（400 局，6133 s）
  → `train_value_prior.py --data vprior/vp_A_k5_m32.npz --epochs 60 --batch-size 256 --lr 1e-3 --seed 0`（6435 s）
  ⇒ 本 ckpt（**只训了位置网**，class/value 冻结）。出处：`docs/experiment_log.md:3616-3680`。
- ckpt 元数据（实读）：`three_net=True, num_heads=3, latent_dim=64, num_resblocks=6, no_bn=False(BN),
  vprior_target_tau=1.0, vprior_tpos=1.0, vprior_epoch=60`。
- 网络：`code/my_ai/network.py:116-290`（骨干 + ①`action_map_conv` 24×19×19 位置图 ②3 个类头
  (B,24) ③value 标量）；输入 `Ant-Game/SDK/utils/features.py`（board(28,19,19) + stats(42)）。
- 搜索读 value 的位置：`code/my_ai/az_intent/az_selfplay.py:117,166,170`（`value_state` → tanh）→
  `bundle_mcts.py:243,258-259,378-384`（逐层变号 backup，PUCT 用 `-child.mean_value`）；
  终局值 = `clip((我方hp-对方hp)/20)`。**语义 = 当前行动者的绝对优势 ∈[-1,1]**。
- **同线旧口径读数**：`posnet_A_k5_m32` vs rule_v4 **54.3%（256 局，`eval.py` 旧口径）**（`experiment_log.md:3727`）
  ⇒ 本轮 128 局配对的先验预期是 **50–55%**（**不是**历史 46.9%）。

## 2. 四个方法层假设（H1–H4）的现状与"最便宜的干预"（全部来自只读调研）

| H | 现状（项目自己写的） | 最便宜干预 | 成本 | 已知反证 / 风险 |
|---|---|---|---|---|
| **H2** 价值视野 tau | 现价值头是 **`terminal` 标签**（`valnet_A2` 元数据 `valnet_label=terminal`）= 整局视野 | 用**已有缓存**重训价值头（`--cache training_history/inject_ex02/value_cache`，**不用重采**），扫 tau | **≈104 s/epoch**，3 epoch ≈ 6 min/档；注入 <1 min；16 局快读数 ≈46 min | 🔴 **方向可能反**：先例 tau20→50 来自**旧 value_warmup 线**（基线 tau=20），而现线是 terminal ⇒ 扫 tau50/100/200 是**把视野改短**。🔴 `abs` 标签 75–90% 方差可被"抄 `stats[1]`"解释（本缓存实测 R²：tau50 geo 0.898 / kgeo 0.777、tau200 geo 0.748 / kgeo 0.573）⇒ 价值头拿不到新信息。🔴 同判据上已有反证：`terminal`(A2) 61.2% > `abs+kgeo`(C) 59.8%（`:3204-3210`）。🔴 `rel`（唯一有真新信号、corr −0.165）与搜索读法不匹配 ⇒ 要配 `AZAI_REL2ABS` = **协议变更**。 |
| **H1** 信用分配 / ~96% HOLD | 无专门 plan；机制读数：空过率 96.7%，掩码 argmax 命中 HOLD 81%/90.5% | ① 先用已有工具**数**"raw 与搜索不一致的决策落在哪"（分钟级，无对局）；② 干预 = 在行权重上加"非 HOLD 类加权"（`train_value_prior.py` 的 `--row-weight-lambda` 现在只区分 argmax/非 argmax） | 只训 7–93 min + 一次配对台 | ⚠️ 该线的 **joint 禁闪电配对台单次 5.5 h**（`svs_joint_ban17` 19814 s）⇒ 判据必须预注册小 n |
| **H3** 策略表达不出多头 bundle（**项目主线**） | **"搜索 → 策略"的 visit/硬蒸馏目标在这条线上根本没实现**：`az_posnet_v2_plan.md:374-386` §8 第 7 条把它列进"第二阶段"，§4 自注"机制已定，实施待定"(:193) | 先跑**可复现性闸门** `_tmp_search_target_similarity.py`（分钟级，同局面换 rng 比目标分布） | 绿灯才采集（400 局，6133 s）+ 训练 93 min；**新损失代码要另写 plan** | 已知基线：目标分布余弦 0.35、bundle 同率 42.5%、value 目标余弦 0.67（`az_q_target_plan.md:40`）⇒ 若目标本身不可复现，"学搜索"就学不到东西 |
| **H4** 类轴 masked/HOLD 坏不动点 | `az_class_axis_plan.md:49-58` 自述**是"自洽的坏不动点"**，靠"当前分布上重训类头"打不开；实测类头排序 = 【不可用类】> HOLD >【可用类】 | ① 类头诊断（分钟级 CPU）；② `az_train.py --policy-only` 在**已有的分布外数据** `training_history/inject_ex02/data`（369,476 帧、38.9% 有塔）上重训类头 | 分钟级诊断；重训 4 epoch | ⚠️ 项目自述"靠这件小事打不开"；且"类轴搜索增益 = 0"（`:44-47`），另有一个未分开的替代解释（`t_class` 下采样从未真正探索其他类） |

## 3. "换价值头"这条流水线的**可行机制**（若将来真要做 H2/价值改造，可照抄）

1. **缓存可用**：`--cache training_history/inject_ex02/value_cache` 读 `board/stats/player/value_target`
   四个 memmap（`az_intent/train_value_net.py:97-129`）；实测 400 局 / 369,476 行；
   逐帧 HP 就在 `stats[:,1]`（`features.py:117,134-136`）⇒ **算 tau 标签不用重新采集**（实测重算 11.4 s）。
2. **只训 value 头**：`train_value_net.py` 会 `load_three_models(--ckpt)` 后只训 value、再 `save_three_net`
   写回三个 state；实测 `posnet_A_k5_m32.pt` 载入回写 **94/94 张量逐位相同** ⇒ class/pos 不会被动。
   ⚠️ `value_state` 的 BN **从未训过**（`num_batches_tracked={0}`）⇒ 必须 `--freeze-bn`（默认开）。
3. **单变量注入**：仓库里**没有**换 `value_state` 的现成脚本（`_tmp_make_swapped_pos_ckpt.py` 换了
   `pos_state` 且 SRC 写死）⇒ 若要做，得写个 ~10 行的"读 ckpt → 替换 `value_state` → 存"的小脚本
   （顶层 3 个 state，每个 state 94 张量，键如 `initial_conv.0.weight`）。
4. **零假设臂**：先把 **A2 价值网**（terminal）按同法换进 `posnet_A` ⇒ 用来量"注入过程本身"的效应。

## 4. 本文给出的**判断**（可被推翻；推翻要照纪律走）

- **H2 的字面路线（abs × tau50/100/200）在现线上大概率是零/负结果**，值得跑的唯一理由是
  **"确认 H2 在现线不成立"**（把它从寄存器里划掉），而不是提升棋力。
- **H3（visit/硬蒸馏）仍是项目主线里唯一"还没被实现过"的大杠杆**，但它前面有一道
  **几分钟就能跑的闸门**（搜索目标可复现性）；闸门不过就别投数据。
- **H1 的诊断部分（数不一致决策的分布）几乎免费**，应该在任何训练干预之前先做。
- 下一步的选择应当**等 B3 基线读数落地之后**再定（预注册里已写死：`p̂<0.70` ⇒ 从 H1–H4 挑一个另写预注册）。
