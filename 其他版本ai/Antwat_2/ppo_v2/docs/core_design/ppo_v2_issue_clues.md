# PPO V2 训练不理想问题线索清单

> 基于代码库全量审读，针对 WinRate 波动大（45%→40%→70%）、Reward 不稳定（270→218→529）、VLoss 偏大且反弹（4467→4278→4883）三大症状，按可能性和影响度综合排序。

---

## 线索 1：VLoss 值域与 vf_coef 严重失衡，价值梯度信号被压制

**问题描述**：当前 `vf_coef=0.05`，VLoss 约 4500，加权贡献约 225，远超 PLoss（~0.1）。即使 `param_tuning_report.md` 建议降至 0.00003，加权贡献仍约 0.135，与 PLoss 量级相当。但更关键的是：VLoss 值域数千是因为 returns 未归一化（终局奖励 ±500 叠加 gamma 折扣后 returns 值域可达数百），value head 输出也在数百量级，MSE 自然为数千。当 vf_coef 极低时，价值函数几乎学不到有效梯度，导致 VLoss 长期居高不下且反弹。

**可能的影响**：高

**调查方向**：
- 检查 `batch_metrics.jsonl` 中 `value_loss`、`value_pred_mean`、`value_pred_std`、`td_error_mean` 的趋势
- 检查 `ppo_trainer.py:447-456` 的 value loss 计算：`0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()`
- 检查 `clip_values.values_min/max`（-500/500）是否与实际 returns 值域匹配

**验证方法**：
1. 打印一个 batch 的 `returns` 统计量（mean/std/min/max），确认值域
2. 计算 `vf_coef * VLoss` 与 `PLoss` 的比值，若 >10x 则确认失衡
3. 对比方案：对 returns 做 z-score 归一化后重跑，观察 VLoss 是否降至 0.1-10 量级

---

## 线索 2：终局奖励 ±500 过大，导致 returns 和 advantages 方差极高

**问题描述**：`REWARD_CONFIG["win_reward"]=500.0`、`loss_reward=-500.0`，而中间步奖励通常在 -10 到 +20 之间。一局 250 步的 episode，终局奖励占总 return 的绝对主导地位。这导致：(1) 赢局和输局的 returns 分布严重双峰化，advantages 方差极大；(2) GAE 虽然能平滑，但终局奖励的冲击仍通过 bootstrap value 传播；(3) 价值函数难以在赢/输之间做出准确预测，VLoss 居高不下。

**可能的影响**：高

**调查方向**：
- 检查 `episode_batch_battle_stats.jsonl` 中 `avg_rw_end_reward` 的占比
- 检查 `batch_metrics.jsonl` 中 `reward_min`/`reward_max`、`return_mean`/`return_std`、`advantages_std`
- 检查 `antwar_env.py:293-299` 终局奖励逻辑

**验证方法**：
1. 统计 `selfplay_battle_log.jsonl` 中每局的 `reward`，绘制分布直方图，确认双峰
2. 计算终局奖励占 episode return 的平均比例
3. 尝试将 `win_reward`/`loss_reward` 降至 ±50 或 ±100，观察 VLoss 和训练稳定性变化

---

## 线索 3：对手池初始全为随机策略，早期训练信号噪声大

**问题描述**：`_create_initial_opponents()` 创建 3 个随机初始化策略作为初始对手。随机策略几乎不执行有效操作（大量 noop），导致早期 episode 的 reward 信号主要由环境固有属性（蚂蚁自动攻击）决定，而非策略行为。当对手池逐渐被真实策略替换后，对手强度突变，导致 WinRate 大幅波动。

**可能的影响**：高

**调查方向**：
- 检查 `selfplay/league/pool.json` 中 `added_episodes`，确认初始对手何时被淘汰
- 检查 `sp_all.log` 中早期 episode 的对手 ID 和对战结果
- 检查 `opponent_update_interval=100`，即前 100 个 episode 只能打随机对手

**验证方法**：
1. 统计前 100 个 episode 的 WinRate 和 Reward 分布，与 100-200 episode 对比
2. 检查 `payoff.json` 中初始对手的 TrueSkill mu 值变化，确认是否长期停留在池中
3. 尝试将初始对手改为预训练的规则策略（如 BasicTowerAI），观察早期训练稳定性

---

## 线索 4：辅助损失标签截断处理不一致，末尾步标签被零填充

**问题描述**：`compute_aux_labels_from_trajectory()` 中 `T = n - max(AUX_HORIZON_STEPS)`，即最后 16 步没有有效标签。`_attach_aux_labels()` 对这些步用零填充，并设置 `aux_valid_mask=False`。但 `compute_auxiliary_loss()` 中直接使用 `F.mse_loss(pred, label)` 计算，**没有使用 `aux_valid_mask` 过滤无效步**。这意味着末尾步的零标签被当作真实标签训练，产生错误的梯度信号。

**可能的影响**：高

**调查方向**：
- 检查 `ant_war_policy_value_network.py:360-414` 的 `compute_auxiliary_loss()`，确认是否使用了 `aux_valid_mask`
- 检查 `ppo_trainer.py:631-696` 的 `_add_auxiliary_loss()`，确认是否传递了 mask
- 检查 `batch.py:78-89` 的 `to_tensors()`，确认 `aux_valid_mask` 是否被传递到训练流程

**验证方法**：
1. 在 `compute_auxiliary_loss()` 中添加 `aux_valid_mask` 参数，仅对 mask=True 的步计算 MSE
2. 对比修改前后的 `aux_tower_loss` 和 `aux_gold_loss` 值变化
3. 检查修改后 VLoss 是否有改善（辅助损失的错误梯度也会影响 value encoder）

---

## 线索 5：双优化器共享 total_loss 反传，辅助损失梯度同时流向策略和价值参数

**问题描述**：`_add_auxiliary_loss()` 将所有辅助损失（包括 policy encoder 和 value encoder 的辅助头损失）加到同一个 `total_loss` 上。然后 `total_loss.backward()` 一次性反传，梯度同时流向策略参数和价值参数。但辅助损失的系数（如 `aux_tower_coef=0.005`）是针对 policy encoder 设计的，对 value encoder 可能过大或过小，导致：(1) value encoder 的辅助梯度干扰价值函数学习；(2) 双方辅助头的梯度通过共享的 `total_loss` 相互耦合。

**可能的影响**：中

**调查方向**：
- 检查 `ppo_trainer.py:673-693`，value encoder 的辅助损失使用与 policy encoder 相同的系数
- 检查 `batch_metrics.jsonl` 中 `vf_aux_tower_loss`/`vf_aux_gold_loss` 与 `aux_tower_loss`/`aux_gold_loss` 的对比
- 检查 `grad_norm_value` 是否因辅助损失而异常

**验证方法**：
1. 分别计算 policy aux loss 和 value aux loss 对各自参数的梯度范数
2. 尝试为 value encoder 的辅助头使用独立的系数（如降低 10x）
3. 或者将 value encoder 的辅助损失从 total_loss 中分离，用 value_optimizer 单独反传

---

## 线索 6：clip_eps_vf=0.2 过大，价值函数裁剪几乎不生效

**问题描述**：`clip_eps_vf=0.2`，而 `clip_values.values_min/max=-500/500`。当 value head 输出在数百量级时，0.2 的裁剪范围几乎不起作用（例如 old_value=300，new_value=350，差值 50 >> 0.2，裁剪后 value_pred_clipped=300.2，与 new_value=350 的 MSE 仍由 unclipped 版本主导）。这导致 value clipping 形同虚设，价值函数可以任意大幅度更新。

**可能的影响**：中

**调查方向**：
- 检查 `ppo_trainer.py:451-456` 的 value clipping 逻辑
- 检查 `batch_metrics.jsonl` 中 `value_pred_mean` 和 `value_input_mean` 的差值
- 计算 `|new_values - old_values|` 的均值，与 `clip_eps_vf` 对比

**验证方法**：
1. 在 `_compute_losses()` 中记录 `value_loss_unclipped` 和 `value_loss_clipped` 的比例
2. 如果 unclipped 几乎总是 > clipped，说明裁剪不生效
3. 尝试将 `clip_eps_vf` 增大至 50-100（与 value 值域匹配），或对 returns 归一化后使用标准 0.2

---

## 线索 7：胜率计算基于 reward 符号而非实际胜负，draw 判定不准确

**问题描述**：`selfplay.py:343` 中 `result_str = "win" if reward > 0 else ("loss" if reward < 0 else "draw")`，但 episode reward 是所有步奖励之和。当终局奖励为 ±500 时，这个判定基本正确；但如果 episode 被截断（达到 max_steps），没有终局奖励，此时 reward 的正负可能不代表实际胜负。此外，`selfplay.py:424` 中 `result = 1 if reward > 0 else (-1 if reward < 0 else 0)` 用于更新 payoff，同样存在此问题。

**可能的影响**：中

**调查方向**：
- 检查 `episode_collector.py:192-196` 的截断处理逻辑
- 检查截断 episode 的比例（`episode_length >= max_steps` 的频率）
- 检查 `selfplay_battle_log.jsonl` 中 result=draw 的记录

**验证方法**：
1. 统计截断 episode 的 reward 分布，确认是否有 reward 接近 0 但实际一方占优的情况
2. 对比使用 env 返回的 `winner` 信息与 reward 符号判定的差异
3. 修改为使用 `info["reward_detail"]["end_reward"]` 判定胜负（非零即有明确胜负）

---

## 线索 8：GAE bootstrap value 在截断时使用当前策略估值，引入偏差

**问题描述**：`episode_collector.py:193-194` 中，截断时 `batch.final_value = self._self_agent.get_value(obs_self)`。这个值来自当前策略的 value head，但：(1) 截断时的观测是 `not done` 分支更新后的 obs_self，而非 done=True 时的观测；(2) 截断后强制设置 `batch.dones[-1] = True`，但 GAE 计算时 `final_value` 被用作 `V(s_T)`，如果 value head 估值不准，会引入系统性偏差；(3) 多个 episode merge 后，每个 episode 的 final_value 独立，但 GAE 倒推时的 episode 边界匹配逻辑（`gae.py:49-76`）依赖 done 标记的正确性。

**可能的影响**：中

**调查方向**：
- 检查 `gae.py:66-76` 的 episode 边界处理，确认 `ep_idx` 递减是否与 done 标记对齐
- 检查截断 episode 的 `final_value` 量级是否合理
- 检查 `batch_metrics.jsonl` 中 `td_error_mean` 和 `td_error_std` 是否异常

**验证方法**：
1. 打印截断 episode 的 `final_value` 和该 episode 的 `returns[-1]`，确认偏差
2. 对比截断 vs 自然终止 episode 的 GAE advantages 分布
3. 尝试对截断 episode 使用 `final_value=0`（即视为终止），观察训练变化

---

## 线索 9：exploit_prob 调度从 0.3 线性增长到 0.9，早期过度探索弱对手

**问题描述**：`OpponentSelector._compute_exploit_prob()` 使用线性调度：`p_effective = 0.3 + 0.6 * progress`。在训练初期（progress < 0.2），exploit_prob < 0.42，即超过一半的时间在探索（选择高 sigma/随机对手）。但初期对手池主要是随机策略，探索意味着打更多随机对手，无法获得有效的策略改进信号。同时，exploit 选择的是 "mu 最高（最强）的对手"，但随机策略的 mu 可能因少量对局而虚高。

**可能的影响**：中

**调查方向**：
- 检查 `opponent_selector.py:122-137` 的调度逻辑
- 检查 `payoff.json` 中各对手的 mu/sigma 分布
- 检查 `sp_all.log` 中被选中对手的 ID 分布

**验证方法**：
1. 统计每个 episode 选择的对手 ID，确认是否集中在少数对手
2. 对比不同 progress 阶段的 WinRate，确认早期是否因对手太弱而虚高
3. 尝试将初始 exploit_prob 提高到 0.5-0.7，观察早期训练信号质量

---

## 线索 10：辅助标签 own_base_dmg 使用 hp_dmg_raw[1]（己方受到的伤害），语义与预测头名称不匹配

**问题描述**：`episode_collector.py:121` 中 `"own_base_dmg": hp_dmg_raw[1]`，而 `hp_dmg_raw` 来自 `antwar_env.py:322` 的 `(round(hp_dmg, 2), round(hp_dmg_received, 2))`。即 `hp_dmg_raw[0]` 是己方对敌方造成的伤害，`hp_dmg_raw[1]` 是己方受到的伤害。但 `BaseDamageHead` 的输出前 5 维是 "own" 部分，标签却是 "己方受到的伤害"，语义上 "own_base_dmg" 应该是 "己方基地受到的伤害" 还是 "己方造成的基地伤害" 存在歧义。如果预测头的语义是 "己方造成的伤害"，则标签取反了。

**可能的影响**：中

**调查方向**：
- 检查 `episode_collector.py:116-123` 的标签提取逻辑
- 检查 `antwar_env.py:225-226` 的 `hp_dmg` 和 `hp_dmg_received` 定义
- 检查 `ant_war_policy_value_network.py:379-386` 的 `aux_base_loss` 计算中 own/enemy 的划分

**验证方法**：
1. 在一局对战中打印 `hp_dmg_raw` 和 `own_base_dmg`/`enemy_base_dmg`，确认语义
2. 检查 `BaseDamageHead` 的 own 部分预测值与标签的相关性
3. 如果语义确实反了，修正后观察 `aux_base_loss` 的变化

---

## 线索 11：logit_noise_std=0.2 在训练时注入噪声，可能干扰 old_log_prob 的一致性

**问题描述**：`get_action()` 中当 `logit_noise_std > 0` 时，在 type_logits 上添加高斯噪声。这意味着采集时的 log_prob 是基于加噪后的 logits 计算的，但 PPO 更新时 `evaluate_actions()` 计算的 new_log_prob 是基于无噪 logits。虽然 PPO 的 ratio = new/old 理论上可以处理这种差异，但噪声引入的额外 KL 偏差会导致：(1) 采集时的策略与训练时的策略不一致；(2) approx_kl 估计偏高，可能触发早停；(3) clip_fraction 虚高。

**可能的影响**：中

**调查方向**：
- 检查 `ant_war_policy_value_network.py:145-148` 的噪声注入逻辑
- 检查 `batch_metrics.jsonl` 中 `approx_kl` 和 `clip_fraction` 的值
- 对比有噪声 vs 无噪声采集时的 `ratio_mean`/`ratio_std`

**验证方法**：
1. 设置 `logit_noise_std=0` 跑 20 个 episode，对比 `clip_fraction` 和 `approx_kl`
2. 如果无噪声时 clip_fraction 显著下降，说明噪声是 clip 偏高的主因之一
3. 考虑仅在 type_probs 上添加 Dirichlet 噪声（类似 AlphaZero），而非 logit 噪声

---

## 线索 12：reward 组成过多且量级差异大，稀疏信号被密集信号淹没

**问题描述**：`_compute_battle_rewards()` 包含 9 个奖励分量：基地伤害(2.0x)、塔伤害(0.2x)、金币获取(0.05x)、敌方金币惩罚(0.01x)、塔存活(0.02x/tower)、科技加成(1.5/1.0 per level)、余额奖励(0.02x)、死亡惩罚(-0.05x/ant)、终局奖励(±500)。此外还有动作奖励（建塔/升级/科技等，最高 15.0）。这些分量的量级从 0.01 到 500 不等，且终局奖励的稀疏大信号可能使策略难以归因哪些行为导致了高 return。

**可能的影响**：中

**调查方向**：
- 检查 `episode_batch_battle_stats.jsonl` 中各 `avg_rw_*` 分量的均值和方差
- 检查 `selfplay_battle_log.jsonl` 中 `reward_sources` 的分布
- 计算各分量对总 reward 方差的贡献比例

**验证方法**：
1. 对 100 局的 reward_sources 做 PCA 或方差分解，找出主导分量
2. 如果终局奖励贡献 >80% 的方差，考虑将其缩小或改为 shaping
3. 尝试移除低贡献分量（如 `enemy_coin_gain_weight`、`balance_reward_weight`），简化信号

---

## 线索 13：PPO epoch=2 且 target_kl=0.05，可能过早触发早停

**问题描述**：`ppo_epochs=2`，`target_kl=0.05`。在 epoch 0 结束后检查 `epoch_approx_kl`，如果 >0.05 则跳过 epoch 1。考虑到 `logit_noise_std=0.2` 导致的 old/new log_prob 偏差（见线索 11），以及 `clip_eps=0.15` 相对较宽，approx_kl 容易超过 0.05，导致实际只训练 1 个 epoch。这减少了策略更新的充分性，可能导致学习速度慢。

**可能的影响**：低

**调查方向**：
- 检查 `batch_metrics.jsonl` 中 `approx_kl` 的分布
- 检查 `training_{time}.log` 中 "Early stopping" 的出现频率
- 检查 `ppo_trainer.py:296-304` 的早停逻辑

**验证方法**：
1. 统计早停触发的比例（epoch 1 被跳过的次数 / 总更新次数）
2. 如果 >50%，考虑提高 `target_kl` 至 0.1 或增加 `ppo_epochs` 至 3
3. 对比早停频繁 vs 不频繁时期的 WinRate 变化

---

## 线索 14：对手池淘汰仅基于 TrueSkill mu，可能淘汰有价值的多样化对手

**问题描述**：`OpponentPool._evict_worst()` 按 `mu + protection_bonus` 排序淘汰最低分对手。但 mu 最低的对手可能是策略风格最不同的（如极端防守型），淘汰后对手池的策略多样性下降。此外，`min_games_threshold=8` 的保护机制可能不够——如果新对手在前 8 局全输（mu 急剧下降），第 9 局后即可被淘汰，来不及展示其训练价值。

**可能的影响**：低

**调查方向**：
- 检查 `pool.json` 中被淘汰对手的 `games_played` 和 `added_episodes`
- 检查 `payoff.json` 中对手的 mu 分布是否过于集中
- 检查对手池大小（`opponent_pool_size=10`）是否足够维持多样性

**验证方法**：
1. 记录每次淘汰的对手 ID 和其 mu/sigma，确认是否淘汰了 sigma 仍较高的对手
2. 增大 `opponent_pool_size` 至 15-20，观察 WinRate 波动是否减小
3. 考虑基于策略距离（如参数 L2 距离）而非仅 mu 来决定淘汰

---

## 线索 15：cosine decay 学习率调度在训练初期就开始衰减

**问题描述**：`LRScheduler` 使用 cosine decay，`progress = episode / total_episodes`。在 episode 1 时 progress ≈ 0，lr ≈ lr_base；到 episode 1000（50%）时 lr 已降至 lr_base * 0.5。对于 2000 episode 的训练，策略在尚未收敛时学习率已大幅衰减，可能导致后期学习停滞。特别是 `lr_vf=0.00025` 在 50% 进度时仅剩 0.000125，价值函数可能来不及拟合。

**可能的影响**：低

**调查方向**：
- 检查 `batch_metrics.jsonl` 中 `learning_rate` 的变化趋势
- 检查 VLoss 在训练后半段是否停止下降
- 检查 `lr_scheduler.py:44-49` 的 cosine decay 计算

**验证方法**：
1. 绘制 lr 和 lr_vf 随 episode 的变化曲线
2. 对比固定 lr vs cosine decay 的训练曲线
3. 考虑增加 warmup 阶段（如前 200 episode）或推迟衰减起点

---

## 线索 16：reward clip=2000 过大，基本不生效

**问题描述**：`REWARD_CONFIG["step_reward_clip"]=2000.0`，而单步 reward 通常在 -10 到 +30 之间（不含终局奖励 ±500 仅在最后一步出现）。clip=2000 基本不起作用，无法防止极端 reward 值（如终局 ±500）对 GAE 计算的冲击。

**可能的影响**：低

**调查方向**：
- 检查 `antwar_env.py:302-303` 的 clip 逻辑
- 检查 `batch_metrics.jsonl` 中 `reward_min`/`reward_max` 是否接近 ±500
- 确认终局奖励是否被 clip（单步 reward 含终局奖励时可能超过 500）

**验证方法**：
1. 打印含终局奖励步的 reward 值，确认是否超过 2000
2. 如果终局奖励 + 其他奖励 < 2000，则 clip 不生效
3. 考虑将 clip 降至 100-200，或对终局奖励单独处理

---

## 线索 17：action reward 中的 downgrade_tower_penalty=-9.0 可能过重

**问题描述**：`REWARD_CONFIG["downgrade_tower_penalty"]=-9.0`，而建塔奖励仅 0.3-0.6，升级奖励 3.0-5.0。降塔惩罚远大于建塔奖励，可能导致策略完全避免降塔操作。但在某些战术场景下（如卖塔重建），降塔是合理操作。过重的惩罚可能限制策略的探索空间。

**可能的影响**：低

**调查方向**：
- 检查 `episode_batch_battle_stats.jsonl` 中 `avg_action_counts` 的 `downgrade_tower` 计数
- 如果 downgrade_tower 计数始终为 0，确认策略是否被惩罚压制
- 检查 `selfplay_battle_log.jsonl` 中 `action_counts` 的分布

**验证方法**：
1. 统计 100 局中 downgrade_tower 的使用频率
2. 如果频率 <0.1%，尝试将惩罚降至 -3.0，观察策略是否开始探索降塔
3. 对比降塔惩罚调整前后的 WinRate

---

## 线索 18：enemy 辅助标签的损失权重与 own 相同，但预测难度不同

**问题描述**：`aux_enemy_tower_coef=0.005` 与 `aux_tower_coef=0.005` 相同，`aux_enemy_gold_coef=0.001` 与 `aux_gold_coef=0.001` 相同。但敌方信息在观测中可能更不完整（如敌方塔的精确 HP 可能不可观测），预测敌方未来伤害/收入的难度更高。相同的权重可能导致 enemy 辅助头的梯度噪声更大，干扰策略学习。

**可能的影响**：低

**调查方向**：
- 检查 `batch_metrics.jsonl` 中 `aux_enemy_tower_loss` vs `aux_tower_loss` 的比值
- 检查观测编码中敌方信息的完整度
- 检查 enemy 辅助头预测值与标签的相关系数

**验证方法**：
1. 如果 `aux_enemy_tower_loss` 始终远大于 `aux_tower_loss`，说明 enemy 预测更难
2. 尝试降低 enemy 辅助头权重至 own 的 1/2 或 1/5
3. 观察调整后 PLoss 和 WinRate 的变化

---

## 综合优先级排序

| 排序 | 线索编号 | 标题 | 可能性 | 影响度 | 综合评级 |
|------|----------|------|--------|--------|----------|
| 1 | 4 | 辅助损失标签截断未过滤无效步 | 高 | 高 | **紧急** |
| 2 | 1 | VLoss 值域与 vf_coef 失衡 | 高 | 高 | **紧急** |
| 3 | 2 | 终局奖励 ±500 过大 | 高 | 高 | **重要** |
| 4 | 3 | 对手池初始全随机 | 高 | 高 | **重要** |
| 5 | 6 | clip_eps_vf 过大不生效 | 中 | 中 | **关注** |
| 6 | 5 | 双优化器辅助损失梯度耦合 | 中 | 中 | **关注** |
| 7 | 7 | 胜率基于 reward 符号判定 | 中 | 中 | **关注** |
| 8 | 8 | GAE bootstrap value 截断偏差 | 中 | 中 | **关注** |
| 9 | 10 | own_base_dmg 标签语义可能反了 | 中 | 中 | **关注** |
| 10 | 11 | logit_noise 干扰 old/new log_prob 一致性 | 中 | 中 | **关注** |
| 11 | 9 | exploit_prob 调度早期过度探索 | 中 | 中 | **关注** |
| 12 | 12 | reward 组成过多量级差异大 | 中 | 中 | **关注** |
| 13 | 13 | PPO epoch 早停过于频繁 | 中 | 低 | **观察** |
| 14 | 15 | cosine decay 过早衰减学习率 | 低 | 低 | **观察** |
| 15 | 16 | reward clip 过大不生效 | 低 | 低 | **观察** |
| 16 | 14 | 对手池淘汰策略单一 | 低 | 低 | **观察** |
| 17 | 17 | downgrade_tower 惩罚过重 | 低 | 低 | **观察** |
| 18 | 18 | enemy 辅助标签权重与难度不匹配 | 低 | 低 | **观察** |

---

## 快速验证路线图

### 第一轮：确认紧急问题（1-2 小时）

1. **线索 4**：在 `compute_auxiliary_loss()` 中添加 `aux_valid_mask` 过滤，重跑 20 episode 对比 aux_loss
2. **线索 1**：打印一个 batch 的 returns 统计量，确认 VLoss 失衡程度
3. **线索 2**：统计 `selfplay_battle_log.jsonl` 中 reward 分布，确认双峰

### 第二轮：参数调整（2-4 小时）

4. **线索 1+2+6 联合**：对 returns 做 z-score 归一化 + 调整 vf_coef + 调整 clip_eps_vf
5. **线索 3**：将初始对手改为 BasicTowerAI 策略
6. **线索 10**：验证 own_base_dmg 标签语义

### 第三轮：深度优化（1-2 天）

7. **线索 5**：分离 value encoder 辅助损失的梯度路径
8. **线索 11**：对比有/无 logit_noise 的训练效果
9. **线索 7+8**：修正胜率判定和 GAE 截断处理
