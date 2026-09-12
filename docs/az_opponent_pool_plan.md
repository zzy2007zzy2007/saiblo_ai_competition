# 对手池自对弈计划（2026-09-12）

## 动机

六个杠杆全部试完，**唯一有效的是数据质量**：

| 杠杆 | 最好结果 | 超越基线 29.7~31.2% |
|------|---------|-------------------|
| **数据质量**（旧 160 局/28% 决定性 → 新 336 局/62%）| **3.1% → 31.2%** | ✅ 唯一有效（10 倍）|
| 数据量 | 25.0% | ❌ |
| 归一化（GN）| 6.2 / 18.8% | ❌ |
| 骨干解冻 | 12.5 / 18.8% | ❌ |
| 标签格式（tau-abs s1/2/4/6）| 9.4 / 21.9 / 12.5 / 9.4% | ❌ |
| 池化（5 种）| 18.8 ~ 40.6%（svs 平手）| ❌ |

**"数据质量"具体指什么？** 旧数据是**两个同等强度的搜索者互磨** → 绝大多数对局以
个位数血量差收场 → terminal 标签（血量差/20）几乎全挤在 0 附近 → 价值头学不到东西。
新数据（336 局、62% 决定性）也是自对弈，但**决定性明显更高**（原因见
`docs/az_data_quality_comparison.md`：随训练阶段上升）。

**所以瓶颈不是训练/架构，是"数据里有没有清晰的学习信号"。** 既然同等强度互磨产生
低信号数据，那就**主动制造强度不对称**——这正是对手池的作用。

## 假设

**learner 与不同强度的对手对局 → 对局一边倒 → 血量差大 → terminal 标签清晰 → 价值头
能学到东西。**

⚠️ 关键前提（必须校准）：**对手必须与 learner 强度接近**。

- 对手太弱 → learner 几乎全胜 → 标签又变成近常数（低信号，问题原样重现）→ 无效
- 对手太强（如未训价值头打 rule_v4）→ learner 几乎全败 → 同样近常数 → 无效
- **对手强度与 learner 相当但风格/水平略有差异 → 胜负参半且分差拉大** → 这才是我们想要的

所以池子的组成与"当前 learner 强度"强相关，**不是随便堆一堆 checkpoint**。

## 实验设计

### 池子组成（可配置）

| 成员 | 来源 | 作用 |
|------|------|------|
| **self** | 当前 learner 自身 | 保持同等强度 regime（价值校准到评测时的对手分布）|
| **past-k** | learner 的历史快照（如 `az_r10`, `az_p6`, `az_p21`）| 略弱/略强的近期对手 |
| **gen0120** | `training_history/az_fixed/gen0120_warm_cpp_gpu.pt`（ES 热启动）| 固定弱锚点（明显弱于当前 learner）|
| **rule_v4** | 规则 AI（`其他版本ai/rule_v4`）| 固定、风格不同的强对手（不走搜索）|

### 采样方案

- 每局开始：`u = rng.random()`；`u < p_self` → self-play，否则从 `{past-k, gen0120, rule_v4}`
  均匀采样一个对手。
- **默认 `p_self = 0.5`**：一半自对弈（保持强度校准 + 数据覆盖），一半异质对手
  （制造决定性）。先只扫 `p_self ∈ {0.5, 0.3}` 两个点，避免一次引入太多变量。
- **每局随机指定 learner 执先/后手**（P0/P1），消除颜色偏置；标签里 `s["player"]`
  已经按视角取符号，方案不受影响。

### 记录什么

- **只记录 learner 的决策**（`bundles / intent_counts / visit / recorded_*`）；
  对手的动作照常 `apply_operation_list` 施加，但**不进样本**（不能去模仿一个更弱
  策略的 visit 分布）。
- 每个 game 的 pickle 里加元信息：`opponent`（名字）、`learner_player`、`p_self`，
  方便训练时按对手分层分析。
- 价值标签**不变**：terminal 血量差 clip ±1（`HP_SCALE=20`）。

## 改动清单（实现阶段）

`code/my_ai/az_intent/az_selfplay.py`：

1. 新增 `OpponentPool` 类
   - `__init__(specs: list[dict], feat, *, iterations, max_depth_rounds, t_class, t_pos, k, sample_mult, c_puct)`
   - 每个 net 成员用现成的 `make_net_fn_from_ckpt(ckpt_path, feat)` 构建
     → 自动按各自 checkpoint 的 `num_heads/gn/value_pool` 建网（已有能力）
   - 规则成员包成 `choose_operations(state, player) -> list[Operation]`（照抄
     `eval.py` 里 `rule_v4` 的加载方式：`其他版本ai/rule_v4/ai.py: AI(seed=seed)`）
   - `.sample(rng) -> entry`，entry 含 `name / kind(net|rule) / net_fn / model`
2. `collect_game(..., opponent_pool=None, learner_player=0, p_self=0.5)`
   - `opponent_pool is None` → 完全走现有 self-play 路径（**向后兼容，默认不变**）
   - 有池子时：开局采样对手 + 建该对手的 `BundleMCTS`；决策循环里
     `player == learner_player` 才记录样本，否则只施加动作
3. `_collect_and_save` / `collect_games_parallel` 透传 `--opponent-pool` 相关参数；
   game pickle 增字段 `opponent/learner_player/p_self`
4. `main()` 增参数：
   - `--opponent-pool <a.pt,b.pt,...>`（逗号分隔；名字含 `rule_v4` 时按规则处理）
   - `--p-self`（默认 0.5）
   - `--pool-past-k` 一类便捷写法

## 复现与兼容

- **默认关闭**（`opponent_pool=None`）→ 现有 `az_selfplay.py` 行为逐位不变，
  已有 336 局数据可复现。
- 池子组成、`p_self`、采样种子写进 `progress` 日志与 pickle 元信息。
- 用 `bash code/run_logged.sh` 记录采集 + 训练，跑完回填 result。

## 实验步骤

1. 实现 + 冒烟（2 局，确认对手切换、样本只记 learner、元信息正确）
2. 采集 **336 局**（与基线同量，**唯一变量 = 对手多样性**），`p_self=0.5`
3. 训练价值头：**完全复用最佳基线配方**（冻结骨干 + no_bn + terminal 标签 + 同 lr/epochs）
4. 评测：
   - **主判据**：search-vs-search 64 局，对手 = 基线 `mix_r10p_vw_pol_frozen.pt`
     （灵敏判据，策略锁死、配对、方差小）
   - **副判据**：vs rule_v4 64 局（只作参考，小局数不可信）
   - **诊断**：采集数据的决定性率、价值标签分布、held-out 价值 MSE

## 判读

- svs > **60%** → 对手池**成立**，继续加大池子多样性 / 扫 `p_self` / 上迭代训练
- svs ≈ 50% → 对手池**不成立**，价值信号瓶颈在别处（可能要重估"决定性高"是否真
  是数据质量的关键变量，而非伴随相关）
- 决定性率没升 → 池子强度没校准好（离 learner 太远），先看分布再调组成

## 风险

- **强度校准是核心难点**：需要按当前 learner 的实测胜率动态维护池子；一次性配好
  的池子会随 learner 变强而失配（gen0120 会越来越弱）。第一版先固定池子 + 事后
  诊断，别一上来就做复杂的 Elo 调度。
- **off-policy 价值**：价值目标是在"对特定对手"的局面下测的，评测时对手是 rule_v4，
  可能不迁移。缓解：保留 50% self-play 让价值校准到自对弈分布。
- **样本效率**：一半的局面（对手回合）不产生样本 → 同样局数下样本减半。第一版把
  局数补到与基线同量（336），保证变量单一。
