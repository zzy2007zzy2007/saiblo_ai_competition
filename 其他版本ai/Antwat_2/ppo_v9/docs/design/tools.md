# ppo_v9 遗传算法训练运维工具设计文档

---

## 1. 概述

### 1.1 背景与目的

ppo_v2 拥有一套成熟的远程运维工具（`tools/manager.sh` + `tools/monitor_training.sh`），通过 SSH 连接远程训练服务器，封装了训练管理、文件传输、资源监控等 20+ 个命令。ppo_v9 目前没有任何运维工具，运维人员需要手动 SSH 到服务器查看日志文件，效率低下且无法做结构化分析。

ppo_v9 与 ppo_v2 本质上是完全不同的训练范式：

| 维度 | ppo_v2（PPO 强化学习） | ppo_v9（遗传算法进化） |
|------|----------------------|----------------------|
| **训练方式** | 梯度反向传播更新网络权重 | 种群 → 杂交 → 变异 → ELO 选择 |
| **进度单位** | 轮次（episode），连续累积 | 代次（generation），离散迭代 |
| **核心健康指标** | policy_loss、value_loss、entropy、grad_norm | best_elo、mean_elo、diversity（权重欧氏距离） |
| **评估方式** | SelfPlay + 定期 baseline 对战 | 每代全量 ELO 评估 + 定期基线验证 |
| **对手管理** | 对手池（Snapshot-based） | 对手池 + TrueSkill 评分 + SharedPayoff |
| **数据文件** | JSONL（逐行累积） | 每代独立 JSON 文件 |
| **模型保存** | 定期保存完整网络权重 | 每代保存种子模型 + 定期 checkpoint |
| **收敛判定** | 奖励曲线趋稳 | 最近 N 代 best_elo 波动 < 阈值 |

因此不能直接复用 ppo_v2 的监控脚本，需要针对 GA 特性重新设计。

### 1.2 设计目标

1. **与现有 manager.sh 集成**：扩展 `tools/manager.sh`，新增 `--model ppo_v9` 模式的专属命令，复用 SSH 连接、服务器配置、文件传输等基础设施。
2. **GA 特化监控**：覆盖代次摘要、种群统计、ELO 趋势、多样性、基线验证、对手池状态、异常检测。
3. **一键 + 按需**：支持 `ga_report` 一键查看全面概览，也支持单独查询某一维度。
4. **可扩展**：模块化设计，后续可轻松添加新的监控项。

### 1.3 调用方式

所有命令通过 `tools/manager.sh` 统一入口调用：

```bash
bash tools/manager.sh --server <server_name> --model ppo_v9 <command> [options]
```

---

## 2. 数据源

### 2.1 训练产出文件布局

ppo_v9 训练由 `ppo_v9/train_ga.py` 入口脚本启动。Run ID 格式为 `ga_run_YYYYMMDD_HHMMSS`，输出目录位于 `<project_root>/outputs/ga_run_<timestamp>/`（由 `train_ga.py` 第 61-62 行动态设置：`config["_output_dir"] = str(path_config.get_run_dir(run_id))`）。

```
outputs/ga_run_YYYYMMDD_HHMMSS/
├── training/                              # 日志文件
│   ├── ga_training_{time}.log             # 主日志（loguru，rotation="100 MB"）
│   ├── ga_training_error_{time}.log       # 错误日志（loguru，rotation="50 MB"）
│   └── training_complete.log              # 训练完成标记（纯文本：Final generation + Total time）
│
├── generations/                           # 每代数据目录
│   ├── gen_0001/
│   │   ├── population_stats.json          # 种群统计摘要
│   │   ├── battle_results.json            # 对战排名表 + 种子名单
│   │   ├── evaluation.json                # 基线验证结果（仅 baseline_battle.enabled=true 时）
│   │   ├── battle_log.jsonl               # 详细对战日志（每条一行 JSON）
│   │   └── seed_models/
│   │       ├── seed_0.pt                  # 排名 #1 种子的完整 state_dict
│   │       ├── seed_1.pt                  # 排名 #2
│   │       └── ...
│   ├── gen_0002/
│   │   └── ...
│   └── ...
│
├── checkpoints/                           # 完整训练状态检查点
│   ├── checkpoint_gen_0010.pt             # 每 save_interval 保存（默认 10 代）
│   └── ...
│
├── league/                                # 对手池持久化状态
│   ├── pool.json                          # 对手池成员 + checkpoint 路径 + 对战场次
│   └── payoff.json                        # TrueSkill 评分矩阵
│
└── tensorboard/                           # TensorBoard 事件文件（logging.tensorboard=true 时）
    └── events.out.tfevents.*
```

### 2.2 各数据文件详解

#### 2.2.1 `population_stats.json`（每代一个）

**来源**：`logging_subsystem.py:_save_population_stats()`（第 100-120 行）

```json
{
  "generation": 42,
  "population_size": 80,
  "best_elo": 1350.5,
  "mean_elo": 1210.2,
  "diversity": 15.73,
  "num_seeds": 8,
  "seed_elo_range": [1205.0, 1350.5]
}
```

**字段说明**：

| 字段 | 类型 | 说明 | 典型值 |
|------|------|------|--------|
| `generation` | int | 代次编号 | 1-200 |
| `population_size` | int | 种群个体数 | 80（默认） |
| `best_elo` | float | 本代最高 ELO | 初始 ~1200，进化后 1300-1800+ |
| `mean_elo` | float | 本代平均 ELO | 初始 ~1200 |
| `diversity` | float | 种群多样性（随机采样 100 对权重向量的平均欧氏距离） | 初始 ~18，随进化下降 |
| `num_seeds` | int | 选拔出的种子数 | 8（默认） |
| `seed_elo_range` | [float, float] | 种子 ELO 范围 [最低, 最高] | — |

**解析方式**：标准 `json.load()`。

#### 2.2.2 `battle_results.json`（每代一个）

**来源**：`logging_subsystem.py:_save_battle_results()`（第 122-147 行）

```json
{
  "generation": 42,
  "rankings": [
    {"id": "gen_042_ind_007", "elo": 1350.5, "seed_rank": 1},
    {"id": "gen_042_ind_023", "elo": 1342.1, "seed_rank": 2},
    ...
  ],
  "seeds": [
    {"id": "gen_042_ind_007", "elo": 1350.5, "rank": 1,
     "parents": ["gen_041_ind_003", "gen_041_ind_012"]},
    ...
  ]
}
```

**字段说明**：
- `rankings`：全体个体的完整 ELO 排名（按 ELO 降序），含 `seed_rank`（-1 表示非种子）
- `seeds`：选拔出的种子列表，含 `parents` 字段记录亲本 ID，可用于谱系追踪

**解析方式**：标准 `json.load()`。

#### 2.2.3 `evaluation.json`（每代一个，仅 baseline_battle.enabled=true 时）

**来源**：`logging_subsystem.py:_save_evaluation_results()`（第 149-156 行），数据由 `BaselineEvaluator.evaluate_seeds()` → `aggregate_results()` 生成。

```json
{
  "gen_042_ind_007": {
    "BasicRandomAI": {
      "total_battles": 12,
      "completed_battles": 12,
      "error_battles": 0,
      "agent1_wins": 10,
      "agent2_wins": 1,
      "draws": 1,
      "agent1_first_move_wins": 5,
      "agent1_first_move_battles": 6,
      "agent1_second_move_wins": 5,
      "agent1_second_move_battles": 6,
      "avg_rounds": 245.3,
      "total_duration": 180.5
    },
    "BasicTowerAI": { ... },
    "MediumRuleAI": { ... }
  },
  "gen_042_ind_023": { ... }
}
```

**字段说明**：
- 按键为种子 ID，值为该种子对各 baseline agent 的对战统计
- `error_battles`：可用于检测对战异常
- `agent1_first_move_wins` / `agent1_second_move_wins`：先手/后手胜率，可分析是否存在先后手偏差
- `agent1` 为我们的种子，`agent2` 为 baseline

**解析方式**：标准 `json.load()`。

#### 2.2.4 `league/pool.json`（对手池状态）

**来源**：`OpponentPool.save_state()`（`opponent_pool.py`）

```json
{
  "opponents": ["gen_001_ind_003", "gen_002_ind_001", ...],
  "checkpoint_paths": {
    "gen_001_ind_003": "outputs/.../generations/gen_0001/seed_models/seed_0.pt",
    ...
  },
  "games_played": {
    "gen_001_ind_003": 45,
    ...
  },
  "added_episodes": {
    "gen_001_ind_003": 1,
    ...
  }
}
```

**字段说明**：
- `opponents`：池中对手 ID 列表，长度反映当前池大小与 `max_size` 的关系
- `checkpoint_paths`：每个对手对应的模型文件路径
- `games_played`：每个对手被选中作为动态参照物的累计场次
- `added_episodes`：对手加入时的代次

**解析方式**：标准 `json.load()`。

#### 2.2.5 `league/payoff.json`（TrueSkill 评分矩阵）

**来源**：`BattleSharedPayoff.save_state()`（`battle_shared_payoff.py`）

```json
{
  "players": ["gen_001_ind_003", "gen_002_ind_001", ...],
  "players_ids": {"gen_001_ind_003": 0, ...},
  "data": {
    "gen_001_ind_003-gen_002_ind_001": {
      "wins": 8, "draws": 2, "losses": 5, "games": 15
    },
    ...
  },
  "trueskill_ratings": {
    "gen_001_ind_003": {"mu": 27.5, "sigma": 3.2, "games_played": 45},
    ...
  }
}
```

**字段说明**：
- `players`：所有参与评分的玩家 ID
- `data`：任意两玩家间的对战记录
- `trueskill_ratings`：TrueSkill 评分，`mu` 为技能估计值，`sigma` 为不确定性

**解析方式**：标准 `json.load()`。

#### 2.2.6 `checkpoint_gen_XXXX.pt`（完整训练状态）

**来源**：`GACheckpointManager.save()`（`checkpoint_manager.py` 第 17-68 行），PyTorch `.pt` 格式。

```python
{
  "generation": 10,
  "population_data": [
    {
      "id": "gen_010_ind_000",
      "genome_weights": [...],       # numpy 数组（完整权重向量）
      "genome_shapes": [...],        # 权重张量形状
      "genome_param_names": [...],
      "elo_rating": 1320.0,
      "seed_rank": 1,
      "parent_ids": [...],
      "mutation_desc": "elite_copy"
    },
    ...
  ],
  "population_stats": {"best_elo": 1320.0, "mean_elo": 1195.3, "diversity": 14.2},
  "config": { ... },
  "league_dir": "outputs/..."
}
```

**注意**：checkpoint 文件体积较大（80 个体 × 完整权重向量），**不参与常规监控读取**——仅在 `ga_resume` 时需要读取 `generation` 和 `population_stats` 字段。

#### 2.2.7 日志文件（`ga_training_{time}.log`）

**来源**：`GALoggingSubsystem._setup_loguru()`（`logging_subsystem.py` 第 18-48 行），loguru 格式。

**格式**：
```
YYYY-MM-DD HH:mm:ss | LEVEL    | module:function:line - message
```

**可用于 grep 的关键日志行模式**：

| 模式 | 来源 | 说明 |
|------|------|------|
| `GA Training started: max_generations=N` | `genetic_evolution_trainer.py:95` | 训练启动 |
| `Generation N: best_elo=X, mean_elo=Y, diversity=Z, duration=T` | `logging_subsystem.py:60-66` | 每代控制台摘要 |
| `Seed ID vs baseline: win_rate=X%, (W/L/D)` | `baseline_evaluator.py:118-124` | 基线对战结果 |
| `Opponent ID added to pool (size: N)` | `opponent_pool.py` | 对手池新增 |
| `Checkpoint saved to PATH at generation N` | `checkpoint_manager.py:68` | checkpoint 保存 |
| `Converged at generation N` | `genetic_evolution_trainer.py:158` | 收敛触发 |
| `GA Training complete: N generations, total_time=T` | `logging_subsystem.py:84-87` | 训练完成 |

#### 2.2.8 TensorBoard 指标

**来源**：`GALoggingSubsystem._write_tensorboard()`（`logging_subsystem.py` 第 158-178 行）。

| 指标名称 | 含义 | 来源 |
|----------|------|------|
| `ga/best_elo` | 当前代最高 ELO | `population.best_elo` |
| `ga/mean_elo` | 当前代平均 ELO | `population.mean_elo` |
| `ga/diversity` | 种群多样性 | `population.diversity` |
| `ga/baseline_win_rate/{baseline_name}` | 各种子对某 baseline 的平均胜率 | 基线验证结果聚合 |

**注意**：当前实现每代创建新的 `SummaryWriter` 并立即 `close()`，因此 TensorBoard 非增量写入，UI 需要手动 reload 才能刷新数据。

---

## 3. 核心监控命令

### 3.1 `ga_status` — 训练状态检查

**功能**：快速判断 GA 训练是否在运行，显示当前进度。

**数据源**：
- 远程进程检查：`screen -ls | grep ga_ppo_v9` 或 `ps aux | grep train_ga.py`
- `training_complete.log` 是否存在 → "已完成"
- 最新代次的 `population_stats.json` → 当前代次与最新统计

**参数**：
- `-v, --verbose`：详细模式，额外输出最近 3 行关键日志

**输出示例**：
```
训练状态: running
当前代次: 42 / 200 (21.0%)
最新统计数据:
  Best ELO: 1350.5 | Mean ELO: 1210.2 | Diversity: 15.73
  种子数: 8 | 种子 ELO 范围: [1205.0, 1350.5]
```

**实现要点**：
1. 通过 `get_latest_ga_run_dir()` 定位最新输出目录
2. 检查 `screen -ls` 或 `ps aux` 判断进程存活
3. 读取最新 `population_stats.json` 获取当前代统计
4. 如在 `-v` 模式，从日志中 `grep 'Generation'` 取最近 3 条

---

### 3.2 `ga_progress` — 综合进度检查

**功能**：全面的训练进度概览，包括代次级 ELO 趋势和预估剩余时间。

**数据源**：
- 遍历 `generations/gen_*/population_stats.json`（最近 N 代）
- 配置中的 `max_generations`
- `training/training_complete.log` 或首代时间戳

**参数**：
- `-g N`：查看最近 N 代（默认 10）

**输出示例**：
```
========================================
  ppo_v9 GA 训练进度
========================================
服务器: server9    状态: running
输出目录: outputs/ga_run_20260612_143052

--- 进度 ---
当前代: 42 / 200 (21.0%)
已用时间: 3h 25m    预估剩余: 12h 52m
收敛窗口 (20): best_elo 波动 12.5 > 阈值 5.0 — 未收敛

--- ELO 趋势 (最近 10 代) ---
 Gen | Best ELO | Mean ELO | ΔBest | Diversity | 耗时
-----|----------|----------|-------|-----------|------
  42 |  1350.5  |  1210.2  | +12.3 |   15.73   | 4m32s
  41 |  1338.2  |  1207.1  |  +8.1 |   16.58   | 4m28s
  40 |  1330.1  |  1205.6  | +15.2 |   16.16   | 4m35s
  ...

--- 多样性趋势 ---
 Gen | Diversity | ΔDiv  | 状态
-----|-----------|-------|------
  42 |   15.73   | -0.85 | ⚠️ 下降
  41 |   16.58   | +0.42 |
  ...
```

**实现要点**：
1. 远程聚合脚本遍历 `generations/gen_*/population_stats.json`，收集最近 N 代的 `generation`、`best_elo`、`mean_elo`、`diversity`
2. 从日志中解析每代耗时（`duration=` 字段），或通过相邻代次时间戳推算
3. 剩余时间 = (已用时间 / 已完成代数) × 剩余代数
4. 收敛判断：最近 `convergence_window` 代 best_elo 波动 < `convergence_threshold`

---

### 3.3 `ga_population` — 种群统计

**功能**：查看当前代的种群详细信息，包括 ELO 分布和种子排名。

**数据源**：
- 最新 `population_stats.json` + `battle_results.json`
- 可选指定代次：`ga_population <gen_id>`

**输出示例**：
```
========================================
  种群统计 — Generation 42
========================================
种群规模: 80 | 种子数: 8
Best ELO: 1350.5 | Mean ELO: 1210.2 | Diversity: 15.73

--- 种子排名 ---
 Rank | ID               | ELO    | 亲本
------|-------------------|--------|-------------------
  1   | gen_042_ind_007   | 1350.5 | gen_041_ind_003, gen_041_ind_012
  2   | gen_042_ind_023   | 1342.1 | gen_041_ind_003, gen_041_ind_017
  3   | gen_042_ind_045   | 1335.8 | gen_041_ind_012, gen_041_ind_028
  ...

--- ELO 分布 ---
1300+  ████████████████████  8  (10.0%)
1250+  ████████████████████████████████████████  16  (20.0%)
1200+  ████████████████████████████████████████████████████████████████████████████████  34  (42.5%)
1150+  ██████████████████████████████████████████████████████  22  (27.5%)
```

**实现要点**：
1. 读取指定代次的 `population_stats.json` 和 `battle_results.json`
2. 从 `rankings` 中提取完整 ELO 分布，按区间统计制作文本直方图
3. 对于种子，从 `seeds[].parents` 提取亲本信息

---

### 3.4 `ga_baseline` — 基线评估结果

**功能**：查看种子 vs 各 baseline agent 的对战胜率，支持横向趋势对比。

**数据源**：
- `generations/gen_*/evaluation.json`

**参数**：
- `-g N`：最近 N 代（默认 5）

**输出示例**：
```
========================================
  基线验证结果
========================================

--- 最新一代 (Gen 42) ---
Baseline      |  胜  |  负  |  平  | 场次 | 胜率
--------------|------|------|------|------|------
BasicRandomAI |  10  |   1  |   1  |  12  | 83.3%
BasicTowerAI  |   8  |   2  |   2  |  12  | 66.7%
MediumRuleAI  |   7  |   4  |   1  |  12  | 58.3%

--- 胜率趋势 (最近 5 代) ---
Baseline      | Gen38 | Gen39 | Gen40 | Gen41 | Gen42 | 趋势
--------------|-------|-------|-------|-------|-------|------
BasicRandomAI |  85%  |  88%  |  91%  |  93%  |  95%  |  ↗
BasicTowerAI  |  62%  |  65%  |  68%  |  67%  |  71%  |  ↗
MediumRuleAI  |  45%  |  48%  |  52%  |  50%  |  55%  |  ↗
```

**实现要点**：
1. 读取最新一代的 `evaluation.json`，按 seed 聚合各 baseline 的胜率（取种子的均值或最佳种子的值）
2. 对于趋势表，遍历最近 N 代的 `evaluation.json`，取每代 best seed 的胜率
3. 胜率趋势箭头判断：首尾代比较

---

### 3.5 `ga_opponents` — 对手池状态

**功能**：查看对手池规模、成分、评分和对战场次分布。

**数据源**：
- `league/pool.json` + `league/payoff.json`

**输出示例**：
```
========================================
  对手池状态
========================================
池大小: 12 / 20 (60%)
活跃对手 (games >= 8): 8 / 12 (66.7%)

--- 对手列表 ---
Opponent             |  mu   | sigma | Games | Added Gen
---------------------|-------|-------|-------|----------
gen_042_ind_007      | 28.5  |  3.1  |  45   |   42
gen_041_ind_003      | 26.2  |  3.8  |  38   |   41
gen_040_ind_012      | 24.1  |  4.5  |  32   |   40
gen_039_ind_005      | 22.8  |  5.2  |  28   |   39
...

--- 来源分布 ---
Gen 42: 1 | Gen 41: 1 | Gen 40: 1 | Gen 39: 1 | Gen 38: 1
Gen 37: 1 | Gen 36: 1 | Gen 35: 1 | Gen 34: 1 | Gen 33: 1
Gen 32: 1 | Gen 31: 1

--- 对战场次分布 ---
 0- 9: ██████████ 4
10-19: ██████████ 4
20-29: ██████████ 2
30-39: ██████████ 2
40-49: ██████████ 1
```

**实现要点**：
1. 合并读取 `pool.json` 和 `payoff.json`
2. 关联每个 opponent 的 TrueSkill mu/sigma（从 `payoff.json.trueskill_ratings`）
3. 关联 `added_episodes` 获取来源代次
4. 统计 `games_played` 分布制作直方图

---

### 3.6 `ga_scan` — 异常扫描

**功能**：扫描以下四类异常并分级报告（CRITICAL / WARN / INFO）。

#### 3.6.1 ELO 停滞检测

**判定逻辑**：
```
输入: 最近 W 代的 best_elo 序列
参数: W = convergence_window (默认 20), 阈值 = convergence_threshold (默认 5.0)
规则: max(best_elo序列) - min(best_elo序列) < 阈值
```

| 条件 | 严重度 | 消息 |
|------|--------|------|
| 波动 < 阈值 且 best_elo < 1300 | CRITICAL | "ELO 严重停滞：种群未明显超越初始水平" |
| 波动 < 阈值 且 best_elo >= 1300 | WARN | "ELO 停滞：最近 W 代无明显提升" |
| 波动 >= 阈值 | — | 无异常 |

#### 3.6.2 种群同质化检测

**判定逻辑**：
```
输入: 当前代 diversity, 初始代 (gen 1) diversity
规则: diversity_current / diversity_initial < 0.30
```

| 条件 | 严重度 | 消息 |
|------|--------|------|
| ratio < 0.15 | CRITICAL | "严重同质化：多样性仅为初始的 {ratio:.0%}" |
| ratio < 0.30 | WARN | "种群同质化：多样性降至初始的 {ratio:.0%}" |
| ratio >= 0.30 | — | 无异常 |

**注意**：diversity 值基于随机采样的权重欧氏距离平均值，初始值约 15-20。随着进化，diversity 自然下降是正常现象——30% 阈值用于捕捉过快的坍塌。

#### 3.6.3 对战异常率检测

**判定逻辑**：
```
输入: 最近 N 代所有 evaluation.json
计算: error_rate = sum(error_battles) / sum(total_battles)
```

| 条件 | 严重度 | 消息 |
|------|--------|------|
| error_rate > 10% | CRITICAL | "对战异常率极高：{rate:.1%}" |
| error_rate > 5% | WARN | "对战异常率偏高：{rate:.1%}" |
| error_rate <= 5% | — | 无异常 |

**补充检查**（当 error_battles > 0 时）：
- evaluation.json 中 `agent2_wins` 对弱 baseline 异常高 → 可能是模型推理崩溃
- 日志中是否存在 "CUDA out of memory" → 资源不足
- `avg_rounds` 异常短（< 10 轮）→ 环境初始化失败

#### 3.6.4 NaN 检测

**判定逻辑**：
扫描最新的日志文件和 JSON 文件中的 NaN 关键字。

搜索模式：
```bash
grep -i "nan\|inf\|NaN\|Inf" <latest_log>
```

| 条件 | 严重度 | 消息 |
|------|--------|------|
| 发现 NaN 且出现在核心指标上下文中 | CRITICAL | "NaN 出现在核心指标中，训练可能已崩溃" |
| 发现 NaN 但仅在警告/调试上下文中 | WARN | "日志中发现 NaN 值，需人工判断" |
| 未发现 NaN | — | 无异常 |

**JSON 文件检查**：虽然标准 `json.load()` 不支持 NaN，但 `json.dumps(stats, default=str)` 会将 NaN 转为字符串 `"nan"`——需要在 JSON 解析后检查关键字段（`best_elo`、`mean_elo`）是否为有限值。

**输出示例**：
```
========================================
  异常扫描结果
========================================
CRITICAL=0 | WARN=1 | INFO=2

🟡 [WARN]     ELO停滞: 最近 20 代 best_elo 波动 3.2 < 阈值 5.0
ℹ️  [INFO]    多样性正常: 当前 15.73, 初始 19.2, 比率 81.9%
ℹ️  [INFO]    对战正常: 异常率 0.6% (1/144)
✅ NaN 检测:  无异常
```

**实现要点**：
1. 实现为 `tools/ppo_v9/lib/ga_anomaly_scanner.sh`，在远程端执行 Python 扫描逻辑
2. 上传扫描脚本到远程执行（类似 ppo_v2 的 `_anomaly_scanner.py` 模式）
3. 返回结构化 JSON：`[{severity, type, msg}]`

---

### 3.7 `ga_report` — 完整代次报告

**功能**：一键查看以上所有核心监控维度的综合报告。相当于按顺序执行 `ga_progress` + `ga_population` + `ga_baseline` + `ga_opponents` + `ga_scan`。

**参数**：
- `-g N`：各维度使用最近 N 代数据（默认 5）

**输出布局**：
```
========================================
  ppo_v9 GA 完整代次报告
========================================
服务器: server9    状态: running
输出目录: outputs/ga_run_20260612_143052

[1/5] 训练进度
...

[2/5] 种群统计 (Gen 42)
...

[3/5] 基线验证
...

[4/5] 对手池状态
...

[5/5] 异常扫描
...
========================================
  报告生成时间: 2026-06-12 15:42:05
========================================
```

**实现要点**：
- 在远程端执行一个聚合脚本，一次性读取所有需要的数据，减少 SSH 往返
- 本地用 Python 格式化脚本统一排版
- 同时提供一个 `monitor_ga.sh` 一键脚本（类似 ppo_v2 的 `monitor_training.sh`），封装为：
  ```bash
  bash tools/ppo_v9/monitor_ga.sh
  ```

---

## 4. 系统监控命令

### 4.1 `ga_gpu` — GPU 监控

**功能**：查看远程服务器 GPU 利用率和显存使用情况。

**数据源**：远程执行 `nvidia-smi`。

**参数**：
- `--watch`：持续监控模式（每 2 秒刷新）

**输出示例**：
```
========================================
  GPU 状态
========================================
GPU 0: NVIDIA RTX 4090
  利用率: 95% | 温度: 72°C | 功耗: 320W / 450W
  显存: 18.2GB / 24.0GB (75.8%)
```

**实现**：复用 ppo_v2 的 `tools/monitor/check_gpu.sh` 逻辑，只需替换命令名和参数传递方式。

### 4.2 `ga_system` — 系统资源监控

**功能**：查看远程服务器 CPU、RAM、磁盘使用情况。

**数据源**：远程执行 `top`、`free -h`、`df -h`。

**参数**：
- `--watch`：持续监控模式

**输出示例**：
```
========================================
  系统资源
========================================
CPU: 45.2% | RAM: 32.1GB / 64.0GB (50.1%)
磁盘: /root/autodl-tmp 使用 245GB / 500GB (49.0%)
```

**实现**：复用 ppo_v2 的 `tools/monitor/check_system.sh` 逻辑。

### 4.3 `ga_tensorboard` — 启动 TensorBoard

**功能**：在本地启动 TensorBoard，指向远程训练目录的 tensorboard 日志（通过 SSH 端口转发或直接下载后本地启用）。

**参数**：
- `--port <port>`：本地端口（默认 6006）

**实现方式**：
1. 通过 SSH 端口转发：`ssh -L <local_port>:localhost:6006 <server> "tensorboard --logdir=<tb_dir> --port 6006"`
2. 或者先下载 `tensorboard/` 目录到本地，再本地启动

**输出**：
```
TensorBoard 已启动：http://localhost:6006
```

---

## 5. 运维操作命令

### 5.1 `ga_start` — 启动训练

**功能**：在远程服务器上通过 screen 会话启动 GA 训练。

**参数**：
- `--config <path>`：YAML 配置文件路径（默认 `configs/ga_evo.yaml`）
- `--population-size <num>`：种群大小
- `--max-generations <num>`：最大代数
- `--num-seeds <num>`：种子数
- `--mutation-scale <num>`：变异幅度
- `--mutation-rate <num>`：变异率
- `--crossover-rate <num>`：杂交率
- `--device <device>`：设备（cuda/cpu）

**实现**：
```bash
# screen 会话命名规则
SCREEN_SESSION="ga_ppo_v9"

# 构建的训练命令
TRAIN_CMD="screen -dmS $SCREEN_SESSION bash -c '
    export PATH=/root/miniconda3/bin:\$PATH
    export PYTHONPATH=<project_dir>/ppo_v9/src:<project_dir>/Ant-Game/SDK/python
    cd <project_dir>/ppo_v9
    python3 train_ga.py --config <config_path> [其他参数]
'"
```

**注意事项**：
- 启动前检查是否已有训练在运行（`screen -ls | grep ga_ppo_v9`）
- ppo_v9 的 Python 入口为 `ppo_v9/train_ga.py`，非 ppo_v2 的 `train.py`
- PYTHONPATH 需包含 `ppo_v9/src` 和 `Ant-Game/SDK/python`

### 5.2 `ga_stop` — 停止训练

**功能**：优雅停止远程 GA 训练（先 SIGTERM，再 SIGKILL）。

**实现**：
```bash
screen -S ga_ppo_v9 -X quit
# 或通过进程树
pkill -f train_ga.py
```

**注意事项**：
- 训练进程会在当前代完成后检查收敛条件并自然退出，或通过 `screen -X quit` 终止
- 停止前应确认当前代的数据已写入磁盘（异步写入可能延迟）

### 5.3 `ga_resume` — 从检查点恢复训练

**功能**：从指定的 checkpoint 恢复训练。

**参数**：
- `--checkpoint <path>`：checkpoint 文件路径（必填）
- 其他参数同 `ga_start`

**实现**：
```bash
python3 train_ga.py --resume <checkpoint_path> [其他参数]
```

**注意**：ppo_v9 的 `train_ga.py` 中 `--resume` 参数已定义（第 21 行），但当前标注了 `# TODO: 实现断点续训加载逻辑`——工具的 `ga_resume` 命令在底层 `train_ga.py` 的恢复逻辑实现完之前只是占位，当前实际上等效于从 checkpoint 提取配置后重新开始训练。

**checkpoint 信息预览**：在恢复前提供 checkpoint 的概要信息：
```
Checkpoint: checkpoint_gen_0042.pt
  Generation: 42
  Best ELO: 1350.5
  Mean ELO: 1210.2
  Population: 80 individuals
```

### 5.4 `ga_deploy` — 部署代码

**功能**：将本地 ppo_v9 代码上传到远程服务器。

**实现**：复用 ppo_v2 的 `tools/deploy/deploy_code.sh` 模式，但源码路径改为 `ppo_v9/`。

**需要上传的内容**：
- `ppo_v9/src/ppo_ga/` — 核心 Python 包
- `ppo_v9/train_ga.py` — 训练入口
- `ppo_v9/configs/` — 配置文件
- `ppo_v9/requirements.txt` — 依赖

**传输方式**：`rsync`（增量同步）或 `tar + scp`。

```bash
# 使用 rsync 同步
rsync -avz --delete \
    <local_ppo_v9>/src/ \
    <server>:<project_dir>/ppo_v9/src/

# 或者打包传输（更可靠）
tar czf /tmp/ppo_v9_src.tar.gz -C <project_dir>/ppo_v9 src train_ga.py configs requirements.txt
scp /tmp/ppo_v9_src.tar.gz <server>:<project_dir>/ppo_v9/
ssh <server> "cd <project_dir>/ppo_v9 && tar xzf ppo_v9_src.tar.gz"
```

### 5.5 `ga_exec` — 执行远程命令

**功能**：在远程服务器的 ppo_v9 环境中执行任意命令。

**用法**：
```bash
bash tools/manager.sh --server server9 --model ppo_v9 ga_exec "ls -la outputs/ga_run_20260612_143052/generations/"
```

**实现**：直接复用 `manager.sh` 的 `exec` 命令逻辑，但自动设置 PYTHONPATH 环境变量。

### 5.6 `ga_logs` — 查看训练日志

**功能**：实时查看或尾部查看远程训练日志。

**参数**：
- `-f, --follow`：实时跟踪（tail -f）
- `-n <num>`：显示最近 N 行（默认 50）
- `--error`：查看错误日志而非主日志

**实现**：
```bash
# 主日志
ssh <server> "tail -n $N -f <latest_run>/training/ga_training_*.log"

# 错误日志
ssh <server> "tail -n $N -f <latest_run>/training/ga_training_error_*.log"
```

### 5.7 `ga_download` — 下载训练结果

**功能**：从远程服务器下载指定代次或全部训练结果到本地。

**参数**：
- `--gen <id>`：下载指定代次目录（如 `42`）
- `--latest`：仅下载最新代次
- `--all`：下载全部结果
- `--output <path>`：本地保存路径

**实现**：
```bash
# 下载最新代次结果
scp -r <server>:<run_dir>/generations/gen_<latest> <local_path>/

# 下载完整 run 目录
scp -r <server>:<run_dir> <local_path>/
```

**注意事项**：`seed_models/` 中的 `.pt` 文件体积较大，建议默认排除模型文件，用 `--include-models` 显式包含。

---

## 6. 架构设计

### 6.1 与现有 manager.sh 的集成方案

**设计决策：扩展 `manager.sh` 而非创建独立的 `ga_manager.sh`。**

**理由**：
1. 复用 SSH 基础设施（`execute_remote_command`、`upload_file_ssh`、`download_file_ssh`）
2. 复用服务器配置（`SERVER/HOST/PORT/PASSWORD`，由 `config_helper.sh` + `server_config.sh` 管理）
3. 统一的 CLI 入口（运维人员只需记住一个命令格式）
4. `--model ppo_v9` 参数自然区分 ppo_v2 和 ppo_v9 的命令路由

**集成方式**：在 `manager.sh` 的 `case` 块中新增 ppo_v9 命令分支：

```bash
# manager.sh 中新增 case 分支：
case "$COMMAND" in
    # ... 现有 ppo_v2 命令：train_start, train_status, ... ...

    # ── ppo_v9 GA 命令 ──
    "ga_status"|"ga_progress"|"ga_population"|"ga_baseline"|\
    "ga_opponents"|"ga_scan"|"ga_report"|\
    "ga_gpu"|"ga_system"|"ga_tensorboard"|\
    "ga_start"|"ga_stop"|"ga_resume"|"ga_deploy"|"ga_exec"|"ga_logs"|"ga_download")
        source "${SCRIPT_DIR}/ppo_v9/lib/ga_config.sh"
        source "${SCRIPT_DIR}/ppo_v9/lib/ga_data_reader.sh"
        exec_ga_command "$SERVER" "$MODEL" "$COMMAND" "$@"
        ;;
esac
```

### 6.2 脚本目录结构

```
tools/
├── manager.sh                          # [扩展] 新增 ppo_v9 命令的 case 分支
├── monitor_training.sh                 # [保持] ppo_v2 一键监控脚本
│
├── ppo_v9/                             # [新建] ppo_v9 专用运维脚本目录
│   ├── lib/
│   │   ├── ga_config.sh               # GA 配置读取与目录定位
│   │   ├── ga_data_reader.sh          # 数据读取通用函数
│   │   └── ga_anomaly_scanner.sh      # 异常扫描脚本（远程执行）
│   │
│   ├── monitor/
│   │   ├── ga_remote_collect.sh       # 远程数据聚合脚本（SSH 端执行，减少往返）
│   │   ├── format_ga_status.py        # 训练状态格式化
│   │   ├── format_ga_progress.py      # 进度概览格式化
│   │   ├── format_ga_population.py    # 种群统计格式化
│   │   ├── format_ga_baseline.py      # 基线对战表格式化
│   │   ├── format_ga_opponents.py     # 对手池状态格式化
│   │   ├── format_ga_scan.py          # 异常扫描结果格式化
│   │   └── format_ga_report.py        # 完整报告格式化
│   │
│   └── monitor_ga.sh                  # [新建] 一键监控脚本（类似 monitor_training.sh）
│
├── lib/
│   ├── config_helper.sh               # [保持] 通用配置
│   ├── ssh_helper.sh                  # [保持] SSH 辅助
│   └── status_manager.sh              # [扩展] 新增 ppo_v9 辅助函数
│
├── deploy/
│   ├── ga_start_training.sh           # [新建] GA 启动训练
│   ├── ga_stop_training.sh            # [新建] GA 停止训练
│   ├── ga_deploy_code.sh              # [新建] GA 代码部署
│   └── ga_tensorboard.sh              # [新建] GA TensorBoard
│
└── config/
    └── server_config.sh               # [保持] 服务器连接信息
```

### 6.3 模块职责

#### 6.3.1 `tools/ppo_v9/lib/ga_config.sh` — 配置模块

**核心函数**：

```bash
# 获取最新 ppo_v9 输出目录
get_latest_ga_run_dir(server_name)
  → 远程执行: ls -td outputs/ga_run_* | head -1

# 获取当前代次编号
get_current_generation(server_name, run_dir)
  → 远程执行: ls -d generations/gen_*/ | sort | tail -1 | sed 's/.*gen_\([0-9]*\).*/\1/'

# 获取 GA 配置值（从远程 YAML 读取）
get_ga_config_value(server_name, run_dir, key_path)
  → 远程执行: python3 -c "import yaml; ..."

# 获取最近 N 代目录列表
get_generation_list(server_name, run_dir, count)
  → 远程执行: ls -d generations/gen_*/ | sort | tail -n $count
```

#### 6.3.2 `tools/ppo_v9/lib/ga_data_reader.sh` — 数据读取模块

**核心函数**：

```bash
# 读取单代 population_stats.json
read_population_stats(server_name, run_dir, gen)
  → 返回 JSON 字符串

# 读取单代 evaluation.json
read_evaluation(server_name, run_dir, gen)
  → 返回 JSON 字符串（文件不存在时返回 "{}"）

# 批量读取多代数据（通过远程聚合脚本）
read_multi_gen_data(server_name, run_dir, gen_list, fields)
  → 返回聚合 JSON
```

#### 6.3.3 `tools/ppo_v9/lib/ga_anomaly_scanner.sh` — 异常扫描模块

参照 ppo_v2 的 `tools/monitor/scan_anomalies.sh` 模式：上传 Python 扫描脚本到远程，执行后返回结构化 JSON，本地解析显示。

**扫描逻辑实现为 Python 脚本**（处理 JSON 文件更方便）：

```python
# ga_anomaly_scanner.py 核心逻辑框架
def scan(run_dir, last_n=20):
    anomalies = []
    
    # 1. ELO 停滞检测
    elos = read_elo_history(run_dir, last_n)
    if max(elos) - min(elos) < threshold:
        anomalies.append({"severity": "WARN", "type": "ELO停滞", ...})
    
    # 2. 同质化检测
    diversity = read_current_diversity(run_dir)
    initial_diversity = read_initial_diversity(run_dir)
    if diversity / initial_diversity < 0.30:
        anomalies.append({"severity": "WARN", "type": "种群同质化", ...})
    
    # 3. 对战异常率
    error_rate = calc_error_rate(run_dir, last_n)
    if error_rate > 0.05:
        anomalies.append({"severity": "WARN", "type": "对战异常率", ...})
    
    # 4. NaN 检测
    nan_found = check_nan_in_logs(run_dir)
    if nan_found:
        anomalies.append({"severity": "CRITICAL", "type": "NaN检测", ...})
    
    return anomalies
```

#### 6.3.4 远程聚合脚本 `tools/ppo_v9/monitor/ga_remote_collect.sh`

参照 ppo_v2 的 `tools/monitor/remote_collect.sh` 模式：在远程服务器执行数据聚合，减少 SSH 往返次数。

**工作流**：
```python
# 内嵌 Python 逻辑
import json, os, glob

def collect(run_dir, last_n, include_evaluation=True, include_league=True):
    gen_dirs = sorted(glob.glob(f"{run_dir}/generations/gen_*"))
    result = {"generations": {}}
    
    for gen_dir in gen_dirs[-last_n:]:
        gen_num = int(os.path.basename(gen_dir).split('_')[1])
        gen_data = {}
        
        # 读 population_stats.json
        with open(f"{gen_dir}/population_stats.json") as f:
            gen_data["population"] = json.load(f)
        
        # 读 evaluation.json（可选）
        if include_evaluation:
            eval_path = f"{gen_dir}/evaluation.json"
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    gen_data["evaluation"] = json.load(f)
        
        result["generations"][str(gen_num)] = gen_data
        result["latest_gen"] = gen_num
    
    # 读 league 状态
    if include_league:
        if os.path.exists(f"{run_dir}/league/pool.json"):
            with open(f"{run_dir}/league/pool.json") as f:
                result["pool"] = json.load(f)
        if os.path.exists(f"{run_dir}/league/payoff.json"):
            with open(f"{run_dir}/league/payoff.json") as f:
                result["payoff"] = json.load(f)
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    collect(sys.argv[1], int(sys.argv[2]))
```

### 6.4 远程执行模型

与 ppo_v2 一致，采用 SSH-based 架构：

```
本地 (manager.sh)                         远程服务器
┌──────────────┐                         ┌────────────────────┐
│ manager.sh   │── SSH upload ──────────→│ ga_remote_collect  │
│              │                         │     .sh            │
│  ppo_v9/lib/ │                         └────────┬───────────┘
│  ppo_v9/     │                                  │
│   monitor/   │←── SSH return JSON ──────────────┘
│              │
│  format_*.py │── 解析 JSON ──→ 格式化输出到终端
└──────────────┘
```

**数据流说明**：
1. `manager.sh` 收到命令 → 定位远程最新 `ga_run_*` 目录
2. 上传 `ga_remote_collect.sh` 到远程
3. 远程执行聚合脚本，遍历 JSON 文件，输出聚合 JSON
4. 本地将聚合 JSON 传给对应的 `format_*.py` 进行格式化
5. 输出到终端

### 6.5 性能注意事项

1. **减少 SSH 往返**：尽量用远程聚合脚本一次完成所有数据读取，避免逐代 SSH 查询
2. **大文件处理**：`checkpoint_gen_*.pt` 不参与监控读取（体积过大），仅在 `ga_resume` 时读取元信息
3. **日志解析**：仅在异常扫描时 grep 最近的日志文件，常规监控不读取日志
4. **pysize**：每代 JSON 文件总体积约 10-50KB，200 代总计约 2-10MB，可安全地在内存中处理

---

## 7. 实施计划

### 7.1 分阶段计划

#### 阶段一：核心基础设施（优先级最高，预估 1-2 天）

**目标**：打通远程数据访问链路，实现最小可用命令。

| 序号 | 任务 | 产出文件 | 说明 |
|------|------|---------|------|
| 1.1 | 创建目录结构 | `tools/ppo_v9/` 及子目录 | 按 6.2 节结构创建 |
| 1.2 | 实现 `ga_config.sh` | `tools/ppo_v9/lib/ga_config.sh` | 定位输出目录、获取当前代次、读取配置 |
| 1.3 | 实现 `ga_data_reader.sh` | `tools/ppo_v9/lib/ga_data_reader.sh` | 远程 JSON 读取函数 |
| 1.4 | 实现远程聚合脚本 | `tools/ppo_v9/monitor/ga_remote_collect.sh` | 远程数据聚合 |
| 1.5 | 扩展 `manager.sh` | `tools/manager.sh` | 新增 case 分支，集成 GA 命令路由 |
| 1.6 | 端到端验证 | — | SSH 连接 → 定位目录 → 读取 JSON → 返回 |

**交付物**：`manager.sh ga_status` 可执行，显示当前代次和运行状态。

#### 阶段二：基础监控命令（预估 2-3 天）

**目标**：实现 4 个常用监控视图。

| 序号 | 任务 | 产出文件 | 说明 |
|------|------|---------|------|
| 2.1 | `ga_status` 完整实现 | `format_ga_status.py` + `status_manager.sh` 扩展 | 进程检查 + 代次摘要 + `-v` 详细模式 |
| 2.2 | `ga_progress` 实现 | `format_ga_progress.py` | ELO 趋势表 + 多样性趋势 + 进度预估 |
| 2.3 | `ga_population` 实现 | `format_ga_population.py` | 种群统计 + ELO 分布直方图 |
| 2.4 | `ga_baseline` 实现 | `format_ga_baseline.py` | 基线胜率表 + 趋势表 |

**交付物**：可查看 ELO 趋势、多样性、种群分布、基线对战。

#### 阶段三：对手池 + 完整报告（预估 1-2 天）

**目标**：对手池监控 + 一键 `ga_report`。

| 序号 | 任务 | 产出文件 | 说明 |
|------|------|---------|------|
| 3.1 | `ga_opponents` 实现 | `format_ga_opponents.py` | 对手池状态 + TrueSkill 评分 |
| 3.2 | `ga_report` 实现 | `format_ga_report.py` | 整合以上所有维度的一键概览 |
| 3.3 | `monitor_ga.sh` | `tools/ppo_v9/monitor_ga.sh` | 一键脚本（类似 monitor_training.sh） |

**交付物**：`bash tools/ppo_v9/monitor_ga.sh` 一键查看综合报告。

#### 阶段四：异常扫描（预估 1-2 天）

**目标**：实现四项异常检测。

| 序号 | 任务 | 产出文件 | 说明 |
|------|------|---------|------|
| 4.1 | ELO 停滞检测 | `ga_anomaly_scanner.sh` 中的扫描逻辑 | 基于 convergence_threshold |
| 4.2 | 同质化检测 | 同上 | 基于 diversity 相对初始值的变化 |
| 4.3 | 对战异常率检测 | 同上 | 基于 evaluation.json 聚合 |
| 4.4 | NaN 检测 | 同上 | 日志 + JSON 双路径扫描 |
| 4.5 | `ga_scan` 集成 | `format_ga_scan.py` | 统一异常报告格式化 |

**交付物**：`manager.sh ga_scan` 和 `ga_report` 中的异常板块。

#### 阶段五：系统监控 + 运维操作（预估 1-2 天）

**目标**：系统资源监控 + 训练生命周期管理。

| 序号 | 任务 | 产出文件 | 说明 |
|------|------|---------|------|
| 5.1 | `ga_gpu` 实现 | 复用 `check_gpu.sh` 模式 | GPU 监控 |
| 5.2 | `ga_system` 实现 | 复用 `check_system.sh` 模式 | 系统资源监控 |
| 5.3 | `ga_tensorboard` 实现 | `deploy/ga_tensorboard.sh` | 远程 TensorBoard 启动 |
| 5.4 | `ga_start` 实现 | `deploy/ga_start_training.sh` | 启动训练 |
| 5.5 | `ga_stop` 实现 | `deploy/ga_stop_training.sh` | 停止训练 |
| 5.6 | `ga_resume` 实现 | 扩展 `ga_start` | 从 checkpoint 恢复 |
| 5.7 | `ga_deploy` 实现 | `deploy/ga_deploy_code.sh` | 代码部署 |
| 5.8 | `ga_logs` 实现 | `manager.sh` 内联逻辑 | 查看日志 |
| 5.9 | `ga_download` 实现 | `manager.sh` 内联逻辑 | 下载结果 |

**交付物**：完整的训练生命周期管理能力。

### 7.2 各阶段交付物汇总

| 阶段 | 关键交付物 | 验证方式 |
|------|-----------|---------|
| 一 | `manager.sh ga_status` | 查看远程训练是否运行、当前代次 |
| 二 | `ga_progress`、`ga_population`、`ga_baseline` | 查看趋势表和分布，数据正确 |
| 三 | `ga_opponents`、`ga_report`、`monitor_ga.sh` | 一键查看完整报告 |
| 四 | `ga_scan` 异常扫描 | 模拟异常场景，验证检测准确 |
| 五 | `ga_start/stop/resume/deploy/logs/download` | 完整的训练生命周期管理 |

### 7.3 优先级排序

1. **阶段一**（最高）— 无基础设施则无法启动任何监控
2. **阶段二** — 训练中最常看的健康指标
3. **阶段三** — 完整视图，提升可用性
4. **阶段四** — 自动化问题发现
5. **阶段五** — 操作便利性（可在阶段一完成后随时插入开发）

---

## 附录 A：ppo_v2 vs ppo_v9 命令对照表

| ppo_v2 命令 | 功能 | ppo_v9 对应命令 | 差异说明 |
|-------------|------|-----------------|---------|
| `train_status` | 训练状态 | `ga_status` | GA 用代次替代轮次 |
| `train_progress_check` | 进度检查 | `ga_progress` | GA 显示代次级 ELO 趋势 + 多样性 |
| `train_metrics_table` | PPO 技术参数表 | — | GA 无 policy_loss/entropy 等，由 ELO 趋势替代 |
| `train_action_table` | 动作统计 | — | GA 单动作空间简单，暂不需要 |
| `train_action_reward_table` | Action Reward | — | 同上 |
| `train_battle_table` | Baseline 对战横表 | `ga_baseline` | 对应关系明确 |
| `train_opponent_table` | 对手池统计 | `ga_opponents` | ppo_v9 对手池含 TrueSkill 评分，信息更丰富 |
| `scan_anomalies` | 异常扫描 | `ga_scan` | 检测内容完全不同（ELO 停滞 vs 梯度爆炸） |
| `train_report` | 完整报告 | `ga_report` | — |
| — | — | `ga_population` | ppo_v9 特有：种群统计与 ELO 分布 |
| `train_start` | 启动训练 | `ga_start` | 入口脚本不同（train_ga.py） |
| `train_stop` | 停止训练 | `ga_stop` | screen 会话名不同 |
| `train_resume` | 恢复训练 | `ga_resume` | checkpoint 格式不同 |
| `deploy_code` | 部署代码 | `ga_deploy` | 源码路径不同 |
| `exec` | 执行命令 | `ga_exec` | 需设置 ppo_v9 的 PYTHONPATH |
| `monitor_gpu` | GPU 监控 | `ga_gpu` | 功能一致 |
| `monitor_system` | 系统监控 | `ga_system` | 功能一致 |
| `tensorboard` | TensorBoard | `ga_tensorboard` | log 目录不同 |
| `file_upload/download` | 文件传输 | `ga_download` | ppo_v9 特化：按代次下载 |

## 附录 B：远程 PYTHONPATH 设置

ppo_v9 使用 `ppo_ga` 作为 Python 包前缀。远程聚合脚本运行在远程服务器上，需确保 Python 能导入 `ppo_ga` 包。在远程执行时设置：

```bash
export PYTHONPATH="/root/autodl-tmp/AntWar/ppo_v9/src:/root/autodl-tmp/AntWar/Ant-Game/SDK/python:\$PYTHONPATH"
```

对于纯 JSON 解析的聚合脚本（推荐方式），只需标准库 `json` + `os` + `glob`，无需依赖 `ppo_ga` 包，减少部署复杂度。
