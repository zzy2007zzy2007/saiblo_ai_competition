# 11. 关键配置参数速查

本章汇总 PPO v1 SelfPlay 与 Baseline Battle 体系中的所有关键配置参数，按功能模块分类，以表格形式呈现。所有参数默认值及含义均来自以下源文件：

- YAML 配置文件：[`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml)
- BaselineBattleConfig：[`battle_config.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_config.py)
- 配置解析器：[`config_parser.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/config/config_parser.py)

## 11.1 PPO 参数

PPO 算法的核心超参数，定义在 [`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L11-L27) 的 `ppo:` 段中。

| 参数名           | 默认值  | 含义                                    |
| ---------------- | ------- | --------------------------------------- |
| `lr`             | 0.0001  | Adam 优化器的初始学习率                  |
| `gamma`          | 0.99    | 折扣因子，用于计算累计回报                |
| `gae_lambda`     | 0.95    | GAE（广义优势估计）的 λ 参数              |
| `clip_eps`       | 0.2     | PPO policy loss 的裁剪范围 ε              |
| `clip_eps_vf`    | 0.2     | PPO value loss 的裁剪范围 ε               |
| `ent_coef`       | 0.15    | 初始熵正则化系数（策略探索权重）           |
| `ent_coef_final` | 0.15    | 熵退火结束时的最终熵系数                  |
| `ent_anneal_power` | 1.0   | 熵退火的幂次（1.0 表示线性退火）           |
| `vf_coef`        | 0.02    | Value loss 的权重系数                     |
| `max_grad_norm`  | 0.5     | 策略网络梯度裁剪的最大范数                 |
| `max_grad_norm_vf` | 0.2  | 价值网络梯度裁剪的最大范数                 |
| `ppo_epochs`     | 4       | 每轮 PPO 更新的 epoch 数                   |
| `batch_size`     | 64      | 每次小批量更新的样本数                     |
| `lr_warmup_episodes` | 1000 | 学习率预热阶段覆盖的 episode 数            |
| `lr_warmup_init` | 0.00001 | 学习率预热的起始值                         |
| `lr_cosine_decay` | true   | 是否启用余弦退火衰减学习率                 |

## 11.2 训练参数

训练流程的整体控制参数，定义在 [`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L44-L52) 的 `training:` 段中。

| 参数名                    | 默认值  | 含义                                              |
| ------------------------- | ------- | ------------------------------------------------- |
| `total_episodes`          | 150000  | 总训练 episode 数                                  |
| `eval_interval`           | 1000    | 评估间隔（每多少个 episode 进行一次评估）           |
| `save_interval`           | 5000    | 模型保存间隔（每多少个 episode 保存一次 checkpoint） |
| `opponent_update_interval` | 500    | 对手池更新间隔（每多少个 episode 添加/淘汰对手）    |
| `num_envs`                | 1       | 并行环境数量                                        |
| `battle_interval`         | 1000    | Baseline Battle 触发间隔（每多少个 episode 执行一次） |
| `n_battles`               | 5       | 每次 Baseline Battle 的对战局数（需 >= 2，用于先后手交替） |
| `max_steps_per_episode`   | 512     | 每个 episode 的最大 step 数（对应最多 256 个完整回合） |

## 11.3 SelfPlay 参数

SelfPlay 对手管理与对手选择的参数，定义在 [`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L54-L58) 的 `selfplay:` 段中。

| 参数名               | 默认值 | 含义                                            |
| -------------------- | ------ | ----------------------------------------------- |
| `opponent_pool_size` | 10     | 对手池最大容量                                   |
| `min_opponent_games` | 8      | 对手入池的最低对战场次阈值（低于此值不会被淘汰）  |
| `exploit_prob`       | 0.7    | 初始 exploit 概率（选择最强对手进行对战）         |
| `explore_prob`       | 0.3    | 初始 explore 概率（随机选择对手进行探索）         |

> **说明**：`exploit_prob` 会在训练过程中自适应增长（从 0.3 到 0.9），详见 [§3.2 对手选择机制](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_outline.md)。配置文件中的 0.7 和 0.3 是初始化值，实际运行时 `OpponentSelector` 会根据训练进度动态调整。

## 11.4 网络参数

神经网络结构的参数，定义在 [`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L64-L71) 的 `network:` 段中。

| 参数名        | 默认值           | 含义                                                 |
| ------------- | ---------------- | ---------------------------------------------------- |
| `hidden_dim`  | 256              | MLP 隐藏层维度（全局特征编码和策略/价值头共用）       |
| `board_shape` | `[28, 19, 19]`   | board 特征形状：`[channels=28, height=19, width=19]` |
| `global_dim`  | 33               | 全局标量特征的维度                                    |
| `action_dim`  | 96               | 动作空间总维度（包含 NO_OP、建塔、科技等所有动作）    |

## 11.5 BaselineBattleConfig 参数

Baseline Battle 的独立配置参数，定义在 [`battle_config.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_config.py#L8-L33) 的 `BaselineBattleConfig` 数据类中。

| 参数名                     | 默认值       | 含义                                                 |
| -------------------------- | ------------ | ---------------------------------------------------- |
| `max_rounds`               | 512          | 单局最大回合数（对应最多 256 个完整回合）             |
| `n_battles`                | 4            | 并行进程数 / 对战局数（先后手各一次为固定逻辑）       |
| `log_level`                | `"INFO"`     | 日志级别（支持 `DEBUG`、`INFO`、`WARNING`、`ERROR`）  |
| `log_dir`                  | `None`       | 日志输出目录（`None` 时使用 PathConfig 默认路径）      |
| `log_to_console`           | `true`       | 是否输出日志到控制台                                  |
| `enable_timing`            | `true`       | 是否启用计时功能（记录各阶段耗时）                     |
| `timing_warning_threshold` | 2.0          | 计时警告阈值（秒），超过此值将输出 warning 日志        |
| `ppo_checkpoint_path`      | `None`       | PPO checkpoint 文件路径                               |
| `baseline_agents`          | `[]`         | baseline agent 列表（用于评估的对手 agent 名）         |
| `agent1_name`              | `None`       | 对战方1的名称（用于日志标识）                          |
| `agent2_name`              | `None`       | 对战方2的名称（用于日志标识）                          |
| `battle_interval`          | 1000         | Baseline Battle 触发间隔（episode 数）                 |
| `checkpoint_save_interval` | `None`       | checkpoint 保存间隔（`None` 时使用 training 中的配置） |
| `device`                   | `"auto"`     | 运行设备（`"auto"` 表示自动选择 CUDA / CPU）          |

> **说明**：`BaselineBattleConfig` 通过 `from_dict()` 从字典创建，也可以通过 `load_from_file()` 从独立 YAML 文件加载（`battle:`、`logging:`、`agents:`、`training:` 四个段），或通过 `load_from_args()` 从命令行参数覆盖。最终通过 `validate()` 进行合法性校验。

## 11.6 数值安全边界参数

为防止训练过程中出现数值溢出、NaN 等问题而设置的裁剪参数，定义在 [`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L29-L42) 的 `ppo.clip_values:` 段中。

| 参数名           | 默认值  | 含义                                                |
| ---------------- | ------- | --------------------------------------------------- |
| `logits_min`     | -20     | 网络输出 logits 的最小裁剪值                          |
| `logits_max`     | 20      | 网络输出 logits 的最大裁剪值                          |
| `values_min`     | -10     | Value head 输出的最小裁剪值                           |
| `values_max`     | 10      | Value head 输出的最大裁剪值                           |
| `returns_min`    | -1e6    | 累计回报（returns）的最小裁剪值                        |
| `returns_max`    | 1e6     | 累计回报（returns）的最大裁剪值                        |
| `advantages_min` | -1e4    | 优势估计（advantages）的最小裁剪值                     |
| `advantages_max` | 1e4     | 优势估计（advantages）的最大裁剪值                     |
| `prob_min`       | 1e-10   | 策略概率的最小值（防止 log(0)）                        |
| `ratio_min`      | 1e-5    | PPO 重要性采样比率的最小值                             |
| `ratio_max`      | 1e5     | PPO 重要性采样比率的最大值                             |
| `loss_max`       | 1e6     | 单次 loss 值的最大裁剪值                               |
| `nan_threshold`  | 5       | NaN 连续出现次数的容忍阈值（超过后触发保护机制）        |

## 11.7 League 参数

League 战绩系统的参数，定义在 [`ppo_antwar.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L60-L62) 的 `league:` 段中。

| 参数名               | 默认值 | 含义                                              |
| -------------------- | ------ | ------------------------------------------------- |
| `decay`              | 0.99   | 历史战绩的指数衰减系数（每轮更新后乘以该系数）      |
| `min_win_rate_games` | 8      | 计算胜率所需的最低对战场次（低于此值胜率不可靠）    |

---

> **配置加载说明**：所有参数通过 [`config_parser.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/config/config_parser.py) 中的 `PPORLConfigParser` 加载。主入口为 `create_ppo_config()` 函数，支持 YAML 文件 + 命令行参数 + 字典覆盖三级合并，其中 `battle_interval` 参数在合并时有特殊优先级处理（命令行参数优先于 YAML 文件）。
