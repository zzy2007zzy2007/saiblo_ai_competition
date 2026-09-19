# 基础设施模块 (Infrastructure)

## 1. 背景与定位

### 1.1 业务问题

在任何软件系统中，总有一部分代码不直接属于某个具体的业务模块，却被多个业务模块共同使用。如果将这些代码分散到各个业务模块中，会导致"循环依赖"或"代码重复"；如果放任不管，则会导致代码组织混乱。因此，需要一个**基础设施模块**来容纳这些通用的、与具体业务目标关联极小的功能。

**基础设施模块**的核心业务目标是：**提供被多个业务模块共享的通用能力**，包括游戏常量定义、配置加载与解析、项目路径管理、观测格式转换工具、回调框架以及跨模块数据聚合工具。这些组件不服务于某个具体的业务目标，而是为所有业务模块提供底层支撑。

### 1.2 在整个系统中的位置

```
┌──────────────────────────────────────────────────────────────┐
│                    基础设施模块                               │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ action_constants │  │ config_parser    │  │ path_config│  │
│  │ 游戏常量定义      │  │ YAML 配置加载    │  │ 路径管理    │  │
│  │ (所有模块依赖)    │  │ (训练/对战依赖)   │  │ (所有模块)  │  │
│  └─────────────────┘  └──────────────────┘  └────────────┘  │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ obs_utils       │  │ base_ppo_callback│  │ constants  │  │
│  │ 观测→张量转换    │  │ 回调框架基类     │  │ 日志/采样  │  │
│  │ (训练引擎使用)   │  │ (训练引擎使用)    │  │ 常量      │  │
│  └─────────────────┘  └──────────────────┘  └────────────┘  │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │ callback_factory│  │episode_batch_    │                   │
│  │ 回调组合工厂     │  │ writer           │                   │
│  │ (自对弈使用)     │  │ 跨模块聚合写入   │                   │
│  └─────────────────┘  └──────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
         ▲            ▲           ▲              ▲
         │            │           │              │
   所有业务模块   自对弈编排   训练引擎        对战执行
```

基础设施模块不依赖任何业务模块，但**所有业务模块都依赖它**。

### 1.3 与其他模块的关系

| 模块 | 依赖本模块的内容 |
|------|----------------|
| 游戏环境 | `action_constants` (TYPE_CONFIG, REWARD_CONFIG, OBS_NORMALIZATION 等) |
| 策略网络 | `action_constants` (TYPE_CONFIG, ACTION_DIM), `aux_labels` (NUM_HORIZONS) |
| PPO 训练引擎 | `action_constants`, `gae`, `obs_utils`, `base_ppo_callback` |
| 自对弈编排 | `config_parser`, `path_config`, `callback_factory`, `episode_batch_writer`, `constants` |
| 对战执行 | `action_constants`, `path_config` |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **action_constants** | 游戏常量集，包含地图定义、动作空间映射、奖励配置、观测归一化参数 |
| **TYPE_CONFIG** | 9 种动作类型在 96 维扁平空间中的索引区间映射 |
| **REWARD_CONFIG** | 完整的奖励权重体系（20+ 参数） |
| **OBS_NORMALIZATION** | 观测特征归一化系数（9 项缩放因子） |
| **Callback** | 回调机制，允许在不修改 PPO 训练核心代码的情况下插入自定义逻辑 |
| **EpisodeBatch 聚合** | 窗口级的对战和训练指标聚合，是跨模块的数据汇总工具 |

### 2.2 数据流概要

```
YAML 配置文件
    │
    ▼
config_parser.create_ppo_config()
    │
    ├─ 合并 CLI 覆盖参数
    ├─ 注入结构化默认值
    ├─ 类型校验
    │
    ▼
完整配置字典 → 传递给 SelfPlayTrainer → 分发给所有子模块
                      │
                      ├─ path_config → 创建目录结构
                      ├─ PPOTrainer → 解析 PPO 超参
                      └─ action_constants → 被各模块直接 import
```

---

## 3. 架构总览

### 3.1 模块组成

```
infrastructure/
│
├── utils/action_constants.py        # 游戏常量（地图/动作/奖励/观测）
├── config/config_parser.py          # YAML 配置加载与合并
├── config/path_config.py            # 项目路径自动解析
├── utils/obs_utils.py               # 观测 numpy→tensor 转换
├── callbacks/base_ppo_callback.py   # 回调基类（BaseCallback/BasePPOCallback）
├── callbacks/callback_factory.py    # 回调工厂 + CallbackList
├── monitor/constants.py             # 监控模块常量（日志格式/采样间隔）
└── monitor/episode_batch_writer.py  # EpisodeBatch 窗口聚合写入器
```

### 3.2 组件关系

```
┌──────────────────────────────────────────────┐
│          基础设施模块总览                       │
│                                                │
│  数据层:                                       │
│  ┌──────────────────────────────────────────┐ │
│  │ action_constants     constants           │ │
│  │  地图/动作/奖励/观测   日志格式/采样间隔   │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  配置层:                                       │
│  ┌──────────────────────────────────────────┐ │
│  │ config_parser     path_config            │ │
│  │ YAML → Dict       路径 → 目录            │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  工具层:                                       │
│  ┌──────────────────────────────────────────┐ │
│  │ obs_utils                                │ │
│  │ numpy → tensor                           │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  框架层:                                       │
│  ┌──────────────────────────────────────────┐ │
│  │ base_ppo_callback → callback_factory     │ │
│  │ 回调基类          → CallbackList + 工厂   │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  聚合层:                                       │
│  ┌──────────────────────────────────────────┐ │
│  │ episode_batch_writer                     │ │
│  │ 跨模块聚合对战+训练指标                   │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## 4. 核心组件详解

### 4.1 action_constants (`utils/action_constants.py`)

#### 4.1.1 职责

这是整个项目中**被引用最广泛的文件**，定义了游戏的核心常量和映射关系。所有其他模块（除联赛管理外）都直接或间接依赖它。

#### 4.1.2 核心常量

**地图相关**：

| 常量 | 值 | 说明 |
|------|-----|------|
| `MAP_SIZE` | 19 | 地图尺寸（19×19 网格） |
| `EDGE` | 10 | 基地到边界的距离 |
| `PLAYER_BASES` | `((2, 9), (16, 9))` | 双方基地坐标 |
| `PLAYER_COUNT` | 2 | 玩家数 |
| `MAX_ROUND` | 512 | 最大回合数 |
| `VALID_CELLS` | List[int] | 可落子的单元格索引列表 |
| `PATH_CELLS` | List[int] | 路径单元格 |
| `HIGHLAND_CELLS` | List[int] | 高地单元格 |
| `MAP_PROPERTY` | `np.ndarray (19,19)` | 完整地形矩阵（包含 VOID/PATH/BARRIER/HIGHLAND） |

**动作空间**：

| 常量 | 值 | 说明 |
|------|-----|------|
| `ACTION_DIM` | 96 | 扁平动作空间总维度 |
| `TYPE_CONFIG` | `Dict[str, TypeConfig]` | 9 种动作类型的起始索引/结束索引映射 |
| `TOWER_POSITIONS` | `List[Tuple[int,int]]` | 16 个塔位坐标 |
| `SuperWeaponType` | Enum | 4 种超级武器枚举 |
| `SUPER_WEAPON_POSITIONS` | `Dict[SuperWeaponType, List[int]]` | 每种武器的 5 个目标位置 |

**奖励配置**（`REWARD_CONFIG`）：

约 20+ 参数，按用途分组：

| 分组 | 参数示例 | 默认值 |
|------|---------|--------|
| 攻击奖励 | `hp_attack_weight`, `tower_attack_weight` | 1.0, 0.5 |
| 生存奖励 | `tower_survival_weight`, `tech_bonus_weight` | 0.1, 0.2 |
| 经济奖励 | `coin_income_weight`, `balance_bonus_weight` | 0.01, 0.005 |
| 惩罚项 | `ant_death_penalty`, `max_noop_penalty` | -0.1, -5.0 |
| 胜负奖励 | `win_bonus`, `lose_penalty` | 10.0, -10.0 |
| 动作奖励 | `build_rewards`, `upgrade_rewards`, `deploy_rewards` | 各操作独立配置 |
| 裁剪 | `step_reward_clip` | 10.0 |

**观测归一化**（`OBS_NORMALIZATION`）：

| 键 | 值 | 用途 |
|-----|-----|------|
| `hp_scale` | 1000.0 | HP 相关特征归一化 |
| `coin_scale` | 500.0 | 金币相关特征归一化 |
| `distance_scale` | 19.0 | 距离相关特征归一化 |
| `round_scale` | 512.0 | 回合进度归一化 |
| 其他 5 项 | 各类特征缩放因子 | 确保所有特征在相近的数值范围内 |

#### 4.1.3 核心函数

```python
def action_id_to_type_and_target(action_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

将扁平 action ID 张量转换为 `(type_id, target_k)` 张量对：
- 遍历 `TYPE_CONFIG` 中的每个类型
- 判断 `action_id` 落在哪个类型的 `[flat_start, flat_end)` 区间
- 计算 `target_k = action_id - flat_start`

此函数在策略网络的结构化动作头和 PPO 训练器的动作类型统计中使用。

### 4.2 config_parser (`config/config_parser.py`)

#### 4.2.1 职责

提供 YAML 配置文件的加载、CLI 覆盖合并和类型校验功能。所有训练相关的配置都通过此模块处理。

#### 4.2.2 核心类与函数

```python
class PPORLConfigParser:
    @staticmethod
    def load_yaml(yaml_path) -> Dict
    @staticmethod
    def merge_yaml_and_args(yaml_path, cli_overrides) -> Dict
    @staticmethod
    def _merge_structural_defaults(config) -> Dict
    @staticmethod
    def _validate_types(data, prefix) -> None

def create_ppo_config(yaml_path=None, cli_overrides=None) -> Dict
```

#### 4.2.3 配置合并流程

```
create_ppo_config()
│
1. 加载 YAML 文件
   config = PPORLConfigParser.load_yaml(yaml_path)
   # 默认: src/ppo_antwar/configs/ppo_antwar.yaml
│
2. 合并结构化默认值
   config = PPORLConfigParser._merge_structural_defaults(config)
   # 注入 system.device (auto检测GPU/CPU)
   # 注入 network.board_shape (19,19), global_dim (33)
   # 注入 env.player_id (0), backend_type ("python")
│
3. 合并 CLI 覆盖
   config = PPORLConfigParser._deep_merge(config, cli_overrides)
   # CLI 参数可以覆盖 YAML 中的任何值
│
4. 类型校验
   PPORLConfigParser._validate_types(config)
   # 校验 21 个关键配置项的类型（如 ppo.lr 必须是 float）
   # 类型不匹配时给出警告并跳过
│
5. 过滤 None 值
   config = PPORLConfigParser._filter_none_values(config)
│
6. 设备检查
   device = "cuda" if torch.cuda.is_available() else "cpu"
   config['system']['device'] = device
│
7. 返回完整配置字典
```

#### 4.2.4 CLI 覆盖参数的类型映射

`_TYPE_MAPPING` 定义了 21 个配置项的类型约束，覆盖：

| 分组 | 配置项 | 类型 |
|------|--------|------|
| PPO | `ppo.lr`, `ppo.gamma`, `ppo.gae_lambda`, `ppo.clip_epsilon` | float |
| PPO | `ppo.ppo_epochs`, `ppo.batch_size` | int |
| 训练 | `training.total_episodes`, `training.update_timesteps` | int |
| 自对弈 | `selfplay.max_opponents`, `selfplay.opponent_save_interval` | int |
| 网络 | `network.hidden_dim` | int |
| 系统 | `system.num_workers` | int |

### 4.3 path_config (`config/path_config.py`)

#### 4.3.1 职责

基于项目根目录自动解析所有路径，统一管理输出目录的创建和访问。采用**快速失败**策略——初始化时一次性验证所有必需的路径。

#### 4.3.2 路径解析

```python
class PathConfig:
    def __init__(self, base_output_dir=""):
        # 项目根目录：向上 4 层
        # ppo_v2/src/ppo_antwar/config/path_config.py
        # → ppo_v2/ → AntWar/
        self.project_root = Path(__file__).resolve().parents[4]

        # SDK 路径
        self.sdk_path = self.project_root / "Ant-Game" / "SDK"
        if not self.sdk_path.exists():
            raise PathConfigError(f"SDK not found: {self.sdk_path}")

        # 基线路径
        self.baselines_path = self.project_root / "baselines"
        if not self.baselines_path.exists():
            raise PathConfigError(f"Baselines not found: {self.baselines_path}")

        # 输出根目录
        self.base_output_dir = Path(base_output_dir) if base_output_dir \
            else self.project_root / "outputs"
```

#### 4.3.3 目录生成

所有 `get_*_dir(run_id)` 方法都自动创建目录并返回路径：

| 方法 | 生成的目录 | 用途 |
|------|-----------|------|
| `get_training_dir(run_id)` | `outputs/{run_id}/training/` | 训练日志 |
| `get_checkpoint_dir(run_id)` | `outputs/{run_id}/checkpoints/` | 模型检查点 |
| `get_battle_dir(run_id)` | `outputs/{run_id}/battle/` | 对战日志 |
| `get_selfplay_dir(run_id)` | `outputs/{run_id}/selfplay/` | 自对弈日志 |
| `get_system_dir(run_id)` | `outputs/{run_id}/system/` | 系统指标 |
| `get_evaluation_dir(run_id)` | `outputs/{run_id}/evaluation/` | 评估日志 |
| `get_run_dir(run_id)` | `outputs/{run_id}/` | 运行根目录 |

### 4.4 obs_utils (`utils/obs_utils.py`)

#### 4.4.1 职责

提供观测数据格式转换工具，将从游戏环境模块获得的 numpy 观测字典转为 PyTorch 张量格式，自动添加 batch 维度，使数据可以直接输入策略网络。

#### 4.4.2 核心函数

```python
def observation_to_tensors(observation: Dict[str, np.ndarray],
                           device: torch.device) -> Dict[str, torch.Tensor]:
    """将观测字典转为张量字典，添加 batch 维度"""
    return {
        'board': torch.from_numpy(observation['board']).float()
                 .unsqueeze(0).to(device),          # (1, 28, 19, 19)
        'global': torch.from_numpy(observation['global']).float()
                  .unsqueeze(0).to(device),          # (1, 33)
        'action_mask': torch.from_numpy(observation['action_mask']).float()
                       .unsqueeze(0).to(device),     # (1, 96)
    }
```

此函数在 `PPOAgent.act()` 和 `PPOTrainer._select_action()` 中方法级导入。

### 4.5 回调框架

#### 4.5.1 base_ppo_callback (`callbacks/base_ppo_callback.py`)

定义 PPO 训练回调的基类体系，提供生命周期钩子：

```
BaseCallback (空标记接口)
    │
    └── BasePPOCallback
        ├── __init__() : 初始化回调
        ├── init_callback(trainer) : 注入 PPOTrainer 引用
        ├── on_training_start() : 训练开始时触发
        ├── on_training_end() : 训练结束时触发
        └── on_step() → bool : 每个训练步触发，返回 False 停止训练
            └── (委托给 _on_step()，子类重写此方法)
```

**设计要点**：
- `_on_step()` 是子类重写点，`on_step()` 是外部调用入口
- 返回 `bool` 控制训练是否继续，任一回调返回 `False` 则停止

#### 4.5.2 callback_factory (`callbacks/callback_factory.py`)

提供回调组合和工厂创建功能：

**CallbackList**：回调容器，按序管理多个回调的完整生命周期：

```python
class CallbackList:
    def append(self, callback)
    def init_callback(self, trainer)     # 逐个初始化
    def on_training_start(self)          # 逐个触发
    def on_step(self) -> bool            # 逐个触发，任一返回 False 则停止
    def on_training_end(self)            # 逐个触发
```

**create_ppo_callbacks()**：工厂函数，按配置创建回调组合：

```python
def create_ppo_callbacks(
    save_interval=100, log_interval=10,
    checkpoint_dir=None, path_config=None,
    enable_checkpoint=True,
    enable_metrics_logging=True,
    enable_tensorboard=False,
    tensorboard_dir=None, tensorboard_interval=10
) -> Optional[CallbackList]
```

根据开关参数组合以下回调：
1. `MetricsLoggingCallback`（终端指标日志）
2. `CheckpointCallback`（周期性检查点保存 + 旧文件清理）
3. `TensorBoardCallback`（TensorBoard 指标记录）

### 4.6 constants (`monitor/constants.py`)

#### 4.6.1 职责

定义监控子系统的共享常量，被自对弈编排模块中的多个日志组件使用。

| 常量 | 值 | 说明 |
|------|-----|------|
| `LOG_FORMAT` | (彩色格式) | loguru 控制台格式 |
| `FILE_LOG_FORMAT` | (无颜色格式) | loguru 文件格式 |
| `BATTLE_LOG_FORMAT` | `{time} \| {level} \| [{extra[category]}] \| {message}` | 对战日志格式 |
| `SP_LOG_FORMAT` | `{time} \| {level} \| [selfplay] \| {message}` | 自对弈格式 |
| `DEFAULT_LEVEL` | `"INFO"` | 默认级别 |
| `SAMPLE_INTERVAL` | 10 | 系统指标采样间隔（秒） |
| `SAMPLES_PER_WRITE` | 6 | 每次写入样本数 |
| `NVIDIA_SMI_TIMEOUT` | 5 | GPU 信息超时（秒） |
| `METRICS_CACHE_SIZE` | 100 | 指标缓存容量 |
| `RECENT_METRICS_WINDOW` | 10 | 近期指标窗口 |
| `HISTORICAL_METRICS_WINDOW` | 100 | 历史指标窗口 |

### 4.7 episode_batch_writer (`monitor/episode_batch_writer.py`)

#### 4.7.1 职责

跨模块的窗口聚合器，读取自对弈对战日志和训练指标日志，按窗口（默认 8 局）聚合为统计摘要。这是唯一需要同时读取两个不同模块输出的组件。

#### 4.7.2 核心类

```python
class EpisodeBatchWriter:
    def __init__(self, battle_log_path, batch_metrics_path,
                 battle_stats_path, train_stats_path, window_size=8)
```

| 参数 | 说明 |
|------|------|
| `battle_log_path` | `selfplay_battle_log.jsonl` 源文件路径 |
| `batch_metrics_path` | `batch_metrics.jsonl` 源文件路径 |
| `battle_stats_path` | 聚合对战统计输出路径 |
| `train_stats_path` | 聚合训练统计输出路径 |
| `window_size` | 聚合窗口大小（默认 8 局） |

#### 4.7.3 公开方法

| 方法 | 功能 |
|------|------|
| `on_episode_complete(episode)` | 递增窗口计数器 |
| `on_batch_complete(episode)` | 触发窗口检查，满时执行聚合 |
| `flush_now()` | 强制刷新（训练结束时调用） |
| `close()` | 调用 `flush_now()` |

#### 4.7.4 聚合逻辑

**对战统计聚合** (`_aggregate_battle`)：
- 读取 `selfplay_battle_log.jsonl` 中窗口内的记录
- 计算：胜率、平率、平均奖励、平均回合
- 聚合：HP/金币/收入平均值、动作计数分布、奖励来源分量

**训练统计聚合** (`_aggregate_train`)：
- 读取 `batch_metrics.jsonl` 中窗口内的记录
- 计算各指标均值：loss、policy_loss、value_loss、entropy、clip_fraction、grad_norm、learning_rate、reward_mean、return_mean

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

基础设施模块**不依赖** `ppo_antwar` 内部的任何其他模块。其依赖仅限于外部库和 Python 标准库：

| 组件 | 依赖 |
|------|------|
| `action_constants` | numpy, torch, SDK.utils.constants |
| `config_parser` | yaml, torch |
| `path_config` | pathlib |
| `obs_utils` | numpy, torch |
| `base_ppo_callback` | loguru, typing |
| `callback_factory` | loguru |
| `constants` | 无（纯常量） |
| `episode_batch_writer` | json, pathlib |

### 5.2 依赖本模块的外部模块

| 组件 | 被哪些模块使用 |
|------|--------------|
| `action_constants` | **所有业务模块**（游戏环境、策略网络、PPO 训练引擎、对战执行） |
| `config_parser` | 自对弈编排 |
| `path_config` | 自对弈编排、PPO 训练引擎、对战执行 |
| `obs_utils` | PPO 训练引擎（方法级导入） |
| `base_ppo_callback` | PPO 训练引擎（各回调子类继承） |
| `callback_factory` | 自对弈编排 |
| `constants` | 自对弈编排（selfplay_logger, system_metrics_sampler） |
| `episode_batch_writer` | 自对弈编排 |

### 5.3 典型交互场景

#### 场景 1：训练启动时的基础设施初始化

1. 外部脚本调用 `create_ppo_config()` → 加载 YAML → 合并 CLI 覆盖 → 返回完整配置
2. 配置传递给 `SelfPlayTrainer.__init__()`
3. `PathConfig` 自动解析项目根目录，验证 SDK 和 baselines 路径
4. 调用 `get_training_dir(run_id)` 等创建输出目录结构
5. `PPOTrainer` 从配置中读取 `ppo.lr`、`ppo.gamma` 等超参
6. `action_constants` 已经在各个模块的 `import` 时完成了常量加载

#### 场景 2：PPO 训练中的回调触发

1. `create_ppo_callbacks()` 根据配置创建 `CallbackList`（含 MetricsLoggingCallback + CheckpointCallback）
2. 回调列表注入到 `PPOTrainer` 中
3. 在每个 `_ppo_update()` 完成后调用 `callbacks.on_step()`
4. `MetricsLoggingCallback` 每隔 N 步输出训练指标到终端
5. `CheckpointCallback` 每隔 N 步保存检查点并清理旧文件
6. 如果任一回调返回 `False`，训练主循环收到信号并优雅停止

#### 场景 3：EpisodeBatch 窗口聚合

1. 自对弈编排每局调用 `episode_batch_writer.on_episode_complete(episode)`
2. PPO 更新后调用 `episode_batch_writer.on_batch_complete(episode)`
3. 当窗口计数器满（累积 8 局）时：
   - 从 `selfplay_battle_log.jsonl` 读取最近 8 局的对战数据
   - 从 `batch_metrics.jsonl` 读取最近 8 局的训练指标
   - 分别聚合为统计摘要
   - 写入 `episode_batch_battle_stats.jsonl` 和 `episode_batch_train_stats.jsonl`
4. 训练结束时调用 `flush_now()` 强制刷新剩余数据

---

## 6. 关键设计决策

### 6.1 action_constants 作为唯一常量中心

整个项目将所有游戏相关常量集中在一个文件中（`action_constants.py`），而非分散在各模块中：

- **单点定义**：地图、动作空间、奖励、观测归一化全部定义在同一位置
- **避免循环依赖**：虽然所有模块都依赖它，但它不依赖任何业务模块
- **配置即代码**：奖励权重等"超参数"写在代码中而非配置文件中，因为训练过程中不会动态调整
- **编译时校验**：Python 在 import 时加载常量，任何错误会被立即发现

### 6.2 config_parser 的三层合并策略

配置合并采用"YAML 默认值 → 结构化默认值 → CLI 覆盖"三层策略：

1. **YAML 默认值**：提供完整的可运行配置
2. **结构化默认值**：补全某些 YAML 中未指定的关键参数（如设备类型、网络输入维度等）
3. **CLI 覆盖**：允许不修改 YAML 文件的情况下临时改变配置

这种设计既保证了默认配置的可运行性，又提供了灵活的覆盖机制。

### 6.3 PathConfig 的快速失败策略

`PathConfig.__init__()` 在初始化时一次性验证所有必需路径（SDK、baselines）：

- **失败时机**：在训练开始前，而非训练过程中
- **失败信息**：明确的 `PathConfigError` 异常，包含缺失路径的详细信息
- **自动创建**：输出目录在第一次访问时自动创建，不需要手动管理

### 6.4 回调框架的轻量设计

回调框架仅提供三个钩子（`on_step`、`on_training_start`、`on_training_end`）：

- **足够**：这三个钩子覆盖了训练引擎需要的所有扩展点
- **简单**：不需要事件总线或复杂的注册机制
- **可组合**：多个回调通过 `CallbackList` 组合，任一回调可终止训练

### 6.5 episode_batch_writer 的跨模块定位

`episode_batch_writer` 是唯一一个需要同时读取两个不同模块输出（对战日志来自自对弈，训练指标来自 PPO 引擎）的组件。将其归入基础设施而非任一业务模块中，是因为：

- **避免循环依赖**：如果放在自对弈模块中，它依赖训练引擎的输出；反之亦然
- **单一职责**：它的职责是"跨模块聚合"，与具体的业务目标无关
- **解耦**：两个业务模块不需要了解对方的日志格式
