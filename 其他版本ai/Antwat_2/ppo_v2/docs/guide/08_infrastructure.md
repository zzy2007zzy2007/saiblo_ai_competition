# 模块8：基础设施

## 1. 概述

### 1.1 模块定位

基础设施模块提供 **跨模块的通用服务**，与具体业务逻辑解耦。包括配置管理（YAML 加载、合并、验证）、路径管理（统一路径解析和目录创建）和 SDK 兼容适配（版本差异封装）。

### 1.2 在整体架构中的位置

```
                          ┌──────────────────────┐
                          │   模块8: 基础设施      │
                          │                        │
                          │  ┌──────────────────┐ │
                          │  │ConfigParser      │ │
                          │  │(YAML → Dict      │ │
                          │  │ + CLI覆盖         │ │
                          │  │ + 类型验证)       │ │
                          │  └──────────────────┘ │
                          │                        │
                          │  ┌──────────────────┐ │
                          │  │PathConfig        │ │
                          │  │(统一路径管理       │ │
                          │  │ 自动创建目录)     │ │
                          │  └──────────────────┘ │
                          │                        │
                          │  ┌──────────────────┐ │
                          │  │BackendAdapter    │ │
                          │  │(SDK版本兼容       │ │
                          │  │ 运行时创建)       │ │
                          │  └──────────────────┘ │
                          └──────────────────────┘
                              ▲        ▲        ▲
                              │        │        │
                     ┌────────┴────────┴────────┴────────┐
                     │  模块1-7 全部依赖模块8              │
                     └─────────────────────────────────────┘
```

该模块被 **所有其他 7 个模块** 直接或间接依赖。

### 1.3 业务目标

- 统一管理训练系统的所有配置参数（YAML + 命令行覆盖）
- 提供项目路径的标准化管理，确保输出目录结构一致
- 封装游戏 SDK 的版本差异，提供统一的运行时创建接口

---

## 2. 背景与概念

### 2.1 配置管理策略

系统配置采用 **分层合并** 策略：

1. **系统默认值**（硬编码在 config_parser 中）
2. **YAML 配置文件**（如 `ppo_antwar.yaml` 或 `custom_antwar.yaml`）
3. **CLI 命令行参数**（最高优先级）

```python
最终配置 = merge(默认值, YAML配置, CLI覆盖)
```

### 2.2 路径管理策略

所有输出文件统一放在 `outputs/{run_id}/` 目录下，子目录按功能划分：

- `training/` — 训练日志
- `selfplay/` — 自对弈数据
- `checkpoint/` — 模型检查点
- `battle/` — 对战日志
- `system/` — 系统监控
- `evaluation/` — 评估结果

`PathConfig` 采用 **快速失败（Fail-Fast）** 策略：初始化时立即检查关键路径（SDK、baselines）是否存在，避免在训练过程中才发现路径错误。

### 2.3 SDK 适配层

游戏 SDK 可能有多版本（Python 后端 vs 原生后端），不同版本的 API 略有差异。`BackendAdapter` 封装了这些差异，为上层提供统一的接口。

---

## 3. 架构设计

### 3.1 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `PPORLConfigParser` | [config/config_parser.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/config/config_parser.py) | 配置加载、合并、验证 |
| `create_ppo_config()` | [config/config_parser.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/config/config_parser.py) | 便捷配置创建入口 |
| `PathConfig` | [config/path_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/config/path_config.py) | 统一路径管理 |
| `BackendAdapter` | [compat/adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/compat/adapters.py) | SDK 后端兼容适配器 |
| `MatchRuntimeWrapper` | [compat/adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/compat/adapters.py) | SDK runtime 兼容包装 |

---

## 4. 核心组件详解

### 4.1 PPORLConfigParser — 配置解析器

位置：[config/config_parser.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/config/config_parser.py)

#### 关键方法

| 方法 | 说明 |
|------|------|
| `load_yaml(yaml_path)` | 读取 YAML 文件 → Dict |
| `merge_yaml_and_args(yaml_path, cli_overrides)` | 加载 YAML + 合并默认值 + 合并 CLI 覆盖 |
| `_merge_structural_defaults(config)` | 补全 system/network/env 字段的结构默认值 |
| `_deep_merge(base, override)` | 递归 dict 合并 |
| `_filter_none_values(data)` | 移除值为 None 的 CLI 参数 |
| `_validate_types(data, prefix)` | 按 `_TYPE_MAPPING` 做类型转换 |

#### 配置合并流程

```
load_yaml(yaml_path)
    │
    ▼
YAML dict
    │
    ▼
_merge_structural_defaults()
    │  补全:
    │    system.device → "cpu"
    │    network.board_shape → [28, 19, 19]
    │    network.global_dim → 33
    │    env.player_id → 0
    │    env.backend_type → "python"
    │
    ▼
merge with CLI overrides:
    │  1. _filter_none_values(cli_overrides) — 移除未设置的 CLI 参数
    │  2. _validate_types() — 按类型映射转换（int/float/bool）
    │  3. _deep_merge() — 递归覆盖
    │
    ▼
最终配置 dict
```

#### 类型映射（_TYPE_MAPPING）

预定义了 30+ 个配置键的类型映射，例如：
- `"ppo.ppo_epochs"` → `int`
- `"ppo.clip_epsilon"` → `float`
- `"ppo.entropy_coef"` → `float`

CLI 传入的字符串参数会按类型映射自动转换。

---

### 4.2 便捷入口

```python
def create_ppo_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict] = None,
) -> Dict:
    """便捷入口：加载配置并验证 CUDA 可用性"""
```

默认使用 `ppo_antwar.yaml`，并自动检测 CUDA 是否可用（若配置指定 CUDA 但系统不支持则降级为 CPU）。

---

### 4.3 PathConfig — 路径管理器

位置：[config/path_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/config/path_config.py)

```python
PathConfig(base_output_dir: str = "")
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `project_root` | `Path` | 项目根目录 |
| `sdk_path` | `Path` | SDK 目录路径 |
| `baselines_path` | `Path` | 外部 AI 目录路径 |

初始化时验证 `sdk_path` 和 `baselines_path` 存在性（Fail-Fast）。

#### 目录方法（均自动创建父目录）

| 方法 | 返回路径 |
|------|---------|
| `get_training_dir(run_id)` | `outputs/{run_id}/training/` |
| `get_checkpoint_dir(run_id)` | `outputs/{run_id}/checkpoint/` |
| `get_battle_dir(run_id)` | `outputs/{run_id}/battle/` |
| `get_selfplay_dir(run_id)` | `outputs/{run_id}/selfplay/` |
| `get_system_dir(run_id)` | `outputs/{run_id}/system/` |
| `get_evaluation_dir(run_id)` | `outputs/{run_id}/evaluation/` |
| `get_run_dir(run_id)` | `outputs/{run_id}/` |

---

### 4.4 BackendAdapter — SDK 后端适配器

位置：[compat/adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/compat/adapters.py)

```python
BackendAdapter:
    create_runtime(player, seed=0, cold_handle_rule_illegal=False,
                   prefer_native=False, movement_policy=None)
        -> MatchRuntimeWrapper
    
    supports_native_backend()
        -> bool
    
    get_backend_info()
        -> Dict[str, Any]
```

#### MatchRuntimeWrapper

包装 SDK 原始 runtime，提供统一的 `resolve_turn` 接口：

```python
MatchRuntimeWrapper(original_runtime)
    resolve_turn(player_0_operations, player_1_operations)
        -> _SimulatedTurnResolution
```

内部根据 `runtime.player` 决定哪个操作是 self，哪个是 opponent，分别应用后 `advance_round()`。

#### 原生后端检测

`supports_native_backend()` 检测 `SDK.native_antwar` 是否可导入。原生后端通常更快，但不是所有环境都能用。

`get_backend_info()` 返回 `{python_available, native_available, cuda_available}` 等信息。

---

## 5. 对外交互

### 5.1 被所有模块依赖

| 依赖方 | 使用什么 | 场景 |
|--------|---------|------|
| **模块1（游戏环境）** | `BackendAdapter` | 创建游戏运行时 |
| **模块2（神经网络）** | 配置参数（network 字段） | `hidden_dim` 等结构参数 |
| **模块3（PPO训练引擎）** | `PathConfig`, 配置参数 | 检查点目录、训练参数 |
| **模块4（自对弈编排）** | `PathConfig`, 配置参数 | 所有输出目录、自对弈参数 |
| **模块5（对战评估）** | `BackendAdapter`, `PathConfig` | 创建运行时、日志路径 |
| **模块6（回调系统）** | `PathConfig` | 检查点目录、TensorBoard 目录 |
| **模块7（监控）** | `PathConfig` | 所有日志/数据输出目录 |

---

## 6. 配置参数体系

### 6.1 配置顶层结构

```yaml
system:
  device: "cuda" | "cpu" | "auto"

network:
  hidden_dim: 256
  enable_auxiliary: true
  board_channels: 28
  board_shape: [28, 19, 19]
  global_dim: 33

env:
  player_id: 0
  backend_type: "python"
  prefer_native: false

ppo:
  # 算法参数
  ppo_epochs: 4
  batch_size: 2048
  minibatch_size: 512
  clip_epsilon: 0.2
  clip_epsilon_vf: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coef: 0.01
  value_loss_coef: 0.5
  aux_loss_weight: 0.5
  max_grad_norm: 0.5
  target_kl: 0.01
  adv_clip: 10.0

  # 学习率
  lr_base: 0.0003
  lr_vf_base: 0.001
  warmup_episodes: 100
  cosine_decay: true

  # 数据收集
  max_steps: 512
  n_steps: 2048
  num_minibatches: 4

selfplay:
  opponent_pool_size: 10
  min_opponent_games: 5
  exploit_prob: 0.5
  opponent_update_interval: 100
  payoff_decay: 0.99

evaluation:
  eval_interval: 200
  baseline_agents: ["BasicTowerAI", "MediumRuleAI"]
  n_battles: 50

callbacks:
  save_interval: 100
  log_interval: 10
  keep_last_n: 5
  enable_tensorboard: false
```

### 6.2 CLI 覆盖示例

```bash
python train.py --ppo.learning_rate 0.0005 --system.device cuda \
                --selfplay.opponent_pool_size 15
```

---

## 7. 常见问题与注意事项

### 7.1 配置合并的优先级

CLI > YAML 文件 > 结构默认值 > 系统默认值

这意味着：
- YAML 中未定义的字段会使用默认值
- CLI 参数会覆盖 YAML 中的值
- CLI 中未指定（值为 None）的参数不会覆盖

### 7.2 路径创建的 Fail-Fast 策略

`PathConfig.__init__()` 中会检查：
- `sdk_path` 是否存在
- `baselines_path` 是否存在

如果不存在则立即抛出 `PathConfigError`，避免在训练中期才报错。子目录（training/checkpoint/ 等）在首次调用 `get_*_dir()` 时自动创建。

### 7.3 类型转换的安全性

`_validate_types()` 在类型转换失败时不会中断程序，而是记录警告并保留原始值。这是有意设计：因为 CLI 参数可能是可选的，一个错误的参数不应该阻止训练启动。

### 7.4 SDK 版本兼容

`BackendAdapter` 是 SDK 版本升级时需要重点关注的代码。如果 SDK 的 runtime API 变更，需要更新 `MatchRuntimeWrapper.resolve_turn()` 的实现，但上层代码（`AntWarEnv`, `BattleSimulator`）不受影响。

### 7.5 自定义配置

创建 `configs/custom_antwar.yaml` 并指定 `--config custom_antwar.yaml` 即可使用自定义配置。自定义配置只需要包含需要覆盖的字段，其他字段使用默认值。
