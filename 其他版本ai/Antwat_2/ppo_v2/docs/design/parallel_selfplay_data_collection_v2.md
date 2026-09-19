# SelfPlay 数据收集并行化技术方案 v2

## 1. 背景与动机

当前 SelfPlay 训练模块的数据收集环节采用串行执行：每次只运行一个 episode，完成后再运行下一个。当累积够 `n_envs`（默认 20）个 episode 的数据后，触发一次 PPO 更新。

以 `total_episodes=2000`、`n_envs=20` 为例，共需收集 2000 个 episode，每个 episode 平均 300 步，串行总耗时约 4.5 小时。其中绝大部分时间消耗在 AntWar 引擎的 CPU 运算上，GPU 推理只占很小比例。

将数据收集改为并行执行，可以显著缩短训练总时长。

---

## 2. 瓶颈分析

### 2.1 单步时间拆解

SelfPlay 数据收集的每一步（step）包含以下操作：

| 操作 | 设备 | 估计耗时 | 占比 |
|------|------|----------|------|
| NeuralAgent 推理（self_agent） | GPU | ~3ms | ~13% |
| OpponentAgent 推理 | GPU/CPU | ~3-10ms | ~13-30% |
| AntWarEnv.step() → GameState.advance_round() | CPU | ~20ms | ~60-70% |
| 观测编码 + 奖励计算 + 快照采集 | CPU | ~3ms | ~10% |
| **单步总计** | | **~27-33ms** | |

### 2.2 引擎运算是瓶颈的代码级论证

`GameState.advance_round()`（[engine.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/backend/engine.py)）是纯 Python 的 CPU 密集型计算，单次调用包含：

1. **`_attack_ants()`**：塔攻击蚂蚁——遍历所有塔，计算攻击范围、目标选择、范围伤害、控制效果
2. **`_move_ants()`**：蚂蚁移动——调用 `_reverse_weighted_plan()` 执行 **Dijkstra 最短路径算法**，复杂度 O(V log V)。每个玩家调用 2 次（worker/combat），加上每个敌方塔 1 次，单回合可达 10+ 次 Dijkstra
3. **`_update_pheromone()`**：信息素更新——操作 (2, MAP_SIZE, MAP_SIZE) 的 numpy 数组
4. **`_refresh_static_risk_fields()`**：风险场刷新——遍历所有塔 × 所有可走格子，更新 (2, MAP_SIZE, MAP_SIZE) 的多个 float32 数组
5. **`_resolve_ant_lifecycle()`**：蚂蚁生命周期——攻击基地、死亡判定、老化
6. **`_spawn_ants()`**：蚂蚁生成
7. **`_tick_effects()`**：武器效果更新

引擎内部维护的大型 numpy 数组：

```
pheromone:                  (2, MAP_SIZE, MAP_SIZE) int32
damage_risk_field:          (2, MAP_SIZE, MAP_SIZE) float32
control_risk_field:         (2, MAP_SIZE, MAP_SIZE) float32
effect_pull_field:          (2, MAP_SIZE, MAP_SIZE) float32
enhanced_worker_costs:      (2, MAP_SIZE, MAP_SIZE) float32
enhanced_combat_base_costs: (2, MAP_SIZE, MAP_SIZE) float32
enhanced_traffic_field:     (2, MAP_SIZE, MAP_SIZE) float32
enhanced_reservations:      (2, MAP_SIZE, MAP_SIZE) float32
```

其中 `MAP_SIZE` 为地图尺寸（约 19-25），仅这些数组的内存占用就超过 100KB，且每回合都要读写。

### 2.3 GPU 推理不是瓶颈

- 网络结构：HexCNNEncoder（3 阶段六边形卷积）+ MLPEncoder + StructuredActionHead + ValueHead，hidden_dim=256
- 输入：board (1,28,19,19) + global (1,33) + action_mask (1,119)
- 单步推理时间 ~3ms（GPU），远小于引擎运算的 ~20ms
- 项目中已有 `profile_env_vs_gpu.py` 工具专门分析此比例，从工具设计可推断团队已确认引擎是瓶颈

### 2.4 计算规模估算

基于 `ppo_antwar.yaml` 配置：

| 参数 | 值 |
|------|-----|
| total_episodes | 2000 |
| n_envs | 20 |
| max_steps_per_episode | 512 |
| 平均每局步数 | ~300 |

**串行数据收集总耗时**：
- 单步 ~27ms × 300 步 × 2000 局 = ~16,200 秒 ≈ **4.5 小时**
- PPO 更新 100 次 × ~5 秒/次 = ~500 秒
- **总计约 4.6 小时**

**并行数据收集预估**（20 个 worker）：
- 单步 ~33ms（CPU 推理） × 300 步 / 20 并行 = ~5 秒/批
- 100 批 × 5 秒 = ~500 秒数据收集
- PPO 更新 ~500 秒
- **总计约 16-20 分钟**

**预估加速比：10-15x**

---

## 3. 方案设计

### 3.1 方案选型

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A. Episode 级进程并行** | 每个完整 episode 在独立子进程中运行，主进程收集结果 | 简单、与 BattleSimulator 模式一致、代码改动小 | Worker 使用 CPU 推理，单步稍慢 |
| B. Step 级混合并行 | 子进程运行引擎，主进程集中做 GPU 批量推理 | GPU 利用率高 | 通信复杂、需自定义同步协议、不同长度 episode 难以对齐 |
| C. 线程并行 | 多线程运行多个 env | 无序列化开销 | Python GIL 限制 CPU 并行，引擎运算无法真正并行 |

**选择方案 A**，理由：
1. 引擎是 CPU 瓶颈，必须用多进程绕过 GIL
2. 项目已有 `BattleSimulator` 使用 `ProcessPoolExecutor` + `spawn` 上下文的成熟模式
3. Episode 级并行天然隔离，无需处理步间同步
4. CPU 推理虽慢于 GPU，但引擎运算占比 ~60-70%，并行收益远大于推理降级损失
5. 符合设计原则：不做过度设计

### 3.2 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ SelfPlayTrainer│  │SelfPlayManager│  │   PPOTrainer    │  │
│  │  (编排逻辑)   │  │  (对手池管理) │  │  (GPU PPO更新)  │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│  ┌──────┴───────────────────────────────────────────────┐   │
│  │        ParallelEpisodeCollector (新增)                │   │
│  │                                                      │   │
│  │  1. 序列化 self_agent 模型 state_dict                 │   │
│  │  2. 提交 N 个任务到 ProcessPoolExecutor               │   │
│  │  3. 收集 N 个 (EpisodeBatch, battle_details, index)  │   │
│  │  4. 按 index 排序，保证结果顺序与提交顺序一致          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ spawn N workers
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Worker 0   │ │  Worker 1   │ │  Worker N-1 │
    │             │ │             │ │             │
    │ AntWarEnv   │ │ AntWarEnv   │ │ AntWarEnv   │
    │ SelfAgent   │ │ SelfAgent   │ │ SelfAgent   │
    │ (CPU推理)   │ │ (CPU推理)   │ │ (CPU推理)   │
    │ OpponentAgent│ │ OpponentAgent│ │ OpponentAgent│
    │ (CPU推理)   │ │ (CPU推理)   │ │ (CPU推理)   │
    │             │ │             │ │             │
    │ EpisodeCollector│ │ EpisodeCollector│ │ EpisodeCollector│
    └─────────────┘ └─────────────┘ └─────────────┘
```

### 3.3 核心设计决策

#### 3.3.1 为什么 Worker 使用 CPU 推理

1. **GPU 显存限制**：20 个 Worker 各加载一个模型到 GPU，需 ~4GB 显存（单模型 ~200MB），不现实
2. **CUDA 上下文不安全**：`fork` 后 CUDA 上下文不可用，必须用 `spawn`，每个子进程独立初始化 CUDA 开销大
3. **引擎是瓶颈**：CPU 推理 ~10ms vs 引擎 ~20ms，推理只占 ~33%，并行引擎运算的收益远大于推理降级的损失
4. **简化架构**：无需处理 GPU 资源竞争和跨进程 GPU 通信

#### 3.3.2 为什么选择 Episode 级并行而非 Step 级

1. **实现简单**：每个 Worker 独立运行完整 episode，无需步间同步
2. **天然隔离**：不同 episode 的 env 状态完全独立，无共享状态问题
3. **PPO 语义正确**：同一批 n_envs 个 episode 使用相同的 self_agent 模型，PPO 重要性采样语义不变（当前串行设计中，同一批内的 episode 也使用相同模型，因为 PPO 更新只在批结束后触发）
4. **已有先例**：`BattleSimulator` 已验证此模式可行

#### 3.3.3 spawn 上下文

沿用 `BattleSimulator` 的 `spawn` 模式：
- `fork` 在有 CUDA 上下文时不安全（可能导致死锁）
- `spawn` 创建全新进程，安全但稍慢（每个 Worker 需重新导入模块）
- Worker 生命周期与一批 episode 对齐，spawn 开销可接受

#### 3.3.4 进程池生命周期

进程池在 `ParallelEpisodeCollector` 构造时创建，训练结束时销毁。不在每批 `collect_batch()` 调用内创建/销毁，原因：

1. spawn 模式启动一个进程约 1s，每批创建/销毁 8 Worker 需 ~8s，100 批就是 ~800s 的纯启动开销
2. `ProcessPoolExecutor` 的 Worker 进程可跨批复用，避免重复导入模块
3. 每批仍需重新序列化 self_agent（PPO 更新后参数变了），但 Worker 进程本身可保留，只需重新加载 state_dict

---

## 4. 详细设计

### 4.1 新增模块：ParallelEpisodeCollector

文件路径：`ppo_v2/src/ppo_antwar/trainer/parallel_episode_collector.py`

```python
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ppo_antwar._agent.protocol import RecordingAgent
from ppo_antwar.battle.opponent_agent import OpponentAgent
from ppo_antwar.env.antwar_env import AntWarEnv
from ppo_antwar.network.ant_war_policy_value_network import (
    AntWarPolicyValueNetwork,
)
from ppo_antwar.trainer.batch import EpisodeBatch
from ppo_antwar.trainer.episode_collector import EpisodeCollector
from ppo_antwar.trainer.neural_agent import NeuralAgent


_EPISODE_TIMEOUT_SECONDS = 600  # 单 episode 超时


class ParallelEpisodeCollector:
    """并行 Episode 收集器 - 使用多进程并行运行多个 episode。

    职责：
    - 管理 Worker 进程池（跨批复用）
    - 序列化 self_agent 模型 state_dict
    - 提交并行收集任务
    - 收集并按提交顺序排序返回结果
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        training_cfg = config.get("training", {})
        self._n_workers = training_cfg.get("parallel_workers", 1)
        self._timeout = training_cfg.get("parallel_timeout", _EPISODE_TIMEOUT_SECONDS)

        # 预提取网络构造参数，避免 Worker 内重复解析 config
        self._hidden_dim = config.get("network", {}).get("hidden_dim", 256)
        self._enable_auxiliary = config.get("ppo", {}).get("enable_auxiliary", True)
        self._exploration_epsilon = config.get("ppo", {}).get("exploration_epsilon", 0.0)

        # 预提取环境构造参数
        env_cfg = config.get("env", {})
        self._env_player_id = env_cfg.get("player_id", 0)
        self._env_backend_type = env_cfg.get("backend_type", "python")
        self._env_prefer_native = env_cfg.get("prefer_native", False)

        # 进程池：构造时创建，跨批复用
        spawn_ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self._n_workers, mp_context=spawn_ctx
        )

    def collect_batch(
        self,
        self_agent: NeuralAgent,
        opponent_specs: List[Dict[str, Any]],
        self_first_list: List[bool],
        episode_count_start: int,
    ) -> List[Tuple[EpisodeBatch, Dict[str, Any]]]:
        """并行收集一批 episode。

        Args:
            self_agent: 当前 self_agent（主进程 GPU 上的模型）
            opponent_specs: 每个 episode 的对手规格列表，
                格式: [{"checkpoint_path": str, "opponent_id": str}, ...]
            self_first_list: 每个 episode 的先手标志列表
            episode_count_start: 起始 episode 编号

        Returns:
            按 episode 提交顺序排列的列表，每个元素为 (EpisodeBatch, battle_details)
        """
        # 1. 序列化 self_agent 的 state_dict
        self_agent_bytes = _serialize_self_agent(self_agent)

        # 2. 构造固定参数元组（所有 Worker 共享，避免重复序列化）
        worker_shared_args = (
            self_agent_bytes,
            self._hidden_dim,
            self._enable_auxiliary,
            self._exploration_epsilon,
            self._env_player_id,
            self._env_backend_type,
            self._env_prefer_native,
            self._config,
        )

        # 3. 提交任务
        futures = {}
        for i, (opp_spec, self_first) in enumerate(
            zip(opponent_specs, self_first_list)
        ):
            future = self._executor.submit(
                _run_episode_in_worker,
                worker_shared_args,
                opp_spec,
                self_first,
                episode_count_start + i,
            )
            # 用字典映射 future → 提交索引，保证结果按顺序排列
            futures[future] = i

        # 4. 收集结果，按提交索引排序
        indexed_results: List[Optional[Tuple[EpisodeBatch, Dict[str, Any]]]] = [
            None
        ] * len(opponent_specs)
        try:
            for future in as_completed(futures, timeout=self._timeout):
                idx = futures[future]
                # Worker 内部异常会在此处抛出（已被 pickle 传播到主进程）
                result = future.result()
                indexed_results[idx] = result
        except TimeoutError as e:
            # as_completed 全局超时：某个 Worker 在 self._timeout 内未完成
            timed_out_indices = [
                futures[f] for f in futures if not f.done()
            ]
            raise RuntimeError(
                f"Worker 超时（{self._timeout}s）："
                f"episode 索引 {timed_out_indices} 未完成。"
                f"可能原因：引擎死循环、CPU 资源严重竞争、或超时配置过短。"
            ) from e
        finally:
            # 终止所有仍在运行的 Worker 进程，防止孤儿进程
            self._terminate_running_workers()

        return indexed_results

    def _terminate_running_workers(self):
        """终止进程池中所有仍在运行的 Worker 进程。

        场景：
        - collect_batch 中有 Worker 超时或异常时调用
        - 确保孤儿进程被清理，不影响下一批次的调度

        实现方式：
        - ProcessPoolExecutor 没有暴露直接终止单个 Worker 的 API
        - 调用 executor.shutdown(wait=False, cancel_futures=True) 关闭当前池
        - 重新创建新的 ProcessPoolExecutor，保证后续批次有干净的 Worker 池

        注意：shutdown(wait=False) 后内部 Worker 进程会被 join，
        但已提交但未开始的任务会被取消（cancel_futures=True）。
        正在执行的任务对应的进程最终会被 terminate。
        """
        self._executor.shutdown(wait=False, cancel_futures=True)
        # 重建进程池，恢复后续批次的收集能力
        spawn_ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self._n_workers, mp_context=spawn_ctx
        )

    def close(self):
        """关闭进程池，释放资源。在训练结束时调用。

        无需额外处理池已被重建的情况：如果 _terminate_running_workers
        已被调用，当前 self._executor 是重建后的池，shutdown(wait=True)
        仍然安全（空池也会正常关闭）。
        """
        self._executor.shutdown(wait=True)
```

**与 v1 方案的关键差异**：

1. **结果按提交顺序返回**：使用 `futures` 字典映射 future → 索引，`as_completed` 收集后按索引填入有序列表，确保 `opponent_id`/`self_first` 与结果的对应关系正确
2. **进程池跨批复用**：`ProcessPoolExecutor` 在 `__init__` 时创建，`close()` 时销毁，避免每批的 spawn 开销
3. **预提取参数**：网络构造参数（`hidden_dim`、`enable_auxiliary`、`exploration_epsilon`）和环境参数在构造时提取，作为固定参数传给 Worker，避免 Worker 内重复解析 config
4. **去掉 `env_factory` 参数**：`env_factory` 是 Callable，不可 pickle 序列化。Worker 内根据 config 直接构造 `AntWarEnv`（见 4.2 节论证）

### 4.2 Worker 函数

```python
def _run_episode_in_worker(
    worker_shared_args: tuple,
    opponent_spec: Dict[str, Any],
    self_first: bool,
    episode_count: int,
) -> Tuple[EpisodeBatch, Dict[str, Any]]:
    """在子进程中运行单个 episode。

    Args:
        worker_shared_args: 所有 Worker 共享的固定参数元组：
            (self_agent_bytes, hidden_dim, enable_auxiliary,
             exploration_epsilon, env_player_id, env_backend_type,
             env_prefer_native, config)
        opponent_spec: 对手规格 {"checkpoint_path": str, "opponent_id": str}
        self_first: 是否先手
        episode_count: episode 编号

    Returns:
        (EpisodeBatch, battle_details)
    """
    (
        self_agent_bytes,
        hidden_dim,
        enable_auxiliary,
        exploration_epsilon,
        env_player_id,
        env_backend_type,
        env_prefer_native,
        config,
    ) = worker_shared_args

    # 1. 反序列化 self_agent（CPU 推理）
    self_agent = _deserialize_self_agent(
        self_agent_bytes,
        device="cpu",
        hidden_dim=hidden_dim,
        enable_auxiliary=enable_auxiliary,
        exploration_epsilon=exploration_epsilon,
    )

    # 2. 加载对手 agent（CPU 推理）
    opponent_agent = OpponentAgent.from_checkpoint(
        checkpoint_path=opponent_spec["checkpoint_path"],
        device=torch.device("cpu"),
        hidden_dim=hidden_dim,
        enable_auxiliary=enable_auxiliary,
    )

    # 3. 创建环境
    env = AntWarEnv(
        player_id=env_player_id,
        backend_type=env_backend_type,
        prefer_native=env_prefer_native,
    )

    # 4. 创建 EpisodeCollector 并运行
    collector = EpisodeCollector(
        config=config,
        episode_count_ref=lambda: episode_count,
        self_agent=self_agent,
    )

    try:
        batch, battle_details = collector.collect(
            env=env,
            opponent_agent=opponent_agent,
            opponent_id=opponent_spec["opponent_id"],
            self_first=self_first,
        )
    finally:
        env.close()

    return batch, battle_details
```

**关键设计说明**：

1. **`env_factory` 的处理**：v1 方案中 `collect_batch` 接受 `env_factory: Callable[[], AntWarEnv]` 参数，但 Callable 不可 pickle 序列化，无法传给子进程。Worker 内改为根据 `config["env"]` 中的参数直接构造 `AntWarEnv`。当前项目中 `env_factory` 的实现也是从 config 构造 `AntWarEnv`，因此行为完全一致。

2. **`EpisodeCollector` 的初始化参数**：`EpisodeCollector.__init__` 需要 3 个参数：
   - `config: Dict[str, Any]` — 传入完整 config，Worker 内复用
   - `episode_count_ref: Callable[[], int]` — 使用 `lambda: episode_count`，仅用于 Worker 内日志
   - `self_agent: RecordingAgent` — 传入反序列化后的 NeuralAgent（它继承自 RecordingAgent）
   - `max_steps_per_episode` 从 `config["training"]["max_steps_per_episode"]` 读取（默认 512），无需额外传递

3. **`episode_count_ref` 在 Worker 内的行为**：当前 `EpisodeCollector` 中 `episode_count_ref` 仅用于 `[ROUND]` 和 `[TERMINAL_1ST_ROUND]` 级别日志。Worker 内这些日志不写入主进程（见 6.3 节），但保留 `episode_count_ref` 以避免修改 `EpisodeCollector` 接口。

### 4.3 模型序列化/反序列化

```python
def _serialize_self_agent(self_agent: NeuralAgent) -> bytes:
    """将 self_agent 的模型 state_dict 序列化为 bytes。

    选择序列化 state_dict 而非整个 agent 对象（cloudpickle），原因：
    1. state_dict 更小（~200MB vs 整个 agent 含计算图可能更大）
    2. 反序列化时可精确控制 device（强制 CPU）
    3. 不依赖 cloudpickle，使用 PyTorch 原生序列化

    与 BattleSimulator 使用 cloudpickle 的差异说明：
    BattleSimulator 需要序列化 SDK BaselineAgent（动态加载的类，必须用 cloudpickle）。
    而 self_agent 是固定的 NeuralAgent + AntWarPolicyValueNetwork，
    其类定义在模块顶层，可被 spawn 后的子进程正常 import，
    因此 state_dict 序列化更安全、更高效。
    """
    buffer = io.BytesIO()
    torch.save(self_agent._policy.state_dict(), buffer)
    return buffer.getvalue()


def _deserialize_self_agent(
    model_bytes: bytes,
    device: str = "cpu",
    hidden_dim: int = 256,
    enable_auxiliary: bool = True,
    exploration_epsilon: float = 0.0,
) -> NeuralAgent:
    """从 bytes 反序列化 self_agent。

    Args:
        model_bytes: 序列化的 state_dict bytes
        device: 推理设备（Worker 内固定为 "cpu"）
        hidden_dim: 网络隐藏层维度，从 config["network"]["hidden_dim"] 提取
        enable_auxiliary: 是否启用辅助任务头，从 config["ppo"]["enable_auxiliary"] 提取
        exploration_epsilon: 探索 epsilon，从 config["ppo"]["exploration_epsilon"] 提取
    """
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location=device, weights_only=True)

    policy = AntWarPolicyValueNetwork(
        hidden_dim=hidden_dim,
        enable_auxiliary=enable_auxiliary,
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    return NeuralAgent(
        policy=policy,
        device=torch.device(device),
        exploration_epsilon=exploration_epsilon,
    )
```

**与 v1 方案的差异**：

1. **`hidden_dim` 和 `enable_auxiliary` 不再是占位符**：从 `ParallelEpisodeCollector` 构造时预提取，作为固定参数传给 Worker，再传给 `_deserialize_self_agent`
2. **`exploration_epsilon` 从 config 读取**：v1 方案硬编码为 `0.0`，但实际训练中 `exploration_epsilon=0.02`（从 `ppo_antwar.yaml` 的 `ppo.exploration_epsilon` 读取），Worker 内应使用与主进程相同的值
3. **`OpponentAgent.from_checkpoint` 的 device 参数**：改为 `torch.device("cpu")`（字符串 `"cpu"` 也能工作，但 `OpponentAgent.from_checkpoint` 内部传给 `NeuralAgent` 的 `device` 参数类型为 `torch.device`，保持类型一致）

### 4.4 EpisodeBatch 序列化

当前 `EpisodeBatch` 包含 numpy 数组和 Python 列表，天然支持 pickle 序列化。`ProcessPoolExecutor` 使用 pickle 传输参数和返回值，因此 `EpisodeBatch` 无需额外修改。

验证要点：
- `observations_board`: `List[np.ndarray]` — pickle 可序列化
- `observations_global`: `List[np.ndarray]` — pickle 可序列化
- `observations_mask`: `List[np.ndarray]` — pickle 可序列化
- `actions`: `List[int]` — pickle 可序列化
- `rewards`: `List[float]` — pickle 可序列化
- `values`: `List[float]` — pickle 可序列化
- `log_probs`: `List[float]` — pickle 可序列化
- `dones`: `List[bool]` — pickle 可序列化
- `aux_tower_damage/gold_income/base_damage`: `Optional[np.ndarray]` — pickle 可序列化
- `aux_valid_mask`: `Optional[np.ndarray]` — pickle 可序列化
- `final_value`: `float` — pickle 可序列化
- `final_values`: `List[float]` — pickle 可序列化

**`EpisodeBatch.merge()` 与并行收集的兼容性**：

`merge()` 的合并逻辑是纯数据拼接：
- 列表字段：`list.extend()` 拼接
- aux 字段：`np.concatenate(axis=0)` 沿第 0 维拼接
- `final_values`：从各子 batch 收集（若 `final_values` 非空则 extend，否则 fallback 取 `final_value`）
- `final_value`：取最后一个子 batch 的值

并行收集后，`SelfPlayTrainer` 将多个 `EpisodeBatch` 追加到 `_batch_pool`，由 `_flush_batch_to_update` 中的 `EpisodeBatch.merge(batch_pool)` 统一合并。`final_values` 的收集逻辑正确——每个 Worker 返回的 `EpisodeBatch` 中 `final_values = [final_value]`（单个 episode 的 bootstrap 值），merge 后拼接为所有 episode 的 bootstrap 值列表，供 GAE 计算使用。

### 4.5 battle_details 数据结构

`EpisodeCollector._build_battle_details()` 返回的 `battle_details` 字典包含以下键：

| 键 | 类型 | 说明 | 主进程消费方式 |
|---|---|---|---|
| `"reward"` | `float` | 累计 episode 总奖励 | 写入日志、滑动窗口统计 |
| `"length"` | `int` | episode 步数 | 写入日志、滑动窗口统计 |
| `"rounds"` | `int` | 环境的 round_count | 写入日志 |
| `"action_counts"` | `Dict[str, int]` | 各动作类型的计数 | 聚合为 `pending_action_stats` |
| `"action_rewards"` | `Dict[str, float]` | 各动作类型的累计奖励 | 聚合为 `pending_action_stats` |
| `"reward_sources"` | `Dict[str, float]` | 8 个奖励分项的累计值 | 聚合写入 PPO 训练上下文 |
| `"snapshots"` | `List[Dict]` | 每步的快照列表 | 提取终局 HP/金币、计算 max_coin/total_income |

每个 snapshot 字典包含：`"round_idx"`, `"own_tower_hp"`, `"enemy_tower_hp"`, `"own_coins"`, `"enemy_coins"`

**并行化后 battle_details 的传递**：Worker 返回的 `(EpisodeBatch, battle_details)` 通过 pickle 传回主进程，所有字段均由 numpy/Python 原生类型组成，可正确序列化。主进程按顺序接收后，对每个 episode 的 `battle_details` 执行与串行模式完全相同的处理。

### 4.6 SelfPlayTrainer 修改

#### 4.6.1 `__init__` 修改

```python
def __init__(self, trainer: PPOTrainer, config: Dict[str, Any],
             path_config: PathConfig, run_id: str = "default"):
    # ... 现有初始化不变 ...

    # 新增：创建并行收集器
    training_cfg = config.get("training", {})
    parallel_workers = training_cfg.get("parallel_workers", 1)
    if parallel_workers > 1:
        self._parallel_collector = ParallelEpisodeCollector(config)
        self._use_parallel = True
    else:
        # parallel_workers=1 时退化为串行模式
        self._parallel_collector = None
        self._use_parallel = False
```

#### 4.6.2 `train()` 方法修改

将串行 episode 循环改为批量并行收集。**核心原则**：并行收集只替换数据收集环节，其余所有逻辑（日志、payoff 更新、回调、周期性任务）保持在主进程中按 episode 顺序执行。

```python
def train(self, num_episodes: int, env_factory: Callable[[], AntWarEnv],
          callbacks=None) -> None:
    self._total_episodes = num_episodes
    self._callbacks = callbacks
    # ... 初始化代码与现有一致 ...
    self._episode_count = 0
    self._last_battle_result = {"wins": 0, "losses": 0, "draws": 0}
    self._create_initial_opponents()
    opponent_win_stats: Dict[str, Dict[str, int]] = {}
    total_steps = 0

    try:
        while self._episode_count < num_episodes:
            n_envs = self._config.get("training", {}).get("n_envs", 20)
            episodes_this_batch = min(
                n_envs, num_episodes - self._episode_count
            )

            if self._use_parallel:
                # 并行路径
                episode_results = self._collect_batch_parallel(
                    episodes_this_batch
                )
            else:
                # 串行路径（parallel_workers=1 时走此分支）
                episode_results = self._collect_batch_serial(
                    episodes_this_batch, env_factory
                )

            # 按顺序处理每个 episode 的结果
            pending_stats = []
            for result in episode_results:
                self._episode_count += 1
                (batch, battle_details, opponent_id,
                 self_first, episode_start_time) = result

                self._logging.time_tracker.start_training()
                self._logging.time_tracker.end_training()
                self._logging.time_tracker.record_episode(
                    time.time() - episode_start_time
                )

                self._batch_pool.append(batch)

                # 处理 episode 结果（日志、payoff、统计）
                pending_info = self._process_episode_result(
                    battle_details, opponent_id, self_first
                )
                pending_stats.append(pending_info)

                total_steps += battle_details["length"]

            # PPO 更新
            self._flush_batch_to_update(
                pending_action_stats=pending_stats if pending_stats else None
            )

            # 周期性任务（基于 episode_count 触发）
            self._run_periodic_tasks()

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        self._finalize_training(total_steps, opponent_win_stats)
        if self._parallel_collector is not None:
            self._parallel_collector.close()
```

#### 4.6.3 `_collect_batch_parallel()` 新增方法

```python
def _collect_batch_parallel(self, n_episodes: int):
    """并行收集 n_episodes 个 episode。

    Returns:
        按 episode 提交顺序排列的列表，每个元素为
        (batch, battle_details, opponent_id, self_first, episode_start_time)
    """
    # 1. 预选所有对手（在主进程中，保证 SelfPlayManager 的状态一致性）
    opponent_specs = []
    self_first_list = []
    for i in range(n_episodes):
        ep_idx = self._episode_count + i + 1
        self_first = ep_idx % 2 == 1
        opponent_id = self._selfplay_manager.select_opponent()
        checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(
            opponent_id
        )
        opponent_specs.append({
            "checkpoint_path": checkpoint_path,
            "opponent_id": opponent_id,
        })
        self_first_list.append(self_first)

    # 2. 更新联赛进度（所有 episode 共享同一进度值）
    self._selfplay_manager.update_progress(self._episode_count + 1)

    # 3. 并行收集（记录开始时间用于后续统计）
    episode_start_time = time.time()
    results = self._parallel_collector.collect_batch(
        self_agent=self._self_agent,
        opponent_specs=opponent_specs,
        self_first_list=self_first_list,
        episode_count_start=self._episode_count,
    )

    # 4. 组装返回结果（附加 opponent_id、self_first、episode_start_time）
    #    results 已按提交顺序排列（由 ParallelEpisodeCollector 保证）
    assembled = []
    for i, (batch, battle_details) in enumerate(results):
        assembled.append((
            batch,
            battle_details,
            opponent_specs[i]["opponent_id"],
            self_first_list[i],
            episode_start_time,
        ))

    return assembled
```

#### 4.6.4 `_collect_batch_serial()` 新增方法

为 `parallel_workers=1` 时提供退化的串行路径，行为与现有 `_run_single_episode` 完全一致：

```python
def _collect_batch_serial(self, n_episodes: int, env_factory):
    """串行收集 n_episodes 个 episode（parallel_workers=1 时的退化路径）。

    Returns:
        列表，每个元素为
        (batch, battle_details, opponent_id, self_first, episode_start_time)
    """
    results = []
    for i in range(n_episodes):
        self._episode_count += 1
        self._logging.time_tracker.start_training()
        episode_start_time = time.time()

        self._selfplay_manager.update_progress(self._episode_count)
        self_first = self._episode_count % 2 == 1
        opponent_id, opponent_agent = self._select_opponent_agent()
        self._logging.selfplay_logger.log_battle_start(
            self._episode_count, opponent_id, self_first
        )

        pending_info = self._collect_and_update_payoff(
            opponent_agent, opponent_id, self_first, env_factory,
            episode_start_time
        )
        pending_info["opponent_id"] = opponent_id

        self._logging.time_tracker.end_training()
        self._logging.time_tracker.record_episode(
            time.time() - episode_start_time
        )
        self._logging.selfplay_logger.log_battle_end(
            self._episode_count, opponent_id,
            pending_info.get("battle_result", "unknown"),
            pending_info.get("reward", 0),
            pending_info.get("length", 0),
        )

        # 从 pending_info 重建 batch 和 battle_details
        batch = self._batch_pool[-1]  # _collect_and_update_payoff 已追加
        self._batch_pool.pop()  # 先弹出，让上层统一管理

        results.append((
            batch,
            pending_info,
            opponent_id,
            self_first,
            episode_start_time,
        ))

    # 回退 episode_count，让上层 while 循环统一递增
    self._episode_count -= n_episodes
    return results
```

**注意**：串行路径保留用于调试和回归测试。当确认并行路径正确后，可考虑移除串行路径，由 `parallel_workers=1` 的并行路径替代（此时 `ProcessPoolExecutor(max_workers=1)` 等效于串行，但仍有 spawn 开销）。

#### 4.6.5 `_process_episode_result()` 新增方法

从 `_collect_and_update_payoff()` 中抽取 episode 结果处理逻辑，供并行/串行路径共用：

```python
def _process_episode_result(
    self,
    battle_details: Dict[str, Any],
    opponent_id: Optional[str],
    self_first: bool,
) -> Dict[str, Any]:
    """处理单个 episode 的结果（日志、payoff、统计）。

    从现有 _collect_and_update_payoff() 中抽取，不包含数据收集和 env 管理。

    Args:
        battle_details: EpisodeCollector.collect() 返回的对战详情
        opponent_id: 对手 ID
        self_first: 是否先手

    Returns:
        pending_info 字典，包含 action_counts, action_rewards, reward_sources,
        reward, length, battle_result 等
    """
    reward = battle_details["reward"]
    length = battle_details["length"]

    # 滑动窗口统计
    self._episode_rewards.append(reward)
    self._episode_lengths.append(length)
    max_stats = self._config.get("training", {}).get("max_episode_stats", 1000)
    if len(self._episode_rewards) > max_stats:
        self._episode_rewards = self._episode_rewards[-max_stats:]
        self._episode_lengths = self._episode_lengths[-max_stats:]

    # 判定胜负
    snapshots = battle_details.get("snapshots", [])
    if snapshots:
        final = snapshots[-1]
        if final["own_tower_hp"] > final["enemy_tower_hp"]:
            result_str = "win"
        elif final["own_tower_hp"] < final["enemy_tower_hp"]:
            result_str = "loss"
        else:
            result_str = "draw"
    else:
        result_str = "unknown"

    # 终局数据
    end_own_hp = snapshots[-1]["own_tower_hp"] if snapshots else 0
    end_enemy_hp = snapshots[-1]["enemy_tower_hp"] if snapshots else 0
    end_own_coins = snapshots[-1]["own_coins"] if snapshots else 0
    end_enemy_coins = snapshots[-1]["enemy_coins"] if snapshots else 0
    max_own_coins = EpisodeCollector.compute_max_coin(snapshots, "self")
    max_enemy_coins = EpisodeCollector.compute_max_coin(snapshots, "opponent")
    total_own_income = EpisodeCollector.compute_total_coin_income(snapshots, "self")
    total_enemy_income = EpisodeCollector.compute_total_coin_income(snapshots, "opponent")

    # 写入对战日志
    battle_record = EpisodeRecord(
        episode=self._episode_count,
        opponent_id=opponent_id or "none",
        self_first=self_first,
        reward=reward,
        length=length,
        result=result_str,
        rounds=battle_details.get("rounds", length),
        end_own_hp=end_own_hp,
        end_enemy_hp=end_enemy_hp,
        end_own_coins=end_own_coins,
        end_enemy_coins=end_enemy_coins,
        max_own_coins=max_own_coins,
        max_enemy_coins=max_enemy_coins,
        total_own_income=total_own_income,
        total_enemy_income=total_enemy_income,
        action_counts=battle_details.get("action_counts", {}),
        action_rewards=battle_details.get("action_rewards", {}),
        reward_sources=battle_details.get("reward_sources", {}),
    )
    self._logging.battle_logger.write_episode(battle_record)
    self._logging.ep_batch_logger.on_episode_complete(
        self._episode_count, battle_record
    )

    # 更新 payoff
    if opponent_id is not None:
        result_val = 1 if result_str == "win" else (-1 if result_str == "loss" else 0)
        self._selfplay_manager.update_payoff(opponent_id, result_val)

    # 对战开始/结束日志
    self._logging.selfplay_logger.log_battle_start(
        self._episode_count, opponent_id, self_first
    )
    self._logging.selfplay_logger.log_battle_end(
        self._episode_count, opponent_id, result_str, reward, length
    )

    return {
        "action_counts": battle_details.get("action_counts", {}),
        "action_rewards": battle_details.get("action_rewards", {}),
        "reward_sources": battle_details.get("reward_sources", {}),
        "reward": reward,
        "length": length,
        "battle_result": result_str,
        "opponent_id": opponent_id,
    }
```

### 4.7 配置参数

在 `ppo_antwar.yaml` 的 `training` 节中新增：

```yaml
training:
  total_episodes: 2000
  save_interval: 20
  opponent_update_interval: 100
  n_envs: 20
  battle_interval: 20
  n_battles: 25
  max_steps_per_episode: 512
  # --- 新增并行参数 ---
  parallel_workers: 8          # 并行 Worker 数量（1 = 串行模式）
  parallel_timeout: 600        # 单 episode 超时（秒）
```

**设计决策**：

1. **不添加 `parallel_inference_device` 参数**：Worker 推理设备固定为 CPU，这是由架构决定的（见 3.3.1），不是用户可配置项。暴露此参数会误导用户尝试设置 `"cuda"`，但实际上多 Worker 共享 GPU 不可行。

2. **`parallel_workers` 默认值说明**：
   - 不默认等于 `n_envs`（20），因为 20 个进程可能超出 CPU 核心数
   - 建议设为 `CPU 核心数 - 2`（留 2 核给主进程和系统）
   - 当 `parallel_workers=1` 时，`SelfPlayTrainer` 走串行路径，行为与现有完全一致

3. **`parallel_timeout` 默认 600 秒**：单 episode 最大 512 步 × 43ms ≈ 22 秒，600 秒超时留有充足余量应对 CPU 资源竞争导致的减速。

### 4.8 Worker 数与 n_envs 的关系

`n_envs` 决定每次 PPO 更新需要的 episode 数，`parallel_workers` 决定同时运行的 Worker 数。两者独立：

- `parallel_workers < n_envs`：Worker 池分多轮运行完 n_envs 个 episode（如 8 Worker 跑 20 episode，需 3 轮：8 + 8 + 4）
- `parallel_workers >= n_envs`：一轮即可跑完所有 episode
- 推荐配置：`parallel_workers = min(n_envs, cpu_count - 2)`

**关键**：`ProcessPoolExecutor` 内部会自动调度——提交 20 个任务到 `max_workers=8` 的池中，前 8 个立即执行，后续任务在 Worker 空闲后自动分配。无需手动分轮。

---

## 5. 数据正确性论证

### 5.1 PPO 重要性采样语义

当前串行流程中，同一批 n_envs 个 episode 使用相同的 self_agent 模型参数（因为 PPO 更新只在批结束后触发）。并行化后，同一批 episode 仍使用相同模型（通过序列化传递同一份 state_dict），因此 PPO 重要性采样的 `ratio = π_new / π_old` 语义完全不变。

### 5.2 对手选择

当前串行流程中，每个 episode 的对手选择依赖 `SelfPlayManager.select_opponent()`，该方法的返回值取决于对手池状态和 payoff 矩阵。并行化后，一批 episode 的对手在主进程中预选，确保对手选择逻辑不变。

注意：同一批中多个 episode 可能选中相同对手，这在串行模式下也可能发生（`exploit_prob < 1` 时有随机性），因此不是新问题。

### 5.3 Episode 顺序

并行化后，episode 的完成顺序可能与提交顺序不同（`as_completed` 不保证顺序）。但 PPO 训练不依赖 episode 的顺序——`EpisodeBatch.merge()` 只是简单拼接数据，顺序不影响 GAE 计算和 PPO 更新。

日志中的 episode 编号仍按提交顺序分配，确保可追溯。`ParallelEpisodeCollector` 通过 `futures` 字典映射保证结果按提交顺序返回。

### 5.4 联赛 Payoff 更新

当前串行流程中，每个 episode 结束后立即更新 payoff。并行化后，一批 episode 全部完成后统一更新 payoff。这意味着同一批中的 episode 无法看到彼此的 payoff 结果。

影响评估：payoff 更新是渐进式的统计量，单批内的延迟对整体训练影响可忽略。且当前 `exploit_prob=0.7`，对手选择本身就有随机性。

### 5.5 联赛进度更新

当前串行流程中，每个 episode 开始前调用 `selfplay_manager.update_progress(episode_count)`。并行化后，一批 episode 共享同一进度值（批次开始时的 `episode_count`），不逐 episode 更新。

影响评估：`update_progress` 仅影响 exploit 概率的渐变（随训练进度从 explore 偏移向 exploit），单批内的进度差异（如 episode 101-120 都使用 episode 101 的进度值）对训练影响可忽略。

---

## 6. 与现有系统的集成

### 6.1 改动范围

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `trainer/parallel_episode_collector.py` | 新增 | 并行收集器 + Worker 函数 + 序列化工具 |
| `trainer/selfplay.py` | 修改 | 新增 `_collect_batch_parallel`、`_collect_batch_serial`、`_process_episode_result`；修改 `train()` 主循环 |
| `configs/ppo_antwar.yaml` | 修改 | `training` 节新增 `parallel_workers`、`parallel_timeout` |
| `trainer/episode_collector.py` | 不变 | Worker 内部直接复用，无需修改 |
| `trainer/batch.py` | 不变 | 天然支持 pickle，merge 逻辑不变 |
| `trainer/ppo_trainer.py` | 不变 | PPO 更新逻辑不变 |
| `battle/opponent_agent.py` | 不变 | Worker 中使用 `torch.device("cpu")` |
| `env/antwar_env.py` | 不变 | 每个 Worker 独立创建 |
| `league/selfplay_manager.py` | 不变 | 对手选择和 payoff 更新仍在主进程 |
| `trainer/logging_subsystem.py` | 不变 | 所有日志仍在主进程写入 |

### 6.2 向后兼容

- `parallel_workers` 未配置或为 1 时，走串行路径，行为与现有完全一致
- 所有现有配置参数不变，新增参数有默认值
- 日志格式不变，episode 编号连续
- `SelfPlayTrainer` 的公共接口（`train()`、`__init__`）签名不变

### 6.3 日志与监控

并行化后需注意的日志问题：

1. **Worker 内日志**：Worker 进程中的 `loguru` 日志默认不会输出到主进程的日志文件。`EpisodeCollector.collect()` 中包含 `[ROUND]` 和 `[TERMINAL_1ST_ROUND]` 级别的详细日志，这些在 Worker 内输出到 stderr（loguru 默认行为），不写入主进程日志文件。关键统计信息通过 `battle_details` 返回主进程，由 `_process_episode_result()` 统一写入日志。

2. **Episode 顺序**：`ParallelEpisodeCollector` 保证结果按提交顺序返回，日志中 episode 编号与提交顺序一致。

3. **系统指标**：`SystemMetricsSampler` 仍在主进程运行，反映主进程资源使用。Worker 进程的资源使用不被监控，这是可接受的——Worker 是短生命周期的 CPU 任务，主进程资源（GPU 显存、PPO 更新）才是关键瓶颈。

4. **`SelfPlayLogger.log_battle_start/end` 的调用时机**：串行模式下在每个 episode 开始/结束时调用。并行模式下，在一批 episode 全部完成后，`_process_episode_result()` 中按顺序对每个 episode 调用，确保日志顺序正确。

### 6.4 回调机制

当前 `callbacks` 仅在 `_flush_batch_to_update()` 中使用，当 PPO 更新成功后调用 `callbacks.on_step()`。并行化后此行为不变——回调仍在主进程中 PPO 更新后触发，不涉及 Worker。

### 6.5 周期性任务

`_run_periodic_tasks()` 基于 `self._episode_count` 取模触发。并行化后，一批 episode 处理完毕后 `_episode_count` 已递增到正确值，周期性任务的触发逻辑不变。

---

## 7. 性能预估

### 7.1 理论加速比

设：
- T_engine = 20ms（引擎单步耗时）
- T_infer_gpu = 3ms（GPU 推理单步耗时）
- T_infer_cpu = 10ms（CPU 推理单步耗时）
- T_other = 3ms（观测编码 + 奖励 + 快照）
- N = parallel_workers
- S = 平均每局步数（~300）

**串行模式单步**：T_serial = T_infer_gpu + T_infer_gpu + T_engine + T_other = 3 + 3 + 20 + 3 = 29ms

**并行模式单步（Worker 内）**：T_worker = T_infer_cpu + T_infer_cpu + T_engine + T_other = 10 + 10 + 20 + 3 = 43ms

**并行模式等效单步**：T_parallel = T_worker / N

**加速比**：T_serial / T_parallel = T_serial × N / T_worker = 29 × N / 43 ≈ 0.67 × N

| parallel_workers | 理论加速比 | 预估训练总时长 |
|------------------|-----------|---------------|
| 1 (串行) | 1x | ~4.6 小时 |
| 4 | ~2.7x | ~1.7 小时 |
| 8 | ~5.4x | ~51 分钟 |
| 12 | ~8.1x | ~34 分钟 |
| 16 | ~10.7x | ~26 分钟 |

实际加速比会因进程启动开销、序列化开销、CPU 资源竞争等因素略低于理论值，预计实际可达理论值的 70-80%。

### 7.2 额外开销

| 开销项 | 估计值 | 说明 |
|--------|--------|------|
| 模型序列化 | ~50ms/批 | 序列化 state_dict 到 bytes |
| 模型反序列化 | ~200ms/Worker | 加载 state_dict + 模型初始化 |
| EpisodeBatch 序列化 | ~10ms/episode | pickle 序列化 numpy 数组 |
| 对手模型加载 | ~300ms/Worker | 从 checkpoint 加载 OpponentAgent |

以 `parallel_workers=8`、`n_envs=20` 为例：
- 每批额外开销：50ms（序列化）+ 20 × (200ms + 300ms)（反序列化+加载）+ 20 × 10ms（结果传输）≈ 10.3 秒（首批）
- 后续批次：Worker 进程已存在，但每批仍需重新加载模型（因为 PPO 更新后参数变了），开销与首批相同
- 每批数据收集时间：300 步 × 43ms × 20 / 8 ≈ 32.3 秒（并行）vs 300 步 × 29ms × 20 = 174 秒（串行）
- 额外开销占比：10.3 / (32.3 + 10.3) ≈ 24%

---

## 8. 风险与缓解

### 8.1 内存压力

**风险**：N 个 Worker 各加载 2 个模型（self + opponent）+ AntWarEnv 引擎状态，内存占用 = N × (2 × ~200MB + ~50MB 引擎) ≈ N × 450MB。8 Worker 需 ~3.6GB。

**缓解**：
- `parallel_workers` 默认值保守（8），用户可根据机器内存调整
- Worker 完成后由 `ProcessPoolExecutor` 自动回收资源
- 若内存不足，减少 `parallel_workers`

### 8.2 CPU 资源竞争

**风险**：N 个 Worker 同时运行引擎运算，可能超出 CPU 核心数，导致上下文切换开销。

**缓解**：
- `parallel_workers` 不超过 `cpu_count - 2`
- `ProcessPoolExecutor` 的 `max_workers` 参数控制并发度

### 8.3 Worker 异常

#### 8.3.1 异常分类

Worker 运行中可能出现三类异常：

| 异常类型 | 触发场景 | `future.result()` 行为 |
|----------|----------|------------------------|
| **Worker 内部异常** | 模型加载失败、引擎异常、pickle 反序列化失败等 | 抛出 `Exception`（原始异常被 pickle 传播到主进程） |
| **Worker 超时** | 引擎死循环、CPU 竞争导致极端延迟 | `as_completed` 抛出 `TimeoutError` |
| **Worker 进程崩溃** | OOM killed、segfault、系统信号 | 抛出 `BrokenProcessPool` 或 `pickle.UnpicklingError` |

#### 8.3.2 核心问题：超时后 Worker 进程不会被自动终止

`future.result(timeout=...)` 和 `as_completed(timeout=...)` 的语义是**仅中断主进程的等待**，不会 kill 正在执行任务的 Worker 进程。超时后 Worker 变成孤儿进程，继续占用 CPU 和内存。

这意味着：
- 超时的 Worker 中的 `AntWarEnv` 资源不会被释放
- 孤儿进程与 `ProcessPoolExecutor` 的调度失去关联，后续批次可能因 Worker 数不足而阻塞
- 多次超时累积后，孤儿进程耗尽系统资源

#### 8.3.3 处理策略

遵循设计原则——**不做容错，让错误尽早暴露**。所有三类异常统一处理：**立即终止训练**。不尝试跳过失败 episode 继续训练，原因：

1. PPO 批数据不完整会导致 GAE 计算和优势估计偏差，训练语义不正确
2. 跳过 episode 后 payoff 矩阵和 episode 计数不一致，引入难以排查的隐含 Bug
3. 超时通常意味着严重问题（引擎死循环、资源耗尽），不应静默跳过

#### 8.3.4 异常处理实现

`collect_batch` 中的异常处理逻辑（见 4.1 节代码）：

1. **`as_completed(futures, timeout=self._timeout)`**：将超时设在 `as_completed` 而非 `future.result()`，因为 `as_completed` 的 `timeout` 参数控制的是整个迭代过程的等待时间，一旦任何一个 future 在此时间内未完成即抛出 `TimeoutError`
2. **`try/except TimeoutError`**：捕获超时后，找出未完成的 future 索引，包装为 `RuntimeError` 并附上诊断信息
3. **`finally: self._terminate_running_workers()`**：无论成功还是异常，都确保清理仍在运行的 Worker 进程

#### 8.3.5 `_terminate_running_workers` 实现

`ProcessPoolExecutor` 不暴露 Worker PID，无法精确 kill 特定进程。因此采用 **shutdown + 重建** 策略：

1. 调用 `executor.shutdown(wait=False, cancel_futures=True)` 关闭当前池
2. 重新创建 `ProcessPoolExecutor`，恢复后续批次的收集能力

设计决策：
- **为什么不用 `os.kill(pid, SIGKILL)`**：无法获取 Worker PID，强行通过内部属性 `_processes` 获取属于私有 API
- **为什么 `shutdown(wait=False)`**：超时的 Worker 可能在运行无限循环，`shutdown(wait=True)` 会无限阻塞
- **重建池的代价**：spawn 一个进程约 1s，8 Worker 约 8s。相比超时本身（600s），重建代价可忽略

#### 8.3.6 异常场景完整行为表

| 场景 | `as_completed` 行为 | `collect_batch` 行为 | Worker 进程 | 下游代码 |
|------|---------------------|---------------------|-------------|----------|
| 所有 Worker 正常完成 | 正常迭代 | 返回完整 `indexed_results` | 空闲等待下一批 | 正常解包 |
| Worker 内部异常 | 该 future 标记 done，迭代到时 `future.result()` 抛异常 | 异常传播，`finally` 终止池中剩余 Worker | 被 shutdown 清理 | 不会执行到下游 |
| Worker 超时 | `as_completed` 超时抛 `TimeoutError` | 抛 `RuntimeError`，`finally` 重建池 | 被 shutdown 清理 | 不会执行到下游 |
| Worker 进程崩溃 | 该 future 标记 done，`future.result()` 抛 `BrokenProcessPool` | 异常传播，`finally` 重建池 | 已崩溃 | 不会执行到下游 |
| 多个 Worker 部分成功 | 先收集已完成结果，遇到首个异常即中断 | 已收集结果丢弃，整体失败 | 被 shutdown 清理 | 不会执行到下游 |

#### 8.3.7 `close` 方法兼容性

`close()` 需要处理池已被 `_terminate_running_workers` 重建的情况。由于重建后 `self._executor` 指向新的 `ProcessPoolExecutor`，`shutdown(wait=True)` 仍然安全——空池也会正常关闭。

### 8.4 对手池一致性

**风险**：并行收集期间，主进程的 `SelfPlayManager` 状态不变（无 PPO 更新），但同一批中多个 episode 可能选中相同对手，payoff 更新延迟到批结束后。

**缓解**：这不是新问题——串行模式下也可能连续选中相同对手。payoff 更新延迟对训练影响可忽略。

---

## 9. 实施步骤

### 阶段一：核心并行化

1. 新增 `ParallelEpisodeCollector` 类（含 `_run_episode_in_worker`、`_serialize_self_agent`、`_deserialize_self_agent`）
2. 修改 `SelfPlayTrainer.__init__()`：根据 `parallel_workers` 配置决定是否创建并行收集器
3. 修改 `SelfPlayTrainer.train()`：将 while 循环改为批量收集 + 顺序处理
4. 新增 `_collect_batch_parallel()`、`_collect_batch_serial()`、`_process_episode_result()` 方法
5. 在 `ppo_antwar.yaml` 中新增 `parallel_workers`、`parallel_timeout` 参数

### 阶段二：验证与调优

1. 对比串行/并行训练的 PPO 指标（loss、reward、entropy）一致性
2. 运行 `profile_env_vs_gpu.py` 验证瓶颈分析
3. 调优 `parallel_workers` 参数
4. 验证 `parallel_workers=1` 时行为与串行完全一致

### 阶段三：优化（可选）

1. Worker 进程复用 + 模型增量更新（避免每批重新加载模型）
2. 批量推理优化：主进程 GPU 批量推理 + Worker 仅运行引擎（Step 级混合并行）
3. 共享内存优化：减少序列化开销

---

## 10. 限制与后续方向

### 当前方案限制

1. Worker 使用 CPU 推理，单步耗时比 GPU 推理慢约 3x
2. 每批需序列化/反序列化模型，有额外开销
3. Worker 内日志无法直接写入主进程日志文件

### 后续优化方向

1. **Step 级混合并行**：Worker 仅运行引擎，主进程集中做 GPU 批量推理。适用于 GPU 推理占比增大的场景（如更大模型）
2. **Ray 框架**：如果后续需要分布式训练，可迁移到 Ray，支持跨机器并行
3. **引擎 C++ 加速**：从根本上优化引擎性能（Dijkstra 算法 C++ 化、numpy 操作向量化）

---

## 附录 A：v1 → v2 变更记录

| 项 | v1 方案 | v2 方案 | 变更原因 |
|---|---|---|---|
| 结果顺序 | `as_completed` 后按完成顺序返回，用 `enumerate(results)` 索引 `opponent_specs[i]` | 用 `futures` 字典映射 future → 提交索引，按索引填入有序列表 | v1 存在逻辑 Bug：`as_completed` 不保证顺序，`enumerate(results)` 的索引不是提交索引，会导致 opponent_id 与 episode 结果错配 |
| 进程池生命周期 | `collect_batch` 内用 `with` 创建/销毁 | `__init__` 时创建，`close()` 时销毁，跨批复用 | 每批 spawn 开销 ~1s/Worker，100 批 × 8 Worker = ~800s 纯启动开销 |
| `env_factory` 参数 | `collect_batch` 接受 `env_factory` | 去掉，Worker 内根据 config 构造 `AntWarEnv` | Callable 不可 pickle 序列化，无法传给子进程 |
| `hidden_dim`/`enable_auxiliary` | `_deserialize_self_agent` 中写 `...`（占位符） | 从 `ParallelEpisodeCollector` 构造时预提取，作为固定参数传给 Worker | 消除占位符，避免 Worker 内重复解析 config |
| `exploration_epsilon` | 硬编码 `0.0` | 从 `config["ppo"]["exploration_epsilon"]` 读取（默认 0.02） | 训练中 Worker 应使用与主进程相同的探索 epsilon |
| `OpponentAgent.from_checkpoint` 的 device | 传字符串 `"cpu"` | 传 `torch.device("cpu")` | 保持与 `NeuralAgent.__init__` 的 `device` 参数类型一致 |
| `parallel_inference_device` 配置项 | 有此配置项 | 移除 | Worker 推理设备固定为 CPU 是架构决策，不是用户可配置项 |
| `_process_episode_result` | 不存在，逻辑内嵌于 `_collect_and_update_payoff` | 抽取为独立方法 | 并行/串行路径共用 episode 结果处理逻辑 |
| battle_details 结构 | 未定义 | 详列 7 个键及类型 | 确保 Worker → 主进程的数据传递无歧义 |
| 内存估算 | N × 400MB | N × 450MB（含引擎状态 ~50MB） | 补充 AntWarEnv 引擎自身的内存消耗 |
| Worker 异常处理 | 未说明 `env.close()` 的异常路径 | 明确 `try/finally: env.close()` | 确保异常时 env 资源被释放 |
| 超时与孤儿进程处理 | 仅简单说"让错误尽早暴露"，未分析超时后行为 | 详列 3 类异常、孤儿进程问题、shutdown+重建策略、完整行为表 | `future.result(timeout=...)` 和 `as_completed(timeout=...)` 不会 kill Worker 进程，超时后 Worker 变孤儿进程，必须主动清理 |
| `_terminate_running_workers` | 不存在 | 新增：shutdown(wait=False, cancel_futures=True) + 重建池 | 确保异常/超时后 Worker 进程被清理，后续批次有干净的 Worker 池 |
| 超时位置 | `future.result(timeout=self._timeout)` 逐 future 超时 | `as_completed(futures, timeout=self._timeout)` 全局超时 | `as_completed` 的 timeout 控制整个迭代等待时间，语义更正确；逐 future 超时需等前一个 future 完成后才开始计时，无法检测首个挂起的 Worker |
