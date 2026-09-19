# SelfPlay 数据收集并行化技术方案

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
│  │  1. 选择 N 个对手                                     │   │
│  │  2. 序列化 self_agent 模型 + 对手 checkpoint 路径      │   │
│  │  3. 提交 N 个任务到 ProcessPoolExecutor               │   │
│  │  4. 收集 N 个 (EpisodeBatch, battle_details) 结果     │   │
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

---

## 4. 详细设计

### 4.1 新增模块：ParallelEpisodeCollector

文件路径：`ppo_v2/src/ppo_antwar/trainer/parallel_episode_collector.py`

```python
class ParallelEpisodeCollector:
    """并行 Episode 收集器 - 使用多进程并行运行多个 episode。

    职责：
    - 管理 Worker 进程池
    - 序列化/反序列化模型和数据
    - 提交并行收集任务
    - 收集并返回结果
    """

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        self._config = config
        self._n_workers = config.get("training", {}).get("parallel_workers", 4)
        self._device = device

    def collect_batch(
        self,
        self_agent: NeuralAgent,
        opponent_specs: List[Dict[str, Any]],
        self_first_list: List[bool],
        env_factory: Callable[[], AntWarEnv],
        episode_count_start: int,
        config: Dict[str, Any],
    ) -> List[Tuple[EpisodeBatch, Dict[str, Any]]]:
        """并行收集一批 episode。

        Args:
            self_agent: 当前 self_agent（主进程 GPU 上的模型）
            opponent_specs: 每个 episode 的对手规格列表
            self_first_list: 每个 episode 的先手标志列表
            env_factory: 环境工厂函数
            episode_count_start: 起始 episode 编号
            config: 训练配置

        Returns:
            列表，每个元素为 (EpisodeBatch, battle_details)
        """
        # 1. 序列化 self_agent 模型
        self_agent_bytes = self._serialize_self_agent(self_agent)

        # 2. 提交任务到进程池
        spawn_ctx = mp.get_context("spawn")
        results = []
        with ProcessPoolExecutor(
            max_workers=self._n_workers, mp_context=spawn_ctx
        ) as executor:
            futures = []
            for i, (opp_spec, self_first) in enumerate(
                zip(opponent_specs, self_first_list)
            ):
                future = executor.submit(
                    _run_episode_in_worker,
                    self_agent_bytes,
                    opp_spec,
                    self_first,
                    episode_count_start + i,
                    config,
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result(timeout=_EPISODE_TIMEOUT_SECONDS)
                results.append(result)

        return results
```

### 4.2 Worker 函数

```python
_EPISODE_TIMEOUT_SECONDS = 600  # 单 episode 超时

def _run_episode_in_worker(
    self_agent_bytes: bytes,
    opponent_spec: Dict[str, Any],
    self_first: bool,
    episode_count: int,
    config: Dict[str, Any],
) -> Tuple[EpisodeBatch, Dict[str, Any]]:
    """在子进程中运行单个 episode。

    Args:
        self_agent_bytes: 序列化的 self_agent 模型
        opponent_spec: 对手规格 {"checkpoint_path": str, "opponent_id": str}
        self_first: 是否先手
        episode_count: episode 编号
        config: 训练配置

    Returns:
        (EpisodeBatch, battle_details)
    """
    # 1. 反序列化 self_agent（CPU 推理）
    self_agent = _deserialize_self_agent(self_agent_bytes, device="cpu")

    # 2. 加载对手 agent（CPU 推理）
    opponent_agent = OpponentAgent.from_checkpoint(
        checkpoint_path=opponent_spec["checkpoint_path"],
        device="cpu",
        hidden_dim=config.get("network", {}).get("hidden_dim"),
        enable_auxiliary=config.get("ppo", {}).get("enable_auxiliary"),
    )

    # 3. 创建环境
    env = AntWarEnv(
        player_id=0,
        backend_type=config.get("env", {}).get("backend_type", "python"),
        prefer_native=config.get("env", {}).get("prefer_native", False),
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

### 4.3 模型序列化/反序列化

```python
import io

def _serialize_self_agent(self_agent: NeuralAgent) -> bytes:
    """将 self_agent 的模型序列化为 bytes。"""
    buffer = io.BytesIO()
    torch.save(self_agent._policy.state_dict(), buffer)
    return buffer.getvalue()

def _deserialize_self_agent(model_bytes: bytes, device: str = "cpu") -> NeuralAgent:
    """从 bytes 反序列化 self_agent。"""
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location=device, weights_only=True)

    policy = AntWarPolicyValueNetwork(
        hidden_dim=...,  # 从 config 获取
        enable_auxiliary=...,
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    return NeuralAgent(
        policy=policy,
        device=torch.device(device),
        exploration_epsilon=0.0,  # 收集时由 config 决定
    )
```

### 4.4 EpisodeBatch 序列化

当前 `EpisodeBatch` 包含 numpy 数组和 Python 列表，天然支持 pickle 序列化。`ProcessPoolExecutor` 使用 pickle 传输参数和返回值，因此 `EpisodeBatch` 无需额外修改。

验证要点：
- `observations_board`: `List[np.ndarray]` — pickle 可序列化
- `observations_global`: `List[np.ndarray]` — pickle 可序列化
- `observations_mask`: `List[np.ndarray]` — pickle 可序列化
- `aux_tower_damage/gold_income/base_damage`: `Optional[np.ndarray]` — pickle 可序列化
- `final_values`: `List[float]` — pickle 可序列化

### 4.5 SelfPlayTrainer 修改

修改 `SelfPlayTrainer.train()` 方法，将串行 episode 循环改为批量并行收集：

```python
def train(self, num_episodes, env_factory, callbacks=None):
    # ... 初始化代码不变 ...

    while self._episode_count < num_episodes:
        # 计算本批要运行的 episode 数
        episodes_this_batch = min(
            n_envs,
            num_episodes - self._episode_count
        )

        # 并行收集一批 episode
        results = self._collect_batch_parallel(
            episodes_this_batch, env_factory
        )

        # 处理每个 episode 的结果
        pending_stats = []
        for batch, battle_details, opponent_id, self_first in results:
            self._episode_count += 1
            self._batch_pool.append(batch)

            # 更新统计、日志、payoff（与现有逻辑相同）
            self._process_episode_result(
                batch, battle_details, opponent_id, self_first
            )
            pending_stats.append(battle_details)

        # PPO 更新
        self._flush_batch_to_update(pending_action_stats=pending_stats)

        # 周期性任务
        self._run_periodic_tasks()
```

新增方法：

```python
def _collect_batch_parallel(self, n_episodes, env_factory):
    """并行收集 n_episodes 个 episode。"""
    # 1. 预选所有对手
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

    # 2. 并行收集
    results = self._parallel_collector.collect_batch(
        self_agent=self._self_agent,
        opponent_specs=opponent_specs,
        self_first_list=self_first_list,
        env_factory=env_factory,
        episode_count_start=self._episode_count,
        config=self._config,
    )

    # 3. 组装返回结果（附加 opponent_id 和 self_first）
    assembled = []
    for i, (batch, battle_details) in enumerate(results):
        assembled.append((
            batch,
            battle_details,
            opponent_specs[i]["opponent_id"],
            self_first_list[i],
        ))

    return assembled
```

### 4.6 配置参数

在 `ppo_antwar.yaml` 中新增：

```yaml
training:
  # ... 现有参数 ...
  parallel_workers: 8          # 并行 Worker 数量（建议 = CPU 核心数 - 2）
  parallel_timeout: 600        # 单 episode 超时（秒）
  parallel_inference_device: "cpu"  # Worker 推理设备（固定为 cpu）
```

`parallel_workers` 默认值说明：
- 不默认等于 `n_envs`（20），因为 20 个进程可能超出 CPU 核心数
- 建议设为 `CPU 核心数 - 2`（留 2 核给主进程和系统）
- 当 `parallel_workers=1` 时退化为串行模式，便于调试

### 4.7 Worker 数与 n_envs 的关系

`n_envs` 决定每次 PPO 更新需要的 episode 数，`parallel_workers` 决定同时运行的 Worker 数。两者独立：

- `parallel_workers < n_envs`：Worker 池分多轮运行完 n_envs 个 episode（如 8 Worker 跑 20 episode，需 3 轮）
- `parallel_workers >= n_envs`：一轮即可跑完所有 episode
- 推荐配置：`parallel_workers = min(n_envs, cpu_count - 2)`

---

## 5. 数据正确性论证

### 5.1 PPO 重要性采样语义

当前串行流程中，同一批 n_envs 个 episode 使用相同的 self_agent 模型参数（因为 PPO 更新只在批结束后触发）。并行化后，同一批 episode 仍使用相同模型（通过序列化传递同一份 state_dict），因此 PPO 重要性采样的 `ratio = π_new / π_old` 语义完全不变。

### 5.2 对手选择

当前串行流程中，每个 episode 的对手选择依赖 `SelfPlayManager.select_opponent()`，该方法的返回值取决于对手池状态和 payoff 矩阵。并行化后，一批 episode 的对手在主进程中预选，确保对手选择逻辑不变。

注意：同一批中多个 episode 可能选中相同对手，这在串行模式下也可能发生（`exploit_prob < 1` 时有随机性），因此不是新问题。

### 5.3 Episode 顺序

并行化后，episode 的完成顺序可能与提交顺序不同（`as_completed` 不保证顺序）。但 PPO 训练不依赖 episode 的顺序——`EpisodeBatch.merge()` 只是简单拼接数据，顺序不影响 GAE 计算和 PPO 更新。

日志中的 episode 编号仍按提交顺序分配，确保可追溯。

### 5.4 联赛 Payoff 更新

当前串行流程中，每个 episode 结束后立即更新 payoff。并行化后，一批 episode 全部完成后统一更新 payoff。这意味着同一批中的 episode 无法看到彼此的 payoff 结果。

影响评估：payoff 更新是渐进式的统计量，单批内的延迟对整体训练影响可忽略。且当前 `exploit_prob=0.7`，对手选择本身就有随机性。

---

## 6. 与现有系统的集成

### 6.1 改动范围

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `trainer/parallel_episode_collector.py` | 新增 | 并行收集器 |
| `trainer/selfplay.py` | 修改 | `train()` 改为批量并行收集 |
| `configs/ppo_antwar.yaml` | 修改 | 新增并行配置参数 |
| `trainer/episode_collector.py` | 不变 | Worker 内部直接复用 |
| `trainer/batch.py` | 不变 | 天然支持 pickle |
| `trainer/ppo_trainer.py` | 不变 | PPO 更新逻辑不变 |
| `battle/opponent_agent.py` | 不变 | Worker 中使用 CPU 设备 |
| `env/antwar_env.py` | 不变 | 每个 Worker 独立创建 |

### 6.2 向后兼容

- `parallel_workers=1` 时退化为串行模式，行为与当前完全一致
- 所有现有配置参数不变，新增参数有默认值
- 日志格式不变，episode 编号连续

### 6.3 日志与监控

并行化后需注意的日志问题：

1. **Worker 内日志**：Worker 进程中的 `loguru` 日志默认不会输出到主进程的日志文件。解决方案：Worker 内的 `[ROUND]` 级别日志降为 `DEBUG`（或移除），关键信息通过 `battle_details` 返回主进程记录
2. **Episode 顺序**：日志中 episode 编号按提交顺序分配，不按完成顺序
3. **系统指标**：`SystemMetricsSampler` 仍在主进程运行，反映主进程资源使用

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
| 进程启动 | ~1s/Worker | spawn 模式首次启动 |
| 对手模型加载 | ~300ms/Worker | 从 checkpoint 加载 OpponentAgent |

以 `parallel_workers=8`、`n_envs=20` 为例：
- 每批额外开销：8 × (200ms + 300ms) + 50ms + 3轮 × 8 × 10ms ≈ 4.5 秒
- 每批数据收集时间：300 步 × 43ms / 8 × 3 轮 ≈ 4.8 秒（并行） vs 300 步 × 29ms × 20 = 174 秒（串行）
- 额外开销占比：4.5 / (4.8 + 4.5) ≈ 48%（首批），后续批次进程复用后降至 ~15%

**优化**：`ProcessPoolExecutor` 可复用 Worker 进程，避免重复启动。但每轮需重新加载模型（因为模型参数可能已更新）。可考虑在 Worker 内缓存模型，仅当 state_dict 变化时重新加载。

---

## 8. 风险与缓解

### 8.1 内存压力

**风险**：N 个 Worker 各加载 2 个模型（self + opponent），内存占用 = N × 2 × ~200MB = N × 400MB。8 Worker 需 ~3.2GB。

**缓解**：
- `parallel_workers` 默认值保守（8），用户可根据机器内存调整
- Worker 完成后立即释放模型和 env 资源
- 若内存不足，减少 `parallel_workers` 或使用模型共享（后续优化）

### 8.2 CPU 资源竞争

**风险**：N 个 Worker 同时运行引擎运算，可能超出 CPU 核心数，导致上下文切换开销。

**缓解**：
- `parallel_workers` 不超过 `cpu_count - 2`
- `ProcessPoolExecutor` 的 `max_workers` 参数控制并发度

### 8.3 Worker 异常

**风险**：Worker 中 episode 运行可能因引擎异常、模型加载失败等原因出错。

**处理**：遵循设计原则——不做容错，让错误尽早暴露。Worker 内异常通过 `future.result()` 传播到主进程，主进程捕获后记录日志并抛出，中断训练。

### 8.4 对手池一致性

**风险**：并行收集期间，主进程的 `SelfPlayManager` 状态不变（无 PPO 更新），但同一批中多个 episode 可能选中相同对手，payoff 更新延迟到批结束后。

**缓解**：这不是新问题——串行模式下也可能连续选中相同对手。payoff 更新延迟对训练影响可忽略。

---

## 9. 实施步骤

### 阶段一：核心并行化

1. 新增 `ParallelEpisodeCollector` 类
2. 实现 `_run_episode_in_worker` Worker 函数
3. 实现模型序列化/反序列化
4. 修改 `SelfPlayTrainer.train()` 使用并行收集
5. 新增配置参数

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
