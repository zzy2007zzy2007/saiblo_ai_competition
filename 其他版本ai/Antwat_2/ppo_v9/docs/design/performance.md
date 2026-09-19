# ppo_v9 遗传算法训练系统并行化改进设计

## 1. 当前瓶颈分析

### 1.1 时间分布估算

以默认配置（`ga_evo.yaml`）为基准分析单代训练耗时：

| 阶段 | 子阶段 | 调用次数 | 每次对战场次 | 总对战场次 | 估算耗时 | 占比 |
|------|--------|----------|-------------|-----------|---------|------|
| 种群评估 | 个体 × 锚定参照物 | 80 × 3 = 240 次 `run_battles` | 2 局/次 | 480 局 | ~60 min | ~86% |
| 种群评估 | 个体 × 动态参照物 | 80 × 2 = 160 次 `run_battles` | 2 局/次 | 320 局 | ~40 min |  |
| 基线验证 | 种子 × 基线 agent | 8 × 3 = 24 次 `run_battles` | 24 局/次 | 288 局 | ~10 min | ~14% |
| 种群生成 | 杂交+变异 | 一次性 | — | — | <1 min | <1% |
| **单代合计** | | | | **~1088 局** | **~110 min** | 100% |

> **说明**：以上为基于 15 秒/局的保守估算（含游戏逻辑 + 网络推理 + SDK 开销）。实际耗时因硬件、`max_steps` 等参数而异。

**200 代完整训练预估**：200 × 110 min ≈ **367 小时（约 15 天）**，其中种群评估占约 **86%**（~316 小时）。

### 1.2 串行瓶颈代码路径

瓶颈的核心在 `battle_selection.py:94-184` 的 `evaluate_population()` 方法：

```python
# 当前实现（battle_selection.py 第 128-143 行）
for individual in population.individuals:        # ⬅ 瓶颈 1：80 个个体串行
    self.elo.ensure_player(individual.id)
    ga_agent = self._create_ga_agent(individual)  # ⬅ 瓶颈 2：每次新建网络 + to(device)

    for ref in references:                         # ⬅ 瓶颈 3：5 个参照物串行
        ref_agent = ref.agent_factory(1)           # ⬅ 瓶颈 4：动态参照物每次从磁盘加载 checkpoint
        battle_results = self._simulator.run_battles(
            ga_agent, ref_agent, n_battles=1       # 仅 2 局，ProcessPool 利用率极低
        )
        # 更新 ELO（串行，但非主要瓶颈）
```

**四个关键问题**：

1. **外层循环串行**：80 个个体逐个评估，每个等上一个完成才能开始。
2. **内层循环串行**：5 个参照物逐个评估，5 次串行 `run_battles`。
3. **每次重建网络**：`_create_ga_agent()` 中 `new AntWarPolicyValueNetwork(...).to(device)` 涉及 ~15 层参数组的 GPU 分配，每个个体执行一次。
4. **动态参照物重复加载**：`_create_reference_from_checkpoint()` 返回的闭包每次调用都执行 `OpponentAgent.from_checkpoint()`（磁盘 I/O + 网络新建 + `torch.load` + `to(device)`），在 80 个个体的循环中被调用 80 次。

### 1.3 当前 ProcessPoolExecutor 的低效利用

`battle_simulator.py:55-91` 已使用 `ProcessPoolExecutor(max_workers=12)`，但每次 `run_battles(n_battles=1)` 仅提交 2 个任务（先手+后手），12 个 worker 中 10 个闲置。这相当于用 12 个进程跑 2 个任务，进程创建/销毁的固定开销占比过高。

---

## 2. 设计方案

### 2.1 方案 A（简单）：个体级并行

**思路**：将外层 `for individual` 循环改为 `ProcessPoolExecutor`，每个 worker 子进程独立完成"一个个体 vs 所有参照物"的评估。

```
主进程                          Worker 1              Worker 2              Worker N
  │                               │                     │                     │
  ├─ 构建任务列表                  │                     │                     │
  │  [(ind_1, refs), (ind_2, refs), ...]               │                     │
  │                               │                     │                     │
  ├─ ProcessPoolExecutor ─────────┤                     │                     │
  │  submit(ind_1, refs) ────────→│ 创建网络             │                     │
  │  submit(ind_2, refs) ────────→│ for ref in refs:     │ 创建网络             │
  │  submit(ind_3, refs) ────────→│   run_battles()      │ for ref in refs:   │
  │  ...                          │   更新局部 ELO        │   run_battles()    │
  │                               │ return (ind, elo)    │   ...              │
  │  as_completed() ──────────────│                     │ return (ind, elo)  │
  │  合并结果                      │                     │                     │
```

**复杂度**：低 —— 仅改动一个循环。

**风险**：低 —— 个体间无共享状态，`Individual` 和 `Genome` 均为可 pickle 的 dataclass。

**局限性**：
- Worker 内部仍是串行遍历 5 个参照物（400 次 `run_battles` 调用不变）
- 每个 worker 仍需独立创建网络实例
- 动态参照物的 checkpoint 仍在 worker 内重复加载
- 不能利用 GPU（CUDA tensor 不可跨进程共享，子进程必须用 CPU 或 `spawn` + 各自初始化 CUDA）

**预估加速比**：
- 若 workers = 12：80 个个体 → 7 波并发，种群评估从 ~100 min → ~10 min
- 基线验证（8 个种子）同理：~10 min → ~2 min
- 单代从 ~110 min → ~12 min，约 **9× 加速**
- 200 代从 15 天 → ~1.7 天

### 2.2 方案 B（中等）：任务级并行 + 网络实例池 + 参照物缓存

**思路**：将 (个体, 参照物) 二元组打平为任务列表，所有任务通过进程池并行提交；同时引入网络实例池和参照物缓存消除重复创建。

```
主进程                                    Worker 池 (N 进程)
  │                                         │
  ├─ 预创建参照物 Agent（缓存，仅一次）        │
  │  序列化所有参照物为 bytes ─────────────────→ 广播给所有 worker
  │                                         │
  ├─ 构建任务列表                             │
  │  tasks = [(genome_1, ref_1),             │
  │           (genome_1, ref_2),             │
  │           ...                            │
  │           (genome_80, ref_5)]            │
  │  = 400 个任务                            │
  │                                         │
  ├─ ProcessPoolExecutor ───────────────────┤
  │  submit all 400 tasks ──────────────────→ Worker 逐个领取任务：
  │                                         │   1. 从网络池获取网络实例
  │  as_completed() ←───────────────────────│   2. load_genome_to_network(net, genome)
  │  收集结果 → 批量更新 ELO                  │   3. 反序列化参照物 agent
  │                                         │   4. run_battles(ga_agent, ref_agent)
  │                                         │   5. 归还网络实例到池
  │                                         │   6. 返回对战结果
```

**复杂度**：中等 —— 需要新增任务分发器、网络池、参照物缓存、ELO 批量更新。

**风险**：中等 —— 进程池 + GPU 设备的处理需要仔细设计（见 4.5 节）。

**预估加速比**：
- 400 任务 / 12 workers = 34 波并发
- 但每个任务内仍有 BattleSimulator 的 12-worker 子池，需改为单局执行模式以提升粒度
- 若任务粒度细化到单局（400 × 2 = 800 局），12 workers → 67 波，每波 ~15 秒 → ~17 分钟
- 相比方案 A，进一步减少了 worker 内串行等待
- 单代从 ~110 min → ~8 min（含基线验证并行），约 **14× 加速**
- 200 代从 15 天 → ~1.1 天

### 2.3 方案 C（高级）：GPU 批量推理（暂不推荐）

**思路**：将多个基因组加载到 batch 维度，一次前向传播同时推理多个个体。结合对弈引擎的批量模拟执行。

**复杂度**：高 —— 需要：
- 修改游戏引擎，支持批量模拟（多局同时推进）
- 网络需支持 batch > 1 的推理（当前已支持，但游戏引擎不批量化）
- 对战编排逻辑全面重构

**风险**：高 —— 游戏引擎是 C++ SDK，批量改造涉及 native 层；投入产出比在当前阶段不划算。

**决策**：暂不采用。

---

## 3. 推荐方案：方案 B（任务级并行 + 网络池 + 参照物缓存）

### 3.1 选择理由

| 维度 | 方案 A | 方案 B | 方案 C |
|------|--------|--------|--------|
| 加速比 | ~9× | ~14× | ~40×+（理论） |
| 代码改动量 | 小（~50 行） | 中（~300 行） | 大（~2000 行+引擎改造） |
| 风险 | 低 | 中 | 高 |
| GPU 利用率 | 无（CPU only） | 可选（ThreadPool + CUDA stream） | 高 |
| 可维护性 | 高 | 中 | 低 |
| 投产时间 | 1-2 天 | 3-5 天 | 2-4 周 |

**推荐方案 B**，理由如下：

1. **加速比显著**：14× 加速意味着 200 代训练从 15 天缩短到 ~1 天，从"不可接受"变为"可接受"。
2. **复杂度可控**：核心改动集中在新增一个 `TaskDispatcher` 模块 + 修改 `BattleSelection` 和 `BaselineEvaluator` 的评估循环，不改动核心算法。
3. **向后兼容**：通过配置开关控制并行/串行模式，出问题可一键回退。
4. **为方案 C 铺路**：任务级并行架构是批量推理的自然前身，任务粒度细化后容易过渡。

### 3.2 方案 B 的关键设计决策

#### 3.2.1 进程池 vs 线程池

| 维度 | ProcessPoolExecutor | ThreadPoolExecutor |
|------|---------------------|-------------------|
| CUDA 支持 | ❌ CUDA tensor 不可跨进程 | ✅ 同进程可共享 CUDA context |
| GIL | 无影响（独立进程） | CUDA 操作不阻塞 GIL |
| 内存开销 | 每进程独立 Python 解释器 | 轻量 |
| pickle 开销 | 需序列化 genome（numpy array） | 无需序列化（共享引用） |
| 调试难度 | 较高 | 低（共享地址空间） |

**决策：优先使用 `ThreadPoolExecutor` + CUDA stream 分离**。

理由：GA 推理的计算量主要在 GPU 上（CNN 编码器推理），CUDA 操作会释放 GIL，因此 Python 线程可真正并行执行 GPU 推理。即使部分游戏逻辑受 GIL 影响（如 SDK backend 的 Python 实现），瓶颈仍在 GPU 推理而非 Python 逻辑。

若 ThreadPool 实测加速不理想（GIL 成为瓶颈），则降级为 ProcessPool + CPU 推理（每个子进程在 CPU 上运行网络）。

#### 3.2.2 网络实例池

```
┌─────────────────────────────────────────────┐
│                 NetworkPool                 │
│                                             │
│  _pool: Queue[AntWarPolicyValueNetwork]     │
│  _device: torch.device                      │
│                                             │
│  acquire() → AntWarPolicyValueNetwork       │
│     从队列取出 → 若无空闲则阻塞等待           │
│                                             │
│  release(network) → None                    │
│     归还到队列（不重置参数，下次 load 会覆盖） │
│                                             │
│  预创建 N = max_workers 个网络实例           │
│  所有实例共享同一 CUDA device               │
└─────────────────────────────────────────────┘
```

工作流：
1. 初始化时创建 N 个网络实例（如 4 个），放入池中。
2. Worker 线程评估一个 (genome, ref) 任务时：`net = pool.acquire()` → `load_genome_to_network(net, genome)` → 创建 GAWarAgent → 对战 → `pool.release(net)`。
3. `load_genome_to_network()` 是 in-place 参数复制（`load_state_dict(strict=False)`），不创建新 tensor，随时间开销可忽略（~10ms vs 网络创建 ~200ms）。

**注意**：`AntWarPolicyValueNetwork` 通过其子模块 `HexCNNEncoder`（policy 和 value 各一个）包含共 **8 个 `BatchNorm2d` 层**。`load_genome_to_network` 使用 `strict=False`，因为 genome 仅编码 `named_parameters()`（weight、bias、LayerNorm 的 weight/bias），不包含 BatchNorm 的 buffer 键（`running_mean`、`running_var`、`num_batches_tracked`）。加载后，网络的可训练参数被正确覆盖，但 BatchNorm 的 running stats **保持加载前的值**。

**对推理的影响**：BatchNorm 在 `eval()` 模式下使用 `running_mean`/`running_var` 做归一化，而非当前 batch 的统计量。如果网络是新实例化的（即网络池中的实例），running stats 为默认初始值（mean=0, var=1），这与训练时的实际统计分布存在偏差。在 GA 场景中：

- **这不是并行化引入的新问题**——当前串行实现中，`_create_ga_agent()` 每次 `new AntWarPolicyValueNetwork(...).to(device)` 同样是新实例，BatchNorm 初始值问题已经存在。
- **实际影响取决于网络对 BatchNorm 统计量的敏感度**——若 CNN 编码器对归一化偏移容忍度较高，影响可能有限；否则可能导致个体实力评估存在系统性偏差（所有个体同等受影响，相对排名可能不变）。
- **网络池不会加剧该问题**——因为所有网络实例初始状态相同，加载不同 genome 后 BatchNorm buffer 保持一致。

**改进方向**（后续迭代考虑）：
1. 将 BatchNorm 的 running stats 纳入 genome 编码（增大 genome 体积 ~8 个 1D tensor × O(128~512)，影响可忽略），并在 `load_genome_to_network` 中同时加载。
2. 或通过网络预热：加载 genome 后，用一个 dummy 输入跑几次前向传播，让 running stats 收敛（需评估收敛速度和效果）。
3. 或替换 BatchNorm 为 `nn.LayerNorm`（`HexCNNEncoder` 中的 `MLPEncoder` 已使用 LayerNorm，但 CNN 编码器使用 BatchNorm 有其结构合理性——2D 特征图归一化）。

当前 GA 训练已在此条件下运行，该问题不阻塞并行化改造。网络池只需关注参数复用，无需额外处理 BatchNorm buffer。

#### 3.2.3 参照物 Agent 缓存

当前问题：

```python
# genetic_evolution_trainer.py:207-208
ref = ReferenceAgent(
    name=opponent_id,
    agent_factory=lambda pid, cp=checkpoint_path: ...  # 闭包捕获 checkpoint_path
)
# 此闭包在 evaluate_population 中被调用 80 次（每个个体一次）
# ref.agent_factory(1) 每次执行 → OpponentAgent.from_checkpoint() → torch.load + to(device)
```

**改进方案**：在 `BattleSelection.evaluate_population()` 开始时，预创建所有参照物 Agent 实例，存入 dict：

```python
# 伪代码
ref_agents_cache: Dict[str, Any] = {}
for ref in references:
    if ref.name not in ref_agents_cache:
        ref_agents_cache[ref.name] = ref.agent_factory(1)  # 仅创建一次

# 后续所有评估复用缓存的 agent
# 通过 AgentLoader.serialize/deserialize 在进程间传递
```

对于静态参照物（BasicRandomAI、BasicTowerAI、MediumRuleAI），它们是纯 Python 规则 agent，创建成本为 0。缓存收益主要体现在动态参照物的消除重复 `torch.load`。

对于进程池方案，参照物 Agent 需序列化为 bytes（`AgentLoader.serialize`，底层 `cloudpickle.dumps`），在 worker 中反序列化。规则 agent 序列化成本极低；动态参照物（PPOWarAgent 包装了 OpponentAgent + 网络）序列化体积较大但仅需一次。

**实现位置**：在 `BattleSelection.__init__()` 或 `evaluate_population()` 开头创建缓存，在 `evaluate_population()` 结束后清理（释放 GPU 显存）。

#### 3.2.4 ELO 评分的并发处理

当前 ELO 更新是顺序的：

```python
self.elo.update(individual.id, ref.name, 1.0)  # 修改 self._ratings dict
```

在并行环境下，多线程同时调用 `elo.update()` 会导致竞态条件（dict 的读写非原子）。

**方案：延迟 ELO 更新（batch ELO update）**

```
并行阶段：只收集对战结果，不更新 ELO
  results = [
    (individual_id, ref_name, outcome),  # outcome ∈ {1.0, 0.5, 0.0}
    ...
  ]

串行阶段（所有对战完成后）：批量更新 ELO
  for ind_id, ref_name, outcome in results:
      elo.update(ind_id, ref_name, outcome)
```

**正确性分析**：
- ELO 是路径依赖的（每次更新依赖当前评分），但在一代内所有比赛可视为"同时发生"。
- 如果严格按照 "先收集所有结果，再按某种顺序批量更新" 的方式，不同顺序会产生不同的最终 ELO 值。
- 但这不影响排名的正确性，因为 ELO 在 GA 中的用途是**代内相对排名**（选 top-N 种子），而非绝对精度。
- 更接近"同时比赛"语义的方案：在对战前拍快照（固定初始 ELO），根据所有结果一次性计算：

```python
# 修正方案：基于初始 ELO 计算最终 ELO
for ind_id in individuals:
    ra = elo_initial
    for ref_name, outcome in ind_results[ind_id]:
        rb = ref_elo_map[ref_name]  # 参照物 ELO 固定
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        ra += k_factor * (outcome - ea)
    final_elo[ind_id] = ra
```

这等价于按固定参照物 ELO 计算，结果是确定性的（与顺序无关），且语义正确。

**实现**：在 `TaskDispatcher` 中收集所有对战结果，然后由 `BattleSelection` 的串行方法计算最终 ELO。

#### 3.2.5 GPU 设备处理（ThreadPool 方案）

使用 `ThreadPoolExecutor` 时，所有线程共享同一进程的内存空间和 CUDA context。关键问题：

1. **CUDA 默认流是串行的**：多个线程向默认 CUDA stream 提交 kernel 会排队执行。需要为每个线程创建独立的 CUDA stream。

```python
import torch

class NetworkPool:
    def __init__(self, n_networks: int, device: torch.device):
        self._pool = queue.Queue(maxsize=n_networks)
        for i in range(n_networks):
            net = AntWarPolicyValueNetwork(...).to(device)
            net.eval()
            # 为每个网络绑定独立的 CUDA stream
            net._cuda_stream = torch.cuda.Stream(device=device)
            self._pool.put(net)
    
    def acquire(self):
        net = self._pool.get()
        # 切换到此网络的 stream
        torch.cuda.set_stream(net._cuda_stream)
        return net
    
    def release(self, net):
        # 同步此 stream（确保操作完成）
        net._cuda_stream.synchronize()
        self._pool.put(net)
```

但这在 PyTorch 中并不简单——`torch.cuda.set_stream()` 影响的是当前线程的默认 stream，而 `net.forward()` 内部的操作不一定会使用该 stream（取决于具体实现）。

**更简单的方案**：使用 `torch.cuda.stream()` context manager 包装推理：

```python
with torch.cuda.stream(net._cuda_stream):
    action_id = net.get_action(board, global_vec, action_mask, deterministic=True)
```

**实测建议**：先用 ThreadPoolExecutor 不带 stream 隔离的简单实现测试加速效果。若 GPU 利用率低（多个线程在默认流上排队），再引入 stream 隔离。CUDA MPS（Multi-Process Service）也可考虑，但增加了运维复杂度。

**备选方案**：如果 ThreadPool + CUDA 实测不理想，切换为 ProcessPool + CPU 推理。CPU 推理虽然单局较慢（~2-5× vs GPU），但 12 进程并行可弥补，总吞吐量可能相近。

---

## 4. 方案 B 详细设计

### 4.1 新增模块：`trainer/task_dispatcher.py`

```python
# ppo_v9/src/ppo_ga/trainer/task_dispatcher.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

@dataclass
class EvalTask:
    """单个评估任务"""
    individual_id: str
    genome: Genome                       # 基因组数据
    reference_name: str                  # 参照物名称
    reference_agent_bytes: bytes         # 预序列化的参照物 agent

@dataclass  
class EvalResult:
    """单个评估结果"""
    individual_id: str
    reference_name: str
    outcomes: List[float]               # [1.0, 0.5, 0.0] × 2 局
    errors: List[str]                    # 错误信息


class NetworkPool:
    """网络实例池 —— 复用网络实例，消除重复创建开销
    
    特性：
    - 有界容量（= n_networks），防止显存膨胀
    - 带超时的 acquire，防止死锁导致训练永久挂起
    - release 前校验网络完整性
    """
    
    def __init__(self, n_networks: int, hidden_dim: int, 
                 enable_auxiliary: bool, device: torch.device,
                 acquire_timeout: float = 120.0):
        self._pool = queue.Queue(maxsize=n_networks)
        self._n_networks = n_networks
        self._acquire_timeout = acquire_timeout
        self._device = device
        self._in_use = 0  # 追踪借出数量，辅助死锁诊断
        
        for i in range(n_networks):
            net = AntWarPolicyValueNetwork(
                hidden_dim=hidden_dim,
                enable_auxiliary=enable_auxiliary,
            ).to(device)
            net.eval()
            self._pool.put(net)
    
    def acquire(self) -> AntWarPolicyValueNetwork:
        """从池中获取网络实例。
        
        若所有实例均在占用中，阻塞等待直到超时。
        
        Raises:
            queue.Empty: 等待超时（默认 120 秒），说明可能存在死锁或任务卡死。
        """
        try:
            net = self._pool.get(timeout=self._acquire_timeout)
            self._in_use += 1
            return net
        except queue.Empty:
            raise queue.Empty(
                f"NetworkPool.acquire() timeout after {self._acquire_timeout}s. "
                f"In use: {self._in_use}/{self._n_networks}. "
                f"Possible deadlock or stuck worker thread."
            )
    
    def release(self, net: AntWarPolicyValueNetwork) -> None:
        """归还网络实例到池。
        
        必须在 finally 块中调用，确保即使对战异常也能归还。
        不做参数重置（下次 load_genome_to_network 会覆盖）。
        """
        if net is None:
            raise ValueError("Cannot release None network to pool")
        self._pool.put(net)
        self._in_use -= 1
    
    @property
    def available(self) -> int:
        """当前可用实例数（用于监控）"""
        return self._n_networks - self._in_use


class TaskDispatcher:
    """任务分发器 —— 将评估任务并行分发给 worker 池
    
    职责：
    - 管理线程池生命周期（init → dispatch → shutdown）
    - 持有 NetworkPool，在 shutdown 时确保归还
    - 收集所有 worker 结果，聚合错误信息
    - 支持超时控制，防止个别任务卡死拖垮整体
    """
    
    def __init__(
        self, 
        max_workers: int,
        network_pool: NetworkPool,
        task_timeout: float = 300.0,  # 单任务最大等待时间（秒）
    ):
        self._max_workers = max_workers
        self._network_pool = network_pool
        self._task_timeout = task_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._logger = logging.getLogger(__name__)
    
    def dispatch_population_eval(
        self, 
        tasks: List[EvalTask],
        battle_config: GABattleConfig,
    ) -> List[EvalResult]:
        """并行执行所有评估任务，收集结果。
        
        错误处理策略：
        - 单个任务异常 → 记录到 EvalResult.errors，不中断其他任务
        - 单个任务超时 → 标记为超时错误，cancel 该 future，继续处理其他
        - 整体超时或过多失败 → 抛出 RuntimeError
        
        Returns:
            List[EvalResult]: 成功和失败的结果混合列表。调用方需检查 errors 字段。
        """
        n_tasks = len(tasks)
        self._logger.info(
            f"Dispatching {n_tasks} eval tasks with {self._max_workers} workers"
        )
        
        futures = {}
        for task in tasks:
            future = self._executor.submit(
                _eval_single_task, task, battle_config, self._network_pool
            )
            futures[future] = task
        
        results: List[EvalResult] = []
        errors: List[str] = []
        start_time = time.monotonic()
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result(timeout=self._task_timeout)
                results.append(result)
                if result.errors:
                    self._logger.warning(
                        f"Task (ind={task.individual_id}, ref={task.reference_name}) "
                        f"completed with {len(result.errors)} error(s): {result.errors}"
                    )
            except TimeoutError:
                future.cancel()
                err_msg = (
                    f"Task (ind={task.individual_id}, ref={task.reference_name}) "
                    f"timed out after {self._task_timeout}s"
                )
                self._logger.error(err_msg)
                errors.append(err_msg)
                results.append(EvalResult(
                    individual_id=task.individual_id,
                    reference_name=task.reference_name,
                    outcomes=[0.5, 0.5],  # 超时视为平局
                    errors=[err_msg],
                ))
            except Exception as e:
                err_msg = (
                    f"Task (ind={task.individual_id}, ref={task.reference_name}) "
                    f"failed: {type(e).__name__}: {e}"
                )
                self._logger.error(err_msg, exc_info=True)
                errors.append(err_msg)
                results.append(EvalResult(
                    individual_id=task.individual_id,
                    reference_name=task.reference_name,
                    outcomes=[0.5, 0.5],
                    errors=[err_msg],
                ))
        
        elapsed = time.monotonic() - start_time
        failure_rate = len(errors) / n_tasks if n_tasks > 0 else 0
        
        self._logger.info(
            f"Dispatch completed in {elapsed:.1f}s. "
            f"Success: {n_tasks - len(errors)}/{n_tasks}, "
            f"Failures: {len(errors)} ({failure_rate:.1%})"
        )
        
        if failure_rate > 0.5:
            raise RuntimeError(
                f"Too many task failures ({len(errors)}/{n_tasks} = {failure_rate:.1%}). "
                f"Aborting to avoid corrupt evaluation results."
            )
        
        return results
    
    def shutdown(self):
        """关闭线程池。必须先于 NetworkPool 销毁调用。"""
        self._logger.info("Shutting down TaskDispatcher executor...")
        self._executor.shutdown(wait=True)
        self._logger.info("TaskDispatcher executor shut down.")
```

### 4.2 修改 `BattleSelection.evaluate_population()`

#### 4.2.1 方法签名与调用方改动

`evaluate_population()` 新增 `task_dispatcher` 参数，由 `GeneticEvolutionTrainer` 在初始化时注入：

```python
# battle_selection.py

class BattleSelection:
    def __init__(self, ...):
        ...
        self._task_dispatcher: Optional[TaskDispatcher] = None  # 由 Trainer 注入
    
    def set_task_dispatcher(self, dispatcher: TaskDispatcher) -> None:
        """注入 TaskDispatcher（由 GeneticEvolutionTrainer 调用）"""
        self._task_dispatcher = dispatcher
    
    def evaluate_population(
        self,
        population: Population,
        generation: int,
    ) -> List[Individual]:
        """评估种群中所有个体，返回按 ELO 降序排列的列表。
        
        当 self._task_dispatcher 不为 None 时使用并行路径，
        否则回退到原始串行路径（向后兼容）。
        """
        if self._task_dispatcher is not None:
            return self._evaluate_population_parallel(population, generation)
        else:
            return self._evaluate_population_serial(population, generation)
```

#### 4.2.2 并行评估实现

```python
def _evaluate_population_parallel(
    self,
    population: Population,
    generation: int,
) -> List[Individual]:
    """并行评估种群（方案 B 路径）"""
    
    # 步骤 1：构建参照物列表
    references = self._build_reference_list(population, generation)
    
    # 步骤 2：预创建并序列化所有参照物 agent（一次性）
    ref_agents_bytes = self._preload_reference_agents(references)
    
    # 步骤 3：编译任务列表
    tasks: List[EvalTask] = []
    for ind in population.individuals:
        for ref in references:
            tasks.append(EvalTask(
                individual_id=ind.id,
                genome=ind.genome,
                reference_name=ref.name,
                reference_agent_bytes=ref_agents_bytes[ref.name],
            ))
    
    # 步骤 4：并行执行所有评估任务
    results = self._task_dispatcher.dispatch_population_eval(
        tasks, self._battle_config
    )
    
    # 步骤 5：聚合结果，批量计算 ELO
    self._batch_update_elo(population, references, results)
    
    # 步骤 6：清理参照物缓存（释放 GPU 显存）
    ref_agents_bytes.clear()
    
    # 步骤 7：按 ELO 降序排列返回
    population.individuals.sort(key=lambda ind: ind.elo_rating, reverse=True)
    return population.individuals
```

#### 4.2.3 ELO 批量计算方法 —— 归属 `BattleSelection`

**设计决策**：ELO 批量计算方法放在 `BattleSelection` 类中（而非 `ELORating` 类），因为：

1. `ELORating` 保持其通用性（单次 `update` 语义不变），不做破坏性修改。
2. 批量计算的"固定参照物 ELO + 结果拍平"语义是 **GA 评估场景特有**的，放在调用方更合适。
3. 两种方法可共存：并行路径用 `_batch_update_elo()`，串行路径仍用 `ELORating.update()`。

```python
# battle_selection.py — BattleSelection 新增方法

def _batch_update_elo(
    self,
    population: Population,
    references: List[ReferenceAgent],
    results: List[EvalResult],
) -> None:
    """批量计算 ELO 评分（确定性，与顺序无关）。
    
    算法：
    - 所有参照物的 ELO 固定（在对战开始前拍快照）
    - 每个个体的 ELO 从其初始值出发，依次应用所有对战结果
    - 参照物 ELO 不随对战结果更新（打破路径依赖，保证确定性）
    
    与 ELORating.update() 的关系：
    - ELORating.update() 保持原有接口，用于串行路径
    - 本方法用于并行路径，二者逻辑等价但本方法结果确定
    """
    k_factor = self._elo_config.k_factor
    initial_elo = self._elo_config.initial_elo
    
    # 1. 拍参照物 ELO 快照
    ref_elo_map: Dict[str, float] = {}
    for ref in references:
        ref_elo_map[ref.name] = self.elo.get_rating(ref.name)
    
    # 2. 按个体聚合结果
    ind_results: Dict[str, List[Tuple[str, List[float]]]] = {}
    for result in results:
        if result.individual_id not in ind_results:
            ind_results[result.individual_id] = []
        ind_results[result.individual_id].append(
            (result.reference_name, result.outcomes)
        )
    
    # 3. 逐个个体计算最终 ELO
    for ind in population.individuals:
        ra = initial_elo
        for ref_name, outcomes in ind_results.get(ind.id, []):
            rb = ref_elo_map.get(ref_name, initial_elo)
            for outcome in outcomes:
                ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                ra += k_factor * (outcome - ea)
        ind.elo_rating = ra
    
    # 4. 同步更新 ELORating 实例（使后续串行路径可用）
    for ind in population.individuals:
        self.elo.set_rating(ind.id, ind.elo_rating)
```

> **注意**：`ELORating` 需新增 `get_rating(name)` 和 `set_rating(name, rating)` 两个只读/只写方法（当前仅有 `update` 方法），改动极小（~5 行）。`get_rating` 返回当前评分，`set_rating` 直接覆盖。二者均不涉及复杂的 ELO 计算逻辑。

#### 4.2.4 串行回退路径（保留原实现）

```python
def _evaluate_population_serial(
    self,
    population: Population,
    generation: int,
) -> List[Individual]:
    """串行评估种群（原始路径，当 parallel.enabled=false 时使用）"""
    # 保持现有实现不变（battle_selection.py:94-184）
    ...
```

### 4.3.0 修改 `BattleSimulator`：新增公开单局对战接口

**问题**：当前 `BattleSimulator` 将两个不相关的职责耦合在一起：

```
BattleSimulator
├── 职责 1：执行单局对战  (_execute_battle)       ← 核心业务逻辑（私有函数）
└── 职责 2：并行调度多局对战 (run_battles → ProcessPool)  ← 执行策略（公开方法）
```

在方案 B 中，worker 线程内只需串行执行 2 局对战，不需要 ProcessPool。但没有公开接口提供"在当前线程中执行单局"的能力——唯一公开方法 `run_battles()` 强制走 ProcessPool。直接调用私有函数 `_execute_battle` 是 workaround，破坏了模块封装。

**方案**：将单局对战提升为公开方法，从根源解耦两个职责：

```python
# battle_simulator.py — 改造后

class BattleSimulator:
    """对局模拟器"""
    
    def execute_single_battle(
        self,
        agent1,
        agent2,
        first_player: int = 0,
        max_steps: int = 200,
    ) -> dict:
        """在当前进程/线程中执行单局对战。
        
        不创建子进程，不嵌套 ProcessPool。
        调用方自行负责并行调度（ThreadPool / 手动多线程等）。
        
        Returns:
            {"result": "agent1_win"|"agent2_win"|"draw", ...} 
            或 {"error": "..."}
        """
        return _execute_battle(
            agent1=agent1,
            agent2=agent2,
            first_player=first_player,
            max_steps=max_steps or self.max_steps,
        )
    
    # run_battles() 保持不变（串行回退路径使用）
    def run_battles(self, ...):
        ...
```

**改动量**：`BattleSimulator` 新增 ~15 行公开方法。`run_battles()` 及所有其他现有代码不受影响。

**为什么优于绕过 API**：
- 封装性：TaskDispatcher 只依赖公开接口，不关心 `_execute_battle` 的实现细节
- 可测试性：`execute_single_battle()` 可独立单测
- 符合原则：从根源解决职责耦合问题，而非 workaround

---

### 4.3 修改 `BaselineEvaluator.evaluate_seeds()`

与种群评估相同的并行化策略，但基线验证有两点关键差异：

1. **`n_battles=12`**（而非 2 局）：每个 (种子, baseline) 组合需执行 12 局，以获取统计显著的结果。
2. **无 ELO 计算**：基线验证直接统计胜率，不涉及 ELO 系统。

#### 4.3.1 入口方法

```python
# baseline_evaluator.py

class BaselineEvaluator:
    def __init__(self, ...):
        ...
        self._task_dispatcher: Optional[TaskDispatcher] = None
    
    def set_task_dispatcher(self, dispatcher: TaskDispatcher) -> None:
        self._task_dispatcher = dispatcher
    
    def evaluate_seeds(
        self,
        seeds: List[Individual],
        generation: int,
    ) -> Dict[str, Dict[str, float]]:
        """评估种子个体 vs 所有 baseline agent。
        
        Returns:
            {seed_id: {baseline_name: win_rate, ...}, ...}
        """
        if self._task_dispatcher is not None:
            return self._evaluate_seeds_parallel(seeds, generation)
        else:
            return self._evaluate_seeds_serial(seeds, generation)
```

#### 4.3.2 并行评估 + `n_battles=12` 的编排策略

**关键问题**：12 局如何划分？有三种可选策略：

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A: 一个 Task = 12 局** | 每个 EvalTask 内部串行执行 12 局 | 简单，对战胜率统计完整 | 12 局串行耗时 ~3 分钟/task，worker 占用长 |
| **B: 一个 Task = 1 局 × 12 个 task** | 将 12 局拆成 12 个独立 task | 并行度最高 | 结果需额外聚合；增加了 task 数量 |
| **C: 一个 Task = 4 局 × 3 个 task** | 折中，3 个 task 各执行 4 局（先手/后手各 2 局） | 平衡并行度与聚合复杂度 | 中等复杂度 |

**推荐策略 A**，理由：基线验证任务总数少（8 种子 × 3 baseline = 24 个组合），即使一个 task 耗时 3 分钟，4 worker 并行也只需 6 波（~18 分钟），远快于当前的串行 ~10 分钟。策略 B/C 带来的额外并行度收益有限，但增加了结果聚合和 task 调度开销。

#### 4.3.3 基线验证 Worker 函数

```python
# task_dispatcher.py — 基线验证专用 worker 函数

def _eval_baseline_task(
    task: EvalTask,
    battle_config: GABattleConfig,
    network_pool: NetworkPool,
) -> EvalResult:
    """在 worker 线程中执行单个基线验证任务。
    
    与种群评估 worker (_eval_single_task) 的关键差异：
    - 执行 n_battles=12 局（而非 2 局），先手/后手各 6 局
    - 通过 BattleSimulator.execute_single_battle() 公开接口执行单局
    - 每局独立创建 BattleSimulator 实例（轻量），避免状态污染
    """
    net = network_pool.acquire()
    try:
        load_genome_to_network(net, task.genome)
        ga_agent = GAWarAgent(player_id=0, network=net, device=net.device)
        ref_agent = AgentLoader.deserialize(task.reference_agent_bytes)
        
        simulator = BattleSimulator(config=battle_config)
        
        outcomes: List[float] = []
        errors: List[str] = []
        
        # 12 局：先手 6 局 + 后手 6 局
        n_battles = battle_config.n_battles  # = 12
        
        for i in range(n_battles):
            first_player = 0 if i < n_battles // 2 else 1
            try:
                result = simulator.execute_single_battle(
                    agent1=ga_agent,
                    agent2=ref_agent,
                    first_player=first_player,
                    max_steps=battle_config.max_steps,
                )
                if "error" in result:
                    outcomes.append(0.5)
                    errors.append(f"battle_{i}: {result['error']}")
                else:
                    game_result = result.get("result")
                    if game_result == "agent1_win":
                        outcomes.append(1.0)
                    elif game_result == "agent2_win":
                        outcomes.append(0.0)
                    else:
                        outcomes.append(0.5)
            except Exception as e:
                outcomes.append(0.5)
                errors.append(f"battle_{i}: {type(e).__name__}: {e}")
        
        return EvalResult(
            individual_id=task.individual_id,
            reference_name=task.reference_name,
            outcomes=outcomes,
            errors=errors,
        )
    finally:
        network_pool.release(net)
```

#### 4.3.4 并行入口实现

```python
def _evaluate_seeds_parallel(
    self,
    seeds: List[Individual],
    generation: int,
) -> Dict[str, Dict[str, float]]:
    """并行评估种子 vs baseline agents"""
    
    # 1. 预加载 baseline agent
    baseline_names = list(self._baseline_agents.keys())
    baseline_bytes = {}
    for name, agent in self._baseline_agents.items():
        baseline_bytes[name] = AgentLoader.serialize(agent)
    
    # 2. 构建任务列表（24 个）
    tasks: List[EvalTask] = []
    for seed in seeds:
        for baseline_name in baseline_names:
            tasks.append(EvalTask(
                individual_id=seed.id,
                genome=seed.genome,
                reference_name=baseline_name,
                reference_agent_bytes=baseline_bytes[baseline_name],
            ))
    
    # 3. 并行执行
    results = self._task_dispatcher.dispatch_population_eval(
        tasks, self._baseline_battle_config
    )
    
    # 4. 聚合结果：按 (seed_id, baseline_name) 计算胜率
    seed_results: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        if r.individual_id not in seed_results:
            seed_results[r.individual_id] = {}
        if r.reference_name not in seed_results[r.individual_id]:
            seed_results[r.individual_id][r.reference_name] = []
        seed_results[r.individual_id][r.reference_name].extend(r.outcomes)
    
    # 5. 计算胜率
    output: Dict[str, Dict[str, float]] = {}
    for seed_id, baseline_outcomes in seed_results.items():
        output[seed_id] = {}
        for baseline_name, outcomes in baseline_outcomes.items():
            wins = sum(1 for o in outcomes if o == 1.0)
            output[seed_id][baseline_name] = wins / len(outcomes)
    
    return output
```

### 4.4 修改 `GeneticEvolutionTrainer._build_dynamic_references()`

当前问题：返回的 `ReferenceAgent.agent_factory` 是闭包，每次调用都重新加载 checkpoint。

改造方案：

1. **保留 `ReferenceAgent` 定义不变**（向后兼容）
2. 在 `evaluate_population()` 入口处预创建所有参照物 Agent：

```python
# battle_selection.py — BattleSelection 新增方法

def _preload_reference_agents(
    self, references: List[ReferenceAgent]
) -> Dict[str, bytes]:
    """预加载并序列化所有参照物 agent。
    
    在评估开始前调用一次，后续所有 EvalTask 复用序列化结果。
    
    异常处理策略：
    - 静态参照物（规则 AI）：创建成本为零，异常概率极低，异常时直接抛出
    - 动态参照物（checkpoint）：可能因 checkpoint 损坏/缺失而失败，此时
      记录 warning 并从参照物列表中移除该动态参照物（降级运行）
    
    Returns:
        Dict[str, bytes]: ref_name → 序列化的 agent bytes
    """
    cache: Dict[str, bytes] = {}
    failed_refs: List[str] = []
    
    for ref in references:
        if ref.name in cache:
            continue  # 去重
        
        try:
            agent = ref.agent_factory(1)
            cache[ref.name] = AgentLoader.serialize(agent)
        except FileNotFoundError as e:
            # checkpoint 文件缺失 → 降级：移除此动态参照物
            logger.warning(
                f"Reference agent '{ref.name}' checkpoint not found: {e}. "
                f"Skipping this reference for current generation."
            )
            failed_refs.append(ref.name)
        except Exception as e:
            # 其他异常（如 torch.load 失败、网络结构不匹配）
            logger.error(
                f"Failed to load reference agent '{ref.name}': {type(e).__name__}: {e}. "
                f"Skipping this reference.",
                exc_info=True,
            )
            failed_refs.append(ref.name)
    
    if failed_refs:
        logger.warning(
            f"{len(failed_refs)} reference(s) skipped: {failed_refs}. "
            f"Evaluation will proceed with {len(cache)} reference(s)."
        )
    
    if len(cache) == 0:
        raise RuntimeError(
            "No reference agents could be loaded. Cannot evaluate population."
        )
    
    return cache
```

> **设计要点**：
> - 去重逻辑：多个 `ReferenceAgent` 可能指向同一名称（如多个锚定参照物共享同一个 `BasicRandomAI`），序列化一次即可。
> - 降级而非崩溃：单个动态参照物加载失败不应阻塞整代训练。参照物减少会导致评估精度下降，但比训练中断好。
> - 调用方 `_evaluate_population_parallel()` 需感知：如果某个 `ref.name` 不在 `ref_agents_bytes` 中，则跳过该 (个体, 参照物) 组合的任务生成。这在上层 `_evaluate_population_parallel()` 的任务编译循环中自然处理（只遍历实际存在于 cache 中的 ref）。

### 4.5 Worker 函数设计

```python
def _eval_single_task(
    task: EvalTask,
    battle_config: GABattleConfig,
    network_pool: NetworkPool,
) -> EvalResult:
    """在 worker 线程中执行单个评估任务
    
    流程：
    1. 从池获取网络 → 加载基因组 → 创建 GAWarAgent
    2. 反序列化参照物 agent
    3. 通过 BattleSimulator.execute_single_battle() 公开接口执行 2 局对战（先手+后手）
       （不嵌套 ProcessPool，见 4.6 节并发控制说明）
    4. 归还网络到池
    5. 返回结果
    
    注意：不对 BattleSimulator 做实例复用。BattleSimulator.__init__ 极其轻量
    （仅存储 config 引用），真正的开销在 execute_single_battle() 内部的 SDK backend
    初始化（load_backend + initial_state）。该开销在每次 execute_single_battle() 调用中产生，
    与 BattleSimulator 实例是否复用无关。
    """
    net = network_pool.acquire()
    try:
        load_genome_to_network(net, task.genome)
        ga_agent = GAWarAgent(player_id=0, network=net, device=net.device)
        
        ref_agent = AgentLoader.deserialize(task.reference_agent_bytes)
        
        # 通过 BattleSimulator 公开接口执行单局，不嵌套 ProcessPool
        # 每个 worker 线程串行执行 2 局（先手+后手），外层 ThreadPool 负责并行
        simulator = BattleSimulator(config=battle_config)
        outcomes = []
        errors = []
        for first_player in [0, 1]:
            result = simulator.execute_single_battle(
                agent1=ga_agent,
                agent2=ref_agent,
                first_player=first_player,
                max_steps=battle_config.max_steps,
            )
            if "error" in result:
                outcomes.append(0.5)
                errors.append(result["error"])
            else:
                game_result = result.get("result")
                if game_result == "agent1_win":
                    outcomes.append(1.0)
                elif game_result == "agent2_win":
                    outcomes.append(0.0)
                else:
                    outcomes.append(0.5)
        
        return EvalResult(
            individual_id=task.individual_id,
            reference_name=task.reference_name,
            outcomes=outcomes,
            errors=errors,
        )
    finally:
        network_pool.release(net)
```

> **关于 `execute_single_battle` 的 SDK 初始化开销**：每次 `execute_single_battle` 调用内部会执行 `load_backend()` + `initial_state()`（C++ SDK 的游戏后端初始化）。在同一个 worker 线程中，先后调用两次（先手 + 后手）意味着 SDK backend 被创建和销毁了两次。考虑到单次 SDK 初始化开销（约数百毫秒级别）远小于单局对战耗时（15 秒级别），暂不优化此部分，后续可作为微调项处理。

### 4.6 整体架构对比

**改造前**：
```
train()
  └─ evaluate_population()  [串行 400 次 run_battles]
       └─ for ind (80)        [串行]
            └─ for ref (5)     [串行]
                 └─ run_battles(n=1)  [内部 12 worker，仅跑 2 局]
```

**改造后**：
```
train()
  └─ evaluate_population()
       ├─ 预加载参照物 agent（一次，缓存）
       ├─ 构建 400 个 EvalTask
       └─ TaskDispatcher.dispatch()
            └─ ThreadPoolExecutor(max_workers=4)
                 ├─ Worker 1: _eval_single_task() ─┐
                 ├─ Worker 2: _eval_single_task() ─┤ 并行 4 个任务
                 ├─ Worker 3: _eval_single_task() ─┤ 每个内串行执行 2 局
                 └─ Worker 4: _eval_single_task() ─┘
```

**注意并发控制**：如果 ThreadPool 4 个 worker 每个都调用 `BattleSimulator.run_battles()`（内部又创建 ProcessPoolExecutor 12 workers），会导致 4 × 12 = 48 个进程同时运行，可能压垮 CPU。

**解决方案**：通过 `BattleSimulator` 新增的 `execute_single_battle()` 公开接口（见 4.3.0 节），worker 函数在线程内串行执行单局对战，不嵌套 ProcessPool。具体实现在 4.5 节 worker 函数中。这样外层 ThreadPool 负责 (个体, 参照物) 级别的并行，每局对战在线程内串行执行，消除了嵌套进程池问题。

> **关于 `BattleSimulator` 实例**：`BattleSimulator.__init__` 极其轻量（仅存储 config 引用），因此不需要对其实例做池化复用。真正的开销在 `execute_single_battle()` 内部的 SDK backend 初始化（`load_backend` + `initial_state`），详见 4.5 节末尾的分析。

### 4.7 `GeneticEvolutionTrainer` 集成 TaskDispatcher

#### 4.7.1 初始化流程

```python
# genetic_evolution_trainer.py

class GeneticEvolutionTrainer:
    def __init__(
        self,
        config: dict,
        # ... 其他参数
    ):
        # ... 现有初始化代码 ...
        
        # --- 并行化组件初始化 ---
        parallel_config = config.get("parallel", {})
        self._parallel_enabled = parallel_config.get("enabled", False)
        
        if self._parallel_enabled:
            self._init_parallel_components(parallel_config)
        else:
            self._task_dispatcher = None
    
    def _init_parallel_components(self, parallel_config: dict):
        """初始化并行化组件：NetworkPool + TaskDispatcher。
        
        初始化顺序很重要：
        1. 先创建 NetworkPool（依赖 device、hidden_dim 等参数）
        2. 再创建 TaskDispatcher（依赖 NetworkPool）
        3. 将 TaskDispatcher 注入 BattleSelection 和 BaselineEvaluator
        """
        max_workers = parallel_config.get("max_workers", 4)
        n_network_instances = parallel_config.get("n_network_instances", max_workers)
        mode = parallel_config.get("mode", "thread")
        acquire_timeout = parallel_config.get("acquire_timeout", 120.0)
        task_timeout = parallel_config.get("task_timeout", 300.0)
        
        device = torch.device(self._config.get("device", "cuda"))
        
        # 步骤 1：创建 NetworkPool
        self._network_pool = NetworkPool(
            n_networks=n_network_instances,
            hidden_dim=self._config["network"]["hidden_dim"],
            enable_auxiliary=self._config["network"].get("enable_auxiliary", False),
            device=device,
            acquire_timeout=acquire_timeout,
        )
        
        logger.info(
            f"Initialized NetworkPool with {n_network_instances} instances "
            f"on {device} (acquire_timeout={acquire_timeout}s)"
        )
        
        # 步骤 2：创建 TaskDispatcher
        if mode == "thread":
            self._task_dispatcher = TaskDispatcher(
                max_workers=max_workers,
                network_pool=self._network_pool,
                task_timeout=task_timeout,
            )
        elif mode == "process":
            raise NotImplementedError(
                "ProcessPool mode not yet implemented. Use 'thread' or 'serial'."
            )
        else:
            raise ValueError(f"Unknown parallel mode: {mode}")
        
        logger.info(
            f"Initialized TaskDispatcher: mode={mode}, "
            f"max_workers={max_workers}, task_timeout={task_timeout}s"
        )
        
        # 步骤 3：注入到子组件
        self._battle_selection.set_task_dispatcher(self._task_dispatcher)
        self._baseline_evaluator.set_task_dispatcher(self._task_dispatcher)
        
        logger.info("TaskDispatcher injected into BattleSelection and BaselineEvaluator")
```

#### 4.7.2 训练循环适配

`train()` 方法中无需修改评估调用，只需在训练结束后清理：

```python
def train(self, n_generations: int):
    """主训练循环（改动极小）"""
    try:
        for gen in range(n_generations):
            # ... 种群生成、评估、选择 ...（现有代码不变）
            # evaluate_population() 和 evaluate_seeds() 内部
            # 自动根据 self._task_dispatcher 选择并行/串行路径
            pass
    finally:
        # 确保资源释放
        if self._task_dispatcher is not None:
            self._task_dispatcher.shutdown()
            logger.info("TaskDispatcher resources released.")
```

#### 4.7.3 资源生命周期

```
GeneticEvolutionTrainer.__init__()
  ├─ NetworkPool.__init__()        # 创建 N 个网络实例 → 占用 GPU 显存
  ├─ TaskDispatcher.__init__()     # 创建线程池（N 线程）
  ├─ battle_selection.set_task_dispatcher()
  └─ baseline_evaluator.set_task_dispatcher()

GeneticEvolutionTrainer.train()    # 每代复用同一 TaskDispatcher

GeneticEvolutionTrainer.train() finally:
  └─ TaskDispatcher.shutdown()     # 关闭线程池，NetworkPool 随 Trainer 销毁释放
```

> **关键约束**：`TaskDispatcher.shutdown()` 必须先于 `NetworkPool` 销毁调用，因为 worker 线程持有对 `NetworkPool` 的引用。`finally` 块确保了即使训练异常退出也能正确释放。

---

## 5. 集成点

### 5.1 需修改的文件

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `trainer/task_dispatcher.py` | **新增** | NetworkPool、TaskDispatcher、EvalTask、EvalResult、Worker 函数 |
| `battle/battle_simulator.py` | **修改** | 新增 `execute_single_battle()` 公开单局对战接口（解耦单局执行与并行调度） |
| `selection/battle_selection.py` | **修改** | `evaluate_population()` 改为使用 TaskDispatcher；新增 ELO 批量计算方法 |
| `evaluation/baseline_evaluator.py` | **修改** | `evaluate_seeds()` 改为使用 TaskDispatcher |
| `trainer/genetic_evolution_trainer.py` | **修改** | 注入 TaskDispatcher 实例；`_build_dynamic_references()` 配合缓存 |
| `config/ga_evo.yaml` | **新增字段** | 并行化配置节：`parallel.enabled`、`parallel.max_workers` 等 |

### 5.2 配置新增字段

```yaml
# ga_evo.yaml 新增
parallel:
  enabled: true                # 是否启用并行评估
  max_workers: 4               # 线程池大小（建议 = GPU 数量 / 网络实例数）
  n_network_instances: 4       # 网络实例池大小（建议 = max_workers）
  mode: "thread"               # "thread" | "process" | "serial"
```

---

## 6. 实施计划

### 6.1 分步实施

| 步骤 | 内容 | 预估工时 | 验证方式 |
|------|------|---------|---------|
| 1 | 新增 `trainer/task_dispatcher.py`：NetworkPool + EvalTask/EvalResult + Worker 函数 | 3h | 单元测试 |
| 2 | 修改 `BattleSelection`：重构 `evaluate_population()` 使用 TaskDispatcher | 2h | 集成测试（单代） |
| 3 | 修改 `BaselineEvaluator`：重构 `evaluate_seeds()` 使用 TaskDispatcher | 1h | 集成测试 |
| 4 | 修改 `GeneticEvolutionTrainer`：注入 TaskDispatcher，调整 `_build_dynamic_references` | 1h | 集成测试 |
| 5 | 添加配置开关 + 串行回退路径 | 0.5h | 配置切换测试 |
| 6 | ELO 批量更新逻辑 | 1h | 对比串行结果一致性 |
| 7 | 端到端性能测试（小规模：3 代 × 20 个体） | 2h | 计时对比 |
| 8 | 完整规模性能测试（10 代 × 80 个体） | 1h | 加速比验证 |
| **总计** | | **~11.5h** | |

### 6.2 测试策略

1. **正确性测试**：
   - 同一代种群，分别在并行模式和串行模式下评估，对比 ELO 排名。
   - 由于 ELO 更新改为批量方式，排名应基本一致（允许微小浮动，因参照物 ELO 固定）。

2. **性能测试**：
   - 计时对比：`evaluate_population()` 和 `evaluate_seeds()` 的端到端耗时。
   - 记录 GPU 利用率和显存峰值。

3. **边界测试**：
   - 网络池耗尽时的行为（阻塞等待 vs 抛异常）。
   - 对战异常（error 结果）在并行模式下的处理。

### 6.3 回滚计划

通过配置开关控制：

```yaml
parallel:
  enabled: false  # 改为 false 即回退到原始串行模式
```

当 `enabled=false` 时，代码路径完全等同当前实现（保留原始 `evaluate_population` 的串行逻辑）。

---

## 7. 风险评估

### 7.1 GPU 显存争用

**风险**：N 个网络实例同时在 GPU 上执行推理，显存占用 = N × 网络参数量 × 4 bytes（float32）。

- 单个 `AntWarPolicyValueNetwork` 参数量估算：hidden_dim=256 时约 **15-20M 参数**，即 60-80 MB。
- 4 个实例：~320 MB（可接受，GPU 通常有 8-24 GB）。
- 加上 PyTorch CUDA context 等开销，4 个实例约 500-800 MB。

**缓解**：限制 `n_network_instances` ≤ 4，通过配置可调。

### 7.2 线程池与 CUDA stream 争用

**风险**：多线程在默认 CUDA stream 上排队，GPU 利用率不升反降。

**缓解**：
1. 先用简单 ThreadPool 实测，监控 GPU 利用率（`nvidia-smi`）。
2. 若利用率低，引入独立 CUDA stream（见 3.2.5）。
3. 若仍有问题，切换为 `ProcessPoolExecutor` + CPU 推理（方案 A 路线）。

### 7.3 Pickle 序列化开销

**风险**：每个 EvalTask 需将 Genome（numpy array，约 15-20M 个 float32 = 60-80 MB）序列化传递给 worker。

- 在 ThreadPool 模式下，可通过共享引用避免（所有对象在同一进程地址空间）。
- 在 ProcessPool 模式下，需要 pickle.dumps + pickle.loads，每个 genome 约 80 MB。

**缓解**：优先使用 ThreadPool。若必须用 ProcessPool，可考虑：
- 使用 `multiprocessing.shared_memory`（Python 3.8+）共享 genome 数据。
- 或将 Genome 的 numpy array 转为 `torch.Tensor` 存入共享内存。

### 7.4 调试复杂度

**风险**：并行模式下错误堆栈复杂、日志交错。

**缓解**：
1. 所有对战日志（logger.debug）保留在 worker 线程内，包含 individual_id 前缀。
2. 异常在 worker 中捕获并返回 `EvalResult.errors`，不中断整体流程。
3. 保留串行模式作为调试 fallback。

### 7.5 BattleSimulator 嵌套进程池

**风险**：ThreadPool worker 调用 `BattleSimulator.run_battles()` 会嵌套创建子 ProcessPool，导致进程爆炸（4 threads × 12 processes = 48 进程）。

**缓解**：通过 `BattleSimulator.execute_single_battle()` 公开接口（见 4.3.0 节），worker 函数在线程内串行执行单局，不嵌套 ProcessPool（见 4.6 节）。

---

## 8. 附录

### 8.1 关键代码路径索引

| 文件 | 关键行 | 说明 |
|------|--------|------|
| `trainer/genetic_evolution_trainer.py` | 97-159 | 主训练循环 |
| `trainer/genetic_evolution_trainer.py` | 114-115 | 调用 `evaluate_population` |
| `trainer/genetic_evolution_trainer.py` | 200-236 | `_create_reference_from_checkpoint` + 闭包工厂 |
| `selection/battle_selection.py` | 94-184 | `evaluate_population` 串行双重循环 |
| `selection/battle_selection.py` | 200-221 | `_create_ga_agent` 网络创建 |
| `selection/elo_rating.py` | 21-41 | `ELORating.update` 顺序更新 |
| `evaluation/baseline_evaluator.py` | 48-126 | `evaluate_seeds` + `_evaluate_single_seed` |
| `battle/battle_simulator.py` | 24-91 | `run_battles` + `_run_battles_parallel` |
| `battle/battle_simulator.py` | 192-367 | `_execute_battle` 单局对战逻辑 |
| `battle/agent_loader.py` | 150-160 | `serialize`/`deserialize` (cloudpickle) |
| `genome/genome.py` | 100-109 | `load_genome_to_network` in-place 加载 |
| `network/ant_war_policy_value_network.py` | 18-80 | 网络结构（双编码器架构） |
| `battle/opponent_agent.py` | 16-33 | `from_checkpoint` (torch.load + to(device)) |

### 8.2 术语表

| 术语 | 含义 |
|------|------|
| 个体 (Individual) | 种群中的一个成员，持有一个基因组（网络权重） |
| 基因组 (Genome) | 扁平化的网络权重 numpy 数组 |
| 参照物 (Reference) | 用于评估个体实力的固定对手（锚定 + 动态） |
| 锚定参照物 | 永不更换的规则 AI baseline（BasicRandomAI 等） |
| 动态参照物 | 从对手池选取的、随训练更新的参照物 |
| 种子 (Seed) | 每代评分最高的 N 个个体，用于生成下一代 |
| ELO 评分 | 基于对战结果的相对实力评分系统 |
| NetworkPool | 预创建的网络实例池，消除重复 `to(device)` 开销 |
| EvalTask | 一个 (个体, 参照物) 对战任务的最小单元 |
