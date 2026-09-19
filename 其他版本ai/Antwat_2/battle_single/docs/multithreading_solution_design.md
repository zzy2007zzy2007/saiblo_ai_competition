# Battle Single 多线程性能问题解决方案设计

## 一、方案概述

针对 `battle_single` 多线程环境下 agent 决策时间显著增加的问题，本方案采用 **多进程并行** 架构替代原有的多线程方案，真正实现高效的并行对战执行。

## 二、方案选择

### 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **当前方案：顺序执行** | 简单、无线程安全问题 | 无法并行，效率低 | 临时修复 |
| **方案一：多进程并行** | 真正并行、无 GIL 限制、进程隔离 | 进程间通信开销、内存占用增加 | **推荐方案** |
| **方案二：线程隔离 Agent** | 线程级并行、启动快 | GIL 限制、内存占用增加 | 轻量级场景 |

### 选择理由

**选择多进程并行作为最终解决方案**，理由如下：

1. **解决 GIL 限制**：每个进程拥有独立的 Python 解释器和 GIL，真正实现 CPU 密集型任务的并行执行
2. **避免线程安全问题**：进程间完全隔离，无需担心共享状态冲突
3. **扩展性好**：可以充分利用多核 CPU 资源
4. **稳定性高**：单个进程崩溃不影响其他进程

## 三、详细设计

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Process (主进程)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│  │  Logger     │  │  Config     │  │  Result Aggregator  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬────────┘     │
└─────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ProcessPoolExecutor                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │            │
│  │ (独立进程)   │  │ (独立进程)   │  │ (独立进程)   │            │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤            │
│  │ Agent1      │  │ Agent1      │  │ Agent1      │            │
│  │ Agent2      │  │ Agent2      │  │ Agent2      │            │
│  │ Simulator   │  │ Simulator   │  │ Simulator   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心实现

#### 3.2.1 进程池初始化

```python
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

def run_battles_parallel(self, agent1_name: str, agent2_name: str,
                        num_episodes: int, parallel_workers: int = 4) -> List[Dict]:
    """
    使用多进程并行执行对战
    
    Args:
        agent1_name: 第一个 agent 名称
        agent2_name: 第二个 agent 名称
        num_episodes: 对战局数
        parallel_workers: 并行进程数
    
    Returns:
        对战结果列表
    """
    self._load_sdk()
    
    # 创建任务列表（每局对战交换先后手）
    tasks = []
    for episode in range(num_episodes):
        seed = episode
        tasks.append((episode, seed, 0, agent1_name, agent2_name))
        tasks.append((episode + 1000, seed + 1000, 1, agent1_name, agent2_name))
    
    # 使用 Manager 共享结果队列（可选）
    with Manager() as manager:
        result_queue = manager.Queue()
        
        # 创建进程池
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            # 提交所有任务
            futures = [
                executor.submit(run_single_battle_process, task, result_queue)
                for task in tasks
            ]
            
            # 等待所有任务完成
            for future in futures:
                future.result()
            
            # 收集结果
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())
    
    results.sort(key=lambda x: x['episode'])
    return results
```

#### 3.2.2 单进程对战函数

```python
def run_single_battle_process(task, result_queue=None):
    """
    在独立进程中执行单局对战
    
    Args:
        task: 任务参数 (episode, seed, first_player, agent1_name, agent2_name)
        result_queue: 结果队列（用于多进程共享）
    """
    episode, seed, first_player, agent1_name, agent2_name = task
    
    try:
        # 在子进程中重新加载 SDK（每个进程独立）
        import sys
        import os
        
        # 设置 SDK 路径
        sdk_paths = [
            '/root/autodl-tmp/AntWar/Ant-Game/SDK',
            os.path.join(os.path.dirname(__file__), '../../Ant-Game/SDK'),
        ]
        
        for path in sdk_paths:
            if os.path.exists(path):
                sdk_parent = os.path.dirname(path)
                if sdk_parent not in sys.path:
                    sys.path.insert(0, sdk_parent)
                if path not in sys.path:
                    sys.path.insert(0, path)
                break
        
        # 加载 agent
        from .agent_loader import AgentLoader
        
        loader = AgentLoader()
        agent1 = loader.load_agent(agent1_name)
        agent2 = loader.load_agent(agent2_name)
        
        if not agent1 or not agent2:
            result = {
                'episode': episode,
                'agent1_name': agent1_name,
                'agent2_name': agent2_name,
                'first_player': agent1_name if first_player == 0 else agent2_name,
                'result': 'error',
                'error': 'Failed to load agents'
            }
            if result_queue:
                result_queue.put(result)
            return result
        
        # 创建独立的模拟器实例
        from .battle_single_simulator import BattleSingleSimulator
        
        simulator = BattleSingleSimulator()
        
        # 执行对战（不记录日志，由主进程统一记录）
        result = simulator.run_battle(
            agent1, agent2,
            agent1_name, agent2_name,
            episode, seed=seed, first_player=first_player
        )
        
        if result_queue:
            result_queue.put(result)
        
        return result
    
    except Exception as e:
        import traceback
        error_msg = str(e) + "\n" + traceback.format_exc()
        result = {
            'episode': episode,
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'first_player': agent1_name if first_player == 0 else agent2_name,
            'result': 'error',
            'error': error_msg
        }
        if result_queue:
            result_queue.put(result)
        return result
```

### 3.3 关键设计要点

#### 3.3.1 进程隔离

| 资源 | 处理方式 | 说明 |
|------|----------|------|
| SDK 模块 | 每个进程独立加载 | 避免模块状态共享 |
| Agent 实例 | 每个进程独立创建 | 避免模型状态共享 |
| 模拟器 | 每个进程独立创建 | 避免状态污染 |
| 日志 | 主进程统一记录 | 避免日志混乱 |

#### 3.3.2 结果收集策略

```python
# 方案A：直接返回（简单，推荐）
results = [future.result() for future in futures]

# 方案B：共享队列（实时进度）
with Manager() as manager:
    result_queue = manager.Queue()
    # 任务完成时将结果放入队列
    # 主进程可以实时读取队列显示进度
```

#### 3.3.3 异常处理

```python
def handle_process_exception(exception, task):
    """
    处理进程执行异常
    
    Args:
        exception: 异常对象
        task: 任务参数
    
    Returns:
        错误结果字典
    """
    episode, seed, first_player, agent1_name, agent2_name = task
    
    return {
        'episode': episode,
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'first_player': agent1_name if first_player == 0 else agent2_name,
        'result': 'error',
        'error': str(exception),
        'seed': seed,
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration': 0.0
    }
```

## 四、实现步骤

### 4.1 步骤分解

| 步骤 | 内容 | 负责人 | 预估时间 |
|------|------|--------|----------|
| 1 | 修改 `run_battles_parallel` 方法，实现多进程并行 | 开发 | 2小时 |
| 2 | 实现 `run_single_battle_process` 独立进程函数 | 开发 | 2小时 |
| 3 | 更新 agent_loader 支持进程独立加载 | 开发 | 1小时 |
| 4 | 测试基本功能（单进程、多进程对比） | 测试 | 2小时 |
| 5 | 性能测试（workers=2,4,8对比） | 测试 | 4小时 |
| 6 | 稳定性测试（长时间运行） | 测试 | 8小时 |

### 4.2 代码修改清单

| 文件 | 修改内容 |
|------|----------|
| `battle_single/src/battle_single_simulator.py` | 修改 `run_battles_parallel` 方法 |
| `battle_single/src/battle_single_simulator.py` | 添加 `run_single_battle_process` 函数 |
| `battle_single/src/agent_loader.py` | 确保 agent 可被多次加载 |
| `battle_single/src/battle_single_cli.py` | 保持接口不变 |

## 五、测试计划

### 5.1 功能测试

| 测试用例 | 参数 | 预期结果 |
|----------|------|----------|
| 单进程对战 | `--episodes 1 --workers 1` | 正常执行 |
| 双进程对战 | `--episodes 1 --workers 2` | 正常执行，无性能下降 |
| 多进程对战 | `--episodes 1 --workers 4` | 正常执行，性能提升 |
| 异常处理 | 不存在的 agent | 返回错误结果，不崩溃 |

### 5.2 性能测试

| 测试指标 | 测试方法 | 预期目标 |
|----------|----------|----------|
| 并行效率 | 对比 workers=1 vs workers=4 | 4进程耗时 ≈ 1进程耗时 |
| agent 决策时间 | 监控 `agent2_choose_operations` | 保持在 1 秒左右 |
| 资源占用 | 监控 CPU 和内存 | 内存增加 ≈ workers 倍 |

### 5.3 测试脚本

```bash
# 基准测试（单进程）
python -m battle_single.src.battle_single_cli --episodes 10 --workers 1 --agents sample nn_gen_32

# 并行测试（多进程）
python -m battle_single.src.battle_single_cli --episodes 10 --workers 2 --agents sample nn_gen_32
python -m battle_single.src.battle_single_cli --episodes 10 --workers 4 --agents sample nn_gen_32
python -m battle_single.src.battle_single_cli --episodes 10 --workers 8 --agents sample nn_gen_32
```

## 六、风险评估

### 6.1 潜在风险

| 风险 | 描述 | 严重程度 | 缓解措施 |
|------|------|----------|----------|
| **Pickle 序列化问题** | agent 对象可能无法序列化 | 高 | 确保 agent 类可序列化，或在进程内创建 |
| **内存占用增加** | 每个进程加载一份模型 | 中 | 限制最大 workers 数量 |
| **启动开销** | 进程启动和 agent 加载耗时 | 中 | 预加载 agent 池 |
| **日志混乱** | 多进程同时写日志 | 低 | 主进程统一记录日志 |
| **资源竞争** | 多个进程竞争系统资源 | 低 | 限制并行度 |

### 6.2 降级策略

```python
def run_battles_parallel(self, agent1_name: str, agent2_name: str,
                        num_episodes: int, parallel_workers: int = 4) -> List[Dict]:
    """
    带有降级策略的并行对战执行
    
    当多进程失败时自动降级为顺序执行
    """
    try:
        # 尝试多进程并行
        return self._run_battles_multiprocess(agent1_name, agent2_name, num_episodes, parallel_workers)
    except Exception as e:
        # 降级为顺序执行
        self.logger.warning('parallel', f"多进程执行失败，降级为顺序执行: {e}")
        return self._run_battles_sequential(agent1_name, agent2_name, num_episodes)
```

## 七、预期收益

### 7.1 性能提升

| 场景 | 当前方案 | 新方案 | 提升 |
|------|----------|--------|------|
| 2 workers | 顺序执行，无并行 | 真正并行 | 约 2x |
| 4 workers | 顺序执行，无并行 | 真正并行 | 约 4x |
| 8 workers | 顺序执行，无并行 | 真正并行 | 约 8x |

### 7.2 资源利用

```
当前方案（顺序执行）:
CPU 利用率: ~12.5% (单线程)

新方案（8 workers）:
CPU 利用率: ~100% (8核并行)
```

## 八、总结

### 8.1 方案要点

| 维度 | 说明 |
|------|------|
| **架构** | 多进程并行，进程间完全隔离 |
| **核心** | 每个进程独立加载 SDK 和 agent |
| **接口** | 保持原有 CLI 接口不变 |
| **降级** | 失败时自动降级为顺序执行 |

### 8.2 下一步行动

1. **审批**：请审阅并批准此方案
2. **实现**：按照步骤分解进行开发
3. **测试**：执行测试计划验证效果
4. **部署**：上线到生产环境

---

**文档版本**: v1.0  
**创建日期**: 2026-05-08  
**状态**: 待审阅