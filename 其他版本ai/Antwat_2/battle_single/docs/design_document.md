# Battle Single 简版对战系统设计文档

## 1. 系统概述

### 1.1 目标

设计一个轻量级的单线程对战评测系统，用于评估两个 agent 之间的对战胜率。相对于完整的 battle 系统，battle_single 追求简洁、高效和易于理解。

### 1.2 核心需求

| 序号 | 需求 | 描述 |
|------|------|------|
| 1 | 简化 SDK 隔离 | 不做 SDK 隔离，统一使用 agent1 的 SDK |
| 2 | 单线程执行 | 不需要多线程，简化执行流程 |
| 3 | 接口一致性 | 与 battle 系统保持一致的 CLI 接口和输出格式 |
| 4 | 对战局数配置 | 支持通过参数设定对战局数 |
| 5 | 先后手公平性 | 每局对战两次，先手后手各一次 |
| 6 | 对战日志 | 记录详细的对战过程，支持诊断 |
| 7 | 胜率统计 | 对战结束后输出详细的胜率统计 |
| 8 | 容错能力 | 对战非正常结束时仍能获取已完成的对战信息 |

### 1.3 与 battle 系统对比

| 特性 | battle | battle_single |
|------|--------|---------------|
| SDK 隔离 | 使用子进程隔离 | 不隔离，使用 agent1 的 SDK |
| 并行执行 | 多进程并行 | 单线程顺序执行 |
| Agent 加载 | 序列化后跨进程传递 | 直接加载到当前进程 |
| 复杂度 | 高 | 低 |
| 适用场景 | 大规模评测 | 快速测试、调试、开发 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────┐
│              Battle Single 简版对战系统                    │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐                      │
│  │   CLI 入口   │───▶│  配置解析器  │                      │
│  └─────────────┘    └──────┬──────┘                      │
│                            ▼                             │
│              ┌───────────────────────┐                    │
│              │     对战模拟器        │                    │
│              │  (BattleSimulator)    │                    │
│              └───────────┬───────────┘                    │
│                          ▼                               │
│              ┌───────────────────────┐                    │
│              │     结果汇总器        │                    │
│              │  (ResultAggregator)   │                    │
│              └───────────┬───────────┘                    │
│                          ▼                               │
│              ┌───────────────────────┐                    │
│              │     日志记录器        │                    │
│              │   (BattleLogger)      │                    │
│              └───────────┬───────────┘                    │
│                          ▼                               │
│              ┌───────────────────────┐                    │
│              │    输出报告           │                    │
│              └───────────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

### 2.2 核心组件说明

| 组件 | 职责 | 说明 |
|------|------|------|
| **CLI 入口** | 命令行参数解析 | 接收用户输入的对战配置 |
| **配置解析器** | 解析配置文件和命令行参数 | 统一管理对战参数 |
| **对战模拟器** | 执行实际对战 | 使用 agent1 的 SDK 作为对战引擎 |
| **结果汇总器** | 汇总所有对战结果 | 计算胜率统计 |
| **日志记录器** | 记录对战过程和结果 | 支持诊断分析 |

---

## 3. 目录结构

```
battle_single/                        # 简版对战系统根目录
├── docs/                             # 文档目录
│   └── design_document.md            # 设计文档（本文件）
├── src/                              # 源代码目录
│   ├── __init__.py
│   ├── battle_single_cli.py          # CLI 入口
│   ├── battle_single_config.py       # 配置管理
│   ├── battle_single_simulator.py    # 对战模拟器
│   ├── battle_single_logger.py       # 日志记录器
│   └── agent_loader.py               # Agent 加载器
├── config/                           # 配置文件目录
│   └── default_config.yaml           # 默认配置
└── scripts/                          # 脚本目录
    └── run_battle_single.sh          # 启动脚本
```

---

## 4. 核心组件设计

### 4.1 Agent 加载器 (agent_loader.py)

**职责**: 加载指定的 agent，直接加载到当前进程

**设计要点**:
- 直接在当前进程加载 agent，不使用子进程隔离
- 使用 agent1 的 SDK 作为对战引擎
- 支持从 baseline 目录加载 agent

**关键接口**:

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `load_agent(agent_name, baseline_path)` | 加载指定的 agent | `agent_name`: str; `baseline_path`: str | 返回 agent 对象 |
| `get_available_agents(baseline_path)` | 获取可用的 agent 名称列表 | `baseline_path`: str | 返回名称列表 |

### 4.2 对战模拟器 (battle_single_simulator.py)

**职责**: 执行两个 agent 之间的对战

**设计要点**:
- 使用 agent1 的 SDK 作为对战引擎
- 支持先手/后手切换
- 记录详细的对战过程日志
- 处理异常情况，确保对战可恢复

**关键接口**:

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `run_battle(agent1, agent2, agent1_name, agent2_name, seed, first_player)` | 执行单场对战 | `agent1/agent2`: agent 对象; `seed`: int; `first_player`: int | 返回对战结果字典 |
| `run_battle_with_swap(agent1, agent2, agent1_name, agent2_name, seed)` | 执行两局对战（交换先后手） | 同上 | 返回两局对战结果 |

**对战结果数据结构**:

```python
{
    'episode': int,              # 对战序号
    'agent1_name': str,          # 玩家1名称
    'agent2_name': str,          # 玩家2名称
    'first_player': str,         # 先手玩家名称
    'result': str,               # 结果: 'agent1_win', 'agent2_win', 'draw', 'error'
    'start_time': str,           # 开始时间
    'end_time': str,             # 结束时间
    'duration': float,           # 耗时（秒）
    'total_rounds': int,         # 总回合数
    'final_hp': {               # 最终血量
        'agent1': int,
        'agent2': int
    },
    'error': str or None         # 错误信息（如有）
}
```

### 4.3 日志记录器 (battle_single_logger.py)

**职责**: 记录对战过程和结果日志

**设计要点**:
- 支持分级日志（DEBUG, INFO, WARNING, ERROR）
- 支持输出到文件和控制台
- 记录对战详情用于诊断

**关键接口**:

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `log_battle_start(agent1, agent2, episode)` | 记录对战开始 | 对战双方和序号 | 无 |
| `log_round(episode, round_num, hp1, hp2)` | 记录回合信息 | 序号、回合数、血量 | 无 |
| `log_battle_end(result)` | 记录对战结束 | 对战结果 | 无 |
| `log_error(episode, error)` | 记录错误 | 序号和错误信息 | 无 |

### 4.4 配置管理 (battle_single_config.py)

**职责**: 管理对战配置参数

**设计要点**:
- 支持从配置文件加载
- 支持从命令行参数覆盖
- 保持与 battle 系统一致的配置项

**配置项**:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `num_episodes` | 10 | 每对对战的局数 |
| `max_rounds` | 512 | 单场对战最大回合数 |
| `log_level` | INFO | 日志级别 |
| `log_dir` | /tmp/battle_single/logs | 日志目录 |
| `log_to_console` | true | 是否输出到控制台 |
| `output_report_file` | /tmp/battle_single/report.txt | 报告输出文件 |

---

## 5. 执行流程

### 5.1 整体流程

```
1. 用户启动命令
       │
       ▼
2. CLI 解析参数
       │
       ▼
3. 加载配置文件
       │
       ▼
4. 加载 agent1 和 agent2
       │
       ▼
5. 加载 agent1 的 SDK 作为对战引擎
       │
       ▼
6. 循环执行对战（每局交换先后手）
       │
       ▼
7. 收集对战结果
       │
       ▼
8. 汇总统计
       │
       ▼
9. 生成报告输出
```

### 5.2 单场对战流程

```
1. 初始化游戏状态（使用 agent1 的 SDK）
       │
       ▼
2. 循环执行回合：
   ├─ 获取 agent1 操作（使用 agent1 的 SDK）
   ├─ 获取 agent2 操作（使用 agent1 的 SDK 进行状态转换）
   ├─ 使用 agent1 的 SDK 执行回合
   └─ 记录回合日志
       │
       ▼
3. 判断游戏结束条件
       │
       ▼
4. 计算对战结果
       │
       ▼
5. 记录对战日志
```

---

## 6. 配置说明

### 6.1 配置文件格式 (YAML)

```yaml
# default_config.yaml
battle:
  num_episodes: 10                    # 每对对战的局数（每局交换先后手，实际对战数 x2）
  max_rounds: 512                    # 单场对战最大回合数
  
logging:
  log_level: INFO                     # 日志级别: DEBUG, INFO, WARNING, ERROR
  log_dir: /tmp/battle_single/logs    # 日志目录
  log_to_console: true                # 是否输出到控制台
  
output:
  report_file: /tmp/battle_single/report.txt  # 报告输出文件
  result_json: /tmp/battle_single/results.json # 结果 JSON 文件
```

### 6.2 命令行参数

```bash
usage: battle_single_cli.py [-h] [--config CONFIG] [--episodes EPISODES]
                            [--max-rounds MAX_ROUNDS] [--log-level LOG_LEVEL]
                            [--agent1 AGENT1] [--agent2 AGENT2] [--output OUTPUT]

Battle Single 简版对战系统

optional arguments:
  -h, --help            显示帮助信息
  --config CONFIG       配置文件路径
  --episodes EPISODES   对战局数（每局交换先后手）
  --max-rounds MAX_ROUNDS  单场对战最大回合数
  --log-level LOG_LEVEL  日志级别: DEBUG, INFO, WARNING, ERROR
  --agent1 AGENT1       第一个 agent 名称
  --agent2 AGENT2       第二个 agent 名称
  --output OUTPUT       报告输出文件路径
```

---

## 7. 容错设计

### 7.1 异常处理

| 异常类型 | 处理策略 |
|----------|----------|
| Agent 加载失败 | 记录错误日志，退出程序 |
| 单场对战异常 | 记录错误，继续下一场对战 |
| Agent 操作获取失败 | 使用空操作继续对战 |

### 7.2 结果完整性

即使部分对战失败，系统仍会：
- 汇总已完成的对战结果
- 统计异常对战数量
- 在报告中明确标识异常情况

---

## 8. 部署与执行

### 8.1 目录结构要求

所有对战测试在 `/tmp/battle_single` 目录执行：

```
/tmp/battle_single/
├── logs/                        # 日志目录
├── results/                     # 结果目录
├── config.yaml                  # 配置文件
└── run_battle_single.sh         # 启动脚本
```

### 8.2 启动命令

```bash
# 使用默认配置
python -m battle_single.src.battle_single_cli --agent1 ann_v1 --agent2 gen99

# 指定参数
python -m battle_single.src.battle_single_cli \
    --agent1 ann_v1 \
    --agent2 gen99 \
    --episodes 10 \
    --output /tmp/battle_single/report.txt

# 使用配置文件
python -m battle_single.src.battle_single_cli --config /tmp/battle_single/config.yaml
```

---

## 9. 输出报告示例

```
========================================
        Battle Single 对战评测报告
========================================

执行时间: 2024-01-15 10:30:00
对战局数: 10 (每局交换先后手)
========================================

【总体统计】
总对战数: 20
完成对战: 20
异常对战: 0
========================================

【对战结果】
┌────────────┬────────────┬───────┬───────┬──────┬────────┐
│   Agent1   │   Agent2   │  胜   │  负  │  平   │  胜率   │
├────────────┼────────────┼───────┼──────┼───────┼────────┤
│   ann_v1   │   gen99    │  12   │  6   │  2    │  60.0% │
└────────────┴────────────┴───────┴──────┴───────┴────────┘

【详细对战记录】
ann_v1 vs gen99: 12胜 6负 2平 (胜率 60.0%)

【先后手胜率】
ann_v1 - 先手: 7胜 3负 (70.0%)
ann_v1 - 后手: 5胜 3负 (62.5%)

========================================
报告生成时间: 2024-01-15 10:32:15
耗时: 2分15秒
========================================
```

---

## 10. 代码安全性

### 10.1 注意事项

| 风险点 | 描述 | 缓解措施 |
|--------|------|----------|
| 路径遍历 | 恶意 agent 路径可能导致文件系统访问 | 使用白名单验证 agent 名称 |
| 资源耗尽 | 长时间对战可能耗尽系统资源 | 设置对战超时时间 |
| 日志注入 | 恶意 agent 名称可能包含特殊字符 | 对输出进行转义处理 |

### 10.2 安全实践

1. **输入验证**: 所有输入参数进行严格验证
2. **资源限制**: 设置最大回合数和超时时间
3. **日志审查**: 定期审查日志内容

---

## 11. 设计原则

### 11.1 KISS 原则

保持代码简洁，去掉冗余：
- 移除多进程调度逻辑
- 移除 SDK 隔离机制
- 使用直接函数调用代替进程间通信

### 11.2 接口一致性

与 battle 系统保持一致：
- 相同的 CLI 参数格式
- 相同的输出报告格式
- 相同的结果数据结构