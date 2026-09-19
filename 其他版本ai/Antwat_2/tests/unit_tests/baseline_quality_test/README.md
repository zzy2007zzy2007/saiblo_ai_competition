# 策略对战测试框架文档

## 1. 概述

本文档详细介绍了 AntWar 项目中的策略对战测试框架，用于测试和评估不同智能体策略的性能。该框架采用模块化设计，支持灵活的配置管理和结果分析。

## 2. 代码结构

```
AntWar/
├── tests/                     # 测试框架
│   ├── evaluation/           # 测试评估模块
│   │   ├── battle_engine.py  # 对战核心引擎
│   │   ├── strategy_loader.py # 策略加载器
│   │   ├── result_analyzer.py # 结果分析器
│   │   ├── compare_strategies.py # 主测试脚本
│   │   ├── config.py         # 配置管理
│   │   └── config_example.json # 配置示例
│   └── strategies/           # 策略定义
│       ├── __init__.py
│       ├── sample.py         # Sample策略
│       ├── gen99.py          # Gen99策略
│       ├── gen199.py         # Gen199策略
│       ├── ann_v1.py         # Ann_v1策略
│       ├── ann_593.py        # Ann_593策略
│       └── basic_tower.py    # BasicTower策略
```

## 3. 核心模块说明

### 3.1 evaluation 模块

#### 3.1.1 battle_engine.py
- **功能**：对战核心引擎，负责运行两个智能体的对战
- **主要方法**：`run_battle(agent1, agent2, episodes, max_rounds, verbose)`
- **参数说明**：
  - `agent1`：第一个智能体实例
  - `agent2`：第二个智能体实例
  - `episodes`：对战轮数
  - `max_rounds`：每轮最大回合数
  - `verbose`：是否打印详细信息
- **返回值**：包含胜场、平均回合数等信息的结果字典

#### 3.1.2 strategy_loader.py
- **功能**：策略加载器，从策略文件加载策略实例
- **主要方法**：`load_strategy(name)`
- **参数说明**：
  - `name`：策略名称（如"sample"、"gen99"等）
- **返回值**：对应策略的实例

#### 3.1.3 result_analyzer.py
- **功能**：结果分析器，分析对战结果并生成报告
- **主要方法**：
  - `analyze_results(results)`：分析对战结果
  - `generate_report(analysis, strategies)`：生成分析报告
  - `export_results(results, analysis, output_file)`：导出结果到文件

#### 3.1.4 compare_strategies.py
- **功能**：主测试脚本，支持通过命令行参数或配置文件控制测试
- **使用方式**：
  - 命令行参数：`python tests/evaluation/compare_strategies.py --strategies sample gen99 --episodes 50`
  - 配置文件：`python tests/evaluation/compare_strategies.py --config tests/evaluation/config_example.json`

#### 3.1.5 config.py
- **功能**：配置管理，支持从文件加载配置和命令行参数覆盖
- **主要方法**：`load_config(args)`
- **配置项**：
  - `episodes`：对战轮数
  - `max_rounds`：每轮最大回合数
  - `strategies`：测试策略列表
  - `output_file`：结果输出文件
  - `verbose`：是否打印详细信息

### 3.2 strategies 模块

- **功能**：策略定义模块，包含各种智能体策略的实现
- **策略列表**：
  - `sample.py`：Sample策略
  - `gen99.py`：Gen99策略
  - `gen199.py`：Gen199策略
  - `ann_v1.py`：Ann_v1策略
  - `ann_593.py`：Ann_593策略
  - `basic_tower.py`：BasicTower策略

## 4. 策略定义

每个策略都需要实现 `select_action` 方法，该方法接收当前游戏状态和玩家座位，返回要执行的操作列表。

**示例**（sample.py）：

```python
class Sample:
    def __init__(self):
        self.name = "sample"
    
    def select_action(self, state, my_seat):
        actions = []
        # 策略逻辑...
        return actions
```

## 5. 使用方法

### 5.1 命令行方式

```bash
# 测试指定策略
python tests/evaluation/compare_strategies.py --strategies sample gen99 ann_v1 --episodes 50

# 使用配置文件
python tests/evaluation/compare_strategies.py --config tests/evaluation/config_example.json

# 导出结果
python tests/evaluation/compare_strategies.py --strategies sample gen99 --episodes 20 --output results.json
```

### 5.2 配置文件方式

**config_example.json**：

```json
{
  "episodes": 50,
  "max_rounds": 500,
  "strategies": ["sample", "gen99", "ann_v1"],
  "output_file": "results.json",
  "verbose": true
}
```

## 6. 测试结果分析

测试完成后，会生成详细的分析报告，包括：

- **策略性能排名**：按胜率排序
- **头对头对战结果**：每个策略对之间的胜率
- **详细对战数据**：胜场、胜率、平均回合数

**示例输出**：

```
================================================================================
策略对战分析报告
================================================================================
总对战场次: 30

策略性能排名:
--------------------------------------------------------------------------------
1. sample     | 胜率: 56.67% | 胜场: 17
2. gen99      | 胜率: 6.67% | 胜场: 2
3. ann_v1     | 胜率: 3.33% | 胜场: 1

头对头对战结果:
--------------------------------------------------------------------------------
sample     vs gen99     
  sample: 80.00% (8胜)
  gen99: 20.00% (2胜)
  平均回合数: 167.20

...
```

## 7. 扩展与维护

### 7.1 添加新策略

1. 在 `tests/strategies/` 目录下创建新的策略文件（如 `new_strategy.py`）
2. 实现 `select_action` 方法
3. 在 `tests/strategies/__init__.py` 中导入并导出新策略
4. 在 `tests/evaluation/strategy_loader.py` 中添加策略加载逻辑

### 7.2 自定义测试逻辑

- 修改 `tests/evaluation/battle_engine.py` 调整对战逻辑
- 修改 `tests/evaluation/result_analyzer.py` 调整结果分析方式
- 修改 `tests/evaluation/compare_strategies.py` 调整测试流程

## 8. 常见问题

### 8.1 策略加载失败
- 检查策略名称是否正确
- 检查策略文件是否存在
- 检查策略类是否正确实现

### 8.2 对战结果异常
- 检查策略的 `select_action` 方法是否正确实现
- 检查游戏状态处理逻辑
- 检查最大回合数设置是否合理

### 8.3 性能问题
- 减少对战轮数（episodes）
- 减少每轮最大回合数（max_rounds）
- 优化策略的 `select_action` 方法

## 9. 最佳实践

- **策略分离**：每个策略单独实现，便于维护和测试
- **配置管理**：使用配置文件管理测试参数，提高可重复性
- **结果分析**：利用分析报告了解策略性能，指导策略优化
- **模块化设计**：保持代码的高内聚低耦合，便于扩展

## 10. 示例测试

### 10.1 测试 sample 与其他策略的对战

```bash
python tests/evaluation/compare_strategies.py --strategies sample gen99 gen199 ann_v1 ann_593 basic_tower --episodes 20
```

### 10.2 测试特定策略组合

```bash
python tests/evaluation/compare_strategies.py --strategies gen99 ann_v1 --episodes 100
```

## 11. 总结

本测试框架为 AntWar 项目提供了一个灵活、强大的策略对战测试工具，支持多种测试场景和结果分析。通过模块化设计和配置管理，使得测试过程更加规范和可重复，为策略的开发和优化提供了有力的支持。