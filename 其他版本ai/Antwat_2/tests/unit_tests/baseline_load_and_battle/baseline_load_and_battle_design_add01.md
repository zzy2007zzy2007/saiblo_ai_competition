# Baseline Load and Battle 设计文档 - 补充01

## 1. ppo_v3 baseline battle 日志分析

### 1.1 日志充分性评估

**ppo_v3/baseline_battle_manager.py 日志特点：**

| 日志类型 | 内容 | 充分性 |
|----------|------|--------|
| 系统资源 | CPU使用率、内存可用量 | ✅ 优秀 |
| 加载状态 | 每个baseline模型的加载成功/失败 | ✅ 优秀 |
| 对战进度 | 环境创建、子进程启动/结束 | ✅ 优秀 |
| 错误处理 | 完整的异常堆栈跟踪 | ✅ 优秀 |
| 时间统计 | 对战开始/结束时间、持续时长 | ✅ 优秀 |
| 结果汇总 | 胜/负/平局数 | ✅ 优秀 |

**ppo_v3/battle_logic.py 日志特点：**

| 日志类型 | 内容 | 充分性 |
|----------|------|--------|
| 回合详情 | 每10回合记录HP状态 | ✅ 优秀 |
| 最终结果 | win/loss/draw/mutual_destroy | ✅ 优秀 |
| 奖励统计 | episode_reward | ✅ 良好 |
| 训练数据 | states/actions/rewards等收集 | ✅ 优秀 |

### 1.2 日志分级评估

**ppo_v3的分级方式：**
- `print()` 用于主要进度信息
- `logger.log_exception()` 用于异常记录
- `logger.log_warning()` 用于警告信息
- `battle_details['rounds']` 用于详细回合信息

**评估：**
- 分级基本合理，但可以更清晰
- 建议增加 DEBUG/INFO/WARNING/ERROR 四级

---

## 2. 代码借鉴清单

### 2.1 从 ppo_v3 借鉴到 unit_tests/baseline_load_and_battle

| # | 借鉴项 | 来源 | 说明 | 优先级 |
|---|--------|------|------|--------|
| 1 | **使用 resolve_turn() API** | 官方文档 | 当前代码使用错误的 apply_operation+advance_round | 🔴 必须 |
| 2 | **使用 state.winner 判断胜负** | 官方文档 | 当前手动比较HP | 🔴 必须 |
| 3 | **使用 state.terminal 判断结束** | 官方文档 | 当前手动检查HP | 🔴 必须 |
| 4 | **MAX_ROUNDS = 512** | constants.py | 当前使用200/1000 | 🔴 必须 |
| 5 | **详细battle_details结构** | battle_logic.py | 包含rounds数组、final_hp、result等 | 🟡 建议 |
| 6 | **时间戳记录** | battle_logic.py | battle_start_time, battle_end_time, duration | 🟡 建议 |
| 7 | **错误处理结构** | battle_logic.py | try-except-return error dict | 🟡 建议 |
| 8 | **mutual_destroy判定** | battle_logic.py | 双亡情况处理 | 🟢 可选 |
| 9 | **每N回合日志间隔** | battle_logic.py | current_round % 10 | 🟢 可选 |
| 10 | **worker_seed管理** | baseline_battle_manager.py | 随机种子可复现性 | 🟢 可选 |

### 2.2 从 unit_tests/baseline_load_and_battle 借鉴到 ppo_v3

| # | 借鉴项 | 来源 | 说明 | 优先级 |
|---|--------|------|------|--------|
| 1 | **独立测试框架** | baseline_load_and_battle | 可单独运行的测试 | 🟡 建议 |
| 2 | **ModelRegistry模式** | baseline_load_and_battle | 集中管理模型配置 | 🟢 可选 |
| 3 | **ModelLoader分离** | baseline_load_and_battle | 加载逻辑与对战分离 | 🟢 可选 |

---

## 3. 重构要点

### 3.1 必须修复的问题

1. **游戏推进API**
   ```python
   # 错误
   state.apply_operation(0, op1)
   state.apply_operation(1, op2)
   state.advance_round()

   # 正确
   state.resolve_turn([op1], [op2])
   ```

2. **胜负判定**
   ```python
   # 错误
   if state.bases[0].hp > 0 and state.bases[1].hp <= 0:
       result = 'win'

   # 正确
   if state.winner == 0:
       result = 'win'
   ```

3. **游戏结束判定**
   ```python
   # 错误
   while state.bases[0].hp > 0 and state.bases[1].hp > 0:

   # 正确
   while not state.terminal:
   ```

4. **最大回合数**
   ```python
   # 错误
   max_rounds = 1000

   # 正确
   MAX_ROUNDS = 512  # 来自 SDK/utils/constants.py
   ```

### 3.2 建议增加的功能

1. **详细battle_details结构**
   ```python
   battle_details = {
       'opponent': opponent_name,
       'start_time': battle_start_str,
       'end_time': battle_end_str,
       'duration': battle_duration,
       'rounds': [],  # 每N回合的HP快照
       'final_hp': {'our': 0, 'enemy': 0},
       'result': 'win/loss/draw',
       'reward': episode_reward,
       'error': None  # 如果有错误
   }
   ```

2. **日志分级输出**
   - DEBUG: 详细调试信息
   - INFO: 主要进度信息
   - WARNING: 资源警告等
   - ERROR: 异常错误

3. **配置常量**
   ```python
   BATTLE_PRINT_INTERVAL = 10  # 每10回合打印一次
   MAX_ROUNDS = 512  # 官方标准
   BATTLE_TIMEOUT = 300  # 超时秒数
   ```

---

## 4. 重构后的代码结构

```
baseline_load_and_battle/
├── main.py                    # 测试入口
├── model_registry.py          # 模型注册表
├── model_loader.py            # 模型加载器
├── battle_simulator.py       # 对战模拟器（核心重构）
├── constants.py               # 新增：配置常量
├── logger.py                  # 新增：日志工具
└── design/
    └── baseline_load_and_battle_design.md
    └── baseline_load_and_battle_design_add01.md  # 本文档
```

---

## 5. 日志分级标准

| 级别 | 触发条件 | 输出内容 |
|------|----------|----------|
| DEBUG | 每回合 | 详细操作、观察、动作 |
| INFO | 每10回合 | HP状态、金币、回合数 |
| WARNING | 资源不足、超时 | 警告信息 |
| ERROR | 异常、API错误 | 错误堆栈、详细信息 |

---

## 6. 预期改进效果

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| API正确性 | ❌ 使用错误API | ✅ 使用resolve_turn |
| 胜负判定 | ❌ 手动HP比较 | ✅ state.winner |
| 最大回合 | ❌ 200/1000 | ✅ 512 |
| 日志详细度 | ⚠️ 简单 | ✅ 完整 |
| 错误处理 | ❌ 无 | ✅ 完善 |
| 可复现性 | ⚠️ 一般 | ✅ 种子管理 |