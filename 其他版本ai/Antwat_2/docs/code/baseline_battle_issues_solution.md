# Baseline Battle 代码改造方案

## 一、现状分析

### 1.1 核心问题

根据文档 `docs/code/baseline_battle_issues.md` 的分析和代码审查，当前 `baseline_battle_runner.py` 存在以下问题：

| 问题类别 | 具体表现 | 影响 |
|---------|---------|------|
| **SDK隔离问题** | 每个baseline agent使用自带的SDK，通过sys.path切换 | 模块冲突、代码复杂、难以维护 |
| **规则重复实现** | `_action_to_string`、`_operations_to_action`硬编码使用`HIGHLAND_CELLS[0]` | 违反设计原则，日志显示错误 |
| **代码重复** | `_run_battle_process`中重复实现了相同的函数 | 维护困难、bug难以同步修复 |
| **验证逻辑分散** | 动作验证在多个位置实现，策略不一致 | 行为不一致、难以追踪 |

### 1.2 当前架构缺陷

```
┌─────────────────────────────────────────────────────────────────┐
│                    BaselineBattleRunner                        │
├─────────────────────────────────────────────────────────────────┤
│  _get_agent_action()                                           │
│    ├── _call_genagent_choose_bundle()  ← 切换SDK路径           │
│    └── _call_genagent_choose_operations() ← 切换SDK路径        │
│  _operations_to_action() ← 硬编码HIGHLAND_CELLS[0]             │
│  _action_to_string() ← 硬编码HIGHLAND_CELLS[0]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    _run_battle_process()                       │
│  ← 完全重复的函数实现，维护困难                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、改造目标

基于设计原则 `/docs/design/principle.txt#L2-10`，改造目标如下：

1. **SDK统一**：baseline agent不再使用自己的SDK，全部使用项目SDK
2. **规则委托**：动作合法检查依赖SDK，而非重复实现游戏规则
3. **代码去重**：消除`_run_battle_process`中的重复代码
4. **异常处理**：合理捕获并记录异常，抛向上层便于追踪

---

## 三、改造方案

### 3.1 方案一：SDK统一注入机制（推荐）

**核心思路**：创建统一的SDK路径管理机制，在agent调用时注入项目SDK，而非切换到baseline自带的SDK。

#### 3.1.1 创建统一的SDK注入工具类

**新增文件**: `ppo/src/ppo_antwar/league/sdk_injector.py`

```python
from __future__ import annotations
import sys
import os
from contextlib import contextmanager
from typing import Optional


class SDKInjector:
    """统一的SDK路径注入器"""
    
    PROJECT_SDK_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        '..', '..', '..', 'SDK'
    )
    
    @classmethod
    @contextmanager
    def inject_project_sdk(cls, baseline_dir: Optional[str] = None):
        """
        注入项目SDK到sys.path
        
        Args:
            baseline_dir: baseline agent所在目录（用于兼容旧版agent）
        """
        original_sys_path = sys.path.copy()
        original_modules = cls._get_sdk_modules()
        
        try:
            # 构建路径：项目SDK优先，然后是baseline目录（如果提供）
            new_path = []
            
            # 添加项目SDK
            project_sdk = os.path.abspath(cls.PROJECT_SDK_PATH)
            if os.path.exists(project_sdk):
                new_path.insert(0, project_sdk)
            
            # 添加项目根目录（确保SDK包可被导入）
            project_root = os.path.dirname(project_sdk)
            if os.path.exists(project_root):
                new_path.insert(0, project_root)
            
            # 如果提供了baseline目录，添加到路径末尾（兼容旧代码）
            if baseline_dir and os.path.exists(baseline_dir):
                new_path.append(baseline_dir)
            
            # 添加原路径（确保其他依赖可用）
            new_path.extend([p for p in original_sys_path if p not in new_path])
            
            sys.path = new_path
            
            # 清理已加载的SDK模块，强制重新导入
            cls._clear_sdk_modules()
            
            yield
        finally:
            # 恢复原始状态
            sys.path = original_sys_path
            cls._restore_sdk_modules(original_modules)
    
    @classmethod
    def _get_sdk_modules(cls) -> dict:
        """获取当前已加载的SDK相关模块"""
        sdk_modules = {}
        for name, module in sys.modules.items():
            if name.startswith('SDK') or (module and hasattr(module, '__file__') and 'SDK' in module.__file__):
                sdk_modules[name] = module
        return sdk_modules
    
    @classmethod
    def _clear_sdk_modules(cls):
        """清理SDK相关模块"""
        modules_to_remove = [name for name in sys.modules.keys() if name.startswith('SDK')]
        for name in modules_to_remove:
            del sys.modules[name]
    
    @classmethod
    def _restore_sdk_modules(cls, modules: dict):
        """恢复SDK模块"""
        sys.modules.update(modules)
```

#### 3.1.2 修改 `baseline_agent.py` 使用统一SDK注入

```python
# 修改 ppo/src/ppo_antwar/league/baseline_agent.py

class GenAgent:
    """外部生成的Agent包装器"""
    
    # ... 其他代码 ...
    
    def _call_agent_method(self, method_name: str, state, player_position: int):
        """
        统一调用agent方法，使用项目SDK
        
        Args:
            method_name: 方法名（choose_bundle 或 choose_operations）
            state: 游戏状态
            player_position: 玩家位置
        
        Returns:
            操作列表或None
        """
        from ppo_antwar.league.sdk_injector import SDKInjector
        
        with SDKInjector.inject_project_sdk(self.baseline_dir):
            agent = self._get_agent()
            
            # 使用项目SDK创建PythonBackendState
            try:
                from SDK.backend.state import PythonBackendState
                raw_state = state
                if hasattr(state, '_state'):
                    raw_state = state._state
                sdk_state = PythonBackendState(raw_state)
            except Exception as e:
                logger.warning(f"Failed to create PythonBackendState: {e}")
                sdk_state = state
            
            method = getattr(agent, method_name, None)
            if method:
                try:
                    result = method(sdk_state, player_position)
                    if hasattr(result, 'operations'):
                        return result.operations
                    return result
                except Exception as e:
                    logger.error(f"Error calling {method_name}: {e}")
                    raise
        
        return None
    
    def choose_bundle(self, state, player_position: int):
        """调用agent的choose_bundle方法"""
        return self._call_agent_method('choose_bundle', state, player_position)
    
    def choose_operations(self, state, player_position: int):
        """调用agent的choose_operations方法"""
        return self._call_agent_method('choose_operations', state, player_position)
```

---

### 3.2 方案二：移除规则重复实现

**核心思路**：修改`_action_to_string`和`_operations_to_action`函数，移除硬编码的`HIGHLAND_CELLS[0]`，改为依赖SDK或接收玩家位置参数。

#### 3.2.1 修改 `_action_to_string` 函数

```python
# 修改 ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py

def _action_to_string(action: int, player_position: int = 0) -> str:
    """
    将动作ID转换为可读字符串（包含参数信息）
    
    Args:
        action: 动作ID
        player_position: 玩家位置（0=先手，1=后手）
    
    Returns:
        可读的动作描述字符串
    """
    from ppo_antwar.utils.action_constants import ACTION_SPACE_CONFIG, HIGHLAND_CELLS
    
    if action == 0:
        return "NO_OP"
    
    # 获取正确的高地坐标
    highland_cells = HIGHLAND_CELLS.get(player_position, HIGHLAND_CELLS.get(0, []))
    
    if ACTION_SPACE_CONFIG['build_tower']['start'] <= action < ACTION_SPACE_CONFIG['build_tower']['end']:
        tower_idx = action - ACTION_SPACE_CONFIG['build_tower']['start']
        if tower_idx < len(highland_cells):
            pos = highland_cells[tower_idx]
            return f"BUILD_TOWER(pos={pos})"
        else:
            return f"BUILD_TOWER(index={tower_idx})"
    
    if ACTION_SPACE_CONFIG['upgrade_tower']['start'] <= action < ACTION_SPACE_CONFIG['upgrade_tower']['end']:
        tower_idx = action - ACTION_SPACE_CONFIG['upgrade_tower']['start']
        if tower_idx < len(highland_cells):
            pos = highland_cells[tower_idx]
            return f"UPGRADE_TOWER(pos={pos})"
        else:
            return f"UPGRADE_TOWER(index={tower_idx})"
    
    if ACTION_SPACE_CONFIG['downgrade_tower']['start'] <= action < ACTION_SPACE_CONFIG['downgrade_tower']['end']:
        tower_idx = action - ACTION_SPACE_CONFIG['downgrade_tower']['start']
        if tower_idx < len(highland_cells):
            pos = highland_cells[tower_idx]
            return f"DOWNGRADE_TOWER(pos={pos})"
        else:
            return f"DOWNGRADE_TOWER(index={tower_idx})"
    
    # ... 其他动作类型处理 ...
    
    return f"UNKNOWN_{action}"
```

#### 3.2.2 修改 `_operations_to_action` 函数

```python
# 修改 ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py

def _operations_to_action(operations, player_position: int = 0) -> int:
    """
    将SDK操作列表转换为动作ID
    
    Args:
        operations: SDK操作列表
        player_position: 玩家位置（0=先手，1=后手）
    
    Returns:
        动作ID（0表示无效）
    """
    from ppo_antwar.utils.action_constants import OperationType, ACTION_SPACE_CONFIG, HIGHLAND_CELLS
    
    if not operations:
        return 0

    op = operations[0]
    if hasattr(op, 'type'):
        op_type = op.type
    elif hasattr(op, 'operation_type'):
        op_type = op.operation_type
    elif hasattr(op, 'op_type'):
        op_type = op.op_type
    else:
        return 0

    op_type_int = int(op_type)
    
    # 获取正确的高地坐标
    highland_cells = HIGHLAND_CELLS.get(player_position, HIGHLAND_CELLS.get(0, []))

    if op_type_int == int(OperationType.BUILD_TOWER):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        try:
            idx = highland_cells.index(pos)
            return ACTION_SPACE_CONFIG['build_tower']['start'] + idx
        except ValueError:
            logger.warning(f"Build tower position {pos} not in player {player_position}'s highland")
            return ACTION_SPACE_CONFIG['build_tower']['start']
    elif op_type_int == int(OperationType.UPGRADE_TOWER):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        try:
            idx = highland_cells.index(pos)
            return ACTION_SPACE_CONFIG['upgrade_tower']['start'] + idx
        except ValueError:
            logger.warning(f"Upgrade tower position {pos} not in player {player_position}'s highland")
            return ACTION_SPACE_CONFIG['upgrade_tower']['start']
    elif op_type_int == int(OperationType.UPGRADE_GENERATION_SPEED):
        return ACTION_SPACE_CONFIG['upgrade_generation_speed']['start']
    elif op_type_int == int(OperationType.UPGRADE_GENERATED_ANT):
        return ACTION_SPACE_CONFIG['upgrade_generated_ant']['start']

    return 0
```

---

### 3.3 方案三：重构 `_get_agent_action` 函数

**核心思路**：简化`_get_agent_action`函数，移除重复的验证逻辑，完全依赖SDK进行合法性检查。

```python
# 修改 ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py

def _get_agent_action(agent, obs: Dict, player_position: int, state, engine: Any = None) -> int:
    """
    统一调用不同类型agent的动作选择方法
    
    Args:
        agent: agent实例
        obs: 观察值
        player_position: 玩家位置
        state: 游戏状态
        engine: 游戏引擎实例（用于SDK调用）
    
    Returns:
        动作ID（0表示无效）
    """
    from ppo_antwar.league.baseline_agent import GenAgent
    
    is_gen_agent = isinstance(agent, GenAgent)
    
    try:
        # GenAgent 或需要SDK状态的agent
        if is_gen_agent or (engine is not None and (hasattr(agent, 'choose_bundle') or hasattr(agent, 'choose_operations'))):
            if engine is None:
                logger.warning("Engine is None, cannot call SDK-based agent")
                return 0
            
            sdk_state = engine.get_sdk_state()
            
            if hasattr(agent, 'choose_bundle'):
                operations = agent.choose_bundle(sdk_state, player_position)
            elif hasattr(agent, 'choose_operations'):
                operations = agent.choose_operations(sdk_state, player_position)
            else:
                return 0
            
            if operations:
                # 依赖SDK进行合法性检查
                valid_ops = []
                for op in operations:
                    if hasattr(engine, 'can_apply_operation'):
                        try:
                            if engine.can_apply_operation(player_position, op):
                                valid_ops.append(op)
                            else:
                                logger.debug(f"SDK rejected operation: {op}")
                        except Exception as e:
                            logger.warning(f"SDK validation error: {e}, allowing operation")
                            valid_ops.append(op)
                    else:
                        valid_ops.append(op)
                
                if valid_ops:
                    return _operations_to_action(valid_ops, player_position)
            return 0
        
        # 标准处理路径（PPO agent）
        if hasattr(agent, 'select_action'):
            return agent.select_action(obs)
        
        # 降级处理（使用简化状态）
        if hasattr(agent, 'choose_operations'):
            operations = agent.choose_operations(state, player_position)
            if operations:
                return _operations_to_action(operations, player_position)
            return 0
        
        if hasattr(agent, 'choose_bundle'):
            operations = agent.choose_bundle(state, player_position)
            if operations:
                return _operations_to_action(operations, player_position)
            return 0
        
    except Exception as e:
        logger.error(f"Error getting agent action: {e}")
        raise
    
    return 0
```

---

### 3.4 方案四：消除代码重复

**核心思路**：将`_run_battle_process`中的重复函数提取为模块级函数，子进程直接调用。

#### 3.4.1 创建独立的子进程执行模块

**新增文件**: `ppo/src/ppo_antwar/league/battle_worker.py`

```python
from __future__ import annotations
import numpy as np
from loguru import logger
from typing import Dict, Any
from .baseline_battle_runner import (
    AntWarBattleEngine,
    BattleOutcome,
    SubProcessLogger,
    _get_agent_action,
    _action_to_string
)


def run_battle_worker(task: Dict[str, Any]) -> BattleOutcome:
    """
    在子进程中执行单场对战
    
    Args:
        task: 任务字典，包含agent1, agent2, episode, player_position等
    
    Returns:
        BattleOutcome对象
    """
    from enum import Enum
    
    class LogLevel(Enum):
        DEBUG = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
    
    agent1 = task['agent1']
    agent2 = task['agent2']
    episode = task['episode']
    battle_identifier = task.get('battle_identifier', f"{task['agent1_name']}_vs_{task['agent2_name']}_episode_{episode}")
    player_position = task['player_position']
    agent1_name = task['agent1_name']
    agent2_name = task['agent2_name']
    log_level_value = task['log_level']
    
    log_level = LogLevel(log_level_value)
    sub_logger = SubProcessLogger(agent1_name, agent2_name, log_level)
    
    start_time = np.datetime64('now')
    sub_logger.log_battle_start(battle_identifier)
    
    try:
        engine = AntWarBattleEngine(player_id=player_position, seed=episode)
        obs = engine.reset(player=player_position, seed=episode)
        
        total_rounds = 0
        hp1, hp2 = 50, 50
        coins1, coins2 = 0, 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated and total_rounds < 512:
            total_rounds += 1
            state = engine.get_state()
            
            # 更新HP状态
            if state.bases:
                if player_position == 0:
                    hp1 = state.bases[0].hp if len(state.bases) > 0 else 50
                    hp2 = state.bases[1].hp if len(state.bases) > 1 else 50
                else:
                    hp1 = state.bases[1].hp if len(state.bases) > 1 else 50
                    hp2 = state.bases[0].hp if len(state.bases) > 0 else 50
            
            # 更新金币状态
            if state.coins:
                if player_position == 0:
                    coins1 = state.coins[0] if len(state.coins) > 0 else 0
                    coins2 = state.coins[1] if len(state.coins) > 1 else 0
                else:
                    coins1 = state.coins[1] if len(state.coins) > 1 else 0
                    coins2 = state.coins[0] if len(state.coins) > 0 else 0
            
            first_player = state.current_player
            action1 = None
            action2 = None
            
            if first_player == 0:
                action1 = _get_agent_action(agent1, obs, player_position, state, engine)
                obs, _, terminated, truncated, _ = engine.step(action1)
                
                if not terminated and not truncated:
                    state = engine.get_state()
                    action2 = _get_agent_action(agent2, obs, 1 - player_position, state, engine)
                    obs, _, terminated, truncated, _ = engine.step(action2)
            else:
                action2 = _get_agent_action(agent2, obs, 1 - player_position, state, engine)
                obs, _, terminated, truncated, _ = engine.step(action2)
                
                if not terminated and not truncated:
                    state = engine.get_state()
                    action1 = _get_agent_action(agent1, obs, player_position, state, engine)
                    obs, _, terminated, truncated, _ = engine.step(action1)
            
            # 记录回合信息
            agent1_is_player0 = (first_player == 0)
            player0_ops = _action_to_string(action1, player_position) if (agent1_is_player0 and action1 is not None) else (_action_to_string(action2, 1 - player_position) if action2 is not None else "")
            player1_ops = _action_to_string(action2, 1 - player_position) if (agent1_is_player0 and action2 is not None) else (_action_to_string(action1, player_position) if action1 is not None else "")
            
            sub_logger.log_round({
                'episode': battle_identifier,
                'round_num': total_rounds,
                'hp1': hp1,
                'hp2': hp2,
                'coins1': coins1,
                'coins2': coins2,
                'player0_ops': player0_ops,
                'player1_ops': player1_ops,
                'first_player': first_player
            })
        
        duration = (np.datetime64('now') - start_time) / np.timedelta64(1, 's')
        
        if hp1 > hp2:
            result = "win"
        elif hp1 < hp2:
            result = "loss"
        else:
            result = "draw"
        
        sub_logger.log_battle_end({
            'episode': battle_identifier,
            'result': result,
            'total_rounds': total_rounds,
            'duration': float(duration),
            'final_hp_our': hp1,
            'final_hp_enemy': hp2,
            'final_coins_our': coins1,
            'final_coins_enemy': coins2
        })
        
        engine.close()
        
        return BattleOutcome(
            episode=battle_identifier,
            opponent_id=agent2_name,
            player_position=player_position,
            result=result,
            total_rounds=total_rounds,
            duration=float(duration),
            final_hp_our=hp1,
            final_hp_enemy=hp2,
            final_coins_our=coins1,
            final_coins_enemy=coins2
        )
    
    except Exception as e:
        sub_logger.log_error(battle_identifier, e)
        raise
```

#### 3.4.2 修改 `BaselineBattleRunner.evaluate_parallel` 方法

```python
# 修改 ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py

def evaluate_parallel(self, agent1, agent2, episode: int, n_battles: int = 2,
                     parallel_workers: int = 2) -> List[BattleOutcome]:
    """
    使用多进程并行执行对战
    
    Args:
        agent1: PPO Agent实例
        agent2: Baseline Agent实例
        episode: 训练episode编号
        n_battles: 对战次数（通常为2，先手后手各一次）
        parallel_workers: 并行进程数
    
    Returns:
        BattleOutcome列表
    """
    import concurrent.futures
    from .battle_worker import run_battle_worker
    
    tasks = []
    for player_position in range(min(n_battles, 2)):
        position_str = "first" if player_position == 0 else "second"
        battle_identifier = f"{self.agent1_name}_vs_{self.agent2_name}_episode_{episode}_{position_str}"
        
        tasks.append({
            'agent1': agent1,
            'agent2': agent2,
            'episode': episode,
            'battle_identifier': battle_identifier,
            'player_position': player_position,
            'agent1_name': self.agent1_name,
            'agent2_name': self.agent2_name,
            'log_level': self.logger.log_level.value,
            'log_dir': self.logger.log_dir,
            'device': self.device
        })
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(run_battle_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results
```

---

## 四、改造步骤与优先级

| 优先级 | 改造项 | 说明 | 风险 |
|--------|--------|------|------|
| **P0** | 方案二：修改`_action_to_string` | 修复日志显示错误，低风险 | 低 |
| **P0** | 方案二：修改`_operations_to_action` | 修复坐标系统不匹配，低风险 | 低 |
| **P1** | 方案三：重构`_get_agent_action` | 简化逻辑，依赖SDK验证 | 中 |
| **P1** | 方案四：提取`battle_worker.py` | 消除代码重复 | 中 |
| **P2** | 方案一：创建`SDKInjector` | 统一SDK注入，需充分测试 | 高 |

---

## 五、验证计划

### 5.1 单元测试

| 测试场景 | 测试内容 | 预期结果 |
|---------|---------|---------|
| 动作字符串转换 | 测试玩家0和玩家1的build_tower动作 | 显示正确的坐标位置 |
| 操作转换 | 测试不同玩家位置的操作转换 | 返回正确的动作ID |
| SDK注入 | 测试baseline agent使用项目SDK | 正确加载项目SDK模块 |
| 异常处理 | 测试agent调用异常 | 异常被捕获并记录，向上抛出 |

### 5.2 集成测试

| 测试场景 | 测试内容 | 预期结果 |
|---------|---------|---------|
| 完整对战 | 执行完整的baseline对战 | 游戏正常进行200+回合 |
| 日志验证 | 检查对战日志 | 显示正确的玩家位置和动作 |
| 并行执行 | 执行并行对战 | 正确处理多个对战任务 |

---

## 六、代码安全性

根据设计原则，改造方案遵循以下安全规范：

1. **异常处理**：所有外部调用都有try-except包裹，异常被记录后向上抛出
2. **日志规范**：使用loguru记录日志，不使用print
3. **敏感信息**：不记录敏感配置信息
4. **模块隔离**：SDK模块在使用完毕后正确清理，避免污染全局命名空间

---

## 七、文件修改清单

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `ppo/src/ppo_antwar/league/sdk_injector.py` | 新增 | 统一SDK注入器 |
| `ppo/src/ppo_antwar/league/battle_worker.py` | 新增 | 子进程对战执行器 |
| `ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py` | 修改 | 重构`_get_agent_action`、`_action_to_string`、`_operations_to_action` |
| `ppo/src/ppo_antwar/league/baseline_agent.py` | 修改 | 使用统一SDK注入 |
| `docs/code/baseline_battle_issues.md` | 更新 | 更新问题状态 |