# Baseline Battle 统一SDK改造设计方案

## 一、需求分析

### 1.1 业务背景

根据设计原则 `/docs/design/principle.txt` 的核心要求：

| 原则编号 | 原则内容 | 对本方案的约束 |
|---------|---------|---------------|
| L2 | 最大限度利用Ant-Game/SDK，避免重复实现游戏规则 | 动作验证必须依赖SDK |
| L3 | 不允许修改Ant-Game/SDK，只能修改ppo/src/ppo_antwar | 所有改造仅限于ppo层 |
| L5 | 错误异常尽早暴露，不设计容错机制 | 异常捕获后记录并向上抛出 |
| L7 | 异常应合理捕获并记录，抛向上层 | 统一异常处理策略 |

### 1.2 核心问题分析

基于对 `baseline_battle_runner.py` 的深度分析，当前架构存在以下核心问题：

#### 问题1：多SDK版本冲突

```
当前状态：
┌────────────────────────────────────────────────────────────────┐
│                    BaselineBattleRunner                        │
│  ├── _call_genagent_choose_bundle()                           │
│  │   └── sys.path = [baseline_dir, sdk_dir]  ← 切换到A的SDK   │
│  └── _call_genagent_choose_operations()                       │
│      └── sys.path = [baseline_dir, sdk_dir]  ← 切换到B的SDK   │
└────────────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌────────────────┐    ┌────────────────┐
│ baseline SDK A │    │ baseline SDK B │
└────────────────┘    └────────────────┘
```

**问题表现**：
- 每个baseline agent携带独立的SDK副本（约10+个版本）
- sys.path频繁切换导致模块缓存污染
- 不同SDK版本的`PythonBackendState`对象不兼容

#### 问题2：规则重复实现

| 重复位置 | 重复内容 | 违反原则 |
|---------|---------|---------|
| `_action_to_string` | 硬编码`HIGHLAND_CELLS[0]` | L2（重复实现游戏规则） |
| `_operations_to_action` | 硬编码`HIGHLAND_CELLS[0]` | L2（重复实现游戏规则） |
| `_run_battle_process` | 完整重复主进程逻辑 | 代码重复 |

#### 问题3：验证逻辑分散

```
当前验证流程：
agent.choose_operations() → 本地验证(HIGHLAND_CELLS) → SDK验证(can_apply_operation) → 执行
```

验证逻辑在多个位置实现，策略不一致，难以维护。

---

## 二、架构设计

### 2.1 设计目标

| 目标 | 描述 | 衡量指标 |
|------|------|---------|
| **SDK统一** | 所有baseline agent使用项目SDK | 单一SDK来源 |
| **规则委托** | 动作验证完全依赖SDK | 移除所有硬编码规则 |
| **代码去重** | 消除重复代码 | `_run_battle_process`删除 |
| **异常可见** | 异常尽早暴露 | 统一异常处理机制 |

### 2.2 整体架构

```
改造后架构：
┌────────────────────────────────────────────────────────────────┐
│                    BaselineBattleRunner                        │
│  ├── SDKInjector.inject_project_sdk()  ← 统一SDK注入          │
│  ├── _get_agent_action()               ← 统一调用入口         │
│  └── evaluate_parallel()               ← 并行执行（调用worker）│
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                      SDKInjector                              │
│  ├── inject_project_sdk()  ← 上下文管理器，注入项目SDK        │
│  ├── _clear_sdk_modules()  ← 清理SDK模块缓存                 │
│  └── _restore_sdk_modules() ← 恢复模块缓存                   │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                     battle_worker.py                          │
│  └── run_battle_worker()  ← 子进程执行入口（复用主进程函数）   │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                     项目Ant-Game SDK                          │
│  ← 唯一权威SDK，所有agent共用                                  │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 核心组件设计

#### 2.3.1 SDKInjector（SDK注入器）

| 属性/方法 | 类型 | 职责 |
|----------|------|------|
| `PROJECT_SDK_PATH` | 类属性 | 项目SDK的绝对路径 |
| `inject_project_sdk()` | 上下文管理器 | 注入项目SDK到sys.path |
| `_get_sdk_modules()` | 私有方法 | 获取当前已加载的SDK模块 |
| `_clear_sdk_modules()` | 私有方法 | 清理SDK模块缓存 |
| `_restore_sdk_modules()` | 私有方法 | 恢复SDK模块缓存 |

**设计要点**：
1. 使用上下文管理器模式，确保资源正确释放
2. 在进入边界前保存sys.path和模块快照
3. 退出边界后完全恢复，避免跨局污染

#### 2.3.2 battle_worker（子进程执行器）

| 属性/方法 | 类型 | 职责 |
|----------|------|------|
| `run_battle_worker()` | 函数 | 子进程对战执行入口 |

**设计要点**：
1. 复用主进程的`_get_agent_action`和`_action_to_string`函数
2. 消除`_run_battle_process`中的重复代码
3. 通过参数传递配置，避免全局状态依赖

#### 2.3.3 GenAgent改造

| 修改点 | 修改内容 | 目的 |
|--------|---------|------|
| `choose_bundle()` | 使用SDKInjector注入项目SDK | 统一SDK环境 |
| `choose_operations()` | 使用SDKInjector注入项目SDK | 统一SDK环境 |

---

## 三、关键流程设计

### 3.1 Agent调用流程

```
┌────────────────────────────────────────────────────────────────┐
│                    _get_agent_action()                        │
├────────────────────────────────────────────────────────────────┤
│  1. 检查agent类型（GenAgent或普通agent）                      │
│           │                                                  │
│           ▼                                                  │
│  2. 获取SDK状态（engine.get_sdk_state()）                     │
│           │                                                  │
│           ▼                                                  │
│  3. 使用SDKInjector注入项目SDK                                │
│           │                                                  │
│           ▼                                                  │
│  4. 调用agent.choose_operations(state, player_position)       │
│           │                                                  │
│           ▼                                                  │
│  5. 使用SDK验证操作合法性（engine.can_apply_operation）        │
│           │                                                  │
│           ▼                                                  │
│  6. 转换为动作ID（_operations_to_action）                     │
│           │                                                  │
│           ▼                                                  │
│  7. 返回动作ID                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 SDK注入流程

```
┌────────────────────────────────────────────────────────────────┐
│              SDKInjector.inject_project_sdk()                 │
├────────────────────────────────────────────────────────────────┤
│  [进入上下文]                                                 │
│     │                                                        │
│     ▼                                                        │
│  1. 保存原始sys.path                                          │
│     │                                                        │
│     ▼                                                        │
│  2. 保存原始SDK模块（sys.modules中以SDK开头的模块）             │
│     │                                                        │
│     ▼                                                        │
│  3. 构建新sys.path：[项目SDK根, 项目根, 原路径]                │
│     │                                                        │
│     ▼                                                        │
│  4. 清理已加载的SDK模块（强制重新导入）                        │
│     │                                                        │
│     ▼                                                        │
│  5. yield（执行用户代码）                                     │
│     │                                                        │
│     ▼                                                        │
│  [退出上下文]                                                 │
│     │                                                        │
│     ▼                                                        │
│  6. 恢复原始sys.path                                          │
│     │                                                        │
│     ▼                                                        │
│  7. 恢复原始SDK模块                                           │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 并行执行流程

```
┌────────────────────────────────────────────────────────────────┐
│            BaselineBattleRunner.evaluate_parallel()           │
├────────────────────────────────────────────────────────────────┤
│  1. 构建任务列表（先手/后手各一个任务）                        │
│           │                                                  │
│           ▼                                                  │
│  2. 创建ProcessPoolExecutor                                  │
│           │                                                  │
│           ▼                                                  │
│  3. 提交任务到worker（battle_worker.run_battle_worker）       │
│           │                                                  │
│           ▼                                                  │
│  4. 等待所有任务完成                                          │
│           │                                                  │
│           ▼                                                  │
│  5. 收集并返回结果                                            │
└────────────────────────────────────────────────────────────────┘
         │
         ▼ (子进程)
┌────────────────────────────────────────────────────────────────┐
│              battle_worker.run_battle_worker()                │
├────────────────────────────────────────────────────────────────┤
│  1. 初始化游戏引擎                                            │
│           │                                                  │
│           ▼                                                  │
│  2. 循环执行对战（调用主进程的_get_agent_action）              │
│           │                                                  │
│           ▼                                                  │
│  3. 收集对战结果                                              │
│           │                                                  │
│           ▼                                                  │
│  4. 返回BattleOutcome                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## 四、代码实现设计

### 4.1 SDKInjector实现

**文件路径**：`ppo/src/ppo_antwar/league/sdk_injector.py`

```python
from __future__ import annotations
import sys
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any


class SDKInjector:
    """
    统一的SDK路径注入器
    
    负责将项目SDK注入到Python的模块搜索路径中，确保所有baseline agent
    使用统一的项目SDK，而非各自携带的SDK副本。
    
    设计原则：
    - 使用上下文管理器模式，确保资源正确释放
    - 在边界点保存/恢复状态，避免跨局污染
    - 支持兼容旧版agent的降级路径
    """
    
    # 项目SDK的相对路径（相对于本文件）
    _RELATIVE_SDK_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        '..', '..', '..', 'SDK'
    )
    
    @classmethod
    def _get_project_sdk_path(cls) -> str:
        """获取项目SDK的绝对路径"""
        return os.path.abspath(cls._RELATIVE_SDK_PATH)
    
    @classmethod
    @contextmanager
    def inject_project_sdk(cls, baseline_dir: Optional[str] = None):
        """
        注入项目SDK到sys.path
        
        Args:
            baseline_dir: baseline agent所在目录（用于兼容旧版agent）
        
        Yields:
            None（上下文管理器）
        
        Note:
            进入上下文时：
            1. 保存原始sys.path和SDK模块
            2. 清理SDK模块缓存
            3. 注入项目SDK
            
            退出上下文时：
            1. 恢复原始sys.path
            2. 恢复原始SDK模块
        """
        # 保存原始状态
        original_sys_path = sys.path.copy()
        original_modules = cls._get_sdk_modules()
        
        try:
            # 构建新的sys.path
            new_path = cls._build_sdk_path(baseline_dir)
            sys.path = new_path
            
            # 清理已加载的SDK模块，强制重新导入
            cls._clear_sdk_modules()
            
            yield
        finally:
            # 恢复原始状态
            sys.path = original_sys_path
            cls._restore_sdk_modules(original_modules)
    
    @classmethod
    def _build_sdk_path(cls, baseline_dir: Optional[str]) -> list:
        """
        构建包含项目SDK的sys.path
        
        路径优先级（从高到低）：
        1. 项目SDK根目录（确保SDK包可被导入）
        2. 项目根目录（确保其他依赖可被导入）
        3. baseline目录（如果提供，用于兼容旧版agent）
        4. 原始sys.path（确保系统依赖可用）
        """
        new_path = []
        
        # 添加项目SDK
        project_sdk = cls._get_project_sdk_path()
        if os.path.exists(project_sdk):
            new_path.append(project_sdk)
        
        # 添加项目根目录
        project_root = os.path.dirname(project_sdk)
        if os.path.exists(project_root):
            new_path.append(project_root)
        
        # 如果提供了baseline目录，添加到路径中（兼容旧代码）
        if baseline_dir and os.path.exists(baseline_dir):
            new_path.append(baseline_dir)
        
        # 添加原始路径（去重）
        for path in original_sys_path:
            if path not in new_path:
                new_path.append(path)
        
        return new_path
    
    @classmethod
    def _get_sdk_modules(cls) -> Dict[str, Any]:
        """
        获取当前已加载的SDK相关模块
        
        Returns:
            模块名到模块对象的字典
        """
        sdk_modules = {}
        for name, module in list(sys.modules.items()):
            if (name.startswith('SDK') or 
                (module is not None and hasattr(module, '__file__') and 'SDK' in str(module.__file__))):
                sdk_modules[name] = module
        return sdk_modules
    
    @classmethod
    def _clear_sdk_modules(cls):
        """
        清理SDK相关模块
        
        Note:
            这会强制Python在下次导入时重新加载SDK模块，
            确保使用正确的SDK版本。
        """
        modules_to_remove = [name for name in list(sys.modules.keys()) if name.startswith('SDK')]
        for name in modules_to_remove:
            del sys.modules[name]
    
    @classmethod
    def _restore_sdk_modules(cls, modules: Dict[str, Any]):
        """
        恢复SDK模块
        
        Args:
            modules: 之前保存的模块字典
        """
        sys.modules.update(modules)
```

### 4.2 GenAgent改造

**文件路径**：`ppo/src/ppo_antwar/league/baseline_agent.py`

```python
from __future__ import annotations
import cloudpickle
import os
from loguru import logger


class GenAgent:
    """
    外部生成的Agent包装器
    
    负责加载和调用外部baseline agent，确保使用统一的项目SDK。
    """
    
    def __init__(self, baseline_dir: str):
        """
        初始化GenAgent
        
        Args:
            baseline_dir: baseline agent所在目录
        """
        self.baseline_dir = baseline_dir
        self._agent = None
    
    def _get_agent(self):
        """
        获取agent实例（延迟加载）
        
        Returns:
            agent实例
        """
        if self._agent is None:
            # 查找agent文件
            agent_path = os.path.join(self.baseline_dir, 'agent.pkl')
            if not os.path.exists(agent_path):
                agent_path = os.path.join(self.baseline_dir, 'main.py')
            
            if os.path.exists(agent_path) and agent_path.endswith('.pkl'):
                with open(agent_path, 'rb') as f:
                    self._agent = cloudpickle.load(f)
            else:
                # 尝试导入main模块
                import importlib.util
                spec = importlib.util.spec_from_file_location("main", agent_path)
                if spec:
                    main_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(main_module)
                    if hasattr(main_module, 'get_agent'):
                        self._agent = main_module.get_agent()
                    elif hasattr(main_module, 'agent'):
                        self._agent = main_module.agent
        
        return self._agent
    
    def _call_agent_method(self, method_name: str, state, player_position: int):
        """
        统一调用agent方法，使用项目SDK
        
        Args:
            method_name: 方法名（choose_bundle 或 choose_operations）
            state: 游戏状态（SDK状态对象）
            player_position: 玩家位置（0=先手，1=后手）
        
        Returns:
            操作列表或None
        
        Raises:
            Exception: 调用过程中发生的任何异常
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
            
            # 调用agent方法
            method = getattr(agent, method_name, None)
            if method is None:
                logger.error(f"Agent has no method: {method_name}")
                return None
            
            try:
                result = method(sdk_state, player_position)
                # 如果返回的是ActionBundle，提取operations
                if hasattr(result, 'operations'):
                    return result.operations
                return result
            except Exception as e:
                logger.error(f"Error calling {method_name}: {e}", exc_info=True)
                raise
    
    def choose_bundle(self, state, player_position: int):
        """
        调用agent的choose_bundle方法
        
        Args:
            state: 游戏状态
            player_position: 玩家位置
        
        Returns:
            ActionBundle或操作列表
        """
        return self._call_agent_method('choose_bundle', state, player_position)
    
    def choose_operations(self, state, player_position: int):
        """
        调用agent的choose_operations方法
        
        Args:
            state: 游戏状态
            player_position: 玩家位置
        
        Returns:
            操作列表
        """
        return self._call_agent_method('choose_operations', state, player_position)
```

### 4.3 baseline_battle_runner改造

**文件路径**：`ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py`

#### 修改`_action_to_string`函数

```python
def _action_to_string(action: int, player_position: int = 0) -> str:
    """
    将动作ID转换为可读字符串（包含参数信息）
    
    Args:
        action: 动作ID
        player_position: 玩家位置（0=先手，1=后手）
    
    Returns:
        可读的动作描述字符串
    
    Note:
        根据玩家位置使用正确的高地坐标，避免硬编码HIGHLAND_CELLS[0]
    """
    from ppo_antwar.utils.action_constants import ACTION_SPACE_CONFIG, HIGHLAND_CELLS
    
    if action == 0:
        return "NO_OP"
    
    # 获取正确的高地坐标（根据玩家位置）
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
    
    # 超级武器动作处理
    from ppo_antwar.utils.action_constants import SUPER_WEAPON_POSITIONS, SuperWeaponType
    
    if ACTION_SPACE_CONFIG['use_lightning_storm']['start'] <= action < ACTION_SPACE_CONFIG['use_lightning_storm']['end']:
        skill_idx = action - ACTION_SPACE_CONFIG['use_lightning_storm']['start']
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.LIGHTNING_STORM]
        if skill_idx < len(positions):
            pos = positions[skill_idx]
            return f"USE_LIGHTNING_STORM(pos={pos})"
        else:
            return f"USE_LIGHTNING_STORM(index={skill_idx})"
    
    if ACTION_SPACE_CONFIG['use_emp_blaster']['start'] <= action < ACTION_SPACE_CONFIG['use_emp_blaster']['end']:
        skill_idx = action - ACTION_SPACE_CONFIG['use_emp_blaster']['start']
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.EMP_BLASTER]
        if skill_idx < len(positions):
            pos = positions[skill_idx]
            return f"USE_EMP_BLASTER(pos={pos})"
        else:
            return f"USE_EMP_BLASTER(index={skill_idx})"
    
    if ACTION_SPACE_CONFIG['use_deflector']['start'] <= action < ACTION_SPACE_CONFIG['use_deflector']['end']:
        skill_idx = action - ACTION_SPACE_CONFIG['use_deflector']['start']
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.DEFLECTOR]
        if skill_idx < len(positions):
            pos = positions[skill_idx]
            return f"USE_DEFLECTOR(pos={pos})"
        else:
            return f"USE_DEFLECTOR(index={skill_idx})"
    
    if ACTION_SPACE_CONFIG['use_emergency_evasion']['start'] <= action < ACTION_SPACE_CONFIG['use_emergency_evasion']['end']:
        skill_idx = action - ACTION_SPACE_CONFIG['use_emergency_evasion']['start']
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.EMERGENCY_EVASION]
        if skill_idx < len(positions):
            pos = positions[skill_idx]
            return f"USE_EMERGENCY_EVASION(pos={pos})"
        else:
            return f"USE_EMERGENCY_EVASION(index={skill_idx})"
    
    if ACTION_SPACE_CONFIG['upgrade_generation_speed']['start'] <= action < ACTION_SPACE_CONFIG['upgrade_generation_speed']['end']:
        return "UPGRADE_GENERATION_SPEED"
    
    if ACTION_SPACE_CONFIG['upgrade_generated_ant']['start'] <= action < ACTION_SPACE_CONFIG['upgrade_generated_ant']['end']:
        return "UPGRADE_GENERATED_ANT"
    
    return f"UNKNOWN_{action}"
```

#### 修改`_operations_to_action`函数

```python
def _operations_to_action(operations, player_position: int = 0) -> int:
    """
    将SDK操作列表转换为动作ID
    
    Args:
        operations: SDK操作列表
        player_position: 玩家位置（0=先手，1=后手）
    
    Returns:
        动作ID（0表示无效）
    
    Note:
        根据玩家位置使用正确的高地坐标，避免硬编码HIGHLAND_CELLS[0]
    """
    from ppo_antwar.utils.action_constants import OperationType, ACTION_SPACE_CONFIG, HIGHLAND_CELLS
    
    if not operations:
        return 0

    op = operations[0]
    
    # 获取操作类型
    if hasattr(op, 'type'):
        op_type = op.type
    elif hasattr(op, 'operation_type'):
        op_type = op.operation_type
    elif hasattr(op, 'op_type'):
        op_type = op.op_type
    else:
        logger.warning("Operation has no type attribute")
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
    
    elif op_type_int == int(OperationType.DOWNGRADE_TOWER):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        try:
            idx = highland_cells.index(pos)
            return ACTION_SPACE_CONFIG['downgrade_tower']['start'] + idx
        except ValueError:
            logger.warning(f"Downgrade tower position {pos} not in player {player_position}'s highland")
            return ACTION_SPACE_CONFIG['downgrade_tower']['start']
    
    elif op_type_int == int(OperationType.UPGRADE_GENERATION_SPEED):
        return ACTION_SPACE_CONFIG['upgrade_generation_speed']['start']
    
    elif op_type_int == int(OperationType.UPGRADE_GENERATED_ANT):
        return ACTION_SPACE_CONFIG['upgrade_generated_ant']['start']
    
    # 超级武器操作
    elif op_type_int == int(OperationType.USE_LIGHTNING_STORM):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        from ppo_antwar.utils.action_constants import SUPER_WEAPON_POSITIONS, SuperWeaponType
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.LIGHTNING_STORM]
        try:
            idx = positions.index(pos)
            return ACTION_SPACE_CONFIG['use_lightning_storm']['start'] + idx
        except ValueError:
            return ACTION_SPACE_CONFIG['use_lightning_storm']['start']
    
    elif op_type_int == int(OperationType.USE_EMP_BLASTER):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        from ppo_antwar.utils.action_constants import SUPER_WEAPON_POSITIONS, SuperWeaponType
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.EMP_BLASTER]
        try:
            idx = positions.index(pos)
            return ACTION_SPACE_CONFIG['use_emp_blaster']['start'] + idx
        except ValueError:
            return ACTION_SPACE_CONFIG['use_emp_blaster']['start']
    
    elif op_type_int == int(OperationType.USE_DEFLECTOR):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        from ppo_antwar.utils.action_constants import SUPER_WEAPON_POSITIONS, SuperWeaponType
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.DEFLECTOR]
        try:
            idx = positions.index(pos)
            return ACTION_SPACE_CONFIG['use_deflector']['start'] + idx
        except ValueError:
            return ACTION_SPACE_CONFIG['use_deflector']['start']
    
    elif op_type_int == int(OperationType.USE_EMERGENCY_EVASION):
        pos = (op.arg0, op.arg1) if hasattr(op, 'arg0') else (0, 0)
        from ppo_antwar.utils.action_constants import SUPER_WEAPON_POSITIONS, SuperWeaponType
        positions = SUPER_WEAPON_POSITIONS[SuperWeaponType.EMERGENCY_EVASION]
        try:
            idx = positions.index(pos)
            return ACTION_SPACE_CONFIG['use_emergency_evasion']['start'] + idx
        except ValueError:
            return ACTION_SPACE_CONFIG['use_emergency_evasion']['start']

    logger.debug(f"Unsupported operation type: {op_type_int}")
    return 0
```

#### 修改`_get_agent_action`函数

```python
def _get_agent_action(agent, obs: Dict, player_position: int, state, engine: Any = None) -> int:
    """
    统一调用不同类型agent的动作选择方法
    
    Args:
        agent: agent实例
        obs: 观察值字典
        player_position: 玩家位置（0=先手，1=后手）
        state: 游戏状态对象
        engine: 游戏引擎实例（用于SDK调用和验证）
    
    Returns:
        动作ID（0表示无效）
    
    Raises:
        Exception: 调用过程中发生的任何异常（记录后向上抛出）
    """
    from ppo_antwar.league.baseline_agent import GenAgent
    
    is_gen_agent = isinstance(agent, GenAgent)
    
    try:
        # GenAgent 或需要SDK状态的agent
        if is_gen_agent or (engine is not None and (hasattr(agent, 'choose_bundle') or hasattr(agent, 'choose_operations'))):
            if engine is None:
                logger.warning("Engine is None, cannot call SDK-based agent")
                return 0
            
            # 获取SDK状态
            sdk_state = engine.get_sdk_state()
            
            # 调用agent方法
            if hasattr(agent, 'choose_bundle'):
                operations = agent.choose_bundle(sdk_state, player_position)
            elif hasattr(agent, 'choose_operations'):
                operations = agent.choose_operations(sdk_state, player_position)
            else:
                logger.warning(f"Agent has no choose_bundle or choose_operations method")
                return 0
            
            # 验证操作合法性（完全依赖SDK）
            if operations:
                valid_ops = []
                for op in operations:
                    if hasattr(engine, 'can_apply_operation'):
                        try:
                            if engine.can_apply_operation(player_position, op):
                                valid_ops.append(op)
                            else:
                                logger.debug(f"SDK rejected operation: {op}")
                        except Exception as e:
                            logger.warning(f"SDK validation error for operation {op}: {e}, allowing operation")
                            valid_ops.append(op)
                    else:
                        valid_ops.append(op)
                
                if valid_ops:
                    return _operations_to_action(valid_ops, player_position)
            
            return 0
        
        # 标准处理路径（PPO agent，使用select_action）
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
        
        logger.warning(f"Agent has no valid action method: {type(agent)}")
    
    except Exception as e:
        logger.error(f"Error getting agent action: {e}", exc_info=True)
        raise
    
    return 0
```

#### 删除`_run_battle_process`函数

删除整个`_run_battle_process`函数，改用`battle_worker.py`中的`run_battle_worker`函数。

#### 修改`evaluate_parallel`方法

```python
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

### 4.4 battle_worker实现

**文件路径**：`ppo/src/ppo_antwar/league/battle_worker.py`

```python
from __future__ import annotations
import numpy as np
from loguru import logger
from typing import Dict, Any
from ppo_antwar.league.baseline_battle.baseline_battle_runner import (
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
        task: 任务字典，包含以下字段：
            - agent1: PPO Agent实例
            - agent2: Baseline Agent实例
            - episode: 训练episode编号
            - battle_identifier: 对战标识字符串
            - player_position: 玩家位置（0=先手，1=后手）
            - agent1_name: agent1名称
            - agent2_name: agent2名称
            - log_level: 日志级别（枚举值）
            - log_dir: 日志目录（可选）
            - device: 运行设备（cuda/cpu）
    
    Returns:
        BattleOutcome对象
    
    Raises:
        Exception: 对战过程中发生的任何异常
    
    Note:
        此函数设计为在子进程中运行，复用主进程的核心函数（_get_agent_action, _action_to_string）
        避免代码重复。
    """
    from enum import Enum
    
    class LogLevel(Enum):
        DEBUG = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
    
    # 提取任务参数
    agent1 = task['agent1']
    agent2 = task['agent2']
    episode = task['episode']
    battle_identifier = task.get('battle_identifier', f"{task['agent1_name']}_vs_{task['agent2_name']}_episode_{episode}")
    player_position = task['player_position']
    agent1_name = task['agent1_name']
    agent2_name = task['agent2_name']
    log_level_value = task['log_level']
    
    # 初始化日志
    log_level = LogLevel(log_level_value)
    sub_logger = SubProcessLogger(agent1_name, agent2_name, log_level)
    
    start_time = np.datetime64('now')
    sub_logger.log_battle_start(battle_identifier)
    
    try:
        # 初始化游戏引擎
        engine = AntWarBattleEngine(player_id=player_position, seed=episode)
        obs = engine.reset(player=player_position, seed=episode)
        
        total_rounds = 0
        hp1, hp2 = 50, 50
        coins1, coins2 = 0, 0
        terminated = False
        truncated = False
        
        # 主对战循环
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
            
            # 根据先手/后手顺序执行动作
            if first_player == 0:
                # 先手玩家行动（agent1）
                action1 = _get_agent_action(agent1, obs, player_position, state, engine)
                obs, _, terminated, truncated, _ = engine.step(action1)
                
                if not terminated and not truncated:
                    state = engine.get_state()
                    # 后手玩家行动（agent2）
                    action2 = _get_agent_action(agent2, obs, 1 - player_position, state, engine)
                    obs, _, terminated, truncated, _ = engine.step(action2)
            else:
                # 后手玩家行动（agent2）
                action2 = _get_agent_action(agent2, obs, 1 - player_position, state, engine)
                obs, _, terminated, truncated, _ = engine.step(action2)
                
                if not terminated and not truncated:
                    state = engine.get_state()
                    # 先手玩家行动（agent1）
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
        
        # 计算对战结果
        duration = (np.datetime64('now') - start_time) / np.timedelta64(1, 's')
        
        if hp1 > hp2:
            result = "win"
        elif hp1 < hp2:
            result = "loss"
        else:
            result = "draw"
        
        # 记录对战结束
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
        
        # 关闭引擎
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

---

## 五、验证计划

### 5.1 单元测试

| 测试模块 | 测试用例 | 预期结果 |
|---------|---------|---------|
| SDKInjector | 测试注入项目SDK后，SDK模块来自正确路径 | 导入的SDK模块路径为项目SDK |
| SDKInjector | 测试退出上下文后，sys.path恢复 | sys.path与注入前一致 |
| _action_to_string | 测试玩家0和玩家1的build_tower动作 | 显示正确的坐标位置 |
| _operations_to_action | 测试不同玩家位置的操作转换 | 返回正确的动作ID |
| GenAgent | 测试choose_operations调用 | 正确使用项目SDK |

### 5.2 集成测试

| 测试场景 | 测试内容 | 预期结果 |
|---------|---------|---------|
| 完整对战 | 执行完整的baseline对战 | 游戏正常进行200+回合 |
| 日志验证 | 检查对战日志 | 显示正确的玩家位置和动作 |
| 并行执行 | 执行并行对战 | 正确处理多个对战任务 |
| SDK隔离 | 在多进程环境中执行对战 | 无模块冲突，结果正确 |

### 5.3 回归测试

| 测试项 | 验证内容 |
|--------|---------|
| 与原代码兼容性 | 确保改造后行为与原代码一致（除修复的bug外） |
| 异常处理 | 确保异常正确捕获、记录并向上抛出 |
| 性能指标 | 确保改造后性能不低于原代码 |

---

## 六、回滚策略

### 6.1 回滚条件

| 条件 | 说明 |
|------|------|
| 单元测试失败 | 任何单元测试不通过 |
| 集成测试失败 | 对战无法正常进行 |
| 性能下降 | 对战时间增加超过20% |
| 结果不一致 | 对战结果与改造前有显著差异 |

### 6.2 回滚步骤

```
回滚流程：
1. 停止所有训练/对战任务
2. 恢复baseline_battle_runner.py到改造前状态
3. 删除新增文件（sdk_injector.py, battle_worker.py）
4. 恢复baseline_agent.py到改造前状态
5. 重新运行验证测试
6. 确认恢复成功后重启任务
```

---

## 七、文件修改清单

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `ppo/src/ppo_antwar/league/sdk_injector.py` | 新增 | 统一SDK注入器 |
| `ppo/src/ppo_antwar/league/battle_worker.py` | 新增 | 子进程对战执行器 |
| `ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py` | 修改 | 重构核心函数，删除_run_battle_process |
| `ppo/src/ppo_antwar/league/baseline_agent.py` | 修改 | GenAgent使用统一SDK注入 |
| `docs/code/baseline_battle_issues.md` | 更新 | 更新问题状态 |

---

## 八、设计原则合规性检查

| 原则 | 合规性 | 说明 |
|------|--------|------|
| L2 | ✅ | 动作验证完全依赖SDK，不重复实现游戏规则 |
| L3 | ✅ | 仅修改ppo/src/ppo_antwar代码，不修改SDK和agent |
| L5 | ✅ | 异常尽早暴露，不设计容错机制 |
| L7 | ✅ | 异常合理捕获并记录，抛向上层 |
| L11 | ✅ | 使用loguru记录日志，不使用print |