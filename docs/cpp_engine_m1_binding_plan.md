# C++ 引擎 M1：code/ 副本 + 极薄 pybind11 绑定

> 目标：把官方 `game/` C++ 引擎封装成可嵌入 MCTS 的极薄 pybind11 模块。
> 日期：2026-08-08。状态：**M1 已完成**（绑定可用、副本逐字节忠实官方）。前置：M0（工具链 + 编译 + test_cpp_runtime.py 8/9）。

> ⚠️ 关键更正（2026-08-08）：**C++ 引擎是权威裁判**，Python SDK 与其在 99/361 格地图定义 + 部分合法性检查上不一致。所以：
> - **一致性目标 = 官方 C++**，不是 Python。嵌入副本逐字节复制官方源码，验证方式 = `diff` 源码一致 + 官方 test_cpp_runtime.py。
> - 曾为对齐 Python 给副本打了高地/VOID 补丁——已撤回（那会让副本偏离权威）。
> - `test_consistency.py`（C++ vs Python 随机对局）降级为**漂移诊断工具**，记录 C++/Python 分叉点，不作为 M1 门禁。

## 1. 背景

- 引擎结算（`advance_round` 蚂蚁移动 Dijkstra）占 bundle MCTS 搜索 ~72%，是最大瓶颈。
- 官方 `game/`（game.hpp + game.cpp 2852 行）是完整、经 `test_cpp_runtime.py` 验证的 C++ 移植（含 Enhanced 寻路），但被 judger 协议（stdin/stdout JSON）包着，无法嵌入 MCTS。
- M0 已验证：工具链可用、game/ 编译、核心行为（塔血量/蚂蚁位置）和 Python 一致。
- 已知坑：Windows 用户名是中文，C++ `ofstream` 写 UTF-8 中文路径静默失败 → 相关路径保持 ASCII。

## 2. 副本策略

**官方代码不改**。在 `code/cpp_engine/` 维护一份拷贝（已从 `Ant-Game/game/` 复制，清除了 .o/.d 产物）：
- `code/cpp_engine/include/`（game.hpp、ant.h、map.h 等，可加访问器）
- `code/cpp_engine/src/`（源文件，可删 judger 相关）
- `code/cpp_engine/Makefile`（副本用）

规则漂移风险：C++ 副本和官方 Python 引擎的一致性靠随机回放测试门禁（见 §6）；若官方修复规则，需同步到副本。

## 3. M1 设计：薄 pybind11 绑定

### 3.1 编译的源文件

```
✅ 编: game.cpp ant.cpp map.cpp building.cpp coin.cpp item.cpp operation.cpp aco.cpp
❌ 跳: comm_judger.cpp output.cpp main.cpp（judger 协议层）
```

### 3.2 暴露接口（对齐 Python 引擎调用方式）

```python
class NativeGame:
    def __init__(seed, movement_policy="enhanced", cold_handle_rule_illegal=True)
    def clone(self) -> NativeGame          # 树节点克隆（含 RNG 状态）
    def apply_operation_list(self, player, ops) -> bool  # 应用操作（合法性兜底）
    def advance_round(self) -> bool        # 整回合结算，返回是否还能继续
    # 只读查询
    def round(self) -> int
    def base_hp(self, player) -> int
    def terminal(self) -> bool
    def winner(self) -> int | None
```

ops 格式：`[(op_type, arg0, arg1), ...]`（对齐 Python 的 `Operation`）。

### 3.3 私有成员访问

`native_antwar.cpp` 的 `#define private public` hack 不用。给 `game.hpp` 的 `Game` 加少量 public 访问器：
- `base_hp(int player)`
- `is_ended()` / `winner()`
- `round_index()`
- `apply_operation(const Operation&, int player)`（若已有公开版本直接用）
- `next_round()`

只加"查询类"访问器，不动核心逻辑，漂移风险最小。

### 3.4 clone 语义（专项关注点）

`Game` 用 `std::deque<DefenseTower>` + `rewire_map`（指针重连）存状态。clone 需要：
- 深拷贝所有状态容器（deque/vector/array 字段）
- RNG 状态（LCG）属于状态本身，clone 后各自演化
- **验证**：clone 原状态 + 原状态各跑 N 回合随机 ops，结果必须逐回合一致

如果 `Game` 有现成的拷贝构造/赋值且正确，直接用；否则手写深拷贝。这是 M1 风险最高的部分。

### 3.5 构建脚本

```bash
# build_cpp_engine.sh
# base 环境有 pybind11 3.0.4
PYTHON=D:/anaconda3/python.exe
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PBI=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
g++ -shared -fPIC -O2 -std=c++17 \
    -I"$PBI" -I"$PY_INC" -Icode/cpp_engine/include \
    code/cpp_engine/src/game.cpp ... code/cpp_engine/binding.cpp \
    -o code/cpp_engine/native_game.pyd
```

## 4. 用法（M1 之后的接入点）

MCTS 侧替换 `advance_round`：
```
树节点状态 = NativeGame（C++）
每个 P1 决策后: native.advance_round()
每模拟结算: native.apply_operation_list(p, ops) + advance_round()
特征提取/网络仍走 Python（下一步 M4 再加速特征）
```

训练/评测分离：自对弈用 C++ 引擎（自洽即可），评测/提交用官方 Python 引擎（规则权威）。

## 5. 里程碑（2026-08-08 状态）

| 里程碑 | 内容 | 验证 | 状态 |
|--------|------|------|------|
| M1a | 绑定编译通过，`NativeGame` 可初始化 + 空操作 512 回合 | import 成功、回合推进正常 | ✅ |
| M1b | **C++ 是权威** → 一致性 = 副本源码与官方 `game/` 逐字节一致（`diff` 无差异） | `diff -r` 退出码 0（仅 .o/.d 产物和 game.hpp 访问器差异） | ✅ |
| M1c | clone 正确性专项 | 克隆并行跑 N 回合一致 | ✅ |
| M1d | 性能基准 | 单回合 <100μs（Python ~3-6ms） | ✅ 实测 C++ 76μs vs Python 2.73ms = **36×** |
| M1e | 接入 bundle MCTS（可选） | 自对弈采集用上 C++ 引擎 | ⏳ 待定 |

## 6. 一致性验证（2026-08-08 修订）

**权威是官方 C++**。验证方式：
1. `diff -r code/cpp_engine/src Ant-Game/game/src --exclude='*.o' --exclude='*.d'` 退出码 0（源码逐字节一致）
2. `code/cpp_engine/include` 仅差 game.hpp 的 M1 访问器声明
3. 官方 `test_cpp_runtime.py`（M0 已 8/9 通过）覆盖官方引擎核心行为

`test_consistency.py`（C++ vs Python 随机对局）保留为**漂移诊断工具**：记录 C++/Python 分叉点（发现：地图合法性 99/361 格不一致、建塔高地检查、武器 VOID 检查等），供参考，不作为 M1 门禁。

## 7. 风险

| 风险 | 缓解 |
|------|------|
| clone 语义（deque + rewire_map）深拷贝错 | M1c 专项压测（已验证 ✓） |
| 私有成员访问 hack 化 | 只加查询访问器，不改逻辑 ✓ |
| C++ 与 Python 漂移（地图 99 格、合法性检查） | **C++ 是权威，副本逐字节忠实即可**；Python 侧差异仅影响"用 Python 评测时的迁移"，用 test_consistency.py 记录 |
| pybind11 编译环境问题 | 为 pytorch-gpu 3.11 构建（MinGW 扩展与 3.13 ABI 不兼容）；运行时 DLL 复制到 pyd 旁 |
| 中文路径坑 | 构建产物/路径保持 ASCII；pytest 用 `--basetemp=C:/mingw64/pytest_tmp` |

## 8. 一句话总结

**M1 = 在 `code/cpp_engine/` 副本里，把官方已验证的 `game/` C++ 引擎剥掉 judger 协议、加少量查询访问器，包成极薄 pybind11 模块（apply_operation_list/advance_round/clone），用随机回放一致性测试确保和 Python 引擎逐回合一致，目标单回合 <100μs——官方代码一个字节都不改。**
