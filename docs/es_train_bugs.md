# es_train.py Bug 修复记录

## 验证结果

| Bug | 类别 | 处理 |
|-----|------|------|
| Bug 1: GameState import | ❌ 非 bug | `SDK.backend` 确实导出了 `GameState`，无需修改 |
| Bug 2: Worker 路径缺少 code/ | ✅ 真 bug | 已修复 |
| Bug 3: 手动循环无 terminal 检查 | ✅ 真 bug | 已改为 `resolve_turn` |
| Bug 4: 缺少 cold_handle_rule_illegal | ✅ 真 bug | 训练时设为 True |
| Bug 5: 多进程串行等待 | ✅ 真 bug | 改为一次提交所有任务 |
| `evaluate()` 死代码 | ✅ 真问题 | 已删除（内联到 step） |

## 修复详情

### Bug 1 — 非 bug
`SDK/backend/__init__.py:10` 有 `from SDK.backend.engine import GameState`，所以 `from SDK.backend import GameState` 是正确的。不做修改。

### Bug 2 — Worker 路径缺 code/
`_eval_worker()` 只把 `Ant-Game` 加入 `sys.path`，但 `from my_ai.network import create_model` 需要 `code/`。

**修复**：在 worker 中同时加入两个路径。

### Bug 3 — 手动循环无 terminal 检查
[旧代码]
```python
ops0 = agent._choose_operations(state, 0)
state.apply_operation_list(0, ops0)  # 可能触发 terminal
ops1 = opp.choose_operations(state, 1)  # state 已 terminal
state.apply_operation_list(1, ops1)
state.advance_round()
```

**修复**：改用 `state.resolve_turn(ops0, ops1)`，它在内部处理顺序执行、terminal 检查和回合推进。

### Bug 4 — 缺少 cold_handle_rule_illegal=True
默认 `cold_handle_rule_illegal=False`，神经网络输出非法操作直接判负，训练无法收集有效样本。

**修复**：`GameState.initial(seed=seed, cold_handle_rule_illegal=True)`

### Bug 5 — 多进程串行等待
[旧代码]
```python
for idx in range(self.population_size):
    for s in ind_seeds:
        results.append(pool.apply_async(...))
    match_scores = [r.get() for r in results]  # 等完一个才提交下一个
```

同一时刻只有 `games_per_individual` 个任务并行。

**修复**：一次提交所有 `pop_size * games_per_individual` 个任务，统一等待。

### 死代码
`evaluate()` 方法从未被调用，`step()` 中有重复的内联逻辑。已删除。
