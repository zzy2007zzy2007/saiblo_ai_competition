# Baseline 对战模块代码审查报告

## 审查概要
- 审查文件数: 10（utils/__init__.py 为空文件）
- 发现问题数: 8
- CRITICAL: 1 | HIGH: 2 | MEDIUM: 3 | LOW: 2

## 问题清单

### Issue 1: MediumRuleAI._upgraded_towers 阻断 L3 升级路径
- **严重程度**: CRITICAL
- **文件**: `battle/rule_based_agents.py`
- **行号**: 376, 323, 455
- **问题描述**: `_upgraded_towers` 集合同时用于 L2 和 L3 升级追踪，导致一旦塔从 BASIC 升级为 HEAVY/QUICK（L2），其 `tower_id` 被加入 `_upgraded_towers`，该塔将永远无法再升级到 L3。

  具体流程：
  1. `_try_upgrade_tower_l2`（行 376）将升级后的 `tower_id` 加入 `_upgraded_towers`
  2. `_select_l2_tower`（行 323）通过 `t.tower_id not in self._upgraded_towers` 过滤掉已升级的塔
  3. `_try_upgrade_tower_l3`（行 455）额外检查 `tower.tower_id in self._upgraded_towers`

  由于 SDK 中 `tower.upgrade()` 不会改变 `tower_id`，上述逻辑导致 BASIC→HEAVY→HEAVY_PLUS 完整升级链完全断裂。游戏内不可能存在开局就是 L2 的塔，因此 `_try_upgrade_tower_l3` 实际上永远不会成功返回任何 L3 升级操作。MediumRuleAI 永远无法建造 HEAVY_PLUS 或 SNIPER 等高级塔。

  此外，`reset_for_episode`（行 225-227）虽能清空 `_upgraded_towers`，但在对战流程中从未被调用（BattleSimulator 通过序列化/反序列化管理 agent 生命周期，不调用此方法）。
- **修复建议**: 使用两个独立的集合分别追踪 L2 和 L3 升级，或仅追踪当前回合内的升级（每回合清空）。推荐方案：
  ```python
  # 将 _upgraded_towers 改为两个独立集合
  self._l2_upgraded_towers: set = set()  # 仅防止同回合重复 L2 升级
  self._l3_upgraded_towers: set = set()  # 仅防止同回合重复 L3 升级
  ```
  或者更简单地，由于每回合只执行一个操作，`_upgraded_towers` 实际上不需要跨回合保留，可以在每回合开始时清空。

---

### Issue 2: MediumRuleAI._try_upgrade_tower_l3 使用错误的金币阈值（300 vs 实际 200）
- **严重程度**: HIGH
- **文件**: `battle/rule_based_agents.py`
- **行号**: 450
- **问题描述**: `_try_upgrade_tower_l3` 中硬编码了 `if self._get_coins(state) < 300:` 作为金币门槛，但：
  1. 类中已定义常量 `_TOWER_L3_COST = 200`（行 201），该常量未被使用
  2. SDK 的 `upgrade_tower_cost()` 函数明确返回 L3 升级费用为 200 金币（HEAVY_PLUS、SNIPER 等）

  这意味着 MediumRuleAI 会额外等待 100 金币才尝试 L3 升级，不必要地延迟了高级塔的建造时机。结合 Issue 1（L3 升级永远无法执行），此问题当前被掩盖，但修复 Issue 1 后将直接暴露。
- **修复建议**: 将行 450 改为 `if self._get_coins(state) < self._TOWER_L3_COST:`，或直接使用 SDK 的 `state.upgrade_tower_cost(target_type)` 动态获取费用。

---

### Issue 3: BattleSimulator 声明支持顺序降级但未实现
- **严重程度**: HIGH
- **文件**: `battle/battle_simulator.py`
- **行号**: 30-53
- **问题描述**: `run_battles` 方法的文档注释（行 30）明确声称"优先多进程并行，失败时降级为顺序执行"，但代码中仅调用了 `_run_battles_parallel`。当所有并行对战失败时（行 49-52），仅打印一条 warning 日志，没有任何降级到顺序执行的逻辑。如果 ProcessPoolExecutor 因环境问题（如内存不足、序列化失败）完全不可用，整个对战评估将返回全错误结果而无法恢复。
- **修复建议**: 实现顺序执行降级路径，或在所有并行对战失败时回退到单进程执行：
  ```python
  if results and all(r.get("error") for r in results):
      logger.warning("All parallel battles failed, falling back to sequential execution")
      results = self._run_battles_sequential(agent1_bytes, agent2_bytes, total_tasks, max_rounds)
  ```

---

### Issue 4: BattleCoordinator 错误结果字典缺少关键字段
- **严重程度**: MEDIUM
- **文件**: `battle/battle_coordinator.py`
- **行号**: 88-93
- **问题描述**: 当某个 baseline agent 评估失败时，异常处理中返回的结果字典仅包含 `error`、`agent1_wins`、`agent2_wins`、`draws` 四个字段，缺少正常聚合结果中的 `total_battles`、`completed_battles`、`error_battles`、`avg_rounds`、`total_duration` 等字段。这会导致下游代码（如训练回调、统计汇总）使用 `results.get('total_battles', 0)` 时得到 0 而非预期的 `n_battles * 2`，可能影响胜率计算和训练日志的准确性。
- **修复建议**: 补全错误结果字典，使其与 `aggregate_results` 的返回格式一致：
  ```python
  results[baseline_name] = {
      "error": str(e),
      "total_battles": n_battles * 2,
      "completed_battles": 0,
      "error_battles": n_battles * 2,
      "agent1_wins": 0,
      "agent2_wins": 0,
      "draws": 0,
      "avg_rounds": 0.0,
      "total_duration": 0.0,
      "agent1_first_move_wins": 0,
      "agent1_first_move_battles": 0,
      "agent1_second_move_wins": 0,
      "agent1_second_move_battles": 0,
  }
  ```

---

### Issue 5: _execute_battle 异常时丢失已积累的回合数据
- **严重程度**: MEDIUM
- **文件**: `battle/battle_simulator.py`
- **行号**: 335-341
- **问题描述**: 当对战某回合抛出异常时，`_execute_battle` 立即返回错误结果，但不包含已成功执行的 `rounds_data`。这导致：1) 无法通过日志分析对战崩溃前的状态变化；2) `_write_round_logs` 无法获取崩溃前的回合数据用于调试；3) 已执行的回合数据被完全丢弃。
- **修复建议**: 在错误返回字典中包含已积累的回合数据：
  ```python
  return {
      "result": "error",
      "error": f"Battle round {round_count} failed: {e}",
      "first_player": first_player,
      "total_rounds": round_count,
      "rounds": rounds_data,  # 保留已执行的回合数据
  }
  ```

---

### Issue 6: opponent_agent.py 使用不安全的 torch.load
- **严重程度**: MEDIUM
- **文件**: `battle/opponent_agent.py`
- **行号**: 27
- **问题描述**: `torch.load(checkpoint_path, map_location=device)` 未指定 `weights_only=True`。PyTorch 使用 pickle 反序列化，加载不受信任的检查点文件可能导致任意代码执行。虽然此代码用于加载自己训练的模型，但在对抗性环境中（如加载第三方提供的 baseline 模型），这是一个安全隐患。PyTorch >= 2.0 已默认 `weights_only=True`，但显式指定更安全。
- **修复建议**:
  ```python
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
  ```
  或如需加载非张量数据：
  ```python
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  ```
  并添加安全警告注释。

---

### Issue 7: 并行对战结果乱序导致 battle_idx 不准确
- **严重程度**: LOW
- **文件**: `battle/battle_simulator.py` + `battle/battle_coordinator.py`
- **行号**: battle_simulator.py 行 81（as_completed）; battle_coordinator.py 行 108（enumerate）
- **问题描述**: `_run_battles_parallel` 使用 `as_completed` 收集结果，返回的列表顺序与提交顺序无关。`battle_coordinator.py` 的 `_write_round_logs` 使用 `enumerate(battle_results)` 生成 `battle_idx`，但此 `battle_idx` 不反映实际的对战序号，也不与 `first_player` 顺序对应。这导致回合日志中 `battle_idx=0` 可能是 agent1 后手的对战，而非先手，影响日志可读性和调试体验。
- **修复建议**: 在 `run_single_battle_process` 返回结果中添加任务序号，或在收集结果时按原始提交顺序排序：
  ```python
  # 方案1：提交时记录序号
  futures = {}
  for i in range(total_tasks):
      future = executor.submit(...)
      futures[future] = i

  # 收集时按序号排序
  results = [None] * total_tasks
  for future in as_completed(futures):
      idx = futures[future]
      results[idx] = future.result(...)
  ```

---

### Issue 8: n_battles=0 被 or 运算符错误覆盖
- **严重程度**: LOW
- **文件**: `battle/battle_coordinator.py`
- **行号**: 51
- **问题描述**: `n_battles = n_battles or getattr(self.config, "n_battles", 8)` 使用 `or` 运算符，当 `n_battles=0` 时会因 0 是 falsy 值而被覆盖为默认值 8。虽然 `n_battles=0` 在业务上不太合理，但如果调用方传入 0 期望执行 0 场对战（如快速验证），此行为会导致意外执行 8 场对战。正确做法是使用 `if n_battles is None` 判断。
- **修复建议**:
  ```python
  n_battles = n_battles if n_battles is not None else getattr(self.config, "n_battles", 8)
  ```
