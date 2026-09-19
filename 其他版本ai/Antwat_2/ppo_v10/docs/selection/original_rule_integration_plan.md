# PPO_v10 基准筛选阶段改造方案 — 接入 original_rule（二分种子）

本方案根据 [original_rule.txt](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/docs/selection/original_rule.txt) 的规则，并复用
[ppo_v10/tools/original_rule_experiment](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/original_rule_experiment) 的实验代码，
将 PPO_v10 的"基准筛选阶段"由当前的 `BenchmarkTest`（外部 BenchmarkPool 全员打分）替换为
"种群内部互打 → 二分种子选 Top16"的方案，后续仍接入 `RoundRobin(Top24 → 8 seeds)`。

---

## 1. 用户决策摘要（已澄清）

| 关键决策点 | 结论 |
| --- | --- |
| 与现有 `BenchmarkTest` 的关系 | **完全替换** —— 不再依赖外部 `BenchmarkPool` 给全员打分 |
| 甄别阶段单 task 的对局数 | **每个 task 2 局**（先后手各一局，对齐 `original_rule.txt` 第 47 条） |
| 批量阶段单 task 的对局数 | **每个 task `2×n_battle` 局**（先后手交替） |
| max_seeds=3 后是否仍走 RoundRobin | **保留** RoundRobin(Top24 → 8 seeds) |
| fallback（`max_candidate_fails=12`） | **取消队列剩余任务**，按累积 mu 取 Top16 进 RR |

> **副作用**：替换 `BenchmarkTest` 后，`benchmark_pool` 相关的"τ_med 计算 / 池更新"流程
> （[genetic_evolution_trainer.py#L253-L379](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/trainer/genetic_evolution_trainer.py#L253-L379)）
> 失去其依赖的 `medium_rule_head_to_head` 数据，需要在本方案内一并处理（见 §6）。

---

## 2. 当前架构梳理（Phase 1 探索结论）

### 2.1 v10 现有筛选流程
- 入口：[GeneticEvolutionTrainer.train](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/trainer/genetic_evolution_trainer.py#L132-L240)
  - 阶段 1 种群生成
  - 阶段 2 `battle_selection.evaluate_and_select(population)` → `(ranked_all, seeds, elites)`
  - 阶段 3 `_update_benchmark_pool(...)` 用 BenchmarkTest 阶段保留的 MediumRuleAI head-to-head 计算 τ_med
- `BattleSelection` ([battle_selection.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/battle_selection.py)) 编排：
  1. `BenchmarkTest.evaluate(population)` 返回 `ranked_all`（全员按 mu 降序）
  2. 取 `ranked_all[:top_n]`（默认 Top24）进 `RoundRobinTournament.run(...)`
- `BenchmarkTest` ([benchmark_test.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/benchmark_test.py)) 行为：
  - 遍历 `BenchmarkPool` 的 builtin/精英 agent（NoviceAI, MediumRuleAI, ...）
  - 每个 entry 对全部个体并行打 `n_battles*2` 局 → 更新 `BattleSharedPayoff` 的 TrueSkill
  - 每个 entry 后计算 Kendall τ，τ > 阈值则早停
  - 副产物：保留 `MediumRuleAI` 的 per-individual head-to-head（供 pool_update 计算 τ_med）

### 2.2 实验代码梳理（可复用项）

[tools/original_rule_experiment](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/original_rule_experiment) 仅 3 个文件：
- [coordinator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/original_rule_experiment/coordinator.py)
  - `OriginalRuleCoordinator` —— 严格对齐 14 条规则的核心协调器（**完全可复用算法逻辑**）
  - `SimplePayoff` —— 模拟实验自用的简化 trueskill 包装（**生产环境用 BattleSharedPayoff，需要适配层**）
  - `CandidateState` 数据类（**可复用**）
- [run_experiment.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/original_rule_experiment/run_experiment.py)
  - 仅是实验入口与汇总报告；**不复用**

[tools/bisection_experiment](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/bisection_experiment) 中：
- [task_queue.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/bisection_experiment/task_queue.py)
  - `Task` / `TaskQueue`（线程安全、按优先级 pop、按 creator 取消、按 seed 取消）—— **完全可复用**
- [battle_executor.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/bisection_experiment/battle_executor.py)
  - `BattleExecutor` —— 用 `ThreadPoolExecutor` 调度，每 task 内顺序跑 `n_games`，每局 callback。
  - **算法骨架可复用**，但执行体 `simulate_battle` 是模拟函数，必须替换为生产环境的 `_execute_battle`（基于真实策略网络对战）。
  - 由于实战对战是 CPU/GPU 密集型，生产版本须改为 **`ProcessPoolExecutor + spawn`**，与现有
    [`_benchmark_battle_worker`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/benchmark_test.py#L37-L62)、
    [`_rr_battle_worker`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/round_robin.py#L35-L60) 一致。
- [individual.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/bisection_experiment/individual.py) —— 模拟个体与 `simulate_battle`，**不复用**（生产用真实 `Individual`）。
- [round_robin.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/bisection_experiment/round_robin.py) —— 实验对照基线，**不复用**（生产用现有 `RoundRobinTournament`）。

### 2.3 复用矩阵汇总

| 实验代码 | 复用方式 | 说明 |
| --- | --- | --- |
| `OriginalRuleCoordinator`（algorithm） | **整体复用** | 仅替换 `SimplePayoff` → `BattleSharedPayoff` 适配层 |
| `Task` / `TaskQueue` | **整体复用** | 直接 import，作为 selection 模块的子组件 |
| `CandidateState` | **整体复用** | 数据结构无需改动 |
| `BattleExecutor`（骨架） | **结构复用，执行体重写** | 模拟函数→真实对战；ThreadPool→ProcessPool |
| `SimplePayoff` | 仅作为适配层参考 | 生产环境用 `BattleSharedPayoff` + 旁路 `battle_count` 计数 |
| `simulate_battle`、`Individual` 模拟版 | 不复用 | 生产环境替换为真实 `Individual` + `_execute_battle` |
| `run_experiment.py` 报告 | 不复用 | 由 `GALoggingSubsystem` 接管 |
| `RoundRobin`（实验版） | 不复用 | 生产环境继续用 `RoundRobinTournament` |

---

## 3. 改造目标

将"阶段 2：个体选择"从「BenchmarkTest（外部对手）→ RoundRobin」改为
「**OriginalRuleSelector（种群内部二分种子）→ RoundRobin**」。

核心约束：
1. **接口对外保持不变**：`BattleSelection.evaluate_and_select(population, generation)` 仍返回
   `(ranked_all, seeds, elites)`，使训练主循环零感知。
2. **按规则 14 个体可重复**：候选失败后允许同一个体被重新选为新候选（见 `original_rule.txt` 注 1）。
3. **线程安全 / callback 解耦**：`BattleExecutor` 的 worker callback 与 `Coordinator` 必须有显式锁
   保护 `seeds / candidates / pair_battle_count` 等共享状态。
4. **fallback 路径**：候选累计失败 ≥ 12 → 队列 `cancel_all`，按累积 mu 取 Top16 进 RR；
   `is_fallback` 标记记录到日志。
5. **n_battle 对接**（用户决策）：
   - 甄别 task：`n_games = 2`（先后手各 1 局），每局完整 callback；M=16 表示对手数。
   - 批量 task：`n_games = 2 × config.n_battle`，"补齐"逻辑保持。

---

## 4. 文件改动清单

### 4.1 新增文件

#### A. [`ppo_v10/src/ppo_ga/selection/original_rule_selector.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/original_rule_selector.py) **(新)**

封装"种群内部二分种子选拔"作为基准筛选阶段。

```python
@dataclass
class OriginalRuleConfig:
    M: int = 16                          # 升级所需最少对手数（规则 2/3）
    max_seeds: int = 3                   # 二分种子数上限（规则 7）
    n_battle: int = 1                    # 批量阶段每对 2×n_battle 局（规则 8）
    max_candidate_fails: int = 12        # fallback 阈值（规则 14）
    max_workers: int = 25                # 并发对战数（参数表）
    discovery_n_games: int = 2           # 甄别 task 内对局数（用户决策）
    max_rounds: int = 512
    device: str = "cuda"
    hidden_dim: int = 256
    enable_auxiliary: bool = True


class OriginalRuleSelector:
    def __init__(self, config: OriginalRuleConfig, payoff: BattleSharedPayoff): ...

    def evaluate(self, population: Population, generation: int) -> List[Individual]:
        """
        执行二分种子选拔。返回按 mu 降序排列的全部个体（用于后续 RoundRobin 取 Top24）。

        副作用：
          - 更新每个 individual 的 trueskill_mu / trueskill_sigma
          - self.stats 记录 battles_total / class1/2/3 / unique_candidates / fallback
          - self.seeds 记录确认的二分种子 id
        """
```

内部职责拆分：
- **协调器**：复用实验代码 `OriginalRuleCoordinator` 的 `start / on_battle_complete /
  _check_candidate / _promote_to_seed / _fail_candidate / _pick_next_candidate /
  _submit_batch_battles / _fallback / finalize_battle_classification`，**完整迁移**。
- **Payoff 适配层**：实验代码用 `SimplePayoff`（自带 `battle_counts`）；生产用
  `BattleSharedPayoff`（无 per-player battle_count）。在 `OriginalRuleSelector` 内部维护
  `self._battle_count: Dict[str, int]`，作为 callback 副产物（避免污染 `BattleSharedPayoff`）。
  协调器通过一个轻量 wrapper 拿到 `get_battle_count(pid)`。
- **Agent 序列化 / 子进程对战**：参考
  [`_benchmark_battle_worker`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/benchmark_test.py#L37-L62)：
  预序列化所有个体的 `GAWarAgent` 一次（Top-level 用 `AgentLoader.serialize`），
  worker 反序列化后跑 N 局返回结果列表。

```text
   ┌────────────────────┐         ┌─────────────────────────┐
   │ OriginalRuleSelector│        │ _OriginalRuleCoordinator│
   │  • TaskQueue        │ submit │  • candidates / seeds   │
   │  • BattleExecutor   │ ──────▶│  • _battle_count        │
   │  • payoff (shared)  │ ◀──────│  • on_battle_complete   │
   └─────────┬──────────┘callback└─────────────────────────┘
             │
             ▼
   ProcessPoolExecutor(spawn) ──▶ _original_rule_battle_worker(...)
```

#### B. [`ppo_v10/src/ppo_ga/selection/_battle_executor.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/_battle_executor.py) **(新)**

生产版 `BattleExecutor`（与实验版同名但实现重写）：
- 输入：`TaskQueue`, `agent_bytes_provider: Callable[[ind_id], bytes]`,
  `callback`, `max_workers`, `max_rounds`。
- 用 `ProcessPoolExecutor(mp_context=spawn)` 调度 worker `_original_rule_battle_worker`。
- 主线程循环逻辑与
  [实验版 `BattleExecutor.run`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/bisection_experiment/battle_executor.py#L65-L101)
  保持一致：补 future 至 `max_workers`，等待至少一个完成再补充，直至队列空。
- 每个 task 完成后，按"先手 → 后手"顺序调用 `callback(home, away, {result_code, ...})`。
- 失败时 callback 一次平局（result_code=0）记账 `n_games` 次，确保 `_battle_count` 计数正确。

worker（top-level 函数，可序列化）：
```python
def _original_rule_battle_worker(
    a_bytes: bytes, b_bytes: bytes, n_games: int, max_rounds: int
) -> List[dict]:
    # 复用 _benchmark_battle_worker 的写法
    # 返回 [{"result": "agent1_win"|"agent2_win"|"draw"|"error", "first_player": 0|1}, ...]
```

> **复用决定**：实验版 `BattleExecutor` 使用 `ThreadPoolExecutor` 是因为
> `simulate_battle` 是纯 Python 计算；真实对战需独立进程隔离 PyTorch CUDA 上下文，
> 因此结构复用、执行层重写。同样 import 实验版的 `Task`/`TaskQueue` 不重复造轮子。

### 4.2 修改文件

#### C. [`ppo_v10/src/ppo_ga/selection/__init__.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/__init__.py)
- 导出 `OriginalRuleSelector`, `OriginalRuleConfig`。
- 移除 `BenchmarkTest`, `BenchmarkTestConfig` 的导出（替换语义）。

#### D. [`ppo_v10/src/ppo_ga/selection/battle_selection.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/battle_selection.py)
- 构造函数签名调整：把 `benchmark_config: BenchmarkTestConfig` 改为
  `original_rule_config: OriginalRuleConfig`；删除 `benchmark_pool` 参数。
- `self._benchmark = OriginalRuleSelector(original_rule_config, payoff)`。
- `evaluate_and_select` 内部调用保持原签名 `self._benchmark.evaluate(population, generation)`。
- `@property benchmark` 仍保留，但返回类型改为 `OriginalRuleSelector`，
  以便训练器读取 `self.battle_selection.benchmark.stats / .seeds / .num_agents_used`
  等字段（保持调用面稳定）。
- **保留** `RoundRobinTournament` 调用链不变。

#### E. [`ppo_v10/src/ppo_ga/selection/benchmark_test.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/benchmark_test.py)
- **删除**（功能完全替代；按 Principles 要求"不做向后兼容"）。

#### F. [`ppo_v10/src/ppo_ga/trainer/genetic_evolution_trainer.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/trainer/genetic_evolution_trainer.py)
- import 替换：`BenchmarkTestConfig` → `OriginalRuleConfig`。
- 构造 `OriginalRuleConfig`：
  ```python
  orig_cfg = OriginalRuleConfig(
      M=config["original_rule"]["M"],
      max_seeds=config["original_rule"]["max_seeds"],
      n_battle=config["original_rule"]["n_battle"],
      max_candidate_fails=config["original_rule"]["max_candidate_fails"],
      max_workers=config["original_rule"]["max_workers"],
      max_rounds=config["env"]["max_steps"],
      device=config["system"]["device"],
      hidden_dim=config["network"]["hidden_dim"],
      enable_auxiliary=config["network"].get("enable_auxiliary", True),
  )
  ```
- `BattleSelection(...)` 不再传 `benchmark_pool`。
- **删除** `_update_benchmark_pool(...)`、`_save_individual_model(...)`、
  `_compute_individual_tau(...)` 三个方法及其调用（及阶段 3 日志）；
  同时删除训练器对 `BenchmarkPool / EliteSeedPool` 中"动态加池"路径的依赖
  （保留 `EliteSeedPool` 添加 seeds 的逻辑）。
- 删除 `self.benchmark_pool` 字段及其 `save_state / load_state`：
  既然不再使用外部基准对手池，没有必要继续维护。
  > **若用户希望保留"benchmark_pool 给精英存档"功能用于其他用途**，本计划默认按规则 1
  > "完全替换"删除；如有不同意见请在审阅本文档时反馈。
- `log_generation` 调用中 `num_benchmark_agents_used` / `benchmark_kendall_tau_final`
  的来源改为 `self.battle_selection.benchmark.stats`：
  - `num_benchmark_agents_used` → 替换为 `unique_candidates`（候选个体去重数）
  - `benchmark_kendall_tau_final` → 设为 `None`（语义不复存在）
- 配置文件中 `benchmark` / `benchmark_pool` 段无人读取，保留即可被静默忽略；本计划**不要求**
  改动 yaml，但**会**新增 `original_rule` 段（见 §4.3）。

#### G. [`ppo_v10/src/ppo_ga/league/benchmark_pool.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/league/benchmark_pool.py)
- 不删除文件（避免影响 checkpoint / log 兼容），但训练器不再实例化它；视作 dead-code。
- 后续如需彻底清理，由独立 PR 处理。本计划范围内不动。

#### H. [`ppo_v10/src/ppo_ga/monitor/ga_logger.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/monitor/ga_logger.py) / `logging_subsystem.py`
- 若 `log_generation` 内部使用 `num_benchmark_agents_used / benchmark_kendall_tau_final`
  写表头/CSV，需要改名或允许 `None`；这部分在实施时按实际签名调整。

### 4.3 配置变更

[`ppo_v10/configs/ga_evo_v10.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/configs/ga_evo_v10.yaml) 新增段：

```yaml
# 二分种子（original_rule）参数
original_rule:
  M: 16                  # 升级所需最少对手数
  max_seeds: 3
  n_battle: 1            # 批量阶段：每对 2×n_battle 局
  max_candidate_fails: 12
  max_workers: 25
```

`benchmark` / `benchmark_pool` 段不再被读取，可以保留也可以删除；
推荐保留以便 git diff 清晰，由后续清理 PR 移除。同步更新
[`ga_evo_v10_test.yaml`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/configs/ga_evo_v10_test.yaml)。

---

## 5. 关键算法点（与规则 1:1 对齐）

| original_rule 规则 | 实现位置 | 备注 |
| --- | --- | --- |
| 1. 第 1 个体作为初始候选 | `OriginalRuleCoordinator.start` | 复用实验代码 |
| 2. 候选并发提交 M 个 task | `_launch_candidate` | `n_games=2`（用户决策） |
| 3. M 完成后判断 mu ∈ Top33-66 → 升级 | `_check_candidate / _promote_to_seed` | 区间从"已对战个体"中取 |
| 4. 不在区间 → fail | `_fail_candidate` | `cumulative_fails+=1` |
| 5. 已对战个体按 mu 升序中位数选新候选 | `_pick_next_candidate` | 偶数取中前/中后均可 |
| 6. 候选优先与有对战记录对手对战 | `_order_opponents` | `with_battles + without_battles` |
| 7. max_seeds=3 | 配置 | 规则 8 触发条件 |
| 8. 3 种子确定后批量提交（已对战补齐） | `_submit_batch_battles` | `target = 2*n_battle - already` |
| 9. 候选降级 / 全部种子定 → 取消"双方非种子/候选"task | `_promote_to_seed` 末尾 + `_fail_candidate` 中 | `TaskQueue.cancel_battles_without_seed` |
| 10. Executor 取 task 时检查 cancelled | `BattleExecutor._execute_task` | 复用 |
| 11. 独立 BattleExecutor 持续从队列取 | 同上 | spawn 进程池 |
| 12. 重复对战取消（计数器） | `Coordinator.battled_pairs` | 与规则 9 共用取消机制 |
| 13. callback 驱动甄别 | `Selector` 在主线程注册 callback | callback 用锁保护 |
| 14. fallback | `_fallback` → `queue.cancel_all` | `is_fallback=True` |
| 15. 累计失败 12 → fallback | 同上 | |

补充注意：
- **规则 2 的"对手顺序选择"**：实验代码已实现 `_order_opponents` 的"battle_count 降序 + 顺序"，
  生产复用即可。
- **规则 8 的"已与种子对战过 k 局，补齐 (n_battle*2 − k) 局"**：实验代码使用
  `pair_battle_count` 维护，复用。同时甄别阶段每 task 2 局，会让 `pair_battle_count` 一次 +2，
  与 `target = 2*n_battle = 2`（n_battle=1 默认）刚好吻合 → 多数对子直接 skip，与规则 8 一致。
- **callback 线程安全**：实验代码运行在 ThreadPool 内，本身就需要锁；生产 ProcessPool 主线程
  以单线程方式收集 future 结果，callback 在主线程同步执行，**无需额外锁**。这与
  [现有 BenchmarkTest as_completed 主线程串行 callback](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/benchmark_test.py#L194-L226) 完全同构。

---

## 6. 副作用：BenchmarkPool 更新逻辑的处置

按用户决策"完全替换 BenchmarkTest"，原来的 [`_update_benchmark_pool`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/trainer/genetic_evolution_trainer.py#L253-L379) 失去依据：
- 它依赖 `benchmark.medium_rule_head_to_head`（替换后不存在）
- 它依赖 `benchmark_pool` 实例（不再实例化）

处置方案：**整体删除该方法及其调用**（参见 §4.2 F）。
本质上，benchmark_pool 的"动态加池"是为外部基准 agent 池服务的，与种群内部互打互斥。
若未来需要重新引入，可作为独立 feature 再设计。

---

## 7. Stats / 日志输出

`OriginalRuleSelector` 暴露的统计字段（与实验脚本一致）：
- `battles_total / class1 / class2 / class3 / effective`
- `unique_candidates`（去重的候选个体数）
- `candidates_promoted / candidates_failed`
- `seeds`（确认的二分种子 id 列表）
- `is_fallback`

`logging_subsystem.log_generation` 的形参兼容性：
- `num_benchmark_agents_used` → 改为接收 `unique_candidates` 数值（语义自然过渡）
- `benchmark_kendall_tau_final` → 改为可选；传入 `None` 时记录为 `null`

---

## 8. 验证步骤

1. **单元 / smoke 测试**
   - 在 `ppo_v10/tools/` 下新增（或扩展现有）一个 dry-run 脚本，使用 ga_evo_v10_test.yaml
     运行 1~2 代，确认：
     - `OriginalRuleSelector.stats.seeds` 长度 ≤ 3
     - `RoundRobin` 正常拿到 `ranked_all[:24]`
     - 训练循环不抛异常，checkpoint 与 log 落盘正常
2. **算法一致性验证**
   - 与 [`tools/original_rule_experiment/run_experiment.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/tools/original_rule_experiment/run_experiment.py) 同种子、同种群规模下，
     生产实现的 `battles_total` / `unique_candidates` / `top16` 与实验代码差距 ≤ 5%
     （仅作回归参考；真实对战与模拟对战无法严格相等）。
3. **fallback 验证**
   - 临时把 `max_candidate_fails` 调成 2 跑一次，确认 `is_fallback=True`，且 `RoundRobin`
     仍能基于 mu Top16 正常进行。
4. **lint / typecheck**
   - 项目根目录执行 `ruff check ppo_v10/src/ppo_ga/selection`（若有配置）；
     若仓库无 lint/typecheck 命令，跳过。

---

## 9. 风险与决策记录

| 风险 / 决策 | 说明 |
| --- | --- |
| 进程池 vs 线程池 | 真实对战需要独立 PyTorch 上下文，必须 ProcessPool；增加 spawn 开销，但与现有 BenchmarkTest 一致。 |
| `pair_battle_count` 与 `BattleSharedPayoff.games` 双计数 | 接受冗余；前者只服务于规则 8 的"补齐"判断，后者是 trueskill 的全局账。 |
| BenchmarkPool 留为 dead-code | 不删除以减小本次 PR diff，由独立清理 PR 处置。 |
| `fallback` 后无 RR 输入怎么办 | fallback 取消队列后仍按 mu Top16 进 RR，与现有逻辑一致。 |
| 日志字段重命名 | `num_benchmark_agents_used` → `unique_candidates` 在 log/csv 列名上有变化，需评审下游分析脚本。 |

---

## 10. 实施顺序（建议）

1. 新增 [`_battle_executor.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/_battle_executor.py)（生产版 worker + queue runner）。
2. 新增 [`original_rule_selector.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/original_rule_selector.py)（迁移 Coordinator + payoff 适配 + Selector 入口）。
3. 改 [`battle_selection.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/battle_selection.py) + [`__init__.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/__init__.py)。
4. 改 [`genetic_evolution_trainer.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/trainer/genetic_evolution_trainer.py)（删 `_update_benchmark_pool` 等 3 个方法及其调用，重写 stats 字段）。
5. 更新 [`logging_subsystem.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/trainer/logging_subsystem.py) 的字段透传。
6. 删除 [`benchmark_test.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v10/src/ppo_ga/selection/benchmark_test.py)。
7. 更新 yaml（`original_rule` 段）。
8. 跑 smoke test → 跑 ga_evo_v10_test 1~2 代 → 提交。
