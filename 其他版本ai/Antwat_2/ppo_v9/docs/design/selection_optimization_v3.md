# 两阶段对战评估优化方案 v3（全圆桌精排版）

> 版本: v3.0
> 日期: 2026-06-12
> 状态: 待实施
> 前身: [v2 精简版](./selection_optimization_v2.md)（精排 24×5×6=720 局）

## 1. 设计动机

### 1.1 v2 方案的瓶颈

v2 将 selection 对局从 2400 降至 1040 局，但精排仍占 69%（720 局）。这 720 局是：

```
24 人 × 5 参照物 × 6 局/对 = 720 局
```

核心问题是**信息效率低**——所有比较通过 5 个参照物间接传递（星形拓扑），24 个个体的两两关系完全依赖参照物"翻译"。

### 1.2 v3 的核心思路

**放弃间接比较，改用直接对战。** 让 24 个晋级个体两两对战（全圆桌），每对只需极少对局即可获得"谁更强"的直接信号。同时保留 2 个锚定参照物维持跨代可比性。

**为什么全圆桌更高效**：

- 星形拓扑（参照物）：120 条边（24×5），每条边 6 局 → 信息通过 5 个节点转述
- 完全图（全圆桌）：276 条边（C(24,2)），每条边 2 局 → 每对直接比较

完全图的边数虽然多（276 vs 120），但**每条边的信息密度远高于星形边**——A 和 B 直接对战一次，胜过 A 和 B 分别跟同一个参照物各打 3 次的间接推断。而且选手互相对战天然在 p≈0.5 区域（最难区分也最关键），效率最高。

---

## 2. 方案设计

### 2.1 整体架构

```
每代 Selection
│
├─ 阶段一：粗排（Course Ranking）
│  ├─ 80 个体
│  ├─ 1 个低门槛参照物（CourseReferencePool）
│  ├─ n_battles=2（每对 4 局，先后手各 2）
│  ├─ 对局数：80 × 1 × 4 = 320 局
│  └─ 按胜率排序，取前 24 个进入精排
│
├─ 阶段二：精排（Fine Ranking）—— 全圆桌 + 锚定参照物
│  ├─ 全圆桌子阶段：24 人两两对战
│  │  ├─ C(24,2) = 276 对
│  │  ├─ 每对 n_battles=1（2 局，先后手各 1）
│  │  ├─ 对局数：276 × 2 = 552 局
│  │  └─ 目的：代内精确排序
│  │
│  ├─ 锚定子阶段：24 人 vs 2 个固定参照物
│  │  ├─ 24 × 2 参照物 × 2 局/对 = 96 局
│  │  └─ 目的：跨代 ELO 锚定
│  │
│  └─ 精排合计：552 + 96 = 648 局
│
└─ 总对局数：320 + 648 = 968 局
```

### 2.2 对局规模演变

| | 当前方案 | v1 | v2 | **v3** |
|--|---------|-----|-----|--------|
| 粗排 | — | 320 | 320 | **320** |
| 精排-参照物 | 2400 | 1440 | 720 | **96** |
| 精排-圆桌 | — | — | — | **552** |
| 头对头 | — | 240 | — | **—** |
| **Selection 合计** | **2400** | **~2000** | **1040** | **968** |
| 节省 | — | 17% | 57% | **60%** |
| 每代预估时间 | ~71min | ~59min | ~42min | **~38min** |

### 2.3 为什么全圆桌排序比参照物方案更准

以极端情况为例：假设有 A、B 两个个体，实力差距为 ELO+50（p≈0.57）。

**参照物方案**：A 和 B 各跟 5 个参照物各打 6 局。A 的观测胜率和 B 的观测胜率各有 ±40% 误差（95%CI），两个噪声叠加后，A 和 B 的 ELO 差可能在 -100 到 +200 之间随机漂移。72% 的概率能正确排序（来自 bootstrap 的 τ=0.805 隐含量）。

**全圆桌方案**：A 和 B 直接对战 2 局。每局是独立的 A vs B 信号。虽然 2 局在 p=0.57 时的误差约 ±69%，但 A 另外还有对 B 之外的 22 个人的对战数据——这些数据通过 ELO 网络传递信息，间接帮助推断 A 和 B 的关系。在全连接图中，每个个体有 46 局的有效信息，排序精度高于 5 参照物 × 6 局的 30 局信息量。

**信息量对比**（每人）：

| | 参照物方案（v2） | 全圆桌（v3） |
|--|----------------|-------------|
| 对局数/人 | 30 | **46**（23对手×2局）+ 4（锚定） = 50 |
| 直接比较比例 | 0%（无人际直接对局） | **100%**（与所有对手都打过） |
| p 分布 | 偏向两端（vs 弱/强参照物） | **集中在 0.5**（vs 同级对手） |
| 排序信息瓶颈 | 5 个参照物的 ELO 锚定值 | 图连通性（完全图，无瓶颈） |

---

## 3. 跨代可比性设计

### 3.1 为什么不能完全放弃参照物

全圆桌方案在代内排序精度很高，但丢失了跨代可比性。例如：
- gen5 的 #1 种子 ELO=1400
- gen10 的 #8 种子 ELO=1380

如果两代的种子从未与相同的参照物对战，这两个 ELO 值处于不同的度量空间，不能直接比较。这会破坏：
1. 对手池的 TrueSkill 评分（需要跨代可比性来淘汰弱对手）
2. 进化曲线的可视化（无法判断是否在进步）
3. 收敛判断（无法判断 best_elo 是否触顶）

### 3.2 锚定参照物的作用

保留 2 个锚定参照物（BasicTowerAI ELO=1000 + MediumRuleAI ELO=1600）作为"度量衡"，每代种子都跟它们对战。

| 锚定参照物 | 用途 | 局数 |
|-----------|------|------|
| BasicTowerAI (ELO=1000) | 低端锚点 | 24×2=48 |
| MediumRuleAI (ELO=1600) | 高端锚点 | 24×2=48 |

这两个参照物提供了跨代的"共同标尺"，使得 ELO 评分在不同代之间可比。类比：全圆桌确定了 24 个人的**相对高度**，锚定参照物确定了他们的**绝对海拔**。

不保留 BasicRandomAI 作为锚定参照物——它在 gen2 已经失去区分度（均值 p=0.83），作为标尺的作用也有限。

---

## 4. 对局数精确核算

### 4.1 每代对局明细

| 阶段 | 个体 | 对手 | 每对局数 | 对局数 | 说明 |
|------|------|------|---------|--------|------|
| **粗排** | 80 | 1 低门槛参照物 | 4 | **320** | n_battles=2 |
| **精排-圆桌** | 24 | C(24,2)=276对 | 2 | **552** | n_battles=1, 先后手各 1 |
| **精排-锚定** | 24 | 2 锚定参照物 | 2 | **96** | n_battles=1 |
| **Selection 合计** | | | | **968** | |
| 基线验证 | 8 | 3 参照物 | 100 | **1200** | 保持不变 |
| **每代总计** | | | | **2168** | |

### 4.2 各版本对比

| | 当前 | v1 | v2 | **v3** |
|--|------|-----|-----|--------|
| Selection 对局 | 2400 | ~2000 | 1040 | **968** |
| Selection 节省 | — | -17% | -57% | **-60%** |
| 每代总计 | 3600 | ~3200 | 2240 | **2168** |
| 每代总节省 | — | -11% | -38% | **-40%** |
| 每代时间（server10） | ~71min | ~59min | ~42min | **~38min** |

---

## 5. 排序模型选择：ELO vs TrueSkill

### 5.1 为什么全圆桌场景下 TrueSkill 更优

代码中已有 TrueSkill 实现（`BattleSharedPayoff` 和 `OpponentSelector` 中使用），但目前 Selection 的代内评估用的是 ELO。

全圆桌场景天然适合 TrueSkill：
- TrueSkill 的 **sigma 收敛机制**能在完全图对战场景下快速收敛（因为信息高度冗余）
- TrueSkill 输出的 **(mu, sigma)** 双参数比 ELO 的单参数更适合"选出 top-8"的需求——我们可以直接用 `mu - 2*sigma` 作为保守下界排序，而不是只看点估计
- TrueSkill 天然处理多人博弈图，不需要像 ELO 那样挨个参照物迭代更新

### 5.2 建议

**Phase 1（实施阶段）**：沿用 ELO。简单，代码改动最小，只需在 `_run_battle_phase()` 的基础上新增一个圆桌对战方法。

**Phase 2（后续优化）**：切换到 TrueSkill。改动的只是评分模型，对战编排逻辑不变。

> 本文档的代码修改方案基于 ELO 实现，TrueSkill 作为可选项在后续单独实现。

---

## 6. 代码修改方案

### 6.1 变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `selection/course_reference_pool.py` | **新建** | 与 v2 相同 |
| `selection/battle_selection.py` | **修改** | 新增圆桌对战方法 |
| `trainer/genetic_evolution_trainer.py` | **修改** | 集成 v3 精排流程 |
| `configs/ga_evo.yaml` | **修改** | 新增 v3 配置字段 |

### 6.2 v3 配置

```yaml
# 两阶段评估配置（v3 全圆桌版）
two_phase:
  enabled: true
  pass_ratio: 0.3              # 粗排晋级比例（80→24）
  course_n_battles: 2          # 粗排每对 4 局

  # 精排 — 全圆桌
  round_robin_enabled: true    # 启用全圆桌对战
  round_robin_n_battles: 1    # 每对 2 局（先后手各 1）

  # 精排 — 锚定参照物（跨代可比性）
  anchor_n_battles: 1          # 每对 2 局
  anchor_refs:                 # 锚定参照物列表
    - "BasicTowerAI"
    - "MediumRuleAI"

  # v3 不需要单独的 fine_n_battles
  # v3 不需要头对头修正

# 低门槛参照物池
course_pool:
  max_size: 5
  refresh_interval: 3
  stale_elo_gap: 400.0
```

### 6.3 新增方法：`_run_round_robin()`

```python
def _run_round_robin(
    self,
    individuals: list,
    n_battles: int,
    phase: str = "round_robin",
) -> list:
    """全圆桌对战评估。

    让 individuals 中每对个体两两对战，每对 n_battles 局
    （先后手各半 = 2 × n_battles 局）。

    Args:
        individuals: 参与圆桌的个体列表
        n_battles: 每对先后手各打的数量（总对局 = n_battles×2）
        phase: 阶段标识（用于日志）

    Returns:
        按 ELO 降序排列的个体列表
    """
    from itertools import combinations

    self.elo.reset_individuals()
    for ind in individuals:
        self.elo.ensure_player(ind.id)

    # 构建所有对战对
    pairs = list(combinations(individuals, 2))
    n_pairs = len(pairs)  # C(n, 2)

    # 预序列化所有个体
    agent_bytes = {}
    for ind in individuals:
        agent = self._create_ga_agent(ind)
        agent_bytes[ind.id] = AgentLoader.serialize(agent)

    # 构建任务列表
    tasks = []
    for ind_a, ind_b in pairs:
        tasks.append({
            "agent_a_id": ind_a.id,
            "agent_a_bytes": agent_bytes[ind_a.id],
            "agent_b_id": ind_b.id,
            "agent_b_bytes": agent_bytes[ind_b.id],
        })

    # 并行执行
    max_rounds = self.config.max_rounds
    max_workers = min(len(tasks), self.config.max_workers)
    task_timeout = n_battles * 2 * _BATTLE_TIMEOUT

    logger.info(
        f"Round-robin: {n_pairs} pairs × {n_battles * 2} battles = "
        f"{n_pairs * n_battles * 2} games, max_workers={max_workers}"
    )

    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
        futures = {}
        for task in tasks:
            future = executor.submit(
                _round_robin_worker,
                task["agent_a_bytes"],
                task["agent_b_bytes"],
                n_battles,
                max_rounds,
            )
            futures[future] = task

        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                battle_results = future.result(timeout=task_timeout)
            except Exception as e:
                logger.error(
                    f"Round-robin task failed: "
                    f"{task['agent_a_id']} vs {task['agent_b_id']}: {e}"
                )
                self.elo.update(task["agent_a_id"], task["agent_b_id"], 0.5)
                completed += 1
                continue

            for result in battle_results:
                if "error" in result:
                    self.elo.update(task["agent_a_id"], task["agent_b_id"], 0.5)
                    continue

                game_result = result.get("result")
                if game_result == "agent1_win":
                    self.elo.update(task["agent_a_id"], task["agent_b_id"], 1.0)
                elif game_result == "agent2_win":
                    self.elo.update(task["agent_a_id"], task["agent_b_id"], 0.0)
                else:
                    self.elo.update(task["agent_a_id"], task["agent_b_id"], 0.5)

            completed += 1
            if completed % 50 == 0 or completed == n_pairs:
                logger.info(
                    f"Round-robin progress: {completed}/{n_pairs} pairs"
                )

    # 回写并排序
    for individual in individuals:
        individual.elo_rating = self.elo.get_rating(individual.id)

    ranked = sorted(individuals, key=lambda ind: ind.elo_rating, reverse=True)
    logger.info(
        f"Round-robin done: best_elo={ranked[0].elo_rating:.1f}, "
        f"worst_elo={ranked[-1].elo_rating:.1f}"
    )
    return ranked
```

### 6.4 新增 worker：`_round_robin_worker()`

```python
def _round_robin_worker(
    ga_bytes_a: bytes,
    ga_bytes_b: bytes,
    n_battles: int,
    max_rounds: int,
) -> List[Dict[str, Any]]:
    """在子进程中执行 (个体A, 个体B) 的全圆桌对战。

    先后手各 n_battles 局，共 2×n_battles 局。
    """
    agent_a = AgentLoader.deserialize(ga_bytes_a)
    agent_b = AgentLoader.deserialize(ga_bytes_b)
    agent_a = _ensure_compatible(agent_a)
    agent_b = _ensure_compatible(agent_b)

    results = []
    for i in range(n_battles * 2):
        first_player = i % 2  # 0: A 先手, 1: B 先手
        result = _execute_battle(agent_a, agent_b, first_player, max_rounds)
        results.append(result)
    return results
```

### 6.5 通用对战阶段方法：`_run_battle_phase()`

从现有 `evaluate_population()` 中提取的通用"个体 vs 参照物"对战方法。粗排和锚定阶段都复用此方法。

**关键设计决策**：
- **不重置 ELO**：调用方控制 ELO 生命周期。粗排前调用方 reset，锚定前不 reset（在圆桌 ELO 上累积）。
- **复用现有 `_flat_battle_worker`**：签名 `(ga_bytes, ref_bytes, n_battles, max_rounds)` 完全匹配粗排和锚定场景。
- **只排序传入的 individuals**：粗排排序 80 人，锚定排序 24 人。

```python
def _run_battle_phase(
    self,
    individuals: List[Individual],
    references: List[ReferenceAgent],
    n_battles: int,
    phase: str = "battle",
) -> List[Individual]:
    """通用对战阶段：individuals vs references，ELO 累积更新后返回排序结果。

    不重置 ELO——调用方负责管理 ELO 生命周期。
    本方法在现有 ELO 状态上增量更新（追加对战数据）。

    Args:
        individuals: 参评个体列表
        references: 参照物列表
        n_battles: 每对先后手各 n_battles 局（总对局 = n_battles × 2）
        phase: 阶段标识（用于日志前缀）

    Returns:
        按 ELO 降序排列的个体列表（仅限传入的 individuals）
    """
    if not individuals or not references:
        logger.warning(
            f"[{phase}] Empty individuals or references, returning unsorted."
        )
        return sorted(individuals, key=lambda ind: ind.elo_rating, reverse=True)

    # ── 步骤 0：确保个体和参照物在 ELO 中注册 ──
    for ind in individuals:
        self.elo.ensure_player(ind.id)

    for ref in references:
        self.elo.ensure_player(ref.name)
        if ref.is_anchor:
            # 锚定参照物：强制硬编码 ELO，不参与更新
            self.elo._ratings[ref.name] = ref.elo
        else:
            # 动态参照物：使用上一轮实际 ELO
            self.elo._ratings[ref.name] = self.elo.get_dynamic_ref_rating(ref.name)

    # ── 步骤 1：预序列化所有参照物 ──
    ref_bytes: Dict[str, bytes] = {}
    for ref in references:
        agent = ref.agent_factory(1)
        ref_bytes[ref.name] = AgentLoader.serialize(agent)

    # ── 步骤 2：构建任务列表 ──
    tasks: List[dict] = []
    max_rounds = self.config.max_rounds

    for individual in individuals:
        ga_agent = self._create_ga_agent(individual)
        ga_bytes = AgentLoader.serialize(ga_agent)
        for ref in references:
            tasks.append({
                "individual_id": individual.id,
                "ga_bytes": ga_bytes,
                "ref_name": ref.name,
                "ref_bytes": ref_bytes[ref.name],
            })

    # ── 步骤 3：并行提交 ──
    max_workers = min(len(tasks), self.config.max_workers)
    task_timeout = n_battles * 2 * _BATTLE_TIMEOUT

    logger.info(
        f"[{phase}] {len(tasks)} tasks ({len(individuals)} inds x "
        f"{len(references)} refs x n_battles={n_battles}), "
        f"max_workers={max_workers}"
    )

    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
        futures = {}
        for task in tasks:
            future = executor.submit(
                _flat_battle_worker,      # 复用现有 worker
                task["ga_bytes"],
                task["ref_bytes"],
                n_battles,
                max_rounds,
            )
            futures[future] = task

        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                battle_results = future.result(timeout=task_timeout)
            except Exception as e:
                logger.error(
                    f"[{phase}] Battle task failed: "
                    f"{task['individual_id']} vs {task['ref_name']}: {e}"
                )
                self.elo.update(task["individual_id"], task["ref_name"], 0.5)
                completed += 1
                continue

            for result in battle_results:
                if "error" in result:
                    self.elo.update(task["individual_id"], task["ref_name"], 0.5)
                    continue

                game_result = result.get("result")
                if game_result == "agent1_win":
                    self.elo.update(task["individual_id"], task["ref_name"], 1.0)
                elif game_result == "agent2_win":
                    self.elo.update(task["individual_id"], task["ref_name"], 0.0)
                else:
                    self.elo.update(task["individual_id"], task["ref_name"], 0.5)

            completed += 1
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                logger.info(f"[{phase}] Progress: {completed}/{len(tasks)}")

    # ── 步骤 4：回写 ELO 并排序（仅限传入的 individuals）──
    for individual in individuals:
        individual.elo_rating = self.elo.get_rating(individual.id)

    # 保存动态参照物当轮 ELO（供下一代使用）
    for ref in references:
        if not ref.is_anchor:
            self.elo.save_dynamic_ref_rating(
                ref.name, self.elo.get_rating(ref.name)
            )

    ranked = sorted(individuals, key=lambda ind: ind.elo_rating, reverse=True)
    logger.info(
        f"[{phase}] Done: best_elo={ranked[0].elo_rating:.1f}, "
        f"worst_elo={ranked[-1].elo_rating:.1f}"
    )
    return ranked
```

### 6.6 `evaluate_population_two_phase()` v3 完整版本

**ELO 生命周期管理**（核心设计）：

三个对战阶段共享同一个 `self.elo` 实例。`_run_round_robin()` 内部会调用 `self.elo.reset_individuals()` 清除粗排评分，因此需要在 reset 前保存快照，在锚定阶段后恢复淘汰个体的 ELO。

```
时间 ──────────────────────────────────────────────────────────────────►

reset ─► 粗排战 ─► 快照 ─► reset ─► 圆桌战 ─► 锚定战 ─► 恢复 ─► 排序
         (80人)    (保存)           (24人)    (24人累   (56人)   (80人)
         80人ELO   到 dict           24人ELO   积更新)   粗排ELO  完整ELO
                                             24人ELO
                                             = 圆桌+锚定综合
```

```python
def evaluate_population_two_phase(
    self,
    population: Population,
    dynamic_references: List[ReferenceAgent],
    course_pool: "CourseReferencePool",
    config: dict,
) -> List[Individual]:
    """两阶段评估（v3：粗排 + 全圆桌精排 + 锚定参照物）。

    ELO 生命周期管理：
      1. 全体 reset → 粗排战 → 得到 80 人 ELO
      2. 保存 80 人 ELO 快照 → reset → 圆桌战 → 得到 24 人 ELO
      3. 不 reset → 锚定战 → 24 人 ELO 增量更新（圆桌+锚定综合）
      4. 恢复 56 人粗排 ELO → 80 人完整排序
    """
    all_individuals = population.individuals
    pass_ratio = config.get("pass_ratio", 0.3)
    n_course_battles = config.get("course_n_battles", 2)
    n_pass = max(1, int(len(all_individuals) * pass_ratio))

    # ═══════════════════════════════════════════════════════════════
    # 阶段 1：粗排
    # ═══════════════════════════════════════════════════════════════
    self.elo.reset_individuals()

    course_refs = course_pool.select(config.get("course_n_refs", 1))
    if not course_refs or len(course_refs) < 1:
        fallback = course_pool.get_fallback_refs()[:1]
        course_ref_agents = [
            ref for ref in self.anchor_references
            if ref.name == fallback[0][0]
        ]
    else:
        course_ref_agents = self._build_course_references(course_refs)

    ranked_course = self._run_battle_phase(
        all_individuals,
        course_ref_agents,
        n_course_battles,
        phase="course",
    )

    # ── 去重检查：粗排参照物无区分度时全部晋级 ──
    course_elos = [ind.elo_rating for ind in ranked_course]
    if len(set(round(e, 1) for e in course_elos)) <= 1:
        logger.warning(
            f"Course ranking has zero discriminability "
            f"(all ELO ≈ {course_elos[0]:.1f}). "
            f"Falling back: all 80 individuals advance to fine ranking."
        )
        n_pass = len(all_individuals)

    passed = ranked_course[:n_pass]
    not_passed = ranked_course[n_pass:]

    # ── 保存粗排 ELO 快照（_run_round_robin 内部会 reset）──
    course_elo_snapshot: Dict[str, float] = {}
    for ind in ranked_course:
        course_elo_snapshot[ind.id] = ind.elo_rating

    logger.info(
        f"Course ranking: {len(passed)}/{len(all_individuals)} passed"
    )

    # ═══════════════════════════════════════════════════════════════
    # 阶段 2a：全圆桌对战（内部调用 elo.reset_individuals()）
    # ═══════════════════════════════════════════════════════════════
    rr_n_battles = config.get("round_robin_n_battles", 1)
    ranked_rr = self._run_round_robin(
        passed,
        n_battles=rr_n_battles,
        phase="round_robin",
    )

    # ═══════════════════════════════════════════════════════════════
    # 阶段 2b：锚定参照物（不 reset，在圆桌 ELO 上增量更新）
    # ═══════════════════════════════════════════════════════════════
    anchor_ref_names = config.get("anchor_refs", ["BasicTowerAI", "MediumRuleAI"])
    anchor_refs = [
        ref for ref in self.anchor_references
        if ref.name in anchor_ref_names
    ]
    anchor_n_battles = config.get("anchor_n_battles", 1)

    if anchor_refs:
        ranked_anchor = self._run_battle_phase(
            passed,
            anchor_refs,
            anchor_n_battles,
            phase="anchor",
        )
        # ranked_anchor 的 ELO = 圆桌 + 锚定综合（已在 self.elo 中累积）
    else:
        ranked_anchor = ranked_rr
        logger.warning(
            "No anchor references configured; cross-gen comparability lost."
        )

    # ═══════════════════════════════════════════════════════════════
    # 最终排序：恢复淘汰个体的粗排 ELO，合成完整 80 人排序
    # ═══════════════════════════════════════════════════════════════
    for ind in not_passed:
        saved_elo = course_elo_snapshot.get(ind.id, self.elo.initial_rating)
        self.elo.set_rating(ind.id, saved_elo)

    for individual in all_individuals:
        individual.elo_rating = self.elo.get_rating(individual.id)

    final_ranked = sorted(all_individuals,
                          key=lambda ind: ind.elo_rating, reverse=True)

    logger.info(
        f"Two-phase selection done: best_elo={final_ranked[0].elo_rating:.1f}"
    )
    return final_ranked
```

### 6.7 Trainer 集成变化

与 v2 基本相同，唯一变化是不再需要 `_build_dynamic_references()` 中的对手池动态参照物和上代冠军（精排的圆桌已覆盖这些信息）。简化为：

```python
# v3: 精排不再使用 dynamic_references
# 对手池动态参照物仅用于 opponent_pool 的 TrueSkill 维护
ranked = self.battle_selection.evaluate_population_two_phase(
    population=population,
    dynamic_references=[],  # v3 不需要
    course_pool=self.course_pool,
    config=two_phase_cfg,
)
```

### 6.8 Worker 模块依赖与 `_round_robin_worker()` 详细定义

#### 6.8.1 依赖分析

`_round_robin_worker()` 作为 `battle_selection.py` 的模块级函数，依赖以下 3 个符号：

| 符号 | 来源 | 在 `battle_selection.py` 中的导入状态 |
|------|------|--------------------------------------|
| `AgentLoader.deserialize()` | `..battle.agent_loader` | ✅ 已有（第 11 行 `from ..battle.agent_loader import AgentLoader`） |
| `_ensure_compatible()` | `..battle.battle_simulator` | ✅ 已有（第 8 行 `from ..battle.battle_simulator import ..., _ensure_compatible, _execute_battle`） |
| `_execute_battle()` | `..battle.battle_simulator` | ✅ 已有（同上） |

**结论**：无需新增任何 import。`_round_robin_worker()` 可以直接定义为 `battle_selection.py` 的模块级函数（与现有 `_flat_battle_worker()` 同级）。

#### 6.8.2 完整定义

与 `_flat_battle_worker()` 对称，但对战双方是平等的个体（不需要标记 `_individual_id` / `_ref_name`）：

```python
# ── v3 新增：全圆桌 worker（模块级函数，与 _flat_battle_worker 同级）──

def _round_robin_worker(
    ga_bytes_a: bytes,
    ga_bytes_b: bytes,
    n_battles: int,
    max_rounds: int,
) -> List[Dict[str, Any]]:
    """在子进程中执行 (个体A, 个体B) 的全圆桌对战。

    先后手各 n_battles 局，共 2×n_battles 局。
    复用模块级 _ensure_compatible 和 _execute_battle（来自 battle_simulator），
    以及 AgentLoader（来自 agent_loader）。

    Args:
        ga_bytes_a: 个体 A 的序列化 agent
        ga_bytes_b: 个体 B 的序列化 agent
        n_battles: 先后手各 n_battles 局
        max_rounds: 单局最大回合数

    Returns:
        [{result, first_player, ...}, ...] 共 n_battles×2 条
    """
    agent_a = AgentLoader.deserialize(ga_bytes_a)
    agent_b = AgentLoader.deserialize(ga_bytes_b)
    agent_a = _ensure_compatible(agent_a)
    agent_b = _ensure_compatible(agent_b)

    results = []
    for i in range(n_battles * 2):
        first_player = i % 2  # 0: A 先手, 1: B 先手
        result = _execute_battle(agent_a, agent_b, first_player, max_rounds)
        results.append(result)
    return results
```

#### 6.8.3 与 `_flat_battle_worker()` 对比

| | `_flat_battle_worker` | `_round_robin_worker` |
|---|---|---|
| 输入 | `(ga_bytes, ref_bytes, ...)` | `(ga_bytes_a, ga_bytes_b, ...)` |
| 对战双方 | 个体 ↔ 参照物 | 个体 A ↔ 个体 B |
| 结果标记 | 写入 `_individual_id`, `_ref_name` | 不写标记（双方平等） |
| 依赖 | `AgentLoader`, `_ensure_compatible`, `_execute_battle` | 完全相同的 3 个依赖 |

### 6.9 补充方法：`_build_course_references()`

将 `CourseReferencePool.select()` 返回的 `(name, elo)` 元组转换为 `ReferenceAgent` 列表，供 `_run_battle_phase()` 使用。

```python
def _build_course_references(
    self, course_refs: List[tuple],
) -> List[ReferenceAgent]:
    """将 CourseReferencePool 返回的 (name, elo) 转换为 ReferenceAgent 列表。

    Args:
        course_refs: [(name, elo), ...] 来自 CourseReferencePool.select()

    Returns:
        ReferenceAgent 列表（is_anchor=False，粗排参照物是动态选择的）
    """
    references = []
    for name, elo in course_refs:
        # 优先从已有 anchor_references 中查找匹配的工厂
        existing = next(
            (ref for ref in self.anchor_references if ref.name == name), None
        )
        if existing:
            factory = existing.agent_factory
        else:
            factory = lambda pid, n=name: AgentLoader.load(n, pid)

        references.append(
            ReferenceAgent(
                name=name,
                elo=elo,
                agent_factory=factory,
                is_anchor=False,  # 粗排参照物是动态选择的，非固定锚定
            )
        )
    return references
```

---

## 7. 时间复杂度分析

### 7.1 对战编排

全圆桌 C(24,2)=276 对，即使顺序提交，也可利用现有的 ProcessPoolExecutor 并行化（max_workers=12~25）。

当前 `_flat_battle_worker` 模式是每对在子进程中独立执行 n_battles×2 局（避免了重复序列化开销）。`_round_robin_worker` 采用相同模式。

### 7.2 ELO 更新

276 对 × 2 局 = 552 次 ELO 更新，每次 O(1)，总计 O(552)。可以忽略不计。

### 7.3 总耗时估算

- 粗排：320 局 ≈ 2~3 分钟
- 圆桌：552 局 ≈ 4~5 分钟
- 锚定：96 局 ≈ 1 分钟

**精排对战总计约 5~6 分钟**（并行执行时），比当前方案的对战阶段（约 50 分钟）节省约 88%。

---

## 8. 实施步骤

| 步骤 | 内容 | 预估 |
|------|------|------|
| 1 | 修复前提 bug | 0.5h |
| 2 | 新建 CourseReferencePool | 1h |
| 3 | 从 evaluate_population() 提取 _run_battle_phase() | 1h |
| 4 | 实现 _round_robin_worker() + _run_round_robin() | 1.5h |
| 5 | 实现 evaluate_population_two_phase()（v3） | 1h |
| 6 | 修改 trainer 集成（移除 dynamic_refs 依赖） | 0.5h |
| 7 | 新增 yaml 配置 | 0.5h |
| 8 | 单元验证 | 0.5h |
| **合计** | | **~6.5h** |

---

## 9. 三版方案对比

| 维度 | v1（参照物） | v2（精简参照物） | **v3（全圆桌）** |
|------|------------|----------------|-----------------|
| 精排人数 | 40 | 24 | **24** |
| 精排方式 | 全参照物（6×6局） | 全参照物（5×6局） | **全圆桌（276对×2局）+ 锚定（2×2局）** |
| 精排对局 | 1440 | 720 | **648** |
| Selection 总计 | ~2000 | 1040 | **968** |
| 节省（vs 当前） | -17% | -57% | **-60%** |
| 跨代可比性 | 有 | 有 | **有**（2 锚定） |
| 代内排序精度 | 间接（通过 ref） | 间接 | **直接（人打人）** |
| 每人有效对局 | 36 | 30 | **50**（46圆桌 + 4锚定） |
| 实现复杂度 | 中 | 中 | **中**（新增圆桌 worker） |
| 预估每代时间 | ~59min | ~42min | **~38min** |

### 选择建议

- **如果追求最大计算量削减**：选 v3（968 局，-60%）
- **如果担心圆桌实现的复杂度**：选 v2（1040 局，-57%，代码改动更少）
- v3 的额外复杂度（约 1.5h 的 _run_round_robin 实现）换来的是更好的排序精度和更少的对局数，性价比高
