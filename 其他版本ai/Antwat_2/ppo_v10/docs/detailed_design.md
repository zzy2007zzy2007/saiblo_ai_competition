# PPO_V10 详细设计文档

> 基于 [requirements_spec.md](./requirements_spec.md)，面向 AI 编程的逐文件、逐函数级设计。

---

## 0. 设计原则（来自 docs/design/principle.txt）

- 最大限度利用 Ant-Game SDK，只做游戏引擎 wrapper，不修改 SDK 代码
- 不做容错机制，让错误尽早暴露；异常应合理捕获记录并抛向上层
- 不做过度设计，只在现有功能上优化代码结构
- 所有日志使用 loguru，不用 print
- GPU 降级 CPU 时输出警告
- 仅考虑 Linux 环境

---

## 1. 文件级变更清单

### 1.1 保留不变的文件

以下文件完整保留，**无需任何修改**：

| 文件路径（相对 `src/ppo_ga/`） | 说明 |
|---|---|
| `network/ant_war_policy_value_network.py` | 网络结构不变 |
| `network/hex_cnn_encoder.py` | HexCNN 编码器 |
| `network/hex_conv.py` | 六边形卷积 |
| `network/mlp_encoder.py` | MLP 编码器 |
| `network/heads.py` | 动作/价值头 |
| `network/__init__.py` | |
| `genome/genome.py` | Genome、cosine_similarity 等 |
| `genome/weight_init.py` | 随机初始化 |
| `genome/__init__.py` | |
| `battle/ant_war_agent.py` | 基类 |
| `battle/ga_war_agent.py` | GA agent |
| `battle/ppo_war_agent.py` | PPO agent |
| `battle/opponent_agent.py` | Opponent agent |
| `battle/agent_loader.py` | Agent 加载器 |
| `battle/rule_based_agents.py` | 规则基 AI |
| `battle/battle_simulator.py` | 对战模拟器 |
| `battle/battle_coordinator.py` | 对战协调器 |
| `battle/result_aggregator.py` | 结果聚合 |
| `battle/report_generator.py` | 报告生成 |
| `battle/utils/__init__.py` | |
| `battle/__init__.py` | |
| `env/observation.py` | 观测编码 |
| `env/action_mask.py` | 动作掩码 |
| `env/__init__.py` | |
| `evaluation/baseline_evaluator.py` | 基线评估（诊断环节） |
| `evaluation/__init__.py` | |
| `monitor/system_metrics_sampler.py` | 系统监控 |
| `monitor/ga_logger.py` | GA 日志器 |
| `monitor/battle_log_writer.py` | 对战日志写入 |
| `monitor/generation_stats_writer.py` | 代统计写入 |
| `monitor/__init__.py` | |
| `utils/action_constants.py` | 动作常量 |
| `utils/obs_utils.py` | 观测工具 |
| `utils/__init__.py` | |
| `compat/adapters.py` | 兼容适配器 |
| `compat/__init__.py` | |
| `config/path_config.py` | 路径配置（不变） |
| `_agent/protocol.py` | Agent 协议 |
| `_agent/__init__.py` | |
| `trainer/neural_agent.py` | 神经 agent 包装器 |

根目录保留不变：
- `train.py`（需小幅调整 CLI 参数）
- `train_ga.py`（需小幅调整 CLI 参数）
- `requirements.txt`

### 1.2 需要删除的文件

| 文件路径 | 删除原因 |
|---|---|
| `src/ppo_ga/selection/elo_rating.py` | ELO 评分系统不再使用 |
| `src/ppo_ga/league/opponent_pool.py` | 被 benchmark_pool 替代 |
| `src/ppo_ga/league/opponent_selector.py` | 不再需要动态对手选择 |

### 1.3 需要新增的文件

| 文件路径 | 说明 |
|---|---|
| `src/ppo_ga/selection/kendall_tau.py` | Kendall τ 计算 |
| `src/ppo_ga/selection/benchmark_test.py` | 基准测试阶段 |
| `src/ppo_ga/selection/round_robin.py` | 循环赛阶段 |
| `src/ppo_ga/league/benchmark_pool.py` | 基准对手池管理 |
| `src/ppo_ga/league/elite_seed_pool.py` | 精英种子池管理 |
| `configs/ga_evo_v10.yaml` | v10 配置文件 |

### 1.4 需要修改的文件

| 文件路径 | 修改程度 | 说明 |
|---|---|---|
| `evolution/population_manager.py` | 小改 | Individual/Population 字段变更 |
| `selection/battle_selection.py` | 重写 | 改为调用 benchmark_test + round_robin |
| `league/battle_shared_payoff.py` | 小改 | 增加 Kendall τ 相关辅助 |
| `league/trueskill_rating.py` | 不变/小改 | 保留，可选增加 to_dict/from_dict |
| `trainer/genetic_evolution_trainer.py` | 重写 | 新主循环 |
| `trainer/checkpoint_manager.py` | 小改 | ELO→TrueSkill 字段 |
| `trainer/logging_subsystem.py` | 小改 | ELO→TrueSkill 日志字段 |
| `config/config_parser.py` | 中改 | 新配置段 + 删除旧配置段 |
| `train.py` | 小改 | 删除不再需要的 CLI 参数 |
| `train_ga.py` | 小改 | 删除不再需要的 CLI 参数 |

---

## 2. 数据结构变更

### 2.1 Individual（`evolution/population_manager.py`）

**变更前：**
```python
@dataclass
class Individual:
    id: str
    generation: int
    genome: Genome
    elo_rating: float = 1200.0          # ← 删除
    seed_rank: int = -1
    parent_ids: List[str] = field(default_factory=list)
    mutation_desc: str = ""
```

**变更后：**
```python
@dataclass
class Individual:
    id: str
    generation: int
    genome: Genome
    trueskill_mu: float = 25.0           # ← 新增：TrueSkill mu
    trueskill_sigma: float = 8.333       # ← 新增：TrueSkill sigma
    seed_rank: int = -1
    parent_ids: List[str] = field(default_factory=list)
    mutation_desc: str = ""
```

### 2.2 Population（`evolution/population_manager.py`）

**变更前：**
```python
@dataclass
class Population:
    generation: int
    individuals: List[Individual]
    best_elo: float = 0.0                # ← 删除
    mean_elo: float = 0.0                # ← 删除
    diversity: float = 0.0
```

**变更后：**
```python
@dataclass
class Population:
    generation: int
    individuals: List[Individual]
    best_rating: float = 0.0             # ← 重命名：最高 TrueSkill mu
    mean_rating: float = 0.0             # ← 重命名：平均 TrueSkill mu
    diversity: float = 0.0
```

### 2.3 新增：BenchmarkPoolEntry（`league/benchmark_pool.py`）

```python
@dataclass
class BenchmarkPoolEntry:
    name: str                            # agent 名称
    order: int                           # 加入次序（0-based，越小越早）
    agent_type: str                      # "builtin" | "individual"
    # 当 agent_type == "individual" 时有以下字段：
    individual_id: str = ""              # 个体 ID
    generation: int = -1                 # 加入时的代数
    checkpoint_path: str = ""            # 个体模型检查点路径
    genome_weights: Optional[np.ndarray] = None  # 基因组权重（用于 cos 相似度过滤）
```

### 2.4 新增：EliteSeedEntry（`league/elite_seed_pool.py`）

```python
@dataclass
class EliteSeedEntry:
    individual_id: str
    generation: int
    seed_rank: int
    trueskill_mu: float
    trueskill_sigma: float
    checkpoint_path: str
    genome_hash: str = ""                # 基因组 SHA256（用于快速去重判断）
```

---

## 3. 新增文件详细设计

### 3.1 `selection/kendall_tau.py`

**功能**：计算 Kendall τ 相关系数。

```python
import numpy as np

def kendall_tau(ranking_a: list, ranking_b: list) -> float:
    """
    计算两个排名列表的 Kendall τ 相关系数。

    Args:
        ranking_a: 排名 A，元素为可比较的个体 ID 列表，按排名从高到低
        ranking_b: 排名 B，格式同上

    Returns:
        τ ∈ [-1, 1]
    """
    n = len(ranking_a)
    if n < 2:
        return 1.0

    # 建立 B 排名中每个元素的位次映射
    rank_b = {item: i for i, item in enumerate(ranking_b)}

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = 1  # ranking_a[i] 排在 ranking_a[j] 前面
            # 在 B 中的相对顺序
            b_order = 1 if rank_b[ranking_a[i]] < rank_b[ranking_a[j]] else -1
            if a_order * b_order > 0:
                concordant += 1
            else:
                discordant += 1

    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def ranking_from_trueskill(
    individual_ids: list,
    trueskill_mus: dict,       # {id: mu}
) -> list:
    """按 TrueSkill mu 降序生成排名列表"""
    return sorted(individual_ids, key=lambda iid: trueskill_mus.get(iid, 0.0), reverse=True)


def ranking_from_head_to_head(
    individual_ids: list,
    results: dict,             # {individual_id: {"wins": int, "losses": int, "draws": int}}
) -> list:
    """根据对战结果生成排名列表（按胜率降序）"""
    def win_rate(iid):
        r = results.get(iid, {})
        total = r.get("wins", 0) + r.get("losses", 0) + r.get("draws", 0)
        if total == 0:
            return 0.0
        return (r.get("wins", 0) + 0.5 * r.get("draws", 0)) / total

    return sorted(individual_ids, key=win_rate, reverse=True)
```

**导出函数**：
- `kendall_tau(ranking_a, ranking_b) -> float`
- `ranking_from_trueskill(individual_ids, trueskill_mus) -> list`
- `ranking_from_head_to_head(individual_ids, results) -> list`

---

### 3.2 `selection/benchmark_test.py`

**功能**：基准测试阶段 — 按次序用基准池 agent 与全体个体对战，累计 TrueSkill，按 Kendall τ 早停。

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger

from ..evolution.population_manager import Individual, Population
from ..battle.battle_config import GABattleConfig
from ..battle.agent_loader import AgentLoader
from ..battle.ga_war_agent import GAWarAgent
from ..battle.battle_simulator import _ensure_compatible, _execute_battle
from ..genome.genome import load_genome_to_network, cosine_similarity
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..league.benchmark_pool import BenchmarkPool, BenchmarkPoolEntry
from ..league.battle_shared_payoff import BattleSharedPayoff
from .kendall_tau import kendall_tau, ranking_from_trueskill, ranking_from_head_to_head


@dataclass
class BenchmarkTestConfig:
    """基准测试配置"""
    n_battles: int = 2
    kendall_tau_threshold: float = 0.7
    max_workers: int = 25
    max_rounds: int = 512
    device: str = "cuda"
    hidden_dim: int = 256
    enable_auxiliary: bool = True


def _benchmark_battle_worker(
    ga_bytes: bytes,
    bench_bytes: bytes,
    n_battles: int,
    max_rounds: int,
) -> list:
    """子进程中执行 (个体, benchmark_agent) 的对战"""
    # 复用 v9 的 _flat_battle_worker 模式（仅名称不同）
    pass  # 实现同 v9 _flat_battle_worker


class BenchmarkTest:
    """基准测试 — 有序遍历基准对手池，按 Kendall τ 早停"""

    def __init__(
        self,
        config: BenchmarkTestConfig,
        benchmark_pool: BenchmarkPool,
        payoff: BattleSharedPayoff,
    ):
        self.config = config
        self._pool = benchmark_pool
        self._payoff = payoff

    def evaluate(
        self,
        population: Population,
    ) -> List[Individual]:
        """
        执行基准测试，返回按 TrueSkill mu 降序排列的个体列表。

        流程：
        for entry in benchmark_pool(按 order 升序):
            1. 加载 entry 的 agent
            2. 与全部个体对战 n_battle=2 局
            3. 在 payoff 中更新 TrueSkill
            4. 计算 entry 的 Kendall τ（R_base vs R_ref）
            5. if τ > threshold: break

        第 1 代且第 1 个 agent 时：τ 直接返回 0（不早停）

        Returns:
            按 trueskill_mu 降序排列的个体列表
        """
        pass

    def _compute_tau_for_benchmark_agent(
        self,
        individual_ids: List[str],
        head_to_head_results: Dict[str, dict],
        current_ranking: List[str],
    ) -> float:
        """
        计算单个 benchmark_agent 的 Kendall τ。

        R_base = 仅基于该 agent 对战结果的排名
        R_ref  = 当前累计 TrueSkill mu 排名

        Returns:
            τ ∈ [-1, 1]
        """
        pass

    def _create_ga_agent(self, individual: Individual):
        """从 Individual 创建 GAWarAgent（同 v9 BattleSelection._create_ga_agent）"""
        pass
```

**关键细节**：
- `_benchmark_battle_worker()` 与 v9 的 `_flat_battle_worker()` 完全相同，仅函数名不同
- `_create_ga_agent()` 与 v9 的 `BattleSelection._create_ga_agent()` 完全相同
- `_reconfigure_loguru()` 保持不变（从 v9 复制或提取到共享模块）
- 第 1 代的第 1 个 agent：`τ` 直接返回 0.0，不早停
- 每轮对战后，所有个体的 `trueskill_mu` 和 `trueskill_sigma` 从 `payoff` 中回写

---

### 3.3 `selection/round_robin.py`

**功能**：循环赛阶段 — Top N 个体全圆桌对战，TrueSkill 排名，基因相似度去重选 seeds。

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from loguru import logger

from ..evolution.population_manager import Individual
from ..battle.battle_config import GABattleConfig
from ..battle.agent_loader import AgentLoader
from ..battle.ga_war_agent import GAWarAgent
from ..battle.battle_simulator import _ensure_compatible, _execute_battle
from ..genome.genome import load_genome_to_network, cosine_similarity
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..league.battle_shared_payoff import BattleSharedPayoff


@dataclass
class RoundRobinConfig:
    """循环赛配置"""
    top_n: int = 24
    n_battles: int = 4
    max_workers: int = 25
    max_rounds: int = 512
    device: str = "cuda"
    hidden_dim: int = 256
    enable_auxiliary: bool = True


def _rr_battle_worker(
    agent_a_bytes: bytes,
    agent_b_bytes: bytes,
    n_battles: int,
    max_rounds: int,
) -> list:
    """子进程中执行两个个体间的对战"""
    pass  # 同 _flat_battle_worker 模式，两个 GA agent 对战


class RoundRobinTournament:
    """循环赛 — 全圆桌对战 + TrueSkill 排名 + 多样性种子选拔"""

    def __init__(
        self,
        config: RoundRobinConfig,
        payoff: BattleSharedPayoff,
    ):
        self.config = config
        self._payoff = payoff

    def run(
        self,
        top_individuals: List[Individual],
        num_seeds: int,
        elitism_count: int,
        dedup_config: dict,
    ) -> Tuple[List[Individual], List[Individual]]:
        """
        执行循环赛，返回 (seeds, elites)。

        流程：
        1. Top N 个体全圆桌对战（每对 n_battle 局，先后手各半）
        2. 更新 payoff TrueSkill
        3. 回写 individual.trueskill_mu / trueskill_sigma
        4. 按 mu 降序排列
        5. 基因相似度过滤 → select_seeds_diverse()
        6. 前 elitism_count 个标记为 elites

        Args:
            top_individuals: 基准测试 Top N 的个体（已含 trueskill_mu）
            num_seeds: 种子数量
            elitism_count: 精英数量
            dedup_config: 去重配置 {"hard_threshold": 0.95, "soft_threshold": 0.85, "soft_max_per_family": 2}

        Returns:
            (seeds: List[Individual], elites: List[Individual])
        """
        pass

    def _generate_matchups(
        self,
        individuals: List[Individual],
    ) -> List[Tuple[Individual, Individual]]:
        """生成全圆桌对战配对列表（每对只出现一次）"""
        pass

    def select_seeds_diverse(
        self,
        ranked: List[Individual],
        hard_threshold: float = 0.95,
        soft_threshold: float = 0.85,
        soft_max_per_family: int = 2,
        num_seeds: int = 8,
    ) -> List[Individual]:
        """
        基因相似度去重选出 seeds。
        与 v9 BattleSelection.select_seeds_diverse() 完全相同的逻辑，
        仅将 ELO 排序改为按 trueskill_mu 排序。

        移入 RoundRobinTournament 是因为循环赛负责种子选拔。
        """
        pass

    def _create_ga_agent(self, individual: Individual, player_id: int = 0) -> GAWarAgent:
        """从 Individual 创建 GAWarAgent"""
        pass
```

**关键细节**：
- `_generate_matchups()` 生成 C(N,2) 个配对，每个配对对战 n_battle 局
- `select_seeds_diverse()` 从 v9 `BattleSelection` 完整移植，入参 `ranked` 按 `trueskill_mu` 降序排列
- 全圆桌对战的并行执行模式与 v9 扁平并行相同
- 从基准测试继承的 TrueSkill 评分直接用于循环赛，不重置

---

### 3.4 `league/benchmark_pool.py`

**功能**：基准对手池管理。

```python
import json
import os
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class BenchmarkPoolEntry:
    """基准池条目"""
    name: str
    order: int
    agent_type: str                  # "builtin" | "individual"
    individual_id: str = ""
    generation: int = -1
    checkpoint_path: str = ""
    genome_weights: Optional[np.ndarray] = None   # 序列化时转为 list


class BenchmarkPool:
    """基准对手池管理器

    特性：
    - 初始池由配置指定（如 ["NoviceAI", "MediumRuleAI"]）
    - 按 order 升序遍历使用
    - 每代最多新增 1 个个
    - 新增时需通过基因相似度过滤
    - 有限数量，不需要淘汰机制
    """

    def __init__(
        self,
        initial_pool: List[str],
        similarity_threshold: float = 0.85,
        max_add_per_gen: int = 1,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_add_per_gen = max_add_per_gen
        self._entries: List[BenchmarkPoolEntry] = []

        # 初始化内置 agent
        for i, name in enumerate(initial_pool):
            entry = BenchmarkPoolEntry(
                name=name,
                order=i,
                agent_type="builtin",
            )
            self._entries.append(entry)

        logger.info(
            f"BenchmarkPool initialized: {len(self._entries)} agents "
            f"({[e.name for e in self._entries]})"
        )

    @property
    def entries(self) -> List[BenchmarkPoolEntry]:
        """按 order 升序返回所有条目"""
        return sorted(self._entries, key=lambda e: e.order)

    @property
    def size(self) -> int:
        return len(self._entries)

    def try_add_individual(
        self,
        individual_id: str,
        generation: int,
        checkpoint_path: str,
        genome_weights: np.ndarray,
        tau: float,
        tau_threshold: float,
    ) -> bool:
        """
        尝试将个体加入基准池。

        条件：
        1. tau > tau_threshold（区分度高于 MediumRuleAI）
        2. cos 相似度 vs 池中所有已有个体 ≤ similarity_threshold

        Returns:
            True 表示成功加入
        """
        if tau <= tau_threshold:
            logger.debug(
                f"BenchmarkPool: {individual_id} tau={tau:.4f} <= threshold={tau_threshold}, skipped"
            )
            return False

        # 基因相似度过滤
        from ..genome.genome import cosine_similarity
        for entry in self._entries:
            if entry.genome_weights is not None:
                cos = cosine_similarity(genome_weights, entry.genome_weights)
                if cos > self.similarity_threshold:
                    logger.debug(
                        f"BenchmarkPool: {individual_id} too similar to {entry.name} "
                        f"(cos={cos:.4f} > {self.similarity_threshold}), skipped"
                    )
                    return False

        # 通过：加入池
        new_order = len(self._entries)
        entry = BenchmarkPoolEntry(
            name=f"ind_{individual_id}",
            order=new_order,
            agent_type="individual",
            individual_id=individual_id,
            generation=generation,
            checkpoint_path=checkpoint_path,
            genome_weights=genome_weights.copy(),
        )
        self._entries.append(entry)
        logger.info(
            f"BenchmarkPool: added {individual_id} at order={new_order} (tau={tau:.4f})"
        )
        return True

    def get_agent_factory(self, entry: BenchmarkPoolEntry):
        """根据条目类型返回 agent 工厂函数

        Returns:
            lambda player_id -> AntWarAgent
        """
        if entry.agent_type == "builtin":
            from ..battle.agent_loader import AgentLoader
            return lambda pid: AgentLoader.load(entry.name, pid)
        else:
            from ..battle.opponent_agent import OpponentAgent
            from ..battle.ppo_war_agent import PPOWarAgent
            from ..env.observation import ObservationEncoder
            from ..env.action_mask import ActionMaskHandler

            def factory(pid, cp=entry.checkpoint_path):
                agent = OpponentAgent.from_checkpoint(checkpoint_path=cp)
                return PPOWarAgent(
                    player_id=pid,
                    ppo_agent=agent,
                    observation_encoder=ObservationEncoder(),
                    action_mask_handler=ActionMaskHandler(),
                )
            return factory

    def save_state(self, filepath: str) -> None:
        """持久化到 JSON"""
        state = {
            "entries": []
        }
        for entry in self._entries:
            e = {
                "name": entry.name,
                "order": entry.order,
                "agent_type": entry.agent_type,
                "individual_id": entry.individual_id,
                "generation": entry.generation,
                "checkpoint_path": entry.checkpoint_path,
            }
            if entry.genome_weights is not None:
                e["genome_weights"] = entry.genome_weights.tolist()
            state["entries"].append(e)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"BenchmarkPool saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """从 JSON 加载"""
        with open(filepath, "r") as f:
            state = json.load(f)

        self._entries = []
        for e in state.get("entries", []):
            gw = None
            if "genome_weights" in e and e["genome_weights"] is not None:
                gw = np.array(e["genome_weights"], dtype=np.float32)
            entry = BenchmarkPoolEntry(
                name=e["name"],
                order=e["order"],
                agent_type=e["agent_type"],
                individual_id=e.get("individual_id", ""),
                generation=e.get("generation", -1),
                checkpoint_path=e.get("checkpoint_path", ""),
                genome_weights=gw,
            )
            self._entries.append(entry)
        logger.info(f"BenchmarkPool loaded from {filepath}: {len(self._entries)} entries")
```

---

### 3.5 `league/elite_seed_pool.py`

**功能**：精英种子池管理（仅做数据记录和持久化）。

```python
import json
import os
import hashlib
from typing import List
from dataclasses import dataclass, asdict

import numpy as np
from loguru import logger


@dataclass
class EliteSeedEntry:
    """精英种子池条目"""
    individual_id: str
    generation: int
    seed_rank: int
    trueskill_mu: float
    trueskill_sigma: float
    checkpoint_path: str
    genome_hash: str = ""


class EliteSeedPool:
    """精英种子池管理器

    仅做数据记录，不参与对战流程。用于离线分析模型迭代效果。
    """

    def __init__(self):
        self._entries: List[EliteSeedEntry] = []

    @property
    def entries(self) -> List[EliteSeedEntry]:
        return self._entries

    def add_seeds(
        self,
        seeds: list,           # List[Individual]
        elites: list,          # List[Individual]
        generation: int,
        checkpoint_dir: str,
    ) -> None:
        """批量添加本代 seeds 和 elites 到池中"""
        for seed in seeds:
            is_elite = seed in elites
            entry = EliteSeedEntry(
                individual_id=seed.id,
                generation=generation,
                seed_rank=seed.seed_rank,
                trueskill_mu=seed.trueskill_mu,
                trueskill_sigma=seed.trueskill_sigma,
                checkpoint_path=os.path.join(
                    checkpoint_dir, f"seed_{seed.seed_rank}.pt"
                ),
                genome_hash=hashlib.sha256(
                    seed.genome.weights.tobytes()
                ).hexdigest()[:16],
            )
            self._entries.append(entry)

        logger.info(
            f"EliteSeedPool: added {len(seeds)} seeds (elites={len(elites)}) "
            f"from gen {generation}, total={len(self._entries)}"
        )

    def save_state(self, filepath: str) -> None:
        """持久化到 JSON"""
        state = {
            "entries": [asdict(e) for e in self._entries]
        }
        # numpy 类型转换
        for e in state["entries"]:
            for k, v in e.items():
                if isinstance(v, np.floating):
                    e[k] = float(v)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"EliteSeedPool saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """从 JSON 加载"""
        with open(filepath, "r") as f:
            state = json.load(f)
        self._entries = [EliteSeedEntry(**e) for e in state.get("entries", [])]
        logger.info(f"EliteSeedPool loaded from {filepath}: {len(self._entries)} entries")
```

---

## 4. 修改文件详细设计

### 4.1 `evolution/population_manager.py`

**变更点**：

1. **Individual 数据类**（第 14-22 行）：
   - 删除 `elo_rating: float = 1200.0`
   - 新增 `trueskill_mu: float = 25.0`
   - 新增 `trueskill_sigma: float = 8.333`

2. **Population 数据类**（第 26-32 行）：
   - `best_elo` → `best_rating`
   - `mean_elo` → `mean_rating`
   - 注释更新

3. **`evolve()` 方法**（第 79-166 行）：
   - 第 121 行：`elo_rating=1200.0` → `trueskill_mu=25.0, trueskill_sigma=8.333`
   - 第 153 行：`elo_rating=1200.0` → `trueskill_mu=25.0, trueskill_sigma=8.333`

---

### 4.2 `selection/battle_selection.py`

**变更方式**：重写为薄包装层，委托给 `BenchmarkTest` 和 `RoundRobinTournament`。

```python
"""
PPO_V10 选拔模块 — 编排基准测试 + 循环赛两阶段流程。

在 v9 中，此文件直接执行 ELO 评估 + 种子选拔。
在 v10 中，它作为 facade，委托给 benchmark_test 和 round_robin 子模块。
"""

from typing import List, Dict, Optional
from loguru import logger

from ..evolution.population_manager import Individual, Population
from ..league.benchmark_pool import BenchmarkPool
from ..league.battle_shared_payoff import BattleSharedPayoff
from .benchmark_test import BenchmarkTest, BenchmarkTestConfig
from .round_robin import RoundRobinTournament, RoundRobinConfig


class BattleSelection:
    """对战选拔 — 编排基准测试 + 循环赛两阶段"""

    def __init__(
        self,
        benchmark_config: BenchmarkTestConfig,
        round_robin_config: RoundRobinConfig,
        benchmark_pool: BenchmarkPool,
        payoff: BattleSharedPayoff,
        dedup_config: dict,
        num_seeds: int = 8,
        elitism_count: int = 2,
    ):
        self._benchmark = BenchmarkTest(benchmark_config, benchmark_pool, payoff)
        self._round_robin = RoundRobinTournament(round_robin_config, payoff)
        self._payoff = payoff
        self._dedup_config = dedup_config
        self.num_seeds = num_seeds
        self.elitism_count = elitism_count

    def evaluate_and_select(
        self,
        population: Population,
    ) -> tuple:
        """
        执行完整的两阶段选拔。

        Returns:
            (ranked_all, seeds, elites)
            - ranked_all: 基准测试后全部个体的 TrueSkill 排名
            - seeds: 循环赛选出的种子列表
            - elites: 前 elitism_count 个种子
        """
        # ── 阶段 1：基准测试 ──
        logger.info("Phase: benchmark test")
        ranked_all = self._benchmark.evaluate(population)

        # ── 阶段 2：循环赛 ──
        logger.info("Phase: round robin")
        top_n = self._round_robin.config.top_n
        top_individuals = ranked_all[:top_n]

        seeds = self._round_robin.run(
            top_individuals=top_individuals,
            num_seeds=self.num_seeds,
            elitism_count=self.elitism_count,
            dedup_config=self._dedup_config,
        )[0]  # (seeds, elites)

        elites = seeds[:self.elitism_count]

        # 更新种群统计
        population.best_rating = ranked_all[0].trueskill_mu if ranked_all else 0.0
        mu_values = [ind.trueskill_mu for ind in ranked_all]
        population.mean_rating = sum(mu_values) / max(len(mu_values), 1)

        return ranked_all, seeds, elites
```

**删除的内容**（从 v9 版本）：
- `ReferenceAgent` 数据类及其所有使用
- `DEFAULT_ANCHOR_REFERENCES`
- `DEFAULT_ELO_MAP`
- `build_anchor_references()`
- `_flat_battle_worker()` — 移入 benchmark_test.py
- `evaluate_population()` — 被 benchmark_test.evaluate() 替代
- `select_seeds()` — 不再需要（仅 diverse 模式）
- `select_seeds_diverse()` — 移入 round_robin.py
- `_create_ga_agent()` — 各模块自行实现
- `ELORating` 导入 — 删除

---

### 4.3 `league/battle_shared_payoff.py`

**变更点**：小改，无需删除任何函数。

1. 新增辅助方法 `get_trueskill_mus()`:
```python
def get_trueskill_mus(self, player_ids: List[str]) -> Dict[str, float]:
    """批量获取多个玩家的 TrueSkill mu"""
    result = {}
    for pid in player_ids:
        rating = self._trueskill_ratings.get(pid, TrueSkillRating())
        result[pid] = rating.mu
    return result
```

2. 新增辅助方法 `get_trueskill_sigmas()`:
```python
def get_trueskill_sigmas(self, player_ids: List[str]) -> Dict[str, float]:
    """批量获取多个玩家的 TrueSkill sigma"""
    result = {}
    for pid in player_ids:
        rating = self._trueskill_ratings.get(pid, TrueSkillRating())
        result[pid] = rating.sigma
    return result
```

---

### 4.4 `trainer/genetic_evolution_trainer.py`

**变更方式**：重写主训练循环。保留 `__init__` 中大部分基础设施初始化。

**删除的初始化代码**：
- `self.elo` (ELORating) — 不再使用
- `self.opponent_pool` — 被 benchmark_pool 替代
- `self.opponent_selector` — 不再需要
- `self.battle_selection` — 创建方式改变（不再传 ELO 参数）

**新增的初始化代码**：
```python
# 基准对手池
from ..league.benchmark_pool import BenchmarkPool
self.benchmark_pool = BenchmarkPool(
    initial_pool=config["benchmark"]["initial_pool"],
    similarity_threshold=config["benchmark_pool"]["similarity_threshold"],
    max_add_per_gen=config["benchmark_pool"]["max_add_per_gen"],
)

# 精英种子池
from ..league.elite_seed_pool import EliteSeedPool
self.elite_seed_pool = EliteSeedPool()

# 选拔模块（重写）
from ..selection.benchmark_test import BenchmarkTestConfig
from ..selection.round_robin import RoundRobinConfig
benchmark_cfg = BenchmarkTestConfig(
    n_battles=config["benchmark"]["n_battles"],
    kendall_tau_threshold=config["benchmark"]["kendall_tau_threshold"],
    max_workers=config["benchmark"]["max_workers"],
    max_rounds=config["env"]["max_steps"],
    device=config["system"]["device"],
    hidden_dim=config["network"]["hidden_dim"],
    enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
)
rr_cfg = RoundRobinConfig(
    top_n=config["round_robin"]["top_n"],
    n_battles=config["round_robin"]["n_battles"],
    max_workers=config["round_robin"]["max_workers"],
    max_rounds=config["env"]["max_steps"],
    device=config["system"]["device"],
    hidden_dim=config["network"]["hidden_dim"],
    enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
)
self.battle_selection = BattleSelection(
    benchmark_config=benchmark_cfg,
    round_robin_config=rr_cfg,
    benchmark_pool=self.benchmark_pool,
    payoff=self.payoff,
    dedup_config=config.get("elite_dedup", {}),
    num_seeds=config["evolution"]["num_seeds"],
    elitism_count=config["evolution"]["elitism_count"],
)
```

**`train()` 方法新流程**：
```python
def train(self) -> None:
    for gen in range(1, max_generations + 1):
        # ── 阶段 1：种群生成（不变）──
        if gen == 1:
            population = self.population_mgr.init_population()
        else:
            population = self.population_mgr.evolve(...)

        # ── 阶段 2：个体选择（两阶段）──
        # 2a: 基准测试 + 2b: 循环赛（在 evaluate_and_select 内部）
        ranked_all, seeds, elites = self.battle_selection.evaluate_and_select(population)
        self._last_seeds = seeds

        # ── 阶段 3：基准对手池更新 ──
        self._update_benchmark_pool(seeds, elites, gen)

        # ── 阶段 4：精英种子池更新 ──
        seed_dir = os.path.join(self._generations_dir, f"gen_{gen:04d}", "seed_models")
        self.elite_seed_pool.add_seeds(seeds, elites, gen, seed_dir)

        # ── 阶段 5：诊断环节（基线验证，不变）──
        baseline_results = self.baseline_evaluator.evaluate_seeds(seeds, gen)

        # ── 阶段 6：日志与持久化 ──
        # 保存 seed_models
        for seed in seeds:
            self._save_seed_model(seed, gen)

        self.logging.log_generation(...)
        self._save_league_state()  # 保存 benchmark_pool + elite_seed_pool + payoff

        # 收敛判断（用 best_rating 替代 best_elo）
        if gen >= convergence_window:
            recent_best = [h for h in self._best_rating_history[-convergence_window:]]
            if max(recent_best) - min(recent_best) < convergence_threshold:
                logger.info(f"Converged at generation {gen}")
                break
```

**新增方法**：

```python
def _update_benchmark_pool(
    self,
    seeds: List[Individual],
    elites: List[Individual],
    generation: int,
) -> None:
    """
    基准对手池更新。

    1. 取 MediumRuleAI 在基准测试中的 τ 作为阈值
    2. 计算循环赛各精英的 Kendall τ
    3. 选 τ 最高且 > 阈值且通过基因过滤的 1 个加入池
    """
    # MediumRuleAI 的 τ 在 benchmark_test 中已计算并存储
    tau_threshold = self.battle_selection._benchmark._medium_rule_tau

    best_candidate = None
    best_tau = -1.0

    for ind in elites:  # 或遍历 seeds/循环赛所有个体
        # 计算个体在循环赛中的 τ
        tau_i = self._compute_individual_tau(ind)
        if tau_i > tau_threshold and tau_i > best_tau:
            best_candidate = ind
            best_tau = tau_i

    if best_candidate:
        checkpoint_path = self._save_seed_model(best_candidate, generation)
        self.benchmark_pool.try_add_individual(
            individual_id=best_candidate.id,
            generation=generation,
            checkpoint_path=checkpoint_path,
            genome_weights=best_candidate.genome.weights,
            tau=best_tau,
            tau_threshold=tau_threshold,
        )


def _compute_individual_tau(self, individual: Individual) -> float:
    """
    计算个体在循环赛中的 Kendall τ。

    用个体在循环赛中的对战结果排名 vs 循环赛 TrueSkill 排名。
    """
    pass


def _save_league_state(self) -> None:
    """持久化所有 league 状态"""
    os.makedirs(self._league_dir, exist_ok=True)
    self.benchmark_pool.save_state(os.path.join(self._league_dir, "benchmark_pool.json"))
    self.elite_seed_pool.save_state(os.path.join(self._league_dir, "elite_seed_pool.json"))
    self.payoff.save_state(os.path.join(self._league_dir, "payoff.json"))
    logger.info(
        f"League state saved "
        f"(benchmark_pool={self.benchmark_pool.size}, "
        f"elite_seed_pool={len(self.elite_seed_pool.entries)})"
    )
```

**删除的方法**：
- `_build_dynamic_references()` — 不再有动态参照物
- `_create_reference_from_checkpoint()` — 不再需要
- `_create_agent_from_individual()` — 移入各子模块

**保留（不变）的方法**：
- `_current_mutation_scale()`
- `_create_battle_config()`
- `_save_seed_model()`
- `_save_checkpoint()`

---

### 4.5 `trainer/checkpoint_manager.py`

**变更点**：

1. **`save()` 方法**（第 17-68 行）：
   - 第 42-48 行序列化 individual 时：
     - `elo_rating` → `trueskill_mu` + `trueskill_sigma`
   - 第 53-57 行 population_stats：
     - `best_elo` → `best_rating`
     - `mean_elo` → `mean_rating`
   - 第 65 行：删除 `opponent_pool` 和 `payoff` 参数（league 状态独立保存）

2. **`load()` 方法**（第 70-134 行）：
   - 第 109-117 行重建 Individual 时：
     - `elo_rating` → `trueskill_mu` + `trueskill_sigma`
   - 第 120-124 行重建 Population 时：
     - `best_elo` → `best_rating`（从 `population_stats` 取 `best_rating`）

具体修改：

```python
# save() 方法 — Individual 序列化部分
population_data.append({
    "id": ind.id,
    "generation": ind.generation,
    "genome_weights": ind.genome.weights,
    "genome_shapes": ind.genome.shapes,
    "genome_param_names": ind.genome.param_names,
    "trueskill_mu": ind.trueskill_mu,        # ← 变更
    "trueskill_sigma": ind.trueskill_sigma,  # ← 变更
    "seed_rank": ind.seed_rank,
    "parent_ids": ind.parent_ids,
    "mutation_desc": ind.mutation_desc,
})

# save() 方法 — population_stats
checkpoint = {
    ...
    "population_stats": {
        "best_rating": population.best_rating,    # ← 变更
        "mean_rating": population.mean_rating,    # ← 变更
        "diversity": population.diversity,
    },
    ...
}

# load() 方法 — Individual 重建
individual = Individual(
    id=ind_data["id"],
    generation=ind_data["generation"],
    genome=genome,
    trueskill_mu=ind_data.get("trueskill_mu", 25.0),      # ← 变更
    trueskill_sigma=ind_data.get("trueskill_sigma", 8.333), # ← 变更
    seed_rank=ind_data["seed_rank"],
    parent_ids=ind_data["parent_ids"],
    mutation_desc=ind_data["mutation_desc"],
)

# load() 方法 — Population 重建
population = Population(
    generation=checkpoint["generation"],
    individuals=individuals,
    best_rating=checkpoint.get("population_stats", {}).get("best_rating", 0.0),  # ← 变更
    mean_rating=checkpoint.get("population_stats", {}).get("mean_rating", 0.0),  # ← 变更
    diversity=checkpoint.get("population_stats", {}).get("diversity", 0.0),
)
```

---

### 4.6 `trainer/logging_subsystem.py`

**变更点**：

1. **`log_generation()` 方法**（第 79-109 行）：
   - 第 90-95 行日志输出：`best_elo` → `best_rating`，`mean_elo` → `mean_rating`
   - 第 98 行 `_save_population_stats()` 调用不变

2. **`_save_population_stats()` 方法**（第 135-155 行）：
   - `best_elo` → `best_rating`
   - `mean_elo` → `mean_rating`
   - `seed_elo_range` → `seed_rating_range`（取 trueskill_mu）
   - 第 148 行：`seeds[-1].elo_rating` → `seeds[-1].trueskill_mu`
   - 第 149 行：`seeds[0].elo_rating` → `seeds[0].trueskill_mu`

3. **`_save_battle_results()` 方法**（第 157-180 行）：
   - 第 163 行：`"elo": ind.elo_rating` → `"trueskill_mu": ind.trueskill_mu`
   - 第 168 行：`"elo": s.elo_rating` → `"trueskill_mu": s.trueskill_mu`
   - 排序键：`x.elo_rating` → `x.trueskill_mu`

4. **`_write_tensorboard()` 方法**（第 191-211 行）：
   - `ga/best_elo` → `ga/best_rating`
   - `ga/mean_elo` → `ga/mean_rating`

5. **新增日志字段**：
   - `_save_population_stats()` 中新增：
     ```python
     "best_trueskill_mu": population.best_rating,
     "mean_trueskill_mu": population.mean_rating,
     "num_benchmark_agents_used": ...,    # 本代基准测试使用了多少个 agent
     "benchmark_kendall_tau_final": ...,  # 最终停止时的 τ 值
     ```

---

### 4.7 `config/config_parser.py`

**变更点**：

1. **`_STRUCTURAL_DEFAULTS` 字典**（第 18-49 行）：

删除以下默认值：
```python
# 删除
"selection": {
    "method": "fixed_reference_elo",
    "anchor_references": ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"],
    "dynamic_references": 2,
    "include_previous_champion": True,
    "elo_k_factor": 64,
    "elo_initial": 1200,
},
"opponent_pool": {
    "max_size": 20,
    "min_games_threshold": 8,
    "exploit_prob": 0.7,
    "decay": 0.99,
    "min_win_rate_games": 8,
},
```

新增以下默认值：
```python
"benchmark": {
    "n_battles": 2,
    "kendall_tau_threshold": 0.7,
    "max_workers": 25,
    "initial_pool": ["NoviceAI", "MediumRuleAI"],
},
"round_robin": {
    "top_n": 24,
    "n_battles": 4,
    "max_workers": 25,
},
"benchmark_pool": {
    "max_add_per_gen": 1,
    "similarity_threshold": 0.85,
},
```

2. **`_TYPE_MAPPING` 字典**（第 51-77 行）：

删除以下类型映射：
```python
"selection.elo_k_factor": float,
"selection.elo_initial": float,
"selection.dynamic_references": int,
"opponent_pool.max_size": int,
"opponent_pool.min_games_threshold": int,
"opponent_pool.exploit_prob": float,
"opponent_pool.decay": float,
```

新增以下类型映射：
```python
"benchmark.n_battles": int,
"benchmark.kendall_tau_threshold": float,
"benchmark.max_workers": int,
"round_robin.top_n": int,
"round_robin.n_battles": int,
"round_robin.max_workers": int,
"benchmark_pool.max_add_per_gen": int,
"benchmark_pool.similarity_threshold": float,
```

3. 在 `_deep_merge` 逻辑不变。

---

### 4.8 `train.py` 和 `train_ga.py`

**变更点**：删除不再需要的 CLI 参数。

`train.py` — 删除（这些参数在 v9 中就被静默忽略，v10 彻底移除）：
- `--episodes` / `--batch-size` / `--n-envs` 等 PPO 参数（已被静默忽略）

`train_ga.py` — 删除：
- 无需删除，保留 `--population_size`、`--max_generations`、`--num_seeds`、`--mutation_rate`、`--mutation_scale`、`--crossover_rate`、`--device`、`--resume`、`--config`

---

### 4.9 配置文件 `configs/ga_evo_v10.yaml`

完整新配置文件：

```yaml
# ============================================================
# GA 进化训练配置文件 - AntWar v10
# ============================================================

# 系统配置
system:
  device: "cuda"
  seed: 42

# 网络结构配置
network:
  hidden_dim: 256
  enable_auxiliary: true

# 进化参数
evolution:
  max_generations: 200
  population_size: 80
  num_seeds: 8
  elitism_count: 2
  convergence_window: 20
  convergence_threshold: 5.0

# 杂交参数
crossover:
  method: "layer_wise"
  rate: 0.8

# 变异参数
mutation:
  method: "gaussian"
  rate: 0.1
  scale: 0.03
  decay: 0.99

# 基准测试参数
benchmark:
  n_battles: 2
  kendall_tau_threshold: 0.7
  max_workers: 25
  initial_pool:
    - "NoviceAI"
    - "MediumRuleAI"

# 循环赛参数
round_robin:
  top_n: 24
  n_battles: 4
  max_workers: 25

# 基因相似度过滤
elite_dedup:
  enabled: true
  hard_threshold: 0.95
  soft_threshold: 0.85
  soft_max_per_family: 2

# 基准对手池参数
benchmark_pool:
  max_add_per_gen: 1
  similarity_threshold: 0.85

# 基线验证参数（诊断环节）
baseline_battle:
  enabled: true
  interval: 1
  n_battles: 16
  max_workers: 25
  agents:
    - "MediumRuleAI"
    - "BasicTowerAI"
    - "nn_gen_32_cuda"
    - "rule_zzy25"

# 环境参数
env:
  player_id: 0
  backend_type: "python"
  max_steps: 512

# 日志参数
logging:
  log_level: "INFO"
  save_interval: 10
  tensorboard: true
```

---

## 5. 函数级变更汇总

### 5.1 删除的函数

| 函数 | 所在文件 | 原因 |
|------|---------|------|
| `ELORating.__init__` | `elo_rating.py` | 整个文件删除 |
| `ELORating.ensure_player` | `elo_rating.py` | 同上 |
| `ELORating.update` | `elo_rating.py` | 同上 |
| `ELORating.get_rating` | `elo_rating.py` | 同上 |
| `ELORating.reset_individuals` | `elo_rating.py` | 同上 |
| `ELORating.get_dynamic_ref_rating` | `elo_rating.py` | 同上 |
| `ELORating.save_dynamic_ref_rating` | `elo_rating.py` | 同上 |
| `OpponentPool.__init__` | `opponent_pool.py` | 整个文件删除 |
| `OpponentPool.add` | `opponent_pool.py` | 同上 |
| `OpponentPool._evict_worst` | `opponent_pool.py` | 同上 |
| `OpponentPool.increment_games` | `opponent_pool.py` | 同上 |
| `OpponentPool.get_checkpoint_path` | `opponent_pool.py` | 同上 |
| `OpponentPool.get_all` | `opponent_pool.py` | 同上 |
| `OpponentPool.save_state` | `opponent_pool.py` | 同上 |
| `OpponentPool.load_state` | `opponent_pool.py` | 同上 |
| `OpponentSelector.__init__` | `opponent_selector.py` | 整个文件删除 |
| `OpponentSelector.select` | `opponent_selector.py` | 同上 |
| `OpponentSelector.select_batch` | `opponent_selector.py` | 同上 |
| `OpponentSelector._exploit_select` | `opponent_selector.py` | 同上 |
| `OpponentSelector._explore_select` | `opponent_selector.py` | 同上 |
| `OpponentSelector._compute_exploit_prob` | `opponent_selector.py` | 同上 |
| `OpponentSelector.update_progress` | `opponent_selector.py` | 同上 |
| `build_anchor_references` | `battle_selection.py` | 不再有锚定参照物 |
| `_flat_battle_worker` | `battle_selection.py` | 移入 benchmark_test.py |
| `BattleSelection.evaluate_population` | `battle_selection.py` | 被 BenchmarkTest.evaluate 替代 |
| `BattleSelection.select_seeds` | `battle_selection.py` | 不再需要（仅 diverse） |
| `BattleSelection.select_seeds_diverse` | `battle_selection.py` | 移入 round_robin.py |
| `BattleSelection._create_ga_agent` | `battle_selection.py` | 各子模块自行实现 |
| `GeneticEvolutionTrainer._build_dynamic_references` | `genetic_evolution_trainer.py` | 不再有动态参照物 |
| `GeneticEvolutionTrainer._create_reference_from_checkpoint` | `genetic_evolution_trainer.py` | 不再需要 |
| `GeneticEvolutionTrainer._create_agent_from_individual` | `genetic_evolution_trainer.py` | 移入子模块 |

### 5.2 新增的函数

| 函数 | 所在文件 | 说明 |
|------|---------|------|
| `kendall_tau` | `kendall_tau.py` | Kendall τ 相关系数 |
| `ranking_from_trueskill` | `kendall_tau.py` | TrueSkill 排名生成 |
| `ranking_from_head_to_head` | `kendall_tau.py` | 对战结果排名生成 |
| `_benchmark_battle_worker` | `benchmark_test.py` | 子进程对战 worker |
| `BenchmarkTest.__init__` | `benchmark_test.py` | 构造函数 |
| `BenchmarkTest.evaluate` | `benchmark_test.py` | 基准测试主流程 |
| `BenchmarkTest._compute_tau_for_benchmark_agent` | `benchmark_test.py` | 单 agent τ 计算 |
| `BenchmarkTest._create_ga_agent` | `benchmark_test.py` | 创建 GA agent |
| `_rr_battle_worker` | `round_robin.py` | 子进程循环赛 worker |
| `RoundRobinTournament.__init__` | `round_robin.py` | 构造函数 |
| `RoundRobinTournament.run` | `round_robin.py` | 循环赛主流程 |
| `RoundRobinTournament._generate_matchups` | `round_robin.py` | 生成对战配对 |
| `RoundRobinTournament.select_seeds_diverse` | `round_robin.py` | 多样性种子选拔 |
| `RoundRobinTournament._create_ga_agent` | `round_robin.py` | 创建 GA agent |
| `BenchmarkPool.__init__` | `benchmark_pool.py` | 构造函数 |
| `BenchmarkPool.try_add_individual` | `benchmark_pool.py` | 尝试入池 |
| `BenchmarkPool.get_agent_factory` | `benchmark_pool.py` | 获取 agent 工厂 |
| `BenchmarkPool.save_state` | `benchmark_pool.py` | 持久化 |
| `BenchmarkPool.load_state` | `benchmark_pool.py` | 加载 |
| `EliteSeedPool.__init__` | `elite_seed_pool.py` | 构造函数 |
| `EliteSeedPool.add_seeds` | `elite_seed_pool.py` | 批量添加种子 |
| `EliteSeedPool.save_state` | `elite_seed_pool.py` | 持久化 |
| `EliteSeedPool.load_state` | `elite_seed_pool.py` | 加载 |
| `BattleSelection.__init__` (重写) | `battle_selection.py` | 接收新参数 |
| `BattleSelection.evaluate_and_select` | `battle_selection.py` | 两阶段选拔 |
| `BattleSharedPayoff.get_trueskill_mus` | `battle_shared_payoff.py` | 批量获取 mu |
| `BattleSharedPayoff.get_trueskill_sigmas` | `battle_shared_payoff.py` | 批量获取 sigma |
| `GeneticEvolutionTrainer._update_benchmark_pool` | `genetic_evolution_trainer.py` | 基准池更新 |
| `GeneticEvolutionTrainer._compute_individual_tau` | `genetic_evolution_trainer.py` | 个体 τ 计算 |
| `GeneticEvolutionTrainer._save_league_state` (重写) | `genetic_evolution_trainer.py` | 保存双池状态 |

### 5.3 修改的函数

| 函数 | 所在文件 | 变更内容 |
|------|---------|---------|
| `Individual.__init__` (dataclass) | `population_manager.py` | `elo_rating` → `trueskill_mu + trueskill_sigma` |
| `Population.__init__` (dataclass) | `population_manager.py` | `best_elo/mean_elo` → `best_rating/mean_rating` |
| `PopulationManager.evolve` | `population_manager.py` | 新 Individual 字段名适配 |
| `BattleSharedPayoff.*` | `battle_shared_payoff.py` | 新增 2 个辅助方法，其余不变 |
| `GACheckpointManager.save` | `checkpoint_manager.py` | ELO→TrueSkill 字段序列化 |
| `GACheckpointManager.load` | `checkpoint_manager.py` | ELO→TrueSkill 字段反序列化 |
| `GALoggingSubsystem.log_generation` | `logging_subsystem.py` | 日志输出字段名变更 |
| `GALoggingSubsystem._save_population_stats` | `logging_subsystem.py` | 统计字段变更 |
| `GALoggingSubsystem._save_battle_results` | `logging_subsystem.py` | elo→trueskill_mu |
| `GALoggingSubsystem._write_tensorboard` | `logging_subsystem.py` | TensorBoard 指标名变更 |
| `GAConfigParser._merge_structural_defaults` | `config_parser.py` | 新增/删除默认配置段 |
| `GAConfigParser._validate_types` (via `_TYPE_MAPPING`) | `config_parser.py` | 类型映射更新 |
| `GeneticEvolutionTrainer.__init__` | `genetic_evolution_trainer.py` | 新增双池初始化，删除旧池 |
| `GeneticEvolutionTrainer.train` | `genetic_evolution_trainer.py` | 新主线流程 |

---

## 6. 数据流图

```
每代数据流:

Population (N 个体)
    │
    ├─→ BenchmarkTest.evaluate()
    │   ├─ for each BenchmarkPoolEntry (按 order):
    │   │   ├─ 全部个体 vs entry agent (n_battle=2)
    │   │   ├─ payoff.update() → TrueSkill
    │   │   ├─ kendall_tau(R_base, R_ref)
    │   │   └─ if τ > threshold → break
    │   └─→ 回写 individual.trueskill_mu/sigma
    │       → 输出 ranked_all (按 mu 降序)
    │
    ├─→ RoundRobinTournament.run(top_N)
    │   ├─ 全圆桌对战 (N*(N-1)/2 配对 × n_battle)
    │   ├─ payoff.update() → TrueSkill
    │   ├─ select_seeds_diverse() (cos 去重)
    │   └─→ (seeds, elites)
    │
    ├─→ _update_benchmark_pool()
    │   ├─ 取 MediumRuleAI τ 作为阈值
    │   ├─ 对 seeds/elites 计算 τ
    │   ├─ cos 相似度过滤 vs 池中个体
    │   └─ 选 τ 最高且通过过滤的 1 个入池
    │
    ├─→ elite_seed_pool.add_seeds()
    │
    ├─→ baseline_evaluator.evaluate_seeds()
    │
    └─→ logging + checkpoint
```

---

## 7. 实施注意事项

1. **代码复用优先级**：`_flat_battle_worker`、`_reconfigure_loguru`、`_execute_battle`、`_ensure_compatible` 等子进程相关函数在 v9 `battle_selection.py` 中，v10 需要提取到共享位置或直接在新模块中复制（遵循"不过度设计"原则，倾向于直接复制）。

2. **`_reconfigure_loguru()`**：在 v9 的 `battle_selection.py` 中定义。v10 的 `benchmark_test.py` 和 `round_robin.py` 都需要它。**方案**：在各自模块中复制定义（只有 10 行代码），或提取到 `battle/utils/` 中。

3. **现有 v9 配置文件处理**：`configs/ga_evo.yaml` 等 7 个 v9 配置文件不移除，v10 使用新的 `ga_evo_v10.yaml`。

4. **tool 脚本**：`tools/` 目录下的工具脚本可能引用 ELO 字段或旧类名，需要后续单独检查适配。

5. **`BattleSharedPayoff` 使用方式**：在 v10 中，同一个 `payoff` 实例在两个阶段（基准测试、循环赛）中共享使用，TrueSkill 评分跨阶段累积。这符合"从基准测试继承"的设计（Open Question 9.7）。

6. **Kendall τ 在基准测试中的 R_ref**：第 1 个 benchmark_agent 没有参考排名。实现方案：
   - 第 1 个 agent：`τ = 0.0`（或直接跳过 τ 检查，不早停）
   - 第 2 个及以后：R_ref = 累计 TrueSkill mu 排名

7. **收敛判断**：`convergence_threshold` 由 ELO 变动改为 best TrueSkill mu 变动。默认值 5.0 可能需要根据 TrueSkill 的 mu 范围（~0-50）重新校准。
