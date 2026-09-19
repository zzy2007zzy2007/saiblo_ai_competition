"""TouchstoneSelector —— 基于 original_rule.txt 的二分种子基准筛选阶段。

设计要点：
  • 完全替换原 BenchmarkTest（外部 BenchmarkPool 全员打分）。
  • 不依赖外部 benchmark agents，纯种群内部互打 → mu 排名 → 选 Top16。
  • 算法严格对齐 original_rule.txt 的 14 条规则：复用
    [tools/original_rule_experiment/coordinator.py] 的 `OriginalRuleCoordinator`
    设计，迁移到 src 内并把 SimplePayoff 适配为 BattleSharedPayoff。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from ..battle.agent_loader import AgentLoader
from ..battle.ga_war_agent import GAWarAgent
from ..evolution.population_manager import Individual, Population
from ..genome.genome import load_genome_to_network
from ..league.battle_shared_payoff import BattleSharedPayoff
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork

from ._battle_executor import BattleExecutor, Task, TaskQueue


# ─── 配置 ──────────────────────────────────────────────────────────────


@dataclass
class TouchstoneConfig:
    """二分种子选拔配置"""
    M: int = 16                          # 升级所需最少对手数
    max_seeds: int = 3                   # 二分种子数上限
    n_battle: int = 1                    # 批量阶段每对 2×n_battle 局
    max_candidate_fails: int = 12        # 累计候选失败上限
    max_workers: int = 25                # 并行执行 workers 数量
    discovery_n_games: int = 2           # 甄别 task 内对局数（先后手各 1 局）
    max_rounds: int = 512
    device: str = "cuda"
    hidden_dim: int = 256
    # 方案 A：按精确排名数字判定（1=最强）
    candidate_min_rank: int = 30         # 候选升级：最低排名要求（含），即不能弱于第 N 名
    candidate_max_rank: int = 15         # 候选升级：最高排名要求（含），即不能强于第 N 名
    next_candidate_rank: int = 20        # 选择下一候选：从剩余已对战个体中选第 N 名（1=最强）

    def __post_init__(self):
        if self.candidate_min_rank <= self.candidate_max_rank:
            raise ValueError(
                f"TouchstoneConfig: candidate_min_rank({self.candidate_min_rank}) "
                f"must be > candidate_max_rank({self.candidate_max_rank})"
            )
        if self.next_candidate_rank < 1:
            raise ValueError(
                f"TouchstoneConfig: next_candidate_rank({self.next_candidate_rank}) "
                f"must be >= 1"
            )


# ─── 候选状态 ──────────────────────────────────────────────────────────


@dataclass
class _CandidateState:
    id: str
    start_time: float
    completed_battles: int = 0
    pending_opponents: List[str] = field(default_factory=list)
    resolved: bool = False
    is_seed: bool = False


# ─── 协调器 ────────────────────────────────────────────────────────────


class _TouchstoneCoordinator:
    """严格对齐 original_rule.txt 的二分种子协调器。

    与实验版 OriginalRuleCoordinator 同构，仅：
      • payoff: SimplePayoff → BattleSharedPayoff
      • _battle_count: 自维护（BattleSharedPayoff 不暴露 per-player count）
    """

    def __init__(
        self,
        config: TouchstoneConfig,
        payoff: BattleSharedPayoff,
        queue: TaskQueue,
        generation: int,
    ):
        self.config = config
        self.payoff = payoff
        self.queue = queue
        self.generation = generation

        self.seeds: List[str] = []
        self.candidates: Dict[str, _CandidateState] = {}
        self.population_ids: List[str] = []
        self.cumulative_fails: int = 0
        self.battled_pairs: Set[Tuple[str, str]] = set()
        self.pair_battle_count: Dict[Tuple[str, str], int] = {}
        self._battle_count: Dict[str, int] = {}
        self._fallback_triggered: bool = False
        self._all_seeds_found: bool = False
        # 逐种子批量：记录当前种子的 batch 对子集合，用于检测 batch 完成
        self._batch_pairs: Set[Tuple[str, str]] = set()

        self._stats = {
            "candidates_launched": 0,
            "candidates_promoted": 0,
            "candidates_failed": 0,
            "discovery_tasks_submitted": 0,
            "batch_tasks_submitted": 0,
            "cancelled_tasks_on_promote": 0,
            "cancelled_tasks_on_fail": 0,
            "cancelled_tasks_on_final": 0,
            "battles_class1": 0,
            "battles_class2": 0,
            "battles_class3": 0,
            "battles_total": 0,
            "battles_effective": 0,
        }
        self._all_battles: List[Tuple[str, str]] = []
        self._battle_results: List[dict] = []
        self._launched_candidate_set: Set[str] = set()

    # ── 公开属性 ──

    @property
    def is_fallback(self) -> bool:
        return self._fallback_triggered

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def unique_candidates_count(self) -> int:
        return len(self._launched_candidate_set)

    def get_battle_count(self, player_id: str) -> int:
        return self._battle_count.get(player_id, 0)

    # ── 公开接口 ──

    def start(self, population_ids: List[str]) -> None:
        """规则 1：第 1 个个体作为初始候选"""
        self.population_ids = list(population_ids)
        n = len(self.population_ids)


        if n < 6:
            logger.warning(
                f"[phase=touchstone][gen={self.generation}] fallback: "
                f"reason=too_few_individuals, individuals={n}, min_individuals=6"
            )
            self._fallback()
            return

        for iid in self.population_ids:
            self.payoff.ensure_player(iid)

        self._launch_candidate(self.population_ids[0])

    def on_battle_complete(self, home: str, away: str, result: dict) -> None:
        """每局对战完成后回调（主线程串行执行，无需锁）。"""
        self._stats["battles_total"] += 1
        self._all_battles.append((home, away))

        seed_set = set(self.seeds)
        cand_active = {cid for cid, st in self.candidates.items() if not st.resolved}
        cand_active -= seed_set

        if home in seed_set or away in seed_set:
            self._stats["battles_class2"] += 1
        elif home in cand_active or away in cand_active:
            self._stats["battles_class1"] += 1

        # 规则 3：更新双边 TrueSkill
        result_code = result.get("result_code", 0)
        self.payoff.update(home, away, result_code)

        # 收集对战结果（用于持久化）
        battle_entry = {
            "home": home,
            "away": away,
            "result_code": result_code,
            "result": result.get("result"),
            "first_player": result.get("first_player"),
            "total_rounds": result.get("total_rounds"),
            "illegal_home": result.get("illegal_actions_agent1", 0),
            "illegal_away": result.get("illegal_actions_agent2", 0),
        }
        # 仅对涉及种子的对局保存回合详情
        if home in seed_set or away in seed_set:
            rounds_data = result.get("rounds", [])
            if rounds_data:
                battle_entry["rounds"] = [
                    {
                        "round": rd.get("round"),
                        "a1_ops": rd.get("agent1_ops"),
                        "a2_ops": rd.get("agent2_ops"),
                        "a1_coins": rd.get("agent1_coins_before"),
                        "a2_coins": rd.get("agent2_coins_before"),
                        "a1_hp": rd.get("agent1_hp_before"),
                        "a2_hp": rd.get("agent2_hp_before"),
                        "a1_towers": rd.get("agent1_towers"),
                        "a2_towers": rd.get("agent2_towers"),
                        "a1_hp_dmg": rd.get("agent1_hp_dmg"),
                        "a2_hp_dmg": rd.get("agent2_hp_dmg"),
                        "a1_ops_cost": rd.get("agent1_ops_cost"),
                        "a2_ops_cost": rd.get("agent2_ops_cost"),
                        "a1_illegal": rd.get("agent1_illegal"),
                        "a2_illegal": rd.get("agent2_illegal"),
                    }
                    for rd in rounds_data
                ]
        self._battle_results.append(battle_entry)

        # 自维护 battle_count
        self._battle_count[home] = self._battle_count.get(home, 0) + 1
        self._battle_count[away] = self._battle_count.get(away, 0) + 1

        pair = tuple(sorted([home, away]))
        self.battled_pairs.add(pair)
        self.pair_battle_count[pair] = self.pair_battle_count.get(pair, 0) + 1

        # 更新候选状态并触发判定
        for pid in (home, away):
            if pid in self.candidates:
                state = self.candidates[pid]
                if state.resolved:
                    continue
                state.completed_battles += 1
                opp = home if pid == away else away
                if opp in state.pending_opponents:
                    state.pending_opponents.remove(opp)
                self._check_candidate(pid)

        # 逐种子批量完成检测：当前种子的 batch 全部打完 → 选下一个候选
        if self._batch_pairs and not self._all_seeds_found:
            target = self.config.n_battle * 2
            all_done = all(
                self.pair_battle_count.get(p, 0) >= target
                for p in self._batch_pairs
            )
            if all_done:
                logger.info(
                    f"[phase=touchstone][gen={self.generation}] seed_batch_done: "
                    f"pairs={len(self._batch_pairs)}, seeds={len(self.seeds)}/{self.config.max_seeds}"
                )
                self._batch_pairs.clear()
                self._pick_next_candidate()

    def finalize_battle_classification(self) -> None:
        final_seeds = set(self.seeds)
        class3 = 0
        for (home, away) in self._all_battles:
            if home not in final_seeds and away not in final_seeds:
                class3 += 1
        self._stats["battles_class3"] = class3
        self._stats["battles_effective"] = self._stats["battles_total"] - class3

    def save_battle_results(self, gen_dir: str) -> None:
        """保存 Touchstone 对战结果到 JSON 文件。

        包含所有对局的胜负信息，以及涉及种子的对局回合详情。
        格式与 baseline _rounds_sample 对齐。
        """
        import json
        import os
        os.makedirs(gen_dir, exist_ok=True)
        filepath = os.path.join(gen_dir, "touchstone_battles.json")
        try:
            with open(filepath, "w") as f:
                json.dump({
                    "seeds": self.seeds,
                    "total_battles": len(self._battle_results),
                    "battles_with_rounds": sum(1 for b in self._battle_results if "rounds" in b),
                    "battles": self._battle_results,
                }, f, indent=2, default=str)
            logger.info(
                f"Touchstone battle results saved: {len(self._battle_results)} battles "
                f"({sum(1 for b in self._battle_results if 'rounds' in b)} with round data) "
                f"→ {filepath}"
            )
        except Exception as e:
            logger.error(f"Failed to save touchstone battle results: {e}")

    def get_ranking_by_mu(self) -> List[str]:
        battled_ids = self._get_battled_player_ids()
        return sorted(
            battled_ids,
            key=lambda iid: self.payoff.get_trueskill_mus([iid]).get(iid, 25.0),
            reverse=True,
        )

    # ── 内部 ──

    def _launch_candidate(self, candidate_id: str) -> None:
        """规则 2/6：候选发起 M 个 task，优先选 battle_count > 0 的对手"""
        state = _CandidateState(id=candidate_id, start_time=time.time())
        self.candidates[candidate_id] = state
        self._stats["candidates_launched"] += 1
        self._launched_candidate_set.add(candidate_id)

        opponents = self._order_opponents(candidate_id)[: self.config.M]

        for opp_id in opponents:
            task = Task(
                task_id=f"disc_{candidate_id}_vs_{opp_id}",
                player_a=candidate_id,
                player_b=opp_id,
                n_games=self.config.discovery_n_games,
                priority=0,
                creator=candidate_id,
                created_at=time.time(),
            )
            self.queue.submit(task)
            state.pending_opponents.append(opp_id)
            self._stats["discovery_tasks_submitted"] += 1

        mu = self.payoff.get_trueskill_mus([candidate_id]).get(candidate_id, 25.0)
        logger.info(
            f"[phase=touchstone][gen={self.generation}] candidate_start: "
            f"candidate={candidate_id}, mu={mu:.2f}, tasks={len(opponents)}"
        )

    def _check_candidate(self, candidate_id: str) -> None:
        """检查候选是否升级或失败。

        按可配置的精确排名判定（1=最强）：
        - candidate_min_rank: 候选不能弱于该排名（mu >= 该排名位置的 mu）
        - candidate_max_rank: 候选不能强于该排名（mu <= 该排名位置的 mu）
        - 边界存在并列时，以并列组的 mu 值作为判定边界（灵活处理并列）
        """
        state = self.candidates.get(candidate_id)
        if not state or state.resolved:
            return

        if state.completed_battles < self.config.M:
            return

        mu = self.payoff.get_trueskill_mus([candidate_id]).get(candidate_id, 25.0)
        battled_ids = self._get_battled_player_ids()
        m = len(battled_ids)
        # 数据太少或不足以覆盖排名范围时跳过
        if m < 3 or m < self.config.candidate_min_rank:
            return

        mu_dict = self.payoff.get_trueskill_mus(battled_ids)
        battled_mus_asc = sorted(mu_dict.values())

        # 排名 → 升序数组位置：rank R（1=最强）→ 位置 m - R
        lo_pos = m - self.config.candidate_min_rank   # 降级下界
        hi_pos = m - self.config.candidate_max_rank   # 降级上界
        # 上界受限于实际人数
        hi_pos = max(hi_pos, 0)

        # 并列处理：lo 边界向左扩展，hi 边界向右扩展，将整个并列组纳入
        lo = battled_mus_asc[lo_pos]
        while lo_pos > 0 and battled_mus_asc[lo_pos - 1] == lo:
            lo_pos -= 1
        lo = battled_mus_asc[lo_pos]

        hi = battled_mus_asc[hi_pos]
        while hi_pos < m - 1 and battled_mus_asc[hi_pos + 1] == hi:
            hi_pos += 1
        hi = battled_mus_asc[hi_pos]

        if lo <= mu <= hi:
            self._promote_to_seed(candidate_id)
        else:
            self._fail_candidate(candidate_id)

    def _get_battled_player_ids(self) -> List[str]:
        return [pid for pid in self.population_ids if self._battle_count.get(pid, 0) > 0]

    def _promote_to_seed(self, candidate_id: str) -> None:
        state = self.candidates[candidate_id]
        state.resolved = True
        state.is_seed = True
        self.seeds.append(candidate_id)
        self._stats["candidates_promoted"] += 1

        mu = self.payoff.get_trueskill_mus([candidate_id]).get(candidate_id, 25.0)
        logger.info(
            f"[phase=touchstone][gen={self.generation}] candidate_promoted: "
            f"candidate={candidate_id}, mu={mu:.2f}, "
            f"seeds={len(self.seeds)}/{self.config.max_seeds}"
        )

        n = self.queue.cancel_by_creator(candidate_id)
        self._stats["cancelled_tasks_on_promote"] += n

        if len(self.seeds) >= self.config.max_seeds:
            self._submit_batch_battles()
            self._all_seeds_found = True
            n = self.queue.cancel_battles_without_seed(self.seeds)
            self._stats["cancelled_tasks_on_final"] += n
            for cid in list(self.candidates.keys()):
                if not self.candidates[cid].resolved:
                    self.candidates[cid].resolved = True
            logger.info(
                f"[phase=touchstone][gen={self.generation}] seeds_finalized: "
                f"seeds={len(self.seeds)}, cancelled_non_seed_tasks={n}"
            )
        else:
            # 逐种子批量：立即提交该种子 vs 全体的对战，
            # 完成后由 on_battle_complete 自动触发下一候选的选拔
            self._submit_batch_battles(seed_id=candidate_id)
            logger.info(
                f"[phase=touchstone][gen={self.generation}] seed_batch_started: "
                f"seed={candidate_id}, pairs={len(self._batch_pairs)}, "
                f"seed_progress={len(self.seeds)}/{self.config.max_seeds}"
            )

    def _submit_batch_battles(self, seed_id: Optional[str] = None) -> None:
        """提交批量对战。

        - seed_id=None: 为所有已确认种子提交批量对战（最终批量）
        - seed_id 指定: 仅为该种子提交批量对战（逐种子模式）
        """
        n_battle = self.config.n_battle
        target_games = 2 * n_battle
        submitted_pairs: Set[Tuple[str, str]] = set()
        candidates = [seed_id] if seed_id is not None else self.seeds

        for s_id in candidates:
            for opp_id in self.population_ids:
                if opp_id == s_id:
                    continue
                pair = tuple(sorted([s_id, opp_id]))
                if pair in submitted_pairs:
                    continue
                already = self.pair_battle_count.get(pair, 0)
                remaining = target_games - already
                if remaining <= 0:
                    continue
                submitted_pairs.add(pair)

                task = Task(
                    task_id=f"batch_{s_id}_vs_{opp_id}",
                    player_a=s_id,
                    player_b=opp_id,
                    n_games=remaining,
                    priority=1,
                    creator=f"batch_{s_id}",
                    created_at=time.time(),
                )
                self.queue.submit(task)
                self._stats["batch_tasks_submitted"] += 1

        # 逐种子模式：记录 batch 对子，用于 on_battle_complete 中检测完成
        if seed_id is not None:
            self._batch_pairs = submitted_pairs.copy()

    def _fail_candidate(self, candidate_id: str) -> None:
        state = self.candidates[candidate_id]
        state.resolved = True
        n = self.queue.cancel_by_creator(candidate_id)
        self._stats["cancelled_tasks_on_fail"] += n
        self._stats["candidates_failed"] += 1
        self.cumulative_fails += 1

        mu = self.payoff.get_trueskill_mus([candidate_id]).get(candidate_id, 25.0)
        logger.info(
            f"[phase=touchstone][gen={self.generation}] candidate_failed: "
            f"candidate={candidate_id}, mu={mu:.2f}, "
            f"completed={state.completed_battles}, "
            f"cumulative_fails={self.cumulative_fails}/{self.config.max_candidate_fails}"
        )

        if self.cumulative_fails >= self.config.max_candidate_fails:
            logger.warning(
                f"[phase=touchstone][gen={self.generation}] fallback: "
                f"reason=max_candidate_fails, "
                f"cumulative_fails={self.cumulative_fails}/{self.config.max_candidate_fails}"
            )
            self._fallback()
            return

        if len(self.seeds) < self.config.max_seeds:
            self._pick_next_candidate()

    def _pick_next_candidate(self) -> None:
        """选出下一个候选（按可配置的精确排名，1=最强）。

        并列处理：若目标排名位置存在多个 mu 相同的个体，选择其中
        battle_count 最高的（更可靠的对战记录）。
        """
        battled_ids = self._get_battled_player_ids()
        used = set(self.seeds) | {
            cid for cid, st in self.candidates.items() if not st.resolved
        }
        remaining = [iid for iid in battled_ids if iid not in used]
        if not remaining:
            logger.info(
                f"[phase=touchstone][gen={self.generation}] no_candidate: "
                "reason=no_battled_unused_individuals"
            )
            return

        mu_dict = self.payoff.get_trueskill_mus(remaining)
        # 按 mu 降序排列（最强在前，1=最强）
        remaining.sort(key=lambda iid: mu_dict.get(iid, 25.0), reverse=True)
        n = len(remaining)

        # 根据可配置排名选候选（1-indexed）
        target_rank = min(self.config.next_candidate_rank, n)
        pos = target_rank - 1  # 0-indexed
        target_mu = mu_dict.get(remaining[pos], 25.0)

        # 并列处理：找到该 mu 值的所有并列个体
        tied_group = [
            iid for iid in remaining
            if mu_dict.get(iid, 25.0) == target_mu
        ]
        # 从并列组中选 battle_count 最高的（对战记录更可靠）
        tied_group.sort(key=lambda iid: self._battle_count.get(iid, 0), reverse=True)
        new_candidate = tied_group[0]

        if len(tied_group) > 1:
            logger.debug(
                f"[phase=touchstone][gen={self.generation}] candidate_tie_group: "
                f"rank={target_rank}, mu={target_mu:.2f}, size={len(tied_group)}"
            )

        self._launch_candidate(new_candidate)

    def _order_opponents(self, candidate_id: str) -> List[str]:
        others = [iid for iid in self.population_ids if iid != candidate_id]
        already = {
            opp for opp in others
            if tuple(sorted([candidate_id, opp])) in self.battled_pairs
        }
        others = [iid for iid in others if iid not in already]
        with_battles = [iid for iid in others if self._battle_count.get(iid, 0) > 0]
        without_battles = [iid for iid in others if self._battle_count.get(iid, 0) == 0]
        with_battles.sort(key=lambda iid: -self._battle_count.get(iid, 0))
        return with_battles + without_battles

    def _fallback(self) -> None:
        self._fallback_triggered = True
        self.queue.cancel_all()
        for cid in self.candidates:
            self.candidates[cid].resolved = True
        logger.warning(
            f"[phase=touchstone][gen={self.generation}] fallback_applied: "
            "remaining_tasks=cancelled, ranking=trueskill_mu"
        )


# ─── Selector ──────────────────────────────────────────────────────────


class TouchstoneSelector:
    """基准筛选阶段（替换 BenchmarkTest）：
    种群内部二分种子互打 → 全员 mu 排名（供后续 RoundRobin 取 Top24）。
    """

    def __init__(
        self,
        config: TouchstoneConfig,
        payoff: BattleSharedPayoff,
    ):
        self.config = config
        self._payoff = payoff
        self._coordinator: Optional[_TouchstoneCoordinator] = None

    @property
    def stats(self) -> dict:
        return self._coordinator.stats if self._coordinator else {}

    @property
    def seeds(self) -> List[str]:
        return list(self._coordinator.seeds) if self._coordinator else []

    @property
    def is_fallback(self) -> bool:
        return self._coordinator.is_fallback if self._coordinator else False

    @property
    def unique_candidates(self) -> int:
        return self._coordinator.unique_candidates_count if self._coordinator else 0

    # 兼容 logging：训练器读取 self.battle_selection.benchmark.num_agents_used。
    # 在二分种子语义下：候选个体去重数视为"使用过的 agent 数"。
    @property
    def num_agents_used(self) -> int:
        return self.unique_candidates

    @property
    def final_tau(self) -> Optional[float]:
        # 不再有 Kendall τ 早停；保持接口存在，返回 None
        return None

    @property
    def illegal_stats(self) -> Dict[str, Any]:
        """返回 touchstone 阶段的非法动作 & 回合数汇总统计。"""
        if self._coordinator is None:
            return {"total_illegal": 0, "num_battles": 0, "avg_rounds": 0}
        battle_results = self._coordinator._battle_results
        if not battle_results:
            return {"total_illegal": 0, "num_battles": 0, "avg_rounds": 0}
        total_illegal = sum(
            b.get("illegal_home", 0) + b.get("illegal_away", 0)
            for b in battle_results
        )
        total_rounds = sum(b.get("total_rounds", 0) for b in battle_results)
        n_battles = len(battle_results)
        return {
            "total_illegal": total_illegal,
            "num_battles": n_battles,
            "avg_rounds": round(total_rounds / max(n_battles, 1), 1),
        }

    def evaluate(
        self,
        population: Population,
        generation: int = 1,
        gen_dir: str = None,
    ) -> List[Individual]:
        """执行二分种子选拔，返回按 trueskill_mu 降序排列的全部个体。"""
        individuals = list(population.individuals)
        for ind in individuals:
            self._payoff.ensure_player(ind.id)

        # 预序列化所有个体的 GAWarAgent
        agent_bytes: Dict[str, bytes] = {}
        for ind in individuals:
            ga_agent = self._create_ga_agent(ind)
            agent_bytes[ind.id] = AgentLoader.serialize(ga_agent)

        queue = TaskQueue()
        coord = _TouchstoneCoordinator(self.config, self._payoff, queue, generation)
        self._coordinator = coord

        executor = BattleExecutor(
            task_queue=queue,
            agent_bytes=agent_bytes,
            callback=coord.on_battle_complete,
            max_workers=self.config.max_workers,
            max_rounds=self.config.max_rounds,
        )

        logger.info(
            f"[phase=touchstone][gen={generation}] start: "
            f"individuals={len(individuals)}, M={self.config.M}, "
            f"max_seeds={self.config.max_seeds}, n_battle={self.config.n_battle}, "
            f"max_candidate_fails={self.config.max_candidate_fails}, "
            f"max_workers={self.config.max_workers}"
        )
        coord.start([ind.id for ind in individuals])
        executor.run()
        coord.finalize_battle_classification()

        # 持久化对战数据
        if gen_dir:
            coord.save_battle_results(gen_dir)

        # 回写 trueskill_mu / trueskill_sigma
        individual_ids = [ind.id for ind in individuals]
        mu_dict = self._payoff.get_trueskill_mus(individual_ids)
        sigma_dict = self._payoff.get_trueskill_sigmas(individual_ids)
        for ind in individuals:
            ind.trueskill_mu = mu_dict.get(ind.id, 25.0)
            ind.trueskill_sigma = sigma_dict.get(ind.id, 8.333)

        ranked = sorted(individuals, key=lambda ind: ind.trueskill_mu, reverse=True)

        mu_values = [ind.trueskill_mu for ind in ranked]
        st = coord.stats
        logger.info(
            f"[phase=touchstone][gen={generation}] done: "
            f"best_mu={mu_values[0]:.2f}, worst_mu={mu_values[-1]:.2f}, "
            f"mean_mu={sum(mu_values)/len(mu_values):.2f}, "
            f"battles={st['battles_total']}, effective_battles={st['battles_effective']}, "
            f"unique_candidates={coord.unique_candidates_count}, "
            f"seeds={len(coord.seeds)}, fallback={coord.is_fallback}"
        )

        # 非法动作 & 回合数统计
        illegal_total = sum(
            b.get("illegal_home", 0) + b.get("illegal_away", 0)
            for b in coord._battle_results
        )
        total_rounds = sum(b.get("total_rounds", 0) for b in coord._battle_results)
        n_battles = max(len(coord._battle_results), 1)
        avg_rounds = round(total_rounds / n_battles, 1)
        logger.info(
            f"[phase=touchstone][gen={generation}] stats: "
            f"illegal={illegal_total}, "
            f"battles={len(coord._battle_results)}, "
            f"avg_rounds={avg_rounds}, "
            f"avg_illegal={illegal_total / n_battles:.1f}/battle"
        )

        return ranked

    def _create_ga_agent(self, individual: Individual) -> GAWarAgent:
        import torch
        device = torch.device(self.config.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.config.hidden_dim,
        ).to(device)
        load_genome_to_network(network, individual.genome)
        network.eval()
        return GAWarAgent(
            player_id=0,
            network=network,
            device=device,
        )


__all__ = ["TouchstoneConfig", "TouchstoneSelector"]
