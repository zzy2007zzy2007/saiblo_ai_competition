"""BisectionCoordinator — 二分种子协调器（v4 §4.4 核心状态机）

完整实现设计文档 14 条规则。
甄别逻辑完全由 callback 驱动（规则 12）。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import trueskill
from loguru import logger

try:
    from .task_queue import Task, TaskQueue
except ImportError:
    from task_queue import Task, TaskQueue  # type: ignore


@dataclass
class CandidateState:
    """单个候选二分种子的甄别状态（v4 §4.4）"""
    id: str
    start_time: float
    completed_battles: int = 0
    in_range_streak: int = 0              # 连续在中位区间的局数
    pending_opponents: List[str] = field(default_factory=list)
    resolved: bool = False                # 已定论（成功或失败）
    is_seed: bool = False


class SimplePayoff:
    """简化的 TrueSkill 评分管理器

    替代 BattleSharedPayoff，只包含 bisection 实验所需的最少接口。
    底层使用 trueskill 库。
    """

    def __init__(self):
        self._ratings: Dict[str, trueskill.Rating] = {}
        self._battle_counts: Dict[str, int] = {}

    def ensure_player(self, player_id: str) -> None:
        """注册玩家"""
        if player_id not in self._ratings:
            self._ratings[player_id] = trueskill.Rating()
            self._battle_counts[player_id] = 0

    def update_match(self, home: str, away: str, result_code: int) -> None:
        """更新 TrueSkill 评分（规则 3）

        Args:
            result_code: 1=home胜, -1=away胜, 0=平
        """
        self.ensure_player(home)
        self.ensure_player(away)

        h = self._ratings[home]
        a = self._ratings[away]

        if result_code == 1:
            new_h, new_a = trueskill.rate_1vs1(h, a)
        elif result_code == -1:
            new_a, new_h = trueskill.rate_1vs1(a, h)
        else:
            new_h, new_a = trueskill.rate_1vs1(h, a, drawn=True)

        self._ratings[home] = new_h
        self._ratings[away] = new_a
        self._battle_counts[home] = self._battle_counts.get(home, 0) + 1
        self._battle_counts[away] = self._battle_counts.get(away, 0) + 1

    def get_trueskill_mu(self, player_id: str) -> float:
        """获取单个玩家的 TrueSkill mu"""
        if player_id in self._ratings:
            return self._ratings[player_id].mu
        return 25.0

    def get_trueskill_sigma(self, player_id: str) -> float:
        """获取单个玩家的 TrueSkill sigma"""
        if player_id in self._ratings:
            return self._ratings[player_id].sigma
        return 8.33

    def get_battle_count(self, player_id: str) -> int:
        """获取玩家的历史对战总局数"""
        return self._battle_counts.get(player_id, 0)

    def get_all_mus_sorted(self, player_ids: List[str]) -> List[float]:
        """获取所有指定玩家的 mu，排序后返回"""
        mus = [self.get_trueskill_mu(pid) for pid in player_ids]
        return sorted(mus)


class BisectionCoordinator:
    """二分种子协调器 — 核心状态机（v4 §4.4）

    状态流转:
        IDLE → DISCOVERING → (SEEDS_FOUND | FALLBACK)
    """

    def __init__(
        self,
        config: dict,
        payoff: SimplePayoff,
        queue: TaskQueue,
    ):
        self.config = config
        self.payoff = payoff
        self.queue = queue
        self.seeds: List[str] = []                      # 已确定的二分种子 ID
        self.candidates: Dict[str, CandidateState] = {} # 候选状态
        self.population_ids: List[str] = []
        self.consecutive_fails: int = 0                  # 规则 14: 连续失败计数
        self.battled_pairs: Set[Tuple[str, str]] = set() # 规则 5: 已对战配对去重
        self._fallback_triggered: bool = False           # 规则 14: 兜底是否已触发

        # 统计
        self._stats = {
            "candidates_launched": 0,
            "candidates_promoted": 0,
            "candidates_failed": 0,
            "discovery_tasks_submitted": 0,
            "batch_tasks_submitted": 0,
            "cancelled_tasks_on_promote": 0,
            "cancelled_tasks_on_fail": 0,
            "cancelled_tasks_on_final": 0,
            # 对局分类（按对局开始时刻动态判定）
            "battles_no_seed": 0,         # 类 1: 对局开始时双方都非种子（父集）
            "battles_with_seed": 0,       # 类 2: 对局开始时已有一方是种子
            "battles_finally_no_seed": 0, # 类 3: bisection 结束后双方仍非种子（类 1 的子集）
            # 候选失败计数
            "failed_before_all_seeds_found": 0,  # 在凑齐 max_seeds 之前的失败计数
            "failed_total": 0,                    # 整个 bisection 阶段的失败计数
        }
        # 用于事后回扫类 3：记录类 1 对局的 (home, away) pair
        self._no_seed_battles: List[Tuple[str, str]] = []
        self._all_seeds_found: bool = False  # 是否已凑齐 max_seeds 个种子

    @property
    def is_fallback(self) -> bool:
        return self._fallback_triggered

    @property
    def stats(self) -> dict:
        return self._stats

    # ── 公开接口 ──

    def start(self, population_ids: List[str]) -> None:
        """启动二分种子选拔（规则 1, 14）"""
        self.population_ids = list(population_ids)
        n = len(self.population_ids)

        logger.info(f"BisectionCoordinator.start: N={n}, "
                     f"P={self.config['P']}, M={self.config['M']}, "
                     f"N_fail={self.config['N']}, max_seeds={self.config['max_seeds']}")

        # 规则 14: N < 6 → 直接 fallback
        if n < 6:
            logger.warning(f"N={n} < 6 → 直接 fallback")
            self._fallback()
            return

        # 注册到 payoff
        for iid in self.population_ids:
            self.payoff.ensure_player(iid)

        # 规则 1: 等概率随机选 P 个候选
        P = min(self.config["P"], n)
        initial_candidates = random.sample(self.population_ids, P)
        for cid in initial_candidates:
            self._launch_candidate(cid)

    def on_battle_complete(self, home: str, away: str, result: dict) -> None:
        """Callback: 每局对战完成后由 BattleExecutor 调用（规则 3, 12）

        Args:
            home: 主队 ID
            away: 客队 ID
            result: {"winner": ..., "result_code": ..., "home": ..., "away": ...}
        """
        # 对局分类（按对局开始时刻动态判定：此时刻 self.seeds 中的就是已升级种子）
        # 注意：BattleExecutor 在 callback 之前完成对战，此时 self.seeds 反映"对局开始时"的种子集合
        # （callback 内部不会在执行同一 task 的过程中插入种子升级，因为升级也是 callback 触发的）
        seed_set = set(self.seeds)
        if home in seed_set or away in seed_set:
            self._stats["battles_with_seed"] += 1
        else:
            self._stats["battles_no_seed"] += 1
            # 记录 pair，bisection 结束后用于回扫类 3
            self._no_seed_battles.append((home, away))

        # 规则 3: 更新双边 TrueSkill
        result_code = result.get("result_code", 0)
        self.payoff.update_match(home, away, result_code)

        # 去重
        pair = tuple(sorted([home, away]))
        self.battled_pairs.add(pair)

        # 规则 12: 检查相关候选状态
        for pid in (home, away):
            if pid in self.candidates:
                state = self.candidates[pid]
                state.completed_battles += 1           # 规则 3: 累加对战记录数
                opp = home if pid == away else away
                if opp in state.pending_opponents:
                    state.pending_opponents.remove(opp)
                self._check_candidate(pid)

    # ── 内部方法 ──

    def _launch_candidate(self, candidate_id: str) -> None:
        """启动一个候选的甄别流程（规则 1, 10）"""
        state = CandidateState(id=candidate_id, start_time=time.time())
        self.candidates[candidate_id] = state
        self._stats["candidates_launched"] += 1

        # 规则 7: 优先与已有对战记录 > 0 的个体对战（按历史局数降序）
        opponents = self._order_opponents(candidate_id)

        for opp_id in opponents:
            # 规则 10: 甄别阶段每对 1 局，优先级 HIGH
            task = Task(
                task_id=f"disc_{candidate_id}_vs_{opp_id}",
                player_a=candidate_id,
                player_b=opp_id,
                n_games=1,              # 甄别阶段 1 局
                priority=0,             # HIGH
                creator=candidate_id,
                created_at=time.time(),
            )
            self.queue.submit(task)
            state.pending_opponents.append(opp_id)
            self._stats["discovery_tasks_submitted"] += 1

        mu = self.payoff.get_trueskill_mu(candidate_id)
        logger.info(f"候选 {candidate_id}: mu={mu:.2f}, "
                     f"提交 {len(opponents)} 个甄别 task")

    def _check_candidate(self, candidate_id: str) -> None:
        """检查候选是否达标或失败（规则 4, 6）

        中位区间严格按规则 v4：
        - 仅基于"已对战个体"（battle_count > 0）计算
        - mu 降序排序，下界 = sorted_desc[⌊9m/20⌋]，上界 = sorted_desc[⌊m/5⌋]
        - lo <= mu <= hi 表示候选在 Top20%-Top45% 区间内（mu 数值落在 [下界, 上界]）
        """
        state = self.candidates.get(candidate_id)
        if not state or state.resolved:
            return

        mu = self.payoff.get_trueskill_mu(candidate_id)

        # 仅已对战个体参与中位区间计算
        battled_ids = self._get_battled_player_ids()
        m = len(battled_ids)

        # 已对战个体数 < 3 时跳过判定
        if m < 3:
            return

        battled_mus_desc = sorted(
            (self.payoff.get_trueskill_mu(pid) for pid in battled_ids),
            reverse=True,
        )
        # 规则 v4: mu 降序时，第 m//5 名 = 数值上界(Top20%)，第 9m//20 名 = 数值下界(Top45%)
        hi = battled_mus_desc[m // 5]
        lo = battled_mus_desc[min(9 * m // 20, m - 1)]
        in_range = lo <= mu <= hi

        if in_range:
            state.in_range_streak += 1
        else:
            state.in_range_streak = 0

        completed = state.completed_battles

        # 规则 4: M 个对手后贯穿全程在中位区间 → 升级
        if completed >= self.config["M"] and state.in_range_streak == completed:
            self.consecutive_fails = 0  # 重置连续失败计数
            self._promote_to_seed(candidate_id)
            return

        # 规则 6: N 个对手后不在中位区间 → 放弃
        if completed >= self.config["N"] and state.in_range_streak < 1:
            self._fail_candidate(candidate_id)
            return

    def _get_battled_player_ids(self) -> List[str]:
        """返回已有对战记录（battle_count > 0）的个体 ID 列表"""
        return [
            pid for pid in self.population_ids
            if self.payoff.get_battle_count(pid) > 0
        ]

    def _promote_to_seed(self, candidate_id: str) -> None:
        """候选升级为正式二分种子（规则 5, 8, 9）"""
        state = self.candidates[candidate_id]
        state.resolved = True
        state.is_seed = True
        self.seeds.append(candidate_id)
        self._stats["candidates_promoted"] += 1

        # 规则 5: 提交与全体的批量 task（去重）
        batch_count = 0
        for opp_id in self.population_ids:
            if opp_id == candidate_id:
                continue
            pair = tuple(sorted([candidate_id, opp_id]))
            if pair in self.battled_pairs:      # 规则 5: 已对战过的不重复
                continue
            self.battled_pairs.add(pair)

            n_battle = self.config.get("n_battle", 2)
            task = Task(
                task_id=f"batch_{candidate_id}_vs_{opp_id}",
                player_a=candidate_id,
                player_b=opp_id,
                n_games=2 * n_battle,           # 规则 2, 10: 批量阶段 n_battle*2 局
                priority=1,                     # LOW
                creator=f"batch_{candidate_id}",
                created_at=time.time(),
            )
            self.queue.submit(task)
            batch_count += 1
            self._stats["batch_tasks_submitted"] += 1

        mu = self.payoff.get_trueskill_mu(candidate_id)
        logger.info(f"种子 {candidate_id}: mu={mu:.2f}, "
                     f"提交 {batch_count} 个批量 task")

        # 取消该候选剩余的甄别 task（规则 6 的取消由 _fail_candidate 处理）
        n = self.queue.cancel_by_creator(candidate_id)
        self._stats["cancelled_tasks_on_promote"] += n

        # 规则 10 (v4 新版): 全部种子确定 → 取消"双方都不是种子"的所有待执行 task
        # 含种子的 task（无论甄别还是批量）一律保留
        if len(self.seeds) >= self.config["max_seeds"]:
            self._all_seeds_found = True
            n = self.queue.cancel_battles_without_seed(self.seeds)
            self._stats["cancelled_tasks_on_final"] += n
            for cid in list(self.candidates.keys()):
                if not self.candidates[cid].resolved:
                    self.candidates[cid].resolved = True
            logger.info(f"全部 {len(self.seeds)} 个种子已确定, 取消 {n} 个'双方非种子'的待执行 task; "
                         f"截至此刻 failed_before_all_seeds_found="
                         f"{self._stats['failed_before_all_seeds_found']}")

    def _fail_candidate(self, candidate_id: str) -> None:
        """候选失败处理（规则 6, 7, 14）"""
        state = self.candidates[candidate_id]
        state.resolved = True
        n = self.queue.cancel_by_creator(candidate_id)  # 规则 6: 取消全部待执行 task
        self._stats["cancelled_tasks_on_fail"] += n
        self._stats["candidates_failed"] += 1
        self._stats["failed_total"] += 1
        # 在凑齐 max_seeds 之前的失败计数（动态判定）
        if not self._all_seeds_found:
            self._stats["failed_before_all_seeds_found"] += 1
        self.consecutive_fails += 1

        mu = self.payoff.get_trueskill_mu(candidate_id)
        logger.info(f"候选 {candidate_id} 失败: mu={mu:.2f}, "
                     f"completed={state.completed_battles}, "
                     f"streak={state.in_range_streak}")

        # 规则 14: 连续 max_consecutive_fails 次候选失败 → fallback
        if self.consecutive_fails >= self.config.get("max_consecutive_fails", 3):
            logger.warning(f"连续 {self.consecutive_fails} 次候选失败 → fallback")
            self._fallback()
            return

        # 规则 7: 选取新候选（还有种子名额时）
        if len(self.seeds) < self.config["max_seeds"]:
            self._pick_next_candidate()

    def _pick_next_candidate(self) -> None:
        """在未做候选 且 已有对战记录 的个体中选 mu 中位数作为新候选（规则 7）

        规则 7 候选池: C = { i ∈ 种群 | i 未做候选 ∧ battle_count(i) > 0 }
        新候选 = C 按 mu 升序后的 Top33% 位置 C[⌊2|C|/3⌋]
        若 C 为空则不再选取新候选。
        """
        used = set(self.candidates.keys()) | set(self.seeds)
        remaining = [
            iid for iid in self.population_ids
            if iid not in used and self.payoff.get_battle_count(iid) > 0
        ]
        if not remaining:
            logger.info("候选池 C 为空（无未做候选且已对战的个体），停止发现新候选")
            return

        remaining.sort(key=lambda iid: self.payoff.get_trueskill_mu(iid))
        new_candidate = remaining[2 * len(remaining) // 3]
        self._launch_candidate(new_candidate)

    def _order_opponents(self, candidate_id: str) -> List[str]:
        """规则 7: 按已有的对战记录数降序排列对手（优先与已对战个体对战）"""
        others = [iid for iid in self.population_ids if iid != candidate_id]
        others.sort(key=lambda iid: -self.payoff.get_battle_count(iid))
        return others

    def _fallback(self) -> None:
        """规则 14: 兜底 — 放弃二分种子方案"""
        self._fallback_triggered = True
        self.queue.cancel_all()
        for cid in self.candidates:
            self.candidates[cid].resolved = True
        logger.warning("二分种子 fallback: 直接用累积 TrueSkill 排序取 Top16")

    def get_ranking_by_mu(self) -> List[str]:
        """按累积 TrueSkill mu 降序排列已对战个体（规则 13）

        仅返回已有对战记录（battle_count > 0）的个体。无对战记录的个体
        不参与排序。
        """
        battled_ids = self._get_battled_player_ids()
        sorted_ids = sorted(
            battled_ids,
            key=lambda iid: self.payoff.get_trueskill_mu(iid),
            reverse=True,
        )
        return sorted_ids

    def finalize_battle_classification(self) -> None:
        """bisection 阶段结束后调用，回扫记录类 3 对局数

        类 3：开始时双方都非种子，且 bisection 结束后双方仍未升级为种子（即"白打"对局）。
        是类 1 的子集。
        """
        final_seed_set = set(self.seeds)
        count = 0
        for (home, away) in self._no_seed_battles:
            if home not in final_seed_set and away not in final_seed_set:
                count += 1
        self._stats["battles_finally_no_seed"] = count
