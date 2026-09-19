"""BisectionCoordinator(original_rule) — 严格对齐 original_rule.txt 14 条规则

核心要点：
- 规则1：第 1 个个体作为初始候选
- 规则3：M 个以上对手后，mu ∈ Top33%-Top66%（基于已对战个体）→ 升级
- 规则4：M 个以上对手后，mu ∉ Top33%-Top66% → 失败
- 规则5：新候选 = 已对战个体按 mu 升序的 Top33% 位置
- 规则6：候选发起对战时，优先选 battle_count > 0 的对手
- 规则8：3 个种子确定后，提交与全体的批量对战（已对战的不重复）
- 规则9：候选降级或种子全部确定后，取消队列中"双方都不是种子/候选"的对战
- 规则14：累计 max_candidate_fails 个候选失败 → fallback
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import trueskill
from loguru import logger

try:
    from ..bisection_experiment.task_queue import Task, TaskQueue
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bisection_experiment"))
    from task_queue import Task, TaskQueue  # type: ignore


@dataclass
class CandidateState:
    """单个候选二分种子的甄别状态"""
    id: str
    start_time: float
    completed_battles: int = 0
    pending_opponents: List[str] = field(default_factory=list)
    resolved: bool = False
    is_seed: bool = False


class SimplePayoff:
    """简化的 TrueSkill 评分管理器"""

    def __init__(self):
        self._ratings: Dict[str, trueskill.Rating] = {}
        self._battle_counts: Dict[str, int] = {}

    def ensure_player(self, player_id: str) -> None:
        if player_id not in self._ratings:
            self._ratings[player_id] = trueskill.Rating()
            self._battle_counts[player_id] = 0

    def update_match(self, home: str, away: str, result_code: int) -> None:
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
        if player_id in self._ratings:
            return self._ratings[player_id].mu
        return 25.0

    def get_battle_count(self, player_id: str) -> int:
        return self._battle_counts.get(player_id, 0)


class OriginalRuleCoordinator:
    """严格对齐 original_rule.txt 的二分种子协调器"""

    def __init__(
        self,
        config: dict,
        payoff: SimplePayoff,
        queue: TaskQueue,
    ):
        self.config = config
        self.payoff = payoff
        self.queue = queue
        self.seeds: List[str] = []
        self.candidates: Dict[str, CandidateState] = {}
        self.population_ids: List[str] = []
        self.cumulative_fails: int = 0  # 规则14: 累计候选失败计数
        self.battled_pairs: Set[Tuple[str, str]] = set()
        # 每对个体已打的局数（用于规则 8 的"补齐"逻辑）
        self.pair_battle_count: Dict[Tuple[str, str], int] = {}
        self._fallback_triggered: bool = False
        self._all_seeds_found: bool = False

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
            # 对局分类（按提交时刻角色判定，一类/二类互斥；三类按最终时刻判定）
            "battles_class1": 0,           # 一类：提交时一方是候选（且无种子参与）
            "battles_class2": 0,           # 二类：提交时一方是正式种子
            "battles_class3": 0,           # 三类：执行完毕时双方都不是正式种子
            "battles_total": 0,            # 总对战数（实际执行）
            "battles_effective": 0,        # 有效 = 总数 − 三类
        }
        # (home, away) 用于回扫三类
        self._all_battles: List[Tuple[str, str]] = []
        # 累计候选个体集合（去重）
        self._launched_candidate_set: Set[str] = set()

    @property
    def is_fallback(self) -> bool:
        return self._fallback_triggered

    @property
    def stats(self) -> dict:
        return self._stats

    # ── 公开接口 ──

    def start(self, population_ids: List[str]) -> None:
        """启动二分种子选拔（规则 1）"""
        self.population_ids = list(population_ids)
        n = len(self.population_ids)

        logger.info(
            f"OriginalRuleCoordinator.start: N={n}, M={self.config['M']}, "
            f"max_seeds={self.config['max_seeds']}, "
            f"max_candidate_fails={self.config['max_candidate_fails']}"
        )

        if n < 6:
            logger.warning(f"N={n} < 6 → 直接 fallback")
            self._fallback()
            return

        for iid in self.population_ids:
            self.payoff.ensure_player(iid)

        # 规则 1: 第 1 个个体作为初始候选
        initial_candidate = self.population_ids[0]
        self._launch_candidate(initial_candidate)

    def on_battle_complete(self, home: str, away: str, result: dict) -> None:
        """每局对战完成后回调"""
        self._stats["battles_total"] += 1
        self._all_battles.append((home, away))

        # 对局分类（按"提交=对战开始时刻"角色判定）
        # 注：BattleExecutor 在执行前才进入这里，此时 self.seeds/self.candidates 反映对战开始时刻
        seed_set = set(self.seeds)
        cand_active = {
            cid for cid, st in self.candidates.items()
            if not st.resolved
        }
        # 已是正式种子的不再当候选
        cand_active -= seed_set

        if home in seed_set or away in seed_set:
            self._stats["battles_class2"] += 1
        elif home in cand_active or away in cand_active:
            self._stats["battles_class1"] += 1
        # 其它情形（双方都不是候选/种子）不计入一/二类，但会在三类回扫时纳入

        # 规则 3: 更新双边 TrueSkill
        result_code = result.get("result_code", 0)
        self.payoff.update_match(home, away, result_code)

        pair = tuple(sorted([home, away]))
        self.battled_pairs.add(pair)
        self.pair_battle_count[pair] = self.pair_battle_count.get(pair, 0) + 1

        # 更新候选状态并检查
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

    # ── 内部方法 ──

    def _launch_candidate(self, candidate_id: str) -> None:
        """启动一个候选（规则 2, 6）"""
        state = CandidateState(id=candidate_id, start_time=time.time())
        self.candidates[candidate_id] = state
        self._stats["candidates_launched"] += 1
        self._launched_candidate_set.add(candidate_id)

        M = self.config["M"]
        # 规则 6: 优先选已有对战记录的对手
        opponents = self._order_opponents(candidate_id)[:M]

        for opp_id in opponents:
            task = Task(
                task_id=f"disc_{candidate_id}_vs_{opp_id}",
                player_a=candidate_id,
                player_b=opp_id,
                n_games=1,
                priority=0,
                creator=candidate_id,
                created_at=time.time(),
            )
            self.queue.submit(task)
            state.pending_opponents.append(opp_id)
            self._stats["discovery_tasks_submitted"] += 1

        mu = self.payoff.get_trueskill_mu(candidate_id)
        logger.info(
            f"候选 {candidate_id}: mu={mu:.2f}, 提交 {len(opponents)} 个甄别 task"
        )

    def _check_candidate(self, candidate_id: str) -> None:
        """检查候选是否升级或失败（规则 3, 4）

        规则 3/4: M 个以上对手后，按 mu 判定 Top20%-Top45% 区间
        - 区间下界 = 第 ⌊11N/20⌋ 名的 mu，上界 = 第 ⌊4N/5⌋ 名的 mu
        - 已对战个体集合作为参考
        """
        state = self.candidates.get(candidate_id)
        if not state or state.resolved:
            return

        if state.completed_battles < self.config["M"]:
            return  # 还没满 M 局

        # 已满 M 局，做一次性判定
        mu = self.payoff.get_trueskill_mu(candidate_id)
        battled_ids = self._get_battled_player_ids()
        m = len(battled_ids)

        if m < 3:
            return  # 数据太少，跳过

        # mu 升序排序
        battled_mus_asc = sorted(
            self.payoff.get_trueskill_mu(pid) for pid in battled_ids
        )
        # 第 ⌊11m/20⌋ 名 = 下界，第 ⌊4m/5⌋ 名 = 上界
        lo_idx = 11 * m // 20
        hi_idx = min(4 * m // 5, m - 1)
        lo = battled_mus_asc[lo_idx]
        hi = battled_mus_asc[hi_idx]

        if lo <= mu <= hi:
            self._promote_to_seed(candidate_id)
        else:
            self._fail_candidate(candidate_id)

    def _get_battled_player_ids(self) -> List[str]:
        return [
            pid for pid in self.population_ids
            if self.payoff.get_battle_count(pid) > 0
        ]

    def _promote_to_seed(self, candidate_id: str) -> None:
        """候选升级（规则 3, 8, 9）"""
        state = self.candidates[candidate_id]
        state.resolved = True
        state.is_seed = True
        self.seeds.append(candidate_id)
        self._stats["candidates_promoted"] += 1

        mu = self.payoff.get_trueskill_mu(candidate_id)
        logger.info(f"种子 {candidate_id}: mu={mu:.2f} (已确认 {len(self.seeds)}/{self.config['max_seeds']})")

        # 取消该候选剩余的甄别 task
        n = self.queue.cancel_by_creator(candidate_id)
        self._stats["cancelled_tasks_on_promote"] += n

        # 规则 8: 凑齐 max_seeds 后，提交批量对战
        if len(self.seeds) >= self.config["max_seeds"]:
            self._submit_batch_battles()
            self._all_seeds_found = True
            # 规则 9: 取消"双方都不是种子/候选"的待执行 task
            n = self.queue.cancel_battles_without_seed(self.seeds)
            self._stats["cancelled_tasks_on_final"] += n
            # 候选降级
            for cid in list(self.candidates.keys()):
                if not self.candidates[cid].resolved:
                    self.candidates[cid].resolved = True
            logger.info(
                f"全部 {len(self.seeds)} 个种子已确定，提交批量对战；"
                f"取消 {n} 个'双方非种子'的待执行 task"
            )
        else:
            # 还需更多种子 → 选下一个候选
            self._pick_next_candidate()

    def _submit_batch_battles(self) -> None:
        """规则 8: 3 个种子确认后，为它们提交与全部其他个体的批量对战。
        已与种子对战过 k 局的，**补齐 (n_battle*2 − k) 局**（不足则补，已满则跳过）。
        每个对子只处理一次（避免两个种子重复补齐同一对子）。
        """
        n_battle = self.config.get("n_battle", 1)
        target_games = 2 * n_battle
        submitted_pairs: Set[Tuple[str, str]] = set()

        for seed_id in self.seeds:
            for opp_id in self.population_ids:
                if opp_id == seed_id:
                    continue
                pair = tuple(sorted([seed_id, opp_id]))
                if pair in submitted_pairs:
                    continue  # 该对子已由前一个种子的循环提交过
                already = self.pair_battle_count.get(pair, 0)
                remaining = target_games - already
                if remaining <= 0:
                    continue  # 已打满 n_battle*2 局，无需再打

                submitted_pairs.add(pair)

                task = Task(
                    task_id=f"batch_{seed_id}_vs_{opp_id}",
                    player_a=seed_id,
                    player_b=opp_id,
                    n_games=remaining,
                    priority=1,
                    creator=f"batch_{seed_id}",
                    created_at=time.time(),
                )
                self.queue.submit(task)
                self._stats["batch_tasks_submitted"] += 1

    def _fail_candidate(self, candidate_id: str) -> None:
        """候选失败（规则 4, 14）"""
        state = self.candidates[candidate_id]
        state.resolved = True
        n = self.queue.cancel_by_creator(candidate_id)
        self._stats["cancelled_tasks_on_fail"] += n
        self._stats["candidates_failed"] += 1
        self.cumulative_fails += 1

        mu = self.payoff.get_trueskill_mu(candidate_id)
        logger.info(
            f"候选 {candidate_id} 失败: mu={mu:.2f}, "
            f"completed={state.completed_battles}, "
            f"cumulative_fails={self.cumulative_fails}/{self.config['max_candidate_fails']}"
        )

        # 规则 14: 累计失败 → fallback
        if self.cumulative_fails >= self.config["max_candidate_fails"]:
            logger.warning(
                f"累计 {self.cumulative_fails} 次候选失败 → fallback"
            )
            self._fallback()
            return

        # 还有种子名额 → 选新候选
        if len(self.seeds) < self.config["max_seeds"]:
            self._pick_next_candidate()

    def _pick_next_candidate(self) -> None:
        """规则 5: 已对战个体按 mu 升序的 Top33% 位置（个体可重复被选）"""
        battled_ids = self._get_battled_player_ids()
        # 排除已确认种子和当前未 resolved 的候选
        used = set(self.seeds) | {
            cid for cid, st in self.candidates.items() if not st.resolved
        }
        # 注意：已 resolved 但失败的候选可重复（规则注 1）
        remaining = [iid for iid in battled_ids if iid not in used]
        if not remaining:
            logger.info("无可选新候选（已对战且不在使用中的为空）")
            return

        remaining.sort(key=lambda iid: self.payoff.get_trueskill_mu(iid))
        new_candidate = remaining[2 * len(remaining) // 3]
        self._launch_candidate(new_candidate)

    def _order_opponents(self, candidate_id: str) -> List[str]:
        """规则 6: 优先与 battle_count > 0 的对手对战，
        不足则从其他个体顺序选择"""
        others = [iid for iid in self.population_ids if iid != candidate_id]
        # 已对战过候选的不再重复
        already = {
            opp for opp in others
            if tuple(sorted([candidate_id, opp])) in self.battled_pairs
        }
        others = [iid for iid in others if iid not in already]

        # 优先级 1: battle_count > 0 的（按 battle_count 降序）
        # 优先级 2: battle_count == 0 的（保持原始顺序）
        with_battles = [iid for iid in others if self.payoff.get_battle_count(iid) > 0]
        without_battles = [iid for iid in others if self.payoff.get_battle_count(iid) == 0]
        with_battles.sort(key=lambda iid: -self.payoff.get_battle_count(iid))
        return with_battles + without_battles

    def _fallback(self) -> None:
        """规则 14: 兜底"""
        self._fallback_triggered = True
        self.queue.cancel_all()
        for cid in self.candidates:
            self.candidates[cid].resolved = True
        logger.warning("二分种子 fallback: 直接用累积 TrueSkill 排序取 Top16")

    def get_ranking_by_mu(self) -> List[str]:
        """按累积 TrueSkill mu 降序"""
        battled_ids = self._get_battled_player_ids()
        return sorted(
            battled_ids,
            key=lambda iid: self.payoff.get_trueskill_mu(iid),
            reverse=True,
        )

    def finalize_battle_classification(self) -> None:
        """阶段结束后回扫：
        - 三类：双方最终都不是正式种子的对局数（浪费的计算量）
        - 有效 = 总数 − 三类
        """
        final_seeds = set(self.seeds)
        class3 = 0
        for (home, away) in self._all_battles:
            if home not in final_seeds and away not in final_seeds:
                class3 += 1
        self._stats["battles_class3"] = class3
        self._stats["battles_effective"] = self._stats["battles_total"] - class3

    @property
    def unique_candidates_count(self) -> int:
        """累计候选个体数（去重）"""
        return len(self._launched_candidate_set)
