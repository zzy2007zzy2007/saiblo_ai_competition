"""TaskQueue — 优先级任务队列（线程安全）

v4 设计文档 §4.2 的精确实现。
取消逻辑由 pop 时检查 cancelled 标记实现（规则 9）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    """对战任务 (v4 §4.1)"""
    task_id: str
    player_a: str
    player_b: str
    n_games: int                    # 对战局数 (甄别=1, 批量=n_battle*2)
    priority: int = 0               # 0=HIGH(甄别), 1=LOW(批量)
    cancelled: bool = False         # 是否已被取消
    creator: str = ""               # 创建者 (candidate_id 或 "batch_{seed_id}")
    created_at: float = 0.0


class TaskQueue:
    """优先级任务队列，线程安全（v4 §4.2）"""

    def __init__(self):
        self._high: List[Task] = []    # 甄别阶段 (priority=0)
        self._low: List[Task] = []     # 批量补齐 (priority=1)
        self._lock = threading.Lock()

    def submit(self, task: Task) -> None:
        """提交任务（规则 10: 高优先级进 _high，低优先级进 _low）"""
        with self._lock:
            if task.priority == 0:
                self._high.append(task)
            else:
                self._low.append(task)

    def pop(self) -> Optional[Task]:
        """取出下一个任务（高优先级优先，跳过已取消的）（规则 9, 10）"""
        with self._lock:
            while self._high:
                t = self._high.pop(0)
                if not t.cancelled:
                    return t
            while self._low:
                t = self._low.pop(0)
                if not t.cancelled:
                    return t
        return None

    def cancel_by_creator(self, creator_id: str) -> int:
        """取消某创建者的所有待执行 task（规则 6: 候选失败时调用）"""
        cancelled = 0
        with self._lock:
            for lst in (self._high, self._low):
                for t in lst:
                    if t.creator == creator_id and not t.cancelled:
                        t.cancelled = True
                        cancelled += 1
        return cancelled

    def cancel_all(self) -> int:
        """取消所有待执行 task"""
        cancelled = 0
        with self._lock:
            for lst in (self._high, self._low):
                for t in lst:
                    if not t.cancelled:
                        t.cancelled = True
                        cancelled += 1
        return cancelled

    def cancel_high_priority(self) -> int:
        """仅取消 priority=0 (HIGH，甄别) 的待执行 task（保留作为兼容/调试用途）"""
        cancelled = 0
        with self._lock:
            for t in self._high:
                if not t.cancelled:
                    t.cancelled = True
                    cancelled += 1
        return cancelled

    def cancel_battles_without_seed(self, seed_ids) -> int:
        """规则 10 (v4 文档新版): 全部种子确定后，取消"双方都不是种子"的所有待执行 task。
        含种子的 task 一律保留。

        Args:
            seed_ids: 已确定的二分种子 id 集合（任何可迭代，会被转为 set）。

        Returns:
            被取消的 task 数量。
        """
        seeds = set(seed_ids)
        cancelled = 0
        with self._lock:
            for lst in (self._high, self._low):
                for t in lst:
                    if t.cancelled:
                        continue
                    if t.player_a not in seeds and t.player_b not in seeds:
                        t.cancelled = True
                        cancelled += 1
        return cancelled

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._high + self._low if not t.cancelled)

    @property
    def total_submitted(self) -> int:
        """已提交过的 task 总数（包括已完成和已取消）"""
        with self._lock:
            return len(self._high) + len(self._low)
