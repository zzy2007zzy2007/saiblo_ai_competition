"""BattleExecutor — 独立任务执行器（v4 §4.3）

从 TaskQueue 取 task → 并行执行对战 → 回调结果。
不感知候选甄别逻辑（规则 11）。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from loguru import logger

try:
    from .individual import Individual, simulate_battle
    from .task_queue import Task, TaskQueue
except ImportError:
    from individual import Individual, simulate_battle  # type: ignore
    from task_queue import Task, TaskQueue  # type: ignore


class BattleExecutor:
    """独立任务执行器（v4 §4.3）

    从 Queue 取 task 并提交给线程池并行执行（规则 11）。
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        individuals: Dict[str, Individual],
        callback: Callable[[str, str, object], None],
        max_workers: int = 25,
    ):
        self._queue = task_queue
        self._individuals = individuals
        self._callback = callback
        self._max_workers = max_workers
        self._running = False
        self._battles_executed: int = 0       # 实际执行的对战局数
        self._tasks_completed: int = 0         # 完成的 task 数
        self._cancelled_skipped: int = 0       # 跳过(已取消)的 task 数
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self._end_time > 0:
            return self._end_time - self._start_time
        return time.time() - self._start_time

    @property
    def battles_executed(self) -> int:
        return self._battles_executed

    @property
    def tasks_completed(self) -> int:
        return self._tasks_completed

    @property
    def cancelled_skipped(self) -> int:
        return self._cancelled_skipped

    def run(self) -> None:
        """主循环：取 task → 并行执行对战 → 回调（阻塞直到队列空）（规则 11）"""
        self._running = True
        self._start_time = time.time()
        self._battles_executed = 0
        self._tasks_completed = 0
        self._cancelled_skipped = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = []
            while self._running:
                # 补充 futures 到 max_workers
                while len(futures) < self._max_workers:
                    task = self._queue.pop()
                    if task is None:
                        break
                    future = pool.submit(self._execute_task, task)
                    futures.append(future)

                if not futures:
                    break

                # 等待任意一个完成
                done = set()
                for f in futures:
                    if f.done():
                        done.add(f)

                if not done:
                    time.sleep(0.02)
                    continue

                for f in done:
                    futures.remove(f)
                    # 异常处理在 _execute_task 内部完成

        self._end_time = time.time()

    def _execute_task(self, task: Task) -> None:
        """执行单个 task 的全部 n_games 局对战（规则 2: 原子提交）

        甄别阶段: n_games=1 (规则 10)
        批量阶段: n_games=n_battle*2 (规则 5)
        """
        # 规则 9: pop 后二次检查 cancelled 标记
        if task.cancelled:
            self._cancelled_skipped += 1
            return

        try:
            for game_idx in range(task.n_games):
                # 交替先后手
                first_id = task.player_a if game_idx % 2 == 0 else task.player_b
                second_id = task.player_b if game_idx % 2 == 0 else task.player_a

                ind_a = self._individuals[first_id]
                ind_b = self._individuals[second_id]

                outcome = simulate_battle(ind_a, ind_b)
                result_code = outcome.to_result_code(first_id, second_id)

                # 规则 3: 每局完成后立即 callback
                self._callback(first_id, second_id, {
                    "winner": outcome.winner_id,
                    "result_code": result_code,
                    "home": first_id,
                    "away": second_id,
                })
                self._battles_executed += 1

            self._tasks_completed += 1

        except Exception as e:
            logger.error(f"Task {task.task_id} 执行异常: {e}")

    def stop(self) -> None:
        self._running = False
