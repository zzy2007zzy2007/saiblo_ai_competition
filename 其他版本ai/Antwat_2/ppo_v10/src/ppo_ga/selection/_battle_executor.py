"""生产版 BattleExecutor —— 服务于 TouchstoneSelector。

与 [tools/bisection_experiment/battle_executor.py] 的实验版同构，
区别：
  • 模拟函数 simulate_battle → 真实 _execute_battle（独立进程，PyTorch 上下文）
  • ThreadPoolExecutor → ProcessPoolExecutor(spawn)
  • Task 内交替先后手由 worker 内部根据 game_idx 决定
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional

from loguru import logger

# 从实验代码复用 Task / TaskQueue
import os
import sys
# 当前文件路径: <ppo_v10>/src/ppo_ga/selection/_battle_executor.py
# 需要上 4 级（selection → ppo_ga → src → ppo_v10），再进 tools/bisection_experiment
_TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "bisection_experiment")
)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from task_queue import Task, TaskQueue  # type: ignore  # noqa: E402

from ..battle.agent_loader import AgentLoader
from ..battle.battle_simulator import _ensure_compatible, _execute_battle


_BATTLE_TIMEOUT = 600
_WORKER_DIAG_LOG = "/tmp/ppo_v10_worker_diag.log"


def _reconfigure_loguru():
    from loguru import logger as _worker_logger
    _worker_logger.remove()
    _worker_logger.add(
        _WORKER_DIAG_LOG,
        level="WARNING",
        rotation="10 MB",
        retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )


def _touchstone_battle_worker(
    a_bytes: bytes,
    b_bytes: bytes,
    n_games: int,
    max_rounds: int,
) -> List[dict]:
    """子进程对战 worker。

    每局交替先后手：game_idx=0 → a 先手（home=a, away=b），
                  game_idx=1 → b 先手（home=b, away=a），以此类推。

    返回每局的 dict：
      {
        "result": "agent1_win"|"agent2_win"|"draw"|"error",
        "first_player": 0|1,        # 0 表示 a 先手
        "home": str, "away": str,
      }
    """
    _reconfigure_loguru()
    a_agent = AgentLoader.deserialize(a_bytes)
    b_agent = AgentLoader.deserialize(b_bytes)
    a_agent = _ensure_compatible(a_agent)
    b_agent = _ensure_compatible(b_agent)
    a_name = getattr(a_agent, "name", None) or "a"
    b_name = getattr(b_agent, "name", None) or "b"

    out: List[dict] = []
    for i in range(n_games):
        # i=0 -> a 先手 (agent1=a)；i=1 -> b 先手 (agent1=b)
        if i % 2 == 0:
            agent1, agent2 = a_agent, b_agent
            home, away = a_name, b_name
            first_player = 0
        else:
            agent1, agent2 = b_agent, a_agent
            home, away = b_name, a_name
            first_player = 0  # 在 worker 视角，agent1 先手；记账时 home=agent1
        result = _execute_battle(agent1, agent2, first_player, max_rounds)
        result["home"] = home
        result["away"] = away
        out.append(result)
    return out


class BattleExecutor:
    """从 TaskQueue 取 task → ProcessPool 并行执行 → 主线程同步 callback。

    主线程同步收集 future，因此 callback 在主线程串行调用，无需额外锁。
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        agent_bytes: Dict[str, bytes],
        callback: Callable[[str, str, dict], None],
        max_workers: int = 12,
        max_rounds: int = 512,
    ):
        self._queue = task_queue
        self._agent_bytes = agent_bytes
        self._callback = callback
        self._max_workers = max_workers
        self._max_rounds = max_rounds

        self._battles_executed: int = 0
        self._tasks_completed: int = 0
        self._cancelled_skipped: int = 0
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    @property
    def battles_executed(self) -> int:
        return self._battles_executed

    @property
    def tasks_completed(self) -> int:
        return self._tasks_completed

    @property
    def cancelled_skipped(self) -> int:
        return self._cancelled_skipped

    @property
    def elapsed_seconds(self) -> float:
        if self._end_time > 0:
            return self._end_time - self._start_time
        return time.time() - self._start_time

    def run(self) -> None:
        """主循环：阻塞直到队列空。"""
        self._start_time = time.time()
        self._battles_executed = 0
        self._tasks_completed = 0
        self._cancelled_skipped = 0

        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self._max_workers, mp_context=spawn_ctx) as pool:
            in_flight: Dict[object, Task] = {}

            while True:
                # 补充 in_flight 至 max_workers
                while len(in_flight) < self._max_workers:
                    task = self._queue.pop()
                    if task is None:
                        break
                    if task.cancelled:
                        self._cancelled_skipped += 1
                        continue
                    a_bytes = self._agent_bytes[task.player_a]
                    b_bytes = self._agent_bytes[task.player_b]
                    future = pool.submit(
                        _touchstone_battle_worker,
                        a_bytes,
                        b_bytes,
                        task.n_games,
                        self._max_rounds,
                    )
                    in_flight[future] = task

                if not in_flight:
                    break

                done = {f for f in in_flight if f.done()}
                if not done:
                    time.sleep(0.02)
                    continue

                for f in done:
                    task = in_flight.pop(f)
                    self._handle_future(f, task)

        self._end_time = time.time()

    def _handle_future(self, future, task: Task) -> None:
        try:
            results = future.result(timeout=task.n_games * _BATTLE_TIMEOUT)
        except Exception as e:
            logger.error(
                f"BattleExecutor: task {task.task_id} ({task.player_a} vs {task.player_b}) failed: {e}"
            )
            for _ in range(task.n_games):
                self._callback(task.player_a, task.player_b, {"result_code": 0})
                self._battles_executed += 1
            return

        for i, result in enumerate(results):
            if i % 2 == 0:
                home, away = task.player_a, task.player_b
            else:
                home, away = task.player_b, task.player_a

            game_result = result.get("result")
            if game_result == "agent1_win":
                code = 1
            elif game_result == "agent2_win":
                code = -1
            else:
                code = 0

            # 附加 result_code 并传递完整对战数据（含 rounds）
            result["result_code"] = code
            result["home"] = home
            result["away"] = away
            self._callback(home, away, result)
            self._battles_executed += 1

        self._tasks_completed += 1


__all__ = ["BattleExecutor", "Task", "TaskQueue"]
