"""对战模拟器 - 使用 ProcessPoolExecutor 实现多进程并行对战"""

import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

from loguru import logger

from .agent_loader import AgentLoader
from .battle_config import BaselineBattleConfig

_BATTLE_TIMEOUT_SECONDS = 300


class BattleSimulator:
    """对战模拟器 - 使用 ProcessPoolExecutor 实现多进程并行对战"""

    def __init__(self, config: BaselineBattleConfig):
        self.config = config
        self._max_workers = getattr(config, "max_workers", 12)

    def execute_single_battle(
        self,
        agent1: Any,
        agent2: Any,
        first_player: int = 0,
        max_steps: int = 200,
    ) -> Dict[str, Any]:
        """在当前进程/线程中执行单局对战。

        不创建子进程，不嵌套 ProcessPool。
        调用方自行负责并行调度（ThreadPool / 手动多线程等）。

        Args:
            agent1: GA agent
            agent2: 参照物 agent
            first_player: 0 表示 agent1 先手，1 表示 agent1 后手
            max_steps: 最大回合数

        Returns:
            {'result': 'agent1_win'|'agent2_win'|'draw'|'error',
             'first_player': 0|1,
             'total_rounds': int,
             'final_hp': {'agent1_hp': int, 'agent2_hp': int},
             'duration': float,
             'rounds': [...]}
        """
        agent1 = _ensure_compatible(agent1)
        agent2 = _ensure_compatible(agent2)
        return _execute_battle(
            agent1=agent1,
            agent2=agent2,
            first_player=first_player,
            max_rounds=max_steps or self.config.max_rounds,
        )

    def run_battles(
        self,
        agent1: Any,
        agent2: Any,
        n_battles: int,
    ) -> List[Dict[str, Any]]:
        """执行对战，优先多进程并行，失败时降级为顺序执行。

        每个 episode 生成 2 场对战（agent1 先手、agent1 后手各一局）。
        """
        agent1_bytes = AgentLoader.serialize(agent1)
        agent2_bytes = AgentLoader.serialize(agent2)

        max_rounds = self.config.max_rounds

        # 每个 episode 生成 2 场：first_player=0（agent1 先手）和 first_player=1（agent1 后手）
        total_tasks = n_battles * 2

        results = self._run_battles_parallel(
            agent1_bytes,
            agent2_bytes,
            total_tasks,
            max_rounds,
        )

        if results and all(r.get("error") for r in results):
            logger.warning(
                "All parallel battles failed"
            )
        return results

    def _run_battles_parallel(
        self,
        agent1_bytes: bytes,
        agent2_bytes: bytes,
        total_tasks: int,
        max_rounds: int,
    ) -> List[Dict[str, Any]]:
        results = []
        max_workers = min(total_tasks, self._max_workers)

        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_ctx
        ) as executor:
            futures = []
            for i in range(total_tasks):
                first_player = i % 2  # 0: agent1 先手, 1: agent1 后手
                future = executor.submit(
                    run_single_battle_process,
                    agent1_bytes,
                    agent2_bytes,
                    first_player,
                    max_rounds,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=_BATTLE_TIMEOUT_SECONDS)
                    if "error" in result:
                        logger.error(f"Battle failed: {result['error']}")
                    results.append(result)
                except Exception as e:
                    logger.error(f"Battle process failed: {e}")
                    results.append({"error": str(e)})

        return results

def run_single_battle_process(
    agent1_bytes: bytes,
    agent2_bytes: bytes,
    first_player: int,
    max_rounds: int,
) -> Dict[str, Any]:
    """在子进程中执行单场对战"""
    # spawn 子进程不继承主进程 loguru sink，需重新配置
    from loguru import logger as _worker_logger
    _worker_logger.remove()
    _worker_logger.add(
        "/tmp/ppo_v9_worker_diag.log",
        level="WARNING",
        rotation="10 MB",
        retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )
    agent1 = AgentLoader.deserialize(agent1_bytes)
    agent2 = AgentLoader.deserialize(agent2_bytes)
    agent1 = _ensure_compatible(agent1)
    agent2 = _ensure_compatible(agent2)
    return _execute_battle(agent1, agent2, first_player, max_rounds)


class SDKAgentAdapter:
    """将 SDK BaseAgent 包装为 ppov2 兼容的 AntWarAgent 接口。

    SDK 外部 agent（如 sample）的 choose_operations(self, state, player, bundles=None)
    与 ppov2 原生 AntWarAgent 的 choose_operations(self, state) 签名不兼容 —— 缺少
    player 参数。此适配器在调用时自动注入 player_id，使外部 agent 可在 _execute_battle
    的对战循环中与原生 agent 统一使用 _get_ops(agent, state) 调用。
    """

    def __init__(self, agent):
        self._agent = agent

    @property
    def name(self):
        return getattr(self._agent, "name", type(self._agent).__qualname__)

    @property
    def player_id(self):
        return self._agent.player_id

    @player_id.setter
    def player_id(self, value):
        self._agent.player_id = value

    def choose_operations(self, state):
        return self._agent.choose_operations(state, player=self.player_id)


def _ensure_compatible(agent):
    """检测 agent 类型，对非 ppov2 原生的 SDK BaseAgent 做适配封装。"""
    from .ant_war_agent import AntWarAgent

    if isinstance(agent, AntWarAgent):
        return agent
    return SDKAgentAdapter(agent)


def _serialize_ops(ops) -> List[Dict]:
    """将 Operation 列表序列化为可 pickled 的 dict 格式。

    兼容两种 Operation 实现：op_type（model）和 type（forecast）。
    """
    if not ops:
        return [{"type": "NOOP"}]
    result = []
    for op in ops:
        op_type = getattr(op, "op_type", None) or getattr(op, "type", None)
        info = {"type": op_type.name if hasattr(op_type, "name") else str(op_type)}
        x = getattr(op, "x", None)
        y = getattr(op, "y", None)
        if x is not None:
            info["x"] = x
        if y is not None:
            info["y"] = y
        result.append(info)
    return result


def _get_ops(agent, state):
    return agent.choose_operations(state)


def _filter_valid_operations(ops, state, player: int):
    """过滤非法操作：通过 can_apply_operation 做最终校验，拦截语义层非法操作。
    """

    if not ops:
        return ops if ops is not None else []

    valid = []
    filtered_count = 0
    for op in ops:
        if state.can_apply_operation(player, op):
            valid.append(op)
        else:
            filtered_count += 1

    if filtered_count > 0:
        logger.warning(
            f"[Battle v2] Player {player}: {filtered_count}/{len(ops)} operations filtered by validity check"
        )

    return valid


def _execute_battle(
    agent1: Any,
    agent2: Any,
    first_player: int,
    max_rounds: int,
) -> Dict[str, Any]:
    """执行单场对战。

    Args:
        first_player: 0 表示 agent1 作为先手（player 0），1 表示 agent1 作为后手（player 1）。
                      结果固定返回 agent1 视角的 HP（agent1_hp = agent1, agent2_hp = agent2）。

    Returns:
        {'result': 'agent1_win'|'agent2_win'|'draw'|'error',
         'first_player': 0|1,
         'total_rounds': int,
         'final_hp': {'agent1_hp': agent1_hp, 'agent2_hp': agent2_hp},
         'duration': float}
    """
    import time
    import os
    import sys
    from ppo_ga.config.path_config import PathConfig

    # 加载 SDK（与 v1 对齐：使用 PythonBackendState.resolve_turn 同时结算）
    path_config = PathConfig()
    sdk_path = str(path_config.sdk_path)
    sdk_parent = os.path.dirname(sdk_path)
    if sdk_parent not in sys.path:
        sys.path.insert(0, sdk_parent)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    from SDK.backend.core import load_backend
    from SDK.backend.state import PythonBackendState

    backend = load_backend(prefer_native=False)

    agent1.player_id = first_player
    agent2.player_id = 1 - first_player

    start_time = time.time()
    game_state = backend.initial_state(cold_handle_rule_illegal=True)
    state = PythonBackendState(game_state)
    round_count = 0
    rounds_data = []
    illegal_total_1 = 0
    illegal_total_2 = 0

    while not state.terminal and round_count < max_rounds:
        try:

            ops_1 = _get_ops(agent1, state)
            ops_2 = _get_ops(agent2, state)
            # 记录本回合每个 agent 的非法动作发生次数
            illegal_this_round_1 = agent1.illegal_action_count - illegal_total_1
            illegal_this_round_2 = agent2.illegal_action_count - illegal_total_2
            illegal_total_1 = agent1.illegal_action_count
            illegal_total_2 = agent2.illegal_action_count
            ops_1 = _filter_valid_operations(ops_1, state, agent1.player_id)
            ops_2 = _filter_valid_operations(ops_2, state, agent2.player_id)

            # ── 操作前采集 ──
            coins_before_p1 = state.coins[agent1.player_id]
            coins_before_p2 = state.coins[agent2.player_id]
            twr_hp_before_p1 = sum(t.hp for t in state.towers_of(agent1.player_id))
            twr_hp_before_p2 = sum(t.hp for t in state.towers_of(agent2.player_id))
            base_hp_before_p1 = state.bases[agent1.player_id].hp
            base_hp_before_p2 = state.bases[agent2.player_id].hp

            # ── 执行操作（分步：先手 → 后手） ──
            if first_player == 0:
                state.apply_operation_list(agent1.player_id, ops_1)
                if not state.terminal:
                    state.apply_operation_list(agent2.player_id, ops_2)
            else:
                state.apply_operation_list(agent2.player_id, ops_2)
                if not state.terminal:
                    state.apply_operation_list(agent1.player_id, ops_1)

            # ── 操作后采集（mid 快照） ──
            coins_mid_p1 = state.coins[agent1.player_id]
            coins_mid_p2 = state.coins[agent2.player_id]
            twr_hp_mid_p1 = sum(t.hp for t in state.towers_of(agent1.player_id))
            twr_hp_mid_p2 = sum(t.hp for t in state.towers_of(agent2.player_id))
            base_hp_mid_p1 = state.bases[agent1.player_id].hp
            base_hp_mid_p2 = state.bases[agent2.player_id].hp

            # ── 推进回合 ──
            if not state.terminal:
                state.advance_round()
            round_count += 1

            # ── 回合后采集 ──
            coins_after_p1 = state.coins[agent1.player_id]
            coins_after_p2 = state.coins[agent2.player_id]
            twr_hp_after_p1 = sum(t.hp for t in state.towers_of(agent1.player_id))
            twr_hp_after_p2 = sum(t.hp for t in state.towers_of(agent2.player_id))
            base_hp_after_p1 = state.bases[agent1.player_id].hp
            base_hp_after_p2 = state.bases[agent2.player_id].hp

            # ── 计算派生值 ──
            # 操作花费 = 初始 − mid（正值表示支出）
            ops_cost_p1 = coins_before_p1 - coins_mid_p1
            ops_cost_p2 = coins_before_p2 - coins_mid_p2
            # 本回合赚取 = mid → after 的金币增量（纯 advance_round 收入）
            coins_earned_p1 = coins_after_p1 - coins_mid_p1
            coins_earned_p2 = coins_after_p2 - coins_mid_p2
            # 塔伤害 = mid → after 的 HP 减少量（纯蚂蚁+武器伤害）
            twr_hp_dmg_p1 = twr_hp_mid_p1 - twr_hp_after_p1
            twr_hp_dmg_p2 = twr_hp_mid_p2 - twr_hp_after_p2
            # 基地伤害 = mid → after
            base_hp_dmg_p1 = base_hp_mid_p1 - base_hp_after_p1
            base_hp_dmg_p2 = base_hp_mid_p2 - base_hp_after_p2

            p1_towers = (len(list(state.towers_of(agent1.player_id))))
            p2_towers = (len(list(state.towers_of(agent2.player_id))))

            rounds_data.append(
                {
                    "round": round_count - 1,
                    "agent1_name": getattr(agent1, "name", type(agent1).__qualname__),
                    "agent2_name": getattr(agent2, "name", type(agent2).__qualname__),
                    "agent1_ops": _serialize_ops(ops_1),
                    "agent2_ops": _serialize_ops(ops_2),
                    "agent1_illegal": illegal_this_round_1,
                    "agent2_illegal": illegal_this_round_2,
                    "agent1_hp": base_hp_after_p1,
                    "agent2_hp": base_hp_after_p2,
                    "agent1_hp_before": base_hp_before_p1,
                    "agent2_hp_before": base_hp_before_p2,
                    "agent1_hp_dmg": base_hp_dmg_p1,
                    "agent2_hp_dmg": base_hp_dmg_p2,
                    "agent1_twr_hp_before": twr_hp_before_p1,
                    "agent2_twr_hp_before": twr_hp_before_p2,
                    "agent1_twr_hp_after": twr_hp_after_p1,
                    "agent2_twr_hp_after": twr_hp_after_p2,
                    "agent1_twr_hp_dmg": twr_hp_dmg_p1,
                    "agent2_twr_hp_dmg": twr_hp_dmg_p2,
                    "agent1_coins": coins_after_p1,
                    "agent2_coins": coins_after_p2,
                    "agent1_coins_before": coins_before_p1,
                    "agent2_coins_before": coins_before_p2,
                    "agent1_ops_cost": ops_cost_p1,
                    "agent2_ops_cost": ops_cost_p2,
                    "agent1_coins_earned": coins_earned_p1,
                    "agent2_coins_earned": coins_earned_p2,
                    "agent1_towers": p1_towers,
                    "agent2_towers": p2_towers,
                }
            )

        except Exception as e:
            return {
                "result": "error",
                "error": f"Battle round {round_count} failed: {e}",
                "first_player": first_player,
                "total_rounds": round_count,
            }

    # 始终从 agent1 视角记录 HP
    bases = state.bases
    agent1_hp, agent2_hp = (bases[agent1.player_id].hp, bases[agent2.player_id].hp)

    if state.terminal:
        if state.winner is not None:
            if agent1.player_id == state.winner:
                result = "agent1_win"
            else:
                result = "agent2_win"
        else:
            result = "draw"
    else:
        result = "draw"

    duration = time.time() - start_time

    return {
        "result": result,
        "first_player": first_player,
        "total_rounds": round_count,
        "final_hp": {"agent1_hp": agent1_hp, "agent2_hp": agent2_hp},
        "duration": duration,
        "rounds": rounds_data,
        "illegal_actions_agent1": illegal_total_1,
        "illegal_actions_agent2": illegal_total_2,
    }
