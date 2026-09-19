import os
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .battle_logger import BattleLogger, SubProcessLogger, LogLevel
from .utils.process_isolation import process_isolation
from .utils.timing import Timer
from .utils.exception_logging import log_exception, get_error_context

DEFAULT_MAX_ROUNDS = 512
BATTLE_PRINT_INTERVAL = 20


def _make_battle_error_result(episode, agent1_name, agent2_name, first_player, error_msg):
    return {
        'episode': episode,
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'first_player': first_player,
        'result': 'error',
        'error': error_msg,
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration': 0.0,
        'total_rounds': 0,
        'final_hp': {'agent1': 0, 'agent2': 0},
        'final_coins': {'agent1': 0, 'agent2': 0}
    }


def run_single_battle_process(task):
    """
    在独立进程中执行单局对战

    Args:
        task: 任务参数 (episode, seed, first_player, agent1_name, agent2_name, max_rounds, log_dir, log_level_str, serialized_agent1, serialized_agent2)

    Returns:
        对战结果字典
    """
    episode, seed, first_player, agent1_name, agent2_name, max_rounds, log_dir, log_level_str, serialized_agent1, serialized_agent2 = task

    original_sys_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    original_cwd = os.getcwd()

    modules_to_remove = []
    for key in list(sys.modules.keys()):
        if key in ['ai', 'common', 'SDK', 'AI'] or \
           key.startswith('ai.') or key.startswith('common.') or \
           key.startswith('SDK.') or key.startswith('AI.'):
            modules_to_remove.append(key)

    removed_modules = {}
    for mod in modules_to_remove:
        if mod in sys.modules:
            removed_modules[mod] = sys.modules[mod]
            del sys.modules[mod]

    result = None
    try:
        import cloudpickle
        
        from ppo_antwar.config.path_config import PathConfig

        path_config = PathConfig()
        sdk_path = str(path_config.sdk_dir)
        sdk_parent = os.path.dirname(sdk_path)
        
        if sdk_parent not in sys.path:
            sys.path.insert(0, sdk_parent)

        from SDK.backend.core import load_backend
        from SDK.backend.state import PythonBackendState

        backend = load_backend(prefer_native=False)
        python_state_class = PythonBackendState

        agent1 = cloudpickle.loads(serialized_agent1)
        agent2 = cloudpickle.loads(serialized_agent2)

        if not agent1 or not agent2:
            logger.bind(category='battle').error(f"Failed to deserialize agents for episode {episode}")
            return _make_battle_error_result(
                episode, agent1_name, agent2_name, first_player,
                'Failed to deserialize agents'
            )

        log_level = LogLevel.from_string(log_level_str)
        sub_logger = SubProcessLogger(agent1_name, agent2_name, log_level=log_level)

        result = _execute_battle(
            agent1, agent2, agent1_name, agent2_name,
            episode, seed, first_player, max_rounds, backend, python_state_class,
            sub_logger=sub_logger
        )

        return result

    except Exception as e:
        import traceback
        error_msg = str(e) + "\n" + traceback.format_exc()
        logger.error('battle', f"Battle {episode} failed: {e}")
        return _make_battle_error_result(
            episode, agent1_name, agent2_name, first_player, error_msg
        )
    finally:
        os.chdir(original_cwd)
        sys.path = original_sys_path
        
        current_modules = set(sys.modules.keys())
        new_modules = current_modules - original_modules
        
        for mod in new_modules:
            if mod in sys.modules:
                del sys.modules[mod]
        
        for mod in removed_modules:
            if mod not in sys.modules:
                sys.modules[mod] = removed_modules[mod]
        
        logger.complete()


def _execute_battle(agent1, agent2, agent1_name, agent2_name,
                    episode, seed, first_player, max_rounds, backend, python_state_class,
                    sub_logger=None):
    """
    执行单局对战的核心逻辑

    Args:
        sub_logger: 子进程日志记录器，用于实时输出日志
    """
    battle_start_time = time.time()
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if sub_logger:
        sub_logger.log_battle_start(episode)

    result = {
        'episode': episode,
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'first_player': first_player,
        'result': 'unknown',
        'start_time': start_time_str,
        'end_time': '',
        'duration': 0.0,
        'total_rounds': 0,
        'final_hp': {'agent1': 0, 'agent2': 0},
        'final_coins': {'agent1': 0, 'agent2': 0},
        'error': None,
        'timing_data': [],
        'round_data': []
    }

    try:
        game_state = backend.initial_state(seed=seed if seed is not None else episode,
                                           cold_handle_rule_illegal=True)
        state = python_state_class(game_state)
        
        if sub_logger:
            sub_logger.debug('state', f"[DEBUG] Battle {episode} initialized: terminal={state.terminal}, winner={state.winner}, bases=[{state.bases[0].hp}, {state.bases[1].hp}]")

        round_count = 0

        while not state.terminal and round_count < max_rounds:
            round_count += 1

            if first_player == 0:
                t0_start = time.time()
                player0_ops = _get_operations(agent1, state, 0)
                t0_end = time.time()

                t1_start = time.time()
                player1_ops = _get_operations(agent2, state, 1)
                t1_end = time.time()
            else:
                t1_start = time.time()
                player0_ops = _get_operations(agent2, state, 0)
                t1_end = time.time()

                t0_start = time.time()
                player1_ops = _get_operations(agent1, state, 1)
                t0_end = time.time()

            resolve_start = time.time()
            state.resolve_turn(player0_ops, player1_ops)
            resolve_end = time.time()

            result['timing_data'].append({
                'round': round_count,
                'agent1_ops_time': t0_end - t0_start,
                'agent2_ops_time': t1_end - t1_start,
                'resolve_time': resolve_end - resolve_start
            })

            hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 50
            hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 50
            coins1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
            coins2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0

            ops1_str = _operations_to_string(player0_ops)
            ops2_str = _operations_to_string(player1_ops)

            result['round_data'].append({
                'round': round_count,
                'hp1': hp1,
                'hp2': hp2,
                'coins1': coins1,
                'coins2': coins2,
                'ops1': ops1_str,
                'ops2': ops2_str
            })

            if sub_logger:
                sub_logger.log_round(
                    episode, round_count, hp1, hp2, coins1, coins2,
                    ops1_str, ops2_str, first_player
                )
            
            if sub_logger:
                sub_logger.debug('state', f"[DEBUG] Round {round_count} after resolve_turn: terminal={state.terminal}, winner={state.winner}, bases=[{hp1}, {hp2}]")

        result['total_rounds'] = round_count
        result['final_hp'] = _get_final_hp(state, first_player)
        result['final_coins'] = _get_final_coins(state)
        result['result'] = _determine_result(state, first_player)

    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['result'] = 'error'
        if sub_logger:
            sub_logger.error('battle', f"Battle {episode} failed: {e}")

    battle_end_time = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['end_time'] = end_time_str
    result['duration'] = battle_end_time - battle_start_time

    if sub_logger:
        sub_logger.log_battle_end(result)

    return result


def _get_operations(agent, state, player, logger_obj=None):
    try:
        if hasattr(agent, 'choose_operations'):
            ops = agent.choose_operations(state, player)
            if ops and hasattr(ops, 'operations'):
                return list(ops.operations)
            return ops
        elif hasattr(agent, 'choose_bundle'):
            result = agent.choose_bundle(state, player)
            if hasattr(result, 'operations'):
                return list(result.operations)
            return result
    except Exception as e:
        error_context = get_error_context(
            player=player,
            agent_type=type(agent).__name__,
            has_choose_operations=hasattr(agent, 'choose_operations'),
            has_choose_bundle=hasattr(agent, 'choose_bundle'),
            state_type=type(state).__name__,
            state_terminal=getattr(state, 'terminal', 'unknown')
        )
        log_exception(logger_obj, "_get_operations", error_context, e)
        raise  # 根据设计原则，异常应抛向上层
    return []


def _get_final_hp(state, first_player):
    try:
        if hasattr(state, 'bases') and len(state.bases) >= 2:
            if first_player == 0:
                return {
                    'agent1': state.bases[0].hp,
                    'agent2': state.bases[1].hp
                }
            else:
                return {
                    'agent1': state.bases[1].hp,
                    'agent2': state.bases[0].hp
                }
    except Exception as e:
        error_context = get_error_context(
            first_player=first_player,
            state_type=type(state).__name__,
            has_bases=hasattr(state, 'bases'),
            bases_len=len(state.bases) if hasattr(state, 'bases') else 0
        )
        log_exception(None, "_get_final_hp", error_context, e)
    return {'agent1': 0, 'agent2': 0}


def _get_final_coins(state):
    try:
        if hasattr(state, 'coins'):
            coins = getattr(state, 'coins')
            if isinstance(coins, (list, tuple)) and len(coins) >= 2:
                return {
                    'agent1': coins[0],
                    'agent2': coins[1]
                }
    except Exception as e:
        error_context = get_error_context(
            state_type=type(state).__name__,
            has_coins=hasattr(state, 'coins'),
            coins_type=type(getattr(state, 'coins', None)).__name__ if hasattr(state, 'coins') else None
        )
        log_exception(None, "_get_final_coins", error_context, e)
    return {'agent1': 0, 'agent2': 0}


def _determine_result(state, first_player):
    try:
        hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 0
        hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 0

        if hp1 > hp2:
            return 'agent1_win' if first_player == 0 else 'agent2_win'
        elif hp2 > hp1:
            return 'agent2_win' if first_player == 0 else 'agent1_win'
        else:
            return 'draw'
    except Exception as e:
        error_context = get_error_context(
            first_player=first_player,
            state_type=type(state).__name__,
            has_bases=hasattr(state, 'bases'),
            bases_len=len(state.bases) if hasattr(state, 'bases') else 0
        )
        log_exception(None, "_determine_result", error_context, e)

    return 'draw'


def _operations_to_string(ops) -> str:
    if ops is None:
        return "None"
    if isinstance(ops, list):
        op_list = []
        for op in ops:
            if hasattr(op, 'op_type'):
                op_type = op.op_type
                arg0 = getattr(op, 'arg0', '')
                arg1 = getattr(op, 'arg1', '')
                op_list.append(f"{op_type}({arg0},{arg1})")
            else:
                op_list.append(str(op))
        return "[" + ", ".join(op_list) + "]"
    return str(ops)


class BattleSimulator:
    def __init__(self, logger: BattleLogger, max_rounds: int = DEFAULT_MAX_ROUNDS):
        """
        Args:
            logger: BattleLogger 实例
            max_rounds: 单场对战最大回合数
        """
        if logger is None:
            raise ValueError("logger cannot be None")
        self.logger = logger
        self.max_rounds = max_rounds
        self._sdk_loaded = False
        self._backend = None
        self._python_state_class = None
        self._lock = threading.Lock()

    def _load_sdk(self):
        """加载 SDK"""
        if self._sdk_loaded:
            return

        from ppo_antwar.config.path_config import PathConfig

        path_config = PathConfig()
        sdk_path = str(path_config.sdk_dir)
        sdk_parent_path = os.path.dirname(sdk_path)
        
        sys.path.insert(0, sdk_parent_path)
        sys.path.insert(0, sdk_path)
        self.logger.info('sdk', f"Loaded SDK from: {sdk_path}")

        from SDK.backend.core import load_backend
        from SDK.backend.state import PythonBackendState

        self._backend = load_backend(prefer_native=False)
        self._python_state_class = PythonBackendState
        self._sdk_loaded = True

    def run_battles_parallel(self, agent1: Any, agent2: Any, agent1_name: str, agent2_name: str,
                            num_episodes: int, parallel_workers: int = 4) -> List[Dict]:
        """
        使用多进程并行执行对战

        Args:
            agent1: agent1 实例
            agent2: agent2 实例
            agent1_name: agent1 名称
            agent2_name: agent2 名称
            num_episodes: 对战局数（每局包含先手/后手各一次）
            parallel_workers: 并行进程数

        Returns:
            对战结果列表
        """
        self._load_sdk()

        self.logger.info('config', f"Starting parallel battles: {num_episodes} episodes, {parallel_workers} workers")

        import cloudpickle
        serialized_agent1 = cloudpickle.dumps(agent1)
        serialized_agent2 = cloudpickle.dumps(agent2)

        tasks = []
        log_level_str = self.logger._get_log_level_str()
        for episode in range(num_episodes):
            seed = episode
            tasks.append((episode, seed, 0, agent1_name, agent2_name, self.max_rounds, self.logger.get_log_dir(), log_level_str, serialized_agent1, serialized_agent2))
            tasks.append((episode + 1000, seed + 1000, 1, agent1_name, agent2_name, self.max_rounds, self.logger.get_log_dir(), log_level_str, serialized_agent1, serialized_agent2))

        results = []
        completed = 0
        total = len(tasks)

        from concurrent.futures import ProcessPoolExecutor

        try:
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(run_single_battle_process, task): task for task in tasks}

                for future in __import__('concurrent.futures').futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.log_result(result)

                    except Exception as e:
                        task = futures[future]
                        self.logger.error('battle', f"Battle execution failed: {task}, error: {e}")
                        result = _make_battle_error_result(
                            task[0], agent1_name, agent2_name, task[2], str(e)
                        )
                        results.append(result)

                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        self.logger.info('progress', f"{completed}/{total} battles completed")

        except Exception as e:
            self.logger.warning('parallel', f"多进程执行失败，降级为顺序执行: {e}")
            results = self._run_battles_sequential(agent1, agent2, agent1_name, agent2_name, num_episodes)

        results.sort(key=lambda x: x['episode'])
        return results

    def _run_battles_sequential(self, agent1: Any, agent2: Any, agent1_name: str, agent2_name: str,
                                num_episodes: int) -> List[Dict]:
        """
        顺序执行对战（降级方案）

        Args:
            agent1: agent1 实例
            agent2: agent2 实例
            agent1_name: agent1 名称
            agent2_name: agent2 名称
            num_episodes: 对战局数

        Returns:
            对战结果列表
        """
        self.logger.info('config', f"Starting sequential battles: {num_episodes} episodes")

        tasks = []
        for episode in range(num_episodes):
            seed = episode
            tasks.append((episode, seed, 0))
            tasks.append((episode + 1000, seed + 1000, 1))

        results = []
        completed = 0
        total = len(tasks)

        for task in tasks:
            episode, seed, first_player = task
            try:
                result = self.run_battle(
                    agent1, agent2,
                    agent1_name, agent2_name,
                    episode, seed=seed, first_player=first_player
                )
                results.append(result)
                self.logger.log_result(result)
            except Exception as e:
                self.logger.error('battle', f"Battle execution failed: {task}, error: {e}")
                result = _make_battle_error_result(
                    episode, agent1_name, agent2_name, first_player, str(e)
                )
                results.append(result)

            completed += 1
            if completed % 10 == 0 or completed == total:
                self.logger.info('progress', f"{completed}/{total} battles completed")

        results.sort(key=lambda x: x['episode'])
        return results

    def run_battle(self, agent1: Any, agent2: Any, agent1_name: str, agent2_name: str,
                   episode: int, seed: int = None,
                   first_player: int = 0) -> Dict:
        """
        执行单局对战

        Args:
            agent1: agent1 实例
            agent2: agent2 实例
            agent1_name: agent1 名称
            agent2_name: agent2 名称
            episode: 回合编号
            seed: 随机种子
            first_player: 先手玩家 (0: agent1, 1: agent2)

        Returns:
            对战结果字典
        """
        self._load_sdk()

        battle_start_time = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.logger.log_battle_start(episode)

        result = {
            'episode': episode,
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'first_player': first_player,
            'result': 'unknown',
            'start_time': start_time_str,
            'end_time': '',
            'duration': 0.0,
            'total_rounds': 0,
            'final_hp': {'agent1': 0, 'agent2': 0},
            'final_coins': {'agent1': 0, 'agent2': 0},
            'error': None
        }

        try:
            self.logger.timing_start('init_battle')
            game_state = self._backend.initial_state(seed=seed if seed is not None else episode,
                                                       cold_handle_rule_illegal=True)
            state = self._python_state_class(game_state)
            self.logger.timing_stop('init_battle')

            round_count = 0

            while not state.terminal and round_count < self.max_rounds:
                round_count += 1

                if first_player == 0:
                    self.logger.timing_start(f'agent1_ops_{episode}_{round_count}')
                    player0_ops = _get_operations(agent1, state, 0, self.logger)
                    self.logger.timing_stop(f'agent1_ops_{episode}_{round_count}')

                    self.logger.timing_start(f'agent2_ops_{episode}_{round_count}')
                    player1_ops = _get_operations(agent2, state, 1, self.logger)
                    self.logger.timing_stop(f'agent2_ops_{episode}_{round_count}')
                else:
                    self.logger.timing_start(f'agent2_ops_{episode}_{round_count}')
                    player0_ops = _get_operations(agent2, state, 0, self.logger)
                    self.logger.timing_stop(f'agent2_ops_{episode}_{round_count}')

                    self.logger.timing_start(f'agent1_ops_{episode}_{round_count}')
                    player1_ops = _get_operations(agent1, state, 1, self.logger)
                    self.logger.timing_stop(f'agent1_ops_{episode}_{round_count}')

                self.logger.timing_start(f'resolve_{episode}_{round_count}')
                state.resolve_turn(player0_ops, player1_ops)
                self.logger.timing_stop(f'resolve_{episode}_{round_count}')

                hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 50
                hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 50
                coins1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
                coins2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0

                player0_ops_str = _operations_to_string(player0_ops)
                player1_ops_str = _operations_to_string(player1_ops)

                self.logger.log_round(episode, round_count, hp1, hp2, coins1, coins2,
                                     player0_ops_str, player1_ops_str, first_player)

            result['total_rounds'] = round_count
            result['final_hp'] = _get_final_hp(state, first_player)
            result['final_coins'] = _get_final_coins(state)
            result['result'] = _determine_result(state, first_player)

        except Exception as e:
            import traceback
            result['error'] = str(e)
            result['result'] = 'error'
            self.logger.log_error(episode, e)
            self.logger.debug('trace', f"Stack trace: {traceback.format_exc()}")

        battle_end_time = time.time()
        end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['end_time'] = end_time_str
        result['duration'] = battle_end_time - battle_start_time

        self.logger.log_battle_end(result)
        self.logger.log_timing_summary(result['duration'])

        return result

