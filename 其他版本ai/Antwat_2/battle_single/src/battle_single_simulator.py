import os
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from .battle_single_logger import BattleLogger, LogLevel


DEFAULT_MAX_ROUNDS = 512
BATTLE_PRINT_INTERVAL = 20


def run_single_battle_process(task):
    """
    在独立进程中执行单局对战（完全独立，不依赖模拟器）

    Args:
        task: 任务参数 (episode, seed, first_player, agent1_name, agent2_name, max_rounds, log_dir, log_level)

    Returns:
        对战结果字典
    """
    episode, seed, first_player, agent1_name, agent2_name, max_rounds, log_dir, log_level_str = task

    try:
        # 在子进程中重新加载 SDK（每个进程独立）
        sdk_paths = [
            '/root/autodl-tmp/AntWar/Ant-Game/SDK',
            os.path.join(os.path.dirname(__file__), '../../Ant-Game/SDK'),
            os.path.join(os.path.dirname(__file__), '../../../Ant-Game/SDK'),
        ]

        sdk_path = None
        for path in sdk_paths:
            if os.path.exists(path):
                sdk_path = path
                sdk_parent = os.path.dirname(path)
                if sdk_parent not in sys.path:
                    sys.path.insert(0, sdk_parent)
                if path not in sys.path:
                    sys.path.insert(0, path)
                break

        if not sdk_path:
            raise RuntimeError(f"Cannot find SDK")

        # 加载 SDK backend
        from SDK.backend.core import load_backend
        from SDK.backend.state import PythonBackendState

        backend = load_backend(prefer_native=False)
        python_state_class = PythonBackendState

        # 加载 agent（每个进程独立创建）
        from .agent_loader import is_builtin_agent, create_builtin_agent, BASELINES_PATH

        def load_agent_directly(agent_name):
            if is_builtin_agent(agent_name):
                return create_builtin_agent(agent_name)

            baseline_path = BASELINES_PATH
            agent_path = os.path.join(baseline_path, agent_name)
            ai_file = os.path.join(agent_path, 'ai.py')

            if not os.path.exists(agent_path):
                return None
            if not os.path.exists(ai_file):
                return None

            original_sys_path = sys.path.copy()
            original_modules = {}

            modules_to_remove = []
            for key in list(sys.modules.keys()):
                if key in ['ai', 'common', 'SDK', 'AI'] or \
                   key.startswith('ai.') or key.startswith('common.') or \
                   key.startswith('SDK.') or key.startswith('AI.'):
                    modules_to_remove.append(key)

            sys.path.insert(0, os.path.dirname(sdk_path))
            sys.path.insert(0, sdk_path)
            sys.path.insert(0, agent_path)

            for mod in modules_to_remove:
                if mod in sys.modules:
                    original_modules[mod] = sys.modules[mod]
                    del sys.modules[mod]

            try:
                os.chdir(agent_path)
                try:
                    from ai import create_agent
                    agent = create_agent()
                except ImportError:
                    from ai import AI
                    agent = AI()

                import torch
                for attr_name in ['network', 'neural_agent', 'model', 'nn_model']:
                    if hasattr(agent, attr_name):
                        model_obj = getattr(agent, attr_name)
                        if hasattr(model_obj, 'to'):
                            model_obj.to('cpu')
                            if hasattr(agent, 'device'):
                                agent.device = torch.device('cpu')
                        break

                return agent
            except Exception:
                return None
            finally:
                sys.path = original_sys_path
                for mod, module in original_modules.items():
                    sys.modules[mod] = module

        agent1 = load_agent_directly(agent1_name)
        agent2 = load_agent_directly(agent2_name)

        if not agent1 or not agent2:
            logger.error('battle', f"Failed to load agents for episode {episode}")
            logger.complete()
            return {
                'episode': episode,
                'agent1_name': agent1_name,
                'agent2_name': agent2_name,
                'first_player': agent1_name if first_player == 0 else agent2_name,
                'result': 'error',
                'error': 'Failed to load agents',
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': 0.0,
                'total_rounds': 0,
                'final_hp': {'agent1': 0, 'agent2': 0},
                'final_coins': {'agent1': 0, 'agent2': 0}
            }

        # 导入子进程日志记录器
        from .battle_single_logger import SubProcessLogger, LogLevel
        log_level = LogLevel.from_string(log_level_str)
        print(f"[DEBUG] log_level_str: {log_level_str}, log_level: {log_level}, log_level.value: {log_level.value}")
        sub_logger = SubProcessLogger(agent1_name, agent2_name, log_level=log_level)

        # 直接执行对战，传入子进程日志记录器以实现实时输出
        result = _execute_battle(
            agent1, agent2, agent1_name, agent2_name,
            episode, seed, first_player, max_rounds, backend, python_state_class,
            sub_logger=sub_logger
        )

        # 退出前确保日志写入
        logger.complete()
        return result

    except Exception as e:
        import traceback
        error_msg = str(e) + "\n" + traceback.format_exc()
        logger.error('battle', f"Battle {episode} failed: {e}")
        logger.complete()
        return {
            'episode': episode,
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'first_player': agent1_name if first_player == 0 else agent2_name,
            'result': 'error',
            'error': error_msg,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': 0.0,
            'total_rounds': 0,
            'final_hp': {'agent1': 0, 'agent2': 0},
            'final_coins': {'agent1': 0, 'agent2': 0}
        }


def _execute_battle(agent1, agent2, agent1_name, agent2_name,
                    episode, seed, first_player, max_rounds, backend, python_state_class,
                    sub_logger=None):
    """
    执行单局对战的核心逻辑（支持实时日志输出）

    Args:
        sub_logger: 子进程日志记录器，用于实时输出日志
    """
    battle_start_time = time.time()
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 记录对战开始（如果提供了子进程日志记录器）
    if sub_logger:
        sub_logger.log_battle_start(episode)

    result = {
        'episode': episode,
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'first_player': agent1_name if first_player == 0 else agent2_name,
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

        round_count = 0
        cumulative_earned = [0, 0]

        while not state.terminal and round_count < max_rounds:
            round_count += 1

            coins_before1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
            coins_before2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0

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

            spent1 = _calculate_ops_cost(player0_ops, state, 0)
            spent2 = _calculate_ops_cost(player1_ops, state, 1)

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
            coins_after1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
            coins_after2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0

            earned1 = coins_after1 - coins_before1 + spent1
            earned2 = coins_after2 - coins_before2 + spent2
            cumulative_earned[0] += earned1
            cumulative_earned[1] += earned2

            ops1_str = _operations_to_string(player0_ops)
            ops2_str = _operations_to_string(player1_ops)

            result['round_data'].append({
                'round': round_count,
                'hp1': hp1,
                'hp2': hp2,
                'coins_before1': coins_before1,
                'coins_before2': coins_before2,
                'coins_spent1': spent1,
                'coins_spent2': spent2,
                'coins_earned1': earned1,
                'coins_earned2': earned2,
                'coins_after1': coins_after1,
                'coins_after2': coins_after2,
                'cumulative_earned1': cumulative_earned[0],
                'cumulative_earned2': cumulative_earned[1],
                'ops1': ops1_str,
                'ops2': ops2_str
            })

            # 实时记录回合日志（如果提供了子进程日志记录器）
            if sub_logger:
                sub_logger.log_round(
                    episode, round_count, hp1, hp2, coins_after1, coins_after2,
                    ops1_str, ops2_str, first_player
                )

        result['total_rounds'] = round_count
        result['final_hp'] = _get_final_hp(state, first_player)
        result['final_coins'] = _get_final_coins(state)
        result['cumulative_earned'] = {
            'agent1': cumulative_earned[0],
            'agent2': cumulative_earned[1]
        }
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

    # 记录对战结束（如果提供了子进程日志记录器）
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
        if logger_obj:
            logger_obj.warning('agent', f"Failed to get operations: {e}")
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
    except Exception:
        pass
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
    except Exception:
        pass
    return {'agent1': 0, 'agent2': 0}


def _determine_result(state, first_player):
    try:
        winner = state.winner
        if winner == 0:
            return 'agent1_win' if first_player == 0 else 'agent2_win'
        elif winner == 1:
            return 'agent2_win' if first_player == 0 else 'agent1_win'
    except Exception:
        pass

    try:
        hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 0
        hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 0

        if hp1 > hp2:
            return 'agent1_win' if first_player == 0 else 'agent2_win'
        elif hp2 > hp1:
            return 'agent2_win' if first_player == 0 else 'agent1_win'
    except Exception:
        pass

    return 'draw'


def _calculate_ops_cost(ops, state, player):
    """
    计算一个玩家操作列表的消耗金币。使用 SDK state 读取实际塔数/科技等级，确保准确。

    Args:
        ops: 操作列表
        state: SDK PythonBackendState（操作执行前）
        player: 玩家索引 (0 或 1)

    Returns:
        spent: 消耗金币总数
    """
    from SDK.utils.constants import SuperWeaponType, TowerType

    spent = 0
    tc = state.tower_count(player)
    gl = state.bases[player].generation_level
    al = state.bases[player].ant_level

    for op in ops:
        op_type = op.op_type if hasattr(op, 'op_type') else None
        if op_type is None:
            continue

        if op_type == 11:  # BUILD_TOWER
            spent += state.build_tower_cost(tc)
            tc += 1
        elif op_type == 12:  # UPGRADE_TOWER
            target = op.arg1 if hasattr(op, 'arg1') else 0
            spent += state.upgrade_tower_cost(TowerType(int(target)))
        elif op_type == 13:  # DOWNGRADE_TOWER (refund, not spending)
            pass
        elif op_type == 21:  # USE_LIGHTNING_STORM (SuperWeaponType=1)
            spent += state.weapon_cost(SuperWeaponType.LIGHTNING_STORM)
        elif op_type == 22:  # USE_EMP_BLASTER (SuperWeaponType=2)
            spent += state.weapon_cost(SuperWeaponType.EMP_BLASTER)
        elif op_type == 23:  # USE_DEFLECTOR (SuperWeaponType=3)
            spent += state.weapon_cost(SuperWeaponType.DEFLECTOR)
        elif op_type == 24:  # USE_EMERGENCY_EVASION (SuperWeaponType=4)
            spent += state.weapon_cost(SuperWeaponType.EMERGENCY_EVASION)
        elif op_type == 31:  # UPGRADE_GENERATION_SPEED
            if gl < 2:
                spent += state.upgrade_base_cost(gl)
                gl += 1
        elif op_type == 32:  # UPGRADE_GENERATED_ANT
            if al < 2:
                spent += state.upgrade_base_cost(al)
                al += 1

    return spent


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


class BattleSingleSimulator:
    def __init__(self, logger: BattleLogger, max_rounds: int = DEFAULT_MAX_ROUNDS):
        """
        Args:
            logger: BattleLogger 实例（必须提供，不允许为 None）
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
        if self._sdk_loaded:
            return

        sdk_paths = [
            '/root/autodl-tmp/AntWar/Ant-Game/SDK',
            os.path.join(os.path.dirname(__file__), '../../Ant-Game/SDK'),
            os.path.join(os.path.dirname(__file__), '../../../Ant-Game/SDK'),
        ]

        sdk_path = None
        for path in sdk_paths:
            if os.path.exists(path):
                sdk_path = path
                break

        if sdk_path:
            sdk_parent_path = os.path.dirname(sdk_path)
            sys.path.insert(0, sdk_parent_path)
            sys.path.insert(0, sdk_path)
            self.logger.info('sdk', f"Loaded SDK from: {sdk_path}")
        else:
            raise RuntimeError(f"Cannot find SDK, tried: {sdk_paths}")

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
            agent1: agent1 实例（仅用于验证，实际在子进程中重新加载）
            agent2: agent2 实例（仅用于验证，实际在子进程中重新加载）
            agent1_name: agent1 名称
            agent2_name: agent2 名称
            num_episodes: 对战局数
            parallel_workers: 并行进程数

        Returns:
            对战结果列表
        """
        self._load_sdk()

        self.logger.info('config', f"Starting parallel battles: {num_episodes} episodes, {parallel_workers} workers")

        tasks = []
        log_level_str = self.logger._get_log_level_str()
        for episode in range(num_episodes):
            seed = episode
            tasks.append((episode, seed, 0, agent1_name, agent2_name, self.max_rounds, self.logger.get_log_dir(), log_level_str))
            tasks.append((episode + 1000, seed + 1000, 1, agent1_name, agent2_name, self.max_rounds, self.logger.get_log_dir(), log_level_str))

        results = []
        completed = 0
        total = len(tasks)

        from concurrent.futures import ProcessPoolExecutor

        try:
            # 使用多进程并行执行
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(run_single_battle_process, task): task for task in tasks}

                for future in __import__('concurrent.futures').futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.log_result(result)

                        if 'timing_data' in result and result['timing_data']:
                            self.logger.timing_start(f"battle_{result['episode']}")
                            for td in result['timing_data']:
                                self.logger.timing_start(f"agent1_ops_{result['episode']}_{td['round']}")
                                self.logger.timing_stop(f"agent1_ops_{result['episode']}_{td['round']}", td['agent1_ops_time'])
                                self.logger.timing_start(f"agent2_ops_{result['episode']}_{td['round']}")
                                self.logger.timing_stop(f"agent2_ops_{result['episode']}_{td['round']}", td['agent2_ops_time'])
                            self.logger.timing_stop(f"battle_{result['episode']}", result['duration'])

                        # 注意：回合日志已在子进程中实时记录，这里不再重复记录
                        # 只保留 round_data 用于后续数据处理
                    except Exception as e:
                        task = futures[future]
                        self.logger.error('battle', f"Battle execution failed: {task}, error: {e}")
                        result = {
                            'episode': task[0],
                            'agent1_name': agent1_name,
                            'agent2_name': agent2_name,
                            'first_player': agent1_name if task[2] == 0 else agent2_name,
                            'result': 'error',
                            'error': str(e),
                            'start_time': datetime.now().isoformat(),
                            'end_time': datetime.now().isoformat(),
                            'duration': 0.0,
                            'total_rounds': 0,
                            'final_hp': {'agent1': 0, 'agent2': 0},
                            'final_coins': {'agent1': 0, 'agent2': 0}
                        }
                        results.append(result)

                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        self.logger.info('progress', f"{completed}/{total} battles completed")

        except Exception as e:
            # 降级为顺序执行
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
                result = {
                    'episode': episode,
                    'agent1_name': agent1_name,
                    'agent2_name': agent2_name,
                    'first_player': agent1_name if first_player == 0 else agent2_name,
                    'result': 'error',
                    'error': str(e),
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration': 0.0,
                    'total_rounds': 0,
                    'final_hp': {'agent1': 0, 'agent2': 0},
                    'final_coins': {'agent1': 0, 'agent2': 0}
                }
                results.append(result)

            completed += 1
            if completed % 10 == 0 or completed == total:
                self.logger.info('progress', f"{completed}/{total} battles completed")

        results.sort(key=lambda x: x['episode'])
        return results

    def run_battle(self, agent1: Any, agent2: Any, agent1_name: str, agent2_name: str,
                   episode: int, seed: int = None,
                   first_player: int = 0) -> Dict:
        self._load_sdk()

        battle_start_time = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.logger.log_battle_start(episode)

        result = {
            'episode': episode,
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'first_player': agent1_name if first_player == 0 else agent2_name,
            'result': 'unknown',
            'start_time': start_time_str,
            'end_time': '',
            'duration': 0.0,
            'total_rounds': 0,
            'final_hp': {'agent1': 0, 'agent2': 0},
            'final_coins': {'agent1': 0, 'agent2': 0},
            'cumulative_earned': {'agent1': 0, 'agent2': 0},
            'round_data': [],
            'error': None
        }

        try:
            self.logger.timing_start('init_battle')
            game_state = self._backend.initial_state(seed=seed if seed is not None else episode,
                                                       cold_handle_rule_illegal=True)
            state = self._python_state_class(game_state)
            self.logger.timing_stop('init_battle')

            round_count = 0
            cumulative_earned = [0, 0]

            while not state.terminal and round_count < self.max_rounds:
                round_count += 1

                coins_before1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
                coins_before2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0

                if first_player == 0:
                    self.logger.timing_start(f'agent1_choose_operations_{episode}_{round_count}')
                    player0_ops = self._get_operations(agent1, state, 0)
                    self.logger.timing_stop(f'agent1_choose_operations_{episode}_{round_count}')

                    self.logger.timing_start(f'agent2_choose_operations_{episode}_{round_count}')
                    player1_ops = self._get_operations(agent2, state, 1)
                    self.logger.timing_stop(f'agent2_choose_operations_{episode}_{round_count}')
                else:
                    self.logger.timing_start(f'agent2_choose_operations_{episode}_{round_count}')
                    player0_ops = self._get_operations(agent2, state, 0)
                    self.logger.timing_stop(f'agent2_choose_operations_{episode}_{round_count}')

                    self.logger.timing_start(f'agent1_choose_operations_{episode}_{round_count}')
                    player1_ops = self._get_operations(agent1, state, 1)
                    self.logger.timing_stop(f'agent1_choose_operations_{episode}_{round_count}')

                spent1 = _calculate_ops_cost(player0_ops, state, 0)
                spent2 = _calculate_ops_cost(player1_ops, state, 1)

                self.logger.timing_start(f'resolve_turn_{episode}_{round_count}')
                state.resolve_turn(player0_ops, player1_ops)
                self.logger.timing_stop(f'resolve_turn_{episode}_{round_count}')

                hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 50
                hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 50
                coins_after1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
                coins_after2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0

                earned1 = coins_after1 - coins_before1 + spent1
                earned2 = coins_after2 - coins_before2 + spent2
                cumulative_earned[0] += earned1
                cumulative_earned[1] += earned2

                player0_ops_str = self._operations_to_string(player0_ops)
                player1_ops_str = self._operations_to_string(player1_ops)

                result['round_data'].append({
                    'round': round_count,
                    'hp1': hp1,
                    'hp2': hp2,
                    'coins_before1': coins_before1,
                    'coins_before2': coins_before2,
                    'coins_spent1': spent1,
                    'coins_spent2': spent2,
                    'coins_earned1': earned1,
                    'coins_earned2': earned2,
                    'coins_after1': coins_after1,
                    'coins_after2': coins_after2,
                    'cumulative_earned1': cumulative_earned[0],
                    'cumulative_earned2': cumulative_earned[1],
                    'ops1': player0_ops_str,
                    'ops2': player1_ops_str
                })

                self.logger.log_round(episode, round_count, hp1, hp2, coins_after1, coins_after2,
                                     player0_ops_str, player1_ops_str, first_player)

            result['total_rounds'] = round_count
            result['final_hp'] = self._get_final_hp(state, first_player)
            result['final_coins'] = self._get_final_coins(state)
            result['cumulative_earned'] = {
                'agent1': cumulative_earned[0],
                'agent2': cumulative_earned[1]
            }
            result['result'] = self._determine_result(state, first_player)

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

    def _get_operations(self, agent: Any, state: Any, player: int) -> List[Any]:
        return _get_operations(agent, state, player, self.logger)

    def _operations_to_string(self, ops) -> str:
        return _operations_to_string(ops)

    def _get_final_hp(self, state, first_player) -> Dict[str, int]:
        return _get_final_hp(state, first_player)

    def _get_final_coins(self, state) -> Dict[str, int]:
        return _get_final_coins(state)

    def _determine_result(self, state, first_player: int) -> str:
        return _determine_result(state, first_player)
