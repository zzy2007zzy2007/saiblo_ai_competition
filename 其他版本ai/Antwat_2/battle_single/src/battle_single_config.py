import os
import yaml
from typing import Optional

from .battle_single_logger import LogLevel


class BattleSingleConfig:
    def __init__(self):
        self.battle_num_episodes: int = 10
        self.battle_max_rounds: int = 512
        self.battle_parallel_workers: int = 4

        self.logging_log_level: str = "INFO"
        self.logging_log_dir: str = "/tmp/battle_single/logs"
        self.logging_log_to_console: bool = True
        self.logging_enable_timing: bool = True
        self.logging_results_format: str = "both"

        self.agent1_name: Optional[str] = None
        self.agent2_name: Optional[str] = None

        self.output_result_json: str = "/tmp/battle_single/results.json"

    def load_from_file(self, config_path: str):
        if not os.path.exists(config_path):
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'battle' in config:
            battle = config['battle']
            if 'num_episodes' in battle:
                self.battle_num_episodes = battle['num_episodes']
            if 'max_rounds' in battle:
                self.battle_max_rounds = battle['max_rounds']
            if 'parallel_workers' in battle:
                self.battle_parallel_workers = battle['parallel_workers']

        if 'logging' in config:
            logging = config['logging']
            if 'log_level' in logging:
                self.logging_log_level = logging['log_level']
            if 'log_dir' in logging:
                self.logging_log_dir = logging['log_dir']
            if 'log_to_console' in logging:
                self.logging_log_to_console = logging['log_to_console']
            if 'enable_timing' in logging:
                self.logging_enable_timing = logging['enable_timing']
            if 'results_format' in logging:
                self.logging_results_format = logging['results_format']

        if 'agents' in config:
            agents = config['agents']
            if 'agent1' in agents:
                self.agent1_name = agents['agent1']
            if 'agent2' in agents:
                self.agent2_name = agents['agent2']

        if 'output' in config:
            output = config['output']
            if 'result_json' in output:
                self.output_result_json = output['result_json']

    def load_from_args(self, args):
        if hasattr(args, 'episodes') and args.episodes is not None:
            self.battle_num_episodes = args.episodes
        if hasattr(args, 'max_rounds') and args.max_rounds is not None:
            self.battle_max_rounds = args.max_rounds
        if hasattr(args, 'workers') and args.workers is not None:
            self.battle_parallel_workers = args.workers
        if hasattr(args, 'log_level') and args.log_level is not None:
            self.logging_log_level = args.log_level
        if hasattr(args, 'agents') and args.agents is not None and len(args.agents) >= 2:
            self.agent1_name = args.agents[0]
            self.agent2_name = args.agents[1]
        if hasattr(args, 'agent1') and args.agent1 is not None:
            self.agent1_name = args.agent1
        if hasattr(args, 'agent2') and args.agent2 is not None:
            self.agent2_name = args.agent2

    def get_log_level(self) -> LogLevel:
        return LogLevel.from_string(self.logging_log_level)