import os
import json
import time
import platform
from typing import Dict, Any, List, Optional

import torch
from loguru import logger

from ..config.path_config import PathConfig
from ..evolution.population_manager import Population, Individual


class GALoggingSubsystem:
    """GA 日志子系统 —— 管理所有日志输出，使用 PathConfig 管理路径。"""

    def __init__(self, config: Dict[str, Any], path_config: PathConfig, run_id: str):
        self.config = config
        self._path_config = path_config
        self._run_id = run_id

        # 通过 PathConfig 获取各子目录
        self._training_dir = str(path_config.get_training_dir(run_id))
        self._system_dir = str(path_config.get_system_dir(run_id))
        self._generations_dir = str(path_config.get_generations_dir(run_id))
        self._tensorboard_dir = str(path_config.get_run_dir(run_id))  # tensorboard 放 run 根目录

        self._setup_loguru()

    def _setup_loguru(self):
        """配置 loguru 日志输出"""
        os.makedirs(self._training_dir, exist_ok=True)

        from loguru import logger as loguru_logger

        # 添加 handler（不 remove 默认 handler，避免影响进程内其他 loguru 用户）
        loguru_logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.get("logging", {}).get("log_level", "INFO"),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        # 主日志文件
        loguru_logger.add(
            os.path.join(self._training_dir, "ga_training_{time}.log"),
            level="INFO",
            rotation="100 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

        # 错误日志文件
        loguru_logger.add(
            os.path.join(self._training_dir, "ga_training_error_{time}.log"),
            level="ERROR",
            rotation="50 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    def log_training_start(self) -> None:
        """记录训练启动配置快照"""
        filepath = os.path.join(self._training_dir, "training_start.log")
        with open(filepath, "w") as f:
            f.write(f"Run ID: {self._run_id}\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"Python: {platform.python_version()}\n")
            f.write(f"PyTorch: {torch.__version__}\n")
            f.write(f"CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
            f.write("\n--- Config ---\n")
            for section, values in self.config.items():
                if section.startswith("_"):
                    continue
                f.write(f"[{section}]\n")
                for k, v in values.items():
                    f.write(f"  {k}: {v}\n")

    def log_generation(
        self,
        generation: int,
        population: Population,
        seeds: List[Individual],
        baseline_results: Dict[str, Dict[str, Any]],
        duration: float,
    ) -> None:
        """记录每代的日志"""
        # 1. 控制台输出
        logger.info(
            f"Generation {generation}: "
            f"best_elo={population.best_elo:.1f}, "
            f"mean_elo={population.mean_elo:.1f}, "
            f"diversity={population.diversity:.4f}, "
            f"duration={duration:.1f}s"
        )

        # 2. 保存种群统计 JSON
        self._save_population_stats(generation, population, seeds)

        # 3. 保存对战结果 JSON
        self._save_battle_results(generation, population, seeds)

        # 4. 保存基线验证结果 JSON
        if baseline_results:
            self._save_evaluation_results(generation, baseline_results)

        # 5. TensorBoard（如果启用）
        if self.config.get("logging", {}).get("tensorboard", False):
            self._write_tensorboard(generation, population, baseline_results)

    def log_training_complete(self, final_generation: int, total_time: float) -> None:
        """记录训练完成"""
        logger.info(
            f"GA Training complete: {final_generation} generations, "
            f"total_time={total_time:.1f}s"
        )

        filepath = os.path.join(self._training_dir, "training_complete.log")
        with open(filepath, "w") as f:
            f.write(f"Final generation: {final_generation}\n")
            f.write(f"Total time: {total_time:.1f}s\n")

    @property
    def system_dir(self) -> str:
        return self._system_dir

    @property
    def generations_dir(self) -> str:
        return self._generations_dir

    def close(self) -> None:
        """关闭日志子系统"""
        pass  # loguru 自动管理

    def _save_population_stats(self, generation: int, population: Population, seeds: List[Individual]) -> None:
        """保存种群统计到 JSON"""
        gen_dir = os.path.join(self._generations_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        stats = {
            "generation": generation,
            "population_size": len(population.individuals),
            "best_elo": population.best_elo,
            "mean_elo": population.mean_elo,
            "diversity": population.diversity,
            "num_seeds": len(seeds),
            "seed_elo_range": (
                seeds[-1].elo_rating if seeds else 0.0,
                seeds[0].elo_rating if seeds else 0.0,
            ),
        }

        filepath = os.path.join(gen_dir, "population_stats.json")
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2, default=str)

    def _save_battle_results(self, generation: int, population: Population, seeds: List[Individual]) -> None:
        """保存对战结果到 JSON"""
        gen_dir = os.path.join(self._generations_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        rankings = [
            {"id": ind.id, "elo": ind.elo_rating, "seed_rank": ind.seed_rank}
            for ind in sorted(population.individuals, key=lambda x: x.elo_rating, reverse=True)
        ]

        seed_list = [
            {"id": s.id, "elo": s.elo_rating, "rank": s.seed_rank, "parents": s.parent_ids}
            for s in seeds
        ]

        results = {
            "generation": generation,
            "rankings": rankings,
            "seeds": seed_list,
        }

        filepath = os.path.join(gen_dir, "battle_results.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def _save_evaluation_results(self, generation: int, baseline_results: Dict) -> None:
        """保存基线验证结果到 JSON"""
        gen_dir = os.path.join(self._generations_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        filepath = os.path.join(gen_dir, "evaluation.json")
        with open(filepath, "w") as f:
            json.dump(baseline_results, f, indent=2, default=str)

    def _write_tensorboard(self, generation: int, population: Population, baseline_results: Dict) -> None:
        """写入 TensorBoard 数据"""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(self._tensorboard_dir, "tensorboard")
            writer = SummaryWriter(tb_dir)

            writer.add_scalar("ga/best_elo", population.best_elo, generation)
            writer.add_scalar("ga/mean_elo", population.mean_elo, generation)
            writer.add_scalar("ga/diversity", population.diversity, generation)

            if baseline_results:
                for seed_id, seed_results in baseline_results.items():
                    for baseline_name, stats in seed_results.items():
                        win_rate = stats.get("agent1_wins", 0) / max(stats.get("total_battles", 1), 1)
                        writer.add_scalar(f"ga/baseline_win_rate/{baseline_name}", win_rate, generation)

            writer.close()
        except ImportError:
            logger.warning("TensorBoard not available, skipping tensorboard logging")
