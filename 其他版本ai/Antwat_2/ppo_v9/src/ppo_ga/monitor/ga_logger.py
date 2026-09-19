from loguru import logger
from typing import Dict, Any, Optional


class GALogger:
    """GA 训练主日志器"""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def log_generation_summary(self, generation: int, stats: Dict[str, Any]) -> None:
        """记录代次摘要"""
        logger.info(
            f"[Gen {generation}] "
            f"best_elo={stats.get('best_elo', 0):.1f}, "
            f"mean_elo={stats.get('mean_elo', 0):.1f}, "
            f"diversity={stats.get('diversity', 0):.4f}"
        )

    def log_seed_info(self, seed_id: str, elo: float, rank: int) -> None:
        """记录种子信息"""
        logger.info(f"  Seed #{rank}: {seed_id} (ELO={elo:.1f})")
