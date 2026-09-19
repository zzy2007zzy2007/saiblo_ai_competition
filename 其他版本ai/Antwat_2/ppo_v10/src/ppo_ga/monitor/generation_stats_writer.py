import json
import os
from typing import Dict, Any
from loguru import logger


class GenerationStatsWriter:
    """种群统计写入器"""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def write_stats(self, generation: int, stats: Dict[str, Any]) -> None:
        """写入种群统计数据"""
        gen_dir = os.path.join(self._output_dir, "generations", f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        filepath = os.path.join(gen_dir, "generation_stats.json")
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2, default=str)
