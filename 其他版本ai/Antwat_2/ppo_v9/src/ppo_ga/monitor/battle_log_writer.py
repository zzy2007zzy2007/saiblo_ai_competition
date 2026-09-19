import json
import os
from typing import Dict, Any, List
from loguru import logger


class BattleLogWriter:
    """对战日志写入器"""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def write_battle_result(
        self,
        generation: int,
        individual_id: str,
        reference_name: str,
        result: str,
    ) -> None:
        """写入单条对战结果"""
        gen_dir = os.path.join(self._output_dir, "generations", f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        filepath = os.path.join(gen_dir, "battle_log.jsonl")
        entry = {
            "generation": generation,
            "individual_id": individual_id,
            "reference_name": reference_name,
            "result": result,
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
