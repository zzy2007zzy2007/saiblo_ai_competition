import json
import os
import hashlib
from typing import List
from dataclasses import dataclass, asdict

import numpy as np
from loguru import logger


@dataclass
class EliteSeedEntry:
    """精英种子池条目"""
    individual_id: str
    generation: int
    seed_rank: int
    trueskill_mu: float
    trueskill_sigma: float
    checkpoint_path: str
    genome_hash: str = ""


class EliteSeedPool:
    """精英种子池管理器

    仅做数据记录，不参与对战流程。用于离线分析模型迭代效果。
    """

    def __init__(self):
        self._entries: List[EliteSeedEntry] = []

    @property
    def entries(self) -> List[EliteSeedEntry]:
        return self._entries

    def add_seeds(
        self,
        seeds: list,           # List[Individual]
        elites: list,          # List[Individual]
        generation: int,
        checkpoint_dir: str,
    ) -> None:
        """批量添加本代 seeds 和 elites 到池中"""
        for seed in seeds:
            is_elite = seed in elites
            entry = EliteSeedEntry(
                individual_id=seed.id,
                generation=generation,
                seed_rank=seed.seed_rank,
                trueskill_mu=seed.trueskill_mu,
                trueskill_sigma=seed.trueskill_sigma,
                checkpoint_path=os.path.join(
                    checkpoint_dir, f"seed_{seed.seed_rank}.pt"
                ),
                genome_hash=hashlib.sha256(
                    seed.genome.weights.tobytes()
                ).hexdigest()[:16],
            )
            self._entries.append(entry)

        logger.info(
            f"EliteSeedPool: added {len(seeds)} seeds (elites={len(elites)}) "
            f"from gen {generation}, total={len(self._entries)}"
        )

    def save_state(self, filepath: str) -> None:
        """持久化到 JSON"""
        state = {
            "entries": [asdict(e) for e in self._entries]
        }
        # numpy 类型转换
        for e in state["entries"]:
            for k, v in e.items():
                if isinstance(v, np.floating):
                    e[k] = float(v)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"EliteSeedPool saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """从 JSON 加载"""
        with open(filepath, "r") as f:
            state = json.load(f)
        self._entries = [EliteSeedEntry(**e) for e in state.get("entries", [])]
        logger.info(f"EliteSeedPool loaded from {filepath}: {len(self._entries)} entries")
