"""Batch 级训练指标写入器 - 写入 batch_metrics.jsonl"""

import json
from pathlib import Path
from typing import Any, Dict

from ..trainer.metrics_schema import MetricsSchema


class BatchMetricsWriter:
    """PPO 更新批次的训练技术指标写入器。

    职责：
    - 每次 PPO update 后将纯训练指标追加写入 batch_metrics.jsonl
    - 仅含训练技术数据（loss、entropy、grad_norm 等），不含对战业务数据
    """

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        # 预计算落盘 key 集合（类型分组）
        self._jsonl_keys = MetricsSchema.get_jsonl_keys()
        self._type_rules = self._build_type_rules()

    @staticmethod
    def _build_type_rules() -> Dict[str, str]:
        rules = {}
        for key in MetricsSchema.get_jsonl_keys():
            defn = MetricsSchema.get(key)
            if defn:
                rules[key] = defn.dtype
        return rules

    def write_batch(self, metrics: Dict[str, Any]) -> None:
        record = {}
        for key in self._jsonl_keys:
            if key in metrics:
                record[key] = MetricsSchema.format_value(key, metrics[key])

        type_ratios = {
            k: round(v, 6)
            for k, v in metrics.items()
            if k.startswith("type_") and k.endswith("_ratio")
        }
        if type_ratios:
            record["type_ratios"] = type_ratios
        type_probs = {
            k: round(v, 6)
            for k, v in metrics.items()
            if k.startswith("type_") and k.endswith("_prob")
        }
        if type_probs:
            record["type_probs"] = type_probs

        with open(self._filepath, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        pass
