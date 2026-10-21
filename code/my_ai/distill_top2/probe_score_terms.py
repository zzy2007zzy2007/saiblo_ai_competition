"""核对 score 里那些"手写静态加成"在实际对局里到底有没有生效。

关注 `mean_c1_bonus_score`（c1_root_bonus）、`mean_future_*`（future threat 系列）、
`mean_lightning_bonus_score`、`mean_reactive_operation_penalty` 的非零比例。
"""
import collections
import json
import sys

path = sys.argv[1]
fields = ["mean_c1_bonus_score", "mean_lightning_bonus_score",
          "mean_reactive_operation_penalty", "mean_future_threat_adjustment_score",
          "mean_future_base_damage_score", "mean_worker_threat_score",
          "mean_combat_threat_score", "mean_tower_value_score"]
nonzero = collections.Counter()
total = 0
c1_max = 0.0
for line in open(path, encoding="utf-8", errors="replace"):
    if "base_heuristic" not in line:
        continue
    record = json.loads(line)
    total += 1
    for field in fields:
        value = abs(float(record.get(field, 0.0)))
        if value > 1e-9:
            nonzero[field] += 1
        if field == "mean_c1_bonus_score":
            c1_max = max(c1_max, value)

print(f"plan 行数 = {total}")
for field in fields:
    print(f"  {field:<38} 非零 {nonzero[field]:>8}  ({100.0 * nonzero[field] / total:6.2f}%)")
print(f"  mean_c1_bonus_score 最大绝对值 = {c1_max}")
