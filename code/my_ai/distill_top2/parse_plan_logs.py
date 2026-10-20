"""解析亚军 plans stderr 日志，搞清候选集结构 + 分数分解。

用法： python parse_plan_logs.py <log 文件或目录> [--max-lines N]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-lines", type=int, default=0, help="0=不限制")
    a = ap.parse_args()
    p = Path(a.path)
    files = [p] if p.is_file() else sorted(p.glob("*.stderr.log"))

    schema = None
    keys = collections.Counter()
    n_lines = 0
    n_plans = 0
    per_round = collections.Counter()      # (seed, round, player) -> #plans
    rounds = collections.Counter()
    names = collections.Counter()
    ranks = collections.Counter()
    score_parts = collections.Counter()
    examples = []

    for f in files:
        n_f = 0
        with f.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"kind"' not in line or '"plan"' not in line:
                    continue
                i = line.find("{")
                if i < 0:
                    continue
                try:
                    r = json.loads(line[i:].strip())
                except Exception:
                    continue
                if r.get("kind") != "plan":
                    continue
                n_lines += 1
                n_f += 1
                for k in r:
                    keys[k] += 1
                if schema is None:
                    schema = list(r.keys())
                per_round[(r.get("seed"), r.get("round"), r.get("player"))] += 1
                rounds[r.get("round")] += 1
                names[r.get("name")] += 1
                ranks[r.get("rank")] += 1
                for k in ("base_heuristic", "lure_heuristic", "lightning_heuristic",
                          "operation_penalty", "followup_penalty", "heuristic", "score"):
                    if k in r:
                        score_parts[k] += 1
                if len(examples) < 2:
                    examples.append(r)
                if a.max_lines and n_lines >= a.max_lines:
                    break
        n_plans += 1
        print(f"  {f.name}: {n_f} plan lines", flush=True)
        if a.max_lines and n_lines >= a.max_lines:
            break

    print(f"\n=== 总 plan 行数: {n_lines} ===")
    print(f"\n=== 字段（schema）===\n{schema}")
    print(f"\n=== 字段出现次数 ===\n{dict(keys)}")
    print(f"\n=== 每题平均候选数 ===")
    if per_round:
        v = list(per_round.values())
        print(f"  题数={len(v)}  min={min(v)} max={max(v)} 均值={sum(v)/len(v):.1f}")
    print(f"\n=== rank 分布 ===\n{dict(sorted(ranks.items()))}")
    print(f"\n=== 不同的候选名数量: {len(names)} ===")
    print(f"  top10: {names.most_common(10)}")
    print(f"\n=== 分数分量出现次数 ===\n{dict(score_parts)}")
    print(f"\n=== 示例（第 1 条）===")
    if examples:
        print(json.dumps(examples[0], ensure_ascii=False, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
