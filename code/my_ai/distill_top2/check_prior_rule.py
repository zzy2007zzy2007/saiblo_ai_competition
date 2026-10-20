"""验证「亚军的先验只有两个非零成分」这个论断（不训练，直接拿日志逐条核对）。

注意：不能只用 key 判断"是否降级"——日志里大量 `13:...`（降级）计划的 heuristic 是 0，
因为只有**降级狙击塔**才吃 `sniper_downgrade_penalty`。所以逐项核对：

 A. base_heuristic  == 80 * [key == "hold"]          （每个决策恰好一个 hold）
 B. lure_heuristic / lightning_heuristic / followup_penalty 全为 0
 C. operation_penalty ∈ {0, 400}
 D. heuristic == base_heuristic + lure + lightning - operation_penalty - followup_penalty  （恒等式）
 E. heuristic == 80 * [hold] - operation_penalty

若 A~E 全部 100% 成立，则先验被完全刻画成"hold 加分 + 降级狙击塔罚分"两条谓词，
主线的"先验可复刻"不需要训练来证明。

用法：
    python check_prior_rule.py match_results/ai0_runnerup_seed732.stderr.log [...]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    total = 0
    ok_a = ok_b = ok_c = ok_d = ok_e = 0
    penalty_values: collections.Counter = collections.Counter()
    downgrade_no_penalty = 0
    downgrade_with_penalty = 0
    examples: list[tuple] = []

    for path_str in args.logs:
        path = Path(path_str)
        print(f"[读] {path.name}")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "base_heuristic" not in line:
                    continue
                try:
                    plan = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                key = plan["key"]
                is_hold = (key == "hold")
                base = float(plan["base_heuristic"])
                lure = float(plan["lure_heuristic"])
                lightning = float(plan["lightning_heuristic"])
                penalty = float(plan["operation_penalty"])
                followup = float(plan["followup_penalty"])
                heuristic = float(plan["heuristic"])

                if abs(base - (80.0 if is_hold else 0.0)) < 1e-9:
                    ok_a += 1
                if abs(lure) < 1e-9 and abs(lightning) < 1e-9 and abs(followup) < 1e-9:
                    ok_b += 1
                penalty_values[round(penalty, 4)] += 1
                if abs(penalty) < 1e-9 or abs(penalty - 400.0) < 1e-9:
                    ok_c += 1
                if abs(heuristic - (base + lure + lightning - penalty - followup)) < 1e-9:
                    ok_d += 1
                if abs(heuristic - ((80.0 if is_hold else 0.0) - penalty)) < 1e-9:
                    ok_e += 1
                elif len(examples) < 10:
                    examples.append((key, heuristic, base, lure, lightning, penalty, followup))

                # 降级但没吃罚分 = 降的不是狙击塔
                has_downgrade = any(segment.split(":")[0] == "13"
                                    for segment in key.split("|")
                                    if not segment.startswith("F:"))
                if has_downgrade:
                    if abs(penalty) < 1e-9:
                        downgrade_no_penalty += 1
                    else:
                        downgrade_with_penalty += 1

    def line(name: str, count: int) -> str:
        return f"  {name} {count:>8} / {total}  ({100.0 * count / max(total, 1):8.4f}%)"

    print(f"\n总 plan 数 = {total}")
    print(line("A base == 80*[hold]           ", ok_a))
    print(line("B lure/lightning/followup == 0", ok_b))
    print(line("C operation_penalty ∈ {0,400} ", ok_c))
    print(line("D 恒等式（四分量之和）        ", ok_d))
    print(line("E heuristic == 80*[hold]-pen  ", ok_e))
    print(f"\noperation_penalty 取值分布: {dict(penalty_values)}")
    print(f"带降级的计划: 无罚分 {downgrade_no_penalty}（降的不是狙击塔） / "
          f"有罚分 {downgrade_with_penalty}（降狙击塔）")
    if examples:
        print("\nE 不成立的例子 (key, heuristic, base, lure, lightning, penalty, followup):")
        for row in examples:
            print("  ", row)
    else:
        print("\n==> A~E 全部成立：先验 = 「hold +80」与「降级狙击塔 -400」两条谓词，可精确复刻")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
