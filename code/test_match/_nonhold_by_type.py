#!/usr/bin/env python
"""把"非 HOLD 注入"按**动作类型**拆开（2026-10-29，父任务任务 1；纯离线，不跑对局）。

数据来源：E3 批（`docs/nonhold_inject_ab_20261029.md`）A 臂的逐回合日志

```
[oracle] inject round=R action=<act> dup=<0/1> new=<0/1> chosen=<act> hit=<0/1>
```

口径（写死，事后不改）：

| 量 | 定义 |
|---|---|
| "注入次数" | 该类型出现在 `action=` 里的回合数（**按 action 的 op 类型集合分类**） |
| 分类 | 按 action 里**出现的 op_type 集合**分类：单 op 用它的 type；多 op 用 `multi:` + 类型组合。`op_type` 编码：11 建塔 / 12 升级 / 13 降级 / 21 闪电 |
| "采纳次数" | `hit=1`（我们最终发出去的那一手 == 它给的那一手） |
| 采纳率 | 采纳 / 注入 |
| 占全部采纳 | 该类型采纳数 / 全部采纳数 |

⚠️ 这批日志**没有** hp/coins 快照（`AZAI_TRACE` 没开），所以"被采纳时当回合的血量/局势"这一项
按任务书"没有就跳过"处理，只给**回合分布**（前/中/后期）作为廉价替代。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P -u code/test_match/_nonhold_by_type.py --tag NONHOLD_INJECT_A \
   --out match_results/us_vs_external/_nonhold_by_type.md
```
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INJ_RE = re.compile(r"\[oracle\] inject round=(\d+) action=(\S+) dup=(\S+) new=(\S+) "
                    r"chosen=(\S+) hit=(\d+)")
OP_RE = re.compile(r"\((\d+),(-?\d+),(-?\d+)\)")
TYPE_NAME = {"11": "BUILD_TOWER", "12": "UPGRADE_TOWER", "13": "DOWNGRADE_TOWER",
             "21": "USE_LIGHTNING_STORM", "22": "USE_EMP_BLASTER", "23": "USE_DEFLECTOR",
             "24": "USE_EMERGENCY_EVASION"}


def classify(action: str) -> str:
    ops = OP_RE.findall(action)
    if not ops:
        return "HOLD/空"
    types = [t for (t, _a, _b) in ops]
    if len(types) == 1:
        return f"{types[0]} {TYPE_NAME.get(types[0], '?')}"
    key = "+".join(sorted(set(types)))
    return f"multi[{key}] " + "+".join(TYPE_NAME.get(t, "?") for t in key.split("+"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="NONHOLD_INJECT_A")
    ap.add_argument("--glob", default="", help="默认 match_results/ai?_az_bridge_ai_seed*_<tag>.stderr.log")
    ap.add_argument("--out", default="match_results/us_vs_external/_nonhold_by_type.md")
    args = ap.parse_args()

    pat = args.glob or f"ai?_az_bridge_ai_seed*_{args.tag}.stderr.log"
    files = sorted(REPO.joinpath("match_results").glob(pat))
    if not files:
        print(f"!! 没找到 {pat}（先确认 E3 批的 stderr 还在 match_results/ 下）")
        return 2

    cnt = collections.Counter()          # type -> injected
    hit = collections.Counter()          # type -> adopted (hit=1)
    rounds: dict[str, list[int]] = collections.defaultdict(list)
    hit_rounds: dict[str, list[int]] = collections.defaultdict(list)
    n_games = 0
    n_lines = 0
    n_new = n_dup = 0

    for f in files:
        n_games += 1
        txt = f.read_text(encoding="utf-8", errors="replace")
        for m in INJ_RE.finditer(txt):
            rnd, act, dup, new, _chosen, h = m.groups()
            n_lines += 1
            n_new += int(new == "1")
            n_dup += int(dup == "1")
            k = classify(act)
            cnt[k] += 1
            rounds[k].append(int(rnd))
            if h == "1":
                hit[k] += 1
                hit_rounds[k].append(int(rnd))

    tot_inj = sum(cnt.values())
    tot_hit = sum(hit.values())
    lines: list[str] = []

    def P(s: str = "") -> None:
        print(s)
        lines.append(s)

    P("\n# 非 HOLD 注入的**动作类型拆解**（自动生成）\n")
    P(f"- 日志文件：`match_results/{pat}`（**{n_games}** 局）")
    P(f"- `[oracle] inject` 行数 = **{n_lines}**（`new=1` {n_new}、`dup=1` {n_dup}）")
    P(f"- 全部注入 = **{tot_inj}**，其中被采纳（`hit=1`）= **{tot_hit}**"
      f"（采纳率 {100.0 * tot_hit / max(1, tot_inj):.1f}%）\n")
    P("| 动作类型 | 注入次数 | 占全部注入 | 采纳次数 | **采纳率** | 占全部采纳 | 注入回合中位 | 采纳回合中位 |")
    P("|---|---|---|---|---|---|---|---|")
    for k, v in cnt.most_common():
        h = hit.get(k, 0)
        P(f"| `{k}` | {v} | {100.0 * v / max(1, tot_inj):.1f}% | {h} | "
          f"**{100.0 * h / max(1, v):.1f}%** | {100.0 * h / max(1, tot_hit):.1f}% | "
          f"{statistics.median(rounds[k]) if rounds[k] else '-'} | "
          f"{statistics.median(hit_rounds[k]) if hit_rounds[k] else '-'} |")
    P(f"| **合计** | **{tot_inj}** | 100% | **{tot_hit}** | "
      f"**{100.0 * tot_hit / max(1, tot_inj):.1f}%** | 100% | — | — |")

    # 单 op vs 多 op
    single = sum(v for k, v in cnt.items() if not k.startswith("multi") and k != "HOLD/空")
    multi = sum(v for k, v in cnt.items() if k.startswith("multi"))
    P(f"\n- 单 op 注入 **{single}**（{100.0 * single / max(1, tot_inj):.1f}%）、"
      f"多 op 注入 **{multi}**（{100.0 * multi / max(1, tot_inj):.1f}%）"
      f"、HOLD/空 **{cnt.get('HOLD/空', 0)}**（`AZAI_ORACLE_SKIP_HOLD=1` 下应为 0）")

    # 回合分布（前/中/后期，按 512 回合上限切）
    P("\n## 采纳的回合分布（替代『当局血量/局势』——本批没开 `AZAI_TRACE`）\n")
    buckets = {"r<128": 0, "128≤r<256": 0, "256≤r<384": 0, "r≥384": 0}
    for k, rs in hit_rounds.items():
        for r in rs:
            if r < 128:
                buckets["r<128"] += 1
            elif r < 256:
                buckets["128≤r<256"] += 1
            elif r < 384:
                buckets["256≤r<384"] += 1
            else:
                buckets["r≥384"] += 1
    P("| 区间 | 采纳次数 | 占比 |")
    P("|---|---|---|")
    for k, v in buckets.items():
        P(f"| {k} | {v} | {100.0 * v / max(1, tot_hit):.1f}% |")

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
