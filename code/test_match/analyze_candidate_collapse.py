"""把 `probe_candidate_collapse.py` 产出的多份 JSON 汇成一张表（只读）。

回答一个问题：**"去重后只剩 1 条候选"的占比，随哪个开关/改法下降多少？**

用法：
    python code/test_match/analyze_candidate_collapse.py \
        --arm "BASE=match_results/collapse/fullscan_B.json" \
        --arm "MC_CELLS=0=match_results/collapse/fullscan_B_mccells0.json" \
        ... --markdown
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def stats(d: dict) -> dict:
    rows = d["rows"]
    n = len(rows)
    nc = Counter(r["c"]["n_cand"] for r in rows)
    pin = Counter(str(r["a"]["pinned"]) for r in rows)
    sm = sorted(nc.elements())
    chk = [r for r in rows if "replay_match" in r]
    ok = sum(1 for r in chk if r["replay_match"])
    return {
        "n": n,
        "p1": 100.0 * nc.get(1, 0) / n,
        "median": sm[n // 2],
        "mean": sum(sm) / n,
        "ge28": 100.0 * sum(v for k, v in nc.items() if k >= 28) / n,
        "eq48": 100.0 * nc.get(48, 0) / n,
        "pin": pin.most_common(4),
        "replay": (ok, len(chk)),
        "n_dist": dict(sorted(nc.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="标签=json 路径（可重复）")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    if not args.arm:
        ap.error("至少给一个 --arm")

    out = []
    for spec in args.arm:
        label, _, path = spec.rpartition("=")      # 标签里允许出现 '='，按**最后一个** '=' 切
        d = load(path)
        s = stats(d)
        s["label"] = label
        s["tag"] = d.get("tag", "")
        out.append(s)

    if args.markdown:
        print("| 配置（相对 BASE 的单变量） | 决策点 | **n_cand==1** | 中位 | 均值 | ≥28 | =48 | 钉住的类 top |")
        print("|---|---|---|---|---|---|---|---|")
        for s in out:
            pin = " / ".join(f"{c}:{100.0 * v / s['n']:.1f}%" for c, v in s["pin"])
            print(f"| {s['label']} | {s['n']} | **{s['p1']:.1f}%** | {s['median']} | "
                  f"{s['mean']:.1f} | {s['ge28']:.1f}% | {s['eq48']:.1f}% | {pin} |")
    else:
        for s in out:
            print(f"\n### {s['label']}  (n={s['n']}, tag={s['tag']})")
            print(f"  n_cand==1 = {s['p1']:.1f}%   中位 {s['median']}  均值 {s['mean']:.1f}  "
                  f"≥28 = {s['ge28']:.1f}%   =48 = {s['eq48']:.1f}%")
            print(f"  重放保真度（n_cand==1 ⇒ 候选==实战动作）：{s['replay'][0]}/{s['replay'][1]}")
            print(f"  钉住的类：{s['pin']}")
            print(f"  n_cand 分布：{s['n_dist']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
