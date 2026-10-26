"""离线检验 v2：亚军走法是否在我们的候选集里（2026-10-26）。

修正了 v1 的两个问题：
  ① 日志里的 ops 是 `Operation(op_type=<...: 11>, arg0=4, arg1=2)` 形式，不是 Python 列表；
  ② 先验证 dump 的 `n` 与日志的 `round` 是否对齐（用"我方 chosen == 日志里我方的 ops"来判断）。

输出：
  · 对齐率（我方 dump 的 chosen 与日志该回合 ops 一致的比例）
  · 候选覆盖率（亚军该回合的操作是否出现在我们的 160 个候选里），按 op_type 分组
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
OP_RE = re.compile(r"op_type=<OperationType\.\w+: (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
LINE_RE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文=.*? -> (\[.*?\])\s*$")


def parse_log(path: Path):
    per = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        rnd, side = int(m.group(1)), m.group(2)
        ops = [(int(a), int(b), int(c)) for a, b, c in OP_RE.findall(m.group(4))]
        per.setdefault(rnd, {})[side] = ops
    return per


def first_op_rounds(per, side):
    """{round: 第一个操作}"""
    return {r: s[side][0] for r, s in per.items() if s.get(side)}


def load_dump(path: Path):
    recs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def cand_key(c):
    op = c[0] if isinstance(c[0], (list, tuple)) else c
    return (int(op[0]), int(op[1]), int(op[2]))


def main() -> int:
    logs = sorted((REPO / "match_results/ladder_logs/VS_RU_dump").glob("*.log"))
    total_hit = total_n = 0
    per_op = {}
    align_hit = align_n = 0
    for lp in logs:
        stem = lp.stem  # 7 / 7r
        seed = int(re.sub(r"\D", "", stem))
        per = parse_log(lp)
        our_cands = sorted((REPO / "training_history/az_dump_ru").glob(f"az_seed{seed:05d}_p*.jsonl"))
        for f in our_cands:
            who = int(f.stem.split("_p")[1])
            our_side = f"p{who}"
            ru_side = "p1" if our_side == "p0" else "p0"
            our_ops = first_op_rounds(per, our_side)
            ru_ops = first_op_rounds(per, ru_side)
            recs = load_dump(f)
            # ① 对齐检验：dump[n].chosen 对应的候选 是否等于日志里我方该回合的第一个操作
            for rec in recs[:60]:
                n = int(rec.get("n", -1))
                cands = rec.get("cands") or []
                ch = rec.get("chosen")
                if not cands or ch is None or n not in our_ops:
                    continue
                try:
                    k = cand_key(cands[int(ch)])
                except Exception:
                    continue
                align_n += 1
                align_hit += int(k == our_ops[n])
            # ② 覆盖率：亚军操作是否在 dump[round] 的候选里
            for rec in recs:
                n = int(rec.get("n", -1))
                if n not in ru_ops:
                    continue
                keys = set()
                for c in rec.get("cands") or []:
                    try:
                        keys.add(cand_key(c))
                    except Exception:
                        pass
                k = ru_ops[n]
                total_n += 1
                total_hit += int(k in keys)
                per_op.setdefault(k[0], [0, 0])[0] += int(k in keys)
                per_op[k[0]][1] += 1
    print(f"  dump.n 与日志 round 的对齐率: {align_hit}/{align_n} = {100*align_hit/max(1,align_n):.1f}%")
    print(f"  亚军走法被我们候选集覆盖: {total_hit}/{total_n} = {100*total_hit/max(1,total_n):.1f}%")
    for op, (h, n) in sorted(per_op.items()):
        print(f"    op_type={op:<3} {h}/{n} = {100*h/n:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
