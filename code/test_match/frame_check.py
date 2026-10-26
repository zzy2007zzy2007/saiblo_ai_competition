"""确定"亚军坐标系 → 我们坐标系"的变换，并据此计算候选覆盖率（2026-10-26）。

背景：亚军（C++ exe）收到的是它自己视角的局面，其 (x, y) 与我们的绝对坐标可能差一个镜像/转置；
v1 检验在 identity 下覆盖率为 0/254，明显是坐标系问题。

做法：
  1. 只用建塔（op_type=11）——坐标语义最明确；
  2. 对 6 种候选变换分别算"亚军动作落在我们候选集里"的比例；
  3. 取覆盖率最高的变换，再算全部 op_type 的覆盖率。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
OP_RE = re.compile(r"op_type=<OperationType\.\w+: (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
LINE_RE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文=.*? -> (\[.*?\])\s*$")

TRANSFORMS = {
    "identity": lambda x, y: (x, y),
    "mirror_x": lambda x, y: (18 - x, y),
    "mirror_y": lambda x, y: (x, 18 - y),
    "mirror_xy": lambda x, y: (18 - x, 18 - y),
    "transpose": lambda x, y: (y, x),
    "transpose_mirror": lambda x, y: (18 - y, x),
}


def parse_log(path: Path):
    per = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        ops = [(int(a), int(b), int(c)) for a, b, c in OP_RE.findall(m.group(4))]
        if ops:
            per.setdefault(int(m.group(1)), {})[m.group(2)] = ops
    return per


def load_records(path: Path):
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


def first_ops(rec) -> list[tuple[int, int, int]]:
    """把一个候选规整成操作列表（HOLD 即空候选 → 返回空表）"""
    out = []
    for c in rec.get("cands") or []:
        if not isinstance(c, (list, tuple)) or not c:
            out.append(None)
            continue
        op = c[0]
        if not isinstance(op, (list, tuple)) or len(op) < 3:
            out.append(None)
            continue
        out.append((int(op[0]), int(op[1]), int(op[2])))
    return out


def collect_pairs():
    """→ [(our_side, n, ru_op, cand_ops)]"""
    pairs = []
    for lp in sorted((REPO / "match_results/ladder_logs/VS_RU_dump").glob("*.log")):
        seed = int(re.sub(r"\D", "", lp.stem))
        per = parse_log(lp)
        for f in sorted((REPO / "training_history/az_dump_ru").glob(f"az_seed{seed:05d}_p*.jsonl")):
            who = int(f.stem.split("_p")[1])
            ru = "p1" if who == 0 else "p0"
            for rec in load_records(f):
                n = int(rec.get("n", -1))
                side = per.get(n, {}).get(ru)
                if not side:
                    continue
                pairs.append((who, n, side[0], first_ops(rec)))
    return pairs


def main() -> int:
    pairs = collect_pairs()
    print(f"  可用 (回合, 亚军动作, 我们的候选) 三元组: {len(pairs)}", flush=True)

    print("\n=== 建塔(op=11) 在各变换下的覆盖率 ===")
    best = ("identity", -1)
    for name, fn in TRANSFORMS.items():
        hit = tot = 0
        for _, _, ru_op, cands in pairs:
            if ru_op[0] != 11:
                continue
            cells = {(o[1], o[2]) for o in cands if o and o[0] == 11}
            if not cells:
                continue
            tot += 1
            hit += int(fn(ru_op[1], ru_op[2]) in cells)
        pct = 100 * hit / max(1, tot)
        print(f"  {name:<16} {hit}/{tot} = {pct:5.1f}%")
        if pct > best[1]:
            best = (name, pct)

    name, pct = best
    fn = TRANSFORMS[name]
    print(f"\n  最佳变换: {name}（建塔覆盖 {pct:.1f}%）")

    print("\n=== 全部 op_type 的覆盖率（用最佳变换）===")
    per_op = {}
    hit = tot = 0
    for _, _, ru_op, cands in pairs:
        op_t, x, y = ru_op
        keys = {o for o in cands if o}
        if not keys:
            continue
        if op_t == 11:
            target = (11, fn(x, y)[0], fn(x, y)[1])
        else:
            target = (op_t, x, y)
        tot += 1
        ok = int(target in keys)
        hit += ok
        per_op.setdefault(op_t, [0, 0])[0] += ok
        per_op.setdefault(op_t, [0, 0])[1] += 1
    print(f"  总覆盖: {hit}/{tot} = {100*hit/max(1,tot):5.1f}%")
    for op_t, (h, n) in sorted(per_op.items()):
        print(f"    op_type={op_t:<3} {h}/{n} = {100*h/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
