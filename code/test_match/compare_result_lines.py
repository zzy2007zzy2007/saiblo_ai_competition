"""对比两份 `match_via_sdk` 日志的 `RESULT` 行（**逐字段**；把 AI 标签归一化后再比）。

2026-10-28；报告 `docs/placebo_inject_ab_20261028.md` §4（"默认行为不变"的对局级证据）。

为什么要把标签归一化：`RESULT` 行里的 `p0=/_p1=winner=` 是**AI 文件名 stem**（`_label_of`）。
"改动前/改动后"两次跑用的是**两个不同文件名**（`_az_bridge_ai_head.py` vs `az_bridge_ai.py`，
前者是 `git show HEAD:` 的原样副本）⇒ 标签必然不同，但它不是行为差异。
真正要比的是 `winner_side / verdict / rounds / terminal / base_hp / coins / illegal`
以及 AI 侧自检行（`illegal-summary`：checked_ops/holds/by_side/by_kind/mismatch）。

用法：
  python code/test_match/compare_result_lines.py <before.log> <after.log>
返回 0 = 逐字段相同（标签已归一化）；1 = 有差异/缺行。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RE_RESULT = re.compile(
    r"RESULT seed=(?P<seed>\d+) .*?winner=(?P<winner>\S+) winner_side=(?P<ws>\S+) "
    r"verdict=(?P<verdict>\S+).*?rounds=(?P<rounds>\d+) terminal=(?P<term>\S+) "
    r"base_hp=(?P<hp0>-?\d+),(?P<hp1>-?\d+) coins=(?P<c0>-?\d+),(?P<c1>-?\d+)"
    r"(?: illegal=(?P<illegal>\d+))?")
FIELDS = ("seed", "ws", "verdict", "rounds", "term", "hp0", "hp1", "c0", "c1", "illegal")


def _norm(txt: str, labels: set[str]) -> str:
    out = txt
    for lab in sorted(labels, key=len, reverse=True):
        if lab:
            out = out.replace(lab, "<AI>")
    # 墙钟读数不是行为：`shadow_ms=4.5` 这类每局都在抖，必须剔掉才能比"行为"
    out = re.sub(r"\bshadow_ms=[\d.]+", "shadow_ms=<t>", out)
    return out


def parse(path: Path) -> dict | None:
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = RE_RESULT.search(txt)
    if not m:
        return None
    d = {k: m.group(k) for k in FIELDS}
    d["_raw"] = m.group(0)
    d["_winner"] = m.group("winner")
    ills = re.findall(r"illegal-summary .*", txt)
    d["_illsum"] = ills[-1] if ills else ""
    return d


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: compare_result_lines.py <before.log> <after.log>")
        return 2
    bp, ap = Path(sys.argv[1]), Path(sys.argv[2])
    b, a = parse(bp), parse(ap)
    if not b or not a:
        print(f"❌ 缺 RESULT 行：before={bool(b)} after={bool(a)}")
        return 1
    labels = {b["_winner"], a["_winner"]}
    print(f"[cmp] before = {bp}")
    print(f"[cmp] after  = {ap}")
    print(f"[cmp] 归一化的 AI 标签 = {sorted(labels)}")
    print(f"[cmp] before: {_norm(b['_raw'], labels)}")
    print(f"[cmp] after : {_norm(a['_raw'], labels)}")
    ok = True
    for k in FIELDS:                       # 只比行为字段（标签不参与）
        if b[k] != a[k]:
            ok = False
            print(f"[cmp] ❌ 字段不同 {k}: before={b[k]!r} after={a[k]!r}")
    bs, as_ = _norm(b["_illsum"], labels), _norm(a["_illsum"], labels)
    print(f"[cmp] before 自检: {bs}")
    print(f"[cmp] after  自检: {as_}")
    if bs != as_:
        ok = False
        print("[cmp] ❌ 自检行不同（如果不是某局没收尾，就是发出去的 op 序列变了）")
    if ok:
        print("[cmp] ✅ 所有行为字段逐字段相同"
              "（winner_side/verdict/rounds/terminal/base_hp/coins/illegal + 自检行）")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
