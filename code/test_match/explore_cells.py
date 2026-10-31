"""把"建塔位置"统一到**策略顺序下标**上（STRATEGIC_BUILD_ORDER[player] 的 idx），
从而可以把 P0/P1 两个座位的数据合起来看（两边的区域形状并不全等，坐标不能直接 mirror）。

数据源：
  - 246 局自对弈 replay（我们 vs 我们）
  - 16 局 vs 冠军的 trace log（我们的 lv / 对手的 lv_opp，逐回合塔集合）
产出：match_results/explore_20261102/cells/ 下的分布表
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from SDK.utils.constants import STRATEGIC_BUILD_ORDER  # noqa: E402

ORDER = {p: [tuple(c) for c in STRATEGIC_BUILD_ORDER[p]] for p in (0, 1)}
IDX = {p: {c: i for i, c in enumerate(ORDER[p])} for p in (0, 1)}


def idx_of(player: int, x: int, y: int):
    return IDX[player].get((x, y))


def selfplay_builds():
    """246 局自对弈：所有 build op + 每局第一个 build。"""
    all_b, first_b, n_games = Counter(), Counter(), 0
    for fp in sorted(glob.glob("training_history/collect_v2_s1/*.replay.json")):
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        n_games += 1
        first = {0: None, 1: None}
        for st in d["steps"]:
            for p in (0, 1):
                for op in (st.get(f"op{p}") or []):
                    if int(op[0]) == 11:
                        i = idx_of(p, int(op[1]), int(op[2]))
                        if i is not None:
                            all_b[i] += 1
                            if first[p] is None:
                                first[p] = i
        for p in (0, 1):
            if first[p] is not None:
                first_b[first[p]] += 1
    return all_b, first_b, n_games


RE_LINE = re.compile(
    r"round=(\d+) hp=\[(\d+), (\d+)\] coins=\[(\d+), (\d+)\] towers=(\d+) "
    r"lv=\[(.*?)\] lv_opp=\[(.*?)\] mismatch=(\d+)")
RE_TRIPLE = re.compile(r"\((\d+), (\d+), (-?\d+)(?:, (\d+))?\)")


def champion_towers():
    """16 局 vs 冠军：我们/对手 每个"格点首次出现"算一次建塔；统计顺序下标分布。"""
    ours_all, ours_first = Counter(), Counter()
    opp_all, opp_first = Counter(), Counter()
    games = 0
    for fp in sorted(glob.glob("match_results/champion_e1/stderr/ai*_az_bridge_ai_seed*.stderr.log")):
        me = None
        seen_me, seen_opp = set(), set()
        first_me, first_opp = True, True
        for line in open(fp, encoding="utf-8", errors="replace"):
            if me is None:
                m = re.search(r"\[azai\] init: player=(\d+)", line)
                if m:
                    me = int(m.group(1))
                    continue
            m = RE_LINE.search(line)
            if not m:
                continue
            lv = [(int(a), int(b), int(c)) for a, b, c, _d in RE_TRIPLE.findall(m.group(7))]
            lvo = [(int(a), int(b), int(c)) for a, b, c, _d in RE_TRIPLE.findall(m.group(8))]
            for x, y, _l in lv:
                if (x, y) not in seen_me:
                    seen_me.add((x, y))
                    i = idx_of(me, x, y)
                    if i is not None:
                        ours_all[i] += 1
                        if first_me:
                            ours_first[i] += 1
                            first_me = False
            for x, y, _l in lvo:
                if (x, y) not in seen_opp:
                    seen_opp.add((x, y))
                    i = idx_of(1 - me, x, y)
                    if i is not None:
                        opp_all[i] += 1
                        if first_opp:
                            opp_first[i] += 1
                            first_opp = False
        if me is not None:
            games += 1
    return ours_all, ours_first, opp_all, opp_first, games


def dump(counter, n, top=20):
    tot = sum(counter.values()) or 1
    rows = []
    for i, c in counter.most_common(top):
        rows.append((i, ORDER[0][i] if i < len(ORDER[0]) else None, c, c / tot))
    return rows


def main():
    out = "match_results/explore_20261102/cells"
    os.makedirs(out, exist_ok=True)
    lines = []

    def P(s=""):
        lines.append(str(s))
        print(s)

    ab, fb, ng = selfplay_builds()
    ca, cf, oa, of, cg = champion_towers()
    P(f"# 建塔位置分布（统一到 STRATEGIC_BUILD_ORDER[player] 的下标）")
    P(f"- 自对弈 replay：{ng} 局，build op 总数 {sum(ab.values())}")
    P(f"- vs 冠军：{cg} 局；我们建塔格点 {sum(ca.values())}，冠军 {sum(oa.values())}")
    P()
    P("## 1. 全部建塔（自对弈，两个座位合并）top20")
    P("idx\tcell_for_P0\tcount\tshare")
    for i, cell, c, sh in dump(ab, None):
        P(f"{i}\t{cell}\t{c}\t{sh:.1%}")
    P()
    P("## 2. 每局第一座塔（开局选择）top20")
    P("idx\tcell_for_P0\tcount\tshare")
    for i, cell, c, sh in dump(fb, None):
        P(f"{i}\t{cell}\t{c}\t{sh:.1%}")
    P()
    P("## 3. vs 冠军：我们 的建塔分布 top15")
    P("idx\tcell_for_P0\tcount\tshare")
    for i, cell, c, sh in dump(ca, None, 15):
        P(f"{i}\t{cell}\t{c}\t{sh:.1%}")
    P()
    P("## 4. vs 冠军：冠军 的建塔分布 top15")
    P("idx\tcell_for_P0\tcount\tshare")
    for i, cell, c, sh in dump(oa, None, 15):
        P(f"{i}\t{cell}\t{c}\t{sh:.1%}")
    P()
    with open(os.path.join(out, "cells_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    json.dump(dict(selfplay_all={str(k): v for k, v in ab.items()},
                   selfplay_first={str(k): v for k, v in fb.items()},
                   champ_ours={str(k): v for k, v in ca.items()},
                   champ_opp={str(k): v for k, v in oa.items()},
                   order={str(p): ORDER[p] for p in (0, 1)}),
              open(os.path.join(out, "cells.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
