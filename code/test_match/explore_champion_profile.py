"""解析 `match_results/champion_e1/stderr/ai*_az_bridge_ai_seed*.stderr.log` 的逐回合 trace，
把「我们 vs 冠军」的同局行为并排量化：塔数/升级/拆除/金币/血量轨迹/塔位。

trace 行格式（az_bridge_ai.py:2901）：
  round=R hp=[h0, h1] coins=[c0, c1] towers=T lv=[(x,y,level,type),...] lv_opp=[(x,y,level),...]

只读。产出 match_results/explore_20261102/champion/：
  per_game.tsv / champion_summary.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

RE_LINE = re.compile(
    r"round=(\d+) hp=\[(\d+), (\d+)\] coins=\[(\d+), (\d+)\] towers=(\d+) "
    r"lv=\[(.*?)\] lv_opp=\[(.*?)\] mismatch=(\d+)"
)
RE_TRIPLE = re.compile(r"\((\d+), (\d+), (-?\d+)(?:, (\d+))?\)")
RE_INIT = re.compile(r"\[azai\] init: player=(\d+) seed=(\d+)")


def parse_file(fp):
    me = None
    seed = None
    rows = []
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if me is None:
                m = RE_INIT.search(line)
                if m:
                    me, seed = int(m.group(1)), int(m.group(2))
                    continue
            m = RE_LINE.search(line)
            if not m:
                continue
            rnd = int(m.group(1))
            hp = (int(m.group(2)), int(m.group(3)))
            coins = (int(m.group(4)), int(m.group(5)))
            lv = [(int(a), int(b), int(c), (int(d) if d else -1)) for a, b, c, d in RE_TRIPLE.findall(m.group(7))]
            lvo = [(int(a), int(b), int(c)) for a, b, c, _d in RE_TRIPLE.findall(m.group(8))]
            rows.append(dict(round=rnd, hp=hp, coins=coins, lv=lv, lvo=lvo,
                             mismatch=int(m.group(9))))
    return me, seed, rows


def side_events(seq):
    """seq: [(round, {(x,y): level})] -> 建/拆/升/降 计数 + 塔数序列"""
    builds = demol = ups = downs = 0
    prev = {}
    counts = []
    for rnd, cur in seq:
        for pos, lv in cur.items():
            if pos not in prev:
                builds += 1
            elif lv > prev[pos]:
                ups += 1
            elif lv < prev[pos]:
                downs += 1
        for pos in prev:
            if pos not in cur:
                demol += 1
        prev = cur
        counts.append(len(cur))
    return dict(builds=builds, demol=demol, ups=ups, downs=downs, counts=counts)


def summarize_file(fp):
    me, seed, rows = parse_file(fp)
    if me is None or not rows:
        return None
    opp = 1 - me
    seq_me, seq_opp = [], []
    for r in rows:
        seq_me.append((r["round"], {(x, y): lv for x, y, lv, _t in r["lv"]}))
        seq_opp.append((r["round"], {(x, y): lv for x, y, lv in r["lvo"]}))
    ev_me, ev_opp = side_events(seq_me), side_events(seq_opp)
    hp_me = [r["hp"][me] for r in rows]
    hp_opp = [r["hp"][opp] for r in rows]
    cm = [r["coins"][me] for r in rows]
    co = [r["coins"][opp] for r in rows]
    # 塔位：(x,y) -> 在该局出现过的最大 level
    pos_me = Counter({p: max(d.values()) + 1 for _r, d in seq_me if d for p in d})
    pos_opp = Counter({p: max(d.values()) + 1 for _r, d in seq_opp if d for p in d})
    return dict(
        file=os.path.basename(fp), seed=seed, seat=me, rounds=len(rows),
        hp_me_end=hp_me[-1], hp_opp_end=hp_opp[-1],
        me=ev_me, opp=ev_opp,
        coins_me_mean=statistics.mean(cm), coins_opp_mean=statistics.mean(co),
        coins_me_ge90=sum(1 for c in cm if c >= 90) / len(cm),
        coins_opp_ge90=sum(1 for c in co if c >= 90) / len(co),
        coins_me_max=max(cm), coins_opp_max=max(co),
        towers_me_mean=statistics.mean(ev_me["counts"]) if ev_me["counts"] else 0,
        towers_opp_mean=statistics.mean(ev_opp["counts"]) if ev_opp["counts"] else 0,
        towers_me_max=max(ev_me["counts"]) if ev_me["counts"] else 0,
        towers_opp_max=max(ev_opp["counts"]) if ev_opp["counts"] else 0,
        hp_me_first_drop=next((r["round"] for r in rows if r["hp"][me] < 50), None),
        hp_opp_first_drop=next((r["round"] for r in rows if r["hp"][opp] < 50), None),
        pos_me=dict(pos_me), pos_opp=dict(pos_opp),
        mismatch_last=rows[-1]["mismatch"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="match_results/champion_e1/stderr")
    ap.add_argument("--glob", default="ai*_az_bridge_ai_seed*.stderr.log")
    ap.add_argument("--out", default="match_results/explore_20261102/champion")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    os.makedirs(a.out, exist_ok=True)
    res = [r for r in (summarize_file(f) for f in files) if r]
    print(f"[info] parsed {len(res)}/{len(files)} files")

    out = []
    def P(s=""):
        out.append(str(s)); print(s)

    P(f"# 我们 vs 冠军：同局行为对照（{len(res)} 局）")
    P()
    n = len(res)
    wins = sum(1 for r in res if r["hp_me_end"] > r["hp_opp_end"])
    P(f"- 血量大者胜口径：我们赢 {wins}/{n}")
    P(f"- 终局血量 我们 mean={statistics.mean(r['hp_me_end'] for r in res):.1f} "
      f"冠军 mean={statistics.mean(r['hp_opp_end'] for r in res):.1f}")
    P(f"- 回合数 mean={statistics.mean(r['rounds'] for r in res):.0f}")
    P()
    P("## 1. 塔经济（每局均值）")
    for key, label in (("me", "我们"), ("opp", "冠军")):
        b = statistics.mean(r[key]["builds"] for r in res)
        d = statistics.mean(r[key]["demol"] for r in res)
        u = statistics.mean(r[key]["ups"] for r in res)
        dn = statistics.mean(r[key]["downs"] for r in res)
        tm = statistics.mean(r["towers_%s_mean" % key] for r in res)
        tx = statistics.mean(r["towers_%s_max" % key] for r in res)
        P(f"- {label}: 建塔={b:.1f} 拆除={d:.1f} 升级={u:.1f} 降级={dn:.1f} "
          f"塔数均值={tm:.2f} 塔数峰值={tx:.1f}")
    P()
    P("## 2. 金币")
    P(f"- 我们：mean={statistics.mean(r['coins_me_mean'] for r in res):.1f} "
      f">=90 占比={statistics.mean(r['coins_me_ge90'] for r in res):.1%} "
      f"max={max(r['coins_me_max'] for r in res)}")
    P(f"- 冠军：mean={statistics.mean(r['coins_opp_mean'] for r in res):.1f} "
      f">=90 占比={statistics.mean(r['coins_opp_ge90'] for r in res):.1%} "
      f"max={max(r['coins_opp_max'] for r in res)}")
    P()
    P("## 3. 血量")
    P(f"- 我们先掉血：{sum(1 for r in res if r['hp_me_first_drop'] is not None)}/{n} 局，"
      f"首次掉血回合（有的话）中位={statistics.median([r['hp_me_first_drop'] for r in res if r['hp_me_first_drop'] is not None]) if any(r['hp_me_first_drop'] is not None for r in res) else None}")
    P(f"- 冠军先掉血：{sum(1 for r in res if r['hp_opp_first_drop'] is not None)}/{n} 局，"
      f"中位={statistics.median([r['hp_opp_first_drop'] for r in res if r['hp_opp_first_drop'] is not None]) if any(r['hp_opp_first_drop'] is not None for r in res) else None}")
    P()
    P("## 4. 塔位（出现过的格点，按出现局数排序）")
    pos_me = Counter()
    pos_opp = Counter()
    for r in res:
        for p in r["pos_me"]:
            pos_me[p] += 1
        for p in r["pos_opp"]:
            pos_opp[p] += 1
    P(f"- 我们用过的不同格点 {len(pos_me)}；冠军 {len(pos_opp)}")
    P(f"- 我们 top: {pos_me.most_common(12)}")
    P(f"- 冠军 top: {pos_opp.most_common(12)}")
    # 重复率：同一局里同一格被建了多次（拆迁重放在同一格）无法直接看，这里看跨局稳定性
    P()
    P("## 5. 逐局")
    P("seed\tseat\trounds\thp_me\thp_opp\tmy_build\tmy_demol\tmy_up\tmy_down\t"
      "opp_build\topp_demol\topp_up\topp_down\tmy_tw\tmy_twmax\topp_tw\topp_twmax\tmy_coin\topp_coin")
    for r in res:
        P(f"{r['seed']}\t{r['seat']}\t{r['rounds']}\t{r['hp_me_end']}\t{r['hp_opp_end']}\t"
          f"{r['me']['builds']}\t{r['me']['demol']}\t{r['me']['ups']}\t{r['me']['downs']}\t"
          f"{r['opp']['builds']}\t{r['opp']['demol']}\t{r['opp']['ups']}\t{r['opp']['downs']}\t"
          f"{r['towers_me_mean']:.2f}\t{r['towers_me_max']}\t"
          f"{r['towers_opp_mean']:.2f}\t{r['towers_opp_max']}\t"
          f"{r['coins_me_mean']:.0f}\t{r['coins_opp_mean']:.0f}")
    with open(os.path.join(a.out, "champion_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(a.out, "per_game.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[ok] wrote {a.out}")


if __name__ == "__main__":
    main()
