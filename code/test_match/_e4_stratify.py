#!/usr/bin/env python
"""E4 / E4-rand 的四臂读数 + **按 |Δscore| 分层**的探索性比较（2026-10-29）。

## 为什么要单独写这一个

`analyze_inject_vs_runnerup.py` 已经能出"配对 McNemar + 配对血量差"，但它不读 `[dscore]`；
`analyze_xoracle.py` 能出 `|Δscore|` 分布，但不做**跨臂**比较。E4 的预注册判据要求：

1. 四臂（E4/E4-rand × 冠军/亚军）各自的**胜率 + 血量差**（Wilson / t 95%CI）；
2. `N_rounds / N_pair_differs / N_one_cand / N_both_hold / N_high_dscore`；
3. `|Δscore|` 的分位（检验力：分差本身有多小）；
4. `pick_hi` 的**正向对照**（E4≈100%、E4-rand≈50%，偏离 ⇒ 臂没按预期跑）；
5. **高 Δscore 子集**上的配对比较（按每局 `d` 的中位数给"局"分档）—— 只有在高档组里
   也无差别，"评估器在这类选择上没贡献"才说得出口。

⚠️ 分层是**探索性**的：两臂的对局不是同一个游戏，"决策级反事实"做不到；
   它只是把"分差大 ⇒ 评估器的选择最可能改变结果"的那些局单独拎出来再看一遍。

## 用法

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P -u code/test_match/_e4_stratify.py --out match_results/us_vs_external/_e4_stratify.md
```
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "test_match", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_DS_RE = re.compile(r"\[dscore\] round=(\d+) n_extra=(\d+) n_net=(\d+) d=(\S+) dvn=(\S+) pick=(\S+)")
_XOR_RE = re.compile(r"\[xor\] round=(\d+) champ=(\S+) ru=(\S+) agree=(\S+) dup_c=(\S+) "
                     r"dup_r=(\S+) chosen=(\S+) hit_c=(\S+) hit_r=(\S+)")
DS_TAGS = ("ONLYEXTRA_VS_CHAMPION", "ONLYEXTRA_RAND_VS_CHAMPION",
           "ONLYEXTRA_VS_RUNNERUP", "ONLYEXTRA_RAND_VS_RUNNERUP")
OUTROOT = "match_results/us_vs_external"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _num(v, default=0.0) -> float:
    return default if v is None else float(v)


def _q(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))]


def _fmt(v, fmt="+.2f", na="-") -> str:
    return na if v is None else format(float(v), fmt)


def parse_dscore_file(path: Path) -> dict:
    """一局的 `[dscore]` 逐决策行 → 有效样本量 + 分差 + 挑法。"""
    if path is None or not path.is_file():
        return {"exists": False}
    txt = path.read_text(encoding="utf-8", errors="replace")
    n = n2 = n1 = 0
    ds: list[float] = []
    pick = {"hi": 0, "lo": 0, "net": 0, "na": 0, "only1": 0}
    for (_r, ne, _nn, d, _dv, pk) in _DS_RE.findall(txt):
        n += 1
        pick[pk] = pick.get(pk, 0) + 1
        if int(ne) >= 2:
            n2 += 1
            if d not in ("NA", "nan"):
                try:
                    ds.append(abs(float(d)))
                except ValueError:
                    pass
        else:
            n1 += 1
    both_hold = 0
    xor_rounds = 0
    ru_nonhold = 0
    eq_ru_when_ru_acts = 0
    eq_champ = 0
    champ_nonhold = 0
    for (_r, champ, ru, _ag, _dc, _dr, _ch, _hc, _hr) in _XOR_RE.findall(txt):
        xor_rounds += 1
        both_hold += int(champ == "()" and ru in ("()", "NA"))
        champ_nonhold += int(champ not in ("()", "NA"))
        eq_champ += int(_hc == "1")
        if ru not in ("()", "NA"):
            ru_nonhold += 1
            eq_ru_when_ru_acts += int(_hr == "1")
    return {"exists": True, "n": n, "n2": n2, "n1": n1, "ds": ds, "pick": pick,
            "d_med": _q(ds, 0.5), "d_p25": _q(ds, 0.25), "d_p75": _q(ds, 0.75),
            "both_hold": both_hold, "xor_rounds": xor_rounds,
            # 2026-10-29：决策层"采纳率"——A 恒为 HOLD 时，它衡量"评估器多愿意放弃 HOLD、
            #   采纳那个外部动作"，以及"亚军出招时我们跟着它走"的比例（E4-rand 按构造≈50%）。
            "ru_nonhold": ru_nonhold, "eq_ru_when_ru_acts": eq_ru_when_ru_acts,
            "eq_champ": eq_champ, "champ_nonhold": champ_nonhold}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUTROOT}/_e4_stratify.md")
    ap.add_argument("--json", default=f"{OUTROOT}/_e4_stratify.json")
    args = ap.parse_args()

    av = _load("_analyze_inj", REPO / "code" / "test_match" / "analyze_inject_vs_runnerup.py")

    arms: dict = {}
    for tag in DS_TAGS:
        logdir = REPO / "match_results" / "ladder_logs" / tag
        if not logdir.is_dir():
            print(f"[skip] 没有 {logdir}")
            continue
        arm = av.read_arm(logdir, REPO / OUTROOT / tag / "stderr")
        for g in arm["games"]:
            seat = g.get("seat")
            sp = (REPO / OUTROOT / tag / "stderr"
                  / f"ai{seat}_az_bridge_ai_seed{g['seed']}.stderr.log") if seat is not None else None
            g["ds"] = parse_dscore_file(sp)
        arm["summary"] = av.summarize(arm)
        arms[tag] = arm

    lines: list[str] = []

    def P(s: str = "") -> None:
        print(s)
        lines.append(s)

    P("\n# E4 / E4-rand 四臂读数 + |Δscore| 分层（自动生成）\n")
    P("## 1. 每个 tag 的汇总\n")
    P("| tag | 局数(valid) | 胜-负 | 胜率 | Wilson 95%CI | 血量差 | t 95%CI | 平均回合 | "
      "N_rounds | N_pair_differs | N_one_cand | both_hold | pick_hi |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled: dict = {}
    for tag in DS_TAGS:
        if tag not in arms:
            continue
        arm = arms[tag]
        s = arm["summary"]
        n = s["n_scored"]
        wr = s["winrate"] or 0.0
        wl, wh = s["winrate_ci"]
        hp = s["hp_diff"]
        ds = {"n": 0, "n2": 0, "n1": 0, "hold": 0, "hi": 0, "lo": 0, "net": 0, "na": 0,
              "only1": 0, "d": []}
        for g in arm["games"]:
            d = g.get("ds") or {}
            if not d.get("exists"):
                continue
            ds["n"] += d["n"]
            ds["n2"] += d["n2"]
            ds["n1"] += d["n1"]
            ds["hold"] += d.get("both_hold", 0)
            for k in ("hi", "lo", "net", "na", "only1"):
                ds[k] += (d.get("pick") or {}).get(k, 0)
            ds["d"] += d["ds"]
        pooled[tag] = {"summary": s, "ds": ds}
        rnds = [float(r) for r in (s.get("rounds") or []) if r not in (None, "", "-")]
        P(f"| {tag} | {n} | {s['wins']}-{s['losses']} | **{100.0 * wr:.1f}%** | "
          f"[{100.0 * wl:.1f}%, {100.0 * wh:.1f}%] | {_fmt(hp.get('mean'))} | "
          f"[{_fmt(hp.get('lo'))}, {_fmt(hp.get('hi'))}] | "
          f"{(statistics.fmean(rnds) if rnds else 0):.0f} | {ds['n']} | {ds['n2']} | {ds['n1']} | "
          f"{ds['hold']} | {ds['hi']}/{ds['n2']} |")

    # ── 决策层：采纳率（A 恒为 HOLD 时这是最直接的"评估器行为"读数）──────────
    P("\n## 0. 决策层：它多愿意放弃 HOLD / 多愿意跟着亚军走\n")
    P("| tag | [xor] 决策数 | 冠军**非**HOLD | 亚军非HOLD(两候选) | 亚军出招时我们采纳 | 我们选 HOLD |")
    P("|---|---|---|---|---|---|")
    for tag in DS_TAGS:
        if tag not in arms:
            continue
        acc = {"xor": 0, "c_nh": 0, "ru_nh": 0, "eq_ru": 0, "eq_c": 0}
        for g in arms[tag]["games"]:
            d = g.get("ds") or {}
            if not d.get("exists"):
                continue
            acc["xor"] += d.get("xor_rounds", 0)
            acc["c_nh"] += d.get("champ_nonhold", 0)
            acc["ru_nh"] += d.get("ru_nonhold", 0)
            acc["eq_ru"] += d.get("eq_ru_when_ru_acts", 0)
            acc["eq_c"] += d.get("eq_champ", 0)
        n_x = max(1, acc["xor"])
        ru = max(1, acc["ru_nh"])
        P(f"| {tag} | {acc['xor']} | {acc['c_nh']}({100.0 * acc['c_nh'] / n_x:.1f}%) | "
          f"{acc['ru_nh']}({100.0 * acc['ru_nh'] / n_x:.1f}%) | "
          f"**{acc['eq_ru']}/{acc['ru_nh']} = {100.0 * acc['eq_ru'] / ru:.1f}%** | "
          f"{acc['eq_c']}({100.0 * acc['eq_c'] / n_x:.1f}%) |")
    P("\n> 亚军出招时我们采纳的比例：E4（评估器挑）vs E4-rand（随机挑，按构造 ≈50%）。")
    P("> 两者若都在 50% 附近，说明**采纳的频次**差不多，差别只在**挑哪些**——那才是评估器的作用面。")

    # ── 分差分布 + 高 Δscore 子集 ────────────────────────────────────────────
    P("\n## 2. `|Δscore|` 分布（检验力：两个外部候选的分差本来有多大）\n")
    P("| tag | 两候选决策数 | |Δ| p25 | 中位 | p75 | p90 | max | N_high(≥全体p75) |")
    P("|---|---|---|---|---|---|---|---|")
    all_d = [x for tag in pooled for x in pooled[tag]["ds"]["d"]]
    thr = _q(all_d, 0.75)
    for tag in DS_TAGS:
        if tag not in pooled:
            continue
        d = pooled[tag]["ds"]["d"]
        if not d:
            P(f"| {tag} | 0 | - | - | - | - | - | 0 |")
            continue
        n_high = sum(1 for x in d if thr is not None and x >= thr)
        P(f"| {tag} | {len(d)} | {_q(d, 0.25):.4f} | {_q(d, 0.5):.4f} | {_q(d, 0.75):.4f} | "
          f"{_q(d, 0.9):.4f} | {max(d):.4f} | {n_high} |")
    if thr is not None:
        P(f"\n> 全体 `|Δscore|` 的 p75 = **{thr:.4f}**（用它定义 `N_high_dscore`）。"
          f"全体两候选决策数 = {len(all_d)}。")

    # ── 配对 + 分层 ──────────────────────────────────────────────────────────
    P("\n## 3. 配对比较（E4 vs E4-rand，同 seeds/座位）+ 高 Δscore 分层（探索性）\n")
    strat: dict = {}
    for opp in ("CHAMPION", "RUNNERUP"):
        tA, tB = f"ONLYEXTRA_VS_{opp}", f"ONLYEXTRA_RAND_VS_{opp}"
        if tA not in arms or tB not in arms:
            continue
        a, b = arms[tA], arms[tB]
        pr = av.paired(a, b)
        tab = pr["table"]
        hp = pr["d_hp"]
        P(f"\n### 对手 = {opp}\n")
        P(f"- 配对数 = {pr['n_pairs']}；两臂都赢 {tab['both_win']}、只 E4 赢 {tab['A_only']}、"
          f"只 E4-rand 赢 {tab['B_only']}、都输 {tab['both_lose']}")
        P(f"- McNemar 精确双侧 **p = {pr['mcnemar_p']:.4f}**；胜率差 = "
          f"{100.0 * (pr['delta_winrate'] or 0):+.1f}pp")
        P(f"- 配对血量差（E4 − E4-rand）= **{_fmt(hp.get('mean'))}** "
          f"[{_fmt(hp.get('lo'))}, {_fmt(hp.get('hi'))}]（n={hp.get('n', 0)}）")
        byA = {g["token"]: g for g in a["games"]}
        byB = {g["token"]: g for g in b["games"]}
        lvl = {}
        for row in pr["rows"]:
            tok = row["token"]
            da = (byA.get(tok, {}).get("ds") or {}).get("d_med")
            db = (byB.get(tok, {}).get("ds") or {}).get("d_med")
            vals = [x for x in (da, db) if x is not None]
            if vals:
                lvl[tok] = sum(vals) / len(vals)
        if lvl:
            med = statistics.median(lvl.values())
            hi_tok = {t for t, v in lvl.items() if v >= med}
            lo_tok = set(lvl) - hi_tok
            P(f"- **高 Δscore 子集**（每对 `d_med` ≥ 配对中位 {med:.4f}）：{len(hi_tok)} 对；"
              f"低档 {len(lo_tok)} 对")
            for name, toks in (("高档", hi_tok), ("低档", lo_tok)):
                sa = {"games": [g for g in a["games"] if g["token"] in toks]}
                sb = {"games": [g for g in b["games"] if g["token"] in toks]}
                pr2 = av.paired(sa, sb)
                if not pr2["n_pairs"]:
                    continue
                h2 = pr2["d_hp"]

                def _wr(arm_games):
                    sc = [g["score"] for g in arm_games if g.get("score") is not None]
                    return (100.0 * sum(1 for x in sc if x == 1.0) / len(sc)) if sc else 0.0

                P(f"  - {name}：n_pair={pr2['n_pairs']}，胜率 E4 {_wr(sa['games']):.1f}% / "
                  f"E4-rand {_wr(sb['games']):.1f}%，McNemar p = {pr2['mcnemar_p']:.4f}，"
                  f"配对 Δhp = {_fmt(h2.get('mean'))} "
                  f"[{_fmt(h2.get('lo'))}, {_fmt(h2.get('hi'))}]")
                strat[f"{opp}_{name}"] = {"n_pairs": pr2["n_pairs"],
                                          "mcnemar_p": pr2["mcnemar_p"],
                                          "d_hp": h2, "tokens": sorted(toks)}
        strat[f"{opp}_all"] = {"n_pairs": pr["n_pairs"], "mcnemar_p": pr["mcnemar_p"],
                               "d_hp": hp, "table": tab}

    # ── 正向对照 ────────────────────────────────────────────────────────────
    P("\n## 4. 正向对照（臂有没有按预期跑）\n")
    for tag in DS_TAGS:
        if tag not in pooled:
            continue
        ds = pooled[tag]["ds"]
        n2 = max(1, ds["n2"])
        P(f"- `{tag}`：pick_hi = {ds['hi']}/{ds['n2']} = **{100.0 * ds['hi'] / n2:.1f}%**、"
          f"pick_lo = {ds['lo']}、pick_net = {ds['net']}、pick_na = {ds['na']}、"
          f"only1 = {ds['only1']}")
    P("\n> 预期：E4（评估器挑）≈ 全 `hi`；E4-rand（随机挑）≈ 一半 `hi`。")
    P("> 若 E4 的 pick_hi 明显 <100% ⇒ 候选在**执行层**被过滤/改写（不是评估器没挑它）。")

    report = "\n".join(lines) + "\n"
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(report, encoding="utf-8")
    print(f"\n[out] {outp}")
    js = REPO / args.json
    js.write_text(json.dumps({"pooled": {k: {"summary": v["summary"],
                                             "n": v["ds"]["n"], "n2": v["ds"]["n2"],
                                             "n1": v["ds"]["n1"],
                                             "pick": {kk: v["ds"][kk] for kk in
                                                      ("hi", "lo", "net", "na", "only1")},
                                             "d_p25": _q(v["ds"]["d"], 0.25),
                                             "d_med": _q(v["ds"]["d"], 0.5),
                                             "d_p75": _q(v["ds"]["d"], 0.75),
                                             "d_p90": _q(v["ds"]["d"], 0.9)}
                                         for k, v in pooled.items()},
                               "thr_p75": thr, "strat": strat},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[out] {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
