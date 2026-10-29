#!/usr/bin/env python
"""把 E2 / E4 / E4-rand 的 **`[xor]` 双 oracle 逐回合记录**算成一张表（2026-10-28）。

## 数据来源

`az_bridge_ai.py` 在我们自己的 stderr 里逐回合打一行（`AZAI_ORACLE_INJECT` /
`AZAI_ORACLE_INJECT_B` 至少一个开着才有；默认零日志）：

```
[xor] round=R champ=<动作或()> ru=<动作或()> agree=<0/1|NA> dup_c=<0/1|NA> dup_r=<0/1|NA>
      chosen=<我们的最终动作> hit_c=<0/1|NA> hit_r=<0/1|NA>
[xor-sum] rounds=N agree=<n>(xx.x%) both_hold=<n>(xx.x%) disagree=<n> champ_new=<n> ru_new=<n>
      chosen_eq_champ=<n> chosen_eq_ru=<n> chosen_eq_neither=<n> [eq_champ_when_disagree=<n>/<n>
      champ_hold=<n>(xx.x%) ru_hold=<n>(xx.x%) champ_ops=<n> ru_ops=<n>]
```

字段口径（与报告 §6 一一对应）：
* `champ` / `ru`：冠军 / 亚军在该局面给出的动作（`()` = HOLD）；E2 只有 A ⇒ `ru=NA`；
* `agree`：两者**完全相同**（0/1）—— **核心研究指标**；
* `both_hold`：两者都是 HOLD；
* `dup_c` / `dup_r`：该动作**已在我们自己的网络候选集**里（`1` = 不带来新信息，`0` = 真·增量）；
* `chosen`：我们最终真的发出去的（过完闸门/执行层过滤之后）；
* `hit_c` / `hit_r`：`chosen` 是否分别等于冠军 / 亚军那一手。

## 这个脚本做什么

1. 读一批归档好的 **我方 stderr**（驱动归档到 `match_results/us_vs_external/<TAG>/stderr/`）；
2. 逐局解析（**优先**用 `[xor-sum]` 汇总行；没有就用逐回合 `[xor]` 行自己数）；
3. 打印"每局 + 合计"的表，并写一份 JSON；
4. 顺手把每局 stderr **复制**到 `match_results/xoracle/<TAG>/`（用户要求的独立归档目录）。

## 用法

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
# E4（对手=冠军）
$P -u code/test_match/analyze_xoracle.py \
    --tag ONLYEXTRA_VS_CHAMPION \
    --stderr match_results/us_vs_external/ONLYEXTRA_VS_CHAMPION/stderr
# E2（只有一个会话；ru 全部 NA）
$P -u code/test_match/analyze_xoracle.py --tag US_VS_CHAMPION_INJ \
    --stderr match_results/us_vs_external/US_VS_CHAMPION_INJ/stderr
```
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUR_LABEL = "az_bridge_ai"

XOR_RE = re.compile(r"\[xor\] round=(\d+) champ=(\S+) ru=(\S+) agree=(\S+) dup_c=(\S+) "
                    r"dup_r=(\S+) chosen=(\S+) hit_c=(\S+) hit_r=(\S+)")
SUM_RE = re.compile(r"\[xor-sum\] (.*)$")
NUM_RE = re.compile(r"([a-z_]+)=([-\d.]+)(?:\(([\d.]+)%\))?")
# 2026-10-29（用户要求的「检验力诊断」）：`[dscore]` 逐决策候选分差
DS_RE = re.compile(r"\[dscore\] round=(\d+) n_extra=(\d+) n_net=(\d+) d=(\S+) dvn=(\S+) pick=(\S+)")
DS_SUM_RE = re.compile(r"\[dscore-sum\] (.*)$")


RESULT_ROUNDS_RE = re.compile(r"RESULT seed=\d+.*?rounds=(\d+)")


def _rounds_of_log(logdir: Path, tok: str) -> int | None:
    """从 ladder 日志读该局**引擎记的回合数**（覆盖度的分母）。缺日志 ⇒ None（不猜）。"""
    f = logdir / f"{tok}.log"
    if not f.is_file():
        return None
    m = RESULT_ROUNDS_RE.search(f.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def _q(xs: list, q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))]


def parse_injection(txt: str) -> dict:
    """解析注入覆盖度（2026-10-29 新增，E2 必需）。

    为什么必须报这个：`[oracle]` 会话**一次读包超时/进程崩掉就永久停用**（读端错位），
    之后整局的注入静默消失 ⇒ 那一局变成"前半局 E2 + 后半局 E1"。
    只看 `asks>0` 会把这种局当成有效局 ⇒ 必须报 **覆盖度 = asks / 本局回合数**。
    典型证据：seed 7 那一局 `asks=261` 而 `rounds=512`、且
    `dead=1 err=OracleDead: 亚军进程已退出 (exit code 3221226505)`。
    """
    out: dict = {"enabled": 0, "asks": None, "ok": None, "inject": None, "new": None,
                 "dup": None, "dup_hold": None, "hit": None, "hit_new": None,
                 "timeout": None, "dead": None, "illegal": None,
                 "p50_ms": None, "p95_ms": None, "max_ms": None, "err": ""}
    out["enabled"] = txt.count("注入已启用")
    stats = re.findall(r"注入统计:\s*(.+)", txt)
    if stats:
        line = stats[-1]
        for k in ("asks", "ok", "inject", "new", "dup", "dup_hold", "hit", "hit_new",
                  "timeout", "dead", "illegal", "p50_ms", "p95_ms", "max_ms"):
            m = re.search(r"(?<![\w_])" + k + r"=([-\d.]+)", line)
            if m:
                out[k] = float(m.group(1))
        m = re.search(r"err=([^|]+)$", line)
        if m:
            out["err"] = m.group(1).strip()[:140]
    return out


def parse_dscore(txt: str) -> dict:
    """解析 [dscore] 逐决策行 → 分差分布 + 三个「有效样本量」分母。

    返回：`n`（有记录决策数）、`n2`（两个额外候选 ⇒ 两臂**才可能**不同）、`n1`（只有一个候选
    ⇒ 两臂**按构造相同**）、`pick_hi/pick_lo/pick_net/pick_na`、`d_*` 与 `dvn_*` 的分位。
    """
    out: dict = {"n": 0, "n2": 0, "n1": 0,
                 "pick_hi": 0, "pick_lo": 0, "pick_net": 0, "pick_na": 0}
    ds: list[float] = []
    dvn: list[float] = []
    for (_r, n_extra, _n_net, d, dv, pick) in DS_RE.findall(txt):
        out["n"] += 1
        if int(n_extra) >= 2:
            out["n2"] += 1
            if d not in ("NA", "nan"):
                try:
                    ds.append(abs(float(d)))
                except ValueError:
                    pass
        else:
            out["n1"] += 1
        if dv not in ("NA", "nan"):
            try:
                dvn.append(float(dv))
            except ValueError:
                pass
        out["pick_" + pick] = int(out.get("pick_" + pick, 0)) + 1
    for k, q in (("d_p25", 0.25), ("d_med", 0.50), ("d_p75", 0.75), ("d_p90", 0.90), ("d_max", 1.0)):
        out[k] = _q(ds, q)
    for k, q in (("dvn_p25", 0.25), ("dvn_med", 0.50), ("dvn_p75", 0.75)):
        out[k] = _q(dvn, q)
    out["_d_list"] = ds
    out["_dvn_list"] = dvn
    return out


def _parse_sum(line: str) -> dict:
    out: dict = {}
    for k, v, pct in NUM_RE.findall(line):
        try:
            out[k] = float(v)
        except ValueError:
            continue
        if pct:
            out[k + "_pct"] = float(pct)
    for k, v in re.findall(r"([a-z_]+)=(\d+/\d+)", line):
        out[k] = v
    return out


def parse_game(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="replace")
    sums = [_parse_sum(m.group(1)) for m in SUM_RE.finditer(txt)]
    steps = XOR_RE.findall(txt)
    rec: dict = {"file": path.name, "steps": len(steps), "sum_lines": len(sums)}
    if sums:
        rec["sum"] = sums[-1]              # 最后一条最全（周期打印 + 收尾）
    # 逐回合自己数一遍（与 sum 交叉核对；也能拿到 sum 里没有的分布）
    c = collections.Counter()
    for (_r, champ, ru, agree, dup_c, dup_r, chosen, hit_c, hit_r) in steps:
        c["rounds"] += 1
        c["agree"] += int(agree == "1")
        c["both_hold"] += int(champ == "()" and ru == "()")
        c["champ_new"] += int(dup_c == "0")
        c["ru_new"] += int(dup_r == "0")
        c["chosen_eq_champ"] += int(hit_c == "1")
        c["chosen_eq_ru"] += int(hit_r == "1")
        c["champ_hold"] += int(champ == "()")
        c["ru_hold"] += int(ru == "()")
    rec["recount"] = dict(c)
    rec["dscore"] = parse_dscore(txt)
    rec["inj"] = parse_injection(txt)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="实验 tag（决定归档目录名）")
    ap.add_argument("--stderr", required=True, help="归档后的我方 stderr 目录")
    ap.add_argument("--tokens", nargs="*", default=[],
                    help="要解析的 token（默认 = 目录里所有 seed 7..14 × 两座位）")
    ap.add_argument("--logdir", default="",
                    help="ladder 日志目录（默认 = match_results/ladder_logs/<tag>；用于覆盖度的分母）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sd = REPO / args.stderr
    if not sd.is_dir():
        print(f"!! 没有 stderr 目录 {sd}", file=sys.stderr)
        return 2
    tokens = args.tokens or [f"{s}{suf}" for s in range(7, 15) for suf in ("", "r")]
    args.logdir = REPO / (args.logdir or f"match_results/ladder_logs/{args.tag}")
    out_dir = REPO / (args.out or f"match_results/xoracle/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for tok in tokens:
        seed = int(re.sub(r"r$", "", tok))
        seat = 1 if tok.endswith("r") else 0
        # 座位：token 无后缀 = 我们执 P0 ⇒ ai0_...；有 r ⇒ ai1_...
        f = sd / f"ai{seat}_{OUR_LABEL}_seed{seed}.stderr.log"
        if not f.is_file():
            rows.append({"token": tok, "missing": True})
            continue
        rec = parse_game(f)
        rec["token"] = tok
        rows.append(rec)
        shutil.copy2(f, out_dir / f.name)      # 用户要求：独立归档目录

    print(f"\n### {args.tag}：双 oracle 逐回合记录（stderr={sd}）\n")
    print("| token | 回合数 | agree | both_hold | champ_new | ru_new | chosen=champ | chosen=ru | chosen=≠二者 |")
    print("|---|---|---|---|---|---|---|---|---|")
    agg: dict = collections.Counter()
    npct: dict = collections.defaultdict(list)
    for r in rows:
        if r.get("missing"):
            print(f"| {r['token']} | **缺 stderr** | - | - | - | - | - | - | - |")
            continue
        s = r.get("sum") or r.get("recount") or {}
        n = int(s.get("rounds", 0))
        for k in ("rounds", "agree", "both_hold", "champ_new", "ru_new",
                  "chosen_eq_champ", "chosen_eq_ru", "chosen_eq_neither",
                  "champ_hold", "ru_hold", "disagree", "champ_ops", "ru_ops"):
            agg[k] += int(s.get(k, 0))

        def _p(k):
            v = int(s.get(k, 0))
            return f"{v}({100.0 * v / max(1, n):.1f}%)"
        print(f"| {r['token']} | {n} | {_p('agree')} | {_p('both_hold')} | {int(s.get('champ_new', 0))} | "
              f"{int(s.get('ru_new', 0))} | {_p('chosen_eq_champ')} | {_p('chosen_eq_ru')} | "
              f"{_p('chosen_eq_neither')} |")
        for k in ("agree", "both_hold", "champ_hold", "ru_hold"):
            if n:
                npct[k].append(100.0 * int(s.get(k, 0)) / n)
        if r.get("sum") and r.get("recount"):
            a, b = int(r["sum"].get("rounds", -1)), int(r["recount"].get("rounds", -2))
            if a != b:
                print(f"  ⚠️ {r['token']}：`[xor-sum]` 的 rounds={a} 与逐回合行数={b} 不一致"
                      f"（收尾那行被 bridge 硬杀吃掉的典型表现，以逐回合为准）")

    n = max(1, int(agg["rounds"]))
    print(f"\n**合计**：{int(agg['rounds'])} 个决策")
    print(f"- 冠亚军**完全一致**：{int(agg['agree'])}/{n} = **{100.0 * agg['agree'] / n:.1f}%**")
    print(f"- 两者都 HOLD：{int(agg['both_hold'])}/{n} = {100.0 * agg['both_hold'] / n:.1f}%"
          f"（冠军 HOLD {int(agg['champ_hold'])}、亚军 HOLD {int(agg['ru_hold'])}）")
    print(f"- 新信息（不在我们网络候选里）：冠军 {int(agg['champ_new'])}、亚军 {int(agg['ru_new'])}")
    print(f"- 我们选了冠军那一手：{int(agg['chosen_eq_champ'])}；选了亚军那一手："
          f"{int(agg['chosen_eq_ru'])}；两边都不是：{int(agg['chosen_eq_neither'])}")
    if agg["disagree"]:
        print(f"- 两者**不一致**的回合 {int(agg['disagree'])} 个；其中我们选冠军那手 "
              f"{int(agg.get('eq_champ_when_disagree', 0))} 次"
              f"（{100.0 * float(agg.get('eq_champ_when_disagree', 0)) / agg['disagree']:.1f}%）")
    if npct["agree"]:
        print(f"- 逐局 `agree` 的分布：min/med/max = "
              f"{min(npct['agree']):.1f}%/{statistics.median(npct['agree']):.1f}%/{max(npct['agree']):.1f}%")

    # ── 注入覆盖度（2026-10-29；E2 的"有效样本"分母）──────────────────────────────
    print("\n### 注入覆盖度（`注入统计` vs 本局回合数）\n")
    print("| token | 回合数 | asks | 覆盖度 | inject | new | dup(HOLD) | hit | hit_new | timeout | dead | p50/p95/max ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("missing"):
            print(f"| {r['token']} | **缺 stderr** | - | - | - | - | - | - | - | - | - | - |")
            continue
        ij = r.get("inj") or {}
        rnd = _rounds_of_log(args.logdir, r["token"])
        asks = ij.get("asks")
        cov = f"{100.0 * asks / rnd:.1f}%" if (asks and rnd) else "-"

        def _i(k):
            v = ij.get(k)
            return "-" if v is None else str(int(v))
        print(f"| {r['token']} | {rnd if rnd else '-'} | {_i('asks')} | **{cov}** | {_i('inject')} | "
              f"{_i('new')} | {_i('dup')}({_i('dup_hold')}) | {_i('hit')} | {_i('hit_new')} | "
              f"{_i('timeout')} | {_i('dead')} | "
              f"{_i('p50_ms')}/{_i('p95_ms')}/{_i('max_ms')} |")
        if ij.get("err"):
            print(f"  - ⚠️ {r['token']} oracle 会话异常：`{ij['err']}`")
    _covs = []
    for r in rows:
        ij = r.get("inj") or {}
        rnd = _rounds_of_log(args.logdir, r["token"])
        if ij.get("asks") and rnd:
            _covs.append(100.0 * float(ij["asks"]) / rnd)
    if _covs:
        print(f"\n- 覆盖度 min/med/max = **{min(_covs):.1f}% / {statistics.median(_covs):.1f}% / "
              f"{max(_covs):.1f}%**（<100% ⇒ 该局后半段退化成基线，读数字段要打折）")
        _dead = sum(1 for r in rows if (r.get("inj") or {}).get("dead"))
        _to = sum(1 for r in rows if (r.get("inj") or {}).get("timeout"))
        print(f"- 会话 dead 的局：{_dead}/{len([r for r in rows if not r.get('missing')])}；"
              f"有 timeout 的局：{_to}")

    # ── 检验力诊断（2026-10-29 用户要求）─────────────────────────────────────────
    #   "E4 ≈ E4-rand" 只有在**两候选真的不同**的决策上才可能是"评估器没用"的证据；
    #   两者都 HOLD / 只有一个候选的决策，两臂**按构造**完全相同 ⇒ 必须先报出这些分母。
    print("\n### 检验力诊断：候选分差 `|Δscore|`（`[dscore]`）\n")
    print("| token | 记录决策数 | 两候选(可分辨) | 单候选(两臂相同) | pick_hi | pick_lo | pick_net | |Δ| p25 | |Δ| 中位 | |Δ| p75 | |Δ| p90 | dvn 中位 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled_d: list[float] = []
    pooled_dvn: list[float] = []
    tot = collections.Counter()
    for r in rows:
        if r.get("missing"):
            print(f"| {r['token']} | **缺 stderr** | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        ds_ = r.get("dscore") or {}
        for k in ("n", "n2", "n1", "pick_hi", "pick_lo", "pick_net", "pick_na"):
            tot[k] += int(ds_.get(k, 0))
        pooled_d += ds_.get("_d_list") or []
        pooled_dvn += ds_.get("_dvn_list") or []

        def _f(v, fmt="%.4f"):
            return "-" if v is None else fmt % v
        print(f"| {r['token']} | {int(ds_.get('n', 0))} | {int(ds_.get('n2', 0))} | "
              f"{int(ds_.get('n1', 0))} | {int(ds_.get('pick_hi', 0))} | {int(ds_.get('pick_lo', 0))} | "
              f"{int(ds_.get('pick_net', 0))} | {_f(ds_.get('d_p25'))} | {_f(ds_.get('d_med'))} | "
              f"{_f(ds_.get('d_p75'))} | {_f(ds_.get('d_p90'))} | {_f(ds_.get('dvn_med'), '%+.4f')} |")
    if tot["n"]:
        print(f"\n**有效样本量**：`N_rounds(有记录)={tot['n']}`、"
              f"`N_pair_differs(两候选不同)={tot['n2']}`、`N_one_cand(两臂按构造相同)={tot['n1']}`"
              f"（其中 both_hold={int(agg['both_hold'])} 一定落在后者里）")
        if pooled_d:
            print(f"- 两候选的 `|Δscore|`：n={len(pooled_d)} 中位 {_q(pooled_d, 0.5):.4f}、"
                  f"p25 {_q(pooled_d, 0.25):.4f}、p75 {_q(pooled_d, 0.75):.4f}、p90 {_q(pooled_d, 0.90):.4f}、"
                  f"max {max(pooled_d):.4f}")
            thr = _q(pooled_d, 0.75)
            n_high = sum(1 for x in pooled_d if x >= thr)
            print(f"- **高 Δscore 子集**（≥p75={thr:.4f}）：{n_high}/{len(pooled_d)} 个决策"
                  f"（占两候选决策的 25%）⇒ 这才是「评估器选对/选错能改变动作」的有效样本")
        if pooled_dvn:
            print(f"- 额外候选 vs **我们自己网络候选**的分数差 `dvn`：n={len(pooled_dvn)} "
                  f"中位 {_q(pooled_dvn, 0.5):+.4f}、p25 {_q(pooled_dvn, 0.25):+.4f}、"
                  f"p75 {_q(pooled_dvn, 0.75):+.4f}"
                  f"（>0 = 外部那一手在我们评估器眼里比我们自己的候选更好）")
        print("- `pick_hi` 是**正向对照**：E4（评估器挑）应≈100%，E4-rand（随机挑）应≈50%；"
              f"本次 pick_hi={tot['pick_hi']}/{tot['n2']}"
              f"（{100.0 * tot['pick_hi'] / max(1, tot['n2']):.1f}%）、pick_lo={tot['pick_lo']}、"
              f"pick_net={tot['pick_net']}、pick_na={tot['pick_na']}")
    print("\n> **口径**：`E4≈E4-rand` 本身**不构成**「评估器无用」的证据——"
          "必须先看 `N_pair_differs`（两候选真的不同）与高 Δscore 子集上的差别；"
          "若在高 Δscore 子集上也一致，才能说评估器在这类选择上没贡献。")

    agg_out = dict(agg)
    agg_out["dscore"] = {k: v for k, v in tot.items()}
    agg_out["d_pooled_d"] = pooled_d
    agg_out["d_pooled_dvn"] = pooled_dvn

    outp = out_dir / "summary.json"
    outp.write_text(json.dumps({"tag": args.tag, "rows": rows, "agg": dict(agg)},
                               ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[out] 逐局 JSON: {outp}")
    print(f"[out] 独立归档: {out_dir}（我方 stderr 副本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
