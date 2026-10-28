"""规则内（≤10 s/决策）口径下「亚军候选注入」A/B 的读数汇总（2026-10-28）。

报告：`docs/oracle_inject_deploy_ab_20261028.md`。

两种模式：

* `--mode probe`（任务 A）：只读 stderr，给**两臂各自的决策耗时**（p50/p95/max +
  `dec_ge10/dec_ge20/dec_ge30` 计数与占比）——用来决定"注入开之后还能不能落在
  游戏规则允许的 10 s/决策之内"。
* `--mode ab`（任务 B）：读 `ab_search_modes --paired-sides` 的产出，给
  ① 胜负/血量差/座位拆分/配对得分/符号检验；
  ② 两臂决策耗时分布（同上）；
  ③ 注入统计（`asks/new/dup/dup_hold/hit/hit_new/hit_dup/hit_sel`）；
  ④ 每局 `verdict` 与 `illegal`；
  ⑤ **事件研究**：把每个决策按"亚军这一手有没有被采纳"分三类
     （`hit_new` = 采纳了**新**候选 / `hit_dup` = 采纳了重复候选（多数是 HOLD）/ `miss` = 没采纳），
     看它们之后 `k` 回合内**血量差的变化**（`m(r+k)-m(r)`，m = 我方基地血 - 对方基地血）。

口径说明（为什么必须这么切）：

* **有效局** = `verdict=engine`（照抄 `analyze_paired.py` §4），其余一律排除主读数；
* `A` = 注入臂（`--flat-env`），日志名 `seed{N}_p{S}.log` 的 `S` 就是 A 的座位；
  A 臂的 stderr 满足"文件名里的 ai 序号 == 座位号"（`ai0_*_p0` / `ai1_*_p1`）；
* 本脚本**只读**日志，不重跑、不改任何东西。

用法：
  python code/test_match/analyze_oracle_inject_deploy.py --mode probe \\
      --stderr-dir match_results/oracle_inject_deploy/probe_canon/stderr
  python code/test_match/analyze_oracle_inject_deploy.py --mode ab \\
      --out-dir match_results/oracle_inject_deploy/ab16
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_oracle_inject_ab import parse_games, RE_ILLSUM  # noqa: E402  复用既有解析

# ── 日志解析正则 ────────────────────────────────────────────────────────────────
RE_CUM = re.compile(r"耗时累计: dec_n=(\d+) dec_max=([\d.]+)s dec_slow=(\d+)")
RE_GE = {th: re.compile(rf"dec_ge{th}=(\d+)\(([\d.]+)%\)") for th in (10, 20, 30)}
RE_PCT = re.compile(r"决策耗时:? n=(\d+) p50=([\d.]+)s p95=([\d.]+)s max=([\d.]+)s")
RE_PCT2 = re.compile(r"决策耗时 p50=([\d.]+)s p95=([\d.]+)s max=([\d.]+)s n=(\d+)")
RE_INJLOG = re.compile(r"\[oracle\] inject round=(\d+) action=(\S+) dup=(\S+) new=(\S+) "
                       r"chosen=(\S+) hit=(\d+)")
RE_TRACE = re.compile(r"^\[azai\] round=(\d+) hp=\[(-?\d+), (-?\d+)\]")
RE_STATS = re.compile(r"注入(?:统计|累计): (?P<body>.+)$", re.MULTILINE)
RE_ORACLE_ON = re.compile(r"注入已启用: exe=(\S+) seed=(\d+) player=(\d+) timeout=([\d.]+)s")
RE_ORACLE_DEAD = re.compile(r"停用注入|启用失败")


def _num(txt: str, key: str, cast=float, default=None):
    m = re.search(rf"\b{key}=([-\d.]+)", txt)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except ValueError:
        return default


def dec_stats(txt: str) -> dict:
    """一局 stderr 的**最终**决策耗时读数（取最后一条累计行；只有 <25 决策的局才没有）。

    * `dec_n` / `dec_max` / `dec_slow` 来自 `⏱️ 耗时累计`（每 25 决策一行）；
    * `p50/p95` 来自 `⏱️ 决策耗时`（同一节奏）；
    * `dec_ge10/20/30` 是**固定阈值**计数（2026-10-28 新增，口径不随环境变量漂移）。
    """
    out: dict = {"dec_n": None, "dec_max": None, "dec_slow": None,
                 "p50": None, "p95": None, "max": None}
    for th in (10, 20, 30):
        out[f"ge{th}"] = None
        out[f"ge{th}_pct"] = None
    cums = RE_CUM.findall(txt)
    if cums:
        out["dec_n"] = int(cums[-1][0])
        out["dec_max"] = float(cums[-1][1])
        out["dec_slow"] = int(cums[-1][2])
    for th in (10, 20, 30):
        m = RE_GE[th].findall(txt)
        if m:
            out[f"ge{th}"] = int(m[-1][0])
            out[f"ge{th}_pct"] = float(m[-1][1])
    p2 = RE_PCT2.findall(txt)
    if p2:
        out["p50"] = float(p2[-1][0])
        out["p95"] = float(p2[-1][1])
        out["max"] = float(p2[-1][2])
        out["n_pct"] = int(p2[-1][3])
    p1 = RE_PCT.findall(txt)
    if p1:
        out["n_pct"] = int(p1[-1][0])
        out["p50"] = float(p1[-1][1])
        out["p95"] = float(p1[-1][2])
        out["max"] = float(p1[-1][3])
    return out


def oracle_stats(txt: str) -> dict:
    m = RE_STATS.findall(txt)
    if not m:
        return {}
    body = m[-1]
    rec = {}
    for k in ("asks", "ok", "inject", "new", "dup", "dup_hold", "dup_nonhold", "hit",
              "hit_new", "hit_dup", "hit_sel", "illegal", "timeout", "dead", "checked",
              "avg_ms", "p50_ms", "p95_ms", "max_ms"):
        rec[k] = _num(body, k, int, None)
    for k in ("hit_rate", "hit_rate_asks", "hit_new_rate", "hit_dup_rate", "hit_sel_rate"):
        v = _num(body, k, float, None)
        rec[k] = v
    return rec


def scan_stderr(stderr_dir: Path) -> tuple[list[dict], list[dict]]:
    """把一目录下所有 stderr 分成 A 臂（注入开）/ B 臂（注入关）两组，各带耗时读数。

    判据 = 日志里有没有 `注入已启用: …`（**不靠文件名**：探针模式下 A 臂有时是 ai0、
    有时是 ai1）。这一判据同时验证了"这一局注入真的起来了"——静默未启用会落成 B 臂。
    """
    arms = {"A": [], "B": []}
    skipped = []
    for f in sorted(set(stderr_dir.glob("ai?_*_seed*.stderr.log"))):
        txt = f.read_text(encoding="utf-8", errors="replace")
        if "决策耗时" not in txt and "耗时累计" not in txt:
            skipped.append(f.name)
            continue
        on = RE_ORACLE_ON.search(txt)
        if on is None and RE_ORACLE_DEAD.search(txt):
            skipped.append(f.name + "(注入启用失败)")
            continue
        m = re.search(r"seed(\d+)", f.name)
        seat_m = re.search(r"_p(\d)\.stderr\.log$", f.name)
        rec = {"file": f.name, "seed": int(m.group(1)) if m else None,
               "seat": int(seat_m.group(1)) if seat_m else None,
               "txt_len": len(txt), **dec_stats(txt)}
        rec["inject_on"] = on is not None
        rec["oracle_timeout"] = float(on.group(4)) if on else None
        if on is not None:
            rec.update({"oracle": oracle_stats(txt),
                        "illegal_sent": max((int(x) for x in RE_ILLSUM.findall(txt)), default=0)})
            arms["A"].append(rec)
        else:
            arms["B"].append(rec)
    return arms, skipped


def _agg(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _tcrit(df: int, alpha: float = 0.05) -> float:
    """双侧 95% 的 t 临界值（优先 scipy，缺了就查小表 —— 本项目只用到 df=3..200）。"""
    try:
        from scipy import stats as _st
        return float(_st.t.ppf(1 - alpha / 2, df))
    except Exception:
        _tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
                8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
                25: 2.060, 30: 2.042}
        return _tab.get(df, 1.96)


def _sign_test(w: int, l: int) -> float:
    """双侧符号检验（忽略平局）：2 * P(X >= max(w,l))，X~Bin(w+l, 0.5)。"""
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def print_timing(arms: dict, title: str) -> None:
    print("=" * 110)
    print(title)
    print(f"{'臂':<4}{'局数':>4}{'决策数':>8}{'p50(s)':>9}{'p95(s)':>9}{'max(s)':>9}"
          f"{'≥10s':>7}{'占比':>8}{'≥20s':>7}{'占比':>8}{'≥30s':>7}{'占比':>8}")
    for arm in ("A", "B"):
        rows = arms[arm]
        if not rows:
            continue
        n_dec = sum(int(r.get("dec_n") or 0) for r in rows)

        # 每局先取统计，再对"局"取中位 + 极值（与 docs/oracle_inject_ab_20261028.md §4.4 同口径）
        def _range(key):
            v = _agg(rows, key)
            return f"{_median(v):.2f}" if v else "   --"
        ge = {th: sum(int(r.get(f"ge{th}") or 0) for r in rows) for th in (10, 20, 30)}
        # 决策数：优先用 dec_n 之和；`dec_geX` 的分母就是这个
        label = "A(注入)" if arm == "A" else "B(基准)"
        print(f"{label:<6}{len(rows):>4}{n_dec:>8}{_range('p50'):>9}{_range('p95'):>9}"
              f"{max(_agg(rows, 'max') or [0]):>9.2f}"
              f"{ge[10]:>7}{(100.0 * ge[10] / max(1, n_dec)):>7.2f}%"
              f"{ge[20]:>7}{(100.0 * ge[20] / max(1, n_dec)):>7.2f}%"
              f"{ge[30]:>7}{(100.0 * ge[30] / max(1, n_dec)):>7.2f}%")
    print("  （p50/p95 = 每局统计后取**中位数**；max = 所有局里最慢的一次；"
          "≥10s/20s/30s 的分母 = 该臂所有局的决策数之和）")
    print()
    print(f"{'文件':<46}{'决策':>6}{'p50':>7}{'p95':>7}{'max':>7}{'≥10s':>6}{'≥20s':>6}{'dec_slow':>9}")
    for arm in ("A", "B"):
        for r in arms[arm]:
            print(f"{r['file']:<46}{int(r.get('dec_n') or 0):>6}"
                  f"{(r.get('p50') or 0):>7.2f}{(r.get('p95') or 0):>7.2f}"
                  f"{(r.get('max') or 0):>7.2f}{int(r.get('ge10') or 0):>6}"
                  f"{int(r.get('ge20') or 0):>6}{int(r.get('dec_slow') or 0):>9}")


# ── 事件研究（hit_new / hit_dup / miss × 后续血量差）──────────────────────────────
def event_study(stderr_dir: Path, a_files: list[dict], horizons=(10, 25, 50)) -> None:
    """把每个决策按"亚军这一手有没有被采纳"分类，看之后 k 回合的基地血量差变化。

    口径：
      * 血量差 `m(r) = hp[我方座位] - hp[对方座位]`，取自我方 stderr 的 `round=` 轨迹行
        （`AZAI_TRACE` 默认 1 ⇒ 每回合一行，两臂都有）；
      * 分类只对有 `[oracle] inject` 行的决策（= 注入真的生效的那些回合）：
          - `hit_new`  = `hit=1 new=1`：亚军给了**我们候选里没有**的一手，且被采纳；
          - `hit_dup`  = `hit=1 dup=1`：亚军这一手与已有候选重复（**绝大多数是 HOLD**）且被采纳；
          - `miss`     = `hit=0`：没采纳（我们的搜索选了别的）；
      * 事件值 `Δm_k = m(r+k) - m(r)`（k 回合后血量差的变化，正 = 变好）；
      * ⚠️ 同一局内的决策高度相关，所以除了"全决策池"的均值，还给**每局均值**（n=局数），
        以及"局内配对差"（同一局里 hit_new 的 Δm 均值 − hit_dup 的 Δm 均值）。
    """
    print("=" * 110)
    print("事件研究：按「亚军这一手有没有被采纳」分类，看之后 k 回合的**血量差变化**")
    per_game = []
    for rec in a_files:
        f = stderr_dir / rec["file"]
        txt = f.read_text(encoding="utf-8", errors="replace")
        seat = rec.get("seat")
        if seat is None:
            continue
        hp = {}
        for m in RE_TRACE.finditer(txt):
            hp[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
        if not hp:
            print(f"  !! {rec['file']} 没有 round= 轨迹行（AZAI_TRACE=0？）")
            continue
        dec = {}
        for m in RE_INJLOG.finditer(txt):
            rnd = int(m.group(1))
            dup, new, hit = m.group(3), m.group(4), int(m.group(6))
            cls = ("hit_new" if (hit == 1 and new == "1")
                   else ("hit_dup" if (hit == 1 and dup == "1") else "miss"))
            dec[rnd] = cls
        if not dec:
            continue
        g = {"file": rec["file"], "seed": rec["seed"], "seat": seat, "n_dec": len(dec),
             "win": None, "classes": {}}
        for h in horizons:
            vals = {c: [] for c in ("hit_new", "hit_dup", "miss")}
            for rnd, cls in dec.items():
                a, b = hp.get(rnd), hp.get(rnd + h)
                if a is None or b is None:
                    continue
                m0, m1 = a[seat] - a[1 - seat], b[seat] - b[1 - seat]
                vals[cls].append(m1 - m0)
            for c, v in vals.items():
                g["classes"].setdefault(c, {})[h] = _mean(v) if v else float("nan")
                g["classes"].setdefault(c + "_n", {})[h] = len(v)
        per_game.append(g)

    if not per_game:
        print("  （没有可用的事件研究数据）")
        return

    print(f"  覆盖 {len(per_game)} 局（A 臂）")
    for h in horizons:
        print(f"\n  ── 之后 {h} 回合的血量差变化 Δm（正 = 我方变好）──")
        print(f"    {'类别':<10}{'决策数':>7}{'Δm 均值(全决策池)':>20}{'每局均值的中位':>16}"
              f"{'每局均值>0 的局数':>16}")
        for c in ("hit_new", "hit_dup", "miss"):
            pooled = []
            gm = []
            for g in per_game:
                v = g["classes"].get(c, {}).get(h)
                n = g["classes"].get(c + "_n", {}).get(h) or 0
                if v is not None and not math.isnan(v) and n:
                    gm.append(v)
                pooled.append((v, n))
            tot_n = sum(n for _v, n in pooled)
            wsum = sum((v * n) for v, n in pooled
                       if v is not None and not math.isnan(v) and n)
            pooled_mean = wsum / tot_n if tot_n else float("nan")
            gmean = _mean(gm)
            pos = sum(1 for x in gm if x > 0)
            print(f"    {c:<10}{tot_n:>7}{pooled_mean:>20.2f}{gmean:>16.2f}"
                  f"{f'{pos}/{len(gm)}':>16}")
        # 局内配对：同一局 hit_new 的 Δm − hit_dup 的 Δm
        diffs = []
        for g in per_game:
            a = g["classes"].get("hit_new", {}).get(h)
            b = g["classes"].get("hit_dup", {}).get(h)
            na = g["classes"].get("hit_new_n", {}).get(h) or 0
            nb = g["classes"].get("hit_dup_n", {}).get(h) or 0
            if (a is not None and b is not None and na >= 3 and nb >= 3
                    and not math.isnan(a) and not math.isnan(b)):
                diffs.append(a - b)
        if diffs:
            print(f"    局内配对（hit_new − hit_dup，只算两类各 ≥3 次的局）：n={len(diffs)} "
                  f"均值={_mean(diffs):+.2f}  "
                  f"正的局={sum(1 for x in diffs if x > 0)}/{len(diffs)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("probe", "ab"), default="ab")
    ap.add_argument("--out-dir", default="match_results/oracle_inject_deploy/ab16")
    ap.add_argument("--stderr-dir", default="")
    ap.add_argument("--no-event", action="store_true", help="跳过事件研究")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO / out
    st = Path(args.stderr_dir) if args.stderr_dir else out / "stderr"
    if not st.is_absolute():
        st = REPO / st

    arms, skipped = scan_stderr(st)
    if not arms["A"] and not arms["B"]:
        print(f"!! {st} 里没有可读的 stderr")
        return 1
    if skipped:
        print(f"⚠️ 跳过 {len(skipped)} 个文件: {', '.join(skipped[:6])}")

    if args.mode == "probe":
        print_timing(arms, f"任务 A：决策耗时（探针，{st}）")
        print()
        print("每局注入读数（A 臂）：")
        for r in arms["A"]:
            o = r.get("oracle") or {}
            print(f"  {r['file']}: timeout={r['oracle_timeout']}s asks={o.get('asks')} "
                  f"new={o.get('new')} dup={o.get('dup')} hit_sel={o.get('hit_sel')} "
                  f"avg_ms={o.get('avg_ms')} max_ms={o.get('max_ms')} "
                  f"timeout={o.get('timeout')} dead={o.get('dead')}")
        return 0

    # ── AB 模式 ────────────────────────────────────────────────────────────────
    games = parse_games(out)
    if not games:
        print(f"!! {out} 里没有 seed*_p*.log")
        return 1
    valid = [g for g in games if g["result"] and g["verdict"] == "engine"]
    print("=" * 110)
    print(f"每局明细（A = 注入臂）：{out}")
    print(f"{'seed':>6} {'A座位':>5} {'结果':>4} {'hp_A':>5} {'hp_B':>5} {'血量差':>6} "
          f"{'回合':>5} {'判词':>12} {'illegal':>7}")
    for g in sorted(games, key=lambda x: (x["seed"], x["a_side"])):
        tag = ("胜" if g["a_score"] == 1.0 else ("负" if g["a_score"] == 0.0 else "平"))
        diff = (g["hp_a"] - g["hp_b"]) if g["hp_a"] is not None else None
        print(f"{g['seed']:>6} {'P' + str(g['a_side']):>5} {tag:>4} "
              f"{(g['hp_a'] if g['hp_a'] is not None else '--'):>5} "
              f"{(g['hp_b'] if g['hp_b'] is not None else '--'):>5} "
              f"{(f'{diff:+d}' if diff is not None else '--'):>6} "
              f"{(g['rounds'] if g['rounds'] is not None else '--'):>5} "
              f"{str(g['verdict']):>12} "
              f"{(g['illegal'] if g['illegal'] is not None else '--'):>7}")

    print()
    print("=" * 110)
    print("汇总（主口径 = 只有 verdict=engine 算有效局）")

    def stat(rows: list[dict], label: str) -> None:
        n = len(rows)
        if not n:
            print(f"  {label}: n=0")
            return
        w = sum(1 for g in rows if g["a_score"] == 1.0)
        d = sum(1 for g in rows if g["a_score"] == 0.5)
        l = sum(1 for g in rows if g["a_score"] == 0.0)
        pts = sum(g["a_score"] for g in rows)
        rate = 100.0 * pts / n
        se = (rate * (100 - rate) / n) ** 0.5
        diffs = [g["hp_a"] - g["hp_b"] for g in rows if g["hp_a"] is not None]
        mean = _mean(diffs)
        sd = (sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5 if len(diffs) > 1 else 0.0
        half = _tcrit(max(1, len(diffs) - 1)) * sd / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        rds = [g["rounds"] for g in rows if g["rounds"] is not None]
        print(f"  {label:<8} n={n:<3} {w}W/{d}D/{l}L  得分={rate:5.1f}% ±{se:.1f}%  "
              f"血量差={mean:+.2f} (95%CI {mean - half:+.2f}~{mean + half:+.2f}, "
              f"sd={sd:.2f})  平均回合={_mean(rds):.0f}")

    stat(valid, "全部")
    stat([g for g in valid if g["a_side"] == 0], "@P0")
    stat([g for g in valid if g["a_side"] == 1], "@P1")
    bad = [g for g in games if g not in valid]
    print(f"  无效局: {len(bad)}/{len(games)}"
          + (f" -> {[(g['seed'], 'P' + str(g['a_side']), g['verdict']) for g in bad]}" if bad else ""))
    ids = [g["illegal"] for g in games if g["illegal"] is not None]
    print(f"  RESULT 行 illegal= 合计: {sum(ids)}（有该字段的 {len(ids)} 局）"
          f"  非零: {[(g['seed'], g['a_side'], g['illegal']) for g in games if g['illegal']]}")
    print(f"  判词分布: {dict((v, sum(1 for g in games if g['verdict'] == v)) for v in {g['verdict'] for g in games})}")

    # 配对得分 + 符号检验（只看有效局）
    by_seed: dict[int, list[float]] = {}
    for g in valid:
        by_seed.setdefault(g["seed"], []).append(g["a_score"])
    paired = {s: _mean(v) for s, v in sorted(by_seed.items())}
    print(f"  配对得分（按 seed 平均）: {paired}")
    print(f"  配对得分总平均 = {_mean(list(paired.values())):.4f}"
          f"（{sum(1 for v in paired.values() if v == 1.0)} 个 seed 2-0、"
          f"{sum(1 for v in paired.values() if v == 0.5)} 个 1-1、"
          f"{sum(1 for v in paired.values() if v == 0.0)} 个 0-2）")
    w = sum(1 for g in valid if g["a_score"] == 1.0)
    l = sum(1 for g in valid if g["a_score"] == 0.0)
    print(f"  符号检验（{w}W vs {l}L，忽略平局）: 双侧 p = {_sign_test(w, l):.4f}")

    print()
    print_timing(arms, f"决策耗时（两臂分开量；A 臂 {len(arms['A'])} 局、B 臂 {len(arms['B'])} 局）")

    print()
    print("=" * 110)
    print("注入侧读数（A 臂）")
    tot = {k: 0 for k in ("asks", "ok", "inject", "new", "dup", "dup_hold", "dup_nonhold",
                          "hit", "hit_new", "hit_dup", "hit_sel", "illegal", "timeout", "dead",
                          "checked")}
    print(f"{'seed':>6}{'座位':>5}{'asks':>6}{'new':>5}{'dup':>6}{'dup_hold':>9}"
          f"{'hit':>6}{'hit_new':>8}{'hit_dup':>8}{'hit_sel':>8}{'avg_ms':>8}{'max_ms':>8}"
          f"{'timeout':>8}{'dead':>5}{'illegal':>8}")
    for r in sorted(arms["A"], key=lambda x: (x["seed"], x["seat"] or 0)):
        o = r.get("oracle") or {}
        for k in tot:
            tot[k] += int(o.get(k) or 0)
        print(f"{r['seed']:>6}{'P' + str(r['seat']):>5}{(o.get('asks') or 0):>6}"
              f"{(o.get('new') or 0):>5}{(o.get('dup') or 0):>6}{(o.get('dup_hold') or 0):>9}"
              f"{(o.get('hit') or 0):>6}{(o.get('hit_new') or 0):>8}{(o.get('hit_dup') or 0):>8}"
              f"{(o.get('hit_sel') or 0):>8}{(o.get('avg_ms') or 0):>8}{(o.get('max_ms') or 0):>8}"
              f"{(o.get('timeout') or 0):>8}{(o.get('dead') or 0):>5}"
              f"{r.get('illegal_sent') or 0:>8}")
    print(f"{'合计':>6}{'':>5}{tot['asks']:>6}{tot['new']:>5}{tot['dup']:>6}{tot['dup_hold']:>9}"
          f"{tot['hit']:>6}{tot['hit_new']:>8}{tot['hit_dup']:>8}{tot['hit_sel']:>8}")
    print(f"  比率: new/asks={100.0 * tot['new'] / max(1, tot['asks']):.1f}%  "
          f"dup_hold/dup={100.0 * tot['dup_hold'] / max(1, tot['dup']):.1f}%  "
          f"hit/asks={100.0 * tot['hit'] / max(1, tot['asks']):.1f}%  "
          f"hit_new/new={100.0 * tot['hit_new'] / max(1, tot['new']):.1f}%  "
          f"hit_dup/dup={100.0 * tot['hit_dup'] / max(1, tot['dup']):.1f}%  "
          f"hit_sel/new={100.0 * tot['hit_sel'] / max(1, tot['new']):.1f}%")
    print(f"  失败: timeout={tot['timeout']} dead={tot['dead']}  "
          f"注入候选被判非法={tot['illegal']}  逐条复核 op={tot['checked']}")
    mx = [int((r.get('oracle') or {}).get('max_ms') or 0) for r in arms["A"]]
    if mx:
        print(f"  oracle 单次最大耗时 = {max(mx)} ms（超时阈值 = "
              f"{arms['A'][0].get('oracle_timeout')} s）")
    print(f"  中途停用注入的局: "
          f"{[r['file'] for r in arms['A'] if (r.get('oracle') or {}).get('dead') or (r.get('oracle') or {}).get('timeout')] or '无'}")

    if not args.no_event:
        print()
        event_study(st, arms["A"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
