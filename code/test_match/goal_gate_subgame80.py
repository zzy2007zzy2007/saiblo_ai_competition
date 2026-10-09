"""**机械门**（v2，2026-10-19 修独立验证指出的两处实现漏洞）：核对本 goal 的判据
（禁闪电子博弈里 vs 官方 example，两段各 ≥80%）。

v1 的两个 bug（独立验证发现）：
  ① `cond["③ 有效局判词全 engine"] = all(...) or True` ⇒ **恒真**，等于没查；
  ② docstring 里的"④ 胜负按边判"**没有真正进入 cond** ⇒ 门挡不住 `winner_side` 缺失的日志。
v2 把 ③④ 改成**可真可假**的检查，并额外输出判词直方图 / 按边计分数 / 标签-边分歧数。

用法:
    python code/test_match/goal_gate_subgame80.py --fixed=S1C_G1_posonly_wide --random=S1C_G1_rand128 \
        --seeds-fixed=7-70 --seeds-random=1000,1091,...

判据（**全部**必须成立，否则"未达标"）：
  ① 两段各 128 局（64 对，双向），且 `有日志 + 缺 == 期望`（没有静默丢局）；
  ② 每段"无效 + 缺失" ≤ 10%；
  ③ **每一个被计分的局，其判词都是 `verdict==engine`**（且无效局全部在 ② 里被计数）；
  ④ **每一个被计分的局都按"边"判**（`winner_side ∈ {p0,p1}`；不依赖 `winner` 标签）；
  ⑤ 两段配对胜率都 ≥ 0.80；
  ⑥ 报告逐段 p̂、CI（按 seed 聚类 bootstrap）、执先/执后拆分、无效明细、判词哈希。
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

LOG_ROOT = REPO / "match_results" / "ladder_logs"
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=(\S+)")
THRESHOLD = 0.80


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def read_game(p: Path) -> dict | None:
    txt = p.read_text(encoding="utf-8", errors="replace")
    line = next((s.strip() for s in txt.splitlines() if s.strip().startswith("RESULT ")), None)
    if not line:
        return None
    return {"kv": dict(KV.findall(line)), "text": line}


def collect(tag: str, seeds: list[int]) -> dict:
    games: dict[str, dict] = {}
    missing: list[str] = []
    for s in seeds:
        for suf in ("", "r"):
            tok = f"{s}{suf}"
            p = LOG_ROOT / tag / f"{tok}.log"
            if not p.is_file():
                missing.append(tok)
                continue
            g = read_game(p)
            (games.__setitem__(tok, g) if g is not None else missing.append(tok))

    verdicts: dict[str, int] = {}
    for g in games.values():
        v = g["kv"].get("verdict")
        verdicts[v] = verdicts.get(v, 0) + 1

    pairs: dict[int, list[float]] = {s: [] for s in seeds}
    per_side = {0: [0, 0], 1: [0, 0]}
    invalid: dict[str, str] = {}
    n_scored = 0
    n_side_ok = 0
    n_side_bad = 0
    label_disagree = 0
    for s in seeds:
        for suf in ("", "r"):
            tok = f"{s}{suf}"
            g = games.get(tok)
            if g is None:
                continue
            kv = g["kv"]
            if kv.get("verdict") != "engine" or kv.get("winner", "INVALID") == "INVALID":
                invalid[tok] = kv.get("verdict")
                continue
            us_is_p0 = not tok.endswith("r")
            ws = kv.get("winner_side")
            if ws not in ("p0", "p1"):
                n_side_bad += 1
                continue
            n_side_ok += 1
            won = (ws == "p0") == us_is_p0
            # 标签口径（仅用于报告分歧，不参与判分）
            lab = kv.get("winner")
            if lab in ("az_bridge_ai", "main"):
                # 我方在两侧都叫 az_bridge_ai ⇒ 标签口径就是 (winner == "az_bridge_ai")；
                # v2 初版这里多乘了 `== us_is_p0`（对 r 局恒反）⇒ 报出假的 63/64 分歧（2026-10-19 修）
                won_lab = (lab == "az_bridge_ai")
                label_disagree += int(won_lab != won)
            pairs[s].append(0.5 if won else 0.0)
            per_side[0 if us_is_p0 else 1][0] += int(won)
            per_side[0 if us_is_p0 else 1][1] += 1
            n_scored += 1

    scores = [sum(v) for v in pairs.values() if len(v) == 2]
    phat = statistics.mean(scores) if scores else float("nan")
    rng = random.Random(20261018)
    boots = []
    for _ in range(10000):
        samp = [rng.choice(scores) for _ in scores] if scores else []
        boots.append(statistics.mean(samp) if samp else float("nan"))
    boots.sort()
    lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]) if boots else (float("nan"),) * 2
    n_expected = 2 * len(seeds)
    bad = len(invalid) + len(missing)
    return dict(tag=tag, seeds=len(seeds), expected=n_expected, have=len(games), missing=missing,
                invalid=invalid, verdicts=verdicts, pairs=len(scores), phat=phat, lo=lo, hi=hi,
                n_scored=n_scored, n_side_ok=n_side_ok, n_side_bad=n_side_bad,
                label_disagree=label_disagree,
                per_side={("p0" if k == 0 else "p1"): tuple(v) for k, v in per_side.items()},
                bad_ratio=bad / n_expected if n_expected else 1.0,
                breakdown={f"{k}-{2 - k}": sum(1 for x in scores if abs(x - k / 2) < 1e-9) for k in (0, 1, 2)},
                sha=hashlib.sha256("".join(sorted(g["text"] for g in games.values())).encode()).hexdigest()[:16])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--seeds-fixed", default="7-70")
    ap.add_argument("--seeds-random", required=True)
    a = ap.parse_args()

    ok_all = True
    for label, tag, spec in (("固定段", a.fixed, a.seeds_fixed), ("随机段", a.random, a.seeds_random)):
        r = collect(tag, parse_seeds(spec))
        cond = {
            "① 局数=128 且 有日志+缺=期望": r["expected"] == 128 and (r["have"] + len(r["missing"])) == r["expected"],
            "② 无效+缺失 ≤10%": r["bad_ratio"] <= 0.10,
            "③ 计分局判词全 engine": r["verdicts"].get("engine", 0) == r["n_scored"] and r["n_scored"] > 0,
            "④ 全部计分局按边判(winner_side)": r["n_side_ok"] == r["n_scored"] and r["n_side_bad"] == 0,
            "⑤ 配对胜率 ≥0.80": (r["phat"] >= THRESHOLD) if r["phat"] == r["phat"] else False,
        }
        ok = all(cond.values())
        ok_all = ok_all and ok
        print(f"=== {label}  tag={tag}")
        print(f"    期望={r['expected']} 有日志={r['have']} 缺={len(r['missing'])} 无效={len(r['invalid'])} "
              f"⇒ 无效+缺失 {r['bad_ratio']:.2%}")
        print(f"    判词直方图={r['verdicts']}  计分局={r['n_scored']}  按边成功={r['n_side_ok']} 按边失败={r['n_side_bad']}"
              f"  标签-边分歧={r['label_disagree']}")
        if r["invalid"]:
            print(f"    无效明细={r['invalid']}")
        print(f"    完整对={r['pairs']} 组成={r['breakdown']}  p̂={r['phat']:.4f} CI=[{r['lo']:.4f}, {r['hi']:.4f}] "
              f"执先={r['per_side']['p0']} 执后={r['per_side']['p1']} 判词哈希={r['sha']}")
        for k, v in cond.items():
            print(f"    {'✅' if v else '❌'} {k}")
    print()
    print("=" * 66)
    print("机械门判词:", "✅ 两段都 ≥80%（且 ①–④ 全过）⇒ 按预注册判据：目标达成"
          if ok_all else "❌ 未达标（至少一段未过线或某条有效性检查失败）")
    print("⚠️ 提醒：p̂ 刚过 0.80 时，其含义是'点估计达到阈值'，不等于统计上证明 >80%（见 docs 记录）")
    print("=" * 66)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
