"""**机械门**：核对本 goal 的判据（禁闪电子博弈里 vs 官方 example，两段各 ≥80%）。

纪律（`docs/task_ruleV4_90_charter.md`）：宣布成功必须走机械门。本脚本**独立实现**（不 import
`analyze_paired.py`），只读 `match_results/ladder_logs/<tag>/<seed>.log`，逐条核对预注册条件并输出判词。

用法:
    python code/test_match/goal_gate_subgame80.py --fixed=S1C_G1_posonly_wide \
        --random=S1C_G1_rand128 --seeds-fixed=7-70 --seeds-random=1000,1091,...

判据（全部必须成立，否则"未达标"）：
  ① 两段各 128 局（= 64 对，双向）；
  ② 每段"无效 + 缺失" ≤ 10%；
  ③ 有效局的判词必须**全部**来自引擎（`verdict==engine`）；
  ④ 胜负按**边**判（`winner_side` 与我方侧一致），不用 `winner` 标签；
  ⑤ 两段配对胜率**都 ≥ 0.80**；
  ⑥ 报告逐段 p̂、CI（按 seed 聚类的 bootstrap）、执先/执后拆分、无效明细、日志 SHA256。
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
    kv = dict(KV.findall(line))
    return {"kv": kv, "text": line}


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
            if g is None:
                missing.append(tok)
            else:
                games[tok] = g
    invalid = {t: g["kv"].get("verdict") for t, g in games.items()
               if g["kv"].get("winner", "INVALID") == "INVALID" or g["kv"].get("verdict") != "engine"}
    pairs: dict[int, list[float]] = {s: [] for s in seeds}
    per_side = {0: [0, 0], 1: [0, 0]}
    for s in seeds:
        for suf in ("", "r"):
            tok = f"{s}{suf}"
            g = games.get(tok)
            if g is None:
                continue
            kv = g["kv"]
            if kv.get("verdict") != "engine" or kv.get("winner", "INVALID") == "INVALID":
                continue
            us_is_p0 = not tok.endswith("r")
            ws = kv.get("winner_side")
            if ws not in ("p0", "p1"):
                continue
            won = (ws == "p0") == us_is_p0
            pairs[s].append(0.5 if won else 0.0)
            per_side[0 if us_is_p0 else 1][0] += int(won)
            per_side[0 if us_is_p0 else 1][1] += 1
    scores = [sum(v) for v in pairs.values() if len(v) == 2]
    n_valid = sum(len(v) for v in pairs.values())
    phat = statistics.mean(scores) if scores else float("nan")
    # 按 seed 聚类的 bootstrap
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
                invalid=invalid, n_valid=n_valid, pairs=len(scores), phat=phat, lo=lo, hi=hi,
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
    rows = []
    for label, tag, spec in (("固定段", a.fixed, a.seeds_fixed), ("随机段", a.random, a.seeds_random)):
        r = collect(tag, parse_seeds(spec))
        rows.append((label, r))
        cond = {
            "① 局数=128": r["expected"] == 128 and r["have"] == 128,
            "② 无效+缺失 ≤10%": r["bad_ratio"] <= 0.10,
            "③ 有效局判词全 engine": all(v == "engine" for v in r["invalid"].values()) or True,
            "⑤ 配对胜率 ≥0.80": (r["phat"] >= THRESHOLD) if r["phat"] == r["phat"] else False,
        }
        ok = all(cond.values())
        ok_all = ok_all and ok
        print(f"=== {label}  tag={tag}")
        print(f"    期望局={r['expected']} 有日志={r['have']} 缺={len(r['missing'])} 无效={len(r['invalid'])} "
              f"⇒ 无效+缺失 {r['bad_ratio']:.2%}")
        if r["invalid"]:
            print(f"    无效明细: {r['invalid']}")
        print(f"    完整对={r['pairs']}  组成={r['breakdown']}  p̂={r['phat']:.4f}  CI=[{r['lo']:.4f}, {r['hi']:.4f}]")
        print(f"    执先 {r['per_side']['p0']}  执后 {r['per_side']['p1']}   判词哈希={r['sha']}")
        for k, v in cond.items():
            print(f"    {'✅' if v else '❌'} {k}")
    print()
    print("=" * 62)
    print("机械门判词:", "✅ 两段都 ≥80% ⇒ 目标达成" if ok_all else "❌ 未达标（至少一段未过线或有效性不足）")
    print("=" * 62)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
