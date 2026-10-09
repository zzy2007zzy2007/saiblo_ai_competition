"""**通用机械门**（goal-2600a56a 要求）：核对"候选 tag 的两段配对胜率 ≥ 阈值"，并**强制核对两侧配置确实不同**。

与 `goal_gate_subgame80.py`（v2，打 example 专用那版）的关系：本文件是**通用版**——
不再假设对手是谁，而是从 run 归档里读出**候选侧**与**对手侧**的配置指纹，检查"它们确实不同"，
从而挡住"把基线换弱/两侧同一配置（镜像）"这类作弊或错误。

用法:
    python code/test_match/goal_gate_vs_baseline.py --tag AB_E_joint_phold10_vs_baseline \
        --seeds-fixed 7-70 --seeds-random 1000,1091,... --threshold 0.80
判据（全部必须成立，否则"未达标"）：
  ① 每段期望局数 == 2×seed 数，且 `有日志 + 缺 == 期望`（无静默丢局）；
  ② 每段"无效 + 缺失" ≤ 10%；
  ③ 每个计分句的判词都是 `verdict == engine`；
  ④ 每个计分句都**按边判**（`winner_side ∈ {p0,p1}`）；
  ⑤ 每段配对胜率 ≥ 阈值；
  ⑥ **两侧配置指纹必须不同**：从 `training_history/runs/<该 tag>` 的 `cmd.txt` 读 ai0/ai1 路径与
     `base_pkg`/`old_pkg` 之类"覆盖型 launcher"，从 `env.txt` 读候选侧 ckpt；若候选侧 ckpt == 基线 ckpt
     且候选侧配置与基线一致 ⇒ 判"同配置（镜像）"，**门直接失败**（防止把基线换成自己）。
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
RUNS = REPO / "training_history" / "runs"
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=(\S+)")

# 冻结基线（写死，与 goal 文本一致）
BASE_CKPT = "checkpoints/grafted_sdk_A.pt"
BASE_ENV = {
    "AZAI_MODE": "pos-only", "AZAI_POSPIN": "mc_quiet", "AZAI_MC_EVERY": "5",
    "AZAI_MC_ROOTONLY": "1", "AZAI_MC_MARGIN": "0", "AZAI_MC_UPG": "1", "AZAI_MC_TOPK": "3",
    "AZAI_ITERS": "256", "AZAI_DEPTH": "4", "AZAI_K": "24", "AZAI_SAMPLE_MULT": "15",
    "AZAI_TEMP": "1e-6", "AZAI_TCLASS": "0.5", "AZAI_TPOS": "1.0", "AZAI_SKIP1": "1",
    "AZAI_POSPRIOR": "off", "AZAI_VERIFY": "1",
}


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


def read_game(p: Path):
    txt = p.read_text(encoding="utf-8", errors="replace")
    line = next((s.strip() for s in txt.splitlines() if s.strip().startswith("RESULT ")), None)
    if not line:
        return None
    return dict(KV.findall(line))


def collect(tag: str, seeds: list[int]) -> dict:
    games, missing = {}, []
    for s in seeds:
        for suf in ("", "r"):
            tok = f"{s}{suf}"
            p = LOG_ROOT / tag / f"{tok}.log"
            if not p.is_file():
                missing.append(tok); continue
            kv = read_game(p)
            games[tok] = kv if kv is not None else missing.append(tok)
    games = {k: v for k, v in games.items() if v is not None}
    verdicts: dict[str, int] = {}
    for g in games.values():
        verdicts[g.get("verdict")] = verdicts.get(g.get("verdict"), 0) + 1
    pairs = {s: [] for s in seeds}
    per_side = {0: [0, 0], 1: [0, 0]}
    invalid = {}
    n_scored = n_side_ok = n_side_bad = 0
    for s in seeds:
        for suf in ("", "r"):
            tok = f"{s}{suf}"
            g = games.get(tok)
            if g is None:
                continue
            if g.get("verdict") != "engine" or g.get("winner", "INVALID") == "INVALID":
                invalid[tok] = g.get("verdict"); continue
            us_is_p0 = not tok.endswith("r")
            ws = g.get("winner_side")
            if ws not in ("p0", "p1"):
                n_side_bad += 1; continue
            n_side_ok += 1; n_scored += 1
            won = (ws == "p0") == us_is_p0
            pairs[s].append(0.5 if won else 0.0)
            per_side[0 if us_is_p0 else 1][0] += int(won)
            per_side[0 if us_is_p0 else 1][1] += 1
    scores = [sum(v) for v in pairs.values() if len(v) == 2]
    phat = statistics.mean(scores) if scores else float("nan")
    rng = random.Random(20261019)
    boots = sorted(statistics.mean([rng.choice(scores) for _ in scores]) for _ in range(10000)) if scores else []
    lo, hi = (boots[250], boots[9749]) if boots else (float("nan"), float("nan"))
    return dict(tag=tag, nseeds=len(seeds), expected=2 * len(seeds), have=len(games), missing=missing,
                invalid=invalid, verdicts=verdicts, pairs=len(scores), phat=phat, lo=lo, hi=hi,
                n_scored=n_scored, n_side_ok=n_side_ok, n_side_bad=n_side_bad,
                per_side={("p0" if k == 0 else "p1"): tuple(v) for k, v in per_side.items()},
                bad_ratio=(len(invalid) + len(missing)) / (2 * len(seeds)),
                breakdown={f"{k}-{2-k}": sum(1 for x in scores if abs(x - k / 2) < 1e-9) for k in (0, 1, 2)})


def config_fingerprint(tag: str) -> dict:
    """从 run 归档读候选侧配置，并与冻结基线比较 ⇒ 判断"两侧是否真的不同"。"""
    out = {"run_dir": None, "ai0": None, "ai1": None, "ckpt": None, "same_config": None, "notes": []}
    cands = sorted([d for d in RUNS.glob(f"*_{tag}") if d.is_dir()])
    if not cands:
        out["notes"].append("找不到该 tag 的 run 归档 ⇒ 无法核对配置指纹")
        out["same_config"] = None
        return out
    d = cands[-1]
    out["run_dir"] = d.name
    cmd = (d / "cmd.txt")
    if cmd.is_file():
        txt = cmd.read_text(encoding="utf-8", errors="replace")
        m0 = re.search(r"--ai0=(\S+)", txt); m1 = re.search(r"--ai1=(\S+)", txt)
        out["ai0"] = m0.group(1) if m0 else None
        out["ai1"] = m1.group(1) if m1 else None
    env = {}
    if (d / "env.txt").is_file():
        for line in (d / "env.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1); env[k.strip()] = v.strip()
    out["ckpt"] = env.get("AZAI_CKPT")
    diffs = {k: (env.get(k), v) for k, v in BASE_ENV.items() if env.get(k) != v}
    same = (out["ckpt"] == BASE_CKPT) and not diffs
    out["same_config"] = same
    out["env_diffs_vs_baseline"] = diffs
    if out["ai1"] and "base_pkg" not in out["ai1"]:
        out["notes"].append(f"⚠️ ai1 不是 base_pkg（实际 {out['ai1']}）⇒ 对手可能不是冻结基线")
    if same:
        out["notes"].append("❌ 候选侧配置与冻结基线**完全相同** ⇒ 这是镜像，不是『打基线』")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="候选臂的 tag（ai0=候选, ai1=base_pkg）")
    ap.add_argument("--seeds-fixed", default="7-70")
    ap.add_argument("--seeds-random", required=True)
    ap.add_argument("--threshold", type=float, default=0.80)
    a = ap.parse_args()

    fp = config_fingerprint(a.tag)
    print("=== 配置指纹核对（防『把基线换弱』）")
    print(f"    run 归档={fp['run_dir']}  ai0={fp['ai0']}  ai1={fp['ai1']}")
    print(f"    候选侧 ckpt={fp['ckpt']}   冻结基线 ckpt={BASE_CKPT}")
    if fp.get("env_diffs_vs_baseline") is not None:
        print(f"    候选侧与基线不同的 env 项: {fp['env_diffs_vs_baseline'] or '（无）'}")
    for n in fp["notes"]:
        print("    " + n)
    fp_ok = (fp["same_config"] is False) and (fp["ai1"] is None or "base_pkg" in (fp["ai1"] or ""))

    ok_all = fp_ok
    for label, spec in (("固定段", a.seeds_fixed), ("随机段", a.seeds_random)):
        r = collect(a.tag, parse_seeds(spec))
        cond = {
            "① 参数/无静默丢局": r["have"] + len(r["missing"]) == r["expected"],
            "② 无效+缺失 ≤10%": r["bad_ratio"] <= 0.10,
            "③ 计分句判词全 engine": r["verdicts"].get("engine", 0) == r["n_scored"] and r["n_scored"] > 0,
            "④ 全部计分句按边判": r["n_side_ok"] == r["n_scored"] and r["n_side_bad"] == 0,
            f"⑤ 配对胜率 ≥{a.threshold:g}": (r["phat"] >= a.threshold) if r["phat"] == r["phat"] else False,
        }
        ok = all(cond.values()); ok_all = ok_all and ok
        print(f"=== {label}  tag={r['tag']}")
        print(f"    期望={r['expected']} 有日志={r['have']} 缺={len(r['missing'])} 无效={len(r['invalid'])} "
              f"⇒ 无效+缺失 {r['bad_ratio']:.2%}  判词={r['verdicts']}")
        if r["invalid"]:
            print(f"    无效明细={r['invalid']}")
        print(f"    完整对={r['pairs']} 组成={r['breakdown']}  p̂={r['phat']:.4f} CI=[{r['lo']:.4f}, {r['hi']:.4f}] "
              f"执先={r['per_side']['p0']} 执后={r['per_side']['p1']}")
        for k, v in cond.items():
            print(f"    {'✅' if v else '❌'} {k}")
    print()
    print("=" * 70)
    print("机械门判词:", "✅ 两段都过阈值（且 ①–⑥ 全过）⇒ 按预注册判据：目标达成"
          if ok_all else "❌ 未达标（某段未过阈值或有效性/指纹检查失败）")
    print("⚠️ 提醒：p̂ 刚过阈值 ⇒ 含义是'点估计达到阈值'，不等于统计上证明 >阈值")
    print("=" * 70)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
