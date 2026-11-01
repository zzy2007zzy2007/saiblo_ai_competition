"""shape_pin vs pin+flat 的 16 局 A/B 分析（2026-11-02）。

主读数：
  1. shape_pin 胜率（+ Wilson 95% CI）；
  2. 血差 Δhp = shape_pin_hp − pin_hp（按 seed 配对，t 95% CI）；
  3. 候选构成：shape_pin 每决策锁定的有序类序列直方图 + n_cand/含HOLD/含升级/长度分布，
     与 pin 臂（同一批对局）对照。

输入：
  · 对局日志 = `match_results/shape_pin_ab/seed<seed>_p<side>.log`（`ab_search_modes.py` 产出）；
  · stderr    = `match_results/ai?_az_bridge_ai_seed<seed>_p<side>.stderr.log`（`MVS_STDERR_SUFFIX=1`）。

座位约定（ab_search_modes.py --paired-sides）：side=0 ⇒ shape_pin 坐 P0；side=1 ⇒ shape_pin 坐 P1。
用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_shape_pin_ab.py \
      --seeds 100-107 [--json match_results/shape_pin_ab/summary.json]
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "match_results" / "shape_pin_ab"
VALID_VERDICT = "engine"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _load_ladder():
    spec = importlib.util.spec_from_file_location("ladder_for_shapepin", REPO / "_tmp_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_seeds(spec: str) -> list[int]:
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
    return sorted(set(out))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def t_ci(xs: list[float], z: float = None) -> tuple[float, float, float, float]:
    """(mean, se, lo, hi)。z=None ⇒ 用 t_{n-1, 0.975}（学生 t 分布）。"""
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = st.mean(xs)
    if n == 1:
        return (mean, float("nan"), mean, mean)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n)
    if z is None:
        # 学生 t 分位数（df=n-1），小样本 95% CI 用这个比 1.96 更诚实。
        # 无内置 t 分位 ⇒ 用一个 df 表 + 线性插值。
        _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
              30: 2.042, 60: 2.000, 120: 1.980}
        df = n - 1
        if df <= 0:
            tv = float("nan")
        elif df in _T:
            tv = _T[df]
        else:
            keys = sorted(_T)
            for a, b in zip(keys, keys[1:]):
                if a <= df <= b:
                    tv = _T[a] + (_T[b] - _T[a]) * (df - a) / (b - a)
                    break
            else:
                tv = _T[keys[-1]] if df > keys[-1] else _T[keys[0]]
        z = tv
    return (mean, se, mean - z * se, mean + z * se)


def _shape_str(seq) -> str:
    """把 locked_seq（通道号或 `cN` 字符串）映射成人类可读动作类序列（legacy decode 语义）。"""
    if not seq:
        return "HOLD"
    name = {0: "建塔", 12: "升级", 13: "降级", 16: "降级", 17: "闪电", 18: "EMP",
            19: "偏转", 20: "闪避", 21: "升速", 22: "升蚁", 23: "HOLD"}
    parts = []
    for c in seq:
        s = str(c)
        if s in ("", "空"):
            continue
        ci = None
        if s.isdigit():
            ci = int(s)
        elif len(s) >= 2 and s[0] == "c" and s[1:].isdigit():
            ci = int(s[1:])
        if ci is not None:
            parts.append("塔" if 1 <= ci <= 15 else name.get(ci, s))
        else:
            parts.append(s)
    return "+".join(parts) or "HOLD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-107")
    ap.add_argument("--out-dir", default=str(OUTDIR))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    ladder = _load_ladder()

    # ── ① 对局结果 ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for seed in seeds:
        for side in (0, 1):
            label = f"seed{seed}_p{side}"
            log_path = out_dir / f"{label}.log"
            if not log_path.is_file():
                missing.append(label)
                continue
            g = ladder.read_game(log_path, str(seed))
            verdict = g.get("verdict")
            if verdict != VALID_VERDICT:
                invalid.append((label, str(verdict)))
                continue
            hp = g.get("hp") or (0, 0)          # (p0_hp, p1_hp)
            ws = g.get("winner_side")
            shapepin_side = side                 # side 0 ⇒ shape_pin=P0；side 1 ⇒ P1
            shapepin_won = (ws == ("p0" if shapepin_side == 0 else "p1"))
            shapepin_hp = hp[shapepin_side]
            pin_hp = hp[1 - shapepin_side]
            rows.append(dict(seed=seed, side=side, label=label,
                             winner_side=ws, rounds=g.get("rounds"),
                             shapepin_won=shapepin_won,
                             dhp=shapepin_hp - pin_hp,
                             shapepin_hp=shapepin_hp, pin_hp=pin_hp))

    n_valid = len(rows)
    wins = sum(1 for r in rows if r["shapepin_won"])
    losses = sum(1 for r in rows if not r["shapepin_won"])
    rate = wins / n_valid if n_valid else float("nan")
    w_lo, w_hi = wilson(wins, n_valid)

    # 按 seed 配对的 Δhp（镜像两局平均 ⇒ 抵消先手偏）
    by_seed: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    paired_dhp: list[float] = []
    for seed in seeds:
        gs = by_seed.get(seed, [])
        if len(gs) == 2:
            paired_dhp.append(st.mean([g["dhp"] for g in gs]))
    mean_dhp, se_dhp, dhp_lo, dhp_hi = t_ci(paired_dhp)

    # 全部 16 局（不配对）的 Δhp 均值（参考）
    all_dhp = [r["dhp"] for r in rows]
    # 按座位拆分
    by_seat = collections.defaultdict(list)
    for r in rows:
        by_seat[r["side"]].append(r["dhp"])

    print("=== shape_pin vs pin+flat：16 局 A/B（seeds "
          f"{seeds[0]}..{seeds[-1]} × 2 座位）===")
    print(f"有日志 {n_valid + len(invalid) + len(missing)} 有效 {n_valid} 无效 {len(invalid)} 缺 {len(missing)}")
    if missing:
        print(f"  缺日志: {missing}")
    if invalid:
        print(f"  无效（不计分）: {invalid}")
    print(f"\n--- 主读数 ①：shape_pin 胜率 ---")
    print(f"  {wins}W-{losses}L / {n_valid} 局 = {rate:.4f}  "
          f"Wilson 95% = [{w_lo:.4f}, {w_hi:.4f}]")
    print(f"\n--- 主读数 ②：血差 Δhp = shape_pin_hp − pin_hp ---")
    print(f"  按 seed 配对（n={len(paired_dhp)} 对）: mean={mean_dhp:+.1f}  "
          f"t95=[{dhp_lo:+.1f}, {dhp_hi:+.1f}]  (se={se_dhp:.1f})")
    if all_dhp:
        print(f"  参考（16 局不配对）: mean={st.mean(all_dhp):+.1f}  "
              f"median={st.median(all_dhp):+.1f}  min={min(all_dhp):+.0f}  max={max(all_dhp):+.0f}")
    for seat in (0, 1):
        xs = by_seat.get(seat, [])
        if xs:
            print(f"  shape_pin 坐 P{seat}（n={len(xs)}）: Δhp mean={st.mean(xs):+.1f}")

    print("\n--- 逐局 ---")
    print("  %-12s %-8s %-10s %-8s %-8s %-7s %-7s" %
          ("token", "座位", "winner", "rounds", "shapepinHP", "pinHP", "Δhp"))
    for r in rows:
        print("  %-12s P%d       %-10s %-8s %-8d %-7d %+7d" %
              (r["label"], r["side"],
               ("shape_pin" if r["shapepin_won"] else "pin"),
               str(r["rounds"]), r["shapepin_hp"], r["pin_hp"], r["dhp"]))

    # ── ② 候选构成（从 stderr）──────────────────────────────────────────────
    print("\n--- 主读数 ③：候选构成（shape_pin vs pin，同一批对局）---")
    sp_locked: collections.Counter = collections.Counter()
    sp_ncand: list[int] = []
    sp_hold: list[int] = []
    sp_upg: list[int] = []
    sp_sizes: collections.Counter = collections.Counter()
    pin_ncand: list[int] = []
    pin_sizes: collections.Counter = collections.Counter()
    n_sp_dec = n_pin_dec = 0

    for seed in seeds:
        for side in (0, 1):
            # shape_pin 坐 side：side0 ⇒ ai0，side1 ⇒ ai1；stderr 后缀 = _p<side>
            sp_file = out_dir.parent / (f"ai{0 if side == 0 else 1}_az_bridge_ai_seed{seed}_p{side}.stderr.log")
            pin_file = out_dir.parent / (f"ai{1 if side == 0 else 0}_az_bridge_ai_seed{seed}_p{side}.stderr.log")
            for f, is_sp in ((sp_file, True), (pin_file, False)):
                try:
                    txt = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if is_sp:
                    for ln in txt.splitlines():
                        m = re.search(r"\[shape_pin\] locked_seq=(\S+) ", ln)
                        if m:
                            sp_locked[m.group(1)] += 1
                        m2 = re.search(r"\[cand\] mode=shape_pin .*?n_cand=(\d+).*?含HOLD=(\d+).*?含升级=(\d+).*?长度分布=\{([^}]*)\}", ln)
                        if m2:
                            n_sp_dec += 1
                            sp_ncand.append(int(m2.group(1)))
                            sp_hold.append(int(m2.group(2)))
                            sp_upg.append(int(m2.group(3)))
                            for kv in m2.group(4).split(","):
                                if ":" in kv:
                                    k, v = kv.split(":", 1)
                                    sp_sizes[int(k.strip())] += int(v.strip())
                else:
                    for ln in txt.splitlines():
                        m = re.search(r"\[cand\] mode=pin .*?n_cand=(\d+).*?长度分布=\{([^}]*)\}", ln)
                        if m:
                            n_pin_dec += 1
                            pin_ncand.append(int(m.group(1)))
                            for kv in m.group(2).split(","):
                                if ":" in kv:
                                    k, v = kv.split(":", 1)
                                    pin_sizes[int(k.strip())] += int(v.strip())

    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return float("nan")
        s = sorted(vals)
        return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]

    print(f"  shape_pin 决策数 = {n_sp_dec}；pin 决策数 = {n_pin_dec}")
    print(f"  shape_pin 锁定的有序类序列直方图（前 12）:")
    for seq, n in sp_locked.most_common(12):
        print(f"      {seq:<22} ({_shape_str(seq.split('+'))}): {n}")
    if sp_ncand:
        print(f"  n_cand 分布: shape_pin p50={_pct(sp_ncand, .5):.0f} p95={_pct(sp_ncand, .95):.0f} "
              f"max={max(sp_ncand)}")
    if pin_ncand:
        print(f"               pin       p50={_pct(pin_ncand, .5):.0f} p95={_pct(pin_ncand, .95):.0f} "
              f"max={max(pin_ncand)}")
    if sp_hold:
        print(f"  含HOLD 决策占比: shape_pin {100.0*sum(sp_hold)/len(sp_hold):.1f}%  "
              f"含升级决策占比: {100.0*sum(sp_upg)/len(sp_upg):.1f}%")
    print(f"  候选长度分布（op 数，按候选条累计）:")
    for sz in sorted(set(list(sp_sizes) + list(pin_sizes))):
        print(f"      len={sz}: shape_pin={sp_sizes.get(sz, 0)}  pin={pin_sizes.get(sz, 0)}")

    out = dict(
        seeds=seeds, n_valid=n_valid, missing=missing, invalid=invalid,
        wins=wins, losses=losses, rate=rate, wilson=[w_lo, w_hi],
        paired_dhp=paired_dhp, dhp_mean=mean_dhp, dhp_t95=[dhp_lo, dhp_hi],
        dhp_by_seat={str(k): v for k, v in by_seat.items()},
        locked_seq_hist=dict(sp_locked.most_common()),
        n_dec=dict(shape_pin=n_sp_dec, pin=n_pin_dec),
        ncand_sp_p50=_pct(sp_ncand, .5), ncand_sp_p95=_pct(sp_ncand, .95),
        ncand_pin_p50=_pct(pin_ncand, .5), ncand_pin_p95=_pct(pin_ncand, .95),
        sizes_sp=dict(sp_sizes), sizes_pin=dict(pin_sizes),
    )
    if args.json:
        p = Path(args.json)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
