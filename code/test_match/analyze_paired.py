"""读 `_tmp_ladder.py` 产的逐局日志，算**配对胜率 + CI + 机制读数**（预注册 §2/§3 的量）。

为什么单独一个脚本（不靠 ladder 的汇总行）：
  1. ladder 的汇总行只打**均值/方差**，没有**配对 bootstrap 的 CI**，也没有**有效性门槛**；
  2. 基线要能被**独立验证**（机械门：第三个 agent 只拿日志 + 本脚本 + 预注册 §2 的定义就能复算）
     ⇒ 判据统计必须**可复算、且写在文件里**，不能只活在我某一轮的对话里。

⚠️ 解析复用 `_tmp_ladder.read_game`（importlib 从仓库根加载）——**绝不在这里重写一遍解析**：
   重写就等于"两处逻辑慢慢分叉"，而历史事故（重打判定把"败"记成"平"）正是这么来的。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_paired.py \
      --tag=B3_baseline_rulev4_128 --seeds=7-70 [--json=out.json] [--us-label=az_bridge_ai]

判据来源：docs/prereg_20261005_baseline_vs_rulev4_128.md §2（配对分 + cluster bootstrap）
         与 §4（有效性门槛：verdict=engine / illegal=0 / winner≠INVALID / terminal）。
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "match_results" / "ladder_logs"
US_DEFAULT = "az_bridge_ai"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def load_ladder():
    spec = importlib.util.spec_from_file_location("ladder_for_analysis", REPO / "_tmp_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_seeds(spec: str) -> list[int]:
    """`7-70` / `7,8,9` / `7-9,20` 都支持。"""
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


def cluster_bootstrap(vals: list[float], iters: int, seed: int,
                      lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """按 **seed（对）** 重采样 —— 对内的两局不独立，所以不能按"局"重采样。"""
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return (means[int(lo / 100 * iters)], means[min(iters - 1, int(hi / 100 * iters))])


def fmt_stats(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return (f"mean={st.mean(xs):.1f} median={st.median(xs):.1f} "
            f"min={min(xs):.0f} max={max(xs):.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="7-70")
    ap.add_argument("--us-label", default=US_DEFAULT)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20261005)
    ap.add_argument("--json", default=None)
    ap.add_argument("--logs-root", default=None,
                    help="日志根目录（默认 match_results/ladder_logs）；"
                         "用于自检与独立验证时指向别处")
    ap.add_argument("--invalid-threshold", type=float, default=0.10,
                    help="无效局比例超过它 ⇒ 整批标 INVALID（预注册 §4 写死 10%%）")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    out_dir = (Path(args.logs_root) if args.logs_root else LOG_ROOT) / args.tag
    if not out_dir.is_dir():
        print(f"!! 找不到日志目录 {out_dir}")
        return 2
    ladder = load_ladder()

    tokens = [f"{s}{suf}" for s in seeds for suf in ("", "r")]
    games: dict[str, dict] = {}
    missing: list[str] = []
    for t in tokens:
        p = out_dir / f"{t}.log"
        if not p.is_file():
            missing.append(t)
            continue
        games[t] = ladder.read_game(p, t)

    # ---------- §4 有效性门槛 ----------
    invalid: dict[str, str] = {}
    # 「太慢」≠「坏」（2026-10-06）：bridge 现在会把读超时标成 verdict=timeout。
    # 单列出来，避免一个方法因为**算得慢**被误读成"无效/有 bug"（曾误导过一次引擎归因）。
    timeouts: dict[str, str] = {}
    for t, g in games.items():
        why = []
        if not g["winner"] or g["winner"] == "INVALID":
            # ⚠️ verdict=None 说明**这一行根本没被 RESULT_RE 解析**（字段缺失），与"真的无判词"不同 ——
            #    2026-10-06：INVALID 行曾因缺 engine_winner= 而永远解析失败，把超时局埋成"winner 缺失"。
            if g["verdict"] is None:
                why.append("winner 缺失/INVALID ⚠️(该行未被解析: 可能字段不全)")
            else:
                why.append("winner 缺失/INVALID")
        if g["verdict"] == "timeout":
            # ⚠️ 仍然**算无效**（没有结果、绝不计分）；同时**单独登记**成"太慢"，让归因看得见。
            #    （我曾错误地在这里 `continue`，导致超时局被当成 0 胜混进配对分 —— 已修，2026-10-06）
            timeouts[t] = "verdict=timeout（AI 单次决策超过读超时 ⇒ 太慢，非逻辑错误）"
            why.append("verdict=timeout（太慢）")
        elif g["verdict"] != "engine":
            why.append(f"verdict={g['verdict']}")
        if g["illegal"]:
            why.append(f"illegal={g['illegal']}")
        if not g["rounds"] or g["rounds"] == "0":
            why.append("rounds=0")
        if str(g["terminal"]) != "True":
            why.append(f"terminal={g['terminal']}")
        if why:
            invalid[t] = "; ".join(why)

    valid = [t for t in tokens if t in games and t not in invalid]
    n_valid = len(valid)
    n_games_expected = len(tokens)
    invalid_ratio = (len(invalid) + len(missing)) / n_games_expected if n_games_expected else 1.0
    if timeouts:
        print(f"  ⏱️ 超时局（太慢 ⇒ 仍计无效、但原因单列，便于区分「慢」与「坏」）: {len(timeouts)} 局 -> {sorted(timeouts)[:12]}"
              + (" ..." if len(timeouts) > 12 else ""), flush=True)

    # ---------- 配对分（主统计量）----------
    per_seed: dict[int, list[dict]] = collections.defaultdict(list)
    for t in valid:
        per_seed[int(re.match(r"(\d+)", t).group(1))].append(games[t])

    pair_scores: list[float] = []
    breakdown = collections.Counter()
    for s in seeds:
        gs = per_seed.get(s, [])
        if len(gs) != 2:
            breakdown["不完整对"] += 1
            continue
        sc = 0.0
        for g in gs:
            # token 不带 r ⇒ 我方(AI0)执 p0；带 r ⇒ 我方执 p1
            us_is_p0 = not g["token"].endswith("r")
            us_label = g["p0_label"] if us_is_p0 else g["p1_label"]
            assert us_label == args.us_label, (g["token"], us_label, args.us_label)
            if g["winner"] == us_label:
                sc += 0.5
            elif g["winner"] == "draw":
                sc += 0.25  # 官方从不判平局；出现即说明判词来源有问题（下面会告警）
        pair_scores.append(sc)
        breakdown["2-0" if sc == 1.0 else ("1-1" if sc == 0.5 else "0-2")] += 1

    n_pairs = len(pair_scores)
    paired_rate = st.mean(pair_scores) if pair_scores else float("nan")
    boot_lo, boot_hi = cluster_bootstrap(pair_scores, args.boot, args.boot_seed) if pair_scores \
        else (float("nan"), float("nan"))

    us_wins = 0.0
    for t in valid:
        g = games[t]
        us_is_p0 = not t.endswith("r")
        us_label = g["p0_label"] if us_is_p0 else g["p1_label"]
        if g["winner"] == us_label:
            us_wins += 1
        elif g["winner"] == "draw":
            us_wins += 0.5
    raw_rate = us_wins / n_valid if n_valid else float("nan")
    w_lo, w_hi = wilson(int(round(us_wins)), n_valid)

    # ---------- 先手侧 & 机制读数 ----------
    side = collections.Counter()
    for t in valid:
        g = games[t]
        us_is_p0 = not t.endswith("r")
        if g["winner"] == "draw":
            side["draw"] += 1
            continue
        us_won = g["winner"] == (g["p0_label"] if us_is_p0 else g["p1_label"])
        if us_is_p0:
            side["我方执先手 胜" if us_won else "我方执先手 负"] += 1
        else:
            side["我方执后手 胜" if us_won else "我方执后手 负"] += 1

    def mech(label: str) -> dict:
        tower, act, ops, light, upg, rounds = [], [], [], [], [], []
        for t in valid:
            g = games[t]
            us_is_p0 = not t.endswith("r")
            who = "p0" if (us_is_p0 == (label == "us")) else "p1"
            h = g["hist"][who]
            tower.append(h.get("BUILD_TOWER", 0))
            upg.append(h.get("UPGRADE_TOWER", 0))
            light.append(h.get("USE_LIGHTNING_STORM", 0))
            act.append(g["act"][who])
            ops.append(g["ops"][who])
            rounds.append(int(g["rounds"]) if str(g["rounds"]).isdigit() else 0)
        return {"name": label, "n": len(tower),
                "build_tower": tower, "upgrade_tower": upg, "lightning": light,
                "act_rounds": act, "ops": ops, "rounds": rounds}

    us_mech = mech("us")
    op_mech = mech("opp")
    # 对手标签从日志里取（别写死 "rule_v4" —— B1 的对手是冠军，写死会误导读数）
    opp_label = "?"
    for t in valid:
        g = games[t]
        opp_label = g["p0_label"] if g["p1_label"] == args.us_label else g["p1_label"]
        break

    dead = [t for t in valid if games[t]["ops"]["p0"] + games[t]["ops"]["p1"] <= 2]
    verdicts = collections.Counter(games[t]["verdict"] for t in valid)
    draws = [t for t in valid if games[t]["winner"] == "draw"]

    # ---------- 报告 ----------
    print(f"=== analyze_paired: tag={args.tag}  seeds={seeds[0]}..{seeds[-1]}（{len(seeds)} 个）===")
    print(f"期望局数={n_games_expected}  有日志={len(games)}  缺日志={len(missing)}  "
          f"无效={len(invalid)}  有效={n_valid}")
    if missing:
        print(f"  缺: {missing}")
    if invalid:
        print("  无效明细（不计分）：")
        for t, why in sorted(invalid.items()):
            print(f"    {t}: {why}")
    print(f"判词来源: {dict(verdicts)}   （必须全是 engine）")
    if dead:
        print(f"  !! 僵局嫌疑（双方操作总数≤2）: {dead}")
    if draws:
        print(f"  !! 出现 draw: {draws} —— 官方级联从不判平局 ⇒ 判词可疑，别当结论。")

    print(f"\n--- 主读数：配对分（预注册 §2）---")
    print(f"  完整配对数 = {n_pairs} / {len(seeds)}   （2-0: {breakdown['2-0']}  "
          f"1-1: {breakdown['1-1']}  0-2: {breakdown['0-2']}  不完整: {breakdown['不完整对']}）")
    print(f"  配对胜率 p̂ = {paired_rate:.4f}   "
          f"95% cluster-bootstrap CI = [{boot_lo:.4f}, {boot_hi:.4f}]  "
          f"({args.boot} 次重采样 seed，seed={args.boot_seed})")
    print(f"  参考（裸胜率，128 局口径）= {raw_rate:.4f}   Wilson 95% = [{w_lo:.4f}, {w_hi:.4f}]"
          f"   （本设计里两者点估计相同；CI 口径不同）")

    print(f"\n--- 先手侧 ---")
    for k in ("我方执先手 胜", "我方执先手 负", "我方执后手 胜", "我方执后手 负", "draw"):
        if side.get(k):
            print(f"  {k}: {side[k]}")

    print(f"\n--- 机制读数（观测，不是优化目标）---")
    for m in (us_mech, op_mech):
        print(f"  [{'我方' if m['name'] == 'us' else opp_label}] n={m['n']}")
        print(f"    建塔 BUILD_TOWER: {fmt_stats(m['build_tower'])}")
        print(f"    升塔 UPGRADE_TOWER: {fmt_stats(m['upgrade_tower'])}")
        print(f"    闪电 USE_LIGHTNING_STORM: {fmt_stats(m['lightning'])}")
        print(f"    出招回合数: {fmt_stats(m['act_rounds'])}")
        print(f"    总操作数: {fmt_stats(m['ops'])}")
        print(f"    对局回合数: {fmt_stats(m['rounds'])}")

    # ---------- 预注册判据检查 ----------
    print(f"\n--- 预注册判据检查（prereg §2/§4/§5）---")
    batch_ok = invalid_ratio <= args.invalid_threshold and not missing
    print(f"  有效性门槛: 无效+缺失比例 = {invalid_ratio:.1%}  "
          f"(门槛 {args.invalid_threshold:.0%}) ⇒ {'通过' if batch_ok else '!! 整批标 INVALID'}")
    if not batch_ok:
        print("  ⇒ 按预注册 §4：本批**不作为基线**，先查原因再重测。")
    elif paired_rate >= 0.70:
        print("  ⇒ 命中预注册判据（p̂ ≥ 0.70）⇒ **必须走机械门**：无上下文独立验证 + 记档，"
              "不许自行宣布成功（prereg §5）。")
    else:
        print(f"  ⇒ 未过 70% 门（差 {0.70 - paired_rate:+.4f}）⇒ 本轮只记基线；"
              "下一轮从方法层假设 H1–H4 挑一个，另写预注册。")
    # 预测 P1/P2（prereg §3）自动核对
    print(f"  P1（点预测 p̂∈[0.35,0.60]）: "
          f"{'符合' if 0.35 <= paired_rate <= 0.60 else '不符合'}")
    med_tower = st.median(us_mech["build_tower"]) if us_mech["build_tower"] else float("nan")
    med_act = st.median(us_mech["act_rounds"]) if us_mech["act_rounds"] else float("nan")
    print(f"  P2（我方建塔中位数≤12、出招回合中位数≤30）: 建塔={med_tower:.0f} 出招={med_act:.0f} "
          f"⇒ {'符合' if med_tower <= 12 and med_act <= 30 else '不符合'}")
    print(f"  P2b（rule_v4 建塔数显著高于我方）: 我方中位={med_tower:.0f} "
          f"对手中位={st.median(op_mech['build_tower']) if op_mech['build_tower'] else float('nan'):.0f}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "tag": args.tag, "seeds": seeds, "n_expected": n_games_expected,
            "n_valid": n_valid, "missing": missing, "invalid": invalid,
            "paired_rate": paired_rate, "paired_ci": [boot_lo, boot_hi], "n_pairs": n_pairs,
            "breakdown": dict(breakdown), "raw_rate": raw_rate, "wilson": [w_lo, w_hi],
            "side": dict(side), "verdicts": dict(verdicts),
            "us_mech": {k: v for k, v in us_mech.items() if k != "name"},
            "opp_mech": {k: v for k, v in op_mech.items() if k != "name"},
            "batch_valid": batch_ok, "pass_70": bool(batch_ok and paired_rate >= 0.70),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
