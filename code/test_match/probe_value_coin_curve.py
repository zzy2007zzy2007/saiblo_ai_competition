"""P0-2：**纯输入空间**扫 `V(coins)` —— 价值网对金币的定价曲线形状。

对应 `docs/value_net_coin_pricing_hypothesis_20261030.md` §E 的 P0-2，
是 `docs/value_net_coin_probe_20261030.md` 的 P0-2 那一半。

为什么这版最干净：
  · 历史坑（`docs/final_report_20261010.md:102-103`）：`facade.coins` 是**缓存副本**，
    写它再 `_cache.clear()` 会退回引擎真值 ⇒ 干预失效、测出 ∂V/∂coins ≡ 0 的假象。
  · 本脚本**完全不碰游戏状态**：不写 coins、不 clone、不做任何状态干预。
    只在**已经编码好的输入张量**上改一个下标（`stats[22]` = 我币/300），然后前向。
    ⇒ 天然绕过那个陷阱。

三组样本（每组内部都是"其余 41 + 28×19×19 个输入逐位不动，只动 stats[22]"）：
  A. `synthetic` —— 全零 board + 全零 stats，只有 `stats[22]` 在动
     （"干净输入张量"，值网的裸函数形态，但不是真实局面分布）
  B. `no_tower`  —— **无塔基准**：`GameStateFacade` 只 `advance_round()`、从不 apply 任何 op
     （= `probe_value_coin_slope.py` / `probe_value_grad.py` 原来那个取样口径，
      也正好是 +0.0274 那个读数的场景）
  C. `tower`     —— **有塔局面**：按 `match_results/ladder_logs/<tag>/<seed>.log`
     逐 op `apply_operation_list` 复现的真实对局（UNIFIED_SPLIT_VERBOSE/8.log 第 0 轮就建了 2 座塔）

读数：
  1. V 随 coins 的曲线（0→300 币，41 个点）
  2. 形状判定：线性 / 饱和 / 非单调（分段斜率 + 线性拟合 R² + 单调性计数）
  3. 关键点斜率：0–45 币（1 座塔）/ 45–90 币 / 90–180 币 / 180–300 币
  4. 换算：每 100 币 ≈ 多少 HP（value 单位 = 20 HP，`code/my_ai/az_intent/mcts.py:46` HP_SCALE）
  5. 附：有塔局面下把 `stats[26]`（我方闪电冷却/35，0=已好）钉死再看斜率（P0-3 的最小版）

用法：
  D:\\anaconda3\\envs\\pytorch-gpu\\python.exe code/test_match/probe_value_coin_curve.py
  ... --tag UNIFIED_SPLIT_VERBOSE --seed 8 --every 10 --points 41
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from SDK.utils.features import FeatureExtractor                      # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models           # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade        # noqa: E402

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"
HP_SCALE = 20.0            # value 单位 = 20 HP（code/my_ai/az_intent/mcts.py:46）
COIN_SCALE = 300.0         # stats[22] = coins[player] / 300（features.py:227）
COIN_IDX = 22              # = extras[0]；summarize 出 22 维，故 extras 从 22 起
LIGHTNING_CD_IDX = 26      # = extras[4] = weapon_cooldowns[player, LIGHTNING(=1)] / 35

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")

#: P0-3 的冷却网格：stats[26] = cd/35，0 = 冷却已好（35 轮里剩 0 轮）
CDS = np.linspace(0.0, 1.0, 11)


# --------------------------------------------------------------------------- #
# 取样
# --------------------------------------------------------------------------- #
def _snap(st, feat: FeatureExtractor, r: int, player: int = 0,
          tag: str = "", seed: int = 0) -> dict:
    return {
        "round": r,
        "tag": tag,
        "seed": seed,
        "board": np.asarray(feat.encode_board(st, player), dtype=np.float32),
        "stats": np.asarray(feat.encode_stats(st, player), dtype=np.float32).ravel(),
        "n_my_tower": len(st.towers_of(player)),
        "n_enemy_tower": len(st.towers_of(1 - player)),
        "coins": int(st.coins[player]),
        "cd_light": float(st.weapon_cooldowns[player][1]),   # 列 1 = LIGHTNING_STORM
    }


def collect_trajectory(feat: FeatureExtractor, seeds=(7, 11), every: int = 25,
                       min_round: int = 40, max_round: int = 320) -> list:
    """**无塔基准**：和 `probe_value_grad.py:42-59` 完全一样的取样口径（只 advance_round）。"""
    out = []
    for seed in seeds:
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        for r in range(0, max_round):
            if r >= min_round and r % every == 0:
                out.append(_snap(st, feat, r, tag="traj_nop", seed=seed))
            st.advance_round()
            if st.terminal:
                break
    return out


def parse_tags(spec: str, default_seed) -> list[tuple[str, str]]:
    """`A:8,B:10` → [("A","8"), ("B","10")]；没写 seed 的用 default_seed。

    seed 一律当**字符串**传下去：日志文件名可能是 `10r.log` 这种重跑档，
    真正喂给引擎的整数 seed 在 `collect_log` 里剥数字得到。
    """
    out = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            t, s = tok.rsplit(":", 1)
            out.append((t.strip(), s.strip()))
        else:
            out.append((tok, str(default_seed)))
    return out


def collect_log(tag: str, seed, feat: FeatureExtractor, every: int = 10,
                min_round: int = 20, max_round: int = 400) -> list:
    """**有塔局面**：按 ladder 日志逐 op `apply_operation_list` 复现真实对局。

    `seed` 可以是 int 或字符串：整串当日志文件名，其中数字当引擎 seed。
    """
    import re as _re

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    stem = str(seed)
    engine_seed = int(_re.sub(r"[^0-9]", "", stem) or 0)
    log = REPO / "match_results" / "ladder_logs" / tag / f"{stem}.log"
    if not log.exists():
        print(f"[复现] ⚠️ 跳过：日志不存在 {log}")
        return []
    txt = log.read_text(encoding="utf-8", errors="replace")
    rounds: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rounds.setdefault(int(m.group(1)), {})[int(m.group(2)[1])] = [
            (int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]
    if not rounds:
        print(f"[复现] ⚠️ 跳过 {tag}/{stem}：日志里解析不出 op 行"
              f"（旧 tag 的中文标签是乱码，`原文` 被写成 `鍘熸枃`）")
        return []

    st = GameStateFacade.initial(seed=engine_seed, cold_handle_rule_illegal=True)
    out, bad = [], 0
    for r in range(0, max_round):
        for p in (0, 1):
            for (t, a0, a1) in rounds.get(r, {}).get(p, []):
                try:
                    st.apply_operation_list(p, [Operation(OperationType(t), a0, a1)])
                except Exception:
                    bad += 1
            if p == 1:
                st.advance_round()
        if st.terminal:
            break
        if r >= min_round and r % every == 0:
            out.append(_snap(st, feat, r, tag=tag, seed=stem))
    print(f"[复现] {tag}/{stem}.log：合法 op 全部 apply（跳过 {bad} 个非法），"
          f"取到 {len(out)} 个局面（round {min_round}..{out[-1]['round'] if out else '-'}，每 {every} 轮）")
    return out


def collect_logs(spec: str, default_seed, feat: FeatureExtractor, every: int = 10,
                 min_round: int = 20) -> list:
    """按 `tag:seed` 列表复现多局。**自动去重**：金币轨迹完全相同的两局视为同一局（日志重复存档）。"""
    out: list = []
    sigs: dict[tuple, str] = {}
    for tg, sd in parse_tags(spec, default_seed):
        grp = collect_log(tg, sd, feat, every=every, min_round=min_round)
        if not grp:
            continue
        sig = (len(grp), tuple(s["coins"] for s in grp),
               tuple(s["n_my_tower"] for s in grp))
        if sig in sigs:
            print(f"[复现] ⚠️ {tg}/{sd} 的金币/塔数轨迹与 {sigs[sig]} 逐项相同"
                  f" ⇒ 判定为同一局，**去重跳过**（避免把一局算两遍）")
            continue
        sigs[sig] = f"{tg}/{sd}"
        out.extend(grp)
    return out


def describe(name: str, grp: list):
    if not grp:
        print(f"[取样] {name}：0 个局面"); return
    coins = [s["coins"] for s in grp]
    cds = [s["cd_light"] for s in grp]
    per_tag = {}
    for s in grp:
        per_tag[s.get("tag", "?")] = per_tag.get(s.get("tag", "?"), 0) + 1
    print(f"[取样] {name}：{len(grp)} 个局面"
          f"（来源 {', '.join(f'{k}×{v}' for k, v in sorted(per_tag.items()))}）；"
          f"有己方塔 {sum(1 for s in grp if s['n_my_tower'] > 0)}、"
          f"有敌方塔 {sum(1 for s in grp if s['n_enemy_tower'] > 0)}")
    print(f"       金币中位 {int(np.median(coins))}（min {min(coins)} / max {max(coins)}）；"
          f"stats[22] 中位 {np.median(np.array([s['stats'][COIN_IDX] for s in grp])):.3f}；"
          f"闪电冷却中位 {np.median(cds):.0f} 轮"
          f"（cd>0 的 {sum(1 for c in cds if c > 0)}/{len(cds)}，"
          f"p90 {np.quantile(cds, 0.9):.0f}）"
          f" ⇒ cd 维度在真实局面里大多为 0，扫到 cd=1.0 属于输入空间外推")


# --------------------------------------------------------------------------- #
# 曲线
# --------------------------------------------------------------------------- #
def sweep(states, value_model, xs: np.ndarray, cd_layer: float | None = None) -> np.ndarray:
    """把 stats[COIN_IDX] 设成 xs 的每个值，逐个前向（batch 化）。返回 (n_states, n_xs)。"""
    n, m = len(states), len(xs)
    if n == 0:
        return np.zeros((0, m), dtype=np.float32)
    board = np.repeat(np.stack([s["board"] for s in states]), m, axis=0)
    stats = np.repeat(np.stack([s["stats"] for s in states]), m, axis=0)
    stats[:, COIN_IDX] = np.tile(xs, n)
    if cd_layer is not None:
        stats[:, LIGHTNING_CD_IDX] = cd_layer
    with torch.no_grad():
        out = value_model(torch.from_numpy(board).float(), torch.from_numpy(stats).float())
        v = (out["value"] if isinstance(out, dict) else out).reshape(n, m).numpy()
    return v


def curve_report(name: str, v: np.ndarray, xs: np.ndarray, grp: list | None = None,
                 per_state: bool = False) -> dict:
    """打印一条（族）曲线的形状读数。v: (n_states, m)。"""
    if v.size == 0:
        print(f"  [{name}] 无样本"); return {}
    mean = v.mean(axis=0)
    per_state_slope = np.polyfit(xs, v.T, 1)[0]
    A = np.stack([xs, np.ones_like(xs)], axis=1)
    coef, *_ = np.linalg.lstsq(A, mean, rcond=None)
    resid = mean - A @ coef
    ss_tot = float(((mean - mean.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    d = np.diff(mean)

    print(f"  [{name}] {v.shape[0]} 局面 × {v.shape[1]} 个金币点")
    print(f"    均值曲线 V 范围 {mean.min():+.5f} .. {mean.max():+.5f}"
          f"（跨度 {(mean.max() - mean.min()) * HP_SCALE:.4f} HP）；"
          f"同一局面内 V 的 std 中位 {np.median(v.std(axis=1)):.5f}")
    print(f"    线性拟合：斜率 {coef[0]:+.6f}/x（= 每 100 币 {(coef[0] / 3) * HP_SCALE:+.4f} HP）"
          f"  R²={r2:.5f}  最大偏离 {np.abs(resid).max():.5f}")
    print(f"    逐局面线性斜率：中位 {np.median(per_state_slope):+.6f}"
          f"  均值 {per_state_slope.mean():+.6f}"
          f"  min {per_state_slope.min():+.6f}  max {per_state_slope.max():+.6f}"
          f"  为正 {int((per_state_slope > 0).sum())}/{len(per_state_slope)}"
          f"  p10 {np.quantile(per_state_slope, 0.1):+.6f}"
          f"  p90 {np.quantile(per_state_slope, 0.9):+.6f}")
    print(f"    单调性：ΔV<0 的步数 {int((d < 0).sum())}/{len(d)}"
          f"（>0 即非单调）；最小步长 {d.min():+.6f}")

    seg = {}
    for lo, hi in ((0, 45), (45, 90), (90, 180), (180, 300)):
        i0 = int(np.argmin(np.abs(xs - lo / COIN_SCALE)))
        i1 = int(np.argmin(np.abs(xs - hi / COIN_SCALE)))
        if i1 <= i0:
            continue
        s = (mean[i1] - mean[i0]) / (xs[i1] - xs[i0])
        seg[(lo, hi)] = float(s)
        print(f"    段 [{lo:>3d},{hi:>3d}] 币：dV/dx = {s:+.6f}"
              f"  ⇒ 每 100 币 = {(s / 3) * HP_SCALE:+.4f} HP"
              f"；这 {hi - lo:>3d} 币共值 {(mean[i1] - mean[i0]) * HP_SCALE:+.4f} HP")

    if per_state and grp is not None:
        print(f"  [{name}] 逐局面明细（斜率 = 该局面自己 0→300 币的线性斜率）")
        print(f"    {'来源':<20} {'round':>5} {'coins':>5} {'我塔':>3} {'敌塔':>3} {'cd/35':>6}"
              f" {'slope':>10} {'V(0)':>9} {'V(300)':>9} {'ΔV×20[HP]':>10}")
        order = np.argsort(-per_state_slope)
        for i in order:
            s = grp[i]
            src = f"{s.get('tag', '?')}:{s.get('seed', '?')}"
            print(f"    {src:<20} {s['round']:>5} {s['coins']:>5} {s['n_my_tower']:>3}"
                  f" {s['n_enemy_tower']:>3} {s['cd_light']:>6.0f} {per_state_slope[i]:>+10.6f}"
                  f" {v[i, 0]:>+9.4f} {v[i, -1]:>+9.4f} {(v[i, -1] - v[i, 0]) * HP_SCALE:>+10.4f}")
    return {"mean": mean, "slope": float(coef[0]), "r2": float(r2),
            "per_state_slope_med": float(np.median(per_state_slope)),
            "per_state_slope_mean": float(per_state_slope.mean()),
            "per_state_slope": per_state_slope, "seg": seg}


def print_table(name: str, v: np.ndarray, xs: np.ndarray):
    mean, sd = v.mean(axis=0), v.std(axis=0)
    print(f"  [{name}] V(coins) 逐点（coins = stats[22]×300；V 单位 = 1，×20 = HP）")
    print("     coins  stats[22]    V(mean)     ±std       ΔV 上一格    V×20 [HP]")
    prev = None
    for i, x in enumerate(xs):
        dv = "" if prev is None else f"{mean[i] - prev:+.6f}"
        print(f"    {int(round(x * COIN_SCALE)):>6d}  {x:8.4f}  {mean[i]:+.6f}  {sd[i]:.6f}"
              f"  {dv:>12}  {mean[i] * HP_SCALE:+.4f}")
        prev = mean[i]


# --------------------------------------------------------------------------- #
# P0-3：V(coins × cd) 二维曲面
# --------------------------------------------------------------------------- #
def _slope_on(xs: np.ndarray, y: np.ndarray, lo: int = 0, hi: int | None = None) -> float:
    """线性最小二乘斜率（只在索引 [lo, hi] 这一段）。"""
    hi = len(xs) - 1 if hi is None else hi
    if hi <= lo:
        return float("nan")
    A = np.stack([xs[lo:hi + 1], np.ones(hi - lo + 1)], axis=1)
    c, *_ = np.linalg.lstsq(A, y[lo:hi + 1], rcond=None)
    return float(c[0])


def surface_report(name: str, grp: list, value_model, xs: np.ndarray, cds: np.ndarray):
    """扫 (cd, coins) 网格，打曲面 + 三个判据读数。"""
    surf = np.zeros((len(cds), len(xs)), dtype=np.float64)   # 均值曲面
    surf_sd = np.zeros_like(surf)
    for i, cd in enumerate(cds):
        v = sweep(grp, value_model, xs, cd_layer=float(cd))
        surf[i] = v.mean(axis=0)
        surf_sd[i] = v.std(axis=0)

    i90 = int(np.argmin(np.abs(xs - 0.30)))     # 90 币
    i45 = int(np.argmin(np.abs(xs - 0.15)))     # 45 币
    key = [0, i45, i90, int(np.argmin(np.abs(xs - 0.60))), len(xs) - 1]

    # ---- (a) 数值表：行 = cd，列 = 若干关键金币点 ---------------------------- #
    print(f"  [曲面] V(coins, cd) 数值表（行=冷却层，列=金币；V 单位 = 1，除以 20 = HP）")
    hdr = "".join(f"{int(round(xs[j] * COIN_SCALE)):>9d}" for j in key)
    print(f"    {'cd/35':>7} {'(轮)':>6}{hdr}")
    for i, cd in enumerate(cds):
        row = "".join(f"{surf[i, j]:+9.4f}" for j in key)
        print(f"    {cd:7.2f} {cd * 35:6.1f}{row}")
    print(f"    （列头单位 = 金币数；第 i 行 = 把 stats[26] 钉成 {', '.join(f'{c:.1f}' for c in cds[:3])} …,"
          f" 共 {len(grp)} 个局面的均值）")

    # ---- (b) 逐 cd 的斜率剖面 ------------------------------------------------ #
    print(f"\n  [斜率剖面] 每个冷却层上，V 沿 coins 的分段斜率（dV/d(stats[22])；每 100 币的 HP = ×20/3）")
    print(f"    {'cd/35':>7} | {'0–90币段':>10} {'90–300币段':>11} {'全域':>10} {'段比 低/高':>10}"
          f" | {'前90币':>8} {'90–300币':>9} | {'线性R²':>7}")
    prof = []
    for i, cd in enumerate(cds):
        y = surf[i]
        s_lo = _slope_on(xs, y, 0, i90)
        s_hi = _slope_on(xs, y, i90, len(xs) - 1)
        s_all = _slope_on(xs, y, 0, len(xs) - 1)
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ c
        r2 = 1 - float((res ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
        prof.append(dict(cd=float(cd), s_lo=s_lo, s_hi=s_hi, s_all=s_all, r2=float(r2)))
        print(f"    {cd:7.2f} | {s_lo:+10.6f} {s_hi:+11.6f} {s_all:+10.6f}"
              f" {s_lo / s_hi if abs(s_hi) > 1e-12 else float('nan'):10.2f}"
              f" | {(y[i90] - y[0]) * HP_SCALE:+8.4f} {(y[-1] - y[i90]) * HP_SCALE:+9.4f}"
              f" | {r2:7.4f}")

    # ---- 判据 1：条件斜率比 cd=0 vs cd≥0.6 ---------------------------------- #
    s_cd0 = prof[0]["s_all"]
    late = [p for p in prof if p["cd"] >= 0.6 - 1e-9]
    s_late = float(np.mean([p["s_all"] for p in late]))
    ratio = abs(s_cd0) / max(abs(s_late), 1e-12)
    print(f"\n  [判据1] 条件斜率比：|∂V/∂coins | cd=0| / |∂V/∂coins | cd≥0.6|")
    print(f"          cd=0   的斜率 {s_cd0:+.6f}"
          f"（每 100 币 {(s_cd0 / 3) * HP_SCALE:+.4f} HP）")
    print(f"          cd≥0.6 的均值斜率 {s_late:+.6f}"
          f"（每 100 币 {(s_late / 3) * HP_SCALE:+.4f} HP）；"
          f"逐层 = " + ", ".join(f"{p['s_all']:+.6f}" for p in late))
    verdict1 = ("几乎不分层（<1.5 ⇒ 没学到「该攒钱的时机」）" if ratio < 1.5
                else "明显分层（≥2 ⇒ 学到了分段）" if ratio >= 2.0
                else "弱分层（1.5–2，灰区）")
    print(f"          ⇒ 比值 {ratio:.2f}× ⇒ **{verdict1}**")

    # ---- 判据 2：coins=0.30（90 币）处有没有折点 ---------------------------- #
    print(f"\n  [判据2] coins=0.30（=90 币）处有没有折点？")
    print("          做法：沿 coins 看每步局部斜率 d1，找「斜率变化最剧烈」的位置；"
          "并给出突变幅度 / 平均斜率（<0.2 视为无实质折点）。")
    d1 = np.diff(surf, axis=1) / np.diff(xs)          # 局部斜率 (len(cds), len(xs)-1)
    for i, cd in enumerate(cds):
        if cd not in (0.0, 0.3, 0.6, 1.0):
            continue
        dd = np.abs(np.diff(d1[i]))
        j_kink = int(np.argmax(dd)) + 1
        below = d1[i, max(i90 - 1, 0)]
        above = d1[i, min(i90, len(xs) - 2)]
        mean_slope = float(np.abs(d1[i]).mean())
        print(f"    cd={cd:4.2f}：突变最大处 coins≈{int(round(xs[j_kink] * COIN_SCALE)):>3d} 币"
              f"（突变幅度 {dd.max():.6f} = 平均斜率的 {dd.max() / max(mean_slope, 1e-12):.1%}）"
              f"；90 币左斜率 {below:+.6f} / 右 {above:+.6f}（比 {abs(below) / max(abs(above), 1e-12):.3f}×）")
    dd_all = np.abs(np.diff(d1, axis=1))
    mag_all = dd_all.max(axis=1) / np.maximum(np.abs(d1).mean(axis=1), 1e-12)
    print(f"    全 {len(cds)} 层里最大突变幅度 / 平均斜率：中位 {np.median(mag_all):.1%}"
          f" / 最大 {mag_all.max():.1%}"
          f"  ⇒ {'没有实质折点（<20%）' if mag_all.max() < 0.2 else '存在 >20% 的斜率突变'}")
    kinks = [int(np.argmax(np.abs(np.diff(d1[i])))) + 1 for i in range(len(cds))]
    print(f"    各层突变点落到的 coins（币）：{sorted({int(round(xs[j] * COIN_SCALE)) for j in kinks})}")
    # 90 币两侧的斜率比在所有 cd 上的范围
    rat = []
    for i in range(len(cds)):
        b = d1[i, max(i90 - 1, 0)]
        a = d1[i, min(i90, len(xs) - 2)]
        rat.append(abs(b) / max(abs(a), 1e-12))
    print(f"    90 币左/右斜率比（11 层）：{', '.join(f'{r:.2f}' for r in rat)}"
          f"  ⇒ 范围 {min(rat):.2f}–{max(rat):.2f}（1.0 = 完全无折点）")

    # ---- 判据 3：∂V/∂cd 的方向与量级 -------------------------------------- #
    print(f"\n  [判据3] 反过来：把 coins 钉住看 ∂V/∂cd（cd 大 = 闪电还早）")
    for j in (0, i45, i90, len(xs) - 1):
        col = surf[:, j]
        sc = _slope_on(cds, col)
        print(f"    coins={int(round(xs[j] * COIN_SCALE)):>3d}：∂V/∂cd = {sc:+.6f}"
              f"（每 10 轮冷却 ≈ {sc * 10 / 35 * HP_SCALE:+.4f} HP）")
    print(f"\n  [曲面] 同一 (coins, cd) 上跨局面的 std：中位 {np.median(surf_sd):.5f}"
          f" / 最大 {surf_sd.max():.5f}（曲面的跨局面波动，越小说明这层定价越一致）")

    # 返回关键数（供文档引用）
    np.savez(REPO / "match_results" / f"probe_p03_surface_{name}.npz",
             xs=xs, cds=cds, surface=surf, surface_sd=surf_sd,
             s_all=np.array([p["s_all"] for p in prof]),
             s_lo=np.array([p["s_lo"] for p in prof]),
             s_hi=np.array([p["s_hi"] for p in prof]))
    print(f"  [落盘] match_results/probe_p03_surface_{name}.npz")
    return {"ratio": ratio, "s_cd0": s_cd0, "s_late": s_late,
            "surf": surf, "prof": prof}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="UNIFIED_SPLIT_VERBOSE:8,UNIFIED_AB_A:10,UNIFIED_AB_A:11",
                    help="有塔组的 ladder 日志，逗号分隔，可带 seed（tag:seed）；自动去重")
    ap.add_argument("--seed", type=int, default=8, help="--tag 里没写 seed 时用它")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--every", type=int, default=10, help="有塔组每几轮取一个局面")
    ap.add_argument("--traj-every", type=int, default=25, help="无塔基准每几轮取一个局面")
    ap.add_argument("--points", type=int, default=41, help="0..300 币上的扫描点数")
    ap.add_argument("--no-synthetic", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    feat = FeatureExtractor(max_actions=96)
    _cls, _pos, value_model = load_three_models(args.ckpt)
    value_model.eval()
    st0 = GameStateFacade.initial(seed=0, cold_handle_rule_illegal=True)
    n_stats = int(feat.encode_stats(st0, 0).size)
    print(f"[值网] {args.ckpt} 的 value_state；stats 维度 = {n_stats}"
          f"；金币下标 = stats[{COIN_IDX}]，闪电冷却 = stats[{LIGHTNING_CD_IDX}]")

    no_t = collect_trajectory(feat, every=args.traj_every)
    log_s = collect_logs(args.tag, args.seed, feat, every=args.every)
    no_tower_from_log = [s for s in log_s if s["n_my_tower"] == 0]
    with_t = [s for s in log_s if s["n_my_tower"] > 0]
    describe("B. no_tower（只 advance_round，从不 apply op）", no_t)
    describe("C. tower（日志复现，场上有己方塔）", with_t)

    xs = np.linspace(0.0, 1.0, args.points)

    if not args.no_synthetic:
        syn = [{"board": np.zeros((28, 19, 19), dtype=np.float32),
                "stats": np.zeros(n_stats, dtype=np.float32),
                "n_my_tower": 0, "n_enemy_tower": 0, "coins": 0, "cd_light": 0.0, "round": -1}]
        print("\n" + "=" * 82)
        print("A. 纯合成输入：全零 board + 全零 stats，只动 stats[22]（值网的裸函数形态）")
        print("=" * 82)
        v_syn = sweep(syn, value_model, xs)
        print_table("synthetic", v_syn, xs)
        curve_report("synthetic", v_syn, xs)

    for tag, name, grp in (("B", "no_tower", no_t), ("C", "tower", with_t)):
        print("\n" + "=" * 82)
        print(f"{tag}. 真实局面 · {name}（{len(grp)} 个）—— 只动 stats[22]")
        print("=" * 82)
        if not grp:
            print("  （没有这一组样本）"); continue
        v = sweep(grp, value_model, xs)
        print_table(name, v, xs)
        curve_report(name, v, xs, grp=grp, per_state=True)

        # 每个局面在**它自己的真实金币值**处的 autograd 斜率（与 P0-1 交叉验证）
        with torch.enable_grad():
            st_t = torch.tensor(np.stack([s["stats"] for s in grp]).copy()); st_t.requires_grad_(True)
            bd_t = torch.tensor(np.stack([s["board"] for s in grp]).copy()); bd_t.requires_grad_(True)
            out = value_model(bd_t, st_t)
            vv = (out["value"] if isinstance(out, dict) else out).sum()
            gs, gb = torch.autograd.grad(vv, [st_t, bd_t])
        g = gs[:, COIN_IDX].numpy()
        print(f"  交叉验证 [autograd @ 各自真实金币] ∂V/∂(我币/300)："
              f"中位 {np.median(g):+.6f}  均值 {g.mean():+.6f}"
              f"  为正 {int((g > 0).sum())}/{len(g)}"
              f"  ⇒ 每 100 币 = {(np.median(g) / 3) * HP_SCALE:+.4f} HP")
        bm = np.abs(gb.numpy()).mean(axis=(0, 2, 3))
        print("  [autograd] 塔相关 board 通道平均 |∂V/∂x|："
              + "  ".join(f"ch{i}={bm[i]:.5f}" for i in (4, 5, 6, 7, 8, 9)))

    if no_tower_from_log:
        print(f"\n  （附注：日志里有 {len(no_tower_from_log)} 个局面还没建塔，未参与统计）")

    # ---- P0-3：V(coins × cd) 二维曲面 ---------------------------------------
    for tag, name, grp in (("P0-3·C", "tower", with_t), ("P0-3·B", "no_tower", no_t)):
        if not grp:
            continue
        print("\n" + "=" * 82)
        print(f"{tag}. V(coins × cd) 二维曲面 · {name}（{len(grp)} 个真实局面）")
        print("   stats[22]=coins/300 扫 0→1.0，stats[26]=我方闪电剩余冷却/35 扫 0→1.0（0=已好）")
        print("   其余 40 维 stats + 整个 board 逐位不动")
        print("=" * 82)
        surface_report(name, grp, value_model, xs, CDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
