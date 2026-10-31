"""安慰剂注入的**确定性 + 合法性**探针（2026-10-28；报告 `docs/placebo_inject_ab_20261028.md` §3）。

## 为什么需要它

`AZAI_ORACLE_PLACEBO=1` 臂的全部说服力都压在两句话上：

1. **可复现**：候选只由 `f(AZAI_FLAT_SEED, 决策序号)` 决定 ⇒ 同一局重跑必须生成同一串候选；
2. **合法**：生成的 op 必须过引擎的只读校验（否则严格模式当场判负，整批读数作废）。

"读一遍代码觉得没问题"不算证据 —— 这个探针**不起对局、不起子进程**，直接用真实 ladder 日志
重放出真实局面，然后逐条验证。跑一次 ~10-30 秒，可以当回归测试。

## 做法

1. 用 `repro_search_crash.read_ops()` 从 `match_results/ladder_logs/<tag>/<seed>.log` 的
   `--trace-ops` 行重放双方操作 ⇒ 拿到**真的**有塔/蚂蚁/金币的局面（p0 决策前 + p1 决策前各一份）；
2. 在每个采样局面上调 `az_bridge_ai._placebo_bundle(...)`：
   * **① 可复现**：同一 `(FLAT_SEED, 决策序号, 局面)` 连调两次，结果逐字相同；
   * **② 独立流**：换 `FLAT_SEED` ⇒ 序列不同；并断言整个探针**没有动 numpy 全局 RNG**
     （搜索用的是新建的 `np.random.default_rng`，这里证明 placebo 也不碰全局流）；
   * **③ 合法**：每个生成的 op 都过 `MVS.check_ops`（**dry + clone 冷路径双查**，权威口径；
     与 `_block_legal_gate` 同一个检查器）—— 非法的记一条 FAIL；
   * **④ 分布**：`hold_p=0` ⇒ hold 占比必须 = 0；`hold_p=0.78` ⇒ 覆盖 600 个决策序号，
     二项 95%CI 必须覆盖 0.78（"对齐亚军 78% HOLD"这刀才成立）；
   * **⑤ 默认零影响**：`AZAI_ORACLE_PLACEBO` 不设时 `_PLACEBO_IMPORTS` 缓存必须是空的
     （= 默认路径连一个 SDK 常量都不 import），且 `_ORACLE["placebo"]` 为 False。

## 用法

```bash
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/probe_placebo_inject.py
# 失败返回 1（可当回归测试）
```
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

FAILS: list[str] = []


def _check(ok: bool, name: str, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return bool(ok)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _ci_binom(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wald 95%CI（n=600 时够用；探针只做数量级核对，不做严谨推断）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    se = (p * (1 - p) / n) ** 0.5
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="VS_RU_dump", help="ladder 日志 tag（match_results/ladder_logs/<tag>/）")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 5, 20, 60, 120, 200, 300, 400])
    ap.add_argument("--flat-seed", type=int, default=53, help="正式 A/B 用的 AZAI_FLAT_SEED")
    ap.add_argument("--n-hold", type=int, default=600, help="hold_p=0.78 的抽样决策数")
    args = ap.parse_args()

    import numpy as np
    import os
    # 探针本身**不启用** placebo（只调纯函数）；显式清掉，免得环境里残留
    os.environ.pop("AZAI_ORACLE_PLACEBO", None)
    os.environ.pop("AZAI_ORACLE_INJECT", None)
    os.environ.pop("AZAI_ORACLE_PLACEBO_HOLD_P", None)

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    bridge = _load("az_bridge_ai", REPO / "code/test_match/az_bridge_ai.py")
    MVS = bridge.MVS
    rsc = _load("_repro_search_crash", REPO / "code/test_match/repro_search_crash.py")

    # ── ⑤ 默认零影响：此刻还没调过 placebo ⇒ 懒加载缓存必须为空 ────────────────────
    _check(bridge._PLACEBO_IMPORTS == {},
           "⑤ 默认路径不 import placebo 的 SDK 常量", f"_PLACEBO_IMPORTS={bridge._PLACEBO_IMPORTS}")
    _check(bridge._ORACLE.get("placebo") is False,
           "⑤ 默认 _ORACLE['placebo']=False（不设 AZAI_ORACLE_PLACEBO）")

    # ── ① 重放真实对局，取局面快照 ───────────────────────────────────────────
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        print(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行，换一个 tag/seed", flush=True)
        return 2
    want = sorted({int(x) for x in args.rounds})
    snaps: list[tuple[int, int, object]] = []
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, 0, st.clone()))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))
        st.advance_round()
    print(f"[probe] tag={args.tag} seed={args.seed} 快照={len(snaps)} 个"
          f"（回合 {want} × 2 座位），FLAT_SEED={args.flat_seed}", flush=True)

    # ── ① 可复现 + ③ 合法（逐条 dry+cold 双查）────────────────────────────────
    #   每个快照上连抽 25 个决策序号 ⇒ 400 个生成样本，类别直方图/合法性都有统计意义
    n_np0 = np.random.get_state()[1].copy()
    kinds: dict = {}
    legal_fail = 0
    n_ops_checked = 0
    n_gen = 0
    reproducible = True
    for (rnd, pl, snap) in snaps:
        dec_n = (rnd * 2) + pl          # 决策序号：一局里第几次决策（0-based 的近似；只用于取种子）
        b1, k1 = bridge._placebo_bundle(snap.clone(), pl, dec_n, args.flat_seed, 0.0)
        b2, k2 = bridge._placebo_bundle(snap.clone(), pl, dec_n, args.flat_seed, 0.0)
        if (b1, k1) != (b2, k2):
            reproducible = False
        for _dn in range(dec_n, dec_n + 25):
            bundle, kind = bridge._placebo_bundle(snap.clone(), pl, _dn, args.flat_seed, 0.0)
            n_gen += 1
            kinds[kind] = kinds.get(kind, 0) + 1
            if not bundle:
                continue
            ops = [Operation(OperationType(int(t)), int(a0), int(a1)) for (t, a0, a1) in bundle]
            stats = MVS.IllegalStats()
            bad = MVS.check_ops(snap.clone(), pl, ops, stats,
                                seed=args.seed, round_index=rnd, tag="placebo_probe",
                                cold=False, log=lambda m: None)
            n_ops_checked += len(ops)
            if bad:
                legal_fail += 1
                print(f"[FAIL] ③ r{rnd} p{pl} 生成的 op 非法: {bundle} -> {bad}", flush=True)
    _check(reproducible, "① 同一 (FLAT_SEED, 决策序号, 局面) 连调两次 ⇒ 结果逐字相同")
    _check(legal_fail == 0, "③ 生成的 op 全部合法（dry + clone 冷路径双查）",
           f"{n_gen} 次生成 / 检查了 {n_ops_checked} 个 op，非法 {legal_fail} 条")
    print(f"[probe] 生成直方图（hold_p=0，{n_gen} 次）: {dict(sorted(kinds.items()))}", flush=True)
    _check(kinds.get("hold", 0) == 0, "④ hold_p=0 ⇒ 一次 HOLD 都不出现",
           f"hold={kinds.get('hold', 0)}")

    # ── ② 独立流：换 FLAT_SEED ⇒ 序列不同；且没动 numpy 全局 RNG ────────────────
    _mid = snaps[len(snaps) // 2]
    seq = {}
    for fs in (args.flat_seed, args.flat_seed + 1):
        seq[fs] = tuple(bridge._placebo_bundle(_mid[2].clone(), _mid[1], n, fs, 0.0)[0]
                        for n in range(40))
    same = sum(1 for a, b in zip(seq[args.flat_seed], seq[args.flat_seed + 1]) if a == b)
    _check(same < 40, "② 换 FLAT_SEED ⇒ 候选序列不同（不是同一个随机流）",
           f"40 个决策里相同 {same} 个")
    n_np1 = np.random.get_state()[1]
    _check(bool((n_np0 == n_np1).all()), "② placebo 生成不消费 numpy 全局 RNG（与搜索的流无关）")

    seq2 = tuple(bridge._placebo_bundle(_mid[2].clone(), _mid[1], n, args.flat_seed, 0.0)[0]
                 for n in range(40))
    _check(seq2 == seq[args.flat_seed], "① 同一串决策序号重跑 ⇒ 候选序列逐字相同（跨调用）")

    # ── ④ hold_p=0.78 的 HOLD 占比（对齐亚军 78% HOLD）───────────────────────
    hs = sum(1 for n in range(args.n_hold)
             if bridge._placebo_bundle(_mid[2].clone(), _mid[1], n, args.flat_seed, 0.78)[1] == "hold")
    lo, hi = _ci_binom(hs, args.n_hold)
    _check(lo <= 0.78 <= hi, "④ hold_p=0.78 ⇒ 实测 HOLD 占比的 95%CI 覆盖 0.78",
           f"{hs}/{args.n_hold} = {hs / args.n_hold:.3f}  95%CI [{lo:.3f}, {hi:.3f}]")

    # ── 耗时（决定"注入成本"的量级；P 臂没有 oracle 查询）─────────────────────
    import time as _t
    _n = 300
    t0 = _t.perf_counter()
    for n in range(_n):
        bridge._placebo_bundle(_mid[2].clone(), _mid[1], n, args.flat_seed, 0.0)
    dt = (_t.perf_counter() - t0) / _n * 1000.0
    print(f"[probe] 单次生成耗时: {dt:.3f} ms/决策（300 次平均；A 臂的 oracle 查询是 "
          f"810-1438 ms/决策）", flush=True)

    print("", flush=True)
    if FAILS:
        print(f"=== {len(FAILS)} 项 FAIL === {FAILS}", flush=True)
        return 1
    print("=== 全部 PASS ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
