"""统一候选生成（方案 D）的**离线冒烟/探针**（2026-10-29）。

在**真实对局局面**上跑 `UnifiedCandidateGen`（不依赖对局进程），打印每条决策的：
  * 候选数 / 长度分布（0/1/2/3 op）
  * 含 HOLD 的条数、含升级（op 12）的条数、各通道/op 的构成
  * 每条候选是否 `can_apply`（**严格自检**：逐条重新 dry-run，一条非法就报警）
  * 阶段 1（idle rollout）耗时 与 候选生成耗时

局面从 `match_results/ladder_logs/<tag>/<seed>.log` 的 `--trace-ops` 行重放得到
（与 `_only_extra_zero_impact.py` 同一个 loader），⇒ 真的有塔/蚁/金币的中后局。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/smoke_unified_cand.py \
    --tag INJECT_VS_RU_B --seed 8 --rounds 20 60 120 200 300 --player 0 --k 48 \
    --out match_results/unified_cand/smoke.log
```
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_REAL_STDOUT = sys.stdout

BASE_SCRIPT = REPO / "code" / "test_match" / "_inject_vs_runnerup.sh"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base_env() -> dict[str, str]:
    txt = BASE_SCRIPT.read_text(encoding="utf-8")
    m = re.search(r'^BASE="(.*)"$', txt, re.M)
    if not m:
        raise SystemExit("读不到 _inject_vs_runnerup.sh 里的 BASE")
    out = {}
    for kv in m.group(1).split(";"):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="INJECT_VS_RU_B")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[20, 60, 120, 200, 300])
    ap.add_argument("--player", type=int, default=-1, help="-1 = 两个座位都跑")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="", help="逐决策读数落 JSON（供报告统计）")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg, file=_REAL_STDOUT, flush=True)
        lines.append(msg)

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"
    os.environ["AZAI_CAND_MODE"] = "unified"
    os.environ["AZAI_K"] = str(int(args.k))

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    rsc = _load("_repro_search_crash_uc", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        # 允许直接给 ladder 目录名（老 tag 的日志文件名可能不是 <seed>.log）
        emit(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行")
        return 2
    want = set(int(x) for x in args.rounds)
    snaps: list = []
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, 0, st.clone()))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))
        st.advance_round()
    if args.player >= 0:
        snaps = [s for s in snaps if int(s[1]) == int(args.player)]

    # ── 建搜索器：与部署**同一条**构造路径（az_bridge_ai.build_engine）─────────────
    bridge = _load("az_bridge_ai_uc", REPO / "code" / "test_match" / "az_bridge_ai.py")
    searcher, _model, _feat = bridge.build_engine(args.seed, 0)
    emit(f"[uc] 搜索器 = {type(searcher).__name__}  k={searcher.k}  "
         f"rollouts_per_candidate={searcher.rollouts_per_candidate}  "
         f"horizon={searcher.horizon}  rounds_budget={searcher.max_total_rounds}  "
         f"rollout_mode={searcher.rollout_mode}")
    emit(f"[uc] 局面 = {args.tag}/seed{args.seed}  快照 {len(snaps)} 个  "
         f"BASE={len(env)} 项（AZAI_K={env.get('AZAI_K')} ckpt={env.get('AZAI_CKPT')}）")

    rows: list[dict] = []
    for (rnd, player, snap) in snaps:
        t0 = time.perf_counter()
        res = searcher.search(snap.clone(), int(player), temperature=0.0)
        dt = (time.perf_counter() - t0) * 1000.0
        stt = dict(getattr(searcher, "last_cand_stats", {}) or {})
        # ── 严格合法性自检：候选逐条 + 选中那手逐条 `can_apply` ───────────────────
        n_bad_cand = 0
        for b in res.bundles:
            _pend: list = []
            for op3 in (b or ()):
                op = Operation(OperationType(int(op3[0])), int(op3[1]), int(op3[2]))
                if not snap.can_apply_operation(int(player), op, _pend):
                    n_bad_cand += 1
                    break
                _pend.append(op)
        _pend = []
        n_bad_chosen = 0
        for op3 in (res.chosen_bundle or ()):
            op = Operation(OperationType(int(op3[0])), int(op3[1]), int(op3[2]))
            if not snap.can_apply_operation(int(player), op, _pend):
                n_bad_chosen += 1
                break
            _pend.append(op)
        sizes = {0: 0, 1: 0, 2: 0, 3: 0}
        n_hold = n_upg = n_build = n_down = n_light = n_super = n_base = 0
        for b in res.bundles:
            sizes[len(b)] = sizes.get(len(b), 0) + 1
            if len(b) == 0:
                n_hold += 1
            for op3 in b:
                t = int(op3[0])
                if t == 12:
                    n_upg += 1
                elif t == 11:
                    n_build += 1
                elif t == 13:
                    n_down += 1
                elif t == 21:
                    n_light += 1
                elif t in (22, 23, 24):
                    n_super += 1
                elif t in (31, 32):
                    n_base += 1
        chosen = tuple(tuple(int(x) for x in op) for op in (res.chosen_bundle or ()))
        row = dict(round=int(rnd), player=int(player),
                   n_cand=len(res.bundles), n_pool=stt.get("n_pool", 0),
                   n_dup=stt.get("n_dup", 0), p1_ms=round(float(stt.get("p1_ms", 0.0)), 2),
                   decide_ms=round(dt, 2),
                   sizes=sizes, n_hold=n_hold, n_upg=n_upg, n_build=n_build, n_down=n_down,
                   n_light=n_light, n_super=n_super, n_base=n_base,
                   chosen=list(chosen), chosen_is_hold=int(len(chosen) == 0),
                   chosen_upg=int(any(op[0] == 12 for op in chosen)),
                   n_rollouts=int(searcher.last_rollouts),
                   n_rounds=int(searcher.last_rounds),
                   bad_cand=n_bad_cand, bad_chosen=n_bad_chosen,
                   scores=[round(float(x), 5) for x in np.asarray(searcher.last_scores).tolist()])
        rows.append(row)
        emit(f"[uc] r{rnd:<4} p{player} n_cand={row['n_cand']:<3} 池={row['n_pool']:<3} "
             f"重复={row['n_dup']:<3} 长度={row['sizes']} "
             f"含HOLD={row['n_hold']} 含升级={row['n_upg']} 建塔={row['n_build']} "
             f"降级={row['n_down']} 闪电={row['n_light']} 超武={row['n_super']} 基地={row['n_base']} | "
             f"p1={row['p1_ms']:.1f}ms 决策={row['decide_ms']:.0f}ms N={row['n_rollouts']} "
             f"选中={row['chosen']} "
             f"{'!! 候选非法' if n_bad_cand else ''}{' !! 选中非法' if n_bad_chosen else ''}")

    n_bad = sum(r["bad_cand"] + r["bad_chosen"] for r in rows)
    emit(f"[uc] 合计：{len(rows)} 个决策，非法候选/选中 = {n_bad}（必须 0）")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        emit(f"[uc] 逐决策读数 -> {out}")
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[uc] 已写入 {out}", file=_REAL_STDOUT, flush=True)
    return 0 if n_bad == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
