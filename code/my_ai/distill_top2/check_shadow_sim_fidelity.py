"""核对「影子模拟器 vs 真实引擎」：同一批候选计划往后 8 回合，两边的估值/排序一致吗？

关键设计（避免把"噪声"误判成"近似误差"）：
  ρ(影子A, 真实)   —— 保真度
  ρ(影子A, 影子B)  —— **可信度上限**（A/B 同候选、不同 rollout 种子的两次独立采样）
  若 ρ(影子A,影子B) 本身就不高，说明该状态里候选之间差异太小、测量被采样噪声主导，
  此时"影子 vs 真实"的低相关不能全算在近似模型头上。同时报告候选分数的离散度。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.distill_top2.lure_value_eval import evaluate_terminal  # noqa: E402
from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
ENTS = _REPO / "training_history/distill_top2/abl_linear2_raw"
LOGS = _REPO / "match_results"
SEED = 732
ROLLOUTS = int(os.environ.get('FID_ROLLOUTS', '16'))
N_CAND = 40
ROUNDS = (60, 90, 120, 150, 180, 210, 240, 300)


def make_env():
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def call_tool(payload: dict) -> dict:
    proc = subprocess.run([str(EXE)], input=(json.dumps(payload) + "\n").encode(),
                          capture_output=True, env=make_env())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])
    return json.loads(proc.stdout.decode("utf-8", "replace").splitlines()[0])


def load_ops(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if '"rank":1,' not in line or "base_heuristic" not in line:
            continue
        record = json.loads(line)
        ops = []
        if record["key"] != "hold":
            for segment in record["key"].split("|"):
                if segment.startswith("F:"):
                    continue
                parts = segment.split(":")
                if len(parts) < 2:
                    continue
                values = [int(x) for x in parts[:3]] + [-1] * (3 - min(3, len(parts)))
                ops.append((values[0], values[1], values[2]))
        out[(record["round"], record["player"])] = ops
    return out


def spear(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 1e-9 else float("nan")


def replay_to(target: int, ops0: dict, ops1: dict) -> GameStateFacade:
    facade = GameStateFacade.initial(SEED)
    for r in range(target):
        for player, table in ((0, ops0), (1, ops1)):
            ops = table.get((r, player), [])
            if ops:
                objs = [Operation(OperationType(int(t)), int(a0), int(a1)) for t, a0, a1 in ops]
                facade.apply_operation_list(player, objs)
        facade.advance_round()
    return facade


def main() -> int:
    ops0 = load_ops(LOGS / f"ai0_runnerup_seed{SEED}.stderr.log")
    ops1 = load_ops(LOGS / f"ai1_ai3_linear_seed{SEED}.stderr.log")
    dumps = [json.loads(line) for line in (ENTS / f"ents_seed{SEED:05d}.jsonl").open(encoding="utf-8")]
    print(f"[日志] {max(r for r, _ in ops0) + 1} 回合")

    bad = 0
    for r in (0, 50, 100, 200, 300, 400):
        facade = replay_to(r, ops0, ops1)
        dump = dumps[r]
        ok = ([int(b.hp) for b in facade.bases] == [int(b[0]) for b in dump["bases"]]
              and [int(c) for c in facade.coins] == [int(c) for c in dump["coins"]]
              and len(list(facade.towers)) == len(dump["towers"])
              and len(list(facade.ants)) == len(dump["ants"]))
        bad += 0 if ok else 1
    print(f"[对账] 6 个抽样回合里不一致 {bad} 个（0 = 重放正确）")

    print("\n=== 同一批候选：影子模拟器 vs 真实引擎（都是往后 8 回合）===")
    print(f"{'回合':>5} {'候选':>4} {'ρ(影子,真实)':>12} {'ρ(影子A,影子B)':>14} "
          f"{'top1一致':>8} {'偏差':>7} {'影子离散':>8} {'真实离散':>8}")
    rows = []
    for target in ROUNDS:
        facade = replay_to(target, ops0, ops1)
        state = dumps[target]
        prior = call_tool(dict(state, seed=SEED, player=0, mode="prior"))
        plans = prior["plans"][:N_CAND]
        candidates = [[[op["type"], op["arg0"], op["arg1"]] for op in plan["ops"]] for plan in plans]

        base = dict(state, seed=SEED, player=0, mode="score_ops", rollouts=ROLLOUTS, candidates=candidates)
        run_a = call_tool(base)
        run_b = call_tool(dict(base, serial=target + 1000))

        def terminal_expectation(result):
            # 只比「8 回合后 evaluate_terminal 的期望」：扣掉 c1_bonus（真实引擎侧没有这一项）和 heuristic
            return np.array([(row["mean"] - row["c1_bonus"]) if row["valid"] else np.nan
                             for row in result["candidates"]], dtype=float)

        shadow_a = terminal_expectation(run_a)
        shadow_b = terminal_expectation(run_b)

        real = []
        for plan in plans:
            sim = facade.clone()
            ops = [Operation(OperationType(int(o["type"])), int(o["arg0"]), int(o["arg1"]))
                   for o in plan["ops"]]
            if ops:
                sim.apply_operation_list(0, ops)
            for _ in range(8):
                if sim.terminal:
                    break
                sim.advance_round()
            real.append(evaluate_terminal(sim, 0))
        real = np.array(real, dtype=float)

        mask = np.isfinite(shadow_a) & np.isfinite(shadow_b) & np.isfinite(real)
        if mask.sum() < 5:
            continue
        rho = spear(shadow_a[mask], real[mask])
        rho_rel = spear(shadow_a[mask], shadow_b[mask])
        top_a = int(np.nanargmax(np.where(mask, shadow_a, -np.inf)))
        top_r = int(np.nanargmax(np.where(mask, real, -np.inf)))
        bias = float(np.mean(shadow_a[mask] - real[mask]))
        rows.append((rho, rho_rel, top_a == top_r))
        print(f"{target:>5} {int(mask.sum()):>4} {rho:>12.3f} {rho_rel:>14.3f} "
              f"{str(top_a == top_r):>8} {bias:>+7.0f} "
              f"{float(np.nanstd(shadow_a[mask])):>8.0f} {float(np.nanstd(real[mask])):>8.0f}")

    if rows:
        rhos = np.array([r[0] for r in rows])
        rels = np.array([r[1] for r in rows])
        print(f"\n保真度 ρ(影子,真实): 中位 {np.median(rhos):+.3f}  均值 {np.mean(rhos):+.3f}")
        print(f"可信度上限 ρ(影子A,影子B): 中位 {np.median(rels):+.3f}  均值 {np.mean(rels):+.3f}")
        print(f"top-1 一致: {sum(1 for r in rows if r[2])}/{len(rows)}")
        print(f"归一化保真度（ρ_真实 ÷ ρ_可信度）: 中位 "
              f"{float(np.median(rhos / np.maximum(rels, 1e-3))):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
