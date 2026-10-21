"""测 rollout 时间与「推进回合数 h」的关系：T(h) ≈ a + b·h。

用 `score_ops` 的 horizon 参数（同一批候选、只改 h），两次采样取中位数。
"""
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
ENTS = Path(r"D:\2026智能体大赛_新2_distill\training_history\distill_top2\abl_linear2_raw\ents_seed00732.jsonl")
N_CAND, ROLLOUTS, REPS = 40, 32, 3


def env():
    e = os.environ.copy()
    e["PATH"] = r"C:\mingw64\bin;" + e.get("PATH", "")
    return e


state = json.loads(ENTS.open(encoding="utf-8").readlines()[200])


def tool(payload):
    proc = subprocess.run([str(EXE)], input=(json.dumps(payload) + "\n").encode(),
                          capture_output=True, env=env())
    if proc.returncode != 0:
        raise RuntimeError(f"rc={proc.returncode} {proc.stderr[:200]}")
    return json.loads(proc.stdout.decode("utf-8", "replace").splitlines()[0])


# 先拿一批合法候选（用 horizon=8 的一次调用顺带确认 n）
prior = tool(dict(state, seed=732, player=0, mode="prior"))
candidates = [[[op["type"], op["arg0"], op["arg1"]] for op in plan["ops"]]
              for plan in prior["plans"][:N_CAND]]

print(f"{'h':>3} {'耗时(ms)':>10} {'实际rollout数':>14} {'每次(µs)':>10}")
rows = {}
for h in (1, 2, 4, 8, 16):
    payload = dict(state, seed=732, player=0, mode="score_ops", rollouts=ROLLOUTS,
                   horizon=h, candidates=candidates)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        result = tool(payload)
        times.append(time.perf_counter() - t0)
    dt = statistics.median(times)
    total = sum(row["n"] for row in result["candidates"] if row.get("valid"))
    rows[h] = (dt, total)
    print(f"{h:>3} {dt*1000:>10.1f} {total:>14} {dt/total*1e6:>10.1f}")

# 线性拟合（用 h=1 与 h=16 两点，抵消固定开销）
(t1, n1), (t16, n16) = rows[1], rows[16]
per_round_us = (t16 - t1) / (n1 * (16 - 1)) * 1e6
fixed_us = (t1 / n1) * 1e6 - per_round_us * 1
print(f"\n拟合：每次 rollout = {fixed_us:.1f} µs（固定） + {per_round_us:.1f} µs × 回合数")
for h in (1, 2, 4, 8, 16):
    print(f"  h={h:>2} → 预测 {(fixed_us + per_round_us*h):7.1f} µs/rollout"
          f"（实测 {rows[h][0]/rows[h][1]*1e6:7.1f}）")
t8 = fixed_us + per_round_us * 8
print(f"\n深度从 8 砍到 4：时间比 = {(fixed_us + per_round_us*4)/t8:.2f}"
      f"（纯线性会是 0.50）")
print(f"深度从 8 砍到 2：时间比 = {(fixed_us + per_round_us*2)/t8:.2f}")
