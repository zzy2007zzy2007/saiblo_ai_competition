"""精确测 C++ 单次 rollout（走 score_ops 模式，避开 mode=rollout 在 209 计划时的栈溢出）。

步骤：① prior 模式拿到亚军自己的合法计划（含 ops）；② 取前 60 个当候选；
      ③ score_ops 用 rollouts=4 / 32 各跑 3 次；④ 用工具返回的 n 求和算斜率。
"""
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
ENTS = Path(r"D:\2026智能体大赛_新2_distill\training_history\distill_top2\abl_linear2_raw\ents_seed00732.jsonl")


def env():
    e = os.environ.copy()
    e["PATH"] = r"C:\mingw64\bin;" + e.get("PATH", "")
    return e


def call(payload: dict) -> dict:
    proc = subprocess.run([str(EXE)], input=(json.dumps(payload) + "\n").encode(),
                          capture_output=True, env=env())
    if proc.returncode != 0:
        print("rc=", proc.returncode, proc.stderr.decode("utf-8", "replace")[:200])
        raise SystemExit(1)
    return json.loads(proc.stdout.decode("utf-8", "replace").splitlines()[0])


state = json.loads(ENTS.open(encoding="utf-8").readlines()[200])
prior = call(dict(state, seed=732, player=0, mode="prior"))
plans = prior["plans"]
print(f"[prior] 拿到 {len(plans)} 个合法计划")

candidates = [[[op["type"], op["arg0"], op["arg1"]] for op in plan["ops"]] for plan in plans[:60]]
times, totals = {}, {}
for n_roll in (4, 32):
    payload = dict(state, seed=732, player=0, mode="score_ops", rollouts=n_roll, candidates=candidates)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = call(payload)
        samples.append(time.perf_counter() - t0)
    valid = [row for row in result["candidates"] if row.get("valid")]
    total = sum(row["n"] for row in valid)
    totals[n_roll] = total
    times[n_roll] = statistics.median(samples)
    print(f"rollouts/候选={n_roll}: 有效候选 {len(valid)}/{len(candidates)}，"
          f"实际 rollout {total}，耗时 {times[n_roll]*1000:.0f} ms")

per_us = (times[32] - times[4]) / (totals[32] - totals[4]) * 1e6
print(f"\n[C++] **单次 8 回合 rollout ≈ {per_us:.1f} µs**")
print(f"      一次决策 14,500 次 ⇒ {per_us * 14500 / 1e6:.2f} s")
