"""亚军每回合决策到底多久：用它自己日志里的 elapsed_us 统计（我们机器上的实测）。

输出：每决策耗时的分位数、每次 rollout 的成本、每回合（双方）合计、以及整局搜索总时间。
"""
import glob
import json
import statistics
import sys

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else
                         "match_results/ai0_runnerup_seed7[3-9]*.stderr.log"))[:6]
print(f"[日志] {len(files)} 局")

all_us, per_rollout, per_game, gaps = [], [], [], []
for path in files:
    times, rollouts = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"kind":"decision"' not in line:
            continue
        record = json.loads(line)
        times.append(record["elapsed_us"])
        rollouts.append(record["total_rollouts"])
    if not times:
        continue
    all_us.extend(times)
    total_roll = sum(rollouts)
    total_us = sum(times)
    per_game.append((total_us, total_roll, len(times)))
    for t, r in zip(times, rollouts):
        if r > 0:
            per_rollout.append(t / r)
    ordered = sorted(times)
    gaps.append((ordered[0], ordered[len(ordered) // 2], ordered[-1]))

ordered = sorted(all_us)
def pct(p):
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

print(f"\n=== 每决策耗时（{len(ordered)} 个决策，我们机器上）===")
print(f"  中位 {statistics.median(ordered)/1000:8.1f} ms")
print(f"  均值 {statistics.mean(ordered)/1000:8.1f} ms")
print(f"  p5/p25/p50/p75/p95 = {pct(0.05)/1000:.0f} / {pct(0.25)/1000:.0f} / "
      f"{pct(0.50)/1000:.0f} / {pct(0.75)/1000:.0f} / {pct(0.95)/1000:.0f} ms")
print(f"  最大 {ordered[-1]/1000:.0f} ms")

print(f"\n=== 每次 rollout 的成本 ===")
print(f"  中位 {statistics.median(per_rollout):.1f} µs   均值 {statistics.mean(per_rollout):.1f} µs")

print(f"\n=== 每局（搜索总时间 / rollout 总数 / 决策数）===")
for total_us, total_roll, n in per_game:
    print(f"  {total_us/1e6:6.1f} s   {total_roll:>7} 次   {n} 决策")
print(f"\n每回合（双方各一次决策）= 约 {2 * statistics.median(ordered)/1000:.0f} ms（中位决策 ×2）")
