"""从 ladder 日志里统计**每臂真实对局中的出招类型构成**。

动机：我用"空转到某回合"的人造局面探测候选类，那种局面没有塔、金币堆积 ⇒ 只看得到闪电，
**不代表真实对局**。这里直接从 256 局真实对局里聚合 op 类型。
"""
import ast
import collections
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

arm = None
per_arm = collections.defaultdict(collections.Counter)
per_arm_games = collections.Counter()
dict_re = re.compile(r"\{[^{}]*\}")

for line in lines:
    m = re.match(r"## ARM (A\d_\w+)", line.strip())
    if m:
        arm = m.group(1)
        continue
    if arm is None or "az_bridge_ai" not in line:
        continue
    dicts = dict_re.findall(line)
    if not dicts:
        continue
    # 第一个 dict = 我们的臂（ai0 = az_bridge_ai）
    try:
        ours = ast.literal_eval(dicts[0])
    except Exception:
        continue
    if not isinstance(ours, dict):
        continue
    per_arm_games[arm] += 1
    for k, v in ours.items():
        per_arm[arm][k] += int(v)

for arm in sorted(per_arm):
    counter = per_arm[arm]
    total = sum(counter.values())
    if not total:
        continue
    print(f"\n=== {arm}（{per_arm_games[arm]} 局，共 {total} 个 op）===")
    for name, count in counter.most_common():
        print(f"  {name:<22} {count:>6}  {100*count/total:5.1f}%")
