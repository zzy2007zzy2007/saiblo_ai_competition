"""审计实验日志里的命令，按脚本统计有没有带 --native-engine（用户 2026-09-22 问）。

动机：`--native-engine` 在 eval.py / az_selfplay.py 里都是 **store_true（默认关）**，
所以"漏了 flag"的 run 会跑在 Python 引擎上，而两个引擎是有区别的 ⇒ 读数不可比。
"""
import pathlib
import re
from collections import Counter, defaultdict

lines = pathlib.Path("docs/experiment_log.md").read_text(encoding="utf-8", errors="replace").splitlines()

SCRIPT_RE = re.compile(r"(code[/\\][\w./\\-]+\.py)")
rows: list[tuple[int, str, bool]] = []
for i, ln in enumerate(lines, 1):
    if "python" not in ln:
        continue
    m = SCRIPT_RE.search(ln)
    if not m:
        continue
    rows.append((i, m.group(1).replace("\\", "/").split("/")[-1], "--native-engine" in ln))

print("日志里可识别的脚本命令总数:", len(rows))
tot: Counter = Counter()
nat: Counter = Counter()
for _, s, n in rows:
    tot[s] += 1
    if n:
        nat[s] += 1

print()
print("%-34s %5s %8s %8s" % ("脚本", "总条数", "带flag", "没带"))
for s, t in tot.most_common():
    print("%-34s %5d %8d %8d" % (s, t, nat[s], t - nat[s]))

# 最近 30 条里"没带 flag 但很可能该带"的（我们的核心脚本）
CORE = {"eval.py", "az_selfplay.py", "az_search_vs_search.py", "collect_value_prior.py",
        "az_raw_vs_raw.py", "az_depth0_greedy.py", "rule_v4_lightning_match.py"}
print()
print("=== 最近 30 条命令里，核心脚本但**没带** --native-engine 的 ===")
miss = [(i, s, ln.strip()[:150]) for i, s, n in rows[-30:] if s in CORE and not n]
if miss:
    for i, s, c in miss:
        print("%6d  %-26s %s" % (i, s, c))
else:
    print("（无）")

print()
print("=== 最近的 12 条命令（不限脚本）===")
for i, s, n in rows[-12:]:
    print("%6d  %-26s %s" % (i, s, "NATIVE" if n else "  --  "))
