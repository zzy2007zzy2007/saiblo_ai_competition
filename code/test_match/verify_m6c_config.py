"""核实：当前 M6c-ROOT 基线到底用的是什么 env 配置。

从 docs/experiment_log.md 里把所有含 AZAI_CKPT + AZAI_MC_ROOTONLY=1 的 env 行抽出来，
按出现顺序（越靠后越新）列出，并解析成 dict 便于对比。
"""
import re
from pathlib import Path

LOG = Path("docs/experiment_log.md")
text = LOG.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

records = []
for index, line in enumerate(lines):
    if "AZAI_CKPT" not in line or "AZAI_MC_ROOTONLY=1" not in line:
        continue
    # 取 env ... 到 D:/anaconda3 之前的部分
    match = re.search(r"(AZAI_CKPT=[^ ]+(?: [A-Z_]+=\S+)*)", line)
    env_part = match.group(1) if match else ""
    tag = ""
    tag_match = re.search(r"--tag=(\S+)", line)
    if tag_match:
        tag = tag_match.group(1)
    env = dict(item.split("=", 1) for item in env_part.split() if "=" in item)
    records.append((index, tag, env))

print(f"共 {len(records)} 条含 MC_ROOTONLY=1 的配置\n")
print("=== 最后 6 条（越后越新）===")
for index, tag, env in records[-6:]:
    print(f"\n[行 {index+1}] tag={tag}")
    for key in sorted(env):
        print(f"    {key}={env[key]}")

# 统计各字段的不同取值，看哪套是"主流"
print("\n=== 各字段取值分布（全部 %d 条）===" % len(records))
keys = set().union(*[set(env) for _, _, env in records])
for key in sorted(keys):
    values = {}
    for _, _, env in records:
        if key in env:
            values[env[key]] = values.get(env[key], 0) + 1
    top = sorted(values.items(), key=lambda kv: -kv[1])
    if len(top) > 1:
        print(f"  {key}: " + ", ".join(f"{v}×{c}" for v, c in top[:4]))
