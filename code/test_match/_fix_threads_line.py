"""修 collect_prior_labels.py 被吃掉引号的那一行（AZAI_TORCH_THREADS）。"""
from pathlib import Path

p = Path("code/test_match/collect_prior_labels.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
good = '_TH = int(__import__("os").environ.get("AZAI_TORCH_THREADS", "0"))\n'
fixed = 0
for i, ln in enumerate(lines):
    if ln.startswith("_TH = int("):
        lines[i] = good
        fixed += 1
p.write_text("".join(lines), encoding="utf-8")
print(f"  修复 {fixed} 行")
