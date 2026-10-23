"""补 ab_k320_vs_k160 的 result。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "ab_k320_vs_k160/output.log"
i = t.find(marker)
if i < 0:
    print("未找到条目")
    raise SystemExit(0)
j = t.find("- **result**: _待填_", i)
new = (
    "- **result**: **广度 160→320 基本饱和**：17W-15L = **53.1%**，血量差 +1.00"
    "（95%CI [-3.3,+5.3]，跨 0）。结合 k=24→160 的 65.6%（CI 不含 0）⇒ "
    "**广度的边际收益在 160 附近收敛**；再往上（320）没有再买到可测的强度，"
    "瓶颈应从『候选数量』转向『候选排序质量（位置网先验）』。"
)
t = t[:j] + new + t[j + len("- **result**: _待填_"):]
p.write_text(t, encoding="utf-8")
print("已补 ab_k320_vs_k160 的 result")
