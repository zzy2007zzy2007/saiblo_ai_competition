"""补 ab_wide_k160_vs_old 的 result。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "ab_wide_k160_vs_old/output.log"
i = t.find(marker)
if i < 0:
    print("未找到条目")
    raise SystemExit(0)
j = t.find("- **result**: _待填_", i)
new = (
    "- **result**: **广度大胜**：k=160/采样1 = **65.6%**（21W-11L），血量差 +5.19"
    "（95%CI [+1.3,+9.0]）vs 部署口径 k=24/采样16；每局 16 分钟、成本相当"
    "（因为部署口径的 16 次采样里 15 次是重复计算）。"
    "⇒ 「rollout 确定性」的发现直接兑现为 **+15.6pp 内部强度**，且**候选束广度仍未饱和**"
    "（不同于 cells4→cells8 的类内加宽，后者已饱和）。"
)
t = t[:j] + new + t[j + len("- **result**: _待填_"):]
p.write_text(t, encoding="utf-8")
print("已补 ab_wide_k160_vs_old 的 result")
