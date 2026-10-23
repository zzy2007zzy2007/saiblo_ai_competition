"""补 ab_search_value_k1 的 result（2026-10-23 关键判据）。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "ab_search_value_k1/output.log"
i = t.find(marker)
if i < 0:
    print("未找到条目")
    raise SystemExit(0)
j = t.find("- **result**: _待填_", i)
new = (
    "- **result**: **搜索价值巨大**：k=160（完整排序）**32W-0L = 100%** 胜 k=1（几乎不搜索），"
    "血量差 **+26.66**（95%CI [+24.7,+28.6]）。⇒ 我们的搜索**绝不是摆设**（相对自身的"
    "\"取第一个候选\"强到 100%）——这与\"打亚军只有 6.2%\"并不矛盾："
    "**搜索有用，但量不够**（每决策约 2,560 模拟回合 vs 亚军 ~114,800，差 45 倍；单回合还贵 10 倍）。"
    "⇒ 路线判定：**应走\"提速换搜索量\"（路线A）**，而不是把搜索蒸馏掉。"
    "附带测量：候选束无重复（k=24/160 去重后仍是 24/160）；k=320 实际只生成 266 个 ⇒ "
    "**该局面的候选空间约 266 个**，解释了 k=160→320 的饱和（53.1%）。"
)
t = t[:j] + new + t[j + len("- **result**: _待填_"):]
p.write_text(t, encoding="utf-8")
print("已补 ab_search_value_k1 的 result")
