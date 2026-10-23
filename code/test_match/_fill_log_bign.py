"""补 ab_random_bign_tree 的 result。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "ab_random_bign_tree/output.log"
i = t.find(marker)
if i < 0:
    print("未找到条目")
    raise SystemExit(0)
j = t.find("- **result**: _待填_", i)
new = (
    "- **result**: **保守随机 + N=64 + 前缀树 = 68.8%（22W-10L），血量差 +5.09（95%CI [+0.2,+10.0]）**"
    " vs 当前冠军。采样曲线：N=8 中性 50.0% → N=32 59.4% → **N=64 68.8%**（随机化改为 C=.3/k=1、开启前缀树）。"
    "代价未增加：平均 46.9 分钟/局（N=32 朴素版是 48.3 分钟/局）⇒ **前缀树把 N 翻倍的成本吃掉了**。"
    "⇒ 用户的判断成立：随机 rollout 需要大采样抹平噪声，而记忆化使其可行。"
)
t = t[:j] + new + t[j + len("- **result**: _待填_"):]
p.write_text(t, encoding="utf-8")
print("已补 ab_random_bign_tree 的 result")
