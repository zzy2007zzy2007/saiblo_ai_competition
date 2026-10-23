"""补 ab_decay_s32 的 result。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "ab_decay_s32/output.log"
i = t.find(marker)
if i < 0:
    print("未找到条目")
    raise SystemExit(0)
j = t.find("- **result**: _待填_", i)
new = (
    "- **result**: **采样 32 让随机 rollout 从\"中性\"变成\"正向\"**：19W-13L = **59.4%**，"
    "血量差 +3.69（95%CI [-1.0,+8.4]，边缘显著）。对比同配置采样 8 = **50.0%**（CI[-5.3,+5.4]，完全中性）"
    " ⇒ **+9.4pp**。⇒ 之前\"随机性无信号\"的判断是错的：**是采样不足让噪声淹没了信号**（用户判断正确）。"
    "代价：平均 48.3 分钟/局（每决策 5,120 次 rollout）。"
    "配套已实现**前缀树记忆化**（`AZAI_FLAT_SHARE_PREFIX=1`）：分数与朴素版逐位一致，"
    "高共享配置（decay C=.3/k=1，共享 85%）下 N=32 省 38%，低共享（43%）反而慢 12-18%。"
)
t = t[:j] + new + t[j + len("- **result**: _待填_"):]
p.write_text(t, encoding="utf-8")
print("已补 ab_decay_s32 的 result")
