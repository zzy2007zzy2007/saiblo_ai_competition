"""把 ab_arbiter_net_v3 的 result 补进实验日志（避免 PowerShell 引号问题）。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "- **output**: `training_history/runs/20261021_205250_ab_arbiter_net_v3/output.log`"
if marker not in t:
    print("未找到该条目，跳过")
    raise SystemExit(0)

idx = t.index(marker)
end = t.index("- **result**: _待填_", idx)
new = (
    "- **result**: 前置**可行性**验证（用户口径：只要不显著变差即通过）：把类仲裁的评估器从"
    "「安静 rollout」换成「我们的 rollout + 值网」（8 次 × 32 回合，批量值函数；"
    "实测与逐叶版同分、耗时 0.82→0.33 s）。64 局内部对打 **36W-28L = 56.2%**，"
    "血量差 **+0.92（95%CI [−2.5, +4.4]）** ⇒ **无损害、也无显著增益**。"
    "⇒ 合成两段 rollout 不会破坏类决策，统一路线可行；且**评估器不是瓶颈** ⇒ "
    "统一搜索的优势只能来自广度（把升级等动作也放进候选逐个真测）。"
    "⚠️ 本对照用的是「取最高分候选」，未测「类边际均值」，后者仍待统一搜索自身验证。"
    "详见 `docs/idea_unified_bandit_search.md` §6c。"
)
t = t[:end] + new + t[end + len("- **result**: _待填_"):]
p.write_text(t, encoding="utf-8")
print("已补 ab_arbiter_net_v3 的 result")
