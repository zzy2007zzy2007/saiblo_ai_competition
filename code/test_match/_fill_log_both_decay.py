"""补 both+decay 的 result，并记录随机 rollout 的完整结论。"""
from pathlib import Path

p = Path("docs/experiment_log.md")
t = p.read_text(encoding="utf-8", errors="replace")
marker = "ab_rollout_decay_both_v2/output.log"
i = t.find(marker)
new = (
    "- **result**: **双方随机+衰减 = 31.2%（10W-22L），血量差 -6.88（95%CI [-11.3,-2.4]）⇒ 显著更差**。"
    "结合同批次读数：恒定随机(双方) 34.4%（CI[-12.3,-2.6]）、**只随机对手+衰减 50.0%（CI[-5.3,+5.4]，中性）**"
    " ⇒ **随机 rollout 路线彻底关闭**：只要我方参与随机就显著变差（破坏候选动作的价值评估），"
    "只随机对手则等于没做。根因：我们的随机源是"建塔/拆塔"这种语义粗糙的动作，"
    "而亚军随机的是**敌方蚂蚁的移动/传送**（`move_phase(rng)`/`teleport_phase(rng)`），有物理意义。"
)
if i < 0:
    print("未找到 both+decay 条目")
else:
    j = t.find("- **result**: _待填_", i)
    t = t[:j] + new + t[j + len("- **result**: _待填_"):]
    p.write_text(t, encoding="utf-8")
    print("已补 both+decay 的 result")
