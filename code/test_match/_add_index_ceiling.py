"""在 prior_repair 文档末尾加一条索引，指向新的天花板结论文档。"""
from pathlib import Path

p = Path("docs/prior_repair_20261023.md")
t = p.read_text(encoding="utf-8")
if "prior_ceiling_20261026" not in t:
    t += """
## 三十一、先验天花板结论（2026-10-26）

本轮先验相关工作的最终判定已单独成文：**docs/prior_ceiling_20261026.md**。

核心读数：亚军的实际走法有 **92.8%** 落在我们位置先验的前 24 名（@4 = 78.1%、@1 = 46.4%；
1,383 个决策点、16 局重放全部忠实）。

⇒ **先验不是瓶颈**（它能覆盖对手会走的好棋），瓶颈在 **rollout 评估与最终选择**。
⇒ 继续"把先验做得更准"收益有限，应转向评估环节。
⇒ 相关前提、时间线、以及被推翻的中间结论（op_type 映射 bug、z-score 损失、
   80 局数据、长训练过拟合、hybrid 拼接、非传递性假象、座位 27pp 效应）全部记录在那份文档里。
"""
    p.write_text(t, encoding="utf-8")
    print("  已加索引")
else:
    print("  索引已存在")
