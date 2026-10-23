"""把"训练脚本设计"追加到 docs/prior_repair_20261023.md（下一轮照此实现）。"""
from pathlib import Path

p = Path("docs/prior_repair_20261023.md")
t = p.read_text(encoding="utf-8")
t += """
## 八、训练脚本设计（`code/my_ai/train_prior_distill.py`，下一轮实现）

### 数据形态（`collect_prior_labels.py` 已产出）
`training_history/prior_labels/g<seed>.npz`：
- `feat` (S, 28, 19, 19) float16 —— `FeatureExtractor.encode_board(state, player)`
- `stats` (S, D) float16 —— `encode_stats(state, player)`
- `cand_cls/x/y/val` —— 变长，按 `cand_off`（长度 S+1）切片
- `rounds` (S,)

### 目标（"往哪建"的位置先验）
采集时记录的是 op-type 码（11=BUILD），**不含塔型** ⇒ 先训"**格子**偏好"：
- 对每个局面，把 `cls in 0..15` 的候选按 `(x,y)` 聚合，教师分 = 该格的最大估值；
- 预测 = `action_map[0:16]` 在 `(x,y)` 上的均值（等价于"这里适合建塔"）；
- **损失 = listwise**：`-Σ w_c log softmax(pred/T)_c`，`w = softmax(val/T')`（软标签），`T=1.0, T'=0.5`；
- 只在**有候选的格**上归一化（天然 mask 掉非法格）。

### 冻结与解冻（防遗忘闪电）
- 冻结：`value_head`、`policy_heads`、`stats_mlp`、`initial_conv` 前两层；
- 解冻：`action_map_conv` 的**0..15 行** 与主干后段；
- 学习率 1e-4，Adam，20 epoch，按**留出一局**做早停。

### 学习曲线（用户要求）
用 1/4、1/2、全量数据各训一次，比较留出局上的 **Spearman(预测, 教师)**：
- 若三者单调上升 ⇒ 数据不够，继续加数据；
- 若 1/2 与全量持平 ⇒ 方法/标签到顶，换标签（更深搜索）或换损失。

### 验收（唯一有效口径）
每一版训练产物都跑 **k=24、32 局配对、对照现冠军**，看**血量差**（能分辨 ~5.5 HP）。
Spearman 只作为训练过程的诊断，**不作为晋级依据**（今天已被证伪两次）。
"""
p.write_text(t, encoding="utf-8")
print("已追加训练脚本设计")
