"""给 train_prior_z 加 --ce-hard（硬 one-hot 标签 CE，2026-10-27）。

背景：旧实现的软标签 T=0.2 相对分数尺度 σ≈0.02 太大 ⇒ 目标近似均匀分布，
相当于教网络"摊平概率"（实测把建塔 @1 从 14% 拉到 11-12%）。
正确做法：对 argmax 位置做 one-hot CE。
"""
from pathlib import Path

p = Path("code/my_ai/train_prior_z.py")
t = p.read_text(encoding="utf-8")

if "ce_hard" not in t:
    # ① argparse
    t = t.replace('    ap.add_argument("--ce-temp", type=float, default=0.2, help="CE 目标温度（越小越接近 one-hot）")',
                  '    ap.add_argument("--ce-temp", type=float, default=0.2, help="CE 目标温度（越小越接近 one-hot）")\n'
                  '    ap.add_argument("--ce-hard", action="store_true", help="用硬 one-hot（教师 argmax）做 CE")')
    # ② 函数签名
    t = t.replace("def groups_loss(out_row: torch.Tensor, targets: dict, clip: float, min_cells: int,\n"
                  "                ce_weight: float = 0.0, ce_temp: float = 0.2) -> list:",
                  "def groups_loss(out_row: torch.Tensor, targets: dict, clip: float, min_cells: int,\n"
                  "                ce_weight: float = 0.0, ce_temp: float = 0.2,\n"
                  "                ce_hard: bool = False) -> list:")
    # ③ CE 分支
    old = """        if ce_weight > 0.0:
            # CE 项：目标用 softmax(v/T)，预测用标准化后的分数（尺度对齐）
            w = torch.softmax(v / max(ce_temp, 1e-6), dim=0)
            logits = zscore(pred)
            loss = loss + ce_weight * (-(w * torch.log_softmax(logits, dim=0)).sum())"""
    new = """        if ce_weight > 0.0:
            if ce_hard:
                # 硬 one-hot：只监督教师 argmax 那一格
                w = torch.zeros_like(v)
                w[int(torch.argmax(v))] = 1.0
            else:
                # 软标签：目标用 softmax(v/T)
                w = torch.softmax(v / max(ce_temp, 1e-6), dim=0)
            logits = zscore(pred)
            loss = loss + ce_weight * (-(w * torch.log_softmax(logits, dim=0)).sum())"""
    assert old in t, "CE 分支锚点未找到"
    t = t.replace(old, new, 1)
    # ④ 调用处
    t = t.replace("args.ce_weight, args.ce_temp)",
                  "args.ce_weight, args.ce_temp, args.ce_hard)")
    p.write_text(t, encoding="utf-8")
    print("  已加 --ce-hard（硬 one-hot CE）")
else:
    print("  已有 --ce-hard")
