"""给 train_prior_z 加"CE + z-score"混合损失（2026-10-26）。

动机：
  · 纯 CE（T=0.5）在 logit 尺度小的情况下梯度极小 ⇒ 学不动（旧版 loss 卡 3.46）
  · 纯 z-score/Pearson 保住整体排序，但头部（@1/@4）不一定最优
  ⇒ 混合：L = L_z + λ·L_ce，其中 L_ce 用**软化后的目标**（T_ce 控制）在支撑集上做 CE

用法：--ce-weight 1.0 --ce-temp 0.2
"""
from pathlib import Path

p = Path("code/my_ai/train_prior_z.py")
t = p.read_text(encoding="utf-8")

if "ce_weight" not in t:
    # ① 参数
    t = t.replace('    ap.add_argument("--min-cells", type=int, default=3)',
                  '    ap.add_argument("--min-cells", type=int, default=3)\n'
                  '    ap.add_argument("--ce-weight", type=float, default=0.0, help="CE 项权重（0=纯 z-score）")\n'
                  '    ap.add_argument("--ce-temp", type=float, default=0.2, help="CE 目标温度（越小越接近 one-hot）")')

    # ② 在 groups_loss 里加 CE（需要知道权重/温度 ⇒ 改成闭包参数）
    t = t.replace("def groups_loss(out_row: torch.Tensor, targets: dict, clip: float, min_cells: int) -> list:",
                  "def groups_loss(out_row: torch.Tensor, targets: dict, clip: float, min_cells: int,\n"
                  "                ce_weight: float = 0.0, ce_temp: float = 0.2) -> list:")

    old_tail = """        zv = zscore(v).clamp(-clip, clip)
        losses.append(((zscore(pred) - zv) ** 2).mean())"""
    new_tail = """        zv = zscore(v).clamp(-clip, clip)
        loss = ((zscore(pred) - zv) ** 2).mean()
        if ce_weight > 0.0:
            # CE 项：目标用 softmax(v/T)，预测用标准化后的分数（尺度对齐）
            w = torch.softmax(v / max(ce_temp, 1e-6), dim=0)
            logits = zscore(pred)
            loss = loss + ce_weight * (-(w * torch.log_softmax(logits, dim=0)).sum())
        losses.append(loss)"""
    assert old_tail in t, "groups_loss 尾部锚点未找到"
    t = t.replace(old_tail, new_tail, 1)

    # ③ 调用处传参
    t = t.replace("losses += groups_loss(out[j], targets, args.clip, args.min_cells)",
                  "losses += groups_loss(out[j], targets, args.clip, args.min_cells,\n"
                  "                                         args.ce_weight, args.ce_temp)")

    p.write_text(t, encoding="utf-8")
    print("  已加 --ce-weight / --ce-temp 混合损失")
else:
    print("  已有混合损失")
