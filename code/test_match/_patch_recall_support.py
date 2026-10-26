"""给 recall 脚本加"支撑集大小"统计：避免把小分母的平凡读数当成好成绩。"""
from pathlib import Path

p = Path("code/test_match/recall_on_runnerup_v3.py")
t = p.read_text(encoding="utf-8")

if "SUP" not in t:
    # ① 记录支撑集大小
    t = t.replace(
        "                    if 0 <= cx < 19 and 0 <= cy < 19 and mask[cx, cy]:",
        "                    SUP = int(mask.sum())\n                    if 0 <= cx < 19 and 0 <= cy < 19 and mask[cx, cy]:",
    )
    t = t.replace(
        "                        ranks.append((op[0], rank))",
        "                        ranks.append((op[0], rank, SUP))",
    )
    t = t.replace(
        "        for op_t, rank in ranks:",
        "        for op_t, rank, sup in ranks:",
    )
    t = t.replace(
        "            d = per_op.setdefault(op_t, {k: [0, 0] for k in ks})",
        "            d = per_op.setdefault(op_t, {k: [0, 0] for k in ks})\n            d.setdefault(\"sup\", []).append(sup)",
    )
    # ② 输出支撑集统计
    t = t.replace(
        '''            print(f"    op_type={op_t:<3} n={n:<5} @1={100*d[1][0]/n:5.1f}%  @4={100*d[4][0]/n:5.1f}%  "
                  f"@24={100*d[24][0]/n:5.1f}%   落在 top-24 之外 {miss} 个（{100*miss/n:4.1f}%）")''',
        '''            sup = d.get("sup") or [0]
            sup_sorted = sorted(sup)
            med = sup_sorted[len(sup_sorted)//2]
            mean = sum(sup)/len(sup)
            base1 = sum(1.0/max(1, s) for s in sup)/len(sup)
            print(f"    op_type={op_t:<3} n={n:<5} 支撑集 均值{mean:5.1f}/中位{med:<4} "
                  f"@1={100*d[1][0]/n:5.1f}%(随机基线{100*base1:5.1f}%)  "
                  f"@4={100*d[4][0]/n:5.1f}%  @24={100*d[24][0]/n:5.1f}%")
            print(f"                 支撑集 <4 的比例 {100*sum(1 for s in sup if s < 4)/len(sup):5.1f}%   "
                  f"<24 的比例 {100*sum(1 for s in sup if s < 24)/len(sup):5.1f}%")''',
    )
    p.write_text(t, encoding="utf-8")
    print("  已加支撑集统计")
else:
    print("  已有支撑集统计")
