"""给 recall_on_runnerup_v3.py 增加"按动作类型拆分"的统计。"""
from pathlib import Path

p = Path("code/test_match/recall_on_runnerup_v3.py")
t = p.read_text(encoding="utf-8")

if "per_op" not in t:
    t = t.replace(
        "    ks = (1, 4, 24)\n    agg = {k: [0, 0] for k in ks}",
        "    ks = (1, 4, 24)\n    agg = {k: [0, 0] for k in ks}\n    per_op = {}",
    )
    t = t.replace(
        """        for op_t, rank in ranks:
            for k in ks:
                agg[k][0] += int(rank <= k)
                agg[k][1] += 1""",
        """        for op_t, rank in ranks:
            d = per_op.setdefault(op_t, {k: [0, 0] for k in ks})
            for k in ks:
                agg[k][0] += int(rank <= k)
                agg[k][1] += 1
                d[k][0] += int(rank <= k)
                d[k][1] += 1""",
    )
    t = t.replace(
        '''        for k in ks:
            print(f"    recall@{k:<3} = {agg[k][0]}/{agg[k][1]} = {100*agg[k][0]/agg[k][1]:5.1f}%")''',
        '''        for k in ks:
            print(f"    recall@{k:<3} = {agg[k][0]}/{agg[k][1]} = {100*agg[k][0]/agg[k][1]:5.1f}%")
        print("\\n  按动作类型拆分：")
        for op_t in sorted(per_op):
            d = per_op[op_t]
            n = d[24][1]
            miss = n - d[24][0]
            print(f"    op_type={op_t:<3} n={n:<5} @1={100*d[1][0]/n:5.1f}%  @4={100*d[4][0]/n:5.1f}%  "
                  f"@24={100*d[24][0]/n:5.1f}%   落在 top-24 之外 {miss} 个（{100*miss/n:4.1f}%）")''',
    )
    p.write_text(t, encoding="utf-8")
    print("  已加按动作类型拆分")
else:
    print("  已有拆分统计")
