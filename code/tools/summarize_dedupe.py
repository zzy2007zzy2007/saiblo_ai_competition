"""从 dedupe_scan2_result.json 汇总关键结论。"""
import json
import os
from collections import defaultdict

d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "dedupe_scan2_result.json"), encoding="utf-8"))
G = 1024 ** 3
total = d["total"]

print(f"总占用        {total/G:9.2f} GiB   ({total/G/1024:.2f} TiB), {d['n_files']} 个文件")
print(f"所属版本      {', '.join(v.replace('智能体大赛备份','') for v in d['versions'])}")
print()

ext = d["ext"]
code_ext = {".py", ".pyc", ".hpp", ".h", ".cpp", ".c", ".sh", ".cpp", ".cu", ".md", ".txt", ".html", ".csv", ".jsonl"}
code_b = sum(v[0] for e, v in ext.items() if e in {".py", ".hpp", ".h", ".cpp", ".c", ".sh", ".cu"})
doc_b = sum(v[0] for e, v in ext.items() if e in {".md", ".txt", ".html", ".log"})
print(f"== 分层 ==")
print(f"  纯源码 (.py/.cpp/.h/.sh)      {code_b/1024**2:9.1f} MiB  ({code_b/total*100:.4f}%)")
print(f"  文档/日志 (.md/.txt/.html/.log) {doc_b/1024**2:9.1f} MiB")
print(f"  模型权重 (.npz/.pth/.pt/.pkl)  {sum(ext.get(e,(0,0))[0] for e in ('.npz','.pth','.pt','.pkl'))/G:9.2f} GiB")
print(f"  数据 (.json/.zip)              {sum(ext.get(e,(0,0))[0] for e in ('.json','.zip'))/G:9.2f} GiB")
print()

print(f"== 判重结果 ==")
print(f"  大小唯一(必然无法去重)  {d['unique_by_size_bytes']/G:9.2f} GiB")
print(f"  重复上界                {d['dup_upper_bound_bytes']/G:9.2f} GiB")
print(f"  指纹实测确认重复        {d['confirmed_saving_bytes']/G:9.2f} GiB")
print(f"    (仅抽查了 {d['covered_bytes']/G:.2f} GiB 的分组,即 '最优情况' 下的抽样)")
print(f"  副本类目录名合计占用    {d['copy_like_bytes']/G:9.2f} GiB")
print()

print(f"== 重复组 Top 15 (按可省字节) ==")
for g in d["groups"][:15]:
    print(f"  {g['size']/1024**2:8.1f} MiB x{g['n']:<4} 省 {g['save']/G:6.2f} GiB  {g['sample'][:100]}")
