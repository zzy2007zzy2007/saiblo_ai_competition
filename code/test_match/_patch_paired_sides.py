"""把 `ab_search_modes.py` 改成支持**同图交换座位（配对）**。

改动（最小、默认关闭以保持向后兼容）：
1. 新增 `--paired-sides`：每个 seed 打 **两局**，A 侧分别坐 P0 / P1（**同一张地图**）。
   ⇒ 座位/地图方差在同一 seed 内相消，SE 大幅下降。
   动机（2026-10-22 实测）：P1 系统性占优 **+19~31pp**（仲裁臂 40.6% vs 71.9%；T1+T2 54.7% vs 73.4%），
   而旧逻辑"seed 偶数 A 打 P0、奇数 A 打 P1"是**不同地图各打一次** ⇒ 座位方差全进噪声。
2. 任务列表由 `(seed,)` 变成 `(seed, flat_side)`；日志名在配对模式下带 `_p0/_p1`。
3. 汇总里额外打印 **先/后手拆分**（P0 胜率 / P1 胜率），以后每条 A/B 都自带座位诊断。

用法：python ab_search_modes.py --paired-sides --seeds 500 501 ...
"""
from __future__ import annotations

from pathlib import Path

p = Path(__file__).resolve().parent / "ab_search_modes.py"
t = p.read_text(encoding="utf-8")

# ── ① 新增命令行开关 ──────────────────────────────────────────────────────
old_arg = '    ap.add_argument("--timeout", type=float, default=4 * 3600.0)'
new_arg = ('    ap.add_argument("--timeout", type=float, default=4 * 3600.0)\n'
           '    # 2026-10-22：同图交换座位（配对）。实测 P1 系统性占优 +19~31pp，\n'
           '    # 旧逻辑是"不同地图各打一次" ⇒ 座位方差全进噪声。\n'
           '    ap.add_argument("--paired-sides", action="store_true",\n'
           '                    help="每个 seed 打两局，A 侧分别坐 P0/P1（同一张地图）")')
assert old_arg in t
t = t.replace(old_arg, new_arg, 1)

# ── ② run(seed) → run(seed, flat_side) ───────────────────────────────────
old_sig = '''    def run(seed: int) -> None:
        # seed 偶数 ⇒ flat 执 P0；奇数 ⇒ flat 执 P1
        flat_is_p0 = (seed % 2 == 0)'''
new_sig = '''    def run(seed: int, flat_side_in: int | None = None) -> None:
        # 默认沿用旧约定（seed 偶数 ⇒ flat 执 P0）；配对模式下由调用方显式指定
        flat_is_p0 = (seed % 2 == 0) if flat_side_in is None else (flat_side_in == 0)'''
assert old_sig in t
t = t.replace(old_sig, new_sig, 1)

# ── ③ 日志名带座位（配对模式下）───────────────────────────────────────────
old_log = '        log_path = out_dir / f"seed{seed}.log"'
new_log = ('        log_path = (out_dir / (f"seed{seed}.log" if flat_side_in is None\n'
           '                                  else f"seed{seed}_p{0 if flat_is_p0 else 1}.log"))')
assert old_log in t
t = t.replace(old_log, new_log, 1)

# ── ④ results 的键 + 打印里带上座位 ─────────────────────────────────────
t = t.replace('''            results[seed] = {"flat_is_p0": flat_is_p0, "score": flat_score,''',
              '''            key = seed if flat_side_in is None else (seed, 0 if flat_is_p0 else 1)
            results[key] = {"flat_is_p0": flat_is_p0, "score": flat_score,''', 1)
t = t.replace('''            print(f"[seed {seed}] flat={'P0' if flat_is_p0 else 'P1'} "''',
              '''            print(f"[seed {seed}] flat={'P0' if flat_is_p0 else 'P1'} "''', 1)

# ── ⑤ 任务列表 + 汇总（含 P0/P1 拆分）────────────────────────────────────
old_tasks = '''    threads = [threading.Thread(target=run, args=(s,)) for s in args.seeds]'''
new_tasks = '''    if args.paired_sides:
        tasks = [(s, side) for s in args.seeds for side in (0, 1)]
    else:
        tasks = [(s, None) for s in args.seeds]
    threads = [threading.Thread(target=run, args=t) for t in tasks]'''
assert old_tasks in t
t = t.replace(old_tasks, new_tasks, 1)

old_sum = '''    scores = [results[s]["score"] for s in args.seeds if s in results]'''
new_sum = '''    scores = [v["score"] for v in results.values()]'''
assert old_sum in t
t = t.replace(old_sum, new_sum, 1)

old_tail = '''              f"±{2*se:.1%}(2SE)  wall={(time.time()-t0)/60:.1f} min ===", flush=True)'''
new_tail = '''              f"±{2*se:.1%}(2SE)  wall={(time.time()-t0)/60:.1f} min ===", flush=True)
        # 2026-10-22：先/后手拆分（座位诊断）。P1 系统性占优时，两条臂若不配对就会互相污染。
        for _side, _name in ((0, "P0"), (1, "P1")):
            _s = [v["score"] for v in results.values() if v["flat_is_p0"] == (_side == 0)]
            if _s:
                _w = sum(1 for x in _s if x == 1.0)
                print(f"    A 在 {_name}: {_w}W-{len(_s)-_w}L = {sum(_s)/len(_s):.1%} (n={len(_s)})",
                      flush=True)'''
assert old_tail in t
t = t.replace(old_tail, new_tail, 1)

p.write_text(t, encoding="utf-8")
print("ab_search_modes.py 已支持 --paired-sides（同图交换座位）")
