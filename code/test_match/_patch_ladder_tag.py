"""修 _tmp_ladder.py：status 代码误用 main() 的局部变量 cfg ⇒ NameError ⇒ 整批退出。

另外给汇总循环加"日志存在"保护（没有日志时不要抛 FileNotFoundError）。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "_tmp_ladder.py"
t = P.read_text(encoding="utf-8")

# ① 模块级 tag，run_one 里用它
if "_TAG = \"ladder\"" not in t:
    t = t.replace("_TOKENS: list = []",
                  "_TAG = \"ladder\"        # 2026-10-23：当前 tag（run_one 里写 status 用）\n_TOKENS: list = []", 1)
t = t.replace('    env["DSH_PROGRESS_TAG"] = str(cfg.get("tag", "ladder"))',
              '    env["DSH_PROGRESS_TAG"] = str(_TAG)', 1)
t = t.replace('                _tag = str(cfg.get("tag", "ladder"))', '                _tag = str(_TAG)', 1)

# ② main 里设置 _TAG
old = "    global _T0\n    _TOKENS.clear(); _TOKENS.append(list(tokens))"
new = ("    global _T0, _TAG\n    _TAG = str(cfg.get(\"tag\", \"ladder\"))\n"
       "    _TOKENS.clear(); _TOKENS.append(list(tokens))")
assert old in t, "② 锚点未命中"
t = t.replace(old, new, 1)

# ③ 汇总循环保护：日志不存在就跳过
old2 = "        g = read_game(out_dir / f\"{t}.log\", t)"
new2 = ("        if not (out_dir / f\"{t}.log\").exists():\n"
        "            print(f\"[ladder] ⚠️ 缺日志 {t}.log（该局未跑/被杀），跳过汇总\", flush=True)\n"
        "            continue\n"
        "        g = read_game(out_dir / f\"{t}.log\", t)")
assert old2 in t, "③ 锚点未命中"
t = t.replace(old2, new2, 1)

P.write_text(t, encoding="utf-8")
print("已修：_TAG 模块级 + 汇总保护")
