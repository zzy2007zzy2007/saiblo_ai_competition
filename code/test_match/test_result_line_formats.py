"""`RESULT` 行的**格式回归测试**（2026-10-06 入库）。

为什么需要它（用户 2026-10-06 指出：这类测试当时只是临时脚本，没入库）：
* `_tmp_ladder.RESULT_RE` 要求行内**必须有 `engine_winner=`**，且 `engine_winner=… rounds=` 必须**相邻**；
* 而 bridge 打 INVALID 行时曾经：① **不打印 `engine_winner=`** ⇒ 正则**静默失配** ⇒ 无效/超时局从未被解析
  （分析器只能显示「winner 缺失 / rounds=0」，把「第 55 回合超时」埋掉）；
  ② 把 `exc=` 塞在 `engine_winner=` 与 `rounds=` 之间 ⇒ **再次失配**。
  ⇒ 这两个坑都不是"运行时报错"，而是**静默**的，所以必须有**入库的**回归测试守住。

覆盖：2 条**必须匹配**的正例（正常局 / 超时局）+ 2 条**必须不匹配**的反例（历史上真实踩过的两种写法）
+ 1 条 `read_game` 端到端解析。

用法（失败返回码 1）：
    python -u code/test_match/test_result_line_formats.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _load_ladder():
    spec = importlib.util.spec_from_file_location("_tmp_ladder_mod", REPO / "_tmp_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tmp_ladder_mod"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---- 用例： (名字, 行内容, 期望匹配, 期望解析出的 (verdict, rounds, winner)) ----
CASES = [
    (
        "正常局（engine）",
        "  RESULT seed=11 p0=az_bridge_ai p1=main winner=az_bridge_ai winner_side=p0 "
        "verdict=engine engine_winner=0 rounds=413 terminal=True base_hp=8,0 coins=1,2",
        True, ("engine", "413", "az_bridge_ai"),
    ),
    (
        "超时局（verdict=timeout，exc= 在行尾 —— 现在 bridge 的写法）",
        "  RESULT seed=11 p0=az_bridge_ai p1=main winner=INVALID winner_side=None "
        "verdict=timeout engine_winner=None rounds=55 terminal=False base_hp=46,47 coins=0,0 exc=TimeoutError",
        True, ("timeout", "55", "INVALID"),
    ),
    (
        "反例①：INVALID 行**缺** engine_winner=（2026-10-06 之前的真实写法）",
        "  RESULT seed=11 p0=az_bridge_ai p1=main winner=INVALID verdict=aborted exc=str "
        "rounds=55 terminal=False base_hp=46,47 coins=0,0",
        False, None,
    ),
    (
        "反例②：exc= 夹在 engine_winner= 与 rounds= 之间（我第一版改法）",
        "  RESULT seed=11 p0=az_bridge_ai p1=main winner=INVALID winner_side=None verdict=timeout "
        "engine_winner=None exc=TimeoutError rounds=55 terminal=False base_hp=46,47 coins=0,0",
        False, None,
    ),
]


def main() -> int:
    mod = _load_ladder()
    ok = True
    for name, line, expect_match, expect_fields in CASES:
        m = mod.RESULT_RE.search(line)
        got_match = bool(m)
        status = "PASS" if got_match == expect_match else "FAIL"
        if got_match != expect_match:
            ok = False
        extra = ""
        if got_match and expect_fields:
            got_fields = (m.group(6), m.group(8), m.group(4))
            if got_fields != expect_fields:
                status, ok = "FAIL", False
                extra = f"  字段不符: got={got_fields} want={expect_fields}"
            else:
                extra = f"  verdict={got_fields[0]} rounds={got_fields[1]} winner={got_fields[2]}"
        print(f"[{status}] 正则: {name}  (匹配={got_match}, 期望={expect_match}){extra}", flush=True)

    # ---- 端到端：把两条正例写进临时日志，交给真正的 read_game 解析 ----
    tmp = REPO / "training_history" / "vprior" / "_regress_result_line.log"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, line, expect_match, expect_fields in CASES[:2]:
            assert expect_fields is not None
            tmp.write_text(line + "\n", encoding="utf-8")
            g = mod.read_game(tmp, "11")
            got = (g["verdict"], g["rounds"], g["winner"])
            status = "PASS" if got == expect_fields else "FAIL"
            if got != expect_fields:
                ok = False
            print(f"[{status}] read_game: {name}  -> {got}", flush=True)
    finally:
        tmp.unlink(missing_ok=True)

    print("=== 全部通过 ===" if ok else "=== 有 FAIL ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
