"""一致性体检：**与 AI 的实现方式无关**的"它到底在下棋吗"检测。

为什么需要它（血的教训，见 memory/project_champion_runnerup_cpp_match.md）：
2026-08-09 两个 C++ AI 在 `main.exe` 裁判下**全程静默弃权**，而"对局跑完、有赢家"完全看不出来；
我一度又用 8 回合上限得出"两个 AI 全弃权"的**错误**结论（实际是前期攒钱的正常策略）。
所以新 AI 接进来的第一道门必须是：**硬失败能判死，软现象只报告、不下结论。**

用法:
    python code/test_match/conformance.py --ai0=<你的程序> [--ai0-args=..] [--rounds=30] [--seed=7]

判据（跑之前写死）:
  HARD FAIL ① 输出里出现 `EOFError: <你的AI> exited` ⇒ AI 在对局结束前就退出了
  HARD FAIL ② `ILLEGAL` 行数 > 0 ⇒ 你的 AI 发了裁判不接受的非法操作
  HARD FAIL ③ 跑够 `--paralysis-rounds`（默认 250）后，对手有操作而**你一次都没出招** ⇒ 疑似弃权
  WARN      ④ 判词来源是 `hp_tiebreak` ⇒ 对局被截断，胜负不是官方规则算的（短跑必然如此）
  WARN      ⑤ 你的 AI 完全没收到过局面（stderr 里若自报则看它）

⚠️ 短跑（默认 30 回合）**不能**判定弃权：rule_v4 前 ~30 回合本来就在攒钱空过。
   要下"弃权"的结论必须 `--rounds>=250`（默认 §判据③ 的阈值就是这么来的）。
"""
from __future__ import annotations

import collections
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OPPONENT = REPO / "code" / "test_match" / "rv4_pkg" / "main.py"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) verdict=(\S+) engine_winner=(\S+) "
    r"rounds=(\S+) terminal=(\S+) base_hp=(\d+),(\d+) coins=(\d+),(\d+)")
OPS_RE = re.compile(r"\[round (\d+)\] p(\d)\((\w+)\) .*-> \[(.*)\]$")


def main() -> int:
    ai0 = None
    ai0_args: tuple[str, ...] = ()
    opponent = DEFAULT_OPPONENT
    rounds = 30
    seed = 7
    paralysis_rounds = 250
    for a in sys.argv[1:]:
        if a.startswith("--ai0="):
            ai0 = Path(a.split("=", 1)[1]).resolve()
        elif a.startswith("--ai0-args="):
            ai0_args = tuple(shlex.split(a.split("=", 1)[1]))
        elif a.startswith("--opponent="):
            opponent = Path(a.split("=", 1)[1]).resolve()
        elif a.startswith("--rounds="):
            rounds = int(a.split("=", 1)[1])
        elif a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
        elif a.startswith("--paralysis-rounds="):
            paralysis_rounds = int(a.split("=", 1)[1])
    if ai0 is None:
        print("用法: python code/test_match/conformance.py --ai0=<你的程序> [--ai0-args=..] "
              "[--opponent=..] [--rounds=30] [--seed=7]")
        return 2

    out_dir = REPO / "match_results" / f"conformance_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "conformance.log"

    cmd = [sys.executable, "-u", "code/test_match/match_via_sdk.py", "--trace-ops",
           f"--rounds={rounds}", f"--ai0={ai0}",
           f"--ai0-args={shlex.join(ai0_args)}" if ai0_args else "",
           f"--ai1={opponent}", str(seed)]
    cmd = [c for c in cmd if c]
    print(f"[conformance] 被测 AI = {ai0}  args={list(ai0_args)}")
    print(f"[conformance] 对手     = {opponent}")
    print(f"[conformance] 回合上限 = {rounds}（<{paralysis_rounds} 时无法判定弃权）")
    print(f"[conformance] 日志     = {log}")

    t0 = time.time()
    with open(log, "wb") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO),
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    dt = time.time() - t0
    txt = log.read_text(encoding="utf-8", errors="replace")

    # ---- 解析 ----
    ops = collections.Counter()
    act = collections.Counter()
    hist: dict[str, collections.Counter] = {"p0": collections.Counter(), "p1": collections.Counter()}
    ill = 0
    early_exit = []
    result = None
    for ln in txt.splitlines():
        if "ILLEGAL" in ln:
            ill += 1
        m = re.search(r"EOFError: (\S+) exited", ln)
        if m:
            early_exit.append(m.group(1))
        m = RESULT_RE.search(ln)
        if m:
            result = {"p0": m.group(2), "p1": m.group(3), "winner": m.group(4),
                      "verdict": m.group(5), "rounds": m.group(7)}
        m = OPS_RE.search(ln)
        if m:
            who, body = "p" + m.group(2), m.group(4)
            n = body.count("Operation(")
            ops[who] += n
            if n:
                act[who] += 1
            for t in re.findall(r"<OperationType\.(\w+):", body):
                hist[who][t] += 1

    # 谁是被测 AI：本工具不交换先手 ⇒ AI0 就是 p0，用标签核对一下而已
    which = "p0"
    if result:
        which = "p0" if result["p0"] == _label_of_path(ai0) else "p1"
    other = "p1" if which == "p0" else "p0"

    print(f"\n=== 原始读数（{dt:.0f}s, bridge rc={rc}）===")
    print(f"  判词:   {result}")
    print(f"  非法:   {ill}")
    print(f"  被测 AI({which}): 出招回合 {act[which]}  操作总数 {ops[which]}  {dict(hist[which].most_common())}")
    print(f"  对手   ({other}): 出招回合 {act[other]}  操作总数 {ops[other]}  {dict(hist[other].most_common())}")

    # ---- 判据 ----
    fails: list[str] = []
    warns: list[str] = []
    if early_exit:
        fails.append(f"① 被测/对手进程在对局结束前退出: {early_exit}")
    if ill > 0:
        fails.append(f"② 出现 {ill} 处非法操作（裁判不接受你的操作）")
    ran = int(result["rounds"]) if result and str(result["rounds"]).isdigit() else 0
    if ran >= paralysis_rounds and ops[other] > 0 and ops[which] == 0:
        fails.append(f"③ 跑了 {ran} 回合，对手出了 {ops[other]} 个操作而你一次都没出 ⇒ 疑似弃权")
    if result and result["verdict"] == "hp_tiebreak":
        warns.append("④ 判词是 hp_tiebreak（对局被 --rounds 截断，胜负不是官方规则算的）")
    if not early_exit and ops[which] == 0 and ran < paralysis_rounds:
        warns.append(f"⑤ 本次只跑了 {ran} 回合，你 0 出招 —— **这不能判定弃权**"
                     f"（rule_v4 前 ~30 回合本来就在攒钱）；要判定请 --rounds>={paralysis_rounds}")

    print("\n=== 结论 ===")
    for w in warns:
        print(f"  WARN {w}")
    if fails:
        for x in fails:
            print(f"  FAIL {x}")
        print("  => 这个 AI 还没法正常接入，先修上面这些。")
        return 1
    print("  PASS 没有硬失败：进程没提前退出、没有非法操作"
          + ("、出招正常" if ops[which] else "（但短跑下'没出招'不算结论）"))
    return 0


def _label_of_path(p: Path) -> str:
    """和 match_via_sdk._label_of 同样的规则（冠军/亚军保持原名，其他用 stem）。"""
    name = p.name.lower()
    if "magica" in name:
        return "champion"
    if "ai_cpp_lure" in name:
        return "runnerup"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in p.stem)


if __name__ == "__main__":
    raise SystemExit(main())
