"""`analyze_paired.py` 的自检：用**手工构造的假日志**验证判据统计的每一处口径。

为什么值得写（不是洁癖）：
  * 真正那份读数（B3 基线 128 局）只有一次机会 —— 统计脚本算错了，我们会**带着错数往下走好几轮**；
  * 机械门要求"无上下文的独立验证"：验证者只需 `analyze_paired.py` + 日志。**如果验证者能拿这个
    自检当靶子**（每个期望值都是手算的），复算就变成了"对着已知答案跑"，比"看代码觉得对"强得多；
  * 假日志同时覆盖**异常路径**（INVALID / illegal / verdict≠engine / 缺日志），这些在真数据里
    可能只出现一两次，肉眼容易漏。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/selftest_analyze_paired.py
退出码 0 = 全过；非 0 = 有断言失败（会打印期望 vs 实际）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYZE = REPO / "code" / "test_match" / "analyze_paired.py"
PY = sys.executable

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def op_line(round_no: int, side: int, ops: list[str]) -> str:
    """复刻 bridge --trace-ops 的行格式（`_tmp_ladder.OPS_RE` 就是认这个）。"""
    body = ", ".join(ops)
    return (f"  [round {round_no}] p{side}(x) 原文='{len(ops)}' -> [{body}]")


BT = "Operation(op_type=<OperationType.BUILD_TOWER: 11>, arg0=5, arg1=5)"
LT = "Operation(op_type=<OperationType.USE_LIGHTNING_STORM: 21>, arg0=5, arg1=5)"
UP = "Operation(op_type=<OperationType.UPGRADE_TOWER: 12>, arg0=5, arg1=-1)"


def result_line(seed: int, p0: str, p1: str, winner: str, *, verdict: str = "engine",
                winner_side: str | None = "p0", rounds: int = 100,
                terminal: bool = True, illegal: bool = False) -> str:
    ws = f"winner_side={winner_side} " if winner_side else ""
    out = []
    if illegal:
        out.append(f"  !! [round 5] ILLEGAL p0(x)=[] p1(x)=[op] terminal=False winner=None")
    out.append(f"  RESULT seed={seed} p0={p0} p1={p1} winner={winner} {ws}verdict={verdict} "
               f"engine_winner=0 rounds={rounds} terminal={terminal} base_hp=50,0 coins=9,9")
    return "\n".join(out)


def write_game(root: Path, tag: str, token: str, text: str) -> None:
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{token}.log").write_text(text + "\n", encoding="utf-8")


def run_analyze(root: Path, tag: str, seeds: str, boot: str = "500") -> str:
    cmd = [PY, "-u", str(ANALYZE), f"--tag={tag}", f"--seeds={seeds}",
           f"--logs-root={root}", f"--boot={boot}"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(f"!! analyze 退出码 {p.returncode}\n{p.stdout}\n{p.stderr}")
    return p.stdout


def expect(out: str, needle: str, label: str, failures: list[str]) -> None:
    if needle in out:
        print(f"  ✓ {label}: 命中 `{needle}`")
    else:
        failures.append(f"{label}: 期望包含 `{needle}`")
        print(f"  ✗ {label}: 没找到 `{needle}`")


def main() -> int:
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="selftest_analyze_"))
    try:
        # ================= 用例 1：口径 / 配对分 / 机制读数 =================
        # 手算期望：seed 7 两局 1 胜 1 负 ⇒ sc=0.5；seed 8 两局全胜 ⇒ sc=1.0
        #          配对胜率 p̂ = (0.5+1.0)/2 = 0.75；2-0:1  1-1:1  0-2:0
        #          我方建塔：token7 我方(p0) 2 座；token7r 我方(p1) 0 座；seed8 我方 1+0 座
        #          ⇒ 我方建塔总数 = 3（4 局），对手 = 2 座
        tag = "UT_case1"
        write_game(tmp, tag, "7", op_line(0, 0, [BT, BT]) + "\n" + op_line(0, 1, [LT]) + "\n" +
                   result_line(7, "az_bridge_ai", "main", "az_bridge_ai", winner_side="p0"))
        # 7r：AI1(main) 执先手 ⇒ p0=main p1=az_bridge_ai，这次 main 赢 ⇒ 我方得 0 分
        write_game(tmp, tag, "7r", op_line(0, 0, [LT]) + "\n" + op_line(0, 1, []) + "\n" +
                   result_line(7, "main", "az_bridge_ai", "main", winner_side="p0"))
        write_game(tmp, tag, "8", op_line(0, 0, [BT]) + "\n" + op_line(0, 1, [UP]) + "\n" +
                   result_line(8, "az_bridge_ai", "main", "az_bridge_ai", winner_side="p0"))
        write_game(tmp, tag, "8r", op_line(0, 0, [UP]) + "\n" +
                   op_line(0, 1, [LT]) + "\n" +
                   result_line(8, "main", "az_bridge_ai", "az_bridge_ai", winner_side="p1"))

        out = run_analyze(tmp, tag, "7-8")
        print("--- 用例 1（正常局：口径/配对分/机制）---")
        expect(out, "有效=4", "四局全部有效", failures)
        expect(out, "配对胜率 p̂ = 0.7500", "配对分 = 0.75（手算）", failures)
        expect(out, "2-0: 1  1-1: 1  0-2: 0", "2-0/1-1/0-2 分布（手算）", failures)
        expect(out, "配对胜率 p̂ = 0.7500   95% cluster-bootstrap CI", "主 CI 有输出", failures)
        # 先手侧：seed7 token7（我方执先手）胜、token7r（我方执后手）负；
        #         seed8 token8（先手）胜、token8r（后手）胜 ⇒ 先手胜 2、后手胜 1
        expect(out, "我方执先手 胜: 2", "先手侧统计可用", failures)
        # p̂=0.75 ≥ 0.70 ⇒ 必须打"走机械门"的措辞（**不许自行宣布成功**）
        expect(out, "必须走机械门", "命中 70% 时打到机械门措辞", failures)
        expect(out, "P1（点预测 p̂∈[0.35,0.60]）: 不符合", "P1 自动核对", failures)
        # 机制读数的手算值：我方 BUILD_TOWER = 2(token7,p0) + 0(7r,我方=p1 只有 LT) + 1(token8,p0) + 0(8r,我方=p1) = 3
        # ⚠️ fmt_stats 只保留 1 位小数 ⇒ 0.75 会打成 "0.8"，比较要带容差
        m = re.search(r"\[我方\] n=4\n    建塔 BUILD_TOWER: mean=([\d.]+)", out)
        if m and abs(float(m.group(1)) - 3 / 4) < 0.06:
            print(f"  ✓ 我方建塔均值 ≈ 0.75（手算 3/4；打印保留 1 位小数）")
        else:
            failures.append("我方建塔均值不等于 0.75")
            print(f"  ✗ 我方建塔均值不对：{m.group(1) if m else '未解析到'}")

        # ================= 用例 2：异常路径（INVALID / illegal / 非 engine）=================
        # 期望：4 局里 3 局无效 ⇒ 无效比例 75% > 10% ⇒ 整批标 INVALID；且有效局只剩 1 局
        tag2 = "UT_case2"
        write_game(tmp, tag2, "7", result_line(7, "az_bridge_ai", "main", "INVALID",
                                                verdict="aborted", winner_side=None, terminal=False))
        write_game(tmp, tag2, "7r", op_line(0, 0, [BT]) + "\n" +
                   result_line(7, "main", "az_bridge_ai", "main", winner_side="p0", illegal=True))
        write_game(tmp, tag2, "8", result_line(8, "az_bridge_ai", "main", "az_bridge_ai",
                                               verdict="hp_tiebreak", winner_side=None))
        write_game(tmp, tag2, "8r", result_line(8, "main", "az_bridge_ai", "az_bridge_ai",
                                                winner_side="p1"))
        out2 = run_analyze(tmp, tag2, "7-8")
        print("--- 用例 2（异常路径：INVALID / illegal / hp_tiebreak）---")
        expect(out2, "无效=3", "三局被判无效", failures)
        expect(out2, "有效=1", "只剩 1 局有效", failures)
        expect(out2, "整批标 INVALID", "无效比例 >10% ⇒ 整批作废", failures)
        expect(out2, "本批**不作为基线**", "作废后的行为写清楚", failures)

        # ================= 用例 3：缺日志 =================
        tag3 = "UT_case3"
        write_game(tmp, tag3, "7", result_line(7, "az_bridge_ai", "main", "az_bridge_ai"))
        out3 = run_analyze(tmp, tag3, "7-8")
        print("--- 用例 3（缺日志）---")
        expect(out3, "有效=1", "只有 1 局有日志", failures)
        expect(out3, "缺: ['7r', '8', '8r']", "缺日志被逐个列出", failures)
        expect(out3, "整批标 INVALID", "有缺失即不作基线", failures)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"=== 自检失败 {len(failures)} 项 ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("=== 自检全部通过（口径、配对分、异常路径、缺日志）===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
