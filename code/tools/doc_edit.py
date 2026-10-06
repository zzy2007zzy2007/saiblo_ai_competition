"""文档改写的小工具：**带"身份校验"的安全替换**（2026-10-06 事故后新增）。

事故（我的错）：一个一次性 patch 脚本里变量名用混了 —— `p` 是 `docs/goal_register.md`、`d` 是
`docs/engine_crash_probe.md`，最后写成 `p.write_text(td)` ⇒ **把 probe 报告的内容写进了结论寄存器**，
并随 commit 提交（幸而用户发现并在下一笔修回）。这类错误**不会报错**，只会静默毁文件。

因此以后改文档一律走本工具：
  1. **身份校验**：先断言"目标文件的首行 == 期望首行"（防止写错文件）；
  2. **锚点校验**：断言被替换的旧串**恰好出现一次**（防止改错位置/漏改）；
  3. 只有两条都过才写盘；写盘后**回读并断言新串已在文件里**。

用法：
    from code.tools.doc_edit import edit_doc
    edit_doc("docs/goal_register.md", "# 结论寄存器（跨会话防绕圈用）", "旧串", "新串")
命令行自检：
    python -u code/tools/doc_edit.py            # 对关键文档做一次"首行身份校验"扫描
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# 关键文档的"身份锚"（首行）。改这些文件前必须先过这一关。
ANCHORS = {
    "docs/goal_register.md": "# 结论寄存器（跨会话防绕圈用）",
    "docs/experiment_log.md": None,       # 首行是自动生成的标题（含时间戳），用前缀匹配
    "docs/engine_crash_probe.md": "# 诊断报告：",
    "docs/task_ruleV4_70_charter.md": "# 任务章程",   # 第一里程碑版（历史）
    "docs/task_ruleV4_90_charter.md": "# 任务章程",   # 第二里程碑版（当前）
    "docs/goal_longrun_plan.md": "#",
    "docs/milestone_20261006_70.md": "# 里程碑报告",
    "docs/state_of_play_20261006.md": "# 状态报告",
}


def _first_line(path: Path) -> str:
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.strip():
            return ln.rstrip()
    return ""


def check_identity(rel: str) -> tuple[bool, str, str]:
    """→ (是否通过, 首行, 期望锚)。期望锚为 None 时只要求"非空且像 markdown 标题"。"""
    path = REPO / rel
    if not path.exists():
        return False, "<文件不存在>", str(ANCHORS.get(rel))
    first = _first_line(path)
    want = ANCHORS.get(rel, None)
    if want is None:
        return first.startswith("#"), first, "<任意 # 开头>"
    # 一律用**前缀**匹配：锚点只用来确认"打开的是哪个文件"，不要求整行全等
    # （2026-10-06：我最初写成全等，结果把 4 个正确文件判成 FAIL —— 锚写短了就误报）
    return first.startswith(want), first, want + "…（前缀匹配）"


def edit_doc(rel: str, expect_first_line: str, old: str, new: str, *, count: int = 1) -> None:
    path = REPO / rel
    first = _first_line(path)
    assert first == expect_first_line, (
        f"身份校验失败：{rel} 的首行是 {first!r}，期望 {expect_first_line!r} —— 拒绝写入（防写错文件）")
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    assert n == count, f"锚点校验失败：旧串在 {rel} 里出现 {n} 次（期望 {count}）—— 拒绝写入"
    path.write_text(text.replace(old, new), encoding="utf-8")
    back = path.read_text(encoding="utf-8")
    assert new in back, f"回读校验失败：{rel} 里没找到新串"
    print(f"[doc_edit] OK: {rel}（首行校验 + 锚点 {n} 次 + 回读）", flush=True)


def main() -> int:
    ok = True
    print("=== 关键文档首行身份校验 ===", flush=True)
    for rel in ANCHORS:
        good, first, want = check_identity(rel)
        flag = "PASS" if good else ("SKIP" if not (REPO / rel).exists() else "FAIL")
        if flag == "FAIL":
            ok = False
        print(f"[{flag}] {rel}\n        首行: {first[:100]}\n        期望: {want}", flush=True)
    print("=== 全部通过 ===" if ok else "=== 有 FAIL ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
