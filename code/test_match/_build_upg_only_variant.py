"""把「只有我的改动」的版本拼出来：HEAD 版 + AZAI_UPG_BUNDLES（不含同事未提交的安慰剂改动）。

为什么要这一步：工作区里的 `code/test_match/az_bridge_ai.py` **同时**有
  ① 同事（安慰剂臂）未提交的改动；② 我加的 `AZAI_UPG_BUNDLES`。
我的 commit 只应该带 ②。做法 = 从工作区文件里把**我的 6 个块**按标记切出来，
插到 `git show HEAD:` 的对应锚点上，产出 `/tmp` 下的目标文件；然后
`git hash-object -w` + `git update-index --cacheinfo` 直接把这个 blob 放进 index
（不碰工作区，同事的改动原样留着）。

用法：python code/test_match/_build_upg_only_variant.py <head.py> <worktree.py> <out.py>
"""
from __future__ import annotations

import sys
from pathlib import Path

head_p, wt_p, out_p = (Path(x) for x in sys.argv[1:4])
head = head_p.read_text(encoding="utf-8")
wt = wt_p.read_text(encoding="utf-8")
H: list[str] = head.splitlines(keepends=True)
W: list[str] = wt.splitlines(keepends=True)


def find(lines: list[str], pred, start: int = 0, what: str = "") -> int:
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"❌ 找不到锚点：{what}")


def block(a: int, b: int, what: str) -> list[str]:
    if not (0 <= a < b <= len(W)):
        raise SystemExit(f"❌ 块越界：{what} ({a},{b})")
    return W[a:b]


# ── 从工作区切出我的 6 个块 ──────────────────────────────────────────────────
a = find(W, lambda l: "AZAI_UPG_BUNDLES=<int>" in l, what="docstring 起")
b = find(W, lambda l: "见 `docs/upgrade_bundles_20261028.md`；日志前缀" in l, a, "docstring 止")
B_DOC = block(a, b + 1, "docstring")

a = find(W, lambda l: l.startswith("# ── 升级候选进束（2026-10-28"), what="_UPG 块起")
b = find(W, lambda l: l.startswith("def _inj_tag() -> str:"), a, "_UPG 块止")
B_UPG = block(a, b, "_UPG")                       # 已含尾部空行
while B_UPG and not B_UPG[-1].strip():
    B_UPG.pop()

a = find(W, lambda l: l.startswith("    if _UPG.get(\"n\", 0) > 0:"), what="_report 块起")
B_REP = block(a, a + 3, "_report")

a = find(W, lambda l: l.startswith("    # ── 升级候选进束（2026-10-28；`AZAI_UPG_BUNDLES=N`"), what="main 辅助块起")
b = find(W, lambda l: l.startswith("    def _probe_upgrade_once() -> None:"), a, "main 辅助块止")
B_MAIN = block(a, b, "main 辅助")

a = find(W, lambda l: "⬆️ 升级候选（`AZAI_UPG_BUNDLES=N`" in l, what="decide 前钩子起")
B_STEP = block(a, a + 3, "decide 前钩子")

a = find(W, lambda l: l.startswith("        _upg_after_search(res)"), what="decide 后钩子")
B_AFTER = block(a, a + 1, "decide 后钩子")

a = find(W, lambda l: "出口数一下" in l, what="出口计数起")
B_SENT = block(a, a + 2, "出口计数")

for nm, blk in (("B_DOC", B_DOC), ("B_UPG", B_UPG), ("B_REP", B_REP),
                ("B_MAIN", B_MAIN), ("B_STEP", B_STEP), ("B_AFTER", B_AFTER),
                ("B_SENT", B_SENT)):
    print(f"[build] {nm}: {len(blk)} 行；首行={blk[0].rstrip()[:70]!r}")

# ── 插到 HEAD 的锚点上（全部用"插入"，保持 HEAD 的顺序不动）───────────────────
ins: list[tuple[int, list[str], str]] = []      # (插到第 i 行之前, 内容, 名字)

i = find(H, lambda l: "fold_small_value_ckpt.py" in l, what="docstring 锚")
ins.append((i + 1, B_DOC, "docstring"))

i = find(H, lambda l: l.startswith("def _fmt_act(act) -> str:"), what="_fmt_act 锚")
ins.append((i, B_UPG + ["\n", "\n"], "_UPG+stats"))

i = find(H, lambda l: l.strip() == '_log(f"[oracle] 注入统计: {_oracle_stats_text()}")',
         what="_report 锚")
ins.append((i + 1, B_REP, "_report"))

i = find(H, lambda l: l.startswith("    def _probe_upgrade_once() -> None:"), what="_probe_upgrade_once 锚")
ins.append((i, B_MAIN, "main 辅助"))

i = find(H, lambda l: l.startswith('        with _T("search"):'), what='with _T("search") 锚')
j = i - 1
while not H[j].lstrip().startswith("from my_ai.az_intent.timing import T as _T"):
    j -= 1
ins.append((j, B_STEP, "decide 前钩子"))

i = find(H, lambda l: l.startswith("        _oracle_after_search()   # 注入统计"), what="_oracle_after_search 锚")
ins.append((i + 1, B_AFTER, "decide 后钩子"))

i = find(H, lambda l: l.strip() == "_oracle_log_round(_ops)", what="_oracle_log_round 锚")
ins.append((i + 1, B_SENT, "出口计数"))

out = list(H)
for idx, blk, nm in sorted(ins, key=lambda t: -t[0]):     # 从后往前插，行号不失效
    out[idx:idx] = blk
    print(f"[build] 插入 {nm} @HEAD:{idx+1}  {len(blk)} 行")

Path(out_p).write_text("".join(out), encoding="utf-8")
print(f"[build] 写出 {out_p}（{len(out)} 行）")
