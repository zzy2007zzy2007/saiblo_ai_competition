"""统计：**每条决策的候选动作数分布**（直接回答"候选 >3 个的决策有多少条"）。

数据：`AZAI_AZ_DUMP` 的 policy JSONL（每行一条决策，含 `cands`（候选束列表）、`labels`、`visits_raw`（真访问计数）、`chosen`）。

报告：
  * `len(cands)` 的直方图（= 搜索的候选**束**数；理论上限 = `K`=24）；
  * **>3 个候选**、**>5**、**≥2** 的条数与占比；
  * 其中"**有效选择**"= `visits_raw` 里**非零**的个数（部署 `TEMP=1e-6` 取 argmax ⇒ 名次实验只在 >1 时才有意义 ✓）；
  * 候选束覆盖的**不同格数**（位置维的真实选择空间 ✓）。

用法：python -u code/my_ai/az_intent/diag_candidate_counts.py --policy-dir data/seedstate_dump/az
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", default="data/seedstate_dump/az")
    ap.add_argument("--max-files", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.policy_dir) / "az_*.jsonl")))
    if args.max_files:
        files = files[: args.max_files]
    ncand_hist = collections.Counter()
    nz_hist = collections.Counter()
    cell_hist = collections.Counter()
    n = 0
    # 候选束数门槛（用户 2026-10-12 要求：≥2 / ≥3 / ≥5 都统计 ✓）
    n_ge2 = n_ge3 = n_ge5 = 0
    # 「有效选择」= `visits_raw` 非零个数（部署取 argmax ⇒ 只有 >1 时名次/分布类目标才有意义 ✓）
    z_ge2 = z_ge3 = z_ge5 = 0
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            cands = d.get("cands")
            if not isinstance(cands, list):
                continue
            n += 1
            nc = len(cands)
            ncand_hist[nc] += 1
            if nc >= 2:
                n_ge2 += 1
            if nc >= 3:
                n_ge3 += 1
            if nc >= 5:
                n_ge5 += 1
            # 非零访问数（有效选择）
            vr = d.get("visits_raw")
            nz = 0
            if isinstance(vr, list):
                nz = sum(1 for x in vr if float(x) > 0)
            nz_hist[nz] += 1
            if nz >= 2:
                z_ge2 += 1
            if nz >= 3:
                z_ge3 += 1
            if nz >= 5:
                z_ge5 += 1
            # 不同格数
            labels = d.get("labels") or []
            cells = {(int(op[1]), int(op[2])) for b in labels for op in b if len(op) >= 3}
            cell_hist[len(cells)] += 1

    def pct(x):
        return f"{x}（{x / max(n, 1) * 100:.1f}%）"

    print(f"[cand] 文件 {len(files)} 个；决策 {n} 条")
    print(f"  候选束数直方图（前 10）：{sorted(ncand_hist.items())[:10]}")
    print(f"  **候选束数 ≥2**：{pct(n_ge2)}；**≥3**：{pct(n_ge3)}；**≥5**：{pct(n_ge5)}")
    print(f"  非零访问数直方图（前 10）：{sorted(nz_hist.items())[:10]}")
    print(f"  **有效选择（非零访问）≥2**：{pct(z_ge2)}；**≥3**：{pct(z_ge3)}；**≥5**：{pct(z_ge5)}"
          f"   ← 名次/分布类目标真正起作用的决策面 ✓")
    print(f"  候选覆盖的不同格数直方图（前 10）：{sorted(cell_hist.items())[:10]}")
    print("  判读：若『有效选择 ≥2』占比很低 ⇒ 名次/分布类目标的可行决策面很窄（蒸馏数据天然稀少 ✓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
