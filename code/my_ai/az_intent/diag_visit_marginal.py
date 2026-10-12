"""诊断：**把搜索的访问分布（在"束"上）按类边缘化到"格"上，量真实局面上的目标质量**。

动机（2026-10-12，计划 `docs/pos_seeded_sampling_plan.md` §3）：
`AZAI_SEED_STATE=1` 让候选集合与访问分布成为**状态的确定性函数**（判据不掉强度 ✓），
但搜索的 visit 是在**束**（≤3 个 op 的组合）上的 ✗，而位置头预测的是**格**上的分布 ✗
⇒ 目标必须先**按类边缘化到格**（`intent_counts` 就是为这个 AZ 边缘化设计的 ✓）。本脚本量：
  * 每条决策里，把 `visits_raw` 的权重按 **(类, x, y)** 累加 ⇒ 得到每类的"格分布"；
  * 报告**权重最大那个类**的格分布：**top1 份额 / 归一化熵 / 参与格数**（并与"均匀"对比）；
  * 以及全局的：束大小分布、目标的自洽性（权重大于 0 的类数）。

数据：`AZAI_AZ_DUMP`（policy jsonl：`cands`/`visits_raw`/`labels`/`chosen` ✓）
       + `AZAI_DUMP_RAW_DIR`（obs .bin：`board(28,19,19)f16 + stats(42,)f16 + player + chosen(3) + mask(24)` ✓）
按**记录下标**配对（两者都是逐决策追加 ✓；取 min 长度防半局 ✗）。

用法：python -u code/my_ai/az_intent/diag_visit_marginal.py --policy-dir data/seedstate_dump/az --obs-dir data/seedstate_dump/obs --games 20
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOARD_SHAPE = (28, 19, 19)
N_CLS = 24
BSZ = int(np.prod(BOARD_SHAPE)) * 2          # 20,216（只含 board）
REC = BSZ + 84 + 1 + 3 + N_CLS               # 20,328


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", default="data/seedstate_dump/az")
    ap.add_argument("--obs-dir", default="data/seedstate_dump/obs")
    ap.add_argument("--games", type=int, default=20)
    args = ap.parse_args()

    pol = Path(args.policy_dir)
    obs = Path(args.obs_dir)
    files = sorted(pol.glob("az_*.jsonl"))[: args.games]
    print(f"[marg] policy 文件 {len(files)} 个；obs 目录存在={obs.exists()}")

    top1, ent, ncell, bsizes, classes = [], [], [], collections.Counter(), collections.Counter()
    n_rec = n_used = n_novis = 0
    for jf in files:
        tok = jf.name.split("seed")[1].split(".")[0]
        of = obs / f"obs_seed{tok}.bin"
        recs = []
        for line in open(jf, encoding="utf-8", errors="replace"):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
        raw = np.fromfile(of, dtype=np.uint8) if of.exists() else np.zeros(0, dtype=np.uint8)
        n_obs = raw.size // REC
        for i in range(min(len(recs), n_obs)):
            d = recs[i]
            vr = d.get("visits_raw")
            cands, labels = d.get("cands"), d.get("labels")
            if not isinstance(vr, list) or not vr or not cands or not labels:
                n_novis += 1
                continue
            w = np.asarray([float(x) for x in vr], dtype=np.float64)
            if w.sum() <= 0:
                n_novis += 1
                continue
            n_rec += 1
            bsizes[len(cands[0]) if cands else 0] += 1
            p = w / w.sum()
            # 按 (类, x, y) 边缘化
            cell_mass: dict = collections.defaultdict(float)
            for k, bundle in enumerate(labels):
                if k >= len(p):
                    break
                for op in bundle:
                    c, x, y = int(op[0]), int(op[1]), int(op[2])
                    cell_mass[(c, x, y)] += float(p[k])
            if not cell_mass:
                continue
            by_cls: dict = collections.defaultdict(dict)
            for (c, x, y), m in cell_mass.items():
                by_cls[c][(x, y)] = by_cls[c].get((x, y), 0.0) + m
            # 取"质量最大"的那个类
            c_best = max(by_cls, key=lambda c: sum(by_cls[c].values()))
            mm = np.asarray(list(by_cls[c_best].values()), dtype=np.float64)
            if mm.sum() <= 0 or mm.size < 2:
                continue
            q = mm / mm.sum()
            n_used += 1
            classes[c_best] += 1
            top1.append(float(q.max()))
            ent.append(float(-(q[q > 0] * np.log2(q[q > 0])).sum()))
            ncell.append(int(q.size))

    if not n_used:
        print(f"[marg] 没有可用记录（n_novis={n_novis}）——dump 是否含 visits_raw？")
        return 0
    print(f"[marg] 决策 {n_rec} 条（可用 {n_used}；无 visits_raw/无效 {n_novis}）；束大小分布 {dict(bsizes.most_common(4))}")
    print(f"       权重最大类分布（前 6）：{classes.most_common(6)}")
    print(f"  ⇒ **边缘化到格后的目标质量**：参与格数中位 {np.median(ncell):.0f}；"
          f"**top1 份额 中位 {np.median(top1):.3f}（均值 {np.mean(top1):.3f}）**；"
          f"**归一化熵 中位 {np.median(ent):.3f} bit**（该格数下均匀 ≈ {np.log2(max(np.median(ncell), 2)):.2f}）")
    print("  ⇒ 判读：top1 明显 >1/n 且熵明显 <log2(n) ⇒ 目标有信息量、值得写训练器 ✓；否则应先修目标 ✗")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
