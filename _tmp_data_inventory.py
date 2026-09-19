"""盘点历史自对弈数据：**哪些 batch 含"闪电以外的动作"**（用户 2026-09-19 提出复用老数据）。

重训价值网只需要 `board/stats/player/value_target` 四个字段 ⇒ 不依赖类索引语义；
"这个 batch 的智能体会不会建塔/升级/降级"可以直接从 `bundles` 的 **op_type** 数出来。

每个 batch 采 1 个 pkl，统计**搜索实际选中的那个候选**（= `bundles[argmax(visit)]`）的 op 构成。

用法: python _tmp_data_inventory.py [--roots training_history/az_cpp/data training_history/az_fixed ...]
"""
from __future__ import annotations
import argparse, pickle, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOTS = ["training_history/az_cpp/data", "training_history/az_fixed"]


def _scan_batch(d: Path):
    pkls = sorted(d.rglob("*.pkl"))
    if not pkls:
        return None
    f = pkls[0]
    try:
        obj = pickle.load(open(f, "rb"))
    except Exception as e:                                    # noqa: BLE001
        return {"batch": d.name, "err": f"{type(e).__name__}: {e}"}
    samples = obj.get("samples", obj) if isinstance(obj, dict) else obj
    if not isinstance(samples, list) or not samples:
        return {"batch": d.name, "err": "no samples"}
    comp = Counter()
    n_act = 0
    vts = []
    for s in samples:
        b = s.get("bundles") or []
        v = s.get("visit")
        if not b:
            continue
        if v is not None and len(v) == len(b):
            import numpy as np
            j = int(np.argmax(v))
        else:
            j = 0
        chosen = b[j]
        if chosen:
            n_act += 1
        for op in chosen:
            comp[int(op[0])] += 1
        vt = s.get("value_target")
        if vt is not None:
            vts.append(float(vt))
    return {"batch": d.name, "file": f.name, "n_pkl": len(pkls), "n": len(samples),
            "acted": n_act, "comp": comp,
            "vt_mean": (sum(vts) / len(vts)) if vts else float("nan"),
            "vt_std": ((sum((x - sum(vts) / len(vts)) ** 2 for x in vts) / len(vts)) ** 0.5
                       if vts else float("nan"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from SDK.utils.constants import OperationType
    name = {int(m): m.name for m in OperationType}

    batches = []
    for r in a.roots:
        p = Path(r)
        if not p.exists():
            continue
        for d in sorted(p.iterdir()):
            if d.is_dir():
                batches.append(d)
    print("[inv] %d 个 batch 目录（根: %s）" % (len(batches), ", ".join(a.roots)), flush=True)
    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        res = [x for x in pool.map(_scan_batch, batches) if x]

    LIGHT = int(OperationType.USE_LIGHTNING_STORM)
    rows = []
    for r in res:
        if "err" in r:
            print("  !! %-16s %s" % (r["batch"], r["err"]))
            continue
        comp = r["comp"]
        tot = sum(comp.values())
        nonlight = tot - comp.get(LIGHT, 0)
        rows.append((nonlight / max(tot, 1), r, nonlight / max(tot, 1)))
    rows.sort(key=lambda x: -x[0])
    print("\n%-16s %6s %6s %7s %7s %7s  %s" %
          ("batch", "n_pkl", "n", "acted%", "vt均值", "vt标准差", "选中动作构成（非闪电占比降序）"))
    for frac, r, _ in rows:
        comp = r["comp"]; tot = sum(comp.values())
        s = ", ".join("%s:%.0f%%" % (name.get(k, str(k)), 100.0 * v / tot)
                      for k, v in comp.most_common(5))
        print("%-16s %6d %6d %6.1f%% %7.2f %7.2f  非闪电%.0f%%  |  %s"
              % (r["batch"], r["n_pkl"], r["n"], 100.0 * r["acted"] / max(r["n"], 1),
                 r["vt_mean"], r["vt_std"], 100 * frac, s), flush=True)


if __name__ == "__main__":
    main()
