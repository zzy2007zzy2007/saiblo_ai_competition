"""把**自对弈**的 pkl（`az_selfplay.py` 的输出）转成"类策略"训练数据：
    {board, stats, player, chosen_cls[3], class_mask[24], value_target}

为什么需要它（2026-10-05，按章程 §2 新增约束）：
* **方法侧**要修的是"**类轴**"（何时/花在哪个类）——寄存器里已证类头是**常数函数**
  （跨局面 logit std 仅 0.024）、而手写储备规则能在这一维拿到 +13pp（diagnostic）。
* 训练这类策略需要 **(state, 实际选的类, 对局结果)**；**自对弈数据不触碰章程 §2 的第三类约束**
  （`rule_v4` 只当测试）。原来的 pkl 里已经有 `intent_counts`/`class_mask`/`value_target`，
  缺的只是"把选中的类抽出来"这一步 ⇒ 本脚本补上。

口径与 `az_bridge_ai.py` / `report_posnet_diag.py` 一致：
  候选 j = `visit` 最大的那个；head h 的类 = `intent_counts[j][h]` 里**计数最大**的那个 intent 的类号；
  没有则记 255（= 该 head 没产出可执行 intent）。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/ingest_selfplay_classes.py \
      --src training_history/vprior/data_pin002_100 --dst training_history/vprior/data_sp_cls100
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def chosen_classes(sample) -> list:
    ic = sample.get("intent_counts")
    out = [255, 255, 255]
    if not ic:
        return out
    vis = sample.get("visit")
    j = 0
    if vis is not None and len(vis):
        j = int(np.argmax(np.asarray(vis, dtype=float)))
    j = min(j, len(ic) - 1)
    per_head = ic[j]
    for h in range(3):
        d = per_head[h] if h < len(per_head) else {}
        if d:
            out[h] = int(max(d.items(), key=lambda kv: kv[1])[0][0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_absolute():
        src = REPO / src
    dst = Path(a.dst)
    if not dst.is_absolute():
        dst = REPO / dst
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("az_selfplay_seed*.pkl"))
    if a.max_games:
        files = files[:a.max_games]
    n_s, n_cls = 0, Counter()
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        samples = d["samples"] if isinstance(d, dict) else d
        rows = []
        for s in samples:
            cc = chosen_classes(s)
            for c in cc:
                if c != 255:
                    n_cls[c] += 1
            rows.append({"board": np.asarray(s["board"], dtype=np.float16),
                         "stats": np.asarray(s["stats"], dtype=np.float16),
                         "player": int(s["player"]),
                         "chosen_cls": np.asarray(cc, dtype=np.int64),
                         "class_mask": np.asarray(s["class_mask"], dtype=bool),
                         "value_target": float(s["value_target"])})
        with open(dst / f.name, "wb") as fh:
            pickle.dump({"samples": rows}, fh)
        n_s += len(rows)
    print(f"转换 {len(files)} 局 -> {n_s} 条 -> {dst}")
    tot = sum(n_cls.values())
    print(f"被选中的类分布（head 次，共 {tot}；只列前 12）:")
    for c, k in n_cls.most_common(12):
        print(f"  类 {c:2d}: {k:7d}  ({k/max(tot,1):5.1%})")
    print(f"未产出 intent 的 head 次 = {3*n_s - tot}（{(3*n_s-tot)/max(3*n_s,1):.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
