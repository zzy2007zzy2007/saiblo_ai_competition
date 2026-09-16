"""把自对弈 pkl 过滤成"只含带位置目标样本"的紧凑数据集（位置-only 阶段用）。

**为什么必须做**：原始 pkl 每局 ~38 MB；500 局 ~19 GB。训练时 `load_samples` 会把
全部样本载入内存 ⇒ **OOM**；而且其中 **97.2% 的样本在位置 CE 上恒为 0**（HOLD 回合
的目标里没有位置），在 anchor_scope=contrib 下连 anchor 也不参与 ⇒ 完全用不到。

只保留带位置目标的样本，并丢掉本阶段不用的字段（`bundles`/`recorded_head_logits`/
`player`/`search_skipped`）⇒ 约 **2.8% 的体量**（500 局 ≈ 0.5 GB）。

判据直接复用 `az_train._marginalize`（训练时用的同一函数），保证过滤器与训练器的
"什么算有位置目标"永远一致，不会两边漂移。

用法：
    python code/my_ai/az_intent/filter_pos_samples.py --src <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO / "Ant-Game", _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KEEP = ("board", "stats", "class_mask", "position_mask",
        "intent_counts", "visit", "recorded_action_map")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 pkl 目录（含 az_selfplay_seed*.pkl）")
    ap.add_argument("--out", required=True, help="输出目录（紧凑数据集）")
    ap.add_argument("--num-heads", type=int, default=3)
    args = ap.parse_args()

    from my_ai.az_intent.az_train import _marginalize

    def has_pos(s: dict) -> bool:
        return any(x >= 0 for h in range(args.num_heads)
                   for (_c, x, _y) in _marginalize(s, h))

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("az_selfplay_seed*.pkl"))
    if not files:
        raise SystemExit(f"{src} 下没有 az_selfplay_seed*.pkl")

    n_in = n_keep = 0
    for f in files:
        with open(f, "rb") as fh:
            samples = pickle.load(fh)["samples"]
        n_in += len(samples)
        kept = [{k: s[k] for k in KEEP if k in s} for s in samples if has_pos(s)]
        n_keep += len(kept)
        with open(out / f.name, "wb") as fh:
            pickle.dump({"samples": kept}, fh)
    print(f"[filter] {len(files)} 局：{n_in} -> {n_keep} 样本 "
          f"({n_keep / max(n_in, 1):.1%}) -> {out}", flush=True)
    sz = sum(p.stat().st_size for p in out.glob("*.pkl"))
    print(f"[filter] 紧凑数据集 {sz/1e6:.1f} MB（每局 {sz/max(len(files),1)/1e6:.2f} MB）",
          flush=True)


if __name__ == "__main__":
    main()
