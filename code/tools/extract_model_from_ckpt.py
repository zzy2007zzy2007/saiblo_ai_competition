"""从 ES checkpoint 里抽出「纯模型」，丢掉 ES 训练状态（种群 / leaderboard）。

动机（2026-10-06，用户提出）：`training_history/ga_ss_*/gen_*.pt` 动辄 300–400 MB，
但**真正是模型的只有 ~2 MB** —— 大头是 `ga_pop`（ES 种群：96 个个体 × ~2.2 MB）+ pickle 开销。
把纯模型抽出来（~4 MB）就能入库 / 传输 / 长期保存。

保留：`model_state`（state_dict 形式）、`mean`（参数向量形式）—— 仓库里两种加载方式都有用
      （`collect_value_*.py` 优先用 `mean`、`expand_to_3heads.py` 优先用 `model_state`），
      外加 `generation / num_heads / no_bn / top2_scores / config` 这些小元数据。
丢弃：`ga_pop`（种群）、`top2_params`（某个体的副本）、`leaderboard`（训练状态）
   ⇒ ⚠️ **产物用于推理 / 血统参照，不能用来续训 ES**（要续训得留原文件）。

用法：
    python -u code/tools/extract_model_from_ckpt.py --src <in.pt> --out <out.pt>
    python -u code/tools/extract_model_from_ckpt.py --src <in.pt> --dry-run   # 只打印各键大小
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# 丢掉这些键（ES 训练状态，不是模型）
DROP = ("ga_pop", "top2_params", "leaderboard")


def _byte_size(v) -> int:
    """逻辑字节数（快，但**对 Python 原生 float/int 会低估**：那些是 pickle 里的大头开销）。

    ⚠️ 2026-10-06 核验发现：`leaderboard`（20 个参数列表，`.tolist()` 出来的 Python float）
    在本函数下算成 **0 MB**，实际占了整份 ckpt 的 **23%**。⇒ 汇报一律用 `_saved_bytes`（实际序列化大小）。
    """
    if torch.is_tensor(v):
        return v.numel() * v.element_size()
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, (int, float, bool)):
        return 8  # Python 标量在 pickle 里约 8B 级（很粗，仅供量级参考）
    if isinstance(v, (list, tuple)):
        return sum(_byte_size(x) for x in v)
    if isinstance(v, dict):
        return sum(_byte_size(x) for x in v.values())
    return 0


def _saved_bytes(v) -> int:
    """**实际序列化大小**（= 这个键在 ckpt 里真正占多少）——权威口径，慢一点但准。"""
    import io
    buf = io.BytesIO()
    torch.save({"_": v}, buf)
    return buf.getbuffer().nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 ckpt（如 training_history/ga_ss_.../gen_0120.pt）")
    ap.add_argument("--out", help="输出路径（--dry-run 时可省）")
    ap.add_argument("--dry-run", action="store_true", help="只打印各键大小，不写文件")
    args = ap.parse_args()

    print(f"[1/4] 载入 {args.src}（{os.path.getsize(args.src)/1e6:.1f} MB）…", flush=True)
    obj = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        print(f"✗ 顶层不是 dict（是 {type(obj).__name__}）⇒ 不支持，停止", flush=True)
        return 2

    print(f"[2/4] 各键**实际序列化大小**（共 {len(obj)} 个键；权威口径）：", flush=True)
    sizes = {k: _saved_bytes(v) for k, v in obj.items()}
    total = sum(sizes.values()) or 1
    for k in sorted(sizes, key=lambda k: -sizes[k]):
        mark = "  ← 丢弃" if k in DROP else ""
        print(f"       {sizes[k]/1e6:10.3f} MB  ({100*sizes[k]/total:5.1f}%)  {k}{mark}", flush=True)

    kept = {k: v for k, v in obj.items() if k not in DROP}
    lost = [k for k in obj if k in DROP]
    print(f"[3/4] 保留 {len(kept)} 个键、丢弃 {len(lost)} 个（{', '.join(lost) or '无'}）；"
          f"保留内容 ≈ {sum(sizes[k] for k in kept)/1e6:.2f} MB", flush=True)

    if args.dry_run:
        print("[4/4] --dry-run ⇒ 不写文件", flush=True)
        return 0
    if not args.out:
        print("✗ 没给 --out，停止", flush=True)
        return 2

    torch.save(kept, args.out)
    print(f"[4/4] 写出 {args.out}（{os.path.getsize(args.out)/1e6:.2f} MB）", flush=True)

    back = torch.load(args.out, map_location="cpu", weights_only=False)
    assert set(back) == set(kept), "回读校验失败：键集合不一致"
    assert back["model_state"].keys() == obj["model_state"].keys(), "回读校验失败：model_state 键不一致"
    print("✓ 回读校验通过（键集合 + model_state 键一致）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
