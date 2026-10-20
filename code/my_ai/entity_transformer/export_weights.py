#!/usr/bin/env python3
"""把 `entity_sim_value_small.pt` 导出成 C++ 可直接读的扁平二进制 + JSON 配置。

产物（同目录）：
  `<out>.bin`   : 所有浮点张量按固定顺序连续存放（float32，行主序）
  `<out>.json`  : 张量清单（名字 / shape / 偏移 / 元素数）+ 模型配置 + 归一化常数
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/entity_sim_value_small.pt")
    ap.add_argument("--out", default="checkpoints/entity_sim_value_small")
    a = ap.parse_args()
    ck = torch.load(_REPO / a.ckpt if not Path(a.ckpt).is_absolute() else a.ckpt,
                    map_location="cpu")
    sd = ck["state_dict"]
    flat = bytearray()
    manifest = []
    off = 0
    for name, t in sd.items():
        arr = t.detach().cpu().numpy().astype(np.float32)
        n = int(arr.size)
        manifest.append({"name": name, "shape": list(arr.shape), "offset": off, "numel": n})
        flat += arr.tobytes()
        off += n
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.name + ".bin")).write_bytes(bytes(flat))
    meta = {
        "config": ck.get("config"),
        "value_mean": float(ck.get("value_mean")),
        "value_std": float(ck.get("value_std")),
        "val_r2": float(ck.get("val_r2")),
        "val_mae": float(ck.get("val_mae")),
        "total_floats": off,
        "tensors": manifest,
    }
    (out.parent / (out.name + ".json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[export] {len(manifest)} tensors, {off} floats -> {out}.bin/.json", flush=True)
    for m in manifest:
        print(f"  {m['name']:<40} {m['shape']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
