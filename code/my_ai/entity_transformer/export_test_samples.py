#!/usr/bin/env python3
"""从投影后的数据集导出若干样本到纯二进制，供 C++ 前向对拍。

格式（小端）：
  int32 n_samples
  each:
    int32 n_tw
      n_tw × (int32 x, int32 y, int32 type_id, float32 hp_ratio)
    int32 n_ant
      n_ant × (int32 x, int32 y, float32 hp_ratio, int32 kind, int32 level, int32 age, int32 beh)
    int32 base_hp, int32 coins
    float32 expected_value
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.entity_transformer.model import TOWER_TYPE_TO_ID  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training_history/distill_top2/sim_value_ds_128.npz")
    ap.add_argument("--out", default="D:/dsh_build_tmp/value_test_samples.bin")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    d = np.load(_REPO / a.data if not Path(a.data).is_absolute() else a.data)
    N = d["value"].shape[0]
    # 选 value 分布上分散的样本
    order = np.argsort(d["value"])[:: max(1, N // a.n)][: a.n]

    # 同时算 PyTorch 的预测（用来验证 C++ 前向是否逐位一致）
    import torch
    from my_ai.entity_transformer.model import EntityTransformerValue
    ck = torch.load(_REPO / "checkpoints/entity_sim_value_small.pt", map_location="cpu")
    cfg = ck["config"]
    model = EntityTransformerValue(d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                   n_heads=cfg["n_heads"], use_cells=cfg["use_cells"],
                                   pos_mode=cfg["pos_mode"])
    model.load_state_dict(ck["state_dict"]); model.eval()
    o = np.asarray(order)
    batch = {
        "tower_cont": torch.from_numpy(d["tower_cont"][o]).float(),
        "tower_type": torch.from_numpy(d["tower_type"][o]).long(),
        "tower_pos": torch.from_numpy(d["tower_pos"][o]).long(),
        "tower_mask": torch.from_numpy(d["tower_mask"][o]).bool(),
        "ant_cont": torch.from_numpy(d["ant_cont"][o]).float(),
        "ant_kind": torch.from_numpy(d["ant_kind"][o]).long(),
        "ant_behavior": torch.from_numpy(d["ant_behavior"][o]).long(),
        "ant_pos": torch.from_numpy(d["ant_pos"][o]).long(),
        "ant_mask": torch.from_numpy(d["ant_mask"][o]).bool(),
        "stats": torch.from_numpy(d["stats"][o]).float(),
    }
    with torch.no_grad():
        pred = model(batch).squeeze(1).numpy() * ck["value_std"] + ck["value_mean"]

    out = bytearray()
    out += struct.pack("<i", len(order))
    for k_idx, i in enumerate(order):
        tw = [(int(d["tower_pos"][i, j] // 19), int(d["tower_pos"][i, j] % 19),
               int(d["tower_type"][i, j]), float(d["tower_cont"][i, j, 1]))
              for j in range(d["tower_mask"][i].shape[0]) if d["tower_mask"][i, j]]
        an = [(int(d["ant_pos"][i, j] // 19), int(d["ant_pos"][i, j] % 19),
               float(d["ant_cont"][i, j, 1]), int(d["ant_kind"][i, j]),
               int(round(d["ant_cont"][i, j, 2] * 2)), int(round(d["ant_cont"][i, j, 3] * 64)),
               int(d["ant_behavior"][i, j]))
              for j in range(d["ant_mask"][i].shape[0]) if d["ant_mask"][i, j]]
        base_hp = int(round(float(d["stats"][i, 0]) * 50))
        coins = int(round(float(d["stats"][i, 1]) * 300))
        out += struct.pack("<i", len(tw))
        for (x, y, t, hr) in tw:
            out += struct.pack("<iiif", x, y, t, hr)
        out += struct.pack("<i", len(an))
        for (x, y, hr, k, lv, age, b) in an:
            out += struct.pack("<iifiiii", x, y, hr, k, lv, age, b)
        out += struct.pack("<ii f f", base_hp, coins, float(d["value"][i]), float(pred[k_idx]))
    Path(a.out).write_bytes(bytes(out))
    print(f"[samples] {len(order)} -> {a.out} ({len(out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
