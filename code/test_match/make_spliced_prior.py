"""按类拼接两个位置网（2026-10-23，用户设想）。

依据（`diag_prior_fair.py`，固定候选集 + 固定教师 posnet_A.value_state）：
  · 建塔通道 tower0-15：我们的网 **-0.113~-0.122**（负相关），SDK 蒸馏网 **+0.022~+0.031** ⇒ 用蒸馏；
  · 闪电通道 LIGHTNING：我们 **+0.607**，蒸馏 +0.554 ⇒ **用我们的**。

拼接口径（24 类，见 decoder.NUM_CLASSES 注释）：
  0-16（16 种塔 + 降级）、21-22（基地升级）→ **grafted_sdk_A**
  17-20（闪电 + 其他超武）              → **posnet_A**
其余（trunk、stats_mlp、value_head）从 posnet_A 全量保留。

产出的权重行：`action_map_conv.weight/bias`（24 行）与 `policy_heads.*.weight/bias`（24 行）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "checkpoints/posnet_A_k5_m32.pt"
DIST = REPO / "checkpoints/grafted_sdk_A.pt"
OUT = REPO / "checkpoints/spliced_posonly_A.pt"

TOWER_CLASSES = list(range(0, 17)) + [21, 22]     # 用蒸馏
SUPER_CLASSES = [17, 18, 19, 20]                  # 用我们（闪电 + 其他超武）


def splice_rows(a: torch.Tensor, b: torch.Tensor, rows_b: list[int]) -> torch.Tensor:
    """a=我们的（基础），b=蒸馏；把 rows_b 指定的行替换成 b 的。"""
    out = a.clone()
    for r in rows_b:
        if r < out.shape[0] and r < b.shape[0]:
            out[r] = b[r]
    return out


def main() -> int:
    ours = torch.load(OURS, map_location="cpu")
    dist = torch.load(DIST, map_location="cpu")
    pa, pb = ours["pos_state"], dist["pos_state"]
    new = dict(pa)
    n_replaced = 0
    for key in list(pa.keys()):
        if "action_map_conv" in key:
            if key not in pb:
                print(f"  ⚠️ 蒸馏网缺少 {key}，跳过")
                continue
            if pa[key].dim() >= 1 and pa[key].shape[0] == 24:
                new[key] = splice_rows(pa[key], pb[key], TOWER_CLASSES)
                n_replaced += 1
                print(f"  拼接 {key}: 用蒸馏替换 {len(TOWER_CLASSES)} 类，保留我们 {len(SUPER_CLASSES)} 类")
    out = dict(ours)
    out["pos_state"] = new
    out["spliced_from"] = f"{OURS.name} + {DIST.name}"
    out["spliced_note"] = ("tower0-16+base21-22 <- grafted_sdk_A; "
                           "lightning17+super18-20 <- posnet_A; rest <- posnet_A")
    torch.save(out, OUT)
    print(f"\n  已写出 {OUT}（替换 {n_replaced} 个张量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
