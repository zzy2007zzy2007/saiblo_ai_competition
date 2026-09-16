"""把一个（单网或二网）checkpoint 拆成三网 checkpoint：类网 / 位置网 / 价值网。

策略网的 state_dict **原样拷贝两份**给 ``class_state`` 与 ``pos_state``
（两者同源、都未训练；骨干因此是两份独立拷贝，之后各自演化），价值网沿用
``value_state``（没有就拷贝策略网）。**故意不写 ``model_state``** —— 见
docs/az_three_net_split_plan.md §3：否则旧的单/二网加载器会静默读到"没训过的
action_map"，是最容易埋雷的地方。

用法：
    python code/my_ai/az_intent/make_three_net_ckpt.py \
        --src training_history/az_fixed/mix_r10p_vw_pol_frozen.pt \
        --out training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO / "Ant-Game", _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_STATE_KEYS = ("model_state", "value_state", "class_state", "pos_state")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"[三网拆分] src = {args.src}")
    print(f"[三网拆分] 源里的权重键: {[k for k in ckpt if k in _STATE_KEYS]}")
    if "model_state" not in ckpt:
        sys.exit("源里没有 model_state（策略网），无法拆分")
    if "class_state" in ckpt or "pos_state" in ckpt:
        sys.exit("源已经是三网 checkpoint，无需再拆")

    policy_sd = ckpt["model_state"]
    value_sd = ckpt.get("value_state", ckpt["model_state"])
    meta = {k: v for k, v in ckpt.items() if k not in _STATE_KEYS}
    print(f"[三网拆分] 透传的元数据键: {sorted(meta.keys())}")

    out = dict(meta)
    out["class_state"] = {k: v.clone() for k, v in policy_sd.items()}
    out["pos_state"] = {k: v.clone() for k, v in policy_sd.items()}
    out["value_state"] = {k: v.clone() for k, v in value_sd.items()}
    out["three_net"] = True
    out["three_net_source"] = str(args.src)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[三网拆分] 写出 -> {args.out}")

    # ── 校验：三份骨干逐位同源、且与源一致 ─────────────────────────────────
    back = torch.load(args.out, map_location="cpu", weights_only=False)
    exact = 0
    for k, v in policy_sd.items():
        same = torch.equal(back["class_state"][k], back["pos_state"][k]) and \
               torch.equal(back["class_state"][k], v)
        exact += int(same)
    print(f"[三网拆分] 校验：{exact}/{len(policy_sd)} 个策略张量在三处逐位相同")
    n_vsame = sum(int(torch.equal(back["value_state"][k], v)) for k, v in value_sd.items())
    print(f"[三网拆分] 校验：价值网 {n_vsame}/{len(value_sd)} 个张量逐位相同")
    print(f"[三网拆分] 校验：无 model_state 键 = {'model_state' not in back}")
    if exact != len(policy_sd) or n_vsame != len(value_sd):
        sys.exit("[三网拆分] !! 逐位校验失败，中止")


if __name__ == "__main__":
    main()
