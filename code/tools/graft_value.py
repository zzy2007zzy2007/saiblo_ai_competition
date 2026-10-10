"""`graft_policy.py` 的**镜像**：把 `--policy`（部署格式三网 ckpt）的 **class_state / pos_state / 元数据**，
与 `--value`（另一个部署格式 ckpt，或只含 `value_state` 的字典）的 **value_state** 合成一个新 ckpt。

为什么需要它（2026-10-09）：`graft_policy.py` 只认**蒸馏格式**（`model_state`）当 `--policy`；
而我们的 `grafted_phold10.pt` 等**本身已是部署格式**（`class_state/pos_state/value_state`）⇒ 用它会 0 覆盖、静默产出错误产物。
本工具显式区分"策略来源"与"价值来源"，并在写出前**逐步核对**（逐位比对 + 打印覆盖计数）。

用法：
    python code/tools/graft_value.py --policy checkpoints/grafted_phold10.pt \\
        --value checkpoints/grafted_sdk_A_vsearch.pt --out checkpoints/grafted_phold10_vsearch.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _state(d: dict, key: str) -> dict | None:
    if key in d and isinstance(d[key], dict):
        return d[key]
    if key == "value_state" and "model_state" in d:
        return d["model_state"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="提供 class_state / pos_state / 元数据的部署格式 ckpt")
    ap.add_argument("--value", required=True, help="提供 value_state 的 ckpt（部署格式或 {value_state:...}）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pol = torch.load(a.policy, map_location="cpu", weights_only=False)
    val = torch.load(a.value, map_location="cpu", weights_only=False)
    cs, ps, vs = _state(pol, "class_state"), _state(pol, "pos_state"), _state(val, "value_state")
    for name, obj in (("class_state", cs), ("pos_state", ps), ("value_state(来自 --value)", vs)):
        if obj is None:
            print(f"   ❌ {name} 取不到 ⇒ 停（检查两个输入 ckpt 的格式）"); return 2
    print(f"   策略来源 {Path(a.policy).name}: class {len(cs)} / pos {len(ps)} 张量")
    print(f"   价值来源 {Path(a.value).name}: value {len(vs)} 张量")

    out = dict(pol)              # 元数据（three_net / num_heads / latent_dim / *_no_bn 等）随策略来源
    out["class_state"] = cs
    out["pos_state"] = ps
    out["value_state"] = vs
    torch.save(out, a.out)

    # 逐步核对：新文件 vs 两个来源
    chk = torch.load(a.out, map_location="cpu", weights_only=False)
    same = lambda A, B: all(k in A and torch.equal(A[k], B[k]) for k in B)
    print(f"   写出 {a.out}")
    print(f"   class_state 与 --policy 逐位相同 = {same(chk['class_state'], cs)}")
    print(f"   pos_state   与 --policy 逐位相同 = {same(chk['pos_state'], ps)}")
    print(f"   value_state 与 --value  逐位相同 = {same(chk['value_state'], vs)}")
    ok = (same(chk["class_state"], cs) and same(chk["pos_state"], ps) and same(chk["value_state"], vs))
    print("   ", "✅ 三方核对通过（这是真正的'策略A+价值B'混合体）" if ok else "❌ 核对失败")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
