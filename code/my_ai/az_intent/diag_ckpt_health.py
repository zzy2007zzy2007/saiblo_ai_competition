"""诊断：新扫描产物为什么冒烟 0 建塔（对照：incumbent / 已知能跑的 visithard / 两个新入围）。

检查项：
  1. 顶层键（架构元数据 `no_bn`/`gn`/`latent_dim`/`num_resblocks`/`three_net` … 是否在 ✓）；
  2. `load_three_models` 往返加载 ✓；
  3. 指纹（相对 incumbent：class/pos/value 各有多少张量逐位相同 ✓）；
  4. 位置图健全性：finite / min / max / std；**该类合法格内能否解出操作**（`decode_head` 逐类试）；
  5. **图与 incumbent 的差**（全图平均 |Δ|）。

用法：python -u code/my_ai/az_intent/diag_ckpt_health.py ckpt1 ckpt2 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")
    os.environ.setdefault("AZAI_BAN_OPS", "21")

    import torch
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, load_three_models
    from my_ai.decoder import decode_head, make_class_mask, make_position_masks

    paths = sys.argv[1:] or ["checkpoints/grafted_sdk_A.pt"]
    feat = FeatureExtractor(max_actions=96)
    st = make_initial_state(90000, True)
    for _ in range(80):
        st.advance_round()
    pmask = make_position_masks(st, 0, intent_decoding=True)
    cmask = make_class_mask(st, 0, position_mask=pmask, intent_decoding=True)
    obs = feat.encode_observation(st, 0, np.zeros(96))
    bb = torch.from_numpy(np.asarray(obs["board"], dtype=np.float32)).unsqueeze(0)
    ss = torch.from_numpy(np.asarray(obs["stats"], dtype=np.float32)).unsqueeze(0)

    base = torch.load("checkpoints/grafted_sdk_A.pt", map_location="cpu", weights_only=False)
    tot = len(base["class_state"])
    ref_am = None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        has_meta = all(k in ck for k in ("no_bn", "latent_dim", "num_resblocks", "three_net"))
        try:
            cm, pm, vm = load_three_models(p)
            rt = "ok"
        except Exception as e:  # noqa: BLE001
            rt = f"FAIL {str(e)[:60]}"
            cm = pm = vm = None
        fp = {}
        for k in ("class_state", "pos_state", "value_state"):
            if k in ck:
                try:
                    fp[k] = sum(1 for t in base[k] if torch.equal(base[k][t], ck[k][t]))
                except Exception:  # noqa: BLE001
                    fp[k] = -1
        line = f"\n  === {Path(p).name}\n    顶层键 {len(ck)}（含架构元数据={has_meta}）、往返加载={rt}、" \
               f"指纹 class {fp.get('class_state')}/{tot} pos {fp.get('pos_state')}/{tot} value {fp.get('value_state')}/{tot}"
        print(line, flush=True)
        if pm is None:
            continue
        pm.eval()
        with torch.no_grad():
            am = pm(bb, ss)["action_map"][0].numpy().astype(np.float64)
        if ref_am is None:
            ref_am = am
        ok_classes = [c for c in range(cmask.shape[0])
                      if cmask[c] and pmask[c].any()
                      and decode_head(am[c], am, cmask, pmask, st, 0, class_id=c,
                                      temperature=0.0, pos_temperature=0.0, intent_decoding=True) is not None]
        econ_ok = [c for c in ok_classes if c <= 16]
        print(f"    finite={np.isfinite(am).all()} min={np.nanmin(am):+.3f} max={np.nanmax(am):+.3f} "
              f"std={np.nanstd(am):.4f}；**可解出 op 的类数={len(ok_classes)}（其中经济类 {len(econ_ok)}：{econ_ok[:8]}）**；"
              f"全图与 incumbent 的 |Δ| 均值={float(np.abs(am - ref_am).mean()):.4f}", flush=True)
        # 经济类里"图上最优格"是否落在合法格里
        for c in (0, 1, 5, 13, 16):
            if c >= am.shape[0]:
                continue
            legal = pmask[c]
            if not legal.any():
                continue
            best_all = np.unravel_index(int(np.argmax(am[c])), am[c].shape)
            inside = bool(legal[best_all])
            print(f"      类 {c:2d}：全图 argmax 格 {best_all} 在合法格内={inside}；"
                  f"合法格内 argmax 值={float(np.max(np.where(legal, am[c], -np.inf))):+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
