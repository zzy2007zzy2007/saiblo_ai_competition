"""诊断：**位置头的 24 个类通道里，哪些通道"随局面变化"？**（= 哪些通道被训练过）

动机（2026-10-05）：`pos_pin=reserve`（储备 180）让我们的**建塔**从 0 变成 ~10.7 次/局，
并带来验收 +13.3pp；但**塔放在哪个格子**是由**该类通道的 action_map** 采样 + 搜索排序决定的。
而位置头的训练历来是"类被钉在 argmax（= 闪电 17）"⇒ **通道 0-16 很可能从没被训练过**
（只被 anchor 维持在旧值）⇒ **我们的塔位置可能是"没有信息"的**。
本脚本用**已有的 checkpoint + 旧 npz 的状态**直接量：每个通道的 action_map **跨局面**是否变化。

判读（先写死）：
  * 若通道 17 的跨局面 std 明显大于通道 0-16 ⇒ **0-16 是"死的"**（没学到状态信息）
    ⇒ "塔放哪儿"与"闪电打哪儿"之间**不能共享同一套位置先验**，这是下一步该动的地方。
  * 若 0-16 也有可比的 std ⇒ 位置头其实训到了多通道 ⇒ 本假设被推翻。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_action_map_channels.py \
      --data training_history/vprior/vp_A_k5_m32.npz --ckpt training_history/vprior/posnet_A_k5_m32.pt \
      --n 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "training_history" / "vprior" / "vp_A_k5_m32.npz"))
    ap.add_argument("--ckpt", default=str(REPO / "training_history" / "vprior" / "posnet_A_k5_m32.pt"))
    ap.add_argument("--n", type=int, default=300, help="取多少个决策（均匀抽样）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt

    z = np.load(a.data, allow_pickle=True)
    board_a, stats_a, player = z["board"], z["stats"], z["player"]
    N = len(player)
    idx = np.linspace(0, N - 1, min(a.n, N)).astype(int)
    board = board_a[idx].astype(np.float32)
    stats = stats_a[idx].astype(np.float32)
    print(f"取 {len(idx)} 个状态（共 {N} 行）；设备={a.device}", flush=True)

    feat = FeatureExtractor(max_actions=96)
    model, _ = make_net_fn_from_ckpt(a.ckpt, feat)
    model.eval()
    dev = "cpu"
    # façade（ThreeNetPolicy）没有 .to()：直接在 CPU 上前向（300 个状态很便宜）

    maps = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            b = torch.from_numpy(board[i:i + 64]).to(dev)
            s = torch.from_numpy(stats[i:i + 64]).to(dev)
            out = model(b, s)
            am = out["action_map"]
            am = am if isinstance(am, np.ndarray) else am.detach().cpu().numpy()
            maps.append(np.asarray(am, dtype=np.float64))
    A = np.concatenate(maps, axis=0)             # (M, 24, 19, 19)
    print(f"action_map 形状 = {A.shape}", flush=True)

    # 每通道：跨局面 std（逐格 std 再取均值）；以及"最热格"的跨局面变化
    print(f"\n{'通道':>4} {'跨局面std':>10} {'最热格不同数':>14} {'均值':>10}")
    rows = []
    for c in range(A.shape[1]):
        M = A[:, c]                                     # (M,19,19)
        std = float(M.std(axis=0).mean())
        flat = M.reshape(len(M), -1)
        hot = flat.argmax(axis=1)
        uniq = int(len(np.unique(hot)))
        rows.append((c, std, uniq, float(M.mean())))
    # 按 std 排序打印
    for c, std, uniq, mu in sorted(rows, key=lambda r: -r[1]):
        mark = "  <-- 闪电(17)" if c == 17 else ("  (HOLD)" if c == 23 else ("  (超武)" if c in (18, 19, 20) else ("  (基地升级)" if c in (21, 22) else "")))
        print(f"{c:>4} {std:>10.5f} {uniq:>14d} {mu:>10.5f}{mark}")
    s17 = [r[1] for r in rows if r[0] == 17][0]
    s_tower = np.mean([r[1] for r in rows if 0 <= r[0] <= 16])
    _ratio = s17 / max(s_tower, 1e-12)
    print(f"\n通道 17 std = {s17:.5f}；通道 0-16（建/升塔）平均 std = {s_tower:.5f} （比值 {_ratio:.1f}×）")
    # ⚠️ 判词必须**跟着数字走**：原来这里写死「比值 ≫1 ⇒ 没信息」，而实测是 2.0× ⇒
    #    判词与读数自相矛盾、会误导后来人（2026-10-06 修）。现在按阈值分档。
    if _ratio >= 5.0:
        print("判读：比值 ≫1（≥5×）⇒ 0-16 通道基本不随局面变化 ⇒ '塔放哪儿'用的是没信息的先验")
    elif _ratio >= 1.5:
        print(f"判读：比值 {_ratio:.1f}×（1.5–5×）⇒ 0-16 通道**随局面变化、但幅度低于闪电**；"
              "「塔放哪儿没信息」**不成立**，但先验好不好仍只能用**打棋力**判定"
              "（逐格命中 / 峰的价值分位都失败过，见 memory feedback_az_policy_prior_role）")
    else:
        print(f"判读：比值 {_ratio:.1f}×（<1.5×）⇒ 0-16 与闪电同量级 ⇒ **原假设被推翻**：多通道都被训到过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
