"""检查坐标/镜像一致性：我们两张座位视角的先验输出，是否满足 180° 旋转关系（2026-10-26）。

背景：覆盖率实验里发现"我们 dump 的候选坐标 vs 亚军日志的操作坐标"需要 mirror_xy(即 180° 旋转)
才能对上（identity 0/18 vs mirror_xy 8/18）。若我们的特征编码是"玩家相对"的，而解码没做逆变换，
就会出现座位不一致的 bug（嫌疑正好对应 27pp 座位差 ≈ 355 Elo）。

本脚本做三件事：
  1. 同一局面下，分别算 encode_board(st, 0) 与 encode_board(st, 1)，检查两者的旋转关系；
  2. 用同一张网分别对两份特征前向，比较 action_map 的 旋转相关 vs 恒等相关；
  3. 打印结论（是否镜像一致）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor

CKPT = str(REPO / "checkpoints/posnet_A_k5_m32.pt")


def build_state(seed: int = 77, rounds: int = 60):
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(rounds):
        if st.terminal:
            break
        st.advance_round()
    return st


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, m, _ = load_three_models(CKPT)
    m.eval()
    st = build_state()
    print(f"  局面: seed=77, round={getattr(st, 'round_index', '?')}, bases={[b.hp for b in st.bases]}", flush=True)

    f0 = np.asarray(feat.encode_board(st, 0), dtype=np.float32)
    f1 = np.asarray(feat.encode_board(st, 1), dtype=np.float32)
    print(f"  encode_board(0) shape {f0.shape}   encode_board(1) shape {f1.shape}", flush=True)
    print("  特征平面相关性：", flush=True)
    print(f"    identity(0 vs 1)        {corr(f0, f1):+.3f}")
    print(f"    flip180(0) vs 1         {corr(f0[:, ::-1, ::-1], f1):+.3f}")
    # 逐通道看：哪些通道在 flip 后与另一座位一致
    same_id, same_fl = [], []
    for c in range(f0.shape[0]):
        same_id.append(corr(f0[c], f1[c]))
        same_fl.append(corr(f0[c, ::-1, ::-1], f1[c]))
    n_fl = sum(1 for a, b in zip(same_id, same_fl) if (b > 0.9 and b > a))
    print(f"    逐通道：flip 后高度一致(>0.9) 且优于 identity 的通道数 = {n_fl}/{f0.shape[0]}", flush=True)

    with torch.no_grad():
        a0 = m(torch.from_numpy(f0).unsqueeze(0), torch.from_numpy(
            np.asarray(feat.encode_stats(st, 0), dtype=np.float32)).unsqueeze(0))["action_map"][0]
        a1 = m(torch.from_numpy(f1).unsqueeze(0), torch.from_numpy(
            np.asarray(feat.encode_stats(st, 1), dtype=np.float32)).unsqueeze(0))["action_map"][0]
    a0 = a0.detach().cpu().numpy()
    a1 = a1.detach().cpu().numpy()
    print("  action_map（建塔 0-15 池化）相关性：", flush=True)
    print(f"    identity                {corr(a0[0:16].mean(0), a1[0:16].mean(0)):+.3f}")
    print(f"    flip180(a0) vs a1       {corr(a0[0:16].mean(0)[::-1, ::-1], a1[0:16].mean(0)):+.3f}")
    print(f"    flip180(a1) vs a0       {corr(a1[0:16].mean(0)[::-1, ::-1], a0[0:16].mean(0)):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
