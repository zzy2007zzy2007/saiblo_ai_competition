"""经验性验证"钉死配置下搜索几乎空转"这条发现（`docs/finding_pinned_config_thin_search_20261005.md`）。

为什么要跑（而不是只读代码）：
  * 这条发现**改变了整个循环的成本模型**（到底是 16 min/局 还是 2 min/局），并且决定
    "配置修复"要不要排在方法层实验之前 ⇒ 值得一个**可复核的直接读数**；
  * 代码阅读会漏掉运行时行为（例如 `selected = sorted(agg)[:k]` 到底会不会真的只剩 1 个）。

做法（**故意便宜**：只 1 个进程、1 个 torch 线程，且**不跑 256 次迭代**）：
  1. 用 raw 策略推演一局，沿途收集若干局面；
  2. 对每个局面，直接调 `BundleMCTS._expand(root, k)`（**只做一次扩展**，约几 ms）
     —— 这正是"根节点候选数"的来源；
  3. 统计三种配置的**根候选数分布**：
       (k=1,  sm=1,  joint,   skip=0)   = 本轮钉死配置
       (k=24, sm=15, pos-only, skip=1)  = 历史标准配置（54.3% 那次）
       (k=24, sm=15, joint,   skip=1)   = 厚 joint
  4. 另测**一次完整 search 的墙钟**（钉死 vs 历史），用来解释 16 min/局 vs ~1.8 min/局。

断言：
  * 钉死配置在**每个**局面上根候选数都 == 1（⇒ 搜索不改变选择，256 次迭代纯浪费）；
  * 厚 joint 至少在多数局面上 ≥2（⇒ 那里的搜索真的有得选）。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/selftest_thin_search_claim.py
退出码 0 = 断言通过。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Ant-Game"))
sys.path.insert(0, str(REPO / "code"))

CKPT = REPO / "training_history" / "vprior" / "posnet_A_k5_m32.pt"
N_STATES = 12          # 采样多少个局面
ROLL_ROUNDS = 60       # raw 推演多少回合（每回合两个玩家 ⇒ 最多 120 个局面可采）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    import numpy as np
    import torch
    torch.set_num_threads(1)

    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode
    from my_ai.decoder import decode_network_output

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt(str(CKPT), feat)

    # ---------- 1) 用 raw 策略推演，沿途收集局面 ----------
    st = make_initial_state(7, True)
    states: list[tuple[object, int, int]] = []
    for rnd in range(ROLL_ROUNDS):
        if st.terminal:
            break
        for pl in (0, 1):
            if st.terminal:
                break
            if len(states) * 2 <= N_STATES * 2 and rnd >= 3:   # 跳过最开始的空局
                states.append((st.clone(), pl, rnd))
            obs = feat.encode_observation(st, pl, np.zeros(96))
            with torch.no_grad():
                o = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                          torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                              intent_decoding=True))
        if not st.terminal:
            st.advance_round()
    # 均匀抽 N_STATES 个
    step = max(1, len(states) // N_STATES)
    picked = states[::step][:N_STATES]
    print(f"参照局面：{len(picked)} 个（raw 推演 {ROLL_ROUNDS} 回合得到 {len(states)} 个候选）\n")

    # ---------- 2) 逐局面只做"一次扩展"，数根候选 ----------
    CFGS = [("钉死 k=1 sm=1 joint skip=0", 1, 1, "joint", False),
            ("历史 k=24 sm=15 pos-only skip=1", 24, 15, "pos-only", True),
            ("厚   k=24 sm=15 joint skip=1", 24, 15, "joint", True)]
    counts: dict[str, list[int]] = {c[0]: [] for c in CFGS}
    for st_i, pl_i, rnd_i in picked:
        for label, k, sm, mode, skip in CFGS:
            m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=k, sample_mult=sm,
                           t_class=0.5, t_pos=1.0, seed=7 + rnd_i, search_mode=mode,
                           pos_pin="argmax", skip_single_candidate=skip)
            root = BundleNode(state=st_i.clone(), player=pl_i)
            m._expand(root, m.k)          # ⚠️ 只扩展一次（不跑迭代）⇒ 便宜
            counts[label].append(len(set(root.bundles)) if root.bundles else 0)

    for label, _, _, _, _ in CFGS:
        c = counts[label]
        frac = sum(1 for x in c if x >= 2) / max(len(c), 1)
        print(f"[{label}] 根候选数 per 局面 = {c}")
        print(f"    中位={sorted(c)[len(c) // 2]}  ≥2 的比例={frac:.1%}  n={len(c)}")

    # ---------- 3) 一次完整 search 的墙钟（解释 16min/局 vs ~1.8min/局）----------
    print()
    st0, pl0, _ = picked[0]
    t: dict[str, float] = {}
    for label, k, sm, mode, skip in (CFGS[0], CFGS[1], CFGS[2]):
        m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=k, sample_mult=sm,
                       t_class=0.5, t_pos=1.0, seed=7, search_mode=mode, pos_pin="argmax",
                       skip_single_candidate=skip)
        t0 = time.time()
        res = m.search(st0, pl0, temperature=1e-6)
        t[label] = time.time() - t0
        print(f"[{label}] 一次 search 墙钟 = {t[label] * 1000:.0f} ms  "
              f"选中动作数={len(res.chosen_bundle or ())}")

    # ---------- 4) 断言 ----------
    print()
    ok = True
    thin = counts[CFGS[0][0]]
    if all(x == 1 for x in thin):
        print(f"✓ 钉死配置（k=1,sm=1）在全部 {len(thin)} 个局面上根候选数都 == 1 "
              f"⇒ 搜索不改变选择（256 次迭代纯浪费）")
    else:
        ok = False
        print(f"✗ 钉死配置出现 >1 候选的局面：{thin} ⇒ 发现需要修正")
    thick = counts[CFGS[2][0]]
    if sum(1 for x in thick if x >= 2) / max(len(thick), 1) >= 0.5:
        print(f"✓ 厚 joint 配置 ≥2 候选的比例 = "
              f"{sum(1 for x in thick if x >= 2) / len(thick):.1%} ⇒ 那里的搜索真的有得选")
    else:
        ok = False
        print(f"✗ 厚 joint 配置 ≥2 候选的比例过低：{thick}")
    ratio = t[CFGS[0][0]] / max(t[CFGS[1][0]], 1e-9)
    print(f"· 单次 search 墙钟比（钉死/历史）= {ratio:.1f}× "
          f"⇒ 与「钉死配置又弱又慢」一致" if ratio > 1 else "· 墙钟比异常，需查")
    print()
    print("=== 断言全部通过 ===" if ok else "=== 有断言失败 ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
