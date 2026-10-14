"""零对局检查：**搜索选中的格，落在位置网评分前 N 名的比例**（K=24 基线 vs K=100 拓宽）。

用户 2026-10-13 提问：选 100 个候选时，排第一的动作落在**位置网评分前 24** 的比例是多少？
  * 若绝大多数仍在前 24 名内 ⇒ 宽菜单没带来新信息（只是加噪声 ✗）；
  * 若很大比例跑到前 24 名之外 ⇒ 宽菜单真的开采了新格 ⇒ 掉强度是"新格本身差" ✓。

做法（禁闪电制度 ✓，零对局 ✓）：对每个探针状态、每个 (K, ITERS)：
  跑一次搜索 ⇒ 取 `argmax(visits)` 的 bundle ⇒ 其第一个带位置 op 的 (类 c, 格 (x,y)) ✓；
  再用**该类的位置合法掩码** `make_position_masks(...)[c]` ✓ 把图上的合法格按分数降序排 ⇒ 求该格**名次** ✓；
  同时算位置网自己的**前 24 名集合**（该类合法格内 ✓）⇒ 看选中格是否属于它 ✓。

用法：
  python -u code/my_ai/az_intent/probe_menu_rank.py --states 8 --rounds 60,120,200 --configs 24:256,100:2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _op_of(bundle):
    for op in bundle:
        if len(op) >= 3:
            return int(op[0]), int(op[1]), int(op[2])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--configs", default="24:256,100:2000")
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--rounds", default="60,120,200")
    ap.add_argument("--sample-mult", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")
    os.environ.setdefault("AZAI_BAN_OPS", "21")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from SDK.utils.features import FeatureExtractor

    from my_ai.decoder import make_position_masks
    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    _m, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    # ⚠️ 修：`net_fn` 的签名是 (state, player) ✗ ⇒ 取图要用**位置网本身**前向 ✓
    from my_ai.az_intent.az_selfplay import load_three_models
    _cm, _pm, _vm = load_three_models(args.ckpt)
    _pm.eval()

    cfgs = []
    for tok in args.configs.split(","):
        k, it = tok.split(":")
        cfgs.append((int(k), int(it)))
    rounds = [int(x) for x in args.rounds.split(",")]
    per = max(1, args.states // max(len(rounds), 1))
    states = []
    for r in rounds:
        for i in range(per):
            st = make_initial_state(91000 + i * 7 + r, True)
            for _ in range(r):
                if st.terminal:
                    break
                st.advance_round()
            if not st.terminal:
                states.append(st)
    print(f"[rank] ckpt={Path(args.ckpt).name} 状态 {len(states)} 个；配置 {cfgs}", flush=True)

    for (k, it) in cfgs:
        ranks, inside24, out24, nochoice, illegal = [], 0, 0, 0, 0
        t0 = time.time()
        for st in states:
            m = BundleMCTS(net_fn, iterations=it, max_depth_rounds=4, k=k, sample_mult=args.sample_mult,
                           t_class=0.5, t_pos=1.0, c_puct=1.25, seed=args.seed, search_mode="pos-only",
                           pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_every=1,
                           mc_root_only=1, mc_topk=3, mc_upg=1)
            m.search(st, 0, temperature=1e-6)
            root = getattr(m, "last_root", None)
            if root is None or len(root.children) < 2:
                nochoice += 1
                continue
            v = np.asarray([float(c.visits) for c in root.children], dtype=np.float64)
            pick = int(np.argmax(v))
            op = None
            for b in [root.bundles[pick]]:
                op = _op_of(b)
            if op is None:
                nochoice += 1
                continue
            c, x, y = op
            pmask = make_position_masks(st, 0, intent_decoding=True)
            legal = np.asarray(pmask[c], dtype=bool) if c < len(pmask) else None
            if legal is None or not legal.any():
                illegal += 1
                continue
            import torch as _t
            obs = feat.encode_observation(st, 0, np.zeros(96))
            bb = _t.from_numpy(np.asarray(obs["board"], dtype=np.float32)).unsqueeze(0)
            ss = _t.from_numpy(np.asarray(obs["stats"], dtype=np.float32)).unsqueeze(0)
            with _t.no_grad():
                _amap = _pm(bb, ss)["action_map"][0].numpy().astype(np.float64)
            am = _amap[c]
            cells = np.argwhere(legal)                      # (M,2)：(x,y) ✓
            score = np.asarray(am, dtype=np.float64)[legal]
            order = np.argsort(-score, kind="stable")
            hit = np.where((cells[order] == np.array([x, y])).all(1))[0]
            if not len(hit):
                illegal += 1
                continue
            r = int(hit[0]) + 1
            ranks.append(r)
            if r <= 24:
                inside24 += 1
            else:
                out24 += 1
        n = len(ranks)
        if not n:
            print(f"  K={k} ITERS={it}: 无有效样本（无选择 {nochoice}，非法 {illegal}）", flush=True)
            continue
        arr = np.asarray(ranks)
        print(f"\n  === K={k} ITERS={it}（有效样本 {n}；无选择 {nochoice}；无法定位 {illegal}；用时 {time.time()-t0:.0f}s）", flush=True)
        print(f"      名次分布：中位 {np.median(arr):.1f}、均值 {arr.mean():.1f}、≤3: {(arr<=3).mean()*100:.1f}%、"
              f"≤10: {(arr<=10).mean()*100:.1f}%、**≤24: {inside24/n*100:.1f}%**、>24: {out24/n*100:.1f}%", flush=True)
        print(f"      最大名次 {int(arr.max())}（该类合法格数中位 {int(np.median([int(np.asarray(make_position_masks(s,0,intent_decoding=True)[0]).sum()) for s in states]))}）", flush=True)
    print("\n[rank] 判读：① K=24 时选中格**应当**几乎总在网的前 24 名内（菜单本身就是它 ✓）；"
          "② 关键看 **K=100 时 >24 的比例**：高 ⇒ 宽菜单真开采了新格（掉强度=新格差 ✓）；≈0 ⇒ 宽菜单只是加噪声 ✓。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
