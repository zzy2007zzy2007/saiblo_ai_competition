"""迁移性检验：把 rule_v4 的**闪电位置**换成我们网络/价值网的判断，其余规则不变，打 rule_v4。

为什么值得做：我们这一路所有证据都是"它对我们自己的智能体有帮助"，而我们自己很弱
（42% vs rule_v4）。换到一个**强策略**上（rule_v4 的其余决策不动），如果闪电位置仍被改好，
说明那份位置知识是真的；如果改不动，说明它可能只是在补偿我们自己的弱先验。

两侧都是 rule_v4 血统（我方 = `rule_v4_mcts_lightning` 的子类，对手 = **原封未动的** rule_v4），
镜像配对、确定性 ⇒ `--null`（我方也用原版）应给出**「每对恰好一胜一负、方差 0」**的零方差对照。
⚠️ 口径：本脚本按**均值口径**打印 ⇒ `0.5000 ± 0.0000`、偏离 0.5 的 pair 数 = 0；
换成**求和口径**（赢一局记 1.0）才是 `1.0`。**同一件事的两种写法，别把 0.5 误读成"没通过"**
（2026-09-22 实测 `--null --pairs 8 --seed 7`：0.5000 / SE 0.0000 / 0/8 偏离 ✓）。

用法:
    python code/test_match/rule_v4_lightning_match.py --null      --pairs 8
    python code/test_match/rule_v4_lightning_match.py --pos-source net   --pairs 32
    python code/test_match/rule_v4_lightning_match.py --pos-source value --pairs 32
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COPY = REPO / "其他版本ai" / "rule_v4_mcts_lightning"
ORIG = REPO / "其他版本ai" / "rule_v4"
for _p in (REPO / "Ant-Game", REPO / "code", COPY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _orig_ai_class():
    """原封未动的 rule_v4 AI（只按路径加载成独立模块名；文件不改）。
    唯一的内存内补丁：把它的 log() 换成空实现 —— 否则跑评测会写 GB 级 ai_decisions.log。"""
    m = _load_by_path("rv4_orig_ai", ORIG / "ai.py")
    m.RuleBasedAI.log = staticmethod(lambda message: None)
    return m.AI


def _worker(job):
    seed, our_player, pos_source, ckpt, native, max_rounds, mode, opp_mode, is_null = job
    import torch
    torch.set_num_threads(1)
    from my_ai.az_intent.az_selfplay import make_initial_state

    OrigAI = _orig_ai_class()
    import mcts_lightning                       # 拷贝目录里的子类
    ours = mcts_lightning.AI_LightningSearch(seed=seed, pos_source=pos_source, ckpt=ckpt,
                                             mode=mode)
    if opp_mode == 'lightning_only':
        opp = mcts_lightning.AI_LightningSearch(seed=seed, pos_source=None, mode='lightning_only')
    else:
        opp = OrigAI(seed=seed)
    if is_null:                                 # --null 对照：我方也用原版
        ours = OrigAI(seed=seed)

    st = make_initial_state(seed, native)
    _rnd = 0
    for _rnd in range(max_rounds):
        if st.terminal:
            break
        for pl in (0, 1):
            if st.terminal:
                break
            ops = (list(ours.choose_bundle(st, pl).operations) if pl == our_player
                   else opp.choose_operations(st, pl))
            st.apply_operation_list(pl, ops)
        if not st.terminal:
            st.advance_round()

    if st.terminal and st.winner is not None:
        score = 1.0 if st.winner == our_player else (0.0 if st.winner == 1 - our_player else 0.5)
    else:
        a, b = st.bases[our_player].hp, st.bases[1 - our_player].hp
        score = 0.5 if a == b else (1.0 if a > b else 0.0)
    print("[game] seed=%d our_player=%d %s us=%d opp=%d rounds=%d"
          % (seed, our_player, "WIN" if score == 1.0 else ("LOSS" if score == 0.0 else "DRAW"),
             st.bases[our_player].hp, st.bases[1 - our_player].hp, _rnd), flush=True)
    return {"score": score,
            "n_lightning": getattr(ours, "n_lightning", 0),
            "n_fired": getattr(ours, "n_fired", 0),
            "n_override": getattr(ours, "n_override", 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-source", default="net", choices=["net", "value", "mcts", "none"])
    ap.add_argument("--null", action="store_true", help="零方差对照：我方也用原版 rule_v4")
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_r1.pt")
    ap.add_argument("--mode", default="full", choices=["full", "lightning_only"])
    ap.add_argument("--opp-mode", default="full", choices=["full", "lightning_only"])
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native-engine", action="store_true", default=True)
    ap.add_argument("--max-rounds", type=int, default=512)
    a = ap.parse_args()
    pos_source = None if a.null else (None if a.pos_source == "none" else a.pos_source)
    tag = ("NULL(原版 vs 原版)" if a.null else
       f"mode={a.mode} opp={a.opp_mode} pos_source={pos_source}")

    jobs = []
    for i in range(a.pairs):
        s = a.seed + i
        for pl in (0, 1):
            jobs.append((s, pl, pos_source, a.ckpt, a.native_engine, a.max_rounds,
                         a.mode, a.opp_mode, a.null))

    import multiprocessing as mp
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            res = pool.map(_worker, jobs)
    else:
        res = [_worker(j) for j in jobs]

    sc = np.asarray([r["score"] for r in res], dtype=np.float64)
    pairs = sc.reshape(a.pairs, 2).sum(axis=1)
    ps = pairs / 2.0                      # 归一到 [0,1]：1.0 = 两局全胜
    n = len(pairs)
    se = ps.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    nl = sum(r["n_lightning"] for r in res)
    no = sum(r["n_override"] for r in res)
    nf = sum(r["n_fired"] for r in res)
    t_str = "  (SE=0：镜像严格抵消)" if se == 0 else f"  t = {(ps.mean() - 0.5) / se:+.2f}"
    print(f"\n=== {tag}  {n} pairs ({2 * n} games) seed={a.seed} ===")
    print(f"  配对胜率均值 = {ps.mean():.4f}   (0.5 = 与 rule_v4 持平, 1.0 = 全胜)  "
          f"SE = {se:.4f}{t_str}")
    print(f"  偏离 0.5 的 pair 数 = {int((np.abs(ps - 0.5) > 1e-9).sum())}/{n}   "
          f"净胜局 = {int(sc.sum() - n)}")
    print(f"  闪电决策次数 = {nl}   其中位置被改 = {no}"
          f"（{100.0 * no / max(nl, 1):.1f}%）"
          + (f"   [lightning_only 实放闪电 = {nf}]" if a.mode == "lightning_only" else ""))


if __name__ == "__main__":
    main()
