# -*- coding: utf-8 -*-
"""机制级诊断 第 3/4 步：扒 flat rollout（s2）的**前缀树**。

对某个真实决策（seed/round/player），重放重建该决策点状态，然后用与采集**逐位一致**的
flat rollout 配置（AZAI_FLAT_ROLLOUTS=64 / HORIZON=16 / random+opponent+decay(1,3) /
SHARE_PREFIX=1 / seed=53 / value_batch=32 / ckpt=posnet_A_k5_m32.pt）跑 `_tree_rollout_value`，
把每条 rollout 的轨迹（共享前缀 + 分叉点 + 叶子值网分）dump 出来。

用法：
    python dump_rollout_tree.py --seed 9000000 --round 128 --player 0 \
        --dumps "lightning,chosen" [--topk 3]

输出：
  1. 重放校验（coins / weapon_cd 与 ents.jsonl 对齐）
  2. 每个候选的前缀树（节点/分叉/叶子分布）+ s2 与 cand.jsonl 对比
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
for p in [ROOT / "code", ROOT / "code" / "cpp_engine", ROOT / "Ant-Game", ROOT / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation                                   # noqa: E402
from SDK.utils.constants import OperationType                             # noqa: E402
from SDK.utils.features import FeatureExtractor                           # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models                 # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy         # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade            # noqa: E402

DATA = ROOT / "training_history" / "collect_v2_s1"
CKPT = str(ROOT / "checkpoints" / "posnet_A_k5_m32.pt")

OP_NAMES = {11: "建塔", 12: "升级", 13: "降级", 21: "闪电", 22: "EMP",
            23: "偏转", 24: "闪避", 31: "升速", 32: "升蚁"}
LIGHT = 21
UPG = 12

# 与采集逐位一致的 flat rollout 配置（见 az_selfplay._collect_and_save + replay.json config）
CFG = dict(k=48, rollouts_per_candidate=64, horizon=16, value_batch=32,
           rollout_mode="random", rollout_random_side="opponent",
           rollout_prob_mode="decay", rollout_decay_c=1.0, rollout_decay_k=3.0,
           rollout_share_prefix=True, rollout_seed=53, max_total_rounds=49152,
           eval_mode="net")


def cat(bundle):
    ts = sorted({int(o[0]) for o in (bundle or ())})
    return "HOLD" if not ts else "+".join(OP_NAMES.get(t, f"op{t}") for t in ts)


def has_op(bundle, op):
    return any(int(o[0]) == op for o in (bundle or ()))


def op_str(op):
    # op = [player, op_type, arg0, arg1]（rollout 里 token 的 op 是四元组）
    pl = int(op[0]); t = int(op[1]); a0 = int(op[2]); a1 = int(op[3])
    if t == 11:
        s = f"建塔({a0},{a1})"
    elif t == 12:
        s = f"升级(id={a0})"
    elif t == 13:
        s = f"降级(id={a0})"
    elif t == 21:
        s = f"闪电({a0},{a1})"
    elif t == 22:
        s = "EMP"
    elif t == 23:
        s = "偏转"
    elif t == 24:
        s = "闪避"
    elif t == 31:
        s = "升速"
    elif t == 32:
        s = "升蚁"
    else:
        s = f"op{t}"
    return f"p{pl}{s}"


def token_str(token):
    return ",".join(op_str(t) for t in token) if token else "·(空)"


def replay_state(seed, target_round, target_player):
    """确定性重放：返回 (target_round, target_player) 决策点的状态。"""
    rp = json.loads((DATA / f"az_selfplay_seed{seed}.replay.json").read_text(encoding="utf-8"))
    steps = rp["steps"]
    by_round = {int(s["round"]): s for s in steps}
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    max_r = max(by_round)
    for r in range(0, min(target_round, max_r) + 1):
        s = by_round.get(r)
        if s is None:
            continue
        if r < target_round:
            st.apply_operation_list(0, [Operation(OperationType(t), a0, a1)
                                        for (t, a0, a1) in s["op0"]])
            st.apply_operation_list(1, [Operation(OperationType(t), a0, a1)
                                        for (t, a0, a1) in s["op1"]])
            if not st.terminal:
                st.advance_round()
        elif r == target_round and target_player == 1:
            st.apply_operation_list(0, [Operation(OperationType(t), a0, a1)
                                        for (t, a0, a1) in s["op0"]])
    return st


def child_for(state, player, bundle):
    from SDK.backend.model import Operation as _Op
    from SDK.utils.constants import OperationType as _OT
    child = state.clone()
    ops = [_Op(_OT(int(t)), int(a0), int(a1)) for (t, a0, a1) in bundle]
    if ops:
        child.apply_operation_list(player, ops)
    if player == 1:
        child.advance_round()
    return child


def render_tree(dump, max_depth=4, topk=5):
    """紧凑前缀树：共享前缀（单子链）+ 分叉点（每节点前 topk 分支，附叶子值均值）。"""
    nodes = dump["nodes"]
    paths = dump["paths"]
    lv = dump["leaf_values"]
    children = defaultdict(list)
    for nid, info in nodes.items():
        if info["parent"] >= 0:
            children[info["parent"]].append(nid)
    # 每个 rollout 经过的节点集合（路径节点序列）
    path_nodes = [[nid for (_, _, nid) in p] for p in paths]
    # 每个节点被哪些 rollout 经过 → 这些 rollout 的叶子值
    node_lv = defaultdict(list)
    for pn, v in zip(path_nodes, lv):
        for nid in pn:
            node_lv[nid].append(v)
    lines = []

    def rec(nid, depth, prefix):
        info = nodes[nid]
        vs = node_lv.get(nid, [])
        n = len(vs)
        mu = float(np.mean(vs)) if vs else 0.0
        if nid == 0:
            lines.append(f"ROOT — {n} rollouts, 叶子 μ={mu:+.3f}")
        else:
            lines.append(f"{prefix}└─ d{info['depth']:>2} {token_str(info['token'])}  "
                         f"n={n} leaf μ={mu:+.3f} σ={np.std(vs):.3f}")
        if depth >= max_depth:
            return
        kids = sorted(children[nid], key=lambda c: -len(node_lv.get(c, [])))
        for cid in kids[:topk]:
            rec(cid, depth + 1, prefix + "   ")
        if len(kids) > topk:
            lines.append(f"{prefix}   …（其余 {len(kids) - topk} 个分支省略）")

    rec(0, 0, "")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--player", type=int, required=True)
    ap.add_argument("--dumps", default="lightning,chosen", help="逗号分隔：lightning / chosen / all / 或候选下标")
    ap.add_argument("--topk", type=int, default=3, help="渲染树时只显示前 topk 大分支")
    ap.add_argument("--max-leaves", type=int, default=12, help="打印多少条叶子分数明细")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（默认有 GPU 就用 GPU）")
    args = ap.parse_args()

    seed, rnd, player = args.seed, args.round, args.player

    # ── 读 cand.jsonl：定位决策行 + 候选 + s1/s2 ──
    cand_lines = (DATA / f"az_selfplay_seed{seed}.cand.jsonl").read_text(encoding="utf-8").splitlines()
    ent_lines = (DATA / f"az_selfplay_seed{seed}.ents.jsonl").read_text(encoding="utf-8").splitlines()
    line_idx = None
    for i, (cl, el) in enumerate(zip(cand_lines, ent_lines)):
        c = json.loads(cl)
        if int(c["round"]) == rnd and int(c["player"]) == player:
            line_idx = i
            cand_rec = c
            ents_rec = json.loads(el)
            break
    if line_idx is None:
        raise SystemExit(f"找不到决策 (round={rnd}, player={player}) in seed={seed}")
    decision_idx = line_idx + 1   # 每个决策都调 search() ⇒ _decision_idx = 行号(1-based)
    cand = cand_rec["cand"]
    s1 = cand_rec["s1"]
    s2 = cand_rec["s2"]
    chosen_idx = cand_rec["chosen_idx"]
    n_cand = len(cand)
    print(f"=== seed={seed} round={rnd} player={player} (决策行 {line_idx} ⇒ decision_idx={decision_idx}) ===")
    print(f"n_cand={n_cand} 金币={ents_rec['coins'][player]} 闪电cd={ents_rec['weapon_cd'][player][1]} "
          f"chosen={cat(cand_rec['chosen'])} (idx={chosen_idx})")
    print(f"cand 概览（按 s2 降序前 8）:")
    for i in sorted(range(n_cand), key=lambda i: -s2[i])[:8]:
        mark = " ← chosen" if i == chosen_idx else (" ← 闪电" if has_op(cand[i], LIGHT) else "")
        print(f"  [{i:>2}] {cat(cand[i]):<14} s1={s1[i]:+.4f} s2={s2[i]:+.4f}{mark}")

    # ── 重放重建状态 + 校验 ──
    st = replay_state(seed, rnd, player)
    ok_coins = int(st.coins[player]) == int(ents_rec["coins"][player])
    ok_cd = int(st.weapon_cooldowns[player, 1]) == int(ents_rec["weapon_cd"][player][1])
    print(f"\n[重放校验] coins={int(st.coins[player])}(期望{ents_rec['coins'][player]}) "
          f"闪电cd={int(st.weapon_cooldowns[player, 1])}(期望{ents_rec['weapon_cd'][player][1]}) "
          f"匹配={'✓' if ok_coins and ok_cd else '✗'}")

    # ── 加载 value model ──
    import torch
    _, _, value_model = load_three_models(CKPT)
    value_model.eval()
    feat = FeatureExtractor(max_actions=96)
    # 采集时值网在 CPU 上跑（az_selfplay 只 torch.set_num_threads(1)，从不 .cuda()）。
    # GPU(cuDNN/TF32) 会把叶子值系统性带偏 ~0.01 ⇒ 要逐位复现 s2 必须用 CPU。
    use_cuda = False
    print(f"value model: {'cuda' if use_cuda else 'cpu'}（与采集一致）")

    searcher = FlatRolloutGreedy(lambda s, p: None, **CFG, value_model=value_model,
                                 feature_extractor=feat)

    # n_rollouts（预算公式：49152 // (n_cand*16)，n_cand≤48 ⇒ 恒 64）
    n_rollouts = int(min(CFG["rollouts_per_candidate"],
                         max(1, CFG["max_total_rounds"] // max(1, n_cand * CFG["horizon"]))))
    print(f"n_rollouts={n_rollouts}")

    # ── 选要 dump 的候选下标 ──
    dumps = args.dumps.split(",")
    idxs = []
    for d in dumps:
        d = d.strip()
        if d == "chosen":
            idxs.append(chosen_idx)
        elif d == "lightning":
            li = [i for i in range(n_cand) if has_op(cand[i], LIGHT)]
            if li:
                idxs.append(max(li, key=lambda i: s2[i]))   # s2 最高的闪电候选
            else:
                print("⚠️ 本轮闪电没进候选")
        elif d == "all":
            idxs.extend(range(n_cand))
        elif d.isdigit():
            idxs.append(int(d))
    idxs = list(dict.fromkeys(idxs))

    for idx in idxs:
        bundle = cand[idx]
        print(f"\n{'='*78}\n候选[{idx}] {cat(bundle)}  bundle={bundle}  s1={s1[idx]:+.4f} "
              f"s2(记录)={s2[idx]:+.4f}\n{'='*78}")
        child = child_for(st, player, bundle)
        _base = ((CFG["rollout_seed"] * 1000003 + decision_idx) * 100003 + idx * 101)
        rngs = [np.random.default_rng(_base + k) for k in range(n_rollouts)]
        searcher._dump_tree = True
        score = searcher._tree_rollout_value(child, player, rngs)
        dump = searcher._tree_dump
        lv = dump["leaf_values"]
        print(f"s2(重算)={score:+.6f}  与记录差={score - s2[idx]:+.6f}")
        print(f"叶子值网分：n={len(lv)} 均值={np.mean(lv):+.4f} 中位={np.median(lv):+.4f} "
              f"min={min(lv):+.4f} max={max(lv):+.4f} std={np.std(lv):+.4f}")
        # 叶子分布直方图
        hist, edges = np.histogram(lv, bins=10)
        print("叶子分布直方图(10 bins):")
        for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
            bar = "#" * int(round(h * 40 / max(1, hist.max())))
            print(f"  [{lo:+.2f},{hi:+.2f}) {h:>3} {bar}")
        # 前缀树
        tree_s = render_tree(dump)
        print("\n前缀树（节点=状态，边=本回合对手随机动作；n=经过该节点的 rollout 条数；leaf μ=该子树叶子均值）:")
        print(tree_s)
        # 叶子明细（前 max_leaves 条 + 按值排序的最差/最好）
        print(f"\n叶子值明细（前 {args.max_leaves} 条 rollout 的轨迹末尾 + 值）:")
        for k, p in enumerate(dump["paths"][:args.max_leaves]):
            trail = " ".join(f"d{r}({token_str(t)})" for r, t, nid in p)
            print(f"  rollout#{k:>2} leaf={lv[k]:+.4f} | {trail}")
        if len(lv) > args.max_leaves:
            order = np.argsort(lv)
            print("  ... 叶子分最低 3 条:")
            for k in order[:3]:
                print(f"    rollout#{k} leaf={lv[k]:+.4f}")
            print("  ... 叶子分最高 3 条:")
            for k in order[-3:]:
                print(f"    rollout#{k} leaf={lv[k]:+.4f}")
        # 保存 dump 到 json（供文档引用）
        out = ROOT / "code" / "test_match" / f"_tree_seed{seed}_r{rnd}_p{player}_idx{idx}.json"
        out.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
        print(f"\n[dump 已存] {out}")


if __name__ == "__main__":
    main()
