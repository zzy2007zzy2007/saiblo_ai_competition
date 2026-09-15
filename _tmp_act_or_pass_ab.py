"""多出招能不能赢？—— 换解码方式的 A/B（不用搜索）。

镜像对战：我方用改过的解码、对手用**现行 raw 解码**，同一个策略网（两网已拆分，
对手/我方都只用策略网）。每个 seed 打两局（我方=P0 / 我方=P1），
⇒ 先后手不对称在**配对内抵消**（score_pair = 两局得分之和 ∈ [0,2]，1.0 = 持平）。

四臂（`--mode`）：
  none          : 我方也用现行 raw 解码 → **基线**，量出纯先后手不对称
  fallback      : 头 argmax 非法时回退到"logit 最高的合法类"（经典 mask-then-argmax）
                  —— 测"解码器不回退"这个设计本身
  fallback-rand : 触发条件与 fallback **完全相同**，但类**随机**取一个合法的
                  —— 对照：收益来自"多出招"还是"选得好"
  value         : 现行 raw 解码空过时，用价值网 1-ply 选出最好的合法类并强制执行
                  —— arm ④ 的价值版（④ 用的是"logit 顺序第一个能执行的"，0W/32L）

判读：`fallback` / `value` 的配对得分显著 >1.0 ⇒ 类头/解码器是瓶颈（该先修类头）；
≈1.0 或 <1.0 ⇒ 多出招没用，空过是对的（用户经验判断成立）。
同时打印我方**出招率**（none 应 ≈2.7%，fallback 应 ≈50%，value 应 ≈97%）
——干预没改变出招率就说明实现错了。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
HOLD = 23
MAX_TURNS_ROUNDS = 600


def _worker(job):
    seed, mode, ckpt = job
    import torch
    torch.set_num_threads(1)

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import load_split_models, make_initial_state
    from my_ai.decoder import (decode_head, decode_network_output, make_class_mask,
                               make_position_masks)

    feat = FeatureExtractor(max_actions=96)
    policy_model, value_model = load_split_models(ckpt)
    policy_model.eval()
    value_model.eval()
    rng = np.random.default_rng(seed + 555)
    DIAG = {"slots": 0, "masked_argmax_is_hold": 0, "trigger_fired": 0,
            "decode_none": 0, "op_ok": 0, "argmax_cls": {}}

    def policy_out(st, pl):
        obs = feat.encode_observation(st, pl, np.zeros(96))
        with torch.no_grad():
            return policy_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                torch.from_numpy(obs["stats"]).unsqueeze(0).float())

    def value_of(st, pl):
        obs = feat.encode_observation(st, pl, np.zeros(96))
        with torch.no_grad():
            return float(value_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                     torch.from_numpy(obs["stats"]).unsqueeze(0).float())["value"])

    def raw_ops(st, pl):
        return decode_network_output(policy_out(st, pl), st, pl, temperature=0.0,
                                     intent_decoding=True)

    def legal_classes(cm):
        return [c for c in range(24) if c != HOLD and cm[c]]

    def decode_with_class(am, hl, cm, pm, st, pl, cid, pending):
        op = decode_head(hl, am, cm.copy(), pm.copy(), st, pl, class_id=int(cid),
                         temperature=0.0, pos_temperature=0.0, intent_decoding=True)
        if op is None or not st.can_apply_operation(pl, op, pending):
            return None
        return op

    def our_ops(st, pl):
        """Return (ops, acted) for our side under the chosen mode."""
        if mode == "none":
            ops = raw_ops(st, pl)
            return ops, bool(ops)
        out = policy_out(st, pl)
        am = out["action_map"][0].numpy()
        heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
        pm = make_position_masks(st, pl, intent_decoding=True)
        cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
        if mode == "value":
            ops = decode_network_output(dict(out), st, pl, temperature=0.0,
                                        intent_decoding=True)
            if ops:
                return ops, True
        ops, pending = [], ()
        for h in range(3):
            hl = np.asarray(out[heads[h]].squeeze(0).numpy(), dtype=np.float64)
            cands = legal_classes(cm)
            if not cands:
                break
            if mode in ("fallback", "fallback-rand"):
                masked = np.where(cm, hl, -1e9)
                c = int(np.argmax(masked))
                DIAG["slots"] += 1
                DIAG["masked_argmax_is_hold"] += int(c == HOLD)
                if c == HOLD:
                    continue  # 合法类里 HOLD 胜出 ⇒ 这个头空过
                DIAG["trigger_fired"] += 1
                DIAG["argmax_cls"][c] = DIAG["argmax_cls"].get(c, 0) + 1
                if mode == "fallback-rand":
                    c = int(rng.choice(cands))
                op = decode_with_class(am, hl, cm, pm, st, pl, c, pending)
                if op is None:
                    DIAG["decode_none"] += 1
                    continue
                DIAG["op_ok"] += 1
            else:  # value: 逐个合法类做 1-ply，取价值最好的
                best_op, best_v = None, -9.9
                for c in cands:
                    op = decode_with_class(am, hl, cm, pm, st, pl, c, ())
                    if op is None:
                        continue
                    post = st.clone()
                    post.apply_operation_list(pl, [op])
                    v = -value_of(post, 1 - pl)
                    if v > best_v:
                        best_v, best_op = v, op
                if best_op is None:
                    continue
                if not st.can_apply_operation(pl, best_op, pending):
                    continue
                op = best_op
            ops.append(op)
            pending = pending + (op,)
        return ops, bool(ops)

    def play(our_player: int) -> tuple[float, int, int]:
        st = make_initial_state(seed, True)
        our_turns = our_acted = 0
        for _ in range(MAX_TURNS_ROUNDS):
            if st.terminal:
                break
            for pl in (0, 1):
                if st.terminal:
                    break
                if pl == our_player:
                    ops, acted = our_ops(st, pl)
                    our_turns += 1
                    our_acted += int(acted)
                else:
                    ops = raw_ops(st, pl)
                st.apply_operation_list(pl, ops)
            if pl == 1 and not st.terminal:
                st.advance_round()
        if st.terminal and st.winner is not None:
            sc = 1.0 if st.winner == our_player else (0.0 if st.winner == 1 - our_player else 0.5)
        else:
            a, b = st.bases[our_player].hp, st.bases[1 - our_player].hp
            sc = 0.5 if a == b else (1.0 if a > b else 0.0)
        return sc, our_turns, our_acted

    s0, t0, a0 = play(0)
    s1, t1, a1 = play(1)
    return {"seed": seed, "pair_score": s0 + s1, "score_p0": s0, "score_p1": s1,
            "our_turns": t0 + t1, "our_acted": a0 + a1, "diag": DIAG}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CK)
    ap.add_argument("--mode", default="fallback",
                    choices=["none", "fallback", "fallback-rand", "value"])
    ap.add_argument("--pairs", type=int, default=32, help="每个 pair = 两局（先后手互换）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = [(args.seed + i, args.mode, args.checkpoint) for i in range(args.pairs)]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            res = pool.map(_worker, jobs)
    else:
        res = [_worker(j) for j in jobs]

    ps = np.asarray([r["pair_score"] for r in res])
    turns = sum(r["our_turns"] for r in res)
    acted = sum(r["our_acted"] for r in res)
    n = len(res)
    se = ps.std(ddof=1) / np.sqrt(n)
    t_str = "  (SE=0：镜像严格抵消)" if se == 0 else f"t = {(ps.mean() - 1.0) / se:+.2f}"
    print(f"\n=== mode={args.mode}  {n} pairs ({2 * n} games) seed={args.seed} "
          f"ckpt={Path(args.checkpoint).name} ===")
    print(f"  配对得分均值 = {ps.mean():.4f}  (1.0 = 与对手持平)   SE = {se:.4f}{t_str}")
    print(f"  偏离 1.0 的 pair 数 = {int((ps != 1.0).sum())}/{n}   "
          f"净增分 = {ps.sum() - n:+.1f} 分（每 pair 满分 2）")
    print(f"  分布: 2分(双杀)={int((ps == 2).sum())}  1.5={int((ps == 1.5).sum())}  "
          f"1分(各赢一局)={int((ps == 1).sum())}  0.5={int((ps == 0.5).sum())}  "
          f"0分(双败)={int((ps == 0).sum())}")
    print(f"  我方出招率 = {acted / max(turns, 1):.1%}   (回合数 {turns})")
    d = {k: 0 for k in ("slots", "masked_argmax_is_hold", "trigger_fired", "decode_none", "op_ok")}
    hist = {}
    for r in res:
        for k in d:
            d[k] += r["diag"][k]
        for c, v in r["diag"]["argmax_cls"].items():
            hist[c] = hist.get(c, 0) + v
    if d["slots"]:
        print(f"  [diag] 头次={d['slots']}  掩码argmax==HOLD {d['masked_argmax_is_hold'] / d['slots']:.1%}  "
              f"触发(非HOLD) {d['trigger_fired'] / d['slots']:.1%}  "
              f"其中解不出动作 {d['decode_none'] / max(d['trigger_fired'], 1):.1%}")
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:6]
        print(f"         触发的 argmax 类分布(前6): {top}")
    print(f"  分先后手: 我方为 P0 时 {np.mean([r['score_p0'] for r in res]):.1%} / "
          f"P1 时 {np.mean([r['score_p1'] for r in res]):.1%}")


if __name__ == "__main__":
    main()
