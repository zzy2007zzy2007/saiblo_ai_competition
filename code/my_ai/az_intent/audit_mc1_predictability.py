"""审计：**"行为克隆 MC1"到底还剩多少可学信号**（只读，零对局）。

背景（2026-10-11）：MC1 = incumbent 网 + `MC_EVERY=1` 的 MC 仲裁配置。问"能不能用位置网去学 MC1"，
必须先把"MC1 的决定有多少是**观测能预测**的"量出来。数据用 **60 局真实 MC1 对局**的 dump：
  * `data/posroll_c_policy/POSROLL_C/az_seed*_p?.jsonl`：每决策 `cands`（候选束，按序）、`labels`（与 cands 对齐，
    每候选是一串 op `[类, x, y]`）、`chosen`（选中束的下标）、`n`、`player`；
  * `data/posroll_c_obs/POSROLL_C/obs_seed*_p?.bin`：定长记录
    `board(28,19,19) f16 + stats(42,) f16 + player(u8) + chosen(3B) + mask(24B)` = 20,328 B（偏移 +84/+85/+88，见 `ingest_bridge_dump.py`）。

指标（都用**部署口径**的前向：`make_net_fn_from_ckpt` ⇒ `head*_logits` + `action_map`）：
  A. **类维**：`head0` 在 `mask` 允许的类里 argmax == **实际执行的首个类** 的比例（对"非空束"与"空束=HOLD"分别报）；
  B. **位置维**：`action_map[实际执行类]` 里，**实际执行格**的排名分布（1 / ≤3 / ≤10 / ≤33 / >33）
     —— 排名 1 即"位置图的 argmax 恰好就是 MC1 执行的那一格"；
     ⚠️ 这是**下界**：真实合法格掩码未知，实际部署是在该类合法格内取 argmax（本轮只能在全图上取），
        所以真实命中率会 ≥ 这里报的值。
  C. 顺带：`chosen` 束的平均 op 数（束大小）与"空束"比例。

用法：python -u code/my_ai/az_intent/audit_mc1_predictability.py --games 40
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HOLD = 23


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", default="data/posroll_c_policy/POSROLL_C")
    ap.add_argument("--obs-dir", default="data/posroll_c_obs/POSROLL_C")
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--out", default="training_history/posroll2/audit_mc1_predictability.json")
    args = ap.parse_args()

    import torch
    from SDK.utils.constants import HIGHLAND_CELLS
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt

    BOARD_SHAPE = (28, 19, 19)
    N_CLS = 24
    bsz = int(np.prod(BOARD_SHAPE)) * 2          # 只含 board（20,216）；stats 紧随其后 84 B（与 ingest 的偏移一致）
    REC = bsz + 84 + 1 + 3 + N_CLS          # board + stats(84) + player(1) + chosen(3) + mask(24) = 20,328
    # 每个玩家的 highland 掩码（部署的候选格 ⊆ highland ⇒ 在这里排名更贴近真实口径）
    HW = {}
    for _pl in (0, 1):
        _m = np.zeros((19, 19), dtype=bool)
        for (_x, _y) in HIGHLAND_CELLS[_pl]:
            _m[int(_x), int(_y)] = True
        HW[_pl] = _m
    feat = FeatureExtractor(max_actions=96)
    net, _ = make_net_fn_from_ckpt(args.ckpt, feat)

    pol_dir, obs_dir = Path(args.policy_dir), Path(args.obs_dir)
    tokens = sorted({p.name.split("seed")[1].split(".")[0] for p in pol_dir.glob("az_*.jsonl")})
    cls_ok = cls_tot = cls_ok_action = cls_tot_action = 0
    hold_n = 0
    bundle_sizes = collections.Counter()
    rank_hist = collections.Counter()
    rank_hist_hw = collections.Counter()
    n_rows = 0
    for tok in tokens[: args.games]:
        jf = pol_dir / f"az_seed{tok}.jsonl"
        of = obs_dir / f"obs_seed{tok}.bin"
        if not (jf.exists() and of.exists()):
            continue
        recs = []
        for line in open(jf, encoding="utf-8", errors="replace"):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
        raw = np.fromfile(of, dtype=np.uint8)
        n_rec = raw.size // REC
        buf = raw[: n_rec * REC].reshape(n_rec, REC)
        n_use = min(len(recs), n_rec)
        for i in range(n_use):
            d = recs[i]
            board = buf[i, :bsz].copy().view(np.float16).reshape(BOARD_SHAPE)
            stats = buf[i, bsz:bsz + 84].copy().view(np.float16).reshape(42)
            cmask = buf[i, bsz + 88: bsz + 88 + N_CLS].astype(bool)
            ch, labels = d.get("chosen"), d.get("labels")
            bundle = labels[ch] if (isinstance(ch, int) and isinstance(labels, list)
                                    and 0 <= ch < len(labels)) else None
            if not bundle:
                hold_n += 1
                continue
            bundle_sizes[len(bundle)] += 1
            c_e, x_e, y_e = int(bundle[0][0]), int(bundle[0][1]), int(bundle[0][2])
            with torch.no_grad():
                o = net(torch.from_numpy(board.astype(np.float32)).unsqueeze(0),
                        torch.from_numpy(stats.astype(np.float32)).unsqueeze(0))
            lg = o["head1_logits"].squeeze(0).numpy().astype(np.float64)
            lg = np.where(cmask, lg, -1e9)
            cls_tot += 1
            cls_ok += int(int(np.argmax(lg)) == c_e)
            if c_e != HOLD:
                cls_tot_action += 1
                cls_ok_action += int(int(np.argmax(lg)) == c_e)
            am = o["action_map"][0].numpy()
            pl_rec = int(buf[i, bsz + 84])
            if 0 <= c_e < am.shape[0]:
                m = am[c_e]
                v = float(m[x_e, y_e])
                flat = m.ravel()
                rank_hist[min(int((flat > v).sum()) + 1, 34)] += 1          # 全图（含 VOID ⇒ 下界）
                hi = m[HW[pl_rec]]
                rank_hist_hw[min(int((hi > v).sum()) + 1, 40)] += 1         # 只在 highland 里
                n_rows += 1

    def rate(a, b):
        return f"{a}/{b} = {a / max(b, 1) * 100:.2f}%"

    def rh(k):
        return f"{rank_hist.get(k, 0)} = {rank_hist.get(k, 0) / max(n_rows, 1) * 100:.2f}%"

    summary = {
        "tokens_used": len(tokens[: args.games]),
        "rows_with_action": n_rows, "hold_rows": hold_n,
        "bundle_size_hist": dict(sorted(bundle_sizes.items())),
        "class_head0_vs_executed_all": rate(cls_ok, cls_tot),
        "class_head0_vs_executed_action_only": rate(cls_ok_action, cls_tot_action),
        "executed_cell_rank_in_highland": {
            "rank1": f"{rank_hist_hw.get(1, 0) / max(n_rows, 1) * 100:.2f}%",
            "rank<=3": f"{sum(rank_hist_hw.get(k, 0) for k in range(1, 4)) / max(n_rows, 1) * 100:.2f}%",
            "rank<=10": f"{sum(rank_hist_hw.get(k, 0) for k in range(1, 11)) / max(n_rows, 1) * 100:.2f}%",
            "rank<=33": f"{sum(rank_hist_hw.get(k, 0) for k in range(1, 34)) / max(n_rows, 1) * 100:.2f}%",
            "rank>33": f"{rank_hist_hw.get(40, 0) / max(n_rows, 1) * 100:.2f}%"},
        "executed_cell_rank_in_map": {"rank1": rh(1), "rank<=3": f"{(rank_hist.get(1, 0) + rank_hist.get(2, 0) + rank_hist.get(3, 0)) / max(n_rows, 1) * 100:.2f}%",
                                      "rank<=10": f"{sum(rank_hist.get(k, 0) for k in range(1, 11)) / max(n_rows, 1) * 100:.2f}%",
                                      "rank<=33": f"{sum(rank_hist.get(k, 0) for k in range(1, 34)) / max(n_rows, 1) * 100:.2f}%",
                                      "rank>33": f"{rank_hist.get(34, 0) / max(n_rows, 1) * 100:.2f}%"},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== Q1 审计：MC1 的决定有多少是'观测可预测'的 =====")
    print(f"  用了 {summary['tokens_used']} 个对局文件；非空束行 {n_rows}，空束(HOLD)行 {hold_n}")
    print(f"  束大小分布：{summary['bundle_size_hist']}")
    print(f"  **类维**（head0 掩码内 argmax == 实际执行类）：全部 {summary['class_head0_vs_executed_all']}；"
          f"仅非 HOLD 动作 {summary['class_head0_vs_executed_action_only']}")
    print(f"  **位置维**（实际执行格在 `action_map[执行类]` 里的排名，1 = 位置图 argmax 就是它）：")
    print("      全图（下界）:")
    for k, v in summary["executed_cell_rank_in_map"].items():
        print(f"        {k:9s} {v}")
    print("      只在 highland 格内（贴近部署口径）:")
    for k, v in summary["executed_cell_rank_in_highland"].items():
        print(f"        {k:9s} {v}")
    print("  ⚠️ 位置维是**下界**：真实部署在该类**合法格掩码内**取 argmax，本轮只能在**全图**上取。")
    print(f"  摘要：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
