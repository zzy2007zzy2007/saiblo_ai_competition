"""审计：**位置头（pos_model 的 action_map）** 与 **教师（搜索/MC 的 intent 分布）** 的落点一致性。

用途：这是**路线 A 的 go/no-go 闸门**（见 `docs/prereg_20261010_actionnet_retrain.md`）——
在花几小时采集之前，先用**现成的 dump** 问一句：我们的位置头离教师还有多远？
- 若已经很近（top-1 一致率高 / 教师 argmax 在我方分布里排名第 1 / KL 小）⇒ **没有头寸，直接停**；
- 若明显有距离（top-1 一致率低、排名靠后、KL 大）⇒ 值得采集 + 蒸馏。

口径（与历史 `train_value_prior.py` / `train_pos_distill.py` 一致，便于对比）：
- 教师分布 p = `intent`（24 候选 × 3 slot 的 `{"[类,x,y]": 计数}`）在**落点**上归一到合法格；
- 我方分布 q = `pos_model(board, stats)["action_map"][pinned_class]` 在**同一批落点**上 z-score 后 softmax；
- 指标：top-1 一致率、教师 argmax 在我方分布中的**排名**、KL(p‖q)、教师分布的归一化熵（目标本身有多"尖"/多可学）。

用法：
  python -u code/test_match/audit_pos_head_vs_teacher.py \
    --policy-dir data/s2_policy/COLLECT_S2 --obs-dir data/s2_obs/COLLECT_S2 \
    --ckpt checkpoints/grafted_sdk_A.pt --max-decisions 3000 --device cpu
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

EPS = 1e-9


def load_teacher(intent=None, cands=None, labels=None, visits=None, mode="visits"):
    """教师分布。

    ⚠️ 语义以桥接端 `code/test_match/az_bridge_ai.py:183-213` 为准（2026-10-10 核对）：
      - `cands`  = **候选束列表**（每束 ≤3 个 op）
      - `visits` = **与 cands 同序的访问分布**  ← 这才是"搜索的选择"
      - `chosen` = **cands 的下标**
      - `labels` = **每个 op 的 (类,x,y)**
      - `intent` = **采样期的 intent 计数**（3 个头）—— **不是**搜索的最终偏好！
    第一版审计误用 `intent` 当教师 ⇒ 自检"chosen == 教师 argmax" 0/2000 即由此而来。

    返回 {(cls,x,y): weight}。
    """
    agg = collections.Counter()
    if mode == "intent" and intent:
        for cand in intent:
            if not isinstance(cand, list):
                continue
            for slot in cand:
                if not isinstance(slot, dict):
                    continue
                for k, c in slot.items():
                    try:
                        kk = json.loads(k) if isinstance(k, str) else k
                        cls, x, y = (int(z) for z in kk)
                    except Exception:
                        continue
                    if x >= 0 and y >= 0:
                        agg[(cls, x, y)] += max(1, int(c))
        return agg

    # mode == "visits"：按候选束的访问量，把束里各 op 的 label(类,x,y) 累加
    if not (cands and labels and visits):
        return agg
    for b_idx, bundle in enumerate(cands):
        if not isinstance(bundle, list):
            continue
        try:
            w = float(visits[b_idx])
        except Exception:
            continue
        if w <= 0:
            continue
        for o_idx in range(min(len(bundle), len(labels[b_idx]) if b_idx < len(labels) else 0)):
            try:
                cls, x, y = (int(z) for z in labels[b_idx][o_idx])
            except Exception:
                continue
            if x >= 0 and y >= 0:
                agg[(cls, x, y)] += w
    return agg


def zsoftmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 1:
        return np.ones_like(v)
    z = (v - v.mean()) / (v.std() + EPS)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + EPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--show-examples", type=int, default=2)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(int(args.threads))
    from my_ai.az_intent.az_selfplay import load_three_models
    from distill.convert_az_dump import read_obs

    dev = torch.device(args.device)
    print(f"[audit] ckpt = {args.ckpt}；device = {dev}", flush=True)
    class_model, pos_model, value_model = load_three_models(args.ckpt)
    pos_model = pos_model.to(dev).eval()
    print("[audit] 已加载 (class, pos, value) 三个模型；用 pos_model 出 action_map", flush=True)

    import re

    pol_files = sorted(glob.glob(os.path.join(args.policy_dir, "*.jsonl")))
    if not pol_files:
        print(f"[audit] ⛔ {args.policy_dir} 下没有 jsonl")
        return 2
    # ⚠️ 实测：jsonl 名前缀是 `az_`、obs 名前缀是 `obs_`（**前缀不同**）⇒ 只能按
    #    `seed<五位>_p<k>` 这个**后缀 token** 配对（2026-10-10 第一版按 stem 配对 ⇒ 全部"缺 obs"）
    def tok(name: str):
        m = re.search(r"(seed\d+_p\d+)", name)
        return m.group(1) if m else None
    obs_by_tok = {}
    for f in glob.glob(os.path.join(args.obs_dir, "*.bin")):
        t = tok(Path(f).name)
        if t:
            obs_by_tok[t] = f
    print(f"[audit] policy 文件 {len(pol_files)} 个；obs 建立索引 {len(obs_by_tok)} 个", flush=True)

    rows = []          # (top1_agree, rank, kl, mass_on_ours, ent, n_cells, cls)
    chosen_match = 0
    chosen_seen = 0
    examples = []
    done = 0
    n_hist = collections.Counter()      # 每个 `n` 出现几次（判断一决策几条记录）
    used_intent = 0                      # 回退到 intent 标签的决策数（应为 0）
    multi_cls = 0                        # 候选中混了多个类的决策数（pos-only 应为 0）

    missing = 0
    for pf in pol_files:
        stem = Path(pf).stem
        t = tok(Path(pf).name)
        obs_path = Path(obs_by_tok[t]) if (t and t in obs_by_tok) else None
        if obs_path is None:
            missing += 1
            if missing <= 3:
                print(f"[audit] ⚠️ 缺 obs：token={t}")
            continue
        # ⚠️ 被硬杀的局会留**半条 JSONL**（实测 `Extra data`）⇒ 必须逐行容错，并
        #    **用 None 占位保持与 obs 的行对齐**（不能过滤掉，否则后面全部错位）
        recs = []
        n_bad = 0
        for l in open(pf, encoding="utf-8"):
            if not l.strip():
                recs.append(None)
                continue
            try:
                recs.append(json.loads(l))
            except Exception:
                recs.append(None)
                n_bad += 1
        if n_bad:
            print(f"[audit] ⚠️ {stem}: 有 {n_bad} 行 JSONL 坏行 ⇒ 跳过该行对应的决策（保持对齐）", flush=True)
        obs = read_obs(obs_path)
        n = min(len(recs), len(obs))
        if len(recs) != len(obs):
            print(f"[audit] ⚠️ {stem}: jsonl={len(recs)} obs={len(obs)} ⇒ 取 min", flush=True)

        for i in range(n):
            if done >= args.max_decisions:
                break
            r, o = recs[i], obs[i]
            if r is None:
                continue
            n_hist[int(r.get("n", -1))] += 1
            # ✅ 正确标签：`visits`（与 cands 同序）按束内 op 的 label(类,x,y) 累加
            agg = load_teacher(cands=r.get("cands"), labels=r.get("labels"),
                               visits=r.get("visits"))
            if not agg:
                agg = load_teacher(intent=r.get("intent"), mode="intent")
                used_intent += 1
            # 钉类纯度：取**权重最大**的类，并统计是否单类
            by_cls = collections.Counter()
            for (c, _, _), w in agg.items():
                by_cls[c] += w
            if not by_cls:
                continue
            c_pin = by_cls.most_common(1)[0][0]
            purity = by_cls[c_pin] / max(1e-9, sum(by_cls.values()))
            if purity < 0.999:
                multi_cls += 1
            cells = sorted({(x, y) for (c, x, y) in agg if c == c_pin})
            if len(cells) < 2:
                continue
            p_counts = np.array([float(agg.get((c_pin, x, y), 0.0)) for (x, y) in cells])
            if p_counts.sum() <= 0:
                continue
            p = p_counts / p_counts.sum()

            board = np.asarray(o["board"], dtype=np.float32)
            stats = np.asarray(o["stats"], dtype=np.float32)
            with torch.no_grad():
                out = pos_model(torch.as_tensor(board, device=dev).unsqueeze(0),
                                torch.as_tensor(stats, device=dev).unsqueeze(0))
            am = out["action_map"]
            am = am[0] if am.dim() == 4 else am
            am = am.detach().cpu().numpy()
            vals = np.array([float(am[c_pin, y, x]) for (x, y) in cells])
            q = zsoftmax(vals)

            p_top = int(np.argmax(p))
            q_top = int(np.argmax(q))
            order = np.argsort(-q)
            rank = int(np.where(order == p_top)[0][0]) + 1
            kl = float(np.sum(p * np.log((p + EPS) / (q + EPS))))
            mass_ours = float(p[q_top])
            ent = float(-np.sum(p * np.log(p + EPS)) / np.log(len(p)))
            rows.append((int(p_top == q_top), rank, kl, mass_ours, ent, len(cells), c_pin))

            # 交叉检查（正确版）：jsonl 的 `chosen` 是 **cands 的下标** ⇒ 看它那一束的 label 格
            #   是否包含教师的 argmax 格（同一束内多 op 时，含任一格即算命中）
            chosen_idx = r.get("chosen")
            try:
                chosen_idx = int(chosen_idx)
            except Exception:
                chosen_idx = -1
            if 0 <= chosen_idx < len(r.get("cands") or []):
                chosen_seen += 1
                lab = r.get("labels") or []
                ccells = set()
                if chosen_idx < len(lab):
                    for z in lab[chosen_idx]:
                        try:
                            _c, _x, _y = (int(v) for v in z)
                            if _x >= 0 and _y >= 0:
                                ccells.add((_x, _y))
                        except Exception:
                            pass
                if cells[p_top] in ccells:
                    chosen_match += 1
            if len(examples) < args.show_examples:
                examples.append((stem, i, c_pin, cells, p, q, rank, kl, ent))
            done += 1
        if done >= args.max_decisions:
            break
        print(f"[audit] {stem}: 累计可用于审计的决策 = {done}", flush=True)

    if not rows:
        print("[audit] ⛔ 没有找到任何'≥2 个落点'的决策")
        return 3

    agree = np.array([r[0] for r in rows])
    ranks = np.array([r[1] for r in rows])
    kls = np.array([r[2] for r in rows])
    mass = np.array([r[3] for r in rows])
    ent = np.array([r[4] for r in rows])
    ncell = np.array([r[5] for r in rows])
    cls_hist = collections.Counter([r[6] for r in rows])

    print("\n===== 审计结果 =====")
    print(f"  可用决策数 = {len(rows)}（来自 {len(pol_files)} 局；每决策落点数中位 = {int(np.median(ncell))}）")
    print(f"  **top-1 一致率 = {agree.mean() * 100:.1f}%**（教师 argmax == 我方 argmax）")
    print(f"  教师 argmax 在我方分布中的排名：中位 {int(np.median(ranks))}，均值 {ranks.mean():.2f}，"
          f"p25={int(np.percentile(ranks, 25))}，p75={int(np.percentile(ranks, 75))}")
    print(f"  排名第 1 占比 = {(ranks == 1).mean() * 100:.1f}%；排名 ≤3 占比 = {(ranks <= 3).mean() * 100:.1f}%")
    print(f"  KL(教师‖我方)：中位 {np.median(kls):.3f}，均值 {kls.mean():.3f}")
    print(f"  教师分布在我方 argmax 上的质量：中位 {np.median(mass):.3f}")
    print(f"  教师分布归一化熵：中位 {np.median(ent):.3f}（越小越尖、越好学）")
    print(f"  钉住的类分布：{cls_hist.most_common(6)}")
    print(f"  ⚠️ 回退到 intent 标签的决策 = {used_intent}（应为 0）；候选中混多类的决策 = {multi_cls}（pos-only 应为 0）")
    print(f"  记录数 / 不同 `n` 数 = {sum(n_hist.values())} / {len(n_hist)}（>1 说明一次决策写多条记录）"
          f"，每 n 出现次数 top：{n_hist.most_common(4)}")
    if chosen_seen:
        print(f"  bridge 选中束含教师 argmax 的比例 = {chosen_match}/{chosen_seen} = "
              f"{chosen_match / chosen_seen * 100:.1f}%（标签一致性自检，应较高）")

    if args.show_examples:
        print("\n----- 例子（p=教师, q=我方）-----")
        for (stem, i, c_pin, cells, p, q, rank, kl, ent) in examples:
            print(f"  {stem} 决策#{i}  钉类={c_pin}  落点数={len(cells)}  排名={rank}  KL={kl:.3f}")
            top = np.argsort(-p)[:5]
            print("     教师 p: " + ", ".join(f"{cells[j]}:{p[j]:.2f}" for j in top))
            topq = np.argsort(-q)[:5]
            print("     我方 q: " + ", ".join(f"{cells[j]}:{q[j]:.2f}" for j in topq))

    # 闸门判读（先写死）
    hi_agree = agree.mean() >= 0.50 or (ranks == 1).mean() >= 0.60
    print("\n===== 闸门判读（预注册口径）=====")
    if hi_agree and np.median(kls) < 0.5:
        print("  ⇒ 🔴 位置头**已贴近教师**（一致率高 / KL 小）⇒ **路线 A 的头寸很小，建议不采集、直接转路线 C/其它**")
    elif agree.mean() < 0.30 or np.median(ranks) >= 4:
        print("  ⇒ 🟢 位置头离教师**明显有距离** ⇒ **值得采集 + 蒸馏**（路线 A 继续）")
    else:
        print("  ⇒ 🟡 中间地带：有距离但不悬殊 ⇒ 先小批量采集（~100 局）试蒸馏，看'目标质量'能否显著改善再决定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
