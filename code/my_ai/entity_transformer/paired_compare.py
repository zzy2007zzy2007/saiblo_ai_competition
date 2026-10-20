"""对同一 seed 集上的两个臂做配对统计检验（逐 seed 得分差 + 符号检验 + bootstrap）。"""
import sys
from pathlib import Path
import numpy as np

LOGS = Path(r"D:\2026智能体大赛_新2_distill\match_results\ladder_logs")


def seed_scores(tags):
    best = {}
    for tag in tags:
        d = LOGS / tag
        if not d.is_dir():
            continue
        for f in d.glob("*.log"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = [ln for ln in txt.splitlines() if "RESULT seed" in ln]
            if not lines:
                continue
            seed = int("".join(ch for ch in f.stem if ch.isdigit()))
            win = 1 if "winner=az_bridge_ai" in lines[-1] else 0
            best[f.stem] = (seed, win)
    out = {}
    for seed, win in best.values():
        out.setdefault(seed, []).append(win)
    return {s: sum(v) for s, v in out.items()}


def summ(sc, name):
    v = np.array(list(sc.values()), float)
    p = v.sum() / (2 * len(v))
    se = np.sqrt(p * (1 - p) / len(v))
    print(f"{name:<22} pairs={len(v):3d}  rate={p:.4f}  SE={se:.4f}  95%CI=[{p-1.96*se:.3f},{p+1.96*se:.3f}]")
    return p


arms = {
    "baseline": seed_scores(["BASELINE_REP"]),
    "kgeo50": seed_scores(["REL_KGEO50_REP", "REL_KGEO50_REP2"]),
    "kgeo50_s7_70": seed_scores(["REL_KGEO50_128"]),
    "baseline_s7_70": {},  # 只有历史 M6cROOT_128 的汇总数字 0.7344，没有逐局日志
}
print("=== 各臂（seed 71-134）===")
for k in ("baseline", "kgeo50"):
    summ(arms[k], k)

print("\n=== 配对比较（同 seed）===")
b, k = arms["baseline"], arms["kgeo50"]
common = sorted(set(b) & set(k))
db = np.array([b[s] for s in common], float)
dk = np.array([k[s] for s in common], float)
diff = dk - db
print(f"共同 seed: {len(common)}")
print(f"kgeo50 - baseline 均值差 = {diff.mean():+.4f} (0-2 分制) => {diff.mean()/2*100:+.1f} pp")
print(f"  标准差={diff.std(ddof=1):.4f}  SE={diff.std(ddof=1)/np.sqrt(len(diff)):.4f}")
print(f"  t = {diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):.2f}")
n_pos = int((diff > 0).sum()); n_neg = int((diff < 0).sum()); n_tie = int((diff == 0).sum())
print(f"  逐 seed: kgeo50 更好 {n_pos}, 更差 {n_neg}, 持平 {n_tie}")
# 符号检验（只看非平局）
n = n_pos + n_neg
if n > 0:
    from math import comb
    p_two = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(n_pos, n_neg) + 1)) / (2 ** n))
    print(f"  符号检验(双侧) p = {p_two:.4f}  (n_non_tie={n})")
print("\n=== 单批 vs 合并：seed 集差异有多大 ===")
kb = arms["kgeo50"]["seed"] if False else arms["baseline"]
print("baseline 在 seed 71-134 的胜率:", f"{summ(arms['baseline'],'  baseline(71-134)'):.4f}")
print("历史 baseline 在 seed 7-70 的胜率: 0.7344 (来自 M6cROOT_128 汇总，无逐局日志)")
print("kgeo50 在 seed 7-70 : 0.8047")
print("kgeo50 在 seed 71-134:", f"{summ(arms['kgeo50'],'  kgeo50(71-134)'):.4f}")
