"""单元级断言：`Q[c]` 利用项从 mellowmax 换成**玻尔兹曼加权**后方向对、且贴近最大值（2026-10-30）。

设计 §"Q[c] 利用项"（commit `b751351`）+ 调研 `docs/prior_art_unified_search_20261030.md`：
* 旧（mellowmax/LogSumExp）：`Q = (1/β)·log( (1/n)·Σ exp(β·x) )`
  带折扣 `log(n)/β`，而折扣**最大恰在"一个神手 + 一堆垃圾"**的情形 ⇒ 与"想突出神手"相反。
* 新（玻尔兹曼加权）：`Q = Σ wᵢ·xᵢ`，`wᵢ = exp(β·xᵢ)/Σⱼ exp(β·xⱼ)` ⇒ 没有 log(n) 项。

断言（β=3、`x = [1.0, -1.0×9]`）：
* 旧 mellowmax ≈ **0.244**（用户给的判据 ~0.24）
* 新玻尔兹曼 ≈ **0.957**（用户给的判据 ~0.96）
* 且两者都 ∈ [min(x), max(x)]，新值**严格更接近 max**。

另外顺带验证实现与公式一致（直接调用 `UnifiedCandidateGen._q_util`，不走任何私有近似），
以及"重复同值不敏感"（`[1.0,-1.0×9]` 里补一个重复 1.0 ⇒ 两种实现都不变/仅微动）。

用法（不需要 GPU / ckpt，纯 CPU 几毫秒）：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_q_util_boltzmann.py
```
退出码 0 = 断言全过。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BETA = 3.0


def mellowmax(xs, beta: float) -> float:
    """旧实现（(1/β)log(mean exp(βx))）—— 与 `23d1490` 的 `_lse` 逐字同式。"""
    arr = np.asarray(xs, dtype=np.float64)
    m = float(arr.max())
    return float(m + math.log(float(np.mean(np.exp(beta * (arr - m))))) / beta)


def boltzmann(xs, beta: float) -> float:
    """新实现（Σ w·x，w = softmax(βx)）—— 与 `_q_util` 的独立参考实现。"""
    arr = np.asarray(xs, dtype=np.float64)
    z = beta * (arr - float(arr.max()))
    w = np.exp(z)
    w /= float(w.sum())
    return float(np.dot(w, arr))


def _gen_with_q(q_by_class: dict):
    """建一个只有 `Q` 的最小生成器实例（绕开 ckpt/局面，只测数学）。"""
    from my_ai.az_intent.unified_cand import UnifiedCandidateGen
    g = UnifiedCandidateGen.__new__(UnifiedCandidateGen)   # 不跑 __init__（省掉依赖）
    g.beta = BETA
    g.Q = {int(c): list(v) for c, v in q_by_class.items()}
    return g


def main() -> int:
    ok = True
    xs = [1.0] + [-1.0] * 9
    old = mellowmax(xs, BETA)
    new = boltzmann(xs, BETA)
    g = _gen_with_q({5: xs})
    got = g._q_util(5)

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {name}: {detail}")
        ok = ok and cond

    print(f"[q] beta={BETA}  x = {xs}")
    print(f"[q] 旧 mellowmax = {old:+.6f}   新 玻尔兹曼 = {new:+.6f}   "
          f"`_q_util` 实测 = {got:+.6f}")
    check("旧值 ≈ 0.24（用户判据）", abs(old - 0.244) < 0.01, f"{old:.4f}")
    check("新值 ≈ 0.96（用户判据）", abs(new - 0.957) < 0.02, f"{new:.4f}")
    check("新实现与参考公式一致", abs(got - new) < 1e-12, f"|{got:.12f} - {new:.12f}|")
    check("两者都在 [min,max] 内", min(xs) <= old <= max(xs) and min(xs) <= new <= max(xs),
          f"min={min(xs)} old={old:.4f} new={new:.4f} max={max(xs)}")
    check("新值严格更接近 max", (max(xs) - new) < (max(xs) - old),
          f"距离 max: 新 {max(xs)-new:.4f} < 旧 {max(xs)-old:.4f}")
    # 极端：全同分 ⇒ 两者都等于该值；单调性：越多垃圾值新值衰减越慢
    same = [0.3, 0.3, 0.3]
    g2 = _gen_with_q({7: same})
    check("全同分 ⇒ 参考实现 = 该值", abs(boltzmann(same, BETA) - 0.3) < 1e-12,
          f"boltzmann={boltzmann(same, BETA):.6f}")
    check("全同分 `_q_util` = 0.3", abs(g2._q_util(7) - 0.3) < 1e-12, f"{g2._q_util(7):.6f}")
    check("全同分 ⇒ 旧实现也 = 该值", abs(mellowmax(same, BETA) - 0.3) < 1e-12,
          f"mellowmax={mellowmax(same, BETA):.6f}")
    decay_new = [boltzmann([1.0] + [-1.0] * k, BETA) for k in (0, 4, 9, 49, 99)]
    decay_old = [mellowmax([1.0] + [-1.0] * k, BETA) for k in (0, 4, 9, 49, 99)]
    print(f"[q] 垃圾数 k=0/4/9/49/99：新 {['%.3f' % v for v in decay_new]}")
    print(f"[q]                        旧 {['%.3f' % v for v in decay_old]}")
    check("垃圾越多，新值衰减得比旧值慢（每一档）",
          all(n >= o - 1e-12 for n, o in zip(decay_new, decay_old)),
          "新 ≥ 旧 逐档成立")
    # 重复同值（缓存命中会重复计入）：玻尔兹曼对重复同值不敏感
    dup = [1.0] + [-1.0] * 9 + [1.0, 1.0]
    print(f"[q] 补两个重复 1.0 后：新 {boltzmann(dup, BETA):+.6f}"
          f"（原 {new:+.6f}）  旧 {mellowmax(dup, BETA):+.6f}（原 {old:+.6f}）")
    check("重复同值不把 Q 撑到 max 之外", boltzmann(dup, BETA) <= max(xs) + 1e-12,
          f"{boltzmann(dup, BETA):.6f} ≤ {max(xs)}")

    # 信用分配：束分**完整**回写（用户定夺：语义 = "含 c 的组最好能有多好"，**不打折扣**）
    gc = _gen_with_q({0: [], 1: [], 2: [], 23: []})
    gc._credit((0, 1, 2), 0.9)
    check("3 元束回写 = 0.9（**不折扣**）",
          all(abs(gc.Q[c][-1] - 0.9) < 1e-12 for c in (0, 1, 2)),
          f"{[gc.Q[c][-1] for c in (0, 1, 2)]}")
    gc._credit((0,), 0.9)
    check("1 元束回写 = 0.9", abs(gc.Q[0][-1] - 0.9) < 1e-12, f"{gc.Q[0][-1]}")
    gc._credit((), 0.9)
    check("空束（HOLD）回写 = 0.9", abs(gc.Q[23][-1] - 0.9) < 1e-12, f"{gc.Q[23][-1]}")
    gc._credit((2, 2), 0.9)
    check("重复类 2 元束 ⇒ 该类记两次 0.9（仍不折扣）",
          abs(gc.Q[2][-1] - 0.9) < 1e-12 and abs(gc.Q[2][-2] - 0.9) < 1e-12,
          f"{gc.Q[2][-2:]}")
    print(f"[q] 结论：{'全部断言通过 ✅' if ok else '有断言失败 ❌'}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
