#!/usr/bin/env python3
"""验证 `AZAI_VALUE_CKPT` 开关的**两种**语义（默认关 / 打开），并覆盖**两处消费点**。

判据（对应任务 B）：
  1. **空值 ⇒ 零影响**：`maybe_override(model) is model`（同一个对象）、`value_override._CACHE` 为空
     （一个额外模型都没加载）、`value_of(dict)` 返回的**就是**原来那个张量；两个消费点
     （`flat_rollout_greedy._leaf_values` / `value_batch.make_batched_value_fn`）的取值路径逐位不变；
  2. **非空 ⇒ 真的换了**：返回适配器、输出形态仍是 `{"value": ...}`（三网形态）、
     `next(model.parameters()).is_cuda` 可用、`tanh` 后与老师的 Spearman ≈0.96、
     **两处消费点都能跑通并给出同量纲的分**。

⚠️ 这个脚本只证明"开关语义 + 消费点兼容"，**不**证明搜索行为不变 ——
后者由 `probe_oracle_inject_zero_impact.py` 的哈希比对负责（见报告 §B）。

用法：
  python code/test_match/probe_value_override_switch.py \
      --folded checkpoints/small32_3/v_32_3_folded.pt --tag VS_RU_dump --seed 8
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.az_intent import value_override as vo                          # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models                # noqa: E402
from my_ai.az_intent.value_batch import make_batched_value_fn            # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade            # noqa: E402
from SDK.utils.features import FeatureExtractor                          # noqa: E402


def spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _load_rsc():
    spec = importlib.util.spec_from_file_location("_repro_search_crash",
                                                  REPO / "code/test_match/repro_search_crash.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_repro_search_crash"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--folded", default="checkpoints/small32_3/v_32_3_folded.pt")
    ap.add_argument("--data", default="training_history/distill_top2/lowq_f64")
    ap.add_argument("--n-frames", type=int, default=256)
    ap.add_argument("--tag", default="VS_RU_dump")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[120, 200])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))

    def _abs(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (REPO / q)

    ckpt, folded, data = _abs(a.ckpt), _abs(a.folded), _abs(a.data)
    feat = FeatureExtractor(max_actions=96)
    _c, _p, teacher = load_three_models(str(ckpt))
    teacher.eval()
    fail = 0

    # ── 情形 1：开关为空 ⇒ 零影响 ────────────────────────────────────────────
    os.environ.pop("AZAI_VALUE_CKPT", None)
    print("=" * 78)
    print(f"[1] 开关为空（AZAI_VALUE_CKPT='{vo.env_path()}'）")
    same = vo.maybe_override(teacher)
    print(f"    maybe_override(teacher) is teacher : {same is teacher}")
    print(f"    已加载的覆盖网个数 (_CACHE)        : {len(vo._CACHE)}   （应为 0）")
    d = {"value": torch.zeros(3, 1)}
    print(f"    value_of(dict) is d['value']       : {vo.value_of(d) is d['value']}")
    t = torch.zeros(4)
    print(f"    value_of(裸张量) is 原张量          : {vo.value_of(t) is t}")
    if not (same is teacher and len(vo._CACHE) == 0
            and vo.value_of(d) is d["value"] and vo.value_of(t) is t):
        print("    ❌ 空值路径不是零影响")
        fail = 1
    else:
        print("    ✅ 空值：同一对象、不加载任何模型、取值逐位不变")

    # ── 情形 2：开关指向折叠后的小值网 ───────────────────────────────────────
    os.environ["AZAI_VALUE_CKPT"] = str(folded)
    print("=" * 78)
    print(f"[2] 开关打开（AZAI_VALUE_CKPT={folded.name}）")
    ov = vo.maybe_override(teacher)
    print(f"    返回类型                            : {type(ov).__name__}")
    print(f"    is_value_override                   : {getattr(ov, 'is_value_override', False)}")
    print(f"    describe()                          : {vo.describe(ov)}")
    print(f"    next(parameters()).is_cuda          : {next(ov.parameters()).is_cuda}")
    print(f"    maybe_override 二次调用同一对象      : {vo.maybe_override(teacher) is ov}"
          f"   （缓存命中，不重复加载）")
    if ov is teacher or not getattr(ov, "is_value_override", False):
        print("    ❌ 开关没有生效")
        fail = 1

    # ── 情形 2a：`_leaf_values` 那条路（直接吃 {"value": ...}）───────────────
    if (data / "game_offsets.npy").exists():
        off = np.load(data / "game_offsets.npy")
    else:
        gid = np.load(data / "game_id.npy")
        off = np.searchsorted(gid, np.arange(int(gid[-1]) + 2))
    n_games = len(off) - 1
    lo = int(off[max(0, n_games - 500)])
    hi = int(off[-1])
    rng = np.random.default_rng(7)
    n = min(a.n_frames, hi - lo)
    idx = np.sort(rng.choice(np.arange(lo, hi), size=n, replace=False))
    board = np.load(data / "board.npy", mmap_mode="r")
    stats = np.load(data / "stats.npy", mmap_mode="r")
    b = torch.from_numpy(np.array(board[idx], np.float32))
    st = torch.from_numpy(np.array(stats[idx], np.float32))
    with torch.no_grad():
        to = teacher(b, st)
        t_out = vo.value_of(to).reshape(-1).numpy()
        s_out = vo.value_of(ov(b, st)).reshape(-1).numpy()
    print(f"    输出形态：teacher dict={isinstance(to, dict)} "
          f"| override dict={isinstance(ov(b, st), dict)} （两者同形态 ⇒ 消费点不用改）")
    print(f"    老师 tanh 范围 [{np.tanh(t_out).min():.3f},{np.tanh(t_out).max():.3f}] "
          f"std={np.tanh(t_out).std():.4f}")
    print(f"    小网 tanh 范围 [{np.tanh(s_out).min():.3f},{np.tanh(s_out).max():.3f}] "
          f"std={np.tanh(s_out).std():.4f}   （同量纲 ⇒ std 应同量级）")
    rho = spearman(np.tanh(s_out), np.tanh(t_out))
    print(f"    Spearman ρ(tanh) = {rho:.4f}   （应 ≈0.96）")
    if not (0.90 <= rho <= 0.99):
        print("    ❌ ρ 不在 0.90–0.99（量纲或折叠有问题）")
        fail = 1

    # ── 情形 2b：`make_batched_value_fn` 那条路（真实局面，含终局处理）────────
    rsc = _load_rsc()
    rounds_ops = rsc.read_ops(a.tag, f"{a.seed}.log")
    states = []
    stt = GameStateFacade.initial(seed=a.seed, cold_handle_rule_illegal=True)
    want = set(int(x) for x in a.rounds)
    for r in range(0, max(want) + 1):
        if r in want:
            states.append(stt.clone())
        for pl in (0, 1):
            ops = rounds_ops.get(r, {}).get(pl, [])
            if ops:
                from SDK.backend.model import Operation
                from SDK.utils.constants import OperationType
                stt.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                              for (t, a0, a1) in ops])
        stt.advance_round()
    fn_t = make_batched_value_fn(teacher, feat, batch_size=32, tanh=True)
    fn_s = make_batched_value_fn(ov, feat, batch_size=32, tanh=True)
    v_t = fn_t(states, 0)
    v_s = fn_s(states, 0)
    print(f"    make_batched_value_fn：{len(states)} 个真实局面（回合 {a.rounds}）")
    for i, (x, y) in enumerate(zip(v_t, v_s)):
        print(f"      #{i}: 老师={x:+.4f}  小网={y:+.4f}  Δ={y - x:+.4f}")
    print(f"    两者的 Spearman（同量纲检查）      : {spearman(v_s, v_t):.4f}")
    if not np.all(np.isfinite(v_s)) or not np.all(np.isfinite(v_t)):
        print("    ❌ 有非有限值")
        fail = 1

    os.environ.pop("AZAI_VALUE_CKPT", None)
    print("=" * 78)
    print("✅ 开关语义与两处消费点都通过" if fail == 0 else "❌ 有判据未通过")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
