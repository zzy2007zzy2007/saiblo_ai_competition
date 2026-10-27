"""**零影响**探针：证明"引擎 RNG 消费"开关默认关闭时，`FlatRolloutGreedy` 的搜索
结果**逐位不变**（2026-10-27）。

## 做法

同一进程里同时装两套 `FlatRolloutGreedy`：
- **before** = 改动前的工作区版本（用 `--before` 指定的文件；默认从 git 取
  `git show <rev>:code/my_ai/az_intent/flat_rollout_greedy.py` 到临时文件）；
- **after** = 当前安装的 `my_ai.az_intent.flat_rollout_greedy`。

两边用**同一个 net_fn / value_model / feature_extractor / 同一串快照 / 同一个
rollout_seed**，逐项打哈希：`last_scores`（float64 原始字节）、`visit_policy`、
`chosen_index`、`chosen_bundle`、`last_rollouts`、`last_rounds`、
`last_effective_samples`、`last_tree_stats`（节点/叶子/去重叶子）、`last_rng_consumed`。

覆盖三条路径（都要逐位一致）：
- `idle` + `share_prefix=1`（不开随机，走朴素路径）
- `random` + `share_prefix=1`（前缀树路径 ← 本次改得最多的一条）
- `random` + `share_prefix=0`（朴素路径 + 随机出招）

## 用法

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P code/test_match/probe_rng_perturb_zero_impact.py --before-rev HEAD \\
    --out code/cpp_engine/build_probe/rng_impl/zero_impact.log
```
（`--before-rev` 要在**提交前**用，提交后请改用 `--before 路径` 指向旧文件。）
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"      # 让 last_tree_stats 被记录
os.environ.setdefault("AZAI_TORCH_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game",
           REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402


def digest(x) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def load_module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def materialize_before(args) -> Path:
    if args.before:
        return Path(args.before)
    # ⚠️ before 文件必须放在 `code/my_ai/az_intent/` 下：该模块用
    # `Path(__file__).parents[3]` 推仓库根，放到 C:\Temp 会因为层级不够而在 import
    # 时抛 IndexError（不是 modulo 的错，是探针的取巧）。用完即删。
    out = REPO / "code" / "my_ai" / "az_intent" / "_before_rng_tmp.py"
    src = subprocess.check_output(
        ["git", "show", f"{args.before_rev}:code/my_ai/az_intent/flat_rollout_greedy.py"],
        cwd=str(REPO))
    out.write_bytes(src)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="", help="改动前的 flat_rollout_greedy.py 路径")
    ap.add_argument("--before-rev", default="HEAD", help="不传 --before 时从该 git rev 取文件")
    ap.add_argument("--log", default="match_results/ladder_logs/VS_RUNNERUP_cells4/10.log")
    ap.add_argument("--rounds", type=int, nargs="+", default=[40, 200])
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    before_path = materialize_before(args)
    emit(f"[zero] before 文件 = {before_path}")
    try:
        mod_before = load_module_from(before_path, "_before_rng_tmp")
    finally:
        if before_path.name == "_before_rng_tmp.py":
            before_path.unlink(missing_ok=True)
    from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy as After
    emit(f"[zero] after 模块 = {sys.modules['my_ai.az_intent.flat_rollout_greedy'].__file__}")

    # ── 模型（两边共用同一份实例，杜绝"两次加载权重不同"的伪差异）──
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
    from SDK.utils.features import FeatureExtractor
    feat = FeatureExtractor(max_actions=96)
    ckpt = str(REPO / os.environ.get("AZAI_CKPT", "checkpoints/posnet_A_k5_m32.pt"))
    _, net_fn = make_net_fn_from_ckpt(ckpt, feat, value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(ckpt)
    vm.eval()
    emit(f"[zero] ckpt={ckpt}")

    # ── 快照：用真实对局日志重放出来的局面（p0/p1 两个座位）──
    from probe_replay_engines import parse_log, parse_ops_text
    from SDK.backend.engine import Operation, OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    info = parse_log(REPO / args.log)
    rows = {r: {p: parse_ops_text(per.get(p, "0\n"), Operation, OperationType)
                for p in (0, 1)} for r, per in info["ops_text"].items()}
    want = set(int(x) for x in args.rounds)
    snaps = []
    st = GameStateFacade.initial(seed=info["seed"], cold_handle_rule_illegal=True)
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, 0, st.clone()))
        for pl in (0, 1):
            ops = rows.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, ops)
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))
        st.advance_round()
    emit(f"[zero] 快照 {len(snaps)} 个：seed={info['seed']} 回合={sorted(want)}")

    CONFIGS = [
        ("idle+prefix", dict(rollout_mode="idle", rollout_share_prefix=True)),
        ("random+prefix", dict(rollout_mode="random", rollout_share_prefix=True,
                               rollout_prob_mode="decay", rollout_decay_c=0.3,
                               rollout_decay_k=1.0)),
        ("random+naive", dict(rollout_mode="random", rollout_share_prefix=False,
                              rollout_prob_mode="decay", rollout_decay_c=0.3,
                              rollout_decay_k=1.0)),
    ]

    bad = 0
    for tag, cfg in CONFIGS:
        kw = dict(rollouts_per_candidate=args.n, horizon=args.horizon,
                  max_total_rounds=10 ** 9, value_model=vm, feature_extractor=feat,
                  value_batch=256, rollout_seed=12345, rollout_random_side="opponent",
                  rollout_act_prob=0.3, **cfg)
        s_before = mod_before.FlatRolloutGreedy(net_fn, k=args.k, **kw)
        s_after = After(net_fn, k=args.k, **kw)
        for (rnd, player, snap) in snaps:
            h = {}
            for name, searcher in (("before", s_before), ("after", s_after)):
                res = searcher.search(snap.clone(), int(player), temperature=0.0)
                stats = getattr(searcher, "last_tree_stats", None)
                h[name] = dict(
                    scores=digest(searcher.last_scores),
                    sum=repr(float(np.nansum(np.asarray(searcher.last_scores)))),
                    policy=digest(res.visit_policy),
                    chosen=int(res.chosen_index),
                    bundle=repr(res.chosen_bundle),
                    rollouts=int(searcher.last_rollouts),
                    rounds=int(searcher.last_rounds),
                    eff=getattr(searcher, "last_effective_samples", None),
                    tree=repr(stats),
                    rng_used=int(getattr(searcher, "last_rng_consumed", -1)),
                )
            same = h["before"] == h["after"]
            bad += 0 if same else 1
            emit(f"[zero] {tag:14s} r{rnd:<4} p{player} | "
                 f"{'一致 ✅' if same else '不一致 ❌'} | "
                 f"scores={h['after']['scores']} tree={h['after']['tree']} "
                 f"chosen={h['after']['chosen']} rng_used={h['after']['rng_used']} "
                 f"rollouts={h['after']['rollouts']} rounds={h['after']['rounds']}")
            if not same:
                for fld in h["before"]:
                    if h["before"][fld] != h["after"][fld]:
                        emit(f"[zero]    ✗ 字段 {fld}: before={h['before'][fld]} "
                             f"after={h['after'][fld]}")

    emit(f"[zero] 结论：{'✅ 全部逐位一致（默认关闭 ⇒ 零影响）' if bad == 0 else f'❌ {bad} 处不一致'}")
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[zero] 已写入 {out}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
