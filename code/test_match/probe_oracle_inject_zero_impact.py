"""**零影响**探针：证明"额外候选注入"这条通道在默认关闭时**数值行为逐位不变**。

## 为什么需要它

`extra_bundles` 是加在 `FlatRolloutGreedy.search` / `UnifiedActionSearch.search` 里的新入参。
它必须满足：**不传 ⇒ 候选列表/顺序/预算/分数/选择全部与改动前一致**。
"读一遍代码觉得没问题"不算证据 —— 这个探针打**哈希**。

## 做法（不依赖搜索代码本身生成局面 ⇒ 前后两跑的局面必然相同）

1. 用 `repro_search_crash.read_ops()` 从**真实对局**的 `--trace-ops` 日志里重放双方操作
   （`match_results/ladder_logs/<tag>/<seed>.log`）⇒ 得到**真的**有塔/蚁/金币的局面；
2. 在若干回合上取快照：p0 决策前一份、p0 已 apply p1 决策前一份（= 桥接里两条决策路径的真实局面）；
3. 对每个 (局面, player) 跑一次 `search`，把 `last_scores`（float64 原始字节）、
   `visit_policy`、`chosen_bundle`、`chosen_index`、候选数、实际 rollout 次数**全部打哈希**。

改动前跑一次、改动后跑一次，两份日志**逐行相同** ⇒ 零影响成立
（这是比"跑两局看看结果"强得多的证据：它逐位比对，而且覆盖了真实中局）。

## 用法

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P code/test_match/probe_oracle_inject_zero_impact.py --tag VS_RU_dump --seed 8 \
    --rounds 10 60 200 --out match_results/oracle_inject/zero_impact_flat_before.log
# …改代码…
$P code/test_match/probe_oracle_inject_zero_impact.py --tag VS_RU_dump --seed 8 \
    --rounds 10 60 200 --out match_results/oracle_inject/zero_impact_flat_after.log
diff <before> <after>      # 必须无输出
```

`--search flat|unified` 决定跑哪条搜索（冠军配置 = `flat`）。
环境变量口径与部署完全一致（`AZAI_CKPT/AZAI_K/AZAI_FLAT_*/AZAI_MODE`），见 `main()` 里的 `ENV`。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# 冠军配置（用户 2026-10-26 指定；与 32 局 A/B 用的完全一致）
ENV = {
    "AZAI_CKPT": "checkpoints/posnet_A_k5_m32.pt",
    "AZAI_K": "160",
    "AZAI_FLAT_ROUNDS": "39000",
    "AZAI_FLAT_ROLLOUT_MODE": "idle",
    "AZAI_MODE": "pos-only",
    "AZAI_TORCH_THREADS": "1",
}


def _digest(x) -> str:
    """float64 原始字节的 sha256 前 16 位。`-inf` 之类也照样参与比较。"""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="VS_RU_dump", help="ladder 日志的 tag（match_results/ladder_logs/<tag>/）")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[10, 60, 200])
    ap.add_argument("--search", default="flat", choices=["flat", "unified"])
    ap.add_argument("--inject", default="",
                    help="验证注入通道本身：逗号分隔的 op（'11,10,11'），多组用 ';' 分隔；"
                         "'hold' = 空 bundle。默认空 ⇒ 与零影响口径完全一致")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for k, v in ENV.items():
        os.environ[k] = v
    os.environ["AZAI_SEARCH"] = args.search
    # ⚠️ 探针本身**绝不**开注入：这条通道的"关"就是本探针要证明的东西。
    os.environ["AZAI_ORACLE_INJECT"] = ""

    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    # ── ① 重放真实对局，取局面快照 ─────────────────────────────────────────
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    rsc = _load("_repro_search_crash", REPO / "code/test_match/repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        print(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行，换一个 tag/seed", flush=True)
        return 2
    want = set(int(x) for x in args.rounds)
    snaps: list[tuple[int, int, object]] = []
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    stop = max(want) + 1
    for r in range(0, stop):
        if r in want:
            snaps.append((r, 0, st.clone()))          # P0 决策时的局面（本回合还没人动）
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))      # P1 决策时的局面（P0 已动）
        st.advance_round()
    emit(f"[probe] tag={args.tag} seed={args.seed} search={args.search} "
         f"快照={len(snaps)} 个（回合 {sorted(want)} × 2 个座位）")

    # ── ② 建引擎（走部署同一条路：az_bridge_ai.build_engine）──────────────
    bridge = _load("az_bridge_ai", REPO / "code/test_match/az_bridge_ai.py")
    searcher, _model, _feat = bridge.build_engine(args.seed, 0)
    emit(f"[probe] 引擎={type(searcher).__name__} ckpt={os.environ['AZAI_CKPT']} "
         f"k={os.environ['AZAI_K']} rounds={os.environ['AZAI_FLAT_ROUNDS']} "
         f"mode={os.environ['AZAI_MODE']} rollout={os.environ['AZAI_FLAT_ROLLOUT_MODE']}")

    # ── ③ 每个快照跑一次 search，全字段打哈希 ──────────────────────────────
    extras = None
    if args.inject:
        extras = []
        for _grp in args.inject.split(";"):
            _grp = _grp.strip()
            if not _grp:
                continue
            if _grp.lower() in ("hold", "()", "empty"):
                extras.append(())
                continue
            _t = [int(x) for x in _grp.split(",") if x.strip()]
            extras.append(tuple(tuple(_t[i:i + 3]) for i in range(0, len(_t) - 2, 3)))
        emit(f"[probe] 注入通道测试：extra_bundles={extras}")

    for (rnd, player, snap) in snaps:
        state = snap.clone()
        res = searcher.search(state, int(player), temperature=0.0, extra_bundles=extras)
        budgets = getattr(searcher, "last_scores", None)
        emit(
            f"[probe] r{rnd:<4} p{player} | 候选={len(res.bundles):<4} "
            f"额外候选={int(getattr(searcher, 'last_extra', 0) or 0):<2} "
            f"rollouts={getattr(searcher, 'last_rollouts', -1):<3} "
            f"回合={getattr(searcher, 'last_rounds', -1):<7} "
            f"| chosen_idx={res.chosen_index} chosen={res.chosen_bundle} "
            f"root_value={res.root_value!r}"
        )
        # 逐候选分数：flat 有 last_scores；unified 只有 last_actions（含每个候选的均值）
        # ⇒ 两个都打哈希（后者对两条搜索都有效，且含浮点 repr ⇒ 连 1ulp 的差都看得见）。
        _scores_sha = ("NA" if budgets is None or np.asarray(budgets).size == 0
                       else _digest(budgets))
        _sum = ("nan" if budgets is None or np.asarray(budgets).size == 0
                else repr(float(np.nansum(np.asarray(budgets, dtype=np.float64)))))
        emit(
            f"[probe]            | scores_sha={_scores_sha} "
            f"actions_sha={hashlib.sha256(repr(getattr(searcher, 'last_actions', None)).encode()).hexdigest()[:16]} "
            f"policy_sha={_digest(res.visit_policy)} "
            f"bundles_sha={hashlib.sha256(repr(res.bundles).encode()).hexdigest()[:16]} "
            f"sum_scores={_sum}"
        )

    emit(f"[probe] 完成：{len(snaps)} 个决策全部打印哈希 ⇒ 与另一份日志逐行 diff 应无差异")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[probe] 已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
