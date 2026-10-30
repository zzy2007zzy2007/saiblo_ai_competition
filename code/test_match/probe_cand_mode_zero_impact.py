"""**零影响**探针：`AZAI_CAND_MODE` 不设（= `pin`）时搜索结果**逐位不变**（2026-10-29）。

## 为什么要它

"统一候选生成"的改动落在两处：
1. `code/my_ai/az_intent/flat_rollout_greedy.py`：新增 `idle_rollout_values()`（新方法）
   + `search()` 里两处 `if _reuse is not None` 钩子（`cand_mode != "unified"` 时恒 None）
   + `cand_mode/cand_verbose/last_cand_stats/last_reused` 四个属性；
2. `code/test_match/az_bridge_ai.py`：`AZAI_CAND_MODE` 读取 + 一个 `if` 决定构造哪个类。
3. `code/my_ai/az_intent/bundle_mcts.py` / `decoder.py` / 网络：**零改动**。

纪律是"开关关 ⇒ 与改动前**逐位相同**"，而"读代码觉得没问题"不算证据 ⇒ 本探针对**同一批
真实局面**跑**改动前/改动后两份代码**，把 `last_scores`（float64 原始字节）/ `visit_policy` /
`chosen_bundle` / 候选数 / rollout 数 / 预算回合数**全部打哈希**，然后 diff 必须无输出。

## 与前例（`_only_extra_zero_impact.py`）的关系

同一个模式：`code/test_match/_flat_rollout_greedy_preunifiedcand.py` 是**动手前那一刻**的副本
（md5 见下方常量），用 `importlib` 单独加载成一个模块，再靠 `sys.modules` 临时替换拿到
「改动前」的搜索器。**不动工作区、不换文件** ⇒ 不会污染正在跑的实验。

## 用法

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_cand_mode_zero_impact.py \
    --tag INJECT_VS_RU_B --seed 8 --rounds 20 60 120 200 \
    --out match_results/unified_cand/zero_impact_probe.log
```
返回 0 = 两侧逐字段相同；4 = 有不一致（会逐行打出）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_REAL_STDOUT = sys.stdout

PRE_COPY = REPO / "code" / "test_match" / "_flat_rollout_greedy_preunifiedcand.py"
# 「加代码那一刻」的副本 md5（副本被动过 ⇒ 比较对象不可信 ⇒ 直接停）
PRE_MD5 = "5B92A4865B849BFCE16BCE65938E1424"


def _digest(x) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest().upper()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base_env() -> dict[str, str]:
    """从**部署脚本**里读 BASE（不在这里重抄一遍：重抄就会漂）。"""
    txt = (REPO / "code" / "test_match" / "_inject_vs_runnerup.sh").read_text(encoding="utf-8")
    m = re.search(r'^BASE="(.*)"$', txt, re.M)
    if not m:
        raise SystemExit("读不到 _inject_vs_runnerup.sh 里的 BASE")
    out = {}
    for kv in m.group(1).split(";"):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="INJECT_VS_RU_B")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[20, 60, 120, 200])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg, file=_REAL_STDOUT, flush=True)
        lines.append(msg)

    if _md5(PRE_COPY) != PRE_MD5:
        emit(f"!! 改动前副本 {PRE_COPY.name} 的 md5 变了（期望 {PRE_MD5}，实得 {_md5(PRE_COPY)}）"
             f" ⇒ 比较对象不可信，终止")
        return 3

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"
    # ⚠️ 探针本身**绝不**设 AZAI_CAND_MODE（要测的就是"不设"）
    for k in ("AZAI_CAND_MODE", "AZAI_ORACLE_INJECT", "AZAI_ORACLE_INJECT_B", "AZAI_ONLY_EXTRA",
              "AZAI_ONLY_EXTRA_RANDOM", "AZAI_ORACLE_PLACEBO", "AZAI_UPG_BUNDLES",
              "AZAI_ALWAYS_HOLD", "AZAI_PIN_K"):
        os.environ.pop(k, None)

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    rsc = _load("_repro_search_crash_zi", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        emit(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行")
        return 2
    want = set(int(x) for x in args.rounds)
    snaps: list = []
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, 0, st.clone()))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))
        st.advance_round()

    # ── 两个搜索器：pre（旧模块）/ live（现行模块），部署代码同一份 ────────────────
    bridge = _load("az_bridge_ai_zi", REPO / "code" / "test_match" / "az_bridge_ai.py")
    import my_ai.az_intent.flat_rollout_greedy as _live_flat

    pre_mod = _load("_flat_preunifiedcand", PRE_COPY)
    if hasattr(pre_mod.FlatRolloutGreedy, "idle_rollout_values"):
        emit("!! 旧模块里居然有 idle_rollout_values ⇒ 副本拿错了，终止")
        return 3
    emit(f"[zi] ✓ 旧模块里没有 idle_rollout_values / cand_mode（确认它是改动前那一份）"
         f" md5={PRE_MD5}")
    sys.modules["my_ai.az_intent.flat_rollout_greedy"] = pre_mod
    try:
        pre_searcher, _m1, _f1 = bridge.build_engine(args.seed, 0)
    finally:
        sys.modules["my_ai.az_intent.flat_rollout_greedy"] = _live_flat
    live_searcher, _m2, _f2 = bridge.build_engine(args.seed, 0)
    emit(f"[zi] pre={type(pre_searcher).__name__}(cand_mode="
         f"{getattr(pre_searcher, 'cand_mode', '<无此属性>')})  "
         f"live={type(live_searcher).__name__}(cand_mode={getattr(live_searcher, 'cand_mode', '?')})")
    emit(f"[zi] BASE 配置 {len(env)} 项（从 _inject_vs_runnerup.sh 读）；"
         f"K={env.get('AZAI_K')} rounds={env.get('AZAI_FLAT_ROUNDS')} "
         f"rollouts={env.get('AZAI_FLAT_ROLLOUTS')} mode={env.get('AZAI_FLAT_ROLLOUT_MODE')} "
         f"AZAI_CAND_MODE=<未设>")
    emit(f"[zi] 局面来源 tag={args.tag} seed={args.seed} 快照={len(snaps)} 个"
         f"（回合 {sorted(want)} × 2 座位）")

    n_bad = 0
    for (rnd, player, snap) in snaps:
        a = pre_searcher.search(snap.clone(), int(player), temperature=0.0)
        b = live_searcher.search(snap.clone(), int(player), temperature=0.0)
        sa = getattr(pre_searcher, "last_scores", None)
        sb = getattr(live_searcher, "last_scores", None)
        same = (
            len(a.bundles) == len(b.bundles)
            and a.chosen_index == b.chosen_index
            and tuple(a.chosen_bundle) == tuple(b.chosen_bundle)
            and repr(a.bundles) == repr(b.bundles)
            and int(pre_searcher.last_rollouts) == int(live_searcher.last_rollouts)
            and int(pre_searcher.last_rounds) == int(live_searcher.last_rounds)
            and _digest(a.visit_policy) == _digest(b.visit_policy)
            and _digest(sa) == _digest(sb)
            and float(a.root_value) == float(b.root_value)
            and repr(a.intent_counts) == repr(b.intent_counts)
        )
        emit(f"[zi] r{rnd:<4} p{player} | 候选={len(a.bundles):<3} "
             f"rollouts={int(pre_searcher.last_rollouts):<3} 回合={int(pre_searcher.last_rounds):<6} | "
             f"chosen_idx={a.chosen_index} chosen={tuple(tuple(int(y) for y in x) for x in a.chosen_bundle)} "
             f"scores_sha={_digest(sa)} policy_sha={_digest(a.visit_policy)} "
             f"bundles_sha={hashlib.sha256(repr(a.bundles).encode()).hexdigest()[:16]} "
             f"root_value={a.root_value!r} | {'✓ 逐字段相同' if same else '!! 不一致'}")
        if not same:
            n_bad += 1
            emit(f"    !! live 侧 候选={len(b.bundles)} rollouts={int(live_searcher.last_rollouts)} "
                 f"回合={int(live_searcher.last_rounds)} chosen_idx={b.chosen_index} "
                 f"chosen={tuple(tuple(int(y) for y in x) for x in b.chosen_bundle)} "
                 f"scores_sha={_digest(sb)} policy_sha={_digest(b.visit_policy)} "
                 f"root_value={b.root_value!r}")

    emit("")
    emit(f"[zi] 小结：{len(snaps)} 个真实局面（pre vs live）不一致 = {n_bad}（必须 0）")
    emit(f"[zi] 判据：`last_scores`(float64 原始字节)/`visit_policy`/`bundles`/`chosen_bundle`/"
         f"`root_value`/`intent_counts`/rollout 数/预算回合数 全等 ⇒ pin 模式逐位不变 ✓"
         if n_bad == 0 else "[zi] ❌ 存在不一致，见上面 !! 行")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[zi] 已写入 {out}", file=_REAL_STDOUT, flush=True)
    return 0 if n_bad == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
