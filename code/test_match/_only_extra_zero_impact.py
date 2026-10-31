"""**零影响**探针：`only_extra`（E4 的候选集开关）在**默认关闭**时逐位不改搜索结果（2026-10-28）。

## 为什么要它

E4 的改动落在**搜索器**里（`code/my_ai/az_intent/flat_rollout_greedy.py:search`）：
新增 `self.only_extra`（默认 False）。纪律是"开关关 ⇒ 与改动前**逐位相同**"，而
"读代码觉得没问题"不算证据 ⇒ 这里对**同一批真实局面**跑**改动前/改动后两份代码**，
把 `last_scores`（float64 原始字节）/ `visit_policy` / `chosen_bundle` / 候选数 / rollout 数
**全部打哈希**，然后逐行 diff。

## 为什么不用"改文件 → 跑 → 改回来"（历史做法）

2026-10-28 跑 E1 时机器上**还有别的实验在跑**（8 局并行）；把工作区的
`flat_rollout_greedy.py` 临时换回旧版，会让**那一刻新起的对局**用旧代码 ⇒ 污染别人（和 E1）的读数。
所以本探针**不动工作区**：它把仓库里的 `_flat_rollout_greedy_preonlyextra.py`
（**加代码那一刻**的副本，md5 见文件头常量）用 `importlib` 单独加载成一个模块，再靠
`sys.modules` 临时替换拿到「改动前」的搜索器。

## 做法

1. 从**真实对局**的 `--trace-ops` 日志重放双方操作（`repro_search_crash.read_ops`）⇒ 真的有塔/蚁/金币的局面；
2. 取若干回合 × 两个座位做快照（= 桥接里两条决策路径的真实局面）；
3. 用**同一份**部署代码 `az_bridge_ai.build_engine` 建两个搜索器：
   `pre`（`sys.modules` 里换成旧模块）与 `live`（现行模块）；两边**同一 ckpt、同一 BASE 配置**；
4. 每个快照**交替**跑两边（`pre` 先、`live` 后，各自 `_decision_idx` 同步增长 ⇒ RNG 流对齐），
   分别在 `extra_bundles=None`（默认路径）和 `extra_bundles=[固定两条]`（注入路径）下跑；
5. 逐字段打印，两跑 diff 必须**无输出**。

## 反向对照（证明开关**不是死代码**）

同一个探针在 `--only-extra` 下再跑一次：候选数必须从 K 条塌成"额外候选的条数"，
且 `chosen_bundle` 必须 ∈ 额外候选。这条**必须变化**，否则说明开关没接上。

## 用法

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P code/test_match/_only_extra_zero_impact.py --tag VS_RU_dump --seed 8 \
    --rounds 20 80 200 --extra "11,5,11;13,3,-1" --out match_results/only_extra/zero_impact_off.log
$P code/test_match/_only_extra_zero_impact.py --tag VS_RU_dump --seed 8 \
    --rounds 20 80 200 --extra "11,5,11;13,3,-1" --only-extra --out match_results/only_extra/on.log
diff <上面两份>   # 第一份必须"与改动前那份逐行相同"；第二份必须显示候选数变了
```
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

# ⚠️ `az_bridge_ai` 一旦被 import 就会把 `sys.stdout` 指到 stderr（协议流保护）⇒ 先留住真正的 stdout
_REAL_STDOUT = sys.stdout

PRE_COPY = REPO / "code" / "test_match" / "_flat_rollout_greedy_preonlyextra.py"
# 「加代码那一刻」的副本 md5（改了副本就说明比较对象被动了 ⇒ 直接停）
PRE_MD5 = "60C895D34D93A9AE6B244FDC4BBDC944"


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
    ap.add_argument("--tag", default="VS_RU_dump", help="ladder 日志 tag（要有 --trace-ops 行）")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[20, 80, 200])
    ap.add_argument("--extra", default="11,5,11;13,3,-1",
                    help="注入路径用的固定额外候选（`;` 分隔每束，每束是 3 的倍数个整数）")
    ap.add_argument("--only-extra", action="store_true",
                    help="反向对照：打开 only_extra（候选数必须变）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg, file=_REAL_STDOUT, flush=True)
        lines.append(msg)

    if _md5(PRE_COPY) != PRE_MD5:
        emit(f"!! 改动前副本 {PRE_COPY.name} 的 md5 变了（期望 {PRE_MD5}）⇒ 比较对象不可信，终止")
        return 3

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"      # 浮点累加顺序要稳 ⇒ 单线程
    # ⚠️ 探针本身**绝不**开注入/only-extra（这些是下面手动切的对象）
    for k in ("AZAI_ORACLE_INJECT", "AZAI_ORACLE_INJECT_B", "AZAI_ONLY_EXTRA",
              "AZAI_ONLY_EXTRA_RANDOM", "AZAI_ORACLE_PLACEBO", "AZAI_UPG_BUNDLES"):
        os.environ.pop(k, None)

    # ── ① 重放真实对局，取局面快照 ───────────────────────────────────────────
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    rsc = _load("_repro_search_crash", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        emit(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行，换一个 tag/seed")
        return 2
    want = set(int(x) for x in args.rounds)
    snaps: list[tuple[int, int, object]] = []
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, 0, st.clone()))            # P0 决策时的局面（本回合还没人动）
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))        # P1 决策时的局面（P0 已动）
        st.advance_round()
    extras: list = []
    for grp in args.extra.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        t = [int(x) for x in grp.split(",") if x.strip()]
        extras.append(tuple(tuple(t[i:i + 3]) for i in range(0, len(t) - 2, 3)))

    emit(f"[zi] 改动前副本 = {PRE_COPY.name} md5={PRE_MD5}（已核对）")
    emit(f"[zi] 局面来源 tag={args.tag} seed={args.seed} 快照={len(snaps)} 个"
         f"（回合 {sorted(want)} × 2 座位）")
    emit(f"[zi] BASE 配置 = {len(env)} 项（从 _inject_vs_runnerup.sh 读）；"
         f"K={env.get('AZAI_K')} rounds={env.get('AZAI_FLAT_ROUNDS')} "
         f"rollouts={env.get('AZAI_FLAT_ROLLOUTS')} mode={env.get('AZAI_FLAT_ROLLOUT_MODE')}")
    emit(f"[zi] 固定额外候选 = {extras}；only_extra={int(args.only_extra)}")

    # ── ② 两个搜索器：pre（旧模块）/ live（现行模块），部署代码同一份 ────────────
    bridge = _load("az_bridge_ai", REPO / "code" / "test_match" / "az_bridge_ai.py")
    import my_ai.az_intent.flat_rollout_greedy as _live_flat

    pre_mod = _load("_flat_preonlyextra", PRE_COPY)
    if not hasattr(pre_mod.FlatRolloutGreedy, "set_only_extra"):
        emit("[zi] ✓ 旧模块里没有 set_only_extra（确认它是改动前那一份）")
    else:
        emit("!! 旧模块里居然有 set_only_extra ⇒ 副本拿错了，终止")
        return 3
    sys.modules["my_ai.az_intent.flat_rollout_greedy"] = pre_mod
    try:
        pre_searcher, _m1, _f1 = bridge.build_engine(args.seed, 0)
    finally:
        sys.modules["my_ai.az_intent.flat_rollout_greedy"] = _live_flat
    live_searcher, _m2, _f2 = bridge.build_engine(args.seed, 0)
    emit(f"[zi] pre={type(pre_searcher).__name__}  live={type(live_searcher).__name__} "
         f"（同一个 ckpt：{env.get('AZAI_CKPT')}）")

    # ── ③ 逐快照交替跑，两边逐字段打哈希 ──────────────────────────────────────
    for (rnd, player, snap) in snaps:
        for tag_extra, ex in (("默认路径(extra=None)", None), ("注入路径(extra=固定2条)", extras)):
            a = pre_searcher.search(snap.clone(), int(player), temperature=0.0, extra_bundles=ex)
            b = live_searcher.search(snap.clone(), int(player), temperature=0.0, extra_bundles=ex)
            sa = getattr(pre_searcher, "last_scores", None)
            sb = getattr(live_searcher, "last_scores", None)
            same = (
                len(a.bundles) == len(b.bundles)
                and a.chosen_index == b.chosen_index
                and tuple(a.chosen_bundle) == tuple(b.chosen_bundle)
                and int(pre_searcher.last_extra) == int(live_searcher.last_extra)
                and int(pre_searcher.last_rollouts) == int(live_searcher.last_rollouts)
                and int(pre_searcher.last_rounds) == int(live_searcher.last_rounds)
                and _digest(a.visit_policy) == _digest(b.visit_policy)
                and _digest(sa) == _digest(sb)
            )
            emit(f"[zi] r{rnd:<4} p{player} {tag_extra:<26} | "
                 f"候选={len(a.bundles):<4} extra={int(pre_searcher.last_extra):<2} "
                 f"rollouts={int(pre_searcher.last_rollouts):<3} "
                 f"回合={int(pre_searcher.last_rounds):<6} | "
                 f"chosen_idx={a.chosen_index} chosen={tuple(tuple(x) for x in a.chosen_bundle)} "
                 f"| scores_sha={_digest(sa)} policy_sha={_digest(a.visit_policy)} "
                 f"bundles_sha={hashlib.sha256(repr([tuple(tuple(x) for x in k) for k in a.bundles]).encode()).hexdigest()[:16]} "
                 f"root_value={a.root_value!r}")
            if not same:
                emit(f"    !! 不一致：live 侧 候选={len(b.bundles)} extra={int(live_searcher.last_extra)} "
                     f"chosen_idx={b.chosen_index} chosen={tuple(tuple(x) for x in b.chosen_bundle)} "
                     f"scores_sha={_digest(sb)} policy_sha={_digest(b.visit_policy)}")

    # ── ④ 反向对照：打开 only_extra（必须**变**，否则说明开关是死代码）──────────────
    if args.only_extra:
        emit("")
        emit("[zi] ===== 反向对照：only_extra=ON（候选集必须塌成额外候选）=====")
        live_searcher.set_only_extra(True)
        _expect = len(extras)
        _n_extra_ok = 0
        for (rnd, player, snap) in snaps[:2]:
            b = live_searcher.search(snap.clone(), int(player), temperature=0.0, extra_bundles=extras)
            chosen = tuple(tuple(int(x) for x in op) for op in (b.chosen_bundle or ()))
            in_extra = chosen in [tuple(tuple(int(x) for x in op) for op in e) for e in extras]
            _n_extra_ok += int(in_extra)
            emit(f"[zi] r{rnd:<4} p{player} only_extra=ON | 候选={len(b.bundles)}"
                 f"（期望 {_expect} = 额外候选去重后条数） extra={int(live_searcher.last_extra)} "
                 f"rollouts={int(live_searcher.last_rollouts)} "
                 f"chosen={chosen} ∈ 额外候选={in_extra} "
                 f"| {'✓ 开关确实改变了候选集' if len(b.bundles) == _expect else '!! 候选数不是额外候选条数'}")
        emit(f"[zi] 反向对照小结：{_n_extra_ok}/{min(2, len(snaps))} 个决策的 chosen 落在额外候选里"
             f"（only_extra 的语义 = 只能从额外候选里选）")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[zi] 已写入 {out}", file=_REAL_STDOUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
