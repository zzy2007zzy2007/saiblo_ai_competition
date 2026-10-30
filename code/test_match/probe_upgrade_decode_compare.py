"""同一个"升级买得起"的局面，`legacy` vs `split` 的候选构成对比（2026-10-31）。

为什么要它：在真局里，"升级买得起（金币 ≥ 60 且有可升的塔）"的决策只占 **~4%**
（`probe_upgrade_supply.py` 实测 14/346）⇒ "含升级的**回合占比**"有硬上界，
`split` **不可能**把那个占比抬高多少。真正该看的是：**在买得起的那些回合里，
两种 decode 各自能不能把升级提出来、提多少条、排第几**。

做法：回放一局（`--trace-ops`），挑出"至少 1 条合法升级 op"的决策点，对每个点用
**同一个生成器**分别以 `decode=legacy` / `decode=split` 跑 `generate()`，比较：
  * 候选里 `op 12`（升级）的条数、涉及多少个不同 `(tower_id, step)`；
  * 阶段 1 分数最高的候选里有没有升级；升级候选的**最高排名**；
  * `n_cand` / 含 HOLD / 同类型多 op / 耗时。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_upgrade_decode_compare.py --tag UNIFIED_CAND_FULL --seed 8
```
"""
from __future__ import annotations

import argparse
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

from SDK.backend.model import Operation            # noqa: E402
from SDK.utils.constants import OperationType      # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.decoder import CHANNEL_TO_TOWER_TYPE, make_position_masks, upgrade_step  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base_env() -> dict:
    txt = (REPO / "code" / "test_match" / "_inject_vs_runnerup.sh").read_text(encoding="utf-8")
    m = re.search(r'^BASE="(.*)"$', txt, re.M)
    out = {}
    for kv in m.group(1).split(";"):
        if kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def _upgrade_ops(st, me: int) -> set:
    pm = make_position_masks(st, me, intent_decoding=False)
    my_cell = np.zeros_like(pm[0], dtype=bool)
    for t in st.towers_of(me):
        my_cell[int(t.x), int(t.y)] = True
    out = set()
    for ch in range(1, 16):
        for x, y in zip(*np.nonzero(pm[ch] & my_cell)):
            tw = st.tower_at(int(x), int(y))
            if tw is None:
                continue
            step = upgrade_step(tw.tower_type, CHANNEL_TO_TOWER_TYPE[ch])
            if step is None:
                continue
            op = Operation(OperationType.UPGRADE_TOWER, int(tw.tower_id), int(step))
            try:
                if st.can_apply_operation(me, op, ()):
                    out.add((int(tw.tower_id), int(step)))
            except Exception:   # noqa: BLE001
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="UNIFIED_CAND_FULL")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--k", type=int, default=48)
    args = ap.parse_args()

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"
    os.environ["AZAI_CAND_MODE"] = "unified"
    os.environ["AZAI_K"] = str(int(args.k))

    rsc = _load("_rsc_udc", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        print("!! 没有 --trace-ops 行")
        return 2
    me = int(args.player)
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    snaps = []
    for r in range(0, max(rounds) + 1):
        ops_ok = _upgrade_ops(st, me)
        if ops_ok:
            snaps.append((r, int(st.coins[me]), ops_ok, st.clone()))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
        st.advance_round()
    print(f"[udc] 局面 = {args.tag}/seed{args.seed}  player={me}  "
          f"'升级买得起'的决策点 = {len(snaps)}")
    if not snaps:
        print("[udc] 该局没有可升级的决策点 ⇒ 无需比较")
        return 0

    bridge = _load("az_bridge_ai_udc", REPO / "code" / "test_match" / "az_bridge_ai.py")
    searcher, _m, _f = bridge.build_engine(args.seed, 0)
    gen = searcher.cand_gen
    gen.decode_mode = "legacy"
    gen.rollout_fn = searcher.idle_rollout_values

    rows = []
    for (r, coins, ops_ok, snap) in snaps:
        net_out = searcher.net_fn(snap.clone(), me)
        out = {}
        for mode in ("legacy", "split"):
            gen.decode_mode = mode
            res = gen.generate(snap.clone(), me, net_out, k=int(args.k))
            bundles = res["bundles"]
            upg_idx = [i for i, b in enumerate(bundles) if any(int(o[0]) == 12 for o in b)]
            upg_ops = set()
            for i in upg_idx:
                for o in bundles[i]:
                    if int(o[0]) == 12:
                        upg_ops.add((int(o[1]), int(o[2])))
            out[mode] = dict(n=len(bundles), n_upg=len(upg_idx),
                             kinds=len(upg_ops), best_rank=(upg_idx[0] + 1 if upg_idx else None),
                             sizes=res["stats"]["sizes"], ms=res["stats"]["p1_ms"])
        rows.append((r, coins, len(ops_ok), out))

    print("")
    print(f"  {'round':>5} {'金币':>4} {'合法升级op':>9} | "
          f"{'legacy: 条数/种类/最高名次':>26} | {'split: 条数/种类/最高名次':>26}")
    better = equal = worse = 0
    for (r, coins, n_ok, out) in rows:
        lg, sp = out["legacy"], out["split"]
        f = lambda d: f"{d['n_upg']:>2}/{d['kinds']:>2}/{str(d['best_rank']):>4}"  # noqa: E731
        print(f"  {r:>5} {coins:>4} {n_ok:>9} | {f(lg):>26} | {f(sp):>26}")
        if sp["kinds"] > lg["kinds"]:
            better += 1
        elif sp["kinds"] == lg["kinds"]:
            equal += 1
        else:
            worse += 1
    print("")
    print(f"[udc] split 的升级**种类数**比 legacy：更多 {better} / 相同 {equal} / 更少 {worse}"
          f"（共 {len(rows)} 个买得起的决策点）")
    avg = lambda key, m: sum(out[m][key] for (_r, _c, _o, out) in rows) / len(rows)  # noqa: E731
    print(f"[udc] 平均：legacy 升级候选 {avg('n_upg','legacy'):.2f} 条 / 种类 "
          f"{avg('kinds','legacy'):.2f}；split {avg('n_upg','split'):.2f} 条 / 种类 "
          f"{avg('kinds','split'):.2f}")
    print(f"[udc] 平均阶段 1 耗时：legacy {avg('ms','legacy'):.0f} ms；"
          f"split {avg('ms','split'):.0f} ms")
    best_l = [out['legacy']['best_rank'] for (_r, _c, _o, out) in rows]
    best_s = [out['split']['best_rank'] for (_r, _c, _o, out) in rows]
    nl = sum(1 for x in best_l if x is not None)
    ns = sum(1 for x in best_s if x is not None)
    print(f"[udc] 候选里**真的有升级**的决策点：legacy {nl}/{len(rows)}，split {ns}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
