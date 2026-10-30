"""`AZAI_CAND_DECODE=legacy|split` 的**单元探针**（2026-10-31）。

验证 `split` 语义（用户定夺）：
* 通道 **0** = 只在**空格**建 `BUILD_TOWER`（我方塔/敌方塔 ⇒ None）；
* 通道 **1-15** = 只在**我方塔格**上升级 `UPGRADE_TOWER(tower_id, step)`（空格 ⇒ None）；
  目标类型仍由 `upgrade_step(tower_type, CHANNEL_TO_TOWER_TYPE[ch])` 决定；
* `legacy`（默认）= 与 `decoder._decode_tower_action` **逐字同语义**（空格建 Basic / 我方塔升级）。

同时打印在真实局面上两种模式的**每类合法格数**（split 应当：通道 0 = 空位数、1-15 = 塔位数）。

用法（纯 CPU，几秒）：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_cand_decode_split.py --rounds 60 200
```
退出码 0 = 断言全过。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_REAL = sys.stdout


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="INJECT_VS_RU_B")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--rounds", type=int, nargs="+", default=[20, 60, 200])
    ap.add_argument("--player", type=int, default=0)
    args = ap.parse_args()

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {name}{('：' + detail) if detail else ''}", file=_REAL)
        ok = ok and cond

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"
    os.environ["AZAI_CAND_MODE"] = "unified"

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from my_ai.decoder import (CHANNEL_TO_TOWER_TYPE, _decode_tower_action, make_position_masks,
                               upgrade_step)
    from my_ai.az_intent.unified_cand import decode_channel_op

    rsc = _load("_rsc_dec", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    want = set(int(x) for x in args.rounds)
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    snaps = []
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, st.clone()))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
        st.advance_round()
    print(f"[dec] 局面 = {args.tag}/seed{args.seed}  {len(snaps)} 个  player={args.player}", file=_REAL)

    print("", file=_REAL)
    print("## A. legacy 与 `decoder._decode_tower_action` 逐格一致", file=_REAL)
    n_same = n_cmp = 0
    for (r, snap) in snaps:
        for ch in range(16):
            for x in range(19):
                for y in range(19):
                    a = decode_channel_op(snap, args.player, ch, x, y, decode_mode="legacy")
                    b = _decode_tower_action(snap, args.player, ch, x, y)
                    n_cmp += 1
                    ta = None if a is None else (int(a.op_type), int(a.arg0), int(a.arg1))
                    tb = None if b is None else (int(b.op_type), int(b.arg0), int(b.arg1))
                    n_same += int(ta == tb)
    check("legacy == _decode_tower_action（全格逐点）", n_same == n_cmp,
          f"{n_same}/{n_cmp}")

    print("", file=_REAL)
    print("## B. split 语义（逐格断言）", file=_REAL)
    bad_build, bad_upg, n_build, n_upg = 0, 0, 0, 0
    for (r, snap) in snaps:
        my = {(int(t.x), int(t.y)): t for t in snap.towers_of(args.player)}
        enemy = {(int(t.x), int(t.y)) for t in snap.towers_of(1 - args.player)}
        for x in range(19):
            for y in range(19):
                a = decode_channel_op(snap, args.player, 0, x, y, decode_mode="split")
                if (x, y) in my or (x, y) in enemy:
                    bad_build += int(a is not None)      # 非空格必须 None
                elif a is not None:
                    n_build += 1
                    if int(a.op_type) != int(OperationType.BUILD_TOWER):
                        bad_build += 1
                for ch in range(1, 16):
                    b = decode_channel_op(snap, args.player, ch, x, y, decode_mode="split")
                    if (x, y) not in my:
                        bad_upg += int(b is not None)     # 非我方塔格必须 None
                        continue
                    if b is None:
                        continue
                    n_upg += 1
                    tw = my[(x, y)]
                    step = upgrade_step(tw.tower_type, CHANNEL_TO_TOWER_TYPE[ch])
                    okb = (int(b.op_type) == int(OperationType.UPGRADE_TOWER)
                           and int(b.arg0) == int(tw.tower_id) and int(b.arg1) == int(step))
                    bad_upg += int(not okb)
    check("通道 0 只在空格出 BUILD_TOWER", bad_build == 0,
          f"违例 {bad_build}（合法建塔点合计 {n_build}）")
    check("通道 1-15 只在我方塔格出 UPGRADE_TOWER(tower_id, step)", bad_upg == 0,
          f"违例 {bad_upg}（合法升级点合计 {n_upg}）")

    print("", file=_REAL)
    print("## C. 每类合法格数（decoder 位置掩码 vs split 收窄后）", file=_REAL)
    print(f"  {'round':>5} {'塔数':>4} {'空格':>4} | {'ch0(legacy)':>11} {'ch0(split)':>10} "
          f"{'ch1..15(legacy) 合计':>20} {'ch1..15(split) 合计':>20}", file=_REAL)
    for (r, snap) in snaps:
        pm = make_position_masks(snap, args.player, intent_decoding=False)
        import numpy as np
        my_cell = np.zeros_like(pm[0], dtype=bool)
        for t in snap.towers_of(args.player):
            my_cell[int(t.x), int(t.y)] = True
        leg0 = int(pm[0].sum())
        leg_up = int(sum(int(pm[ch].sum()) for ch in range(1, 16)))
        sp0 = int((pm[0] & (~my_cell)).sum())
        sp_up = int(sum(int((pm[ch] & my_cell).sum()) for ch in range(1, 16)))
        print(f"  {r:>5} {int(my_cell.sum()):>4} {int((~my_cell).sum()):>4} | "
              f"{leg0:>11} {sp0:>10} {leg_up:>20} {sp_up:>20}", file=_REAL)
    check("split 的通道 0 合法点 ≤ legacy（只在空格）", True, "见上表")

    print("", file=_REAL)
    print(f"[dec] 结论：{'全部断言通过 ✅' if ok else '有断言失败 ❌'}", file=_REAL)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
