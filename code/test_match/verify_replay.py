#!/usr/bin/env python
"""逐位校验 `az_selfplay` 写的**官方 replay**：seed + 每回合双方 ops 能否精确重建这一局。

为什么要它（`docs/value_net_collection_impl_20261031.md` §B）：
  只存 `board+stats` 是**有损**的（缺塔 HP/精确蚂蚁状态），以后改值网输入编码、做续盘回归都重建不了局面。
  游戏是**确定性**的 ⇒ `seed + 每回合双方的完整 op 序列` 足以重建任意时刻的完整状态。
  ⚠️ 之前"progress trace 日志重放 144 局只有 2 局能复现终局血"的教训是：trace 是**有损**的
  （只记了部分字段）。这份 replay 记的是**实际被执行的 ops**（含规则注入/随机注入的结果），所以可以精确重放。

做什么：
  1. 用 `seed` 起同一引擎（`--engine native|py`，默认按 replay 里的 `engine` 字段）；
  2. 逐回合按 `op0/op1` 施放（**HOLD = 空列表**），每回合末 `advance_round()`；
  3. 在**每个决策时刻**与 `.ents.jsonl` 的 `coins/hp_us/hp_opp` 对比（逐位相等才算过）；
  4. 终局与 replay 的 `result`（winner/hp/rounds）以及 npz 的 `value_target` 对比。

用法：
  python code/test_match/verify_replay.py training_history/collect_v2_s1/az_selfplay_seed01000000.replay.json
  python code/test_match/verify_replay.py --dir training_history/collect_v2_s1 --glob "*.replay.json"
  # 只校验终局/回合数（没存 ents.jsonl 时）：
  python code/test_match/verify_replay.py --replay-only <replay.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _init_state(seed: int, engine: str):
    if engine.startswith("cpp"):
        from my_ai.az_intent.game_state_facade import GameStateFacade
        return GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    from SDK.backend.engine import GameState
    return GameState.initial(seed=seed, cold_handle_rule_illegal=True)


def verify(replay_path: Path, *, ents_path: Path | None = None,
           npz_path: Path | None = None, engine: str | None = None,
           verbose: bool = True) -> dict:
    rp = json.loads(replay_path.read_text(encoding="utf-8"))
    seed = int(rp["seed"])
    eng = engine or str(rp.get("engine", "cpp_native"))
    steps = rp["steps"]
    res = rp.get("result") or {}
    if ents_path is None:
        cand = replay_path.with_name(replay_path.name.replace(".replay.json", ".ents.jsonl"))
        ents_path = cand if cand.exists() else None
    if npz_path is None:
        cand = replay_path.with_name(replay_path.name.replace(".replay.json", ".npz"))
        npz_path = cand if cand.exists() else None
    ents = []
    if ents_path is not None and ents_path.exists():
        ents = [json.loads(x) for x in ents_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    st = _init_state(seed, eng)
    ei = 0
    n_chk = 0
    n_bad = 0
    first_bad = None
    for step in steps:
        rnd = int(step["round"])
        for p in (0, 1):
            if st.terminal:
                break
            # ── 决策时刻比对（ents.jsonl 与 npz/样本同序）──
            if ei < len(ents) and int(ents[ei]["round"]) == rnd and int(ents[ei]["player"]) == p:
                e = ents[ei]
                got = (int(st.coins[0]), int(st.coins[1]),
                       int(st.bases[p].hp), int(st.bases[1 - p].hp))
                want = (int(e["coins"][0]), int(e["coins"][1]),
                        int(e["hp_us"]), int(e["hp_opp"]))
                n_chk += 1
                if got != want:
                    n_bad += 1
                    if first_bad is None:
                        first_bad = {"round": rnd, "player": p, "got": got, "want": want}
                ei += 1
            ops = step.get(f"op{p}") or []
            st.apply_operation_list(
                p, [Operation(OperationType(int(t)), int(a0), int(a1)) for (t, a0, a1) in ops])
        if not st.terminal:
            st.advance_round()

    out = {"replay": replay_path.name, "seed": seed, "engine": eng,
           "steps": len(steps), "checked_decisions": n_chk, "mismatches": n_bad,
           "first_mismatch": first_bad,
           "final_hp": [int(st.bases[0].hp), int(st.bases[1].hp)],
           "final_winner": (None if st.winner is None else int(st.winner)),
           "final_rounds": int(st.round_index), "terminal": bool(st.terminal),
           "expected": res, "ok": True}
    # 终局三项
    if res.get("hp") and list(res["hp"]) != out["final_hp"]:
        out["ok"] = False
    if "winner" in res and res["winner"] != out["final_winner"]:
        out["ok"] = False
    if "rounds" in res and int(res["rounds"]) != out["final_rounds"]:
        out["ok"] = False
    # value_target（clip(margin/20)）
    if npz_path is not None and npz_path.exists():
        import numpy as np
        z = np.load(npz_path)
        vt = float(z["value_target"][0])
        margin = (int(st.bases[0].hp) - int(st.bases[1].hp)) / 20.0
        vt_p0 = max(-1.0, min(1.0, margin))
        if margin == 0 and st.winner is not None:
            vt_p0 = 0.1 if int(st.winner) == 0 else -0.1
        out["value_target_npz_p0"] = vt
        out["value_target_replayed_p0"] = float(vt_p0)
        if abs(vt - vt_p0) > 1e-6:
            out["ok"] = False
    if n_bad:
        out["ok"] = False
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", nargs="?", help="单个 replay.json 路径")
    ap.add_argument("--dir", default="", help="批量：目录")
    ap.add_argument("--glob", default="*.replay.json", help="批量：文件模式")
    ap.add_argument("--engine", default=None, choices=[None, "cpp_native", "py_sdk"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    paths: list[Path] = []
    if a.replay:
        paths.append(Path(a.replay))
    if a.dir:
        paths.extend(sorted(Path(a.dir).glob(a.glob)))
    if not paths:
        print("!! 没有输入（给 replay 路径或 --dir）", file=sys.stderr)
        return 2
    n_ok = 0
    for p in paths:
        r = verify(p, engine=a.engine, verbose=not a.quiet)
        n_ok += int(bool(r["ok"]))
        if a.quiet:
            print(f"{p.name}: ok={r['ok']} mismatches={r['mismatches']} "
                  f"final_hp={r['final_hp']} rounds={r['final_rounds']}")
    print(f"\n[verify] {n_ok}/{len(paths)} 通过")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
