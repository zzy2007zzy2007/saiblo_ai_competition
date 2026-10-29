#!/usr/bin/env python
"""临时自测：`RunnerUpOracle` 的 history / replay / rebuild_session 是否工作（2026-10-29）。

不跑对局，只起一个亚军会话，喂 20 回合，然后**故意关掉**它，再重建 + 回放，看能不能无缝续上。
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

spec = importlib.util.spec_from_file_location("ro", REPO / "code" / "test_match" / "runnerup_oracle.py")
ro = importlib.util.module_from_spec(spec)
sys.modules["ro"] = ro
spec.loader.exec_module(ro)

from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

seed = 7
st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
orc = ro.RunnerUpOracle(seed=seed, player=1, timeout=20.0,
                        stderr_path=REPO / "match_results" / "oracle_logs" / "_tmp_selftest.stderr.log")
t0 = time.time()
for _ in range(20):
    ops = orc.ask(st, opponent_ops=[])
    st.apply_operation_list(1, ops)
    st.advance_round()
print(f"[selftest] 20 回合 OK, history={len(orc.history)} next={orc._next_round} "
      f"stats={orc.stats} {time.time() - t0:.1f}s")
print(f"[selftest] history[0] 帧标签={[f[0] for f in orc.history[0][1]]} "
      f"history[5] 帧标签={[f[0] for f in orc.history[5][1]]}")

# 故意"杀死"会话（模拟 OracleDead），然后用 rebuild_session 重建 + 回放到第 20 回合
orc.close()
t1 = time.time()
new, n, dt = ro.rebuild_session(orc, upto_round=int(st.round_index))
print(f"[selftest] rebuild: 回放 {n} 回合 {dt:.1f}s next={new._next_round} "
      f"（真回合={st.round_index}）history={len(new.history)}")
assert new._next_round == int(st.round_index), "回放后 next_round 必须等于当前回合"
ops = new.ask(st, opponent_ops=[])
print(f"[selftest] 重建后第 {st.round_index} 回合 ask OK -> {len(ops)} ops  "
      f"总耗时 {time.time() - t1:.1f}s stats={new.stats}")
new.close()
print("[selftest] ✅ PASS")
