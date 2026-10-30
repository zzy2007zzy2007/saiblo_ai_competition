"""临时检查：阶段 2 复用（`AZAI_FLAT_ROLLOUT_MODE=idle` 时 P1 样本被 P2 复用）确实生效。

`_phase2_cache` 只在 **P1 估计器 == P2 估计器** 时非 None（默认 idle vs random ⇒ None）。
本脚本把 BASE 的 `AZAI_FLAT_ROLLOUT_MODE` 改成 idle 跑两个真实局面，打印 `last_reused`。
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

txt = (REPO / "code" / "test_match" / "_inject_vs_runnerup.sh").read_text(encoding="utf-8")
m = re.search(r'^BASE="(.*)"$', txt, re.M)
base = {}
for kv in m.group(1).split(";"):
    if kv:
        k, v = kv.split("=", 1)
        base[k] = v
base["AZAI_FLAT_ROLLOUT_MODE"] = "idle"      # ← 关键：让 P2 与 P1 同估计器
base["AZAI_K"] = "8"
base["AZAI_FLAT_ROUNDS"] = "8192"
for k, v in base.items():
    os.environ[k] = v
os.environ["AZAI_TORCH_THREADS"] = "1"
os.environ["AZAI_CAND_MODE"] = "unified"
os.environ["AZAI_CAND_VERBOSE"] = "1"

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType
from my_ai.az_intent.game_state_facade import GameStateFacade


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rsc = _load("_rsc_reuse", REPO / "code" / "test_match" / "repro_search_crash.py")
rounds = rsc.read_ops("INJECT_VS_RU_B", "8.log")
st = GameStateFacade.initial(seed=8, cold_handle_rule_illegal=True)
snaps = []
for r in range(0, 121):
    if r in (20, 60):
        snaps.append((r, 0, st.clone()))
    for pl in (0, 1):
        ops = rounds.get(r, {}).get(pl, [])
        if ops:
            st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
    st.advance_round()

bridge = _load("az_bridge_ai_reuse", REPO / "code" / "test_match" / "az_bridge_ai.py")
searcher, _m, _f = bridge.build_engine(8, 0)
print(f"[reuse] searcher={type(searcher).__name__} rollout_mode={searcher.rollout_mode} k={searcher.k}")
for (rnd, player, snap) in snaps:
    before = int(getattr(searcher, "last_reused", 0))
    res = searcher.search(snap.clone(), player, temperature=0.0)
    after = int(getattr(searcher, "last_reused", 0))
    print(f"[reuse] r{rnd} p{player} n_cand={len(res.bundles)} "
          f"phase2_cache={'on' if getattr(searcher, '_phase2_cache', None) else 'off'} "
          f"复用样本数={after - before} n_rollouts={searcher.last_rollouts} "
          f"last_rounds={searcher.last_rounds}")
