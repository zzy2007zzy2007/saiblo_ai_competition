"""一次性探针：**冠军当 oracle 时的单次查询延迟**（2026-10-28；给 E2/E4 的 `AZAI_ORACLE_TIMEOUT=20`
一个实测依据）。

做法：`RunnerUpOracle(seed=7, player=0, exe=冠军, timeout=20)` 从 r0 连续喂 30 回合
（协议要求连续喂，跳回合会读端错位），把每次 ask 的墙钟打出来。

⚠️ 它**不参与**任何正式实验；跑完可删。冠军 exe 的 PATH 修正由 `match_via_sdk.start_ai` 负责。
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent      # 本文件在仓库根 ⇒ parent 就是仓库根
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
PLAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEED = 7


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    from my_ai.az_intent.game_state_facade import GameStateFacade

    ro = _load("runnerup_oracle", REPO / "code" / "test_match" / "runnerup_oracle.py")
    champ = REPO / "其他版本ai" / "ant-war2-magica-v3" / "magica_v3_O0.exe"
    print(f"[lat] 冠军 oracle: {champ.name} seed={SEED} player={PLAYER} rounds={ROUNDS}")
    ms: list[float] = []
    with ro.RunnerUpOracle(seed=SEED, player=PLAYER, exe=str(champ), timeout=20.0) as orc:
        st = GameStateFacade.initial(seed=SEED, cold_handle_rule_illegal=True)
        for r in range(ROUNDS):
            if st.terminal:
                print(f"[lat] r{r} 已终局，提前结束")
                break
            t0 = time.monotonic()
            try:
                ops = orc.ask(st)
            except Exception as exc:  # noqa: BLE001
                print(f"[lat] r{r} 失败: {type(exc).__name__}: {exc}")
                return 1
            dt = (time.monotonic() - t0) * 1000.0
            ms.append(dt)
            print(f"[lat] r{r:<3} {dt:8.1f} ms  ops={[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops]}")
            if ops:
                st.apply_operation_list(PLAYER, list(ops))
            st.advance_round()
    if ms:
        print(f"[lat] 汇总 n={len(ms)} p50={statistics.median(ms):.0f}ms "
              f"mean={statistics.fmean(ms):.0f}ms max={max(ms):.0f}ms "
              f"（E2/E4 的 AZAI_ORACLE_TIMEOUT=20s ≈ {20000.0 / max(1.0, max(ms)):.0f}× 本批最大值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
