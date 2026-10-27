"""**部署链路冒烟**：只用环境变量（不动任何命令行）把"引擎 RNG 消费"打开，
走 `az_bridge_ai.build_engine()` 真实构造路径，确认

1. `AZAI_FLAT_RNG_PERTURB=1` 能让 AI 进程用上 RNG 变体引擎（`rng_build/native_game.pyd`）；
2. 建局（`GameStateFacade.initial`）拿到的状态**支持** `consume_rng`；
3. `search()` 真的消费了 RNG（`last_rng_consumed > 0`），且消费只发生在 rollout 克隆上
   —— 裁判/本局状态不受影响（这里用"search 前后 `state` 的 rng_state 不变"来验证）；
4. 前缀树路径（`AZAI_FLAT_SHARE_PREFIX=1`）也生效（`last_tree_stats` 有值）。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P code/test_match/smoke_rng_perturb_deploy.py                 # 打开 perturb
AZAI_FLAT_RNG_PERTURB=0 $P code/test_match/smoke_rng_perturb_deploy.py   # 关闭（对照）
```
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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

# 冠军配置的 flat 口径（只改随机性来源相关项）
ENV = {
    "AZAI_CKPT": "checkpoints/posnet_A_k5_m32.pt",
    "AZAI_SEARCH": "flat",
    "AZAI_K": "48",
    "AZAI_FLAT_ROLLOUTS": "64",
    "AZAI_FLAT_HORIZON": "16",
    "AZAI_FLAT_ROUNDS": "1000000000",
    "AZAI_FLAT_ROLLOUT_MODE": "idle",
    "AZAI_FLAT_SHARE_PREFIX": "1",
    "AZAI_FLAT_PROB_MODE": "const",
    "AZAI_FLAT_ACT_PROB": "0.5",
    "AZAI_FLAT_SEED": "53",
    "AZAI_MODE": "pos-only",
    "AZAI_TORCH_THREADS": "1",
    "AZAI_ROLLOUT_SAMPLES": "1",
}


def main() -> int:
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    perturb = os.environ.get("AZAI_FLAT_RNG_PERTURB", "0")
    import importlib.util

    spec = importlib.util.spec_from_file_location("az_bridge_ai",
                                                  REPO / "code/test_match/az_bridge_ai.py")
    bridge = importlib.util.module_from_spec(spec)
    sys.modules["az_bridge_ai"] = bridge
    assert spec.loader is not None
    spec.loader.exec_module(bridge)

    from my_ai.az_intent import rng_engine
    print(f"[smoke] AZAI_FLAT_RNG_PERTURB={perturb}  rng_build 已构建={rng_engine.built()}",
          flush=True)
    mcts, _model, _feat = bridge.build_engine(10, 0)
    print(f"[smoke] searcher={type(mcts).__name__} "
          f"rollout_rng_perturb={getattr(mcts, 'rollout_rng_perturb', None)} "
          f"_rng_on={getattr(mcts, '_rng_on', None)}", flush=True)

    from my_ai.az_intent.game_state_facade import GameStateFacade
    st = GameStateFacade.initial(seed=10, cold_handle_rule_illegal=True)
    for _ in range(60):
        st.advance_round()
    g = getattr(st, "_g", st)
    rng_before = int(g.get_rng_state()) if hasattr(g, "get_rng_state") else None
    print(f"[smoke] 建局引擎支持 consume_rng={hasattr(g, 'consume_rng')} "
          f"rng_state={rng_before}  {rng_engine.describe()}", flush=True)

    res = mcts.search(st, 0)
    rng_after = int(g.get_rng_state()) if hasattr(g, "get_rng_state") else None
    stats = getattr(mcts, "last_tree_stats", None)
    print(f"[smoke] 候选={len(res.bundles)} chosen={res.chosen_index} "
          f"tree(node,leaf,uniq)={stats} "
          f"rng_consumed={getattr(mcts, 'last_rng_consumed', None)} "
          f"rng_missing={getattr(mcts, 'last_rng_missing', None)}", flush=True)
    print(f"[smoke] 本局状态 rng_state: before={rng_before} after={rng_after} "
          f"-> {'未被 rollout 影响 ✅' if rng_before == rng_after else '被改了 ❌'}",
          flush=True)
    ok = True
    if perturb == "1":
        ok &= bool(getattr(mcts, "_rng_on", False))
        ok &= hasattr(g, "consume_rng")
        ok &= int(getattr(mcts, "last_rng_consumed", 0)) > 0
        ok &= stats is not None
    ok &= (rng_before == rng_after)
    print(f"[smoke] 结论: {'✅ 通过' if ok else '❌ 失败'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
