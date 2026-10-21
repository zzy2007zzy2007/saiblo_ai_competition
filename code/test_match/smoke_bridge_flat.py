"""冒烟：通过**部署桥接**（az_bridge_ai.build_engine）构造 flat 搜索器并跑一次决策。

验证 AZAI_SEARCH=flat 这条接入路径真的能用（kwargs 透传、值网加载、时间预算）。
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

# 桥接用 AZAI_CKPT 指定权重；k/sample_mult 等沿用部署默认
os.environ.setdefault("AZAI_CKPT", str(REPO / "checkpoints/grafted_sdk_A.pt"))
os.environ["AZAI_SEARCH"] = "flat"
os.environ["AZAI_K"] = "24"
os.environ["AZAI_SAMPLE_MULT"] = "15"
os.environ["AZAI_FLAT_ROLLOUTS"] = "8"
os.environ["AZAI_FLAT_HORIZON"] = "32"
os.environ["AZAI_FLAT_BATCH"] = "32"
os.environ["AZAI_FLAT_VERBOSE"] = "1"
os.environ["AZAI_TORCH_THREADS"] = "4"
os.environ["AZAI_HOME"] = os.environ.get("AZAI_HOME", "")

spec = importlib.util.spec_from_file_location("az_bridge_ai", REPO / "code/test_match/az_bridge_ai.py")
bridge = importlib.util.module_from_spec(spec)
sys.modules["az_bridge_ai"] = bridge
spec.loader.exec_module(bridge)

from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

t0 = time.perf_counter()
searcher, model, feat = bridge.build_engine(732, 0)
print(f"[build_engine] {time.perf_counter() - t0:.1f} s，类型 {type(searcher).__name__}")

facade = GameStateFacade.initial(732)
for _ in range(200):
    facade.advance_round()
print(f"[局面] round={facade.round_index} 蚁 {len(list(facade.ants))}")

t0 = time.perf_counter()
res = searcher.search(facade, 0, temperature=0.0)
dt = time.perf_counter() - t0
print(f"[search] {dt:.2f} s  候选 {len(res.bundles)} 个  选中 {res.chosen_bundle}")
if hasattr(searcher, "last_timing"):
    print(f"[相位] expand {searcher.last_timing.get('expand_s', 0):.2f}s  "
          f"rollout {searcher.last_timing.get('rollout_s', 0):.2f}s   "
          f"每候选 {searcher.last_rollouts} 次 × {searcher.horizon} 回合 = {searcher.last_rounds} 回合")
