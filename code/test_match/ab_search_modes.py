"""A/B：**扁平 rollout 搜索 vs 原版 M6c（BundleMCTS）**，同一 ckpt、同一候选生成。

用法：
    python code/test_match/ab_search_modes.py --seeds 7 8 9 10 --jobs 4 --out-dir training_history/ab_flat

设计：
  · 两个 AI 都是 `az_bridge_ai.py`（部署桥接），只有 `AZAI_SEARCH` 不同：
        flat 侧 = AZAI_SEARCH=flat + AZAI_FLAT_*
        mcts 侧 = 默认（AZAI_SEARCH=mcts）
  · **交替执先**：seed 偶数 → flat 当 ai0；seed 奇数 → flat 当 ai1（抵消先手偏差）
  · 复用 `_tmp_ladder.read_game` 解析胜负（和既有实验同一套判据）
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "code", REPO / "Ant-Game"]:
    sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
sys.modules["_tmp_ladder"] = _ladder
_spec.loader.exec_module(_ladder)

BRIDGE = REPO / "code/test_match/az_bridge_ai.py"
# 部署口径：**逐字照抄 `M6cROOT_128`**（docs/experiment_log.md 行 6332；该臂 = 对 rule_v4 0.7344，
# 并被 docs/experiment_log.md 行 11337 明确标为"M6c-ROOT 基线"）。
#   ⚠️ 注意：不是 `grafted_sdk_A.pt` + MC_EVERY=1 + MC_TOPK=3 + MC_UPG=1 那套（那是后来的嫁接/加宽实验臂）。
DEFAULT_CANON_ENV = (
    "AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;"
    "AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;"
    "AZAI_MC_EVERY=5;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;"
    "AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off"
)
# flat 侧：同一套配置 + 只把"评估器"换成扁平 rollout
DEFAULT_FLAT_ENV = (DEFAULT_CANON_ENV + ";AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;"
                    "AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000")
DEFAULT_MCTS_ENV = (DEFAULT_CANON_ENV + ";AZAI_SEARCH=mcts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out-dir", default="training_history/ab_flat")
    ap.add_argument("--flat-env", default=DEFAULT_FLAT_ENV)
    ap.add_argument("--mcts-env", default=DEFAULT_MCTS_ENV)
    ap.add_argument("--timeout", type=float, default=4 * 3600.0)
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["PYTHONIOENCODING"] = "utf-8"

    sem = threading.Semaphore(args.jobs)
    results: dict[int, dict] = {}
    lock = threading.Lock()

    def run(seed: int) -> None:
        # seed 偶数 ⇒ flat 执 P0；奇数 ⇒ flat 执 P1
        flat_is_p0 = (seed % 2 == 0)
        env0 = args.flat_env if flat_is_p0 else args.mcts_env
        env1 = args.mcts_env if flat_is_p0 else args.flat_env
        cmd = [sys.executable, "-u", str(REPO / "code/test_match/match_via_sdk.py"),
               "--trace-ops",
               f"--ai0={BRIDGE.as_posix()}",
               f"--ai1={BRIDGE.as_posix()}",
               f"--ai0-env={env0}",
               f"--ai1-env={env1}",
               str(seed)]
        log_path = out_dir / f"seed{seed}.log"
        t0 = time.time()
        with sem:
            with open(log_path, "wb") as handle:
                proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                      env=env_base, cwd=str(REPO), timeout=args.timeout)
        elapsed = time.time() - t0
        parsed = _ladder.read_game(log_path, str(seed))
        flat_side = 0 if flat_is_p0 else 1
        winner_side = parsed.get("winner_side")
        if winner_side in (None, "None", ""):
            flat_score = 0.5
        else:
            flat_score = 1.0 if int(winner_side) == flat_side else 0.0
        with lock:
            results[seed] = {"flat_is_p0": flat_is_p0, "score": flat_score,
                             "rc": proc.returncode, "minutes": elapsed / 60,
                             "rounds": parsed.get("rounds"), "hp": parsed.get("hp"),
                             "terminal": parsed.get("terminal")}
            print(f"[seed {seed}] flat={'P0' if flat_is_p0 else 'P1'} "
                  f"score={flat_score} rc={proc.returncode} "
                  f"rounds={parsed.get('rounds')} hp={parsed.get('hp')} "
                  f"({elapsed/60:.1f} min)", flush=True)

    threads = [threading.Thread(target=run, args=(s,)) for s in args.seeds]
    t0 = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    scores = [results[s]["score"] for s in args.seeds if s in results]
    if scores:
        wins = sum(1 for x in scores if x == 1.0)
        draws = sum(1 for x in scores if x == 0.5)
        losses = sum(1 for x in scores if x == 0.0)
        rate = sum(scores) / len(scores)
        se = (0.25 / len(scores)) ** 0.5
        print(f"\n=== flat vs mcts：{len(scores)} 局  "
              f"flat 胜率 {rate:.1%}  ({wins}W/{draws}D/{losses}L)  "
              f"±{2*se:.1%}(2SE)  wall={(time.time()-t0)/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
