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
    # 2026-10-22：同图交换座位（配对）。实测 P1 系统性占优 +19~31pp，
    # 旧逻辑是"不同地图各打一次" ⇒ 座位方差全进噪声。
    ap.add_argument("--paired-sides", action="store_true",
                    help="每个 seed 打两局，A 侧分别坐 P0/P1（同一张地图）")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["PYTHONIOENCODING"] = "utf-8"
    # 2026-10-23（用户要求）：逐回合进度文件用这个 tag（match_via_sdk 会写
    #   training_history/progress/<tag>_seed<seed>.txt）
    TAG = out_dir.name
    env_base["DSH_PROGRESS_TAG"] = TAG
    try:
        import progress as _progress
    except Exception:            # 进度输出绝不影响对局
        _progress = None
    T_START = time.time()

    sem = threading.Semaphore(args.jobs)
    results: dict[int, dict] = {}
    lock = threading.Lock()
    INVALID = {"n": 0}   # 2026-10-22：无结果/异常局计数（原来直接丢，不可见）

    def run(seed: int, flat_side_in: int | None = None) -> None:
        # 默认沿用旧约定（seed 偶数 ⇒ flat 执 P0）；配对模式下由调用方显式指定
        flat_is_p0 = (seed % 2 == 0) if flat_side_in is None else (flat_side_in == 0)
        env0 = args.flat_env if flat_is_p0 else args.mcts_env
        env1 = args.mcts_env if flat_is_p0 else args.flat_env
        cmd = [sys.executable, "-u", str(REPO / "code/test_match/match_via_sdk.py"),
               "--trace-ops",
               f"--ai0={BRIDGE.as_posix()}",
               f"--ai1={BRIDGE.as_posix()}",
               f"--ai0-env={env0}",
               f"--ai1-env={env1}",
               str(seed)]
        log_path = (out_dir / (f"seed{seed}.log" if flat_side_in is None
                                  else f"seed{seed}_p{0 if flat_is_p0 else 1}.log"))
        # 2026-10-23：配对模式同一 seed 打两局 ⇒ 进度文件必须带座位后缀，否则互相覆盖
        env_game = dict(env_base)
        env_game["DSH_PROGRESS_SUFFIX"] = "" if flat_side_in is None else f"_p{0 if flat_is_p0 else 1}"
        t0 = time.time()
        with sem:
            with open(log_path, "wb") as handle:
                proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                      env=env_game, cwd=str(REPO), timeout=args.timeout)
        elapsed = time.time() - t0
        parsed = _ladder.read_game(log_path, str(seed))
        flat_side = 0 if flat_is_p0 else 1
        winner_side = parsed.get("winner_side")
        # 新格式的 winner_side 是 'p0'/'p1' 字符串；旧格式是 None/缺失 ⇒ 按平局记
        # 2026-10-22 修：`_ladder.read_game` 在"没打完/无结果"时会返回**字符串 'None'**，
        # 而这里原来直接 int() ⇒ ValueError ⇒ 该局线程崩掉、结果丢失（实测一次丢 6/32 局）。
        if isinstance(winner_side, str):
            _ws = winner_side.strip().lower()
            if _ws in ("p0", "p1"):
                winner_side = int(_ws[1])
            elif _ws in ("none", "null", ""):
                winner_side = None
                with lock:
                    INVALID["n"] += 1
            else:
                try:
                    winner_side = int(_ws)
                except ValueError:
                    winner_side = None
                    with lock:
                        INVALID["n"] += 1
        elif winner_side is not None:
            winner_side = int(winner_side)
        if winner_side is None:
            flat_score = 0.5
        else:
            flat_score = 1.0 if int(winner_side) == flat_side else 0.0
        with lock:
            key = seed if flat_side_in is None else (seed, 0 if flat_is_p0 else 1)
            results[key] = {"flat_is_p0": flat_is_p0, "score": flat_score,
                             "rc": proc.returncode, "minutes": elapsed / 60,
                             "rounds": parsed.get("rounds"), "hp": parsed.get("hp"),
                             "terminal": parsed.get("terminal")}
            print(f"[seed {seed}] flat={'P0' if flat_is_p0 else 'P1'} "
                  f"score={flat_score} rc={proc.returncode} "
                  f"rounds={parsed.get('rounds')} hp={parsed.get('hp')} "
                  f"({elapsed/60:.1f} min)", flush=True)
            # 2026-10-23：整批进度文件（每完成一局覆盖重写，用户可随时打开看）
            if _progress is not None:
                try:
                    _n = len(results)
                    _w = sum(1 for v in results.values() if v["score"] >= 1.0)
                    _l = sum(1 for v in results.values() if v["score"] == 0.0)
                    _recent = [
                        f"seed{k} {'P0' if v['flat_is_p0'] else 'P1'} "
                        f"{'胜' if v['score'] >= 1.0 else ('负' if v['score'] == 0.0 else '平')} "
                        f"hp={v.get('hp')} ({v.get('rounds')} 回合)"
                        for k, v in list(results.items())[-6:]]
                    _inflight = []
                    for f in (REPO / "training_history/progress").glob(f"{TAG}_seed*.txt"):
                        try:
                            last = f.read_text(encoding="utf-8").strip().splitlines()[-1]
                            if last.startswith("round="):
                                _inflight.append(int(last.split("=")[1].split()[0]))
                        except Exception:
                            pass
                    _progress.write_status(
                        TAG, total=len(tasks), done=_n, wins=_w, losses=_l,
                        invalid=INVALID["n"], started=T_START,
                        inflight_rounds=_inflight, recent=_recent)
                except Exception:
                    pass

    if args.paired_sides:
        tasks = [(s, side) for s in args.seeds for side in (0, 1)]
    else:
        tasks = [(s, None) for s in args.seeds]
    threads = [threading.Thread(target=run, args=t) for t in tasks]
    t0 = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    scores = [v["score"] for v in results.values()]
    if scores:
        wins = sum(1 for x in scores if x == 1.0)
        draws = sum(1 for x in scores if x == 0.5)
        losses = sum(1 for x in scores if x == 0.0)
        rate = sum(scores) / len(scores)
        se = (0.25 / len(scores)) ** 0.5
        print(f"\n=== flat vs mcts：{len(scores)} 局  "
              f"flat 胜率 {rate:.1%}  ({wins}W/{draws}D/{losses}L)  "
              f"±{2*se:.1%}(2SE)  wall={(time.time()-t0)/60:.1f} min  "
              f"无效局={INVALID[chr(110)]} ===", flush=True)
        # 2026-10-22：先/后手拆分（座位诊断）。P1 系统性占优时，两条臂若不配对就会互相污染。
        for _side, _name in ((0, "P0"), (1, "P1")):
            _s = [v["score"] for v in results.values() if v["flat_is_p0"] == (_side == 0)]
            if _s:
                _w = sum(1 for x in _s if x == 1.0)
                print(f"    A 在 {_name}: {_w}W-{len(_s)-_w}L = {sum(_s)/len(_s):.1%} (n={len(_s)})",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
