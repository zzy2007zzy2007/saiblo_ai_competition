"""把亚军 `generate_root_plans` 的先验（heuristic）批量抽成标签。

链路：`ents_seed*.jsonl`（逐回合状态） → stdin → `dump_plan_prior.exe`
      → 每状态的全部候选计划 + heuristic 分解。

详见 docs/plan_prior_distill.md。

用法：
    # 抽标签（一个 seed 一个输出文件）
    python dump_prior_labels.py --globs "training_history/distill_top2/abl_linear2_raw/ents_seed*.jsonl" \
        --out-dir training_history/distill_top2/prior_labels --players 0,1

    # 只看先验值分布（不落盘）
    python dump_prior_labels.py --globs "..." --stats-only

    # 与亚军自己的日志对齐（验证工具正确性）
    python dump_prior_labels.py --align-log-dir match_results --align-seed 732 --align-round 100
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
CHUNK = 400  # 每次喂给 exe 的状态数（避免管道缓冲死锁，同时摊薄启动开销）


def make_env() -> dict:
    """MinGW 编的 exe 需要 C:\\mingw64\\bin 里的 DLL。"""
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def run_states(states: list[dict], env: dict | None = None) -> list[dict]:
    """把一个 state dict 列表喂给 exe，返回逐条结果（顺序一致）。"""
    env = env or make_env()
    results: list[dict] = []
    for start in range(0, len(states), CHUNK):
        chunk = states[start:start + CHUNK]
        payload = "".join(json.dumps(item) + "\n" for item in chunk)
        proc = subprocess.run([str(EXE)], input=payload.encode("utf-8"),
                              capture_output=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"exe 退出码 {proc.returncode}: "
                               f"{proc.stderr.decode('utf-8', 'replace')[:500]}")
        lines = [ln for ln in proc.stdout.decode("utf-8", "replace").splitlines() if ln.strip()]
        if len(lines) != len(chunk):
            raise RuntimeError(f"结果条数不符: 期望 {len(chunk)} 实得 {len(lines)}")
        results.extend(json.loads(ln) for ln in lines)
    return results


def seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", path.stem)
    return int(match.group(1)) if match else None


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def track_to_json(plan: dict) -> dict:
    return {
        "k": plan["key"],
        "h": plan["heuristic"],
        "bh": plan["base_heuristic"],
        "lh": plan["lure_heuristic"],
        "gh": plan["lightning_heuristic"],
        "op": plan["operation_penalty"],
        "fp": plan["followup_penalty"],
        "hz": plan["horizon"],
        "hl": int(bool(plan["has_lightning"])),
        "f": plan["followup"],
        "ops": [[o["type"], o["arg0"], o["arg1"]] for o in plan["ops"]],
    }


def cmd_extract(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for pattern in args.globs:
        paths.extend(sorted(Path(p) for p in glob.glob(pattern)))
    if not paths:
        print("没匹配到输入文件", file=sys.stderr)
        return 2
    print(f"[输入] {len(paths)} 个文件")
    players = [int(x) for x in args.players.split(",")]
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env()

    t_start = time.time()
    states_done = 0
    plans_done = 0
    for path in paths:
        seed = seed_from_name(path) or 0
        rows = load_rows(path)
        if args.max_rounds:
            rows = rows[:args.max_rounds]
        out_path = out_dir / (path.stem.replace("ents_", "prior_") + ".jsonl") if out_dir else None
        sink = out_path.open("w", encoding="utf-8") if out_path else None
        try:
            for player in players:
                batch = []
                for row in rows:
                    state = dict(row)
                    state["seed"] = seed
                    state["player"] = player
                    batch.append(state)
                results = run_states(batch, env)
                for state, result in zip(batch, results):
                    if result.get("ok") != 1:
                        print(f"[warn] {path.name} round={state.get('round')} "
                              f"player={player} 失败: {result.get('error')}", file=sys.stderr)
                        continue
                    record = {
                        "seed": seed,
                        "round": state.get("round"),
                        "player": player,
                        "n": result["n_plans"],
                        "n_base": result["n_base"],
                        "n_lure": result["n_lure"],
                        "n_lightning": result["n_lightning"],
                        "plans": [track_to_json(pl) for pl in result["plans"]],
                    }
                    plans_done += len(record["plans"])
                    states_done += 1
                    if sink:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            if sink:
                sink.close()
        elapsed = time.time() - t_start
        print(f"[ok] {path.name}: states={states_done} plans={plans_done} "
              f"elapsed={elapsed:.1f}s ({states_done / max(elapsed, 1e-9):.2f} state/s)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for pattern in args.globs:
        paths.extend(sorted(Path(p) for p in glob.glob(pattern)))
    print(f"[输入] {len(paths)} 个文件")
    players = [int(x) for x in args.players.split(",")]
    env = make_env()
    all_nonzero_ratio = []
    all_distinct = []
    examples = []
    for path in paths[:args.limit]:
        seed = seed_from_name(path) or 0
        rows = load_rows(path)
        batch = []
        for row in rows:
            for player in players:
                state = dict(row)
                state["seed"] = seed
                state["player"] = player
                batch.append(state)
        results = run_states(batch, env)
        for result in results:
            if result.get("ok") != 1:
                continue
            values = [pl["heuristic"] for pl in result["plans"]]
            if not values:
                continue
            nonzero = sum(1 for v in values if v != 0)
            all_nonzero_ratio.append(nonzero / len(values))
            all_distinct.append(len(set(values)))
            if len(examples) < 6:
                counter = collections.Counter(values)
                examples.append({
                    "seed": seed,
                    "round": result["round_index"],
                    "player": result["player"],
                    "n": len(values),
                    "nonzero": nonzero,
                    "distinct": len(set(values)),
                    "max": max(values),
                    "top": counter.most_common(5),
                })
        print(f"[ok] {path.name}")
    if all_nonzero_ratio:
        print(f"\n状态数={len(all_nonzero_ratio)}")
        print(f"非零比例: 均值 {statistics.mean(all_nonzero_ratio):.3f} "
              f"中位 {statistics.median(all_nonzero_ratio):.3f} "
              f"min {min(all_nonzero_ratio):.3f} max {max(all_nonzero_ratio):.3f}")
        print(f"不同取值个数: 均值 {statistics.mean(all_distinct):.2f} "
              f"中位 {statistics.median(all_distinct)} max {max(all_distinct)}")
    for item in examples:
        print(json.dumps(item, ensure_ascii=False))
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    """拿亚军自己的 plan 日志核对：同一 (seed, round, player) 的 key 集与 heuristic 是否逐位一致。"""
    log_paths = sorted(Path(args.align_log_dir).glob(f"*runnerup_seed{args.align_seed}*.stderr.log"))
    if not log_paths:
        print("找不到日志", file=sys.stderr)
        return 2
    log_path = log_paths[0]
    print(f"[日志] {log_path}")
    target_round = args.align_round
    log_plans: dict[int, dict[str, dict]] = {0: {}, 1: {}}
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "base_heuristic" not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("round") != target_round:
                continue
            log_plans.setdefault(record["player"], {})[record["key"]] = record
    print(f"[日志] round={target_round} player0={len(log_plans[0])} player1={len(log_plans[1])}")

    rows_paths = sorted(Path(args.globs[0]).parent.glob(f"ents_*{args.align_seed}.jsonl"))
    if not rows_paths:
        print("找不到该 seed 的状态文件", file=sys.stderr)
        return 2
    rows = load_rows(rows_paths[0])
    row = rows[target_round]

    ok = True
    checked = 0
    for player in (0, 1):
        if not log_plans.get(player):
            # ai0/ai1 各自一份日志，只有一个玩家有数据
            print(f"player={player} 日志无记录，跳过")
            continue
        checked += 1
        state = dict(row)
        state["seed"] = args.align_seed
        state["player"] = player
        result = run_states([state])[0]
        if result.get("ok") != 1:
            print(f"[失败] player={player}: {result.get('error')}")
            ok = False
            continue
        mine = {pl["key"]: pl["heuristic"] for pl in result["plans"]}
        theirs = {k: float(v["heuristic"]) for k, v in log_plans[player].items()}
        common = set(mine) & set(theirs)
        diffs = [abs(mine[k] - theirs[k]) for k in common]
        print(f"player={player} 我的={len(mine)} 日志={len(theirs)} 交集={len(common)} "
              f"仅我有={len(set(mine) - set(theirs))} 仅日志={len(set(theirs) - set(mine))} "
              f"max_diff={max(diffs) if diffs else 0}")
        if diffs and max(diffs) > 1e-6:
            bad = sorted(common, key=lambda k: -abs(mine[k] - theirs[k]))[:5]
            for key in bad:
                print(f"   {key}: mine={mine[key]} log={theirs[key]}")
            ok = False
        if set(mine) != set(theirs):
            ok = False
    print(f"[对齐] 核对了 {checked} 个玩家 -> " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--globs", nargs="+", required=True)
    parser.add_argument("--players", default="0,1")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2, help="stats 模式最多看几个文件")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--align-log-dir", default=None)
    parser.add_argument("--align-seed", type=int, default=732)
    parser.add_argument("--align-round", type=int, default=100)
    args = parser.parse_args()
    if args.align_log_dir:
        return cmd_align(args)
    if args.stats_only:
        return cmd_stats(args)
    return cmd_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
