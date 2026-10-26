"""探针：批量把 ladder 日志的出招序列用 GameStateFacade 重放，和日志 RESULT 对拍。

目的：量化"重放不忠实"到底有多少局、什么条件下发生。
不需要编译任何东西（纯 in-process 引擎 B）。

用法：
  python code/test_match/probe_replay_batch.py                       # 默认扫 VS_RUNNERUP_cells4
  python code/test_match/probe_replay_batch.py <dir> [<dir> ...] --limit 40
  python code/test_match/probe_replay_batch.py ... --cold 0          # 用 cold_handle_rule_illegal=False
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_replay_engines import (parse_log, parse_ops_text, first_div_towers,  # noqa: E402
                                  run_facade)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path,
                    default=[REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4"])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--cold", type=int, default=1, choices=(0, 1))
    args = ap.parse_args()

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    logs: list[Path] = []
    for d in args.dirs:
        if d.is_file():
            logs.append(d)
        else:
            logs.extend(sorted(d.glob("*.log")))
    logs = logs[: args.limit]

    n_ok = n_bad = n_skip = 0
    bad_list = []
    tower_bad = []
    for lg in logs:
        info = parse_log(lg)
        if info["seed"] is None or info["result"] is None:
            n_skip += 1
            print(f"  skip {lg.name}: seed={info['seed']} result={info['result']}")
            continue
        n_rounds = info["result"]["rounds"]
        rows = []
        ok = True
        for r in range(n_rounds):
            d = info["ops_text"].get(r, {})
            if 0 not in d or 1 not in d:
                ok = False
                break
            rows.append({p: parse_ops_text(d[p], Operation, OperationType) for p in (0, 1)})
        if not ok:
            n_skip += 1
            print(f"  skip {lg.name}: 日志缺某些回合的出招")
            continue
        try:
            fr = run_facade(rows, n_rounds, info["seed"], cold=bool(args.cold))
        except Exception as exc:  # noqa: BLE001
            n_skip += 1
            print(f"  skip {lg.name}: facade {type(exc).__name__}: {exc}")
            continue
        got = fr[-1]["hp"]
        want = info["result"]["hp"]
        tw = first_div_towers(fr, info["towers"])
        if got == want:
            n_ok += 1
            print(f"  OK   {lg.name:10s} seed={info['seed']:3d} rounds={n_rounds:3d} "
                  f"hp={got}  towers={'ok' if tw is None else f'DIFF@{tw[0]}'}")
            if tw is not None:
                tower_bad.append((lg.name, tw[0]))
        else:
            n_bad += 1
            bad_list.append(lg.name)
            print(f"  BAD  {lg.name:10s} seed={info['seed']:3d} rounds={n_rounds:3d} "
                  f"facade={got} log={want}  towers={'ok' if tw is None else f'DIFF@{tw[0]}'}")

    print(f"\n== cold={args.cold}: OK={n_ok}  BAD={n_bad}  SKIP={n_skip} / {len(logs)} logs ==")
    if bad_list:
        print("BAD:", bad_list)
    if tower_bad:
        print("塔快照先于 RESULT 就已偏离:", tower_bad[:10])
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
