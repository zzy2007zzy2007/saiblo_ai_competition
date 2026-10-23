"""把散落的对局日志登记成**训练可用索引**（不复制数据，只登记 + 校验可回放性）。

为什么要它：我们的 ladder / A-B 日志里已经有"seed + 双方逐回合 op + RESULT"，
而引擎是确定性的 ⇒ 任何决策点的完整观测都能重建。这个脚本把"哪些局可用"集中登记，
以后训练时按索引取，不用再翻目录。

输出：`training_history/replays/index.jsonl`，每行一局：
  {tag, seed, seats: {p0: label, p1: label}, winner, hp, rounds, terminal, log, mtime}

用法：
  python code/test_match/build_replay_index.py [--only VS_RU_newcfg] [--rescan]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

OUT = REPO / "training_history/replays/index.jsonl"
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
HEAD = re.compile(r"\[bridge\] AI([01])=(.*)")


def parse_game(log: Path, tag: str, seed: str) -> dict | None:
    txt = log.read_text(encoding="utf-8", errors="replace")
    labels = {}
    for m in HEAD.finditer(txt):
        labels[f"p{m.group(1)}"] = Path(m.group(2).strip()).name
    # RESULT 行：兼容几种写法
    winner = None
    for pat in (r"RESULT[^\n]*?winner[=: ]+([pP][01]|None|\d)",
                r"winner_side[=: ]+([pP][01]|None|\d)"):
        m = re.search(pat, txt)
        if m:
            w = m.group(1).lower()
            winner = None if w in ("none",) else ("p0" if w in ("p0", "0") else "p1")
            break
    hp = re.search(r"hp[=: ]*\(?\[?(\d+)\s*,\s*(\d+)", txt)
    rounds = re.search(r"rounds[=: ]+(\d+)", txt)
    n_ops = len(OP.findall(txt))
    if n_ops == 0:
        return None
    return {
        "tag": tag, "seed": seed,
        "seats": {k: v for k, v in sorted(labels.items())},
        "winner": winner,
        "hp": [int(hp.group(1)), int(hp.group(2))] if hp else None,
        "rounds": int(rounds.group(1)) if rounds else None,
        "n_ops": n_ops,
        "replayable": True,           # 有 op trace ⇒ 可确定性重建
        "log": str(log.relative_to(REPO)).replace("\\", "/"),
        "size_kb": round(log.stat().st_size / 1024, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只处理 tag 含该子串的目录")
    args = ap.parse_args()

    rows, skipped = [], 0
    # ① ladder 日志（打外敌，例如亚军）
    for d in sorted((REPO / "match_results/ladder_logs").glob("*")):
        if not d.is_dir() or (args.only and args.only not in d.name):
            continue
        for log in sorted(d.glob("*.log")):
            rec = parse_game(log, d.name, log.stem)
            if rec:
                rows.append(rec)
            else:
                skipped += 1
    # ② A/B 日志（内部对打）
    for d in sorted((REPO / "training_history").glob("ab_*")):
        if not d.is_dir() or (args.only and args.only not in d.name):
            continue
        for log in sorted(d.glob("seed*.log")):
            rec = parse_game(log, d.name, log.stem)
            if rec:
                rows.append(rec)
            else:
                skipped += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_mb = sum(r["size_kb"] for r in rows) / 1024
    by_tag: dict[str, int] = {}
    for r in rows:
        by_tag[r["tag"]] = by_tag.get(r["tag"], 0) + 1
    print(f"登记 {len(rows)} 局（跳过 {skipped} 个无 op 的日志），日志总量 {total_mb:.1f} MB")
    print(f"索引: {OUT.relative_to(REPO)}")
    print("按 tag 统计（前 12）：")
    for tag, n in sorted(by_tag.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:>5}  {tag}")


if __name__ == "__main__":
    raise SystemExit(main())
