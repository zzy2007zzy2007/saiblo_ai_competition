"""按动作类型的 recall 统一报告：口径 A（亚军决策点）+ 口径 B（我们 80 局留出）（2026-10-27）。

合并 / 取代了三份重叠代码：
  · `compare_priors_on_runnerup.py` —— 口径 A 的原始实现。本脚本 **import 它**，复用其
    `parse_log` / `collect_decisions` 的日志重放与口径（不重写解析）。
  · `eval_per_op_recall.py` / `eval_multi_prior.py` —— 口径 B 的原始实现，已在头部标 DEPRECATED。
  · `my_ai/train_prior_z.py:recall_eval` 保持不动（训练内环用，不属于"报告"）。

两套口径的差别（**别把 A、B 的数字直接横比**）：
  A 亚军决策点：状态由亚军对局日志重放得到；支撑集 = 该动作类的**合法格数**（position mask）；
               目标 = **亚军实际走法**，且只在"重放后仍然合法"的决策点上统计。
  B 我们 80 局留出：状态取自 `training_history/prior_labels/g9376-g9379.npz`；
               支撑集 = npz 里该动作组**出现过的候选格数**；目标 = **教师 argmax 候选**。

共同口径（这是"统一"的实质）：
  · 动作类型固定 6 类：建塔 11 / 降级 13 / 闪电 21 / EMP 22 / deflector 23 / evasion 24，
    样本为 0 的类型也打印一行 `n=0`；
  · `--min-cells` 可配（默认 2；设 1 等于不设门槛，可复现旧脚本）；
  · 每行都给随机基线（`1/n`、`min(4,n)/n`、`min(24,n)/n` 的均值，按该行支撑集逐决策点平均）；
  · `n < --insufficient-n`（默认 20）的行标注 `样本不足`；
  · `--json-out` 落一份 JSON，便于以后逐项比较。

用法：
    python code/test_match/eval_recall_report.py \\
        --nets posnet_A=checkpoints/posnet_A_k5_m32.pt,prior_hard_ce_w1=checkpoints/prior_hard_ce_w1.pt \\
        --min-cells 2 --json-out docs/eval_recall_report_20261027.json

    只跑一套口径：`--only A` / `--only B`
    复现旧脚本（无门槛）：`--min-cells 1`
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game",
           REPO / "Ant-Game" / "SDK", REPO / "code" / "test_match"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models
from my_ai.train_prior_z import load_games

# 口径 A 的重放/解析：直接复用旧脚本，避免第三份副本
import compare_priors_on_runnerup as _runnerup

KS = (1, 4, 24)
OPS = (11, 13, 21, 22, 23, 24)
CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}
NAME_ZH = {11: "建塔", 13: "降级", 21: "闪电", 22: "EMP", 23: "deflector", 24: "evasion"}
assert CH == _runnerup.CH, "CH 与 compare_priors_on_runnerup 不一致，口径 A 会算错"

DEFAULT_NETS = (
    "posnet_A=checkpoints/posnet_A_k5_m32.pt,"
    "prior_hybrid_z_only=checkpoints/prior_hybrid_z_only.pt,"
    "prior_hard_ce_w1=checkpoints/prior_hard_ce_w1.pt,"
    "prior_hard_ce_w3=checkpoints/prior_hard_ce_w3.pt"
)
DEFAULT_LOGS_DIR = "match_results/ladder_logs/VS_RUNNERUP_cells4"
DEFAULT_HOLDOUT = "g9376,g9377,g9378,g9379"
HOLDOUT_DIR = "training_history/prior_labels"

NOTE = ("NOTE 降级类（op=13）样本量通常极小，支撑集常常只有 1-3 格 ⇒ @4/@24 接近平凡；"
        "n<{n} 的行标注「样本不足」，比例不可过度解读。")


def p(*args, **kwargs) -> None:
    print(*args, flush=True, **kwargs)


# ─── 宽度感知的表格对齐（中英混排时列不会错位）────────────────────────────
def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _ljust(s: str, n: int) -> str:
    return s + " " * max(0, n - _w(s))


def _rjust(s: str, n: int) -> str:
    return " " * max(0, n - _w(s)) + s


# ─── 统一的"决策点"表示：两套口径只在"排名域 + 目标"上不同 ──────────────────
@dataclass
class Decision:
    """一个待评估的决策点。

    board/stats：网络输入；op：动作类型；
    A 口径用 mask + target（在合法格内排名，目标是实际走法）；
    B 口径用 cells + vals（只在候选格内排名，目标是教师 argmax）。
    """
    board: np.ndarray
    stats: np.ndarray
    op: int
    mask: np.ndarray | None = None
    target: tuple[int, int] | None = None
    cells: list[tuple[int, int]] | None = None
    vals: list[float] | None = None

    def support(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else len(self.cells)  # type: ignore[arg-type]

    def rank(self, sc: np.ndarray) -> int:
        """1-based 名次。两种域各自的写法与旧脚本逐位一致。"""
        if self.mask is not None:                      # 口径 A
            v = np.where(self.mask, sc, -1e9).reshape(-1)
            tgt = int(self.target[0]) * 19 + int(self.target[1])  # type: ignore[index]
            return int(np.where(np.argsort(-v) == tgt)[0][0]) + 1
        s = np.asarray([sc[x, y] for (x, y) in self.cells], dtype=sc.dtype)   # 口径 B
        v = np.asarray(self.vals, dtype=np.float64)
        return int(np.where(np.argsort(-s) == int(np.argmax(v)))[0][0]) + 1


def build_runnerup_decisions(logs) -> list[Decision]:
    """口径 A：复用 compare_priors_on_runnerup.collect_decisions 的重放结果。"""
    raw = _runnerup.collect_decisions(logs)
    return [
        Decision(board=np.asarray(b, np.float32), stats=np.asarray(s, np.float32), op=int(op),
                 mask=np.asarray(mask, bool), target=(int(cell[0]), int(cell[1])))
        for b, s, op, cell, mask in raw
    ]


def build_holdout_decisions(files) -> tuple[list[Decision], int, int]:
    """口径 B：复用 my_ai.train_prior_z.load_games 的窗口与候选。

    → (决策点, 被跳过的"不支持动作组"数, npz 窗口数)
    """
    games = load_games([str(f) for f in files])
    out: list[Decision] = []
    n_unsupported = 0
    for board, stats, targets in games:
        for op, cells in targets.items():
            if int(op) not in CH:
                n_unsupported += 1
                continue
            keys = list(cells.keys())
            out.append(Decision(board=np.asarray(board, np.float32), stats=np.asarray(stats, np.float32),
                                op=int(op), cells=keys, vals=[cells[k] for k in keys]))
    return out, n_unsupported, len(games)


# ─── 评估 ────────────────────────────────────────────────────────────────
def support_table(decisions: list[Decision], min_cells: int) -> tuple[dict, dict]:
    kept = {op: [] for op in OPS}
    dropped = {op: 0 for op in OPS}
    for d in decisions:
        s = d.support()
        if s >= min_cells:
            kept[d.op].append(s)
        else:
            dropped[d.op] += 1
    return kept, dropped


def baseline(sups: list[int]) -> dict[str, float] | None:
    """随机基线的逐行期望：mean(1/n)、mean(min(4,n)/n)、mean(min(24,n)/n)。"""
    if not sups:
        return None
    return {str(k): sum(min(k, s) / max(1, s) for s in sups) / len(sups) for k in KS}


def evaluate(net_path: str, decisions: list[Decision], min_cells: int,
             dev: torch.device) -> tuple[dict, dict]:
    _, model, _ = load_three_models(str(REPO / net_path))
    model.to(dev)
    model.eval()
    n = {op: 0 for op in OPS}
    hit = {op: {k: 0 for k in KS} for op in OPS}
    with torch.no_grad():
        for d in decisions:
            if d.support() < min_cells:
                continue
            amap = model(torch.from_numpy(d.board).unsqueeze(0).to(dev),
                         torch.from_numpy(d.stats).unsqueeze(0).to(dev))["action_map"][0]
            amap = amap.detach().cpu()
            sc = (amap[0:16].mean(0) if d.op == 11 else amap[CH[d.op][0]]).numpy()
            r = d.rank(sc)
            n[d.op] += 1
            for k in KS:
                hit[d.op][k] += int(r <= k)
    return n, hit


def eval_caliber(key: str, label: str, semantics: str, source: dict, decisions: list[Decision],
                 nets: list[tuple[str, str]], min_cells: int, dev: torch.device) -> dict:
    t0 = time.time()
    kept, dropped = support_table(decisions, min_cells)
    per_net: dict[str, tuple[dict, dict]] = {}
    for name, path in nets:
        per_net[name] = evaluate(path, decisions, min_cells, dev)
        p(f"    [ok] {name}")

    ops_json = {}
    for op in OPS:
        sups = kept[op]
        base = baseline(sups)
        med = statistics.median(sups) if sups else None
        nets_json = {}
        for name, _ in nets:
            n_op = per_net[name][0][op]
            hits = per_net[name][1][op]
            nets_json[name] = {
                "n": n_op,
                "hit": {str(k): hits[k] for k in KS},
                "recall": {str(k): (hits[k] / n_op if n_op else None) for k in KS},
                "support_median": med,
                "random": base,
                "insufficient": bool(0 < n_op < ARGS.insufficient_n),
            }
        ops_json[NAME[op]] = {
            "op": op, "name_zh": NAME_ZH[op],
            "n": len(sups), "support_median": med,
            "support_min": min(sups) if sups else None,
            "support_max": max(sups) if sups else None,
            "random": base,
            "dropped_by_min_cells": dropped[op],
            "nets": nets_json,
        }

    all_sups = [s for op in OPS for s in kept[op]]
    overall_nets = {}
    for name, _ in nets:
        n_tot = sum(per_net[name][0][op] for op in OPS)
        hit_tot = {k: sum(per_net[name][1][op][k] for op in OPS) for k in KS}
        overall_nets[name] = {
            "n": n_tot,
            "hit": {str(k): hit_tot[k] for k in KS},
            "recall": {str(k): (hit_tot[k] / n_tot if n_tot else None) for k in KS},
            "random": baseline(all_sups),
            "insufficient": bool(0 < n_tot < ARGS.insufficient_n),
        }

    return {
        "key": key,
        "label": label,
        "semantics": semantics,
        "source": source,
        "min_cells": min_cells,
        "decisions_total": len(decisions),
        "decisions_kept": len(all_sups),
        "dropped_by_min_cells_total": sum(dropped.values()),
        "elapsed_sec": round(time.time() - t0, 1),
        "ops": ops_json,
        "overall": {"n": len(all_sups), "support_median": statistics.median(all_sups) if all_sups else None,
                    "random": baseline(all_sups), "nets": overall_nets},
    }


# ─── 打印 ────────────────────────────────────────────────────────────────
def print_caliber(res: dict, names: list[str]) -> None:
    W = max(20, max(_w(n) for n in names) + 2)
    p("")
    p(f"{'=' * 96}")
    p(f"口径 {res['key']}：{res['label']}")
    p(f"  语义  {res['semantics']}")
    for k, v in res["source"].items():
        p(f"  {k:<8}{v}")
    p(f"  决策点 {res['decisions_total']} 个，全部网络一致；min-cells={res['min_cells']} "
      f"过滤掉 {res['dropped_by_min_cells_total']} 个")
    p("  " + NOTE.format(n=ARGS.insufficient_n))

    for op in OPS:
        d = res["ops"][NAME[op]]
        sups_n = d["n"]
        head = f"  -- {NAME_ZH[op]} {NAME[op]}（op={op}）  n={sups_n}"
        if d["support_median"] is not None:
            head += f"  支撑集中位={d['support_median']:.0f}（{d['support_min']}-{d['support_max']}）"
        if d["dropped_by_min_cells"]:
            head += f"  被 min-cells 丢弃 {d['dropped_by_min_cells']}"
        p("")
        p(head)
        if sups_n == 0:
            why = "该类决策点全部被 min-cells 过滤" if d["dropped_by_min_cells"] else "该口径样本中未出现此动作类型"
            p(f"    {_ljust('（无样本）', W)} {_rjust('n=0', 6)}   -- {why}；仍列出以保持 6 类动作覆盖")
            continue
        p(f"    {_ljust('网络', W)} {_rjust('n', 6)} {_rjust('支撑中位', 9)} "
          f"{_rjust('@1', 7)} {_rjust('@4', 7)} {_rjust('@24', 7)}  "
          f"{_rjust('随机@1', 7)} {_rjust('随机@4', 7)} {_rjust('随机@24', 8)}")
        for name in names:
            row = d["nets"][name]
            n = row["n"]
            if n == 0:
                p(f"    {_ljust(name, W)} {_rjust('0', 6)}   （该网络无样本）")
                continue
            r, b = row["recall"], row["random"] or {}
            pct = {k: f"{100 * r[str(k)]:.1f}%" for k in KS}
            rnd = {k: f"{100 * b[str(k)]:.1f}%" for k in KS}
            med = f"{d['support_median']:.0f}"
            line = (f"    {_ljust(name, W)} {_rjust(str(n), 6)} {_rjust(med, 9)} "
                    f"{_rjust(pct[1], 7)} {_rjust(pct[4], 7)} {_rjust(pct[24], 7)}  "
                    f"{_rjust(rnd[1], 7)} {_rjust(rnd[4], 7)} {_rjust(rnd[24], 8)}")
            if row["insufficient"]:
                line += "  <- 样本不足"
            p(line)

    ov = res["overall"]
    b = ov["random"] or {}
    rnd = {k: f"{100 * b[str(k)]:.1f}%" for k in KS}
    p("")
    p(f"  -- 总体（6 类合并，仍按 min-cells 过滤）  n={ov['n']}  支撑集中位={ov['support_median']:.0f}")
    p(f"    {_ljust('网络', W)} {_rjust('n', 6)} {_rjust('', 9)} "
      f"{_rjust('@1', 7)} {_rjust('@4', 7)} {_rjust('@24', 7)}")
    for name in names:
        row = ov["nets"][name]
        n = row["n"]
        if n == 0:
            continue
        r = row["recall"]
        pct = {k: f"{100 * r[str(k)]:.1f}%" for k in KS}
        line = (f"    {_ljust(name, W)} {_rjust(str(n), 6)} {'':>9} "
                f"{_rjust(pct[1], 7)} {_rjust(pct[4], 7)} {_rjust(pct[24], 7)}   "
                f"[随机 {rnd[1]} / {rnd[4]} / {rnd[24]}]")
        p(line)


# ─── 参数 / 主流程 ───────────────────────────────────────────────────────
def parse_nets(spec: str) -> list[tuple[str, str]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--nets 格式错误（要 name=path）：{item}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not (REPO / path).exists():
            raise SystemExit(f"--nets 里的 checkpoint 不存在：{path}")
        out.append((name, path))
    if not out:
        raise SystemExit("--nets 为空")
    return out


def git_info() -> dict:
    def _run(*cmd):
        try:
            return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                                  encoding="utf-8", timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001 - 只是元信息
            return ""
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain"))}


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description="统一的按动作类型 recall 报告（口径 A 亚军决策点 + 口径 B 80 局留出）")
    ap.add_argument("--nets", default=DEFAULT_NETS, help="name=path,name=path,...")
    ap.add_argument("--min-cells", type=int, default=2, help="支撑集门槛（默认 2；1=不设门槛）")
    ap.add_argument("--insufficient-n", type=int, default=20, help="n 低于此值标注「样本不足」（默认 20）")
    ap.add_argument("--only", choices=["A", "B", "all"], default="all")
    ap.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    ap.add_argument("--n-logs", type=int, default=16, help="口径 A 用前 N 局（0=全部，默认 16，与旧脚本一致）")
    ap.add_argument("--holdout", default=DEFAULT_HOLDOUT, help="口径 B 留出局 stem，逗号分隔")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="默认 cpu（与旧脚本逐位可比）")
    ap.add_argument("--threads", type=int, default=4, help="CPU 线程上限（默认 4，避免干扰正在跑的实验）")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("logs", nargs="*", help="可直接给日志路径（覆盖 --logs-dir/--n-logs）")
    ARGS = ap.parse_args()

    torch.set_num_threads(max(1, ARGS.threads))
    dev = torch.device(ARGS.device)
    nets = parse_nets(ARGS.nets)
    names = [n for n, _ in nets]
    p(f"网络 {len(nets)} 个：" + "、".join(names))
    p(f"min-cells={ARGS.min_cells}  insufficient-n={ARGS.insufficient_n}  device={ARGS.device}  "
      f"threads={ARGS.threads}")

    result = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "script": "code/test_match/eval_recall_report.py",
            "cwd_repo": str(REPO),
            **git_info(),
            "min_cells": ARGS.min_cells,
            "insufficient_n": ARGS.insufficient_n,
            "ks": list(KS),
            "ops": {NAME[op]: op for op in OPS},
            "nets": [{"name": n, "path": pth} for n, pth in nets],
            "device": ARGS.device,
            "note": NOTE.format(n=ARGS.insufficient_n),
        },
        "calibers": {},
    }

    if ARGS.only in ("A", "all"):
        if ARGS.logs:
            logs = [Path(x) for x in ARGS.logs]
        else:
            all_logs = sorted((REPO / ARGS.logs_dir).glob("*.log"))
            logs = all_logs[:ARGS.n_logs] if ARGS.n_logs > 0 else all_logs
        p(f"\n[口径 A] 亚军决策点：重放 {len(logs)} 局日志 ...")
        t0 = time.time()
        decisions = build_runnerup_decisions(logs)
        p(f"    可用决策点 {len(decisions)} 个（{time.time() - t0:.1f}s）")
        p("    评估网络 ...")
        result["calibers"]["A"] = eval_caliber(
            "A", "亚军决策点（日志重放）",
            "支撑集=该动作类的合法格数（position mask）；目标=亚军实际走法（仅统计重放后仍合法的决策点）",
            {"日志目录": str(REPO / ARGS.logs_dir) if not ARGS.logs else "(命令行指定)",
             "日志局数": f"{len(logs)} 局：" + ", ".join(x.stem for x in logs)},
            decisions, nets, ARGS.min_cells, dev)

    if ARGS.only in ("B", "all"):
        files = [REPO / HOLDOUT_DIR / f"{s.strip()}.npz" for s in ARGS.holdout.split(",") if s.strip()]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit("留出 npz 不存在：" + ", ".join(missing))
        p(f"\n[口径 B] 我们自己的留出局：{', '.join(f.stem for f in files)} ...")
        t0 = time.time()
        decisions, n_unsup, n_windows = build_holdout_decisions(files)
        p(f"    可用决策点 {len(decisions)} 个（npz 窗口 {n_windows} 个；"
          f"另有 {n_unsup} 个不支持的动作组被跳过；{time.time() - t0:.1f}s）")
        p("    评估网络 ...")
        result["calibers"]["B"] = eval_caliber(
            "B", "我们自己的 80 局留出（g9376-g9379）",
            "支撑集=npz 里该动作组出现过的候选格数；目标=教师 argmax 候选（候选内排名，不引入合法掩码）",
            {"留出目录": HOLDOUT_DIR, "留出局": ", ".join(f.stem for f in files),
             "npz 窗口数": str(n_windows), "不支持动作组": str(n_unsup)},
            decisions, nets, ARGS.min_cells, dev)

    for key in ("A", "B"):
        if key in result["calibers"]:
            print_caliber(result["calibers"][key], names)

    if ARGS.json_out:
        out = Path(ARGS.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        p(f"\nJSON 已写入 {out}（{out.stat().st_size} B）")
    else:
        p("\n提示：加 --json-out <path> 可落一份 JSON 供以后比较")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
