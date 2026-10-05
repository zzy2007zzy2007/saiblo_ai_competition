"""把 `az_bridge_ai.py` 的**裸观测 dump**（`AZAI_DUMP_RAW_DIR`）变成价值网能吃的**带标签 pkl**。

为什么要有这一步：bridge 收尾时会**硬杀**我们的 AI 进程，实测**结束时写文件根本来不及**
（`[dump] 调用: samples=367` 之后进程就没了）⇒ 采集改成**边打边追加**定长记录的 `.bin`，
而**终局标签不由 AI 写**，而是事后从 **ladder 日志的 `RESULT ... base_hp=`** 反推
（那个日志由 bridge 写，不会被杀）⇒ 竞态消失。

记录格式（每个决策一条，小端、定长）：
    board: 28*19*19 个 float16
    stats: 42 个 float16
    player: 1 个 uint8

标签口径与 `az_selfplay.py:428-433` **逐字相同**：
    `v_p0 = clip((hp0-hp1)/HP_SCALE)`；样本标签 = `v_p0` 若该样本的 player==0 否则 `-v_p0`。
其中 `hp0/hp1` 取自 ladder 日志里 `RESULT seed=<S> p0=... p1=... base_hp=<hp0>,<hp1>`；
本文件属于 `obs_seed<S>_p<P>.bin` ⇒ **P 必须与日志里的 p0/p1 标签对得上**（我们的标签是 `az_bridge_ai`）。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/ingest_bridge_dump.py \
      --raw-dir training_history/vprior/dump_x --ladder-tag S3_depth8_128 \
      --out-dir training_history/vprior/data_vs_rv4
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.az_intent.mcts import HP_SCALE  # noqa: E402

BOARD_SHAPE = (28, 19, 19)
N_CLS = 24
REC_BYTES = int(np.prod(BOARD_SHAPE)) * 2 + 42 * 2 + 1 + 3 + N_CLS   # +player +chosen(3) +mask(24)

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) (?:winner_side=(\S+) )?verdict=(\S+) "
    r"engine_winner=(\S+) rounds=(\d+) terminal=(\S+) base_hp=(\d+),(\d+) coins=(\d+),(\d+)")
NAME_RE = re.compile(r"obs_seed(\d+)_p(\d+)\.bin$")


def load_ladder_labels(tag: str, us_label: str) -> dict:
    """(seed, player) -> v_p0（从 ladder 日志的 RESULT 行反推）。"""
    d = REPO / "match_results" / "ladder_logs" / tag
    out = {}
    for f in sorted(d.glob("*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RESULT_RE.search(txt)
        if not m:
            continue
        seed = int(m.group(1))
        hp0, hp1 = int(m.group(10)), int(m.group(11))
        v_p0 = float(np.clip((hp0 - hp1) / HP_SCALE, -1.0, 1.0))
        # 记录"我方是 p0 还是 p1"（用于核对该 bin 的 player 是否可信）
        us_is_p0 = (m.group(2) == us_label)
        out[(seed, 0 if us_is_p0 else 1)] = (v_p0, m.group(4))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--ladder-tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--us-label", default="az_bridge_ai")
    ap.add_argument("--tower-only", action="store_true",
                    help="只保留 board[4]（己方塔）非零的样本")
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    if not raw.is_absolute():
        raw = REPO / raw
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_ladder_labels(a.ladder_tag, a.us_label)
    print(f"ladder 日志里拿到 {len(labels)} 局的终局标签（tag={a.ladder_tag}）")

    n_files = n_samp = n_keep = 0
    for f in sorted(raw.glob("obs_seed*_p*.bin")):
        m = NAME_RE.search(f.name)
        if not m:
            continue
        seed, player = int(m.group(1)), int(m.group(2))
        if (seed, player) not in labels:
            print(f"  ⚠️ {f.name}: ladder 日志里找不到对应局，跳过")
            continue
        v_p0, winner = labels[(seed, player)]
        n_rec = f.stat().st_size // REC_BYTES
        if n_rec == 0:
            continue
        buf = np.fromfile(f, dtype=np.uint8)
        buf = buf[: n_rec * REC_BYTES].reshape(n_rec, REC_BYTES)
        bsz = int(np.prod(BOARD_SHAPE)) * 2
        board = buf[:, :bsz].copy().view(np.float16).reshape(n_rec, *BOARD_SHAPE)
        stats = buf[:, bsz:bsz + 84].copy().view(np.float16).reshape(n_rec, 42)
        pl = buf[:, bsz + 84].astype(np.int64)
        chosen = buf[:, bsz + 85: bsz + 88].astype(np.int64)          # 3 个类号（255 = 无）
        cmask = buf[:, bsz + 88: bsz + 88 + N_CLS].astype(bool)       # 合法类掩码
        if a.tower_only:
            keep = (board[:, 4] != 0).any(axis=(1, 2))
            board, stats, pl, chosen, cmask = (board[keep], stats[keep], pl[keep],
                                               chosen[keep], cmask[keep])
        n_keep += len(pl)
        samples = [{"board": board[i], "stats": stats[i], "player": int(pl[i]),
                    "chosen_cls": chosen[i].astype(np.int64),
                    "class_mask": cmask[i].astype(bool),
                    "value_target": float(v_p0 if pl[i] == 0 else -v_p0)}
                   for i in range(len(pl))]
        dst = out_dir / f"az_selfplay_seed{seed:05d}_p{player}.pkl"
        with open(dst, "wb") as fh:
            pickle.dump({"samples": samples}, fh)
        n_files += 1
        n_samp += len(samples)
        print(f"  {f.name}: 记录 {n_rec} -> 写出 {len(samples)} 条 "
              f"(v_p0={v_p0:+.3f}, winner={winner}) -> {dst.name}")
    print(f"\n合计：{n_files} 个 pkl、{n_samp} 条样本（tower_only={a.tower_only} 保留 {n_keep}）")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
