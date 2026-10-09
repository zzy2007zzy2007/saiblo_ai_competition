"""把"观测 dump + AZ 软策略 dump"转成**价值网训练用 npz**，目标是**搜索自己回传的根节点价值 `root_value`**。

动机（2026-10-09）：S2 已证明 `value_state` **确实进入搜索叶评估**（同局面换网 ⇒ 叶值从 −0.168 变到 +0.439）。
用**终局胜负**当目标训出的网在开局就给出 +0.44（过度自信，因为训练样本以中局为主）；
而搜索回传的 `root_value` 是**稠密、平滑、且已被搜索（前瞻）改进过**的信号（实测范围 [−0.637, +0.786]、1322 个不同取值、相邻决策平滑、开局 ≈ −0.085 中性）
⇒ **"从搜索蒸馏价值"比"用终局当目标"信息更好**，是本管线里还没试过的一条杠杆（标准 AZ 的价值自举）。

对齐：桥接端 `_record_az` 在写观测**之前**调用 ⇒ 同一 (seed, player) 的两串记录**按决策序号一一对应**（个别末尾会因硬杀差 1 条 ⇒ 取较短者）。

输出（与 `pkl_to_value_npz.py` 同 schema，供 `train_value_antwar.py` 直接吃）：
  `board (T,28,19,19) f16` / `stats (T,42) f16` / `target (T,) f32`（= 该决策的 `root_value`）
另存 `value (T,2) = (t, −t)` 以保持与旧格式兼容。

用法：
    python code/my_ai/az_intent/azjsonl_to_value_npz.py --obs-dir data/s2_obs/COLLECT_S2 \
        --az-dir data/s2_policy/COLLECT_S2 --out data/s2v_npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REC = 28 * 19 * 19 * 2 + 42 * 2 + 1 + 3 + 24  # = 20328（与 convert_az_dump.py 一致）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def read_obs(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    n = len(buf) // REC
    buf = buf[: n * REC].reshape(n, REC)
    board = buf[:, :28 * 19 * 19 * 2].copy().view(np.float16).reshape(n, 28, 19, 19)
    o = 28 * 19 * 19 * 2
    stats = buf[:, o:o + 84].copy().view(np.float16).reshape(n, 42)
    return board, stats, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--az-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    obsd, azd, outd = Path(a.obs_dir), Path(a.az_dir), Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    n_files = n_samples = 0
    for az in sorted(azd.glob("az_seed*_p*.jsonl")):
        stem = az.stem.replace("az_", "")
        ob = obsd / f"obs_{stem}.bin"
        if not ob.is_file():
            print(f"  ⚠️ 缺 {ob.name} ⇒ 跳过 {az.name}"); continue
        board, stats, no = read_obs(ob)
        recs = []
        for s in az.read_text(encoding="utf-8", errors="replace").splitlines():
            if not s.strip():
                continue
            try:
                recs.append(json.loads(s))
            except Exception:  # noqa: BLE001  硬杀可能留下截断行
                print(f"  ⚠️ {az.name}: 跳过 1 行坏 JSON")
        n = min(no, len(recs))
        tgt = np.asarray([float(r.get("root_value", 0.0)) for r in recs[:n]], dtype=np.float32)
        np.savez_compressed(outd / f"{stem}.npz", board=board[:n], stats=stats[:n],
                            target=tgt, value=np.stack([tgt, -tgt], axis=1))
        n_files += 1
        n_samples += n
    print(f"   ✅ 转出 {n_files} 个 npz、{n_samples} 条样本（目标 = 搜索 root_value）-> {outd}")
    return 0 if n_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
