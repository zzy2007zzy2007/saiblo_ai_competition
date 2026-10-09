"""把 `ingest_bridge_dump.py` 产出的**带标签 pkl** 转成 `train_value_net.py` 能吃的 **npz**。

为什么需要：`ingest` 给的是每局一个 pkl（样本含 `board/stats/player/value_target`，其中 `value_target` 是**该样本视角**的终局价值），
而 `code/my_ai/train_value_net.py` 的 `ValueDataset` 要的是 **npz**，键为
`board (T,28,19,19) f16` / `stats (T,42) f16` / **`value (T,2) f32`**（两方各自的价值）。

本脚本按 pkl 里每个样本的 `player` 把标签**还原成 p0 视角**再写成 `value=(v_p0, -v_p0)`：
- `player == 0` ⇒ `v_p0 = value_target`
- `player == 1` ⇒ `v_p0 = -value_target`

用法：
    python code/my_ai/az_intent/pkl_to_value_npz.py --src data/s2_labeled --dst data/s2_value_npz
    python code/my_ai/az_intent/pkl_to_value_npz.py --selftest      # 用随机数据自测（不读真实数据）
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _stack(samples: list) -> dict:
    """把样本列表（dict 或对象）堆成 npz 需要的三个数组。"""
    boards, statss, vals = [], [], []
    for s in samples:
        g = (lambda k: s[k]) if isinstance(s, dict) else (lambda k: getattr(s, k))
        b = np.asarray(g("board"), dtype=np.float16)
        st = np.asarray(g("stats"), dtype=np.float16)
        vt = float(g("value_target"))
        pl = int(g("player"))
        v_p0 = vt if pl == 0 else -vt
        boards.append(b)
        statss.append(st)
        vals.append((v_p0, -v_p0))
    return dict(board=np.stack(boards).astype(np.float16),
                stats=np.stack(statss).astype(np.float16),
                value=np.asarray(vals, dtype=np.float32))


def convert_dir(src: Path, dst: Path) -> tuple[int, int]:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.pkl"))
    n_files = n_samples = 0
    for f in files:
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        samples = obj if isinstance(obj, list) else obj.get("samples", obj)
        if not samples:
            print(f"  ⚠️ {f.name}: 0 样本 ⇒ 跳过"); continue
        d = _stack(list(samples))
        np.savez_compressed(dst / (f.stem + ".npz"), **d)
        n_files += 1
        n_samples += d["board"].shape[0]
    return n_files, n_samples


def selftest() -> int:
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = d / "src"; src.mkdir()
    fake = [dict(board=np.zeros((28, 19, 19), np.float16), stats=np.zeros(42, np.float16), player=0, value_target=0.5),
            dict(board=np.ones((28, 19, 19), np.float16), stats=np.ones(42, np.float16), player=1, value_target=0.5)]
    with open(src / "g1.pkl", "wb") as fh:
        pickle.dump(fake, fh)
    nf, ns = convert_dir(src, d / "dst")
    z = np.load(d / "dst" / "g1.npz")
    ok = (nf == 1 and ns == 2 and z["board"].shape == (2, 28, 19, 19)
          and np.allclose(z["value"][0], [0.5, -0.5]) and np.allclose(z["value"][1], [-0.5, 0.5]))
    print(f"   文件={nf} 样本={ns} board={z['board'].shape} value[0]={z['value'][0]} value[1]={z['value'][1]}")
    print("自测:", "✅ 全过（player=0 取 +vt、player=1 取 −vt）" if ok else "❌ 失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--dst")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.src and a.dst):
        ap.error("需要 --src/--dst（或用 --selftest）")
    nf, ns = convert_dir(Path(a.src), Path(a.dst))
    print(f"   ✅ 转出 {nf} 个 npz、{ns} 条样本 -> {a.dst}")
    return 0 if nf else 1


if __name__ == "__main__":
    raise SystemExit(main())
