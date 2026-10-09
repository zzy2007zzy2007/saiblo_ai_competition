"""离线诊断：三张价值网在**同一批留出局**上，对**两种目标**的误差。

三张网：
  * `old`   = 部署原网（`checkpoints/grafted_sdk_A.pt` 的 `value_state`）
  * `vs2`   = 用**终局胜负**当目标训出的部署形状网（`checkpoints/grafted_sdk_A_vs2.pt`）
  * `vsearch` = 用**搜索回传的 root_value** 当目标训出（`checkpoints/grafted_sdk_A_vsearch.pt`）

两种目标：
  * `search` = `data/s2v_npz` 的 `target`（搜索值）
  * `outcome` = `data/s2_value_npz` 的 `target`（该样本视角的终局价值）

留出集与训练**逐位一致**：`default_rng(0).permutation(len(games))` 的前 12 局（`games` 顺序 = `sorted(glob)`）。

用法：
    python code/test_match/diag_value_nets.py --search-npz data/s2v_npz --outcome-npz data/s2_value_npz --val-games 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.network import create_model, model_kwargs_from_ckpt  # noqa: E402

NETS = {
    "old(部署原网)": ("checkpoints/grafted_sdk_A.pt", "value_state"),
    "终局目标网": ("checkpoints/grafted_sdk_A_vs2.pt", "value_state"),
    "搜索值网": ("checkpoints/grafted_sdk_A_vsearch.pt", "value_state"),
}


def load_model(path: str, key: str, base: str, dev: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    b = torch.load(base, map_location="cpu", weights_only=False)
    m = create_model(**model_kwargs_from_ckpt(b)).to(dev)
    sd = ck[key] if key in ck else ck["model_state"]
    m.load_state_dict(sd)
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-npz", default="data/s2v_npz")
    ap.add_argument("--outcome-npz", default="data/s2_value_npz")
    ap.add_argument("--base", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--val-games", type=int, default=12)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    fs = sorted(Path(a.search_npz).glob("*.npz"))
    games = [np.load(f) for f in fs]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(games))
    val = [games[i] for i in idx[: a.val_games]]

    # 终局目标（同一局同一决策顺序；两套 npz 的文件名一致 ⇒ 直接按文件名配）
    tgt_out = []
    _out_files = {q.name: q for q in Path(a.outcome_npz).glob("*.npz")}
    for g, f in zip(games, fs):
        # 两套 npz 的命名前缀不同（`seed10001_p0.npz` vs `az_selfplay_seed10001_p0.npz`）⇒ 按后缀匹配
        hit = _out_files.get(f.name) or next((q for n, q in _out_files.items() if n.endswith(f.stem + ".npz")), None)
        if hit is None:
            print(f"   ⚠️ 终局 npz 里找不到 {f.name} 的对应文件 ⇒ 跳过该局"); continue
        z = np.load(hit)
        tgt_out.append(z["target"].astype(np.float32))
    tgt_out_val = [tgt_out[i] for i in idx[: a.val_games]]

    print(f"   留出局数 = {len(val)}（与训练逐位一致的 default_rng(0) 切分）")
    rows = []
    for name, (path, key) in NETS.items():
        m = load_model(path, key, a.base, dev)
        es, eo = [], []
        with torch.no_grad():
            for g, to in zip(val, tgt_out_val):
                xb = torch.from_numpy(g["board"].astype(np.float32)).to(dev)
                xs = torch.from_numpy(g["stats"].astype(np.float32)).to(dev)
                v = torch.tanh(m(xb, xs)["value"].squeeze(-1)).cpu().numpy()
                es.append(np.abs(v - g["target"]))
                eo.append(np.abs(v - to))
        mae_s = float(np.concatenate(es).mean())
        mae_o = float(np.concatenate(eo).mean())
        rows.append((name, mae_s, mae_o))
    print(f"   {'网':<16} {'对搜索值 MAE':>12} {'对终局值 MAE':>12}")
    for name, ms, mo in rows:
        print(f"   {name:<16} {ms:>12.4f} {mo:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
