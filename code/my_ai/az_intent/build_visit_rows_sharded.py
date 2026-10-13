"""分片建集器：把 dump **逐局**转成小分片，再合并成训练用的 npz（**峰值内存 ≈ 单局** ✓）。

动因（用户 2026-10-12 提问）：400 局 ≈ **30k 行 ≈ 0.62 GB**（实测估）本不紧张 ✓，
但 1,200 局 ≈ 1.9 GB、而 value-prior 那种密度（862,134 行 = **17.4 GB** ✗）会撞墙
⇒ 先把"建集"这条路做成**永不整体驻留**的形式 ✓。

做法：
  1. 逐文件（= 逐局）读 policy JSONL + obs `.bin`（一次只持有一局的 obs ~9 MB ✓）；
  2. 每局产出一个分片 `<shard-dir>/shard_<token>.npz`（board/stats/cls/cell/prob/cnt ✓）；
  3. 收尾把分片**合并**成 `--out` 的训练 npz（合并需要 total 量级的内存：30k 行 0.62 GB ✓ 可接受；
     若将来 >200k 行，请改用"惰性读分片"而不是合并 ✓ —— 脚本会打印内存估计并给阈值警告 ✓）。

目标构造与 `train_visit_marginal.py` **完全一致** ✓（`visits_raw` 归一化后按 (类, x, y) 边缘化 ✓，
取质量最大的类、≥2 格 ✓；`cell` 存 (x, y) ✓ 与 `action_map[类, x, y]` 的索引一致 ✓）。

用法：
  python -u code/my_ai/az_intent/build_visit_rows_sharded.py \
    --policy-dir data/seedstate_dump2/az --obs-dir data/seedstate_dump2/obs \
    --max-games 0 --shard-dir training_history/visitmarg/shards30k \
    --out training_history/visitmarg/seedstate_rows30k.npz
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

BOARD_SHAPE = (28, 19, 19)
N_CLS = 24
BSZ = int(np.prod(BOARD_SHAPE)) * 2
REC = BSZ + 84 + 1 + 3 + N_CLS


def build_one(jf: str, obs_dir: Path) -> dict | None:
    tok = Path(jf).name.split("seed")[1].split(".")[0]
    of = obs_dir / f"obs_seed{tok}.bin"
    if not of.exists():
        return None
    recs = []
    for line in open(jf, encoding="utf-8", errors="replace"):
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    raw = np.fromfile(of, dtype=np.uint8)
    n_obs = raw.size // REC
    buf = raw[: n_obs * REC].reshape(n_obs, REC)
    boards, stats_l, clss, cells, probs, cnts = [], [], [], [], [], []
    n_skip_novis = n_skip_single = 0
    for i in range(min(len(recs), n_obs)):
        d = recs[i]
        vr, labels = d.get("visits_raw"), d.get("labels")
        if not (isinstance(vr, list) and vr and labels):
            n_skip_novis += 1
            continue
        w = np.asarray([float(x) for x in vr], dtype=np.float64)
        s = w.sum()
        if s <= 0:
            n_skip_novis += 1
            continue
        p = w / s
        by = collections.defaultdict(lambda: collections.defaultdict(float))
        for k, bundle in enumerate(labels):
            if k >= len(p):
                break
            for op in bundle:
                c, x, y = int(op[0]), int(op[1]), int(op[2])
                by[c][(x, y)] += float(p[k])
        if not by:
            continue
        c_best = max(by, key=lambda c: sum(by[c].values()))
        items = sorted(by[c_best].items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        if len(items) < 2:
            n_skip_single += 1
            continue
        cells.append([xy for xy, _ in items])
        probs.append([m for _, m in items])
        clss.append(c_best)
        cnts.append(len(items))
        boards.append(buf[i, :BSZ].copy().view(np.float16).reshape(BOARD_SHAPE))
        stats_l.append(buf[i, BSZ:BSZ + 84].copy().view(np.float16).reshape(42))
    if not clss:
        return dict(tok=tok, n=0, n_skip_novis=n_skip_novis, n_skip_single=n_skip_single)
    C = max(cnts)
    cell = np.full((len(clss), C, 2), -1, dtype=np.int16)
    prob = np.zeros((len(clss), C), dtype=np.float32)
    for r, (cs, ps) in enumerate(zip(cells, probs)):
        cell[r, :len(cs)] = np.asarray(cs, dtype=np.int16)
        prob[r, :len(ps)] = np.asarray(ps, dtype=np.float32)
    return dict(tok=tok, n=len(clss), board=np.asarray(boards, dtype=np.float16),
                stats=np.asarray(stats_l, dtype=np.float16), cls=np.asarray(clss, dtype=np.int16),
                cell=cell, prob=prob, cnt=np.asarray(cnts, dtype=np.int16),
                n_skip_novis=n_skip_novis, n_skip_single=n_skip_single)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", default="data/seedstate_dump2/az")
    ap.add_argument("--obs-dir", default="data/seedstate_dump2/obs")
    ap.add_argument("--max-games", type=int, default=0, help="0 = 全部")
    ap.add_argument("--shard-dir", default="training_history/visitmarg/shards30k")
    ap.add_argument("--out", default="training_history/visitmarg/seedstate_rows30k.npz")
    ap.add_argument("--merge-threshold", type=int, default=200000,
                    help="合并所需内存超过该行数时给警告（>200k 行请改惰性读分片）")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.policy_dir) / "az_*.jsonl")))
    if args.max_games:
        files = files[: args.max_games]
    sdir = Path(args.shard_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"[shard] 文件 {len(files)} 个 ⇒ 分片目录 {sdir}", flush=True)

    tot_rows = tot_skip_novis = tot_skip_single = 0
    shards = []
    for k, jf in enumerate(files, 1):
        r = build_one(jf, Path(args.obs_dir))
        if r is None:
            continue
        if r["n"] == 0:
            continue
        fp = sdir / f"shard_{r['tok']}.npz"
        np.savez_compressed(fp, board=r["board"], stats=r["stats"], cls=r["cls"],
                            cell=r["cell"], prob=r["prob"], cnt=r["cnt"])
        shards.append(fp)
        tot_rows += r["n"]
        tot_skip_novis += r.get("n_skip_novis", 0)
        tot_skip_single += r.get("n_skip_single", 0)
        if k % 25 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] 累计行 {tot_rows}（跳过：无 visits_raw {tot_skip_novis}，单格 {tot_skip_single}）", flush=True)

    est_gb = tot_rows * (int(np.prod(BOARD_SHAPE)) * 2 + 84 + 25 * 2 + 25 * 4 + 2) / 1e9
    print(f"[shard] 分片 {len(shards)} 个；总行 {tot_rows}；**合并内存估计 ≈ {est_gb:.2f} GB**")
    if tot_rows > args.merge_threshold:
        print(f"[shard] ⚠️ 超过合并阈值 {args.merge_threshold} 行 ⇒ 建议改**惰性读分片**（本次仍会合并，请确认内存 ✓）")

    # 合并（流式：逐分片读入后 concatenate；峰值 = 总量级）
    # ⚠️ 各分片的"候选格数上限"可能不同（本批 24 vs 25 ✗）⇒ 必须**统一 pad 到全局最大宽度** ✓
    #    （`cell` 填充 -1、`prob` 填充 0；真实格数由 `cnt` 记录 ⇒ 训练器按 cnt 掩码 ✓）
    widths = []
    for fp in shards:
        with np.load(fp) as z:
            widths.append(int(z["cell"].shape[1]))
    Cmax = max(widths) if widths else 0
    print(f"[shard] 分片宽度范围 {min(widths)}–{Cmax} ⇒ 统一 pad 到 {Cmax} ✓")

    def cat(key, dtype):
        parts = []
        for fp in shards:
            with np.load(fp) as z:
                a = np.asarray(z[key], dtype=dtype)
                if key == "cell" and a.shape[1] < Cmax:
                    a = np.concatenate([a, np.full((a.shape[0], Cmax - a.shape[1], 2), -1, dtype=dtype)], axis=1)
                elif key == "prob" and a.shape[1] < Cmax:
                    a = np.concatenate([a, np.zeros((a.shape[0], Cmax - a.shape[1]), dtype=dtype)], axis=1)
                parts.append(a)
        return np.concatenate(parts, axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, board=cat("board", np.float16), stats=cat("stats", np.float16),
                        cls=cat("cls", np.int16), cell=cat("cell", np.int16),
                        prob=cat("prob", np.float32), cnt=cat("cnt", np.int16))
    print(f"[shard] 写出 {out}（{out.stat().st_size / 1e9:.2f} GB 压缩后）✓ 训练器接口不变 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
