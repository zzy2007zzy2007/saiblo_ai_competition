"""把 AZ 环的桥接端 dump 转成训练用 npz（与 `distill_data/` **同 schema**）。

输入（都来自 `code/test_match/az_bridge_ai.py`，默认关闭）：
  * `--az-dir`  : `AZAI_AZ_DUMP` 的产物 `az_seed{seed:05d}_p{player}.jsonl`
                  每行 = 一次决策：`cands`(候选束，每束 ≤3 个 op) `labels`(已解出的 (class,x,y)) `visits`(访问分布)
                  `chosen` `intent` `root_value`。
  * `--obs-dir` : `AZAI_DUMP_RAW_DIR` 的产物 `obs_seed{seed:05d}_p{player}.bin`
                  定长记录 **20328 字节**：board f16 (28,19,19) | stats f16 (42,) | player u8 | chosen u8×3 | 类掩码 u8×24。

输出：`--out` 目录下 `az_seed{seed:05d}_p{player}.npz`，字段与 `distill_data/` 一致：
  `board (N,28,19,19) f16` `stats (N,42) f16` `class_ (N,3) i64` `action_map (N,24,19,19) f16` `head_logits (N,3,24) f16`
  —— `action_map`/`head_logits` 是**搜索访问分布的软目标**（AZ 环的教师），`class_` 是**实际执行**的类（每头一个）。

用法：
    python code/distill/convert_az_dump.py --az-dir <dir> --obs-dir <dir> --out <dir>
    python code/distill/convert_az_dump.py --selftest          # 合成数据自测（不跑对局）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REC = 28 * 19 * 19 * 2 + 42 * 2 + 1 + 3 + 24  # = 20328
HOLD = 23
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def read_obs(path: Path) -> list[dict]:
    buf = path.read_bytes()
    if len(buf) % REC:
        raise ValueError(f"{path.name}: 字节数 {len(buf)} 不是记录长度 {REC} 的整数倍")
    out = []
    for i in range(len(buf) // REC):
        o = i * REC
        board = np.frombuffer(buf, dtype=np.float16, count=28 * 19 * 19, offset=o).reshape(28, 19, 19)
        o2 = o + 28 * 19 * 19 * 2
        stats = np.frombuffer(buf, dtype=np.float16, count=42, offset=o2)
        o3 = o2 + 42 * 2
        player = int(buf[o3]); o4 = o3 + 1
        chosen = list(buf[o4:o4 + 3]); o5 = o4 + 3
        mask = np.frombuffer(buf, dtype=np.uint8, count=24, offset=o5)
        out.append(dict(board=board, stats=stats, player=player, chosen=chosen, mask=mask))
    return out


def convert_pair(obs_path: Path, az_path: Path, min_visit: float = 0.0) -> dict:
    obs = read_obs(obs_path)
    lines = [json.loads(s) for s in az_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    n = min(len(obs), len(lines))
    if len(obs) != len(lines):
        print(f"  ⚠️ {az_path.name}: 观测 {len(obs)} 条 vs AZ {len(lines)} 条 ⇒ 取前 {n} 条（对齐按序号）")
    board = np.zeros((n, 28, 19, 19), dtype=np.float16)
    stats = np.zeros((n, 42), dtype=np.float16)
    cls = np.full((n, 3), HOLD, dtype=np.int64)
    amap = np.zeros((n, 24, 19, 19), dtype=np.float16)
    hlog = np.zeros((n, 3, 24), dtype=np.float16)
    kept = 0
    for i in range(n):
        board[i] = obs[i]["board"]
        stats[i] = obs[i]["stats"]
        rec = lines[i]
        labels = rec.get("labels") or []
        visits = rec.get("visits") or []
        # ① 硬标签 = 实际执行的那个束里各 op 的类（每头一个；不足补 HOLD）
        ci = int(rec.get("chosen", -1))
        if 0 <= ci < len(labels):
            for h, lab in enumerate(labels[ci][:3]):
                cls[i, h] = int(lab[0])
        # ② 软目标 = 访问分布在 (类, 格) 与 (头, 类) 上投影（归一化）
        tot = float(sum(visits)) or 1.0
        for b_i, lab_list in enumerate(labels):
            w = float(visits[b_i]) / tot if b_i < len(visits) else 0.0
            if w <= min_visit:
                continue
            for h, lab in enumerate(lab_list[:3]):
                c, x, y = int(lab[0]), int(lab[1]), int(lab[2])
                if 0 <= c < 24:
                    hlog[i, h, c] += w
                    if 0 <= x < 19 and 0 <= y < 19:
                        amap[i, c, x, y] += w
        kept += 1
    return dict(board=board[:kept], stats=stats[:kept], class_=cls[:kept], action_map=amap[:kept], head_logits=hlog[:kept])


def selftest() -> int:
    import tempfile
    d = Path(tempfile.mkdtemp())
    az = d / "az_seed00007_p0.jsonl"
    ob = d / "obs_seed00007_p0.bin"
    # 合成 3 条决策：第 1 条两个候选（建塔 (12,3,4)→cls0、HOLD），第 2 条一个候选，第 3 条无候选
    recs = [
        {"n": 0, "seed": 7, "player": 0, "cands": [[[12, 3, 4]], [[23, 0, 0]]],
         "labels": [[[0, 3, 4]], [[23, -1, -1]]], "visits": [0.75, 0.25], "chosen": 0, "intent": [1, 0, 0], "root_value": 0.1},
        {"n": 1, "seed": 7, "player": 0, "cands": [[[18, 5, 6]]],
         "labels": [[[16, 5, 6]]], "visits": [1.0], "chosen": 0, "intent": [0, 1, 0], "root_value": 0.2},
        {"n": 2, "seed": 7, "player": 0, "cands": [], "labels": [], "visits": [], "chosen": -1, "intent": [0, 0, 0], "root_value": 0.3},
    ]
    az.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    with open(ob, "wb") as f:
        for i in range(3):
            f.write(np.full((28, 19, 19), i, dtype=np.float16).tobytes())
            f.write(np.full(42, i, dtype=np.float16).tobytes())
            f.write(bytes([0]))
            f.write(bytes([0, 23, 23]))
            f.write(bytes([1] * 24))
    out = convert_pair(ob, az)
    ok = True
    shapes = {"board": (3, 28, 19, 19), "stats": (3, 42), "class_": (3, 3),
              "action_map": (3, 24, 19, 19), "head_logits": (3, 3, 24)}
    for k, shp in shapes.items():
        same = tuple(out[k].shape) == shp
        ok = ok and same
        print(f"   {k:12s} shape={tuple(out[k].shape)} 期望={shp} {'✅' if same else '❌'}")
    checks = [
        ("class_[0] = [0,23,23]", list(out["class_"][0]) == [0, 23, 23]),
        ("class_[1] = [16,23,23]", list(out["class_"][1]) == [16, 23, 23]),
        ("head_logits[0,0,0] ≈ 0.75", abs(float(out["head_logits"][0, 0, 0]) - 0.75) < 1e-3),
        ("head_logits[0,0,23] ≈ 0.25", abs(float(out["head_logits"][0, 0, 23]) - 0.25) < 1e-3),
        ("action_map[0,0,3,4] ≈ 0.75", abs(float(out["action_map"][0, 0, 3, 4]) - 0.75) < 1e-3),
        ("action_map[1,16,5,6] ≈ 1.0", abs(float(out["action_map"][1, 16, 5, 6]) - 1.0) < 1e-3),
        ("board[0] 全 0 / board[2] 全 2", float(out["board"][0].sum()) == 0 and abs(float(out["board"][2].mean()) - 2) < 1e-3),
    ]
    for name, val in checks:
        ok = ok and bool(val)
        print(f"   {name}: {'✅' if val else '❌'}")
    print("自测:", "✅ 全过" if ok else "❌ 有失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--az-dir"); ap.add_argument("--obs-dir"); ap.add_argument("--out")
    ap.add_argument("--min-visit", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.az_dir and a.obs_dir and a.out):
        ap.error("需要 --az-dir/--obs-dir/--out（或用 --selftest）")
    azd, obd, outd = Path(a.az_dir), Path(a.obs_dir), Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    pairs = 0
    for az in sorted(azd.glob("az_seed*_p*.jsonl")):
        seed_p = az.stem.replace("az_", "")
        ob = obd / f"obs_{seed_p}.bin"
        if not ob.is_file():
            print(f"  ⚠️ 缺观测文件 {ob.name} ⇒ 跳过 {az.name}"); continue
        d = convert_pair(ob, az, a.min_visit)
        np.savez_compressed(outd / f"{seed_p}.npz", **d)
        print(f"  ✅ {seed_p}: {d['board'].shape[0]} 条决策 -> {outd / (seed_p + '.npz')}")
        pairs += 1
    print(f"共 {pairs} 个 (seed, player) 文件")
    return 0 if pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
