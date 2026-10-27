#!/usr/bin/env python3
"""前向耗时实测：32/3 小值网 vs 64/6 现役值网（**同进程、同线程数、同 batch**）。

为什么单独一个脚本：搜索侧的收益全部来自"每帧值网前向多少钱"，
必须在同一台机器、同一时刻、同一线程配置下**背靠背**测，跨机器/跨负载的绝对值没意义
（本机在跑对局/训练时绝对值会偏慢，**但比值可信**）。

两侧都用**真实 ckpt 权重**（不是随机初始化）：
  * 学生 = `--student-ckpt`（独立 `ValueNetwork`，`config` 决定 width/blocks）；
  * 老师 = `--teacher-ckpt` 的 `value_state`（`AntWarNetwork`，部署时真正跑的那个网）。

用法：
  python code/test_match/bench_value_forward.py \
      --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt --threads 4 --batch 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.value_network import ValueNetwork          # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models  # noqa: E402


def bench(model, bs: int, iters: int, warm: int, teacher: bool) -> float:
    b = torch.randn(bs, 28, 19, 19)
    s = torch.randn(bs, 42)
    with torch.no_grad():
        for _ in range(warm):
            model(b, s)
        t0 = time.perf_counter()
        for _ in range(iters):
            o = model(b, s)
            if teacher and isinstance(o, dict):   # 部署侧真的会取 value
                o = o["value"]
        dt = time.perf_counter() - t0
    return dt / iters * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--teacher-ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warm", type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"cuda_avail={torch.cuda.is_available()}  (CPU 计时，比部署侧偏慢，看比值)")

    sc = Path(a.student_ckpt)
    if not sc.is_absolute():
        sc = _REPO / sc
    sk = torch.load(sc, map_location="cpu", weights_only=False)
    cfg = sk.get("config") or {}
    student = ValueNetwork(latent_dim=int(cfg.get("latent_dim", 64)),
                           num_resblocks=int(cfg.get("num_resblocks", 6))).eval()
    student.load_state_dict(sk["state_dict"])

    tc = Path(a.teacher_ckpt)
    if not tc.is_absolute():
        tc = _REPO / tc
    _c, _p, teacher = load_three_models(str(tc))
    teacher.eval()

    ps = student.count_parameters()
    pt = sum(p.numel() for p in teacher.parameters())
    print(f"student {sc.name}: ValueNetwork({cfg.get('latent_dim')},{cfg.get('num_resblocks')}) "
          f"params={ps:,}")
    print(f"teacher {tc.name}: value_state(AntWarNetwork) params={pt:,} (整网，value 通路同在)")
    print(f"{'batch':>6} {'64/6 ms':>10} {'32/3 ms':>10} {'speedup':>8}")
    for bs in a.batch:
        t = bench(teacher, bs, a.iters, a.warm, True)
        s = bench(student, bs, a.iters, a.warm, False)
        print(f"{bs:>6} {t:>10.2f} {s:>10.2f} {t/s:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
