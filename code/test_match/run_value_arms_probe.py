#!/usr/bin/env python
"""对一批值网 ckpt 批量跑 **P0-2 / P0-3** 探针，并把两个判据数字抓成一张表。

为什么要它：`probe_value_coin_curve.py` 一次只吃一个 `--ckpt`、输出长篇；
重训 9 个标签臂后要横向比"90 币折点 %"与"条件斜率比"，手抄容易错。

用法：
  python code/test_match/run_value_arms_probe.py                 # 默认一批（incumbent + 9 臂）
  python code/test_match/run_value_arms_probe.py --ckpt a.pt b.pt --names a b
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(Path(sys.executable))
PROBE = REPO / "code/test_match/probe_value_coin_curve.py"
OUTD = REPO / "match_results/value_arms_probe"

# 顺序 = 报告表格顺序（先对照，再主力，再 tau 扫，再类型对照，最后 aux2）
DEFAULT_ARMS = [
    ("incumbent", "checkpoints/posnet_A_k5_m32.pt"),
    ("terminal", "training_history/value_retrain_v1/arm_terminal.pt"),
    ("kgeo50", "training_history/value_retrain_v1/arm_kgeo50.pt"),
    ("kgeo25", "training_history/value_retrain_v1/arm_kgeo25.pt"),
    ("kgeo10", "training_history/value_retrain_v1/arm_kgeo10.pt"),
    ("kgeo100", "training_history/value_retrain_v1/arm_kgeo100.pt"),
    ("rel50(rel+geo)", "training_history/value_retrain_v1/arm_rel50.pt"),
    ("relk50(rel+kgeo)", "training_history/value_retrain_v1/arm_relk50.pt"),
    ("abs50(abs+geo)", "training_history/value_retrain_v1/arm_abs50.pt"),
    ("aux2(kgeo50)", "training_history/value_retrain_v1/arm_aux2_kgeo50.pt"),
]

RE_RATIO = re.compile(r"⇒ 比值\s*([\d.]+)×")
RE_KINK = re.compile(r"最大突变幅度 / 平均斜率：中位\s*([\d.]+)%")
RE_KINK_AT = re.compile(r"各层突变点落到的 coins（币）：(.*)")
RE_Y90 = re.compile(r"90 币左/右斜率比（11 层）：(.*)")
RE_S0 = re.compile(r"cd=0\s+的斜率\s*([+-][\d.]+)")
RE_SLATE = re.compile(r"cd≥0\.6 的均值斜率\s*([+-][\d.]+)")
RE_VR2 = re.compile(r"线性 R².*?([\d.]+)")


def parse(txt: str) -> dict:
    """取**最后一次**出现的读数（= 最后一组样本；探针顺序是 synthetic → no_tower → tower）。"""
    def last(rx, cast=float, default=None):
        m = rx.findall(txt)
        if not m:
            return default
        return cast(m[-1])
    out = {"ratio": last(RE_RATIO), "kink_pct": last(RE_KINK),
           "s_cd0": last(RE_S0), "s_late": last(RE_SLATE)}
    m = RE_Y90.findall(txt)
    if m:
        vals = [float(x) for x in re.findall(r"[\d.]+", m[-1])]
        if vals:
            out["y90_ratio_min"] = min(vals)
            out["y90_ratio_max"] = max(vals)
    m = RE_KINK_AT.findall(txt)
    if m:
        out["kink_at"] = m[-1].strip()
    # 组别标题（确认最后一组是 tower）
    tags = re.findall(r"^([A-Z])\. (纯合成输入|真实局面 · \S+)", txt, re.M)
    out["last_group"] = tags[-1] if tags else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=None)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--tag", default="UNIFIED_SPLIT_VERBOSE:8,UNIFIED_AB_A:10,UNIFIED_AB_A:11")
    a = ap.parse_args()
    if a.ckpt:
        arms = list(zip(a.names or [Path(p).stem for p in a.ckpt], a.ckpt))
    else:
        arms = DEFAULT_ARMS
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, rel in arms:
        ck = Path(rel) if Path(rel).is_absolute() else REPO / rel
        log = OUTD / f"{name.split('(')[0].replace('+', '_')}.log"
        if not ck.exists():
            print(f"[probe] !! 缺 ckpt {ck}")
            rows.append({"name": name, "ckpt": str(ck), "missing": True})
            continue
        print(f"[probe] {name} <- {ck.name} ...", flush=True)
        r = subprocess.run([PY, "-u", str(PROBE), "--ckpt", str(ck), "--tag", a.tag],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        txt = (r.stdout or "") + "\n" + (r.stderr or "")
        log.write_text(txt, encoding="utf-8")
        p = parse(txt)
        p.update({"name": name, "ckpt": str(ck), "rc": r.returncode})
        rows.append(p)
        print(f"        折点={p.get('kink_pct')}% 条件斜率比={p.get('ratio')}× "
              f"(cd0={p.get('s_cd0')}, cd≥0.6={p.get('s_late')})", flush=True)

    # 汇总表
    print("\n" + "=" * 100)
    print(f"{'臂':<18}{'P0-2 折点%':>11}{'P0-3 比值×':>11}{'∂V/∂coins|cd=0':>16}"
          f"{'|cd≥0.6':>12}{'90币左/右比':>14}")
    print("-" * 100)
    for r in rows:
        if r.get("missing"):
            print(f"{r['name']:<18}{'(缺 ckpt)':>11}")
            continue
        yr = (f"{r.get('y90_ratio_min'):.2f}–{r.get('y90_ratio_max'):.2f}"
              if "y90_ratio_min" in r else "—")
        print(f"{r['name']:<18}{(r.get('kink_pct') if r.get('kink_pct') is not None else float('nan')):>11.1f}"
              f"{(r.get('ratio') if r.get('ratio') is not None else float('nan')):>11.2f}"
              f"{(r.get('s_cd0') if r.get('s_cd0') is not None else float('nan')):>+16.5f}"
              f"{(r.get('s_late') if r.get('s_late') is not None else float('nan')):>+12.5f}"
              f"{yr:>14}")
    out = OUTD / "summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[json] {out}")
    print("判据：P0-2 折点 ≥20%（现役 3.0%）；P0-3 比值 ≥2（现役 0.99×）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
