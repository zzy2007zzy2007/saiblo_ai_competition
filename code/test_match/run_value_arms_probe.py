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
RE_P03 = re.compile(r"\*\*P0-3 (过线|不过线)\*\*")


def judge_p03(s_cd0, s_late) -> dict:
    """**带符号、带方向**的 P0-3（2026-11-01 用户口径）。

    假说："冷却好（cd=0）时金币更值钱" ⇒ 需要
      ① `s_cd0 > 0`（金币对价值是正的）
      ② `s_cd0 > s_late`（冷却好时比冷却久时更值钱）
    ⚠️ 旧的 `|s_cd0| / |s_late| ≥ 2` 会被**负斜率**骗过线（relk50 就是这样"假过线"），已废。
    """
    if s_cd0 is None or s_late is None:
        return {"sign_ok": None, "dir_ok": None, "p03_ok": None}
    sign_ok = bool(s_cd0 > 0)
    dir_ok = bool(s_cd0 > s_late)
    return {"sign_ok": sign_ok, "dir_ok": dir_ok, "p03_ok": bool(sign_ok and dir_ok),
            "diff": float(s_cd0 - s_late),
            "old_abs_ratio": (abs(s_cd0) / max(abs(s_late), 1e-12))}


def parse(txt: str) -> dict:
    """取**最后一次**出现的读数（= 最后一组样本；探针顺序是 synthetic → no_tower → tower）。"""
    def last(rx, cast=float, default=None):
        m = rx.findall(txt)
        if not m:
            return default
        return cast(m[-1])
    out = {"ratio": last(RE_RATIO), "kink_pct": last(RE_KINK),
           "s_cd0": last(RE_S0), "s_late": last(RE_SLATE)}
    out.update(judge_p03(out["s_cd0"], out["s_late"]))
    m = RE_P03.findall(txt)
    if m:
        out["p03_text"] = m[-1]
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
    ap.add_argument("--from-json", default="", help="不跑探针，直接用已有 summary.json 重算（P0-3 口径改了就用它）")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.from_json:
        src = Path(a.from_json) if Path(a.from_json).is_absolute() else REPO / a.from_json
        rows = json.loads(src.read_text(encoding="utf-8"))
        for r in rows:                       # 用原始斜率重算带符号判据
            r.update(judge_p03(r.get("s_cd0"), r.get("s_late")))
        print_table(rows)
        (OUTD / "summary_signed.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[json] {OUTD / 'summary_signed.json'}（源 {src}）")
        return 0
    if a.ckpt:
        arms = list(zip(a.names or [Path(p).stem for p in a.ckpt], a.ckpt))
    else:
        arms = DEFAULT_ARMS
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
        print(f"        折点={p.get('kink_pct')}% | cd0斜率={p.get('s_cd0')} "
              f"cd≥0.6={p.get('s_late')} ⇒ 符号{'✓' if p.get('sign_ok') else '✗'} "
              f"方向{'✓' if p.get('dir_ok') else '✗'} ⇒ P0-3 "
              f"{'过线' if p.get('p03_ok') else '不过线'}", flush=True)

    print_table(rows)
    out = OUTD / "summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[json] {out}")
    return 0


def print_table(rows: list) -> None:
    """汇总表（2026-11-01 新口径：P0-3 带符号 + 带方向）。"""
    print("\n" + "=" * 108)
    print(f"{'臂':<18}{'P0-2 折点%':>10}{'slope|cd=0':>12}{'slope|cd≥0.6':>14}"
          f"{'符号(>0)':>10}{'方向(cd0>late)':>15}{'P0-3':>8}{'(旧|比|)':>9}")
    print("-" * 108)
    for r in rows:
        if r.get("missing"):
            print(f"{r['name']:<18}{'(缺 ckpt)':>10}")
            continue
        f = lambda k, w, p=".5f": (f"{r[k]:+{w}{p}}" if r.get(k) is not None else "—")  # noqa: E731
        mark = lambda b: ("✓" if b else "✗") if b is not None else "—"                  # noqa: E731
        print(f"{r['name']:<18}{(r.get('kink_pct') if r.get('kink_pct') is not None else float('nan')):>10.1f}"
              f"{f('s_cd0', 12)}{f('s_late', 14)}"
              f"{mark(r.get('sign_ok')):>10}{mark(r.get('dir_ok')):>15}"
              f"{('过线' if r.get('p03_ok') else '不过线'):>8}"
              f"{(r.get('old_abs_ratio') if r.get('old_abs_ratio') is not None else float('nan')):>9.2f}")
    print("\n判据：P0-2 折点 ≥20%（现役 3.0–4.3%）｜P0-3 = **cd=0 斜率为正 且 cd=0 > cd≥0.6**"
          "（旧 |比|≥2 会被负斜率假通过，仅列作对照）")


if __name__ == "__main__":
    raise SystemExit(main())
