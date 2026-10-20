"""二分定位 replay 里让 sdk_lure_inspector 报错的那条记录。

做法：把 replay 截成前 N 条，逐次跑 inspector 的 plan_rollouts(round=N-1)，
看从哪个 N 开始报 "cannot use value() with array"。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

EXE = r"D:\dsh_build_tmp\sdk_lure_inspector.exe"
SRC = Path(r"D:\dsh_build_tmp\replay_obj.json")
TMP = Path(r"D:\dsh_build_tmp\_probe.json")

# ⚠️ MinGW 编出来的 exe 需要 C:\mingw64\bin 里的 DLL；subprocess 不继承
#    当前 pwsh 会话临时改的 PATH，所以这里显式加。
ENV = os.environ.copy()
ENV["PATH"] = r"C:\mingw64\bin;" + ENV.get("PATH", "")


def run(n: int) -> str:
    recs = json.loads(SRC.read_text(encoding="utf-8"))[:n]
    TMP.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    req = {"mode": "plan_rollouts", "replay_path": str(TMP).replace("\\", "/"),
           "round": n - 1, "player": 0, "plan_key": "hold"}
    p = subprocess.run([EXE], input=json.dumps(req).encode(), capture_output=True, env=ENV)
    out = p.stdout.decode("utf-8", errors="replace").strip()
    if not out:
        out = "[EMPTY] rc=%d stderr=%s" % (p.returncode,
                                           p.stderr.decode("utf-8", errors="replace")[:120])
    return out[:200]


def main() -> int:
    total = len(json.loads(SRC.read_text(encoding="utf-8")))
    print(f"总记录数 {total}")
    lo, hi = 1, total
    # 先确认 1 正常、total 失败
    print(f"  N=1      -> {run(1)}")
    last_ok = 1
    for n in (2, 3, 5, 10, 20, 40, 80, 160, total):
        r = run(n)
        ok = r.startswith('{"mode"')
        print(f"  N={n:<4} -> {'OK ' if ok else 'ERR'} {r[:100]}")
        if ok:
            last_ok = n
        else:
            hi = n
            break
    else:
        print("全部通过？")
        return 0
    lo = last_ok
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = run(mid)
        if r.startswith('{"mode"'):
            lo = mid
        else:
            hi = mid
    print(f"\n第一条坏记录: 索引 {hi-1}（截断到 {hi} 条时开始报错）")
    rec = json.loads(SRC.read_text(encoding="utf-8"))[hi - 1]
    print("该记录:", json.dumps(rec, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
