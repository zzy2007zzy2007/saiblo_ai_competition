"""修 probe_runnerup_oracle.py 的 ProcIO 构造（ProcIO 需要 Popen 对象）。"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

if "subprocess.Popen" not in t:
    t = t.replace(
        "import sys\nfrom pathlib import Path",
        "import subprocess\nimport sys\nfrom pathlib import Path",
        1,
    )
    old = "    io = ProcIO(RUNNERUP, cwd=str(REPO))"
    new = (
        "    proc = subprocess.Popen([RUNNERUP], stdin=subprocess.PIPE,\n"
        "                            stdout=subprocess.PIPE, cwd=str(REPO))\n"
        '    io = ProcIO(proc, "runnerup", strip_cr=True)'
    )
    assert old in t, "锚点未找到"
    t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("  已改成 Popen + ProcIO(proc, label, strip_cr=True)")
else:
    print("  已是 Popen 版本")
