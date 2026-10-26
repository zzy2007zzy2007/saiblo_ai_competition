"""给 probe_runnerup_oracle.py 补上 DLL 搜索路径与 cwd（照抄 match_via_sdk 的做法）。"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

if "add_dll_directory" not in t:
    t = t.replace("import subprocess\nimport sys", "import os\nimport subprocess\nimport sys", 1)
    anchor = "RUNNERUP = str(REPO"
    idx = t.index(anchor)
    ins = 'os.add_dll_directory(r"C:/mingw64/bin")\n'
    t = t[:idx] + ins + t[idx:]
    p.write_text(t, encoding="utf-8")
    print("  已加 add_dll_directory")
else:
    print("  已有 DLL 目录设置")

# cwd 改成 exe 所在目录
if "cwd=str(REPO))" in t:
    t = Path("code/test_match/probe_runnerup_oracle.py").read_text(encoding="utf-8")
    t = t.replace("cwd=str(REPO))", "cwd=str(Path(RUNNERUP).parent))", 1)
    Path("code/test_match/probe_runnerup_oracle.py").write_text(t, encoding="utf-8")
    print("  已把 cwd 改为 exe 目录")
