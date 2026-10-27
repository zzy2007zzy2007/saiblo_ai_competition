"""构建**独立**的 RNG 可控引擎模块 `code/cpp_engine/rng_build/native_game.pyd`。

设计要点（务必保持）：
1. **不覆盖现役 `code/cpp_engine/native_game.pyd`**：产物落在子目录 `rng_build/`，
   目录名不同、文件名仍是 `native_game.pyd`（pybind 模块名必须等于文件名，所以不能改
   名字；"独立"靠**目录隔离 + 导入路径选择**实现，见 `code/my_ai/az_intent/rng_engine.py`）。
2. **`game.cpp` 一个字节都不改**：唯一改动是 `include/game.hpp` 里加 3 个 header-inline
   方法（`get_rng_state / set_rng_state / consume_rng`）+ `binding.cpp` 加 3 行 `.def`。
3. 编译 flags 与 `code/cpp_engine/build.sh` **完全一致**（`-std=c++17 -O2 -fPIC -shared`），
   这样"不消费 RNG 时与现役 pyd 逐位一致"才可比（见
   `code/test_match/probe_engine_rng.py parity`）。

坑（都已在脚本里处理）：
- `TMP/TEMP/TMPDIR` 必须指纯 ASCII 目录，否则 MinGW 的 cc1plus 写 .o 会 Permission denied；
- 从 base 环境取 pybind11 头文件、从 pytorch-gpu 环境取 python 3.11 头文件与 libs；
- 产物旁边必须放 MinGW 运行库（libstdc++-6 / libgcc_s_seh-1 / libwinpthread-1）。

用法：
    python code/cpp_engine/build_rng_variant.py            # 构建 + 打印产物路径
    python code/cpp_engine/build_rng_variant.py --check    # 构建 + 与现役 pyd 逐位对拍
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CE = REPO / "code" / "cpp_engine"
BUILD = Path(r"C:\Temp\rng_variant_build")     # 纯 ASCII
ENG = BUILD / "eng"
OUT = CE / "rng_build"
TMP = BUILD / "tmp"
GXX = Path(r"C:\mingw64\bin\g++.exe")
PYTHON = Path(r"D:\anaconda3\envs\pytorch-gpu\python.exe")
PBI_PYTHON = Path(r"D:\anaconda3\python.exe")

SOURCES = ["game.cpp", "ant.cpp", "map.cpp", "building.cpp", "coin.cpp",
           "item.cpp", "operation.cpp", "aco.cpp", "comm_judger.cpp",
           "output.cpp"]

# ── 补丁 1：game.hpp 加 3 个 header-inline 方法（唯一改动）──
HPP_ANCHOR = "    unsigned long long rng_state_now() const { return rng_state; }\n"
HPP_NEW = HPP_ANCHOR + """
    // ── RNG 控制接口（rollout 主动消费用；只暴露状态，不改任何游戏逻辑）──
    // 说明：引擎 RNG 是 48-bit LCG（src/game.cpp:25-27），单一状态 `rng_state`，
    //       `next_random()` 是唯一的推进入口（private）。
    unsigned long long get_rng_state() const { return rng_state; }
    void set_rng_state(unsigned long long state) {
        // 折叠到 48 bit，保持 LCG 不变量（乘子是奇数 ⇒ 状态空间上双射）。
        rng_state = state & ((1ULL << 48) - 1);
    }
    void consume_rng(int count) {
        for (int i = 0; i < count; ++i) next_random();
    }
"""

# ── 补丁 2：binding.cpp 加 3 行 .def ──
BIND_ANCHOR = '        .def("rng_state_now", &Game::rng_state_now)\n'
BIND_NEW = BIND_ANCHOR + """        .def("get_rng_state", &Game::get_rng_state)
        .def("set_rng_state", &Game::set_rng_state, py::arg("state"))
        .def("consume_rng", &Game::consume_rng, py::arg("count"))
"""


def apply(text: str, old: str, new: str, name: str, path: Path) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[build] ✗ {path.name}: 锚点 '{name}' 出现 {n} 次（应为 1）"
                         f"\n----- 锚点原文 -----\n{old}")
    return text.replace(old, new, 1)


def prepare() -> None:
    if ENG.exists():
        shutil.rmtree(ENG)
    (ENG / "src").mkdir(parents=True)
    (ENG / "include").mkdir(parents=True)
    for f in sorted((CE / "src").glob("*.cpp")):
        shutil.copy2(f, ENG / "src" / f.name)
    for f in sorted((CE / "include").glob("*.h*")):
        shutil.copy2(f, ENG / "include" / f.name)
    for f in ("m1_support.cpp", "binding.cpp"):
        shutil.copy2(CE / f, ENG / f)

    hpp = ENG / "include" / "game.hpp"
    t = hpp.read_text(encoding="utf-8")
    hpp.write_text(apply(t, HPP_ANCHOR, HPP_NEW, "rng api", hpp), encoding="utf-8")

    bind = ENG / "binding.cpp"
    t = bind.read_text(encoding="utf-8")
    bind.write_text(apply(t, BIND_ANCHOR, BIND_NEW, "rng binding", bind), encoding="utf-8")

    # 自检：game.cpp 必须与官方逐字节相同（防止"顺手改了规则"）
    import hashlib
    for name in ("game.cpp",):
        a = hashlib.md5((REPO / "Ant-Game/game/src" / name).read_bytes()).hexdigest()
        b = hashlib.md5((ENG / "src" / name).read_bytes()).hexdigest()
        if a != b:
            raise SystemExit(f"[build] ✗ {name} 与官方源码不同（{a} vs {b}）")
    print(f"[build] ✓ 副本已打补丁（game.cpp 与官方 MD5 相同）: {ENG}")


def compile_pyd() -> Path:
    def q(py: Path, code: str) -> str:
        return subprocess.check_output([str(py), "-c", code], text=True).strip()

    py_inc = q(PYTHON, "import sysconfig; print(sysconfig.get_paths()['include'])")
    py_libdir = q(PYTHON, "import sysconfig, os; print(os.path.join(sysconfig"
                          ".get_paths()['data'], 'libs'))")
    pbi_inc = q(PBI_PYTHON, "import pybind11; print(pybind11.get_include())")
    py_ver = q(PYTHON, "import sys; print(f'{sys.version_info.major}"
                       "{sys.version_info.minor}')")
    print(f"[build] python {py_ver} include={py_inc} libdir={py_libdir}\n"
          f"[build] pybind11 include={pbi_inc}")

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    # ⚠️ MinGW 的 ld 用窄字符 fopen：**不能**直接把 -o 指到含中文的仓库路径
    #    （会报 `ld.exe: cannot open output file ...: No such file or directory`，
    #     中间目录存在也没用）。所以先链到纯 ASCII 的暂存目录，再用 python 拷回去
    #    （python 的 open 走宽字符 API，中文路径没问题）。
    stage = BUILD / "out"
    stage.mkdir(parents=True, exist_ok=True)
    staged_pyd = stage / "native_game.pyd"
    out_pyd = OUT / "native_game.pyd"
    cmd = [str(GXX), "-std=c++17", "-O2", "-fPIC", "-shared",
           f"-I{pbi_inc}", f"-I{py_inc}", f"-I{ENG / 'include'}",
           *[str(ENG / "src" / s) for s in SOURCES],
           str(ENG / "m1_support.cpp"), str(ENG / "binding.cpp"),
           f"-L{py_libdir}", f"-lpython{py_ver}", "-o", str(staged_pyd)]
    env = dict(os.environ)
    env.update(TMP=str(TMP), TEMP=str(TMP), TMPDIR=str(TMP))
    print("[build] g++ ...")
    r = subprocess.run(cmd, env=env, capture_output=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-8000:])
        raise SystemExit(f"[build] ✗ g++ rc={r.returncode}")
    if r.stderr.strip():
        print("[build] g++ stderr:\n" + r.stderr[-2000:])
    shutil.copy2(staged_pyd, out_pyd)
    for dll in ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"):
        shutil.copy2(Path(r"C:\mingw64\bin") / dll, OUT / dll)
    print(f"[build] ✓ {out_pyd}  ({out_pyd.stat().st_size} bytes)")
    print(f"[build] 现役 pyd 未被触碰: {CE / 'native_game.pyd'}")
    return out_pyd


SMOKE = r"""
import sys
sys.path.insert(0, r'{out}')
import native_game as ng
print('[smoke] module:', ng.__file__)
g = ng.NativeGame()
g.init_game(seed=7, movement_policy='enhanced', cold=True)
s0 = g.get_rng_state()
g.consume_rng(5)
s5 = g.get_rng_state()
M, INC, MASK = 25214903917, 11, (1 << 48) - 1
s = s0
for _ in range(5):
    s = (M * s + INC) & MASK
print('[smoke] consume_rng(5) == 手推 LCG:', s == s5)
g.set_rng_state(s0)
print('[smoke] set_rng_state 回滚:', g.get_rng_state() == s0)
print('[smoke] set_rng_state 折叠到 48bit:', (g.set_rng_state(1 << 50) or g.get_rng_state()) == ((1 << 50) & MASK))
"""


def smoke() -> bool:
    r = subprocess.run([str(PYTHON), "-c", SMOKE.format(out=OUT)],
                       capture_output=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return False
    return True


def check_parity() -> bool:
    """与现役 pyd 逐位对拍（真实对局 440 回合，逐回合强摘要 + rng_state 序列）。"""
    probe = REPO / "code/test_match/probe_engine_rng.py"
    outs = {}
    for tag, extra in (("live", []), ("rng", ["--engine-dir", str(OUT)])):
        out = BUILD / f"parity_{tag}.json"
        cmd = [str(PYTHON), str(probe), "digest", "--out", str(out), *extra]
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr[-3000:])
            return False
        outs[tag] = out
    import json
    a = json.loads(outs["live"].read_text(encoding="utf-8"))
    b = json.loads(outs["rng"].read_text(encoding="utf-8"))
    same = a["digests"] == b["digests"] and a["rng"] == b["rng"]
    print(f"[parity] 现役 vs rng_build 逐回合摘要+rng_state 序列完全一致: {same}")
    return same


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="构建后做冒烟 + 与现役 pyd 对拍")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    if not args.skip_build:
        prepare()
        compile_pyd()
    if args.check:
        ok = smoke() and check_parity()
        print(f"[check] {'✅ 通过' if ok else '❌ 失败'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
