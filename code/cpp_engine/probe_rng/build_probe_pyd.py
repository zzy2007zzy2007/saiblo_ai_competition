"""构建**探针版** pybind 引擎（`native_game_rng.pyd`）——不改现役 `native_game.pyd`。

做的事（全部在 `C:\\Temp\\rng_probe\\` 下，中文路径不进编译链）：
  1. 把 `code/cpp_engine/{src,include,m1_support.cpp,binding.cpp}` 复制到临时目录；
  2. 在**副本**上打 4 处补丁：
       a. `game.hpp`: 新增 `get_rng_state() / set_rng_state(s) / consume_rng(k)`
          （**只加接口，不动任何游戏逻辑**），外加探针专用的 `rng_site_counts()`；
       b. `game.cpp` : `next_random()` 里加计数器；每个 RNG 消费点挂一个 site tag；
       c. `game.cpp` : 文件尾补 `rng_site_counts()` 的定义；
       d. `binding.cpp`: 模块名改 `native_game_rng` + 绑定 4 个新方法；
  3. 用与 `code/cpp_engine/build.sh` **相同的 flags**（`-std=c++17 -O2 -fPIC -shared`）
     编出 `C:\\Temp\\rng_probe\\out\\native_game_rng.pyd`，并把 MinGW 运行库拷到旁边。

为什么必须同 flags：本探针要用来证明"不消费 RNG 时，新引擎与现役 pyd 逐位一致"，
优化级别不同就可能引入无关差异。

用法：
    python code/cpp_engine/probe_rng/build_probe_pyd.py            # 只构建
    python code/cpp_engine/probe_rng/build_probe_pyd.py --smoke    # 构建 + 冒烟

注意：`TMP/TEMP/TMPDIR` 必须指到纯 ASCII 目录，否则 MinGW 的 cc1plus 写不出 .o
（默认 `C:\\Users\\<中文名>\\AppData\\Local\\Temp` 会报 Permission denied）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CE = REPO / "code" / "cpp_engine"
BUILD = Path(r"C:\Temp\rng_probe")
ENG = BUILD / "eng"
OUT = BUILD / "out"
TMP = BUILD / "tmp"
GXX = Path(r"C:\mingw64\bin\g++.exe")
PYTHON = Path(r"D:\anaconda3\envs\pytorch-gpu\python.exe")
PBI_PYTHON = Path(r"D:\anaconda3\python.exe")

# ─────────────────────── 补丁 1：game.hpp 新接口 ───────────────────────
HPP_ANCHOR = "    unsigned long long rng_state_now() const { return rng_state; }\n"
HPP_NEW = HPP_ANCHOR + """    // ── RNG 控制接口（探针/改造提案；只暴露状态，不改任何游戏逻辑）──
    unsigned long long get_rng_state() const { return rng_state; }
    void set_rng_state(unsigned long long state) {
        // 引擎 RNG 是 48-bit LCG（game.cpp:25-27）⇒ 折叠到 48 bit 保持不变量。
        rng_state = state & ((1ULL << 48) - 1);
    }
    void consume_rng(int count) {
        for (int i = 0; i < count; ++i) next_random();
    }
    // 探针专用：每个消费点累计被调用次数（game.cpp 里 site tag 统计）。
    std::vector<long long> rng_site_counts() const;
"""

# ─────────────────────── 补丁 2：game.cpp 计数器 ───────────────────────
CPP_SITE_ANCHOR = "constexpr std::size_t MAX_JUDGER_PACKET_SIZE = 16 * 1024 * 1024;\n"
CPP_SITE_NEW = CPP_SITE_ANCHOR + """
// ── 探针专用：RNG 消费点计数器（不属于游戏逻辑，不影响任何判定）──
enum RngSite {
    SITE_OTHER = 0,
    SITE_CHOOSE_RANDOM_LEGAL = 1,
    SITE_BEWITCH_TARGET = 2,
    SITE_MOVE_LEGACY_RANDOM = 3,
    SITE_MOVE_ENH_RANDOM = 4,
    SITE_RANDOMIZED_MOVE_LEGACY = 5,
    SITE_RANDOMIZED_MOVE_ENH = 6,
    SITE_RANDOM_MOVE_STEPS = 7,
    SITE_TELEPORT = 8,
    SITE_DRIFT_ITEM = 9,
    SITE_SPAWN_PROFILE = 10,
    SITE_SIEGE_SPAWN = 11,
    SITE_COUNT = 12
};
int g_rng_site = SITE_OTHER;
long long g_rng_site_draws[SITE_COUNT] = {0};
unsigned long long g_rng_draws_total = 0;
struct RngSiteScope {
    int prev;
    explicit RngSiteScope(int site) : prev(g_rng_site) { g_rng_site = site; }
    ~RngSiteScope() { g_rng_site = prev; }
};
"""

CPP_NEXT_RANDOM_OLD = """unsigned long long Game::next_random() {
    rng_state = (RNG_MULTIPLIER * rng_state + RNG_INCREMENT) & RNG_MASK;
    return rng_state;
}
"""
CPP_NEXT_RANDOM_NEW = """unsigned long long Game::next_random() {
    rng_state = (RNG_MULTIPLIER * rng_state + RNG_INCREMENT) & RNG_MASK;
    ++g_rng_draws_total;
    ++g_rng_site_draws[g_rng_site];
    return rng_state;
}
"""

CPP_TAIL = """
// ── 探针专用访问器（定义在 game.cpp 里才能看到匿名 namespace 的计数器）──
std::vector<long long> Game::rng_site_counts() const {
    return std::vector<long long>(g_rng_site_draws, g_rng_site_draws + SITE_COUNT);
}
"""

# ─────────────────────── 补丁 3：消费点打 tag ───────────────────────
# 每条 = (名字, old, new)；old 必须在副本里**恰好出现 1 次**，否则脚本中止。
SITE_PATCHES: list[tuple[str, str, str]] = [
    ("choose_random_legal_move",
     """    if (candidates.empty())
        return Ant::NoMove;
    return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
}

void Game::mark_risk_fields_dirty() {
""",
     """    if (candidates.empty())
        return Ant::NoMove;
    RngSiteScope _rng_site_scope(SITE_CHOOSE_RANDOM_LEGAL);
    return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
}

void Game::mark_risk_fields_dirty() {
"""),
    ("random_bewitch_target",
     """    if (cells.empty())
        return {own.x, own.y};
    return cells[random_index(static_cast<int>(cells.size()))];
""",
     """    if (cells.empty())
        return {own.x, own.y};
    RngSiteScope _rng_site_scope(SITE_BEWITCH_TARGET);
    return cells[random_index(static_cast<int>(cells.size()))];
"""),
    ("choose_ant_move_legacy: Randomized",
     """    if (ant.get_behavior() == Ant::Behavior::Randomized)
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);

    std::vector<double> damage_scores =
""",
     """    if (ant.get_behavior() == Ant::Behavior::Randomized) {
        RngSiteScope _rng_site_scope(SITE_RANDOMIZED_MOVE_LEGACY);
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
    }

    std::vector<double> damage_scores =
"""),
    ("choose_ant_move_legacy: softmax",
     """    if (total <= 0.0)
        return std::get<0>(candidates[0]);
    double threshold = random_float();
""",
     """    if (total <= 0.0)
        return std::get<0>(candidates[0]);
    double threshold;
    {
        RngSiteScope _rng_site_scope(SITE_MOVE_LEGACY_RANDOM);
        threshold = random_float();
    }
"""),
    ("choose_ant_move_enhanced: Randomized",
     """    if (ant.get_behavior() == Ant::Behavior::Randomized)
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
    if (ant.get_behavior() == Ant::Behavior::Bewitched)
        return choose_ant_move_legacy(ant);
""",
     """    if (ant.get_behavior() == Ant::Behavior::Randomized) {
        RngSiteScope _rng_site_scope(SITE_RANDOMIZED_MOVE_ENH);
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
    }
    if (ant.get_behavior() == Ant::Behavior::Bewitched)
        return choose_ant_move_legacy(ant);
"""),
    ("choose_ant_move_enhanced: softmax",
     """    double threshold = random_float();
    double cumulative = 0.0;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        cumulative += probs[i] / total;
        if (threshold <= cumulative) {
            commit_annotation(i);
""",
     """    double threshold;
    {
        RngSiteScope _rng_site_scope(SITE_MOVE_ENH_RANDOM);
        threshold = random_float();
    }
    double cumulative = 0.0;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        cumulative += probs[i] / total;
        if (threshold <= cumulative) {
            commit_annotation(i);
"""),
    ("resolve_random_move_steps",
     """        int move = random_index(3) < 2 ? choose_random_legal_move(ant)
                                       : choose_ant_move(ant);
""",
     """        int rng_step_roll;
        {
            RngSiteScope _rng_site_scope(SITE_RANDOM_MOVE_STEPS);
            rng_step_roll = random_index(3);
        }
        int move = rng_step_roll < 2 ? choose_random_legal_move(ant)
                                     : choose_ant_move(ant);
"""),
    ("teleport_ants",
     """        int ant_idx = random_index(static_cast<int>(eligible.size()));
""",
     """        int ant_idx;
        {
            RngSiteScope _rng_site_scope(SITE_TELEPORT);
            ant_idx = random_index(static_cast<int>(eligible.size()));
        }
"""),
    ("drift_items",
     """            auto cell = cells[random_index(static_cast<int>(cells.size()))];
            it.x = cell.first;
""",
     """            std::pair<int, int> cell;
            {
                RngSiteScope _rng_site_scope(SITE_DRIFT_ITEM);
                cell = cells[random_index(static_cast<int>(cells.size()))];
            }
            it.x = cell.first;
"""),
    ("generate_ants: spawn profile",
     """    auto draw_spawn_profile = [this]() {
        double roll = random_float();
""",
     """    auto draw_spawn_profile = [this]() {
        RngSiteScope _rng_site_scope(SITE_SPAWN_PROFILE);
        double roll = random_float();
"""),
    ("generate_ants: siege spawn",
     """        SpawnProfile profile = draw_spawn_profile();
        spawn_from_tower(tower, profile.kind, profile.behavior);
        if (tower.get_type() == TowerType::ProducerSiege &&
            random_float() <= tower.get_siege_spawn_chance()) {
""",
     """        SpawnProfile profile = draw_spawn_profile();
        spawn_from_tower(tower, profile.kind, profile.behavior);
        bool siege_spawn_roll = false;
        if (tower.get_type() == TowerType::ProducerSiege) {
            RngSiteScope _rng_site_scope(SITE_SIEGE_SPAWN);
            siege_spawn_roll = random_float() <= tower.get_siege_spawn_chance();
        }
        if (siege_spawn_roll) {
"""),
]

# ─────────────────────── 补丁 4：binding.cpp ───────────────────────
BIND_PATCHES: list[tuple[str, str, str]] = [
    ("module name",
     "PYBIND11_MODULE(native_game, m) {",
     "PYBIND11_MODULE(native_game_rng, m) {"),
    ("new methods",
     '        .def("rng_state_now", &Game::rng_state_now)\n',
     '        .def("rng_state_now", &Game::rng_state_now)\n'
     '        .def("get_rng_state", &Game::get_rng_state)\n'
     '        .def("set_rng_state", &Game::set_rng_state, py::arg("state"))\n'
     '        .def("consume_rng", &Game::consume_rng, py::arg("count"))\n'
     '        .def("rng_site_counts", &Game::rng_site_counts)\n'),
]

SOURCES = ["game.cpp", "ant.cpp", "map.cpp", "building.cpp", "coin.cpp",
           "item.cpp", "operation.cpp", "aco.cpp", "comm_judger.cpp",
           "output.cpp"]


def apply(text: str, old: str, new: str, name: str, path: Path) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[patch] ✗ {path.name}: 锚点 '{name}' 出现 {n} 次（应为 1）"
                         f"\n----- 锚点原文 -----\n{old}")
    return text.replace(old, new, 1)


def prepare_sources() -> None:
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
    t = apply(t, HPP_ANCHOR, HPP_NEW, "rng api", hpp)
    hpp.write_text(t, encoding="utf-8")

    cpp = ENG / "src" / "game.cpp"
    t = cpp.read_text(encoding="utf-8")
    t = apply(t, CPP_SITE_ANCHOR, CPP_SITE_NEW, "site counters", cpp)
    t = apply(t, CPP_NEXT_RANDOM_OLD, CPP_NEXT_RANDOM_NEW, "next_random", cpp)
    for name, old, new in SITE_PATCHES:
        t = apply(t, old, new, name, cpp)
    t += CPP_TAIL
    cpp.write_text(t, encoding="utf-8")

    bind = ENG / "binding.cpp"
    t = bind.read_text(encoding="utf-8")
    for name, old, new in BIND_PATCHES:
        t = apply(t, old, new, name, bind)
    bind.write_text(t, encoding="utf-8")
    print(f"[patch] ✓ 副本已打补丁: {ENG}")


def compile_pyd() -> Path:
    def q(py: Path, code: str) -> str:
        return subprocess.check_output([str(py), "-c", code], text=True).strip()

    py_inc = q(PYTHON, "import sysconfig; print(sysconfig.get_paths()['include'])")
    py_libdir = q(PYTHON, "import sysconfig, os; print(os.path.join(sysconfig"
                          ".get_paths()['data'], 'libs'))")
    pbi_inc = q(PBI_PYTHON, "import pybind11; print(pybind11.get_include())")
    py_ver = q(PYTHON, "import sys; print(f'{sys.version_info.major}"
                       "{sys.version_info.minor}')")
    print(f"[build] python {py_ver}  include={py_inc}\n"
          f"[build] libdir={py_libdir}  pybind11={pbi_inc}")

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    out_pyd = OUT / "native_game_rng.pyd"
    cmd = [str(GXX), "-std=c++17", "-O2", "-fPIC", "-shared",
           f"-I{pbi_inc}", f"-I{py_inc}", f"-I{ENG / 'include'}",
           *[str(ENG / "src" / s) for s in SOURCES],
           str(ENG / "m1_support.cpp"), str(ENG / "binding.cpp"),
           f"-L{py_libdir}", f"-lpython{py_ver}", "-o", str(out_pyd)]
    env = dict(os.environ)
    # MinGW 的 cc1plus 用 TMP 写中间文件：必须纯 ASCII，否则静默 Permission denied
    env.update(TMP=str(TMP), TEMP=str(TMP), TMPDIR=str(TMP))
    print("[build] g++ ...")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-8000:])
        raise SystemExit(f"[build] ✗ g++ rc={r.returncode}")
    if r.stderr.strip():
        print("[build] g++ stderr:\n" + r.stderr[-3000:])
    for dll in ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"):
        shutil.copy2(Path(r"C:\mingw64\bin") / dll, OUT / dll)
    sz = out_pyd.stat().st_size
    print(f"[build] ✓ {out_pyd}  ({sz} bytes)")
    return out_pyd


SMOKE = r"""
import sys
sys.path.insert(0, r'C:\Temp\rng_probe\out')
import native_game_rng as ng
print('[smoke] module:', ng.__file__)
print('[smoke] has:', [n for n in ('get_rng_state', 'set_rng_state', 'consume_rng',
                                   'rng_site_counts', 'rng_state_now') if hasattr(ng.NativeGame, n)])
g = ng.NativeGame()
g.init_game(seed=7, movement_policy='enhanced', cold=True)
s0 = g.get_rng_state()
g.consume_rng(5)
s5 = g.get_rng_state()
# 手推一遍 LCG 验证 consume_rng 就是"推进流"
M, INC, MASK = 25214903917, 11, (1 << 48) - 1
s = s0
for _ in range(5):
    s = (M * s + INC) & MASK
print('[smoke] consume_rng(5) 与手推 LCG 一致:', s == s5)
g.set_rng_state(s0)
print('[smoke] set_rng_state 回滚成功:', g.get_rng_state() == s0)
g.set_rng_state(1 << 50)   # 超 48 bit 应被折叠
print('[smoke] set_rng_state 折叠到 48 bit:', g.get_rng_state() == ((1 << 50) & MASK))
g.advance_round()
print('[smoke] site counts:', g.rng_site_counts())
"""


def smoke() -> None:
    r = subprocess.run([str(PYTHON), "-c", SMOKE], capture_output=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit(f"[smoke] ✗ rc={r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    if not args.skip_build:
        prepare_sources()
        compile_pyd()
    if args.smoke:
        smoke()
    return 0


if __name__ == "__main__":
    sys.exit(main())
