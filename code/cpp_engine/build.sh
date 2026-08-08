#!/usr/bin/env bash
# M1: build the thin pybind11 binding (native_game.pyd) from the cpp_engine copy.
# Builds for the TRAINING env python (pytorch-gpu, 3.11) — MinGW-built
# extensions are ABI-OK with MSVC Python 3.11, but NOT 3.13 (base env).
# pybind11 headers come from the base env (header-only, version-independent).
set -e

REPO="D:/2026智能体大赛_新2"
CE="$REPO/code/cpp_engine"
PYTHON="D:/anaconda3/envs/pytorch-gpu/python.exe"
PBI_PYTHON="D:/anaconda3/python.exe"
GXX="/c/mingw64/bin/g++.exe"

PY_INC="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
PY_LIBDIR="$("$PYTHON" -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()['data'], 'libs'))")"
PBI_INC="$("$PBI_PYTHON" -c "import pybind11; print(pybind11.get_include())")"
PY_VER="$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
echo "python: $("$PYTHON" -c "import sys; print(sys.version.split()[0])") include=$PY_INC"
echo "python libdir: $PY_LIBDIR"
echo "pybind11 include: $PBI_INC"

# comm_judger.cpp + output.cpp are compiled too (not just the engine core):
# the core logic (generate_ants etc.) calls Output recording methods, so the
# symbols must resolve.  They are never CALLED through the binding (we use
# init_game, not the judger flow) — they just sit in the module unused.
"$GXX" -std=c++17 -O2 -fPIC -shared \
    -I"$PBI_INC" -I"$PY_INC" \
    -I"$CE/include" \
    "$CE/src/game.cpp" "$CE/src/ant.cpp" "$CE/src/map.cpp" \
    "$CE/src/building.cpp" "$CE/src/coin.cpp" "$CE/src/item.cpp" \
    "$CE/src/operation.cpp" "$CE/src/aco.cpp" \
    "$CE/src/comm_judger.cpp" "$CE/src/output.cpp" \
    "$CE/m1_support.cpp" "$CE/binding.cpp" \
    -L"$PY_LIBDIR" -lpython$PY_VER \
    -o "$CE/native_game.pyd"

# Copy the MinGW runtime DLLs next to the pyd so the Windows DLL loader finds
# them without relying on PATH (C:/mingw64/bin).
for dll in libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll; do
    cp -f "/c/mingw64/bin/$dll" "$CE/$dll" 2>/dev/null || true
done

echo "built: $CE/native_game.pyd"
