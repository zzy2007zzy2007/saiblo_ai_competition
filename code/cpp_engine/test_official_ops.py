"""Targeted op-type coverage: our pyd vs the official C++ binary.

Complements the random test (test_official_consistency.py) by exercising each
operation type with hand-crafted legal / illegal / edge-case scenarios, fed
identically to both engines and compared per round (same reliable fields:
camps / coins / base levels / weapon cooldowns exact, tower events subset).

Usage:
    python code/cpp_engine/test_official_ops.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "Ant-Game"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_game
from test_official_consistency import official_run, our_state, check_round

# Each entry: (name, [P0 ops, P1 ops]) — one round.
# Ops are the SAME for both engines; legality differences are part of the test
# (both must agree on what applies and what is skipped).
SCENARIOS: list[tuple[str, list[list[list[int]]]]] = [
    # ── build (11) ──
    ("build own highland + opponent highland + occupied-vs-camp",
     [[(11, 5, 7), (11, 12, 9)], [(11, 13, 10)]]),
    ("build on occupied cell (repeat) + out-of-bounds",
     [[(11, 5, 7), (11, 18, 18)], []]),
    # ── upgrade (12) ──
    ("upgrade own tower to HEAVY + invalid type + non-existent id + opponent tower",
     [[(12, 0, 1), (12, 99, 1)], [(12, 0, 2)]]),
    ("upgrade twice same tower same round (used_tower)",
     [[(12, 0, 3)], []]),
    # ── downgrade (13) ──
    ("downgrade non-Basic tower (was upgraded) + downgrade opponent's",
     [[(13, 0, 0)], [(13, 0, 0)]]),
    ("downgrade Basic tower (destroy) + non-existent id",
     [[(13, 99, 0)], [(13, 1, 0)]]),
    # ── build+destroy same tower same round (used_tower) ──
    ("build then downgrade same tower (used_tower keeps it)",
     [[(11, 6, 16), (13, 3, 0)], []]),
    # ── weapons (21/22/23/24) ──
    ("lightning + emp + deflector + evasion (valid positions)",
     [[(21, 5, 7), (23, 9, 9)], [(22, 13, 10), (24, 13, 10)]]),
    ("weapons again same round -> cooldown skip",
     [[(21, 5, 7)], [(22, 13, 10)]]),
    ("weapons on VOID/edge cells",
     [[(23, 0, 0), (24, 18, 0)], [(21, 0, 18)]]),
    # ── base upgrades (31/32) ──
    ("barrack + ant upgrade",
     [[(31, 0, 0), (32, 0, 0)], [(31, 0, 0), (32, 0, 0)]]),
    ("base upgrade again (camp_upgraded_flag + level cap)",
     [[(31, 0, 0), (32, 0, 0), (31, 0, 0)], [(31, 0, 0), (32, 0, 0)]]),
    # ── EMP-shielded build ──
    ("emp near tower zone then build there (shielded)",
     [[(11, 6, 16)], [(22, 6, 16)]]),
    ("build in EMP zone after deploy",
     [[(11, 6, 16)], []]),
]


def main() -> None:
    seed = 99
    rounds_ops = [ro for _, ro in SCENARIOS]
    replay_path = Path("C:/mingw64/test_official_ops.json")

    official = official_run(seed, rounds_ops, replay_path)
    g = native_game.NativeGame()
    g.init_game(seed=seed, movement_policy="enhanced", cold=True)

    print(f"running {len(SCENARIOS)} curated op-type rounds...")
    for i, ((name, _), o) in enumerate(zip(SCENARIOS, official)):
        for p, ops in enumerate(SCENARIOS[i][1]):
            g.apply_operation_list(p, ops)
        g.advance_round()
        u = our_state(g)
        err = check_round(o, u, i)
        if err:
            print(f"FAIL at round {i+1} [{name}]: {err}")
            print(f"  official: camps={o['camps']} coins={o['coins']} towers={o['towers']} "
                  f"wc={o['weaponCds']}")
            print(f"  ours    : camps={u['camps']} coins={u['coins']} towers={u['towers']} "
                  f"wc={u['weaponCds']}")
            sys.exit(1)
        print(f"  OK round {i+1}: {name}")
    print(f"\nall {len(SCENARIOS)} curated op-type rounds consistent "
          f"(official binary vs our pyd)")


if __name__ == "__main__":
    main()
