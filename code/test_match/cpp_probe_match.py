"""`cpp_probe_match.dump_state(state)` —— 把一局状态序列化成可 JSON 的 dict。

## 为什么需要这个文件
`match_via_sdk.py:459` 里写着 `import cpp_probe_match as _cpm` 用于 `--dump-round` 的
快照落盘，但**这个模块在全仓从不存在**（2026-10-20 两个 worktree 都搜过）⇒ 那条 dump
路径历史上从未真正跑通（一直静默失败在 `ModuleNotFoundError`）。

## 字段与亚军 `PublicState`（antgame_sdk/sdk.hpp:504）对齐
  round_index / coins / bases(hp) / towers / ants / active_effects /
  super_weapon_usage / weapon_cooldowns / die_count / old_count
这样 `state_dump_to_replay.py` → `sdk_lure_inspector` 的链路才能把状态喂进亚军的
`make_defense_simulator` / `generate_root_plans`。

序列化格式**复用** `match_via_sdk._dump_raw_state` 里已经验证过的那套（facade 的属性名
已在那里踩过坑），保证与 `ents_seed*.jsonl` 逐字段一致。
"""
from __future__ import annotations


def dump_state(state) -> dict:
    """把 BackendState / GameStateFacade 序列化成 dict（所有叶子都可 JSON）。"""
    towers = [[int(t.tower_id), int(t.player), int(t.x), int(t.y),
               int(t.tower_type), int(t.hp), float(t.cooldown_clock)]
              for t in state.towers]
    ants = [[int(a.ant_id), int(a.player), int(a.x), int(a.y), int(a.hp),
             int(a.max_hp), int(a.kind), int(a.age), int(a.level),
             int(a.behavior), int(a.status), int(bool(a.frozen))]
            for a in state.ants]
    try:
        effects = [[int(e.weapon_type), int(e.player), int(e.x), int(e.y),
                    int(e.remaining_turns)] for e in state.active_effects]
    except Exception:  # noqa: BLE001  某些实现没有 active_effects
        effects = []
    try:
        wcd = [[int(state.weapon_cooldowns[p, w]) for w in range(5)] for p in (0, 1)]
    except Exception:  # noqa: BLE001
        wcd = [[0] * 5, [0] * 5]
    try:
        usage = [int(state.super_weapon_usage[p]) for p in (0, 1)]
    except Exception:  # noqa: BLE001
        usage = [0, 0]
    try:
        die = [int(state.die_count[p]) for p in (0, 1)]
        old = [int(state.old_count[p]) for p in (0, 1)]
    except Exception:  # noqa: BLE001
        die, old = [0, 0], [0, 0]

    return {
        "round_index": int(getattr(state, "round_index", 0)),
        "terminal": bool(getattr(state, "terminal", False)),
        "winner": int(getattr(state, "winner", -1) or -1),
        "coins": [int(c) for c in state.coins],
        "bases": [[int(b.hp), int(b.generation_level), int(b.ant_level)]
                  for b in state.bases],
        "towers": towers,
        "ants": ants,
        "active_effects": effects,
        "weapon_cooldowns": wcd,
        "super_weapon_usage": usage,
        "die_count": die,
        "old_count": old,
    }
