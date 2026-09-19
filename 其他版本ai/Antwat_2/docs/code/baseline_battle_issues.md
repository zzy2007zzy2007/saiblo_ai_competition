# Baseline Battle Issues Analysis

## Overview

This document describes critical issues found in the baseline battle system that cause premature game termination and abnormal battle outcomes.

---

## Issue 1: Coordinate System Mismatch in Sample Agent

### Root Cause
The `sample agent` uses a different coordinate system for strategic build positions compared to the main game engine.

### Evidence

**Sample Agent's STRATEGIC_BUILD_ORDER (baselines/sample/SDK/utils/constants.py:288-301):**
```python
STRATEGIC_BUILD_ORDER = {
    0: ((2, 9), (4, 9), (5, 9), ...),
    1: ((16, 9), (14, 9), (13, 9), ...),  # Player 1 build positions
}
```

**Main Engine's HIGHLAND_CELLS (ppo/src/ppo_antwar/utils/action_constants.py:46-49):**
```python
HIGHLAND_CELLS = {
    1: [(9, 2), (9, 4), (9, 14), (9, 16), (10, 7), ...],  # Player 1 highland cells
}
```

### Impact
- Sample agent generates tower build operations at positions that don't belong to its highland
- Engine validation rejects these operations
- Games end prematurely (1-12 rounds instead of normal 200+ rounds)
- Battle results show excessive draws

### Solution
Align `STRATEGIC_BUILD_ORDER` in sample agent with the main engine's `HIGHLAND_CELLS` definition.

---

## Issue 2: Violation of Design Principles in Action Validation

### Principle Violated
> "关于游戏对战和游戏规则的部分，最大限度利用Ant-Game/SDK，尽量避免涉及游戏规则的实现，只做游戏引擎的wrapper"

### Current Implementation Issues

| Problem | Description |
|---------|-------------|
| **Rule Re-implementation** | Manual position validation against `HIGHLAND_CELLS` in `_validate_action()` |
| **Multiple Sources of Truth** | Map definitions exist in multiple places (main project, sample agent, battle_single) |
| **Validation Logic Fragmentation** | Validation occurs at both action and operation levels inconsistently |

### Example of Non-Compliant Code
```python
# baseline_battle_runner.py:441-450
if ACTION_SPACE_CONFIG['build_tower']['start'] <= action < ACTION_SPACE_CONFIG['build_tower']['end']:
    pos = _action_to_position(action, player_position)
    if pos is not None:
        if pos not in HIGHLAND_CELLS[player_position]:  # Manual rule implementation
            logger.debug(f"[VALIDATION] Position {pos} does not belong to player {player_position}'s highland")
            return 0
```

### Recommended Approach
```python
def _validate_action(action: int, player_position: int, engine=None) -> int:
    # Delegate validation to the engine
    if engine and hasattr(engine, 'can_apply_action'):
        if not engine.can_apply_action(player_position, action):
            logger.debug(f"[VALIDATION] Engine rejected action")
            return 0
    return action
```

---

## Issue 3: Inconsistent Position Display in Logs

### Problem
The `_action_to_string()` function was hardcoded to use `HIGHLAND_CELLS[0]`, causing incorrect position display for player 1 actions.

### Impact
- Logs showed incorrect tower positions for player 1, making debugging difficult
- Example: Action 5 for player 1 was displayed as `BUILD_TOWER(pos=(4, 16))` instead of `BUILD_TOWER(pos=(10, 7))`

### Fix Applied
Modified `_action_to_string()` to accept `player_position` parameter and use the correct highland cells.

---

## Summary of Recommendations

| Priority | Issue | Solution | Status |
|----------|-------|----------|--------|
| **High** | Coordinate mismatch | Align sample agent's STRATEGIC_BUILD_ORDER with main engine | ⚠️ Pending |
| **Medium** | Principle violation | Remove manual rule implementations, use engine validation | ✅ Fixed |
| **Low** | Log inconsistency | Modified _action_to_string() to accept player_position | ✅ Fixed |

---

## Files Affected

1. **baselines/sample/SDK/utils/constants.py** - STRATEGIC_BUILD_ORDER definition
2. **ppo/src/ppo_antwar/league/baseline_battle/baseline_battle_runner.py** - Action validation logic
3. **ppo/src/ppo_antwar/utils/action_constants.py** - Main engine coordinate definitions

---

## Timeline

- **2026-05-11**: Issues discovered during baseline battle testing
- **2026-05-11**: Log display fix applied (`_action_to_string`)
- **2026-05-11**: Validation refactoring completed - all agents now use project SDK, removed manual rule implementations
- **2026-05-11**: Created `SDKInjector` for unified SDK injection
- **2026-05-11**: Created `battle_worker.py` to eliminate code duplication
- **Pending**: Coordinate alignment fix in sample agent (requires modifying Ant-Game/SDK which is not allowed)