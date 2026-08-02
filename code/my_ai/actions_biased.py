"""Biased ActionCatalog that boosts scores for rare action classes.

This subclass overrides specific scoring methods to make rare classes
(like HEAVY tower, LIGHTNING_STORM) more attractive to the AI.

Purpose: when collecting distillation training data, the biased teacher
will naturally produce more samples of rare classes, giving the student
model more examples to learn from.

Usage:
    from my_ai.actions_biased import BiasedActionCatalog
    catalog = BiasedActionCatalog()
    agent = ExampleAI()
    agent.catalog = catalog  # swap in the biased catalog
"""
from __future__ import annotations

from SDK.utils.actions import ActionCatalog, ActionBundle
from SDK.utils.constants import (
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TowerType,
    OperationType,
)
from SDK.backend.state import BackendState
from SDK.backend.model import Operation


class BiasedActionCatalog(ActionCatalog):
    """ActionCatalog with boosted scores for rare action classes.

    Tuned biases (applied per-candidate):
      - HEAVY family (class 1,2):  +5.0 fit bonus  (was 13 samples)
      - SNIPER (class 8):          +4.0 fit bonus  (not seen in training data)
      - MISSILE (class 12):        +3.0 fit bonus  (not seen)
      - PRODUCER_FAST (class 13):  +3.0 fit bonus  (was 162 samples)
      - LIGHTNING_STORM (class17): +3.0 score bonus, threshold lowered 1.5→0.5
    """

    # ── Tower type fit bonuses ──────────────────────────────────────────────

    _RARE_TOWER_BOOST: dict[TowerType, float] = {
        TowerType.HEAVY: 5.0,
        TowerType.HEAVY_PLUS: 3.0,
        TowerType.SNIPER: 4.0,
        TowerType.MISSILE: 3.0,
        TowerType.PRODUCER_FAST: 3.0,
        TowerType.PRODUCER_SIEGE: 2.0,
        TowerType.PRODUCER_MEDIC: 2.0,
        TowerType.BEWITCH: 2.0,
    }

    def _tower_type_fit(self, tower_type: TowerType, local_density: float,
                        forward_distance: int) -> float:
        base = super()._tower_type_fit(tower_type, local_density, forward_distance)
        return base + self._RARE_TOWER_BOOST.get(tower_type, 0.0)

    # ── Build candidates: also boost building at more positions ─────────────

    def _build_candidates(self, state: BackendState,
                          player: int) -> list[ActionBundle]:
        results = super()._build_candidates(state, player)
        # Slightly boost all build scores (leads to more non-HOLD actions)
        for b in results:
            b.score += 30.0  # big boost to make building more attractive vs HOLD
        return results

    # ── Super weapon thresholds and scoring ─────────────────────────────────

    def _superweapon_candidates(self, state: BackendState,
                                 player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        enemy = 1 - player
        enemy_ants = state.ants_of(enemy)
        my_ants = state.ants_of(player)
        enemy_towers = state.towers_of(enemy)

        # --- LIGHTNING_STORM (class 17, was 6 samples) ---
        # Boost value and lower threshold from 1.5 → 0.5
        if (enemy_ants or enemy_towers) \
                and state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0 \
                and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost:
            centers = {(ant.x, ant.y) for ant in enemy_ants}
            centers.update((tower.x, tower.y) for tower in enemy_towers)
            best = max(
                ((x, y, self._storm_value(state, player, x, y) + 3.0) for x, y in centers),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 0.5:  # lowered threshold
                op = Operation(OperationType.USE_LIGHTNING_STORM, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"storm@{best[0]},{best[1]}", (op,), best[2],
                        ("weapon", "storm")))

        # --- EMP_BLASTER (class 18, not seen in training data) ---
        if enemy_towers \
                and state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER] == 0 \
                and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost:
            centers = {(tower.x, tower.y) for tower in enemy_towers}
            scored = [(x, y, self._emp_value(state, player, x, y)) for x, y in centers]
            best = max(scored, key=lambda item: item[2], default=None)
            if best and best[2] > 1.0:  # lowered from 2.0
                op = Operation(OperationType.USE_EMP_BLASTER, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"emp@{best[0]},{best[1]}", (op,), best[2],
                        ("weapon", "emp")))

        # --- DEFLECTOR (class 19, already 1420 samples) ---
        # Keep original threshold, slight score boost
        if my_ants \
                and state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR] == 0 \
                and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost:
            best = max(
                ((ant.x, ant.y, self._deflector_value(state, player, ant.x, ant.y) + 1.0)
                 for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.5:
                op = Operation(OperationType.USE_DEFLECTOR, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"deflect@{best[0]},{best[1]}", (op,), best[2],
                        ("weapon", "shield")))

        # --- EMERGENCY_EVASION (class 20, already 1227 samples) ---
        if my_ants \
                and state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION] == 0 \
                and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost:
            best = max(
                ((ant.x, ant.y, self._evasion_value(state, player, ant.x, ant.y) + 1.0)
                 for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.0:
                op = Operation(OperationType.USE_EMERGENCY_EVASION, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(
                        f"evasion@{best[0]},{best[1]}", (op,), best[2],
                        ("weapon", "panic")))

        return results
