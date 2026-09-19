"""单元测试：BUILD_ORDERS 正确性验证

注意：本测试直接验证数据结构，不依赖 SDK 环境，
因此可以直接运行而无需安装 SDK 依赖。
"""

from typing import Dict, List, Tuple

# ── 从 action_constants.py 复制的数据结构 ──────────────────────────────────────
TOWER_POSITIONS: List[Tuple[int, int]] = [
    (4, 2),
    (4, 9),
    (5, 6),
    (5, 9),
    (6, 7),
    (6, 14),
    (7, 8),
    (8, 7),
    (10, 7),
    (11, 5),
    (11, 14),
    (12, 6),
    (12, 9),
    (13, 9),
    (13, 15),
    (14, 9),
]

HIGHLAND_CELLS: Dict[int, List[Tuple[int, int]]] = {
    0: [
        (4, 2),
        (4, 3),
        (4, 9),
        (4, 15),
        (4, 16),
        (5, 3),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 11),
        (5, 12),
        (5, 15),
        (6, 1),
        (6, 2),
        (6, 4),
        (6, 7),
        (6, 9),
        (6, 11),
        (6, 14),
        (6, 16),
        (6, 17),
        (7, 1),
        (7, 5),
        (7, 8),
        (7, 10),
        (7, 13),
        (7, 17),
        (8, 2),
        (8, 4),
        (8, 7),
        (8, 11),
        (8, 14),
        (8, 16),
    ],
    1: [
        (9, 2),
        (9, 4),
        (9, 14),
        (9, 16),
        (10, 7),
        (10, 8),
        (10, 10),
        (10, 11),
        (11, 1),
        (11, 2),
        (11, 4),
        (11, 5),
        (11, 13),
        (11, 14),
        (11, 16),
        (11, 17),
        (12, 1),
        (12, 6),
        (12, 7),
        (12, 9),
        (12, 11),
        (12, 12),
        (12, 17),
        (13, 2),
        (13, 3),
        (13, 7),
        (13, 9),
        (13, 11),
        (13, 15),
        (13, 16),
        (14, 3),
        (14, 9),
        (14, 15),
    ],
}

# ── 从 rule_based_agents.py 复制的 BUILD_ORDERS ────────────────────────────────
BUILD_ORDERS: Dict[int, List[Tuple[int, int]]] = {
    0: [
        (4, 9),
        (5, 9),
        (7, 8),
        (8, 7),
        (6, 7),
        (5, 6),
        (6, 14),
        (4, 2),
    ],
    1: [
        (12, 9),
        (13, 9),
        (14, 9),
        (10, 7),
        (12, 6),
        (11, 5),
        (11, 14),
        (13, 15),
    ],
}


class TestBuildOrders:
    """验证 BUILD_ORDERS 的全量正确性"""

    def test_player0_has_8_unique_positions(self):
        order = BUILD_ORDERS[0]
        assert len(order) == 8, f"Player 0: expected 8 positions, got {len(order)}"
        assert len(set(order)) == 8, "Player 0: has duplicate positions"

    def test_player1_has_8_unique_positions(self):
        order = BUILD_ORDERS[1]
        assert len(order) == 8, f"Player 1: expected 8 positions, got {len(order)}"
        assert len(set(order)) == 8, "Player 1: has duplicate positions"

    def test_coverage_matches_tower_positions(self):
        all_positions = set(BUILD_ORDERS[0]) | set(BUILD_ORDERS[1])
        expected = set(TOWER_POSITIONS)
        assert all_positions == expected, (
            f"Coverage mismatch. "
            f"Missing: {expected - all_positions}. "
            f"Extra: {all_positions - expected}"
        )

    def test_player0_positions_in_highlands(self):
        for pos in BUILD_ORDERS[0]:
            assert pos in HIGHLAND_CELLS[0], (
                f"Player 0 position {pos} is not in Player 0's highlands"
            )

    def test_player1_positions_in_highlands(self):
        for pos in BUILD_ORDERS[1]:
            assert pos in HIGHLAND_CELLS[1], (
                f"Player 1 position {pos} is not in Player 1's highlands"
            )

    def test_no_overlap_between_players(self):
        overlap = set(BUILD_ORDERS[0]) & set(BUILD_ORDERS[1])
        assert not overlap, f"Overlapping positions: {overlap}"

    def test_player0_first_position_is_midline(self):
        assert BUILD_ORDERS[0][0] == (4, 9), (
            f"Player 0 first build position should be (4, 9), "
            f"got {BUILD_ORDERS[0][0]}"
        )

    def test_player1_first_position_is_midline(self):
        assert BUILD_ORDERS[1][0] == (12, 9), (
            f"Player 1 first build position should be (12, 9), "
            f"got {BUILD_ORDERS[1][0]}"
        )

    def test_ordering_player0_distances_ascending_within_tiers(self):
        """验证 Player 0 的 Tier 2 （非中线位）按到 (9,9) 距离升序"""
        # Tier 1（中线位）: (4,9), (5,9) - 不检查距离
        # Tier 2 剩余位应按距离升序
        tier2 = BUILD_ORDERS[0][2:]
        distances = [((x - 9) ** 2 + (y - 9) ** 2) ** 0.5 for x, y in tier2]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 0.001, (
                f"Player 0 Tier 2 not sorted by distance asc: "
                f"{list(zip(tier2, [round(d, 2) for d in distances]))}"
            )

    def test_ordering_player1_distances_ascending_within_tiers(self):
        """验证 Player 1 的 Tier 2（非中线位）按到 (9,9) 距离升序"""
        # Tier 1（中线位）: (12,9), (13,9), (14,9) - 不检查距离
        tier2 = BUILD_ORDERS[1][3:]
        distances = [((x - 9) ** 2 + (y - 9) ** 2) ** 0.5 for x, y in tier2]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 0.001, (
                f"Player 1 Tier 2 not sorted by distance asc: "
                f"{list(zip(tier2, [round(d, 2) for d in distances]))}"
            )
