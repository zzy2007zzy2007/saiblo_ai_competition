"""故障注入 AI（2026-10-27）：每回合发一条**确定非法**的操作，用来验证裁判侧的非法处理。

为什么需要它：`cold_handle_rule_illegal` 的两种口径（官方默认 = 判负 / 旧本地默认 = 静默跳过）
**在没有非法操作时行为完全相同**（这也是"48 局 0 次"看起来没事的原因）。要演示差异、
并证明"严格模式真的抓得住"，就必须有一个**保证非法**的对手。

默认故障形态：在自己**基地格**上建塔 `11 <我方基地 x> <我方基地 y>`。
官方 `Game::apply_operation` 对 TowerBuild 的头两条检查就是
"position out of range" 与 "there is already a camp" ⇒ 必非法，且与金币/塔数无关
（不会因为"这一回合恰好有钱"而偶尔变成合法）。
`--op=12` 换成历史 bug 的形态（`12 <x> <y>`：把 (x,y) 当成 (tower_id, target_type)）——
在有塔、有钱时它**也**一定非法，但在冷模式下会被静默吞掉、塔等级永远是 0。

用法（`--ai1=` 让它执后手；两种模式各跑一次对比）：
    python code/test_match/match_via_sdk.py --ai1=code/test_match/faulty_ai.py 7
    python code/test_match/match_via_sdk.py --cold-illegal --ai1=code/test_match/faulty_ai.py 7

⚠️ 只做故障注入，不参与任何训练/评测口径。
"""
from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from example_ai import log, read_ops, read_state, recv_line, send_ops  # noqa: E402

try:                                     # bridge 会给 .py AI 设 PYTHONPATH=<repo>/Ant-Game
    from SDK.utils.constants import PLAYER_BASES
except Exception:                        # noqa: BLE001
    EDGE = 10
    MAP_SIZE = 2 * EDGE - 1
    PLAYER_BASES = ((2, EDGE - 1), (MAP_SIZE - 3, EDGE - 1))

KIND = "build_on_base"                   # 由 --op= 覆盖


def decide(player: int, round_idx: int) -> list[tuple[int, ...]]:
    """返回一条**必然非法**的操作。空列表 = 空过（合法）。"""
    if KIND == "upgrade_xy":             # 历史 bug：升级 op 用了 (x, y)
        return [(12, 7, 3)]
    bx, by = PLAYER_BASES[int(player)]   # 在自家基地格上建塔
    if round_idx % 10 == 0:
        log(f"round={round_idx} 注入非法 op=(11,{bx},{by}) kind={KIND}")
    return [(11, int(bx), int(by))]


def main() -> int:
    global KIND
    for a in sys.argv[1:]:
        if a.startswith("--op="):
            KIND = a.split("=", 1)[1]
        elif a == "--upgrade-xy":
            KIND = "upgrade_xy"
    init = recv_line()
    if init is None:
        log("没收到 init 行")
        return 1
    player, seed = (int(x) for x in init.split())
    log(f"player={player} seed={seed} kind={KIND}")
    state = None
    round_idx = 0
    try:
        while True:
            # 官方读写次序：先手方"先写后读"，后手方"先读后写"
            if player == 0:
                send_ops(decide(player, round_idx))
                if read_ops() is None:
                    break
            else:
                if read_ops() is None:
                    break
                send_ops(decide(player, round_idx))
            st = read_state()
            if st is None:
                break
            state = st
            round_idx += 1
    except BrokenPipeError:
        log("裁判关掉了管道（正常收尾）")
    log(f"结束: rounds={round_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
