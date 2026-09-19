"""探针：**当场上已有塔时，ExampleAI 的选法（`bundles[1:8]` 里按 score 取最大）会不会变成拆塔？**

背景（2026-09-19）：`_example_legal_bundle`（`--inject-example-prob` 的注入器）里加了个
`_PRODUCTIVE` 过滤器（排除 DOWNGRADE / 超武），理由是"官方启发式的评分里金币权重高，
一旦场上有塔它打分最高的就变成拆塔"。**但这个理由是我推的，没测过。**
而 `cand_pick_k96` 的 `first` 臂（同样是"取 score 最大的 bundle"）在实战里是
`BUILD 93% / UPGRADE 5% / DOWNGRADE 1%` —— 说明那个推断可能是错的。

本探针直接量：
  1. 用 ExampleAI 的选法走一局（它建塔多 ⇒ 场上有塔的局面很多）；
  2. 每个"该玩家场上有塔"的决策点上，打印 top-8 的 (name, score)，
     并标出 ExampleAI 实际会选哪个；
  3. 统计：ExampleAI 的选择里 BUILD / UPGRADE / DOWNGRADE / 其他 各占多少。

判读：
  * 降级占比 ~0 ⇒ **过滤器没作用**（不改变行为）⇒ 应该删掉，让 `_example_legal_bundle`
    与 `ai_example.py` 逐字一致（我的注释理由也是错的）。
  * 降级占比高 ⇒ 过滤器确实在防"建↔拆循环"⇒ 保留。

用法: python _tmp_probe_example_pick.py [--seed 0] [--rounds 200]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--verbose", type=int, default=12, help="打印前 N 个有塔决策点")
    a = ap.parse_args()

    from SDK.utils.actions import ActionCatalog
    from SDK.utils.features import FeatureExtractor
    from SDK.backend.model import OperationType
    from my_ai.az_intent.az_selfplay import make_initial_state

    name = {int(m): m.name for m in OperationType}
    feat = FeatureExtractor(max_actions=96)
    cat = ActionCatalog(max_actions=96, feature_extractor=feat)
    st = make_initial_state(a.seed, native_engine=False)

    pick = Counter()          # ExampleAI 实际选的（顶层 op）
    pick_withtower = Counter()  # 只在"场上有自己的塔"时
    n_tower_dec, n_dec = 0, 0
    shown = 0

    for rnd in range(a.rounds):
        if st.terminal:
            break
        for pl in (0, 1):
            if st.terminal:
                break
            b = cat.build(st, pl)
            if not b:
                continue
            n_dec += 1
            short = b[1:8] or b[0:1]
            best = max(short, key=lambda x: (x.score, -len(x.operations)))
            has_tower = len([t for t in st.towers if t.player == pl]) > 0
            tag = "?"
            if best.operations:
                tops = {int(o.op_type) for o in best.operations}
                if len(tops) == 1:
                    tag = name.get(next(iter(tops)), "?")
                else:
                    tag = "+".join(sorted(name.get(t, "?") for t in tops))
            pick[tag] += 1
            if has_tower:
                n_tower_dec += 1
                pick_withtower[tag] += 1
                if shown < a.verbose:
                    shown += 1
                    top8 = ", ".join("%s=%.1f" % (x.name, x.score) for x in short)
                    print("[r%02d p%d 有%d塔] 选:%s | top8: %s"
                          % (rnd, pl, len([t for t in st.towers if t.player == pl]), tag, top8),
                          flush=True)
            st.apply_operation_list(pl, list(best.operations))
        if not st.terminal:
            st.advance_round()

    def dump(c: Counter, label: str) -> None:
        tot = sum(c.values())
        print("%s (n=%d): %s" % (label, tot, ", ".join(
            "%s:%d(%.0f%%)" % (k, v, 100.0 * v / max(tot, 1)) for k, v in c.most_common(8))))

    print("\n===== seed=%d rounds=%d terminal=%s =====" % (a.seed, rnd, st.terminal), flush=True)
    dump(pick, "全部决策点 ExampleAI 的选择")
    dump(pick_withtower, "仅'场上有自己的塔'的决策点 (%d/%d)" % (n_tower_dec, n_dec))


if __name__ == "__main__":
    main()
