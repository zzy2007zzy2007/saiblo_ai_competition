"""把 AZ dump 转成行为克隆数据集（2026-10-24，v2 按 round 索引 + coins 匹配）。

背景与踩坑记录：
  · 日志重放不可行（144 局只有 2 局能复现终局血量）；
  · AZ dump 每决策被写 2 条记录，且 **p1 的记录顺序错乱**（n=3 处出现 round=0 的陈旧记录）
    ⇒ 不能按列表下标或 n 配对；
  · 改为：每条记录按它**自己的** `state.round` 归位，重放时用 **coins 匹配**挑记录。

输出：`training_history/bc_dataset/seed<S>.npz`
  board(S,28,19,19) / stats(S,D) / label(S,3)=(op_type,x,y) / cmask(S,24) / pmask(S,19,19)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType
from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.game_state_facade import GameStateFacade
from my_ai.decoder import make_class_mask, make_position_masks

DUMP = REPO / "training_history/az_dump_bc"
OUT = REPO / "training_history/bc_dataset"


def load_by_round(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stt = rec.get("state") or {}
            r = stt.get("round")
            if r is None:
                continue
            out.setdefault(int(r), []).append(rec)
    return out


def chosen_ops(rec: dict) -> list:
    cands = rec.get("cands") or []
    ch = int(rec.get("chosen", -1))
    if not (0 <= ch < len(cands)):
        return []
    return [tuple(int(v) for v in op) for op in cands[ch]]


def pick(records: list, facade):
    """挑与当前局面最匹配的记录。

    优先级：① coins 完全相同；② bases_hp 相同（p1 的 round-0 记录是"p0 已出手后"的
    快照，coins 已扣钱但基地血量一样）；③ 退化为该回合第一条（宁可冒一点风险也要保持
    重放不中断——中断会让后续所有回合级联失配）。
    """
    if not records:
        return None
    coins = [int(c) for c in facade.coins]
    for rec in records:
        if [int(c) for c in (rec.get("state") or {}).get("coins", [])] == coins:
            return rec
    hp = [int(b.hp) for b in facade.bases]
    for rec in records:
        if [int(h) for h in (rec.get("state") or {}).get("bases_hp", [])] == hp:
            return rec
    return records[0]


def convert(seed: int, feat: FeatureExtractor):
    d0 = DUMP / f"az_seed{seed:05d}_p0.jsonl"
    d1 = DUMP / f"az_seed{seed:05d}_p1.jsonl"
    if not (d0.exists() and d1.exists()):
        return 0, 0
    by_round = {0: load_by_round(d0), 1: load_by_round(d1)}
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    boards, stats_l, labels, cms, pms = [], [], [], [], []
    skipped = 0
    for _ in range(600):
        if st.terminal:
            break
        recs = {}
        for side in (0, 1):
            rec = pick(by_round[side].get(int(st.round_index), []), st)
            recs[side] = rec
            if rec is None:
                skipped += 1
                continue
            ops = chosen_ops(rec)
            if not ops:
                continue
            t, a0, a1 = ops[0]
            # 2026-10-24：降级动作的 arg0 是**塔 id**（不是坐标）⇒ 用当前局面的塔表映射成格子
            if t == int(OperationType.DOWNGRADE_TOWER):
                _tw = {int(x.tower_id): (int(x.x), int(x.y)) for x in st.towers}
                if int(a0) not in _tw:
                    continue
                a0, a1 = _tw[int(a0)]
            try:
                cm = make_class_mask(st, side, intent_decoding=False)
                pm = make_position_masks(st, side, intent_decoding=False)
                if t == int(OperationType.BUILD_TOWER):
                    pos = pm[0:16].any(axis=0)
                elif t == int(OperationType.DOWNGRADE_TOWER):
                    pos = pm[16]
                elif t == int(OperationType.USE_LIGHTNING_STORM):
                    pos = pm[17]
                else:
                    pos = np.zeros_like(pm[0], dtype=bool)
                boards.append(np.asarray(feat.encode_board(st, side), dtype=np.float16))
                stats_l.append(np.asarray(feat.encode_stats(st, side), dtype=np.float16))
                labels.append((t, a0, a1))
                cms.append(np.asarray(cm, dtype=bool))
                pms.append(np.asarray(pos, dtype=bool))
            except Exception:
                pass
        for side in (0, 1):
            rec = recs.get(side)
            if rec is None:
                continue
            for (t, a0, a1) in chosen_ops(rec):
                try:
                    st.apply_operation_list(side, [Operation(OperationType(int(t)), int(a0), int(a1))])
                except Exception:
                    pass
        st.advance_round()
    if not labels:
        return 0, skipped
    np.savez_compressed(
        OUT / f"seed{seed}.npz",
        board=np.stack(boards), stats=np.stack(stats_l),
        label=np.asarray(labels, dtype=np.int16),
        cmask=np.stack(cms), pmask=np.stack(pms))
    return len(labels), skipped


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    feat = FeatureExtractor(max_actions=96)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    seeds = sorted({int(p.stem.split("_")[1].replace("seed", ""))
                    for p in DUMP.glob("az_seed*_p0.jsonl")})
    total = games = skipped_all = 0
    for seed in seeds[:limit]:
        n, sk = convert(seed, feat)
        if n:
            games += 1
            total += n
            skipped_all += sk
            print(f"  ✓ seed={seed}: {n} 样本（跳过 {sk}）累计 {total}", flush=True)
        else:
            print(f"  ✗ seed={seed}: 无样本", flush=True)
    print(f"\n  完成：{games} 局 / {total} 样本 / 跳过 {skipped_all}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
