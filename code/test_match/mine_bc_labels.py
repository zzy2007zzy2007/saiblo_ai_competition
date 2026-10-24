"""从现有对局日志挖"搜索选中动作"的标签（行为克隆数据），2026-10-24。

背景：昨晚用 rollout 分数当教师失败（Spearman +0.428 而棋力 50%）。
用户建议改为**以搜索选中的动作为学习目标**（信息密度低但免采集、且直接优化 recall@1）。

数据来源：冠军配置（k=160 / 采样1 / h16 / 值网仲裁 / 不升级）的对局日志，逐回合含完整操作：
    [round 0] p0(az_bridge_ai) 原文='2\n11 4 15\n11 4 2' -> [Operation(...)]
本脚本：解析 → 用 GameStateFacade 重放 → 在每个决策记录
    (board 28x19x19 fp16, stats, 选中的 op, 合法类掩码, 该类的合法格掩码)

输出：training_history/bc_labels/<arm>_<game>.npz
"""
from __future__ import annotations

import re
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

# arm -> 哪一侧是冠军配置（A 或 B）
ARMS = {
    "ab_wide_k160_vs_old": "A",
    "ab_k320_vs_k160": "B",
    "ab_mcts16_k160": "B",
    "ab_enum300_vs_160": "B",
}
LINE_RE = re.compile(r"\[round (\d+)\] p(0|1)\(([^)]*)\) 原文='(.*?)' ->")


def parse_log(path: Path):
    """返回 (seed, {round: {side: [(t,a0,a1), ...]}})"""
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"RESULT seed=(\d+)", txt)
    seed = int(m.group(1)) if m else None
    hp_m = re.search(r"base_hp=(\d+),(\d+)", txt)
    hp_logged = (int(hp_m.group(1)), int(hp_m.group(2))) if hp_m else None
    rounds: dict[int, dict[int, list]] = {}
    for ln in txt.splitlines():
        mm = LINE_RE.search(ln)
        if not mm:
            continue
        r, side, _label, raw = int(mm.group(1)), int(mm.group(2)), mm.group(3), mm.group(4)
        raw = raw.replace("\\n", "\n")
        parts = [p.strip() for p in raw.splitlines() if p.strip()]
        ops = []
        if parts:
            try:
                n = int(parts[0])
            except ValueError:
                n = 0
            for p in parts[1:1 + n]:
                f = p.split()
                if len(f) >= 3:
                    try:
                        ops.append((int(f[0]), int(f[1]), int(f[2])))
                    except ValueError:
                        pass
        rounds.setdefault(r, {})[side] = ops
    return seed, rounds, hp_logged


def mine_one(arm: str, log_path: Path, feat: FeatureExtractor, out_dir: Path) -> int:
    seed, rounds, hp_logged = parse_log(log_path)
    if seed is None or not rounds:
        print(f"  ✗ {log_path.name}: 解析失败（无 RESULT 或无操作行）", flush=True)
        return 0
    seat_A = 0 if log_path.stem.endswith("_p0") else 1
    champ_side = seat_A if ARMS.get(arm) == "A" else 1 - seat_A

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    boards, stats_l, labels, cmasks, pcmasks = [], [], [], [], []
    for r in sorted(rounds):
        by_side = rounds[r]
        for side in (0, 1):
            ops = by_side.get(side, [])
            if side == champ_side and ops:
                try:
                    board = np.asarray(feat.encode_board(st, side), dtype=np.float16)
                    stats_v = np.asarray(feat.encode_stats(st, side), dtype=np.float16)
                    cm = np.asarray(make_class_mask(st, side, intent_decoding=False), dtype=bool)
                    pm = np.asarray(make_position_masks(st, side, intent_decoding=False))
                    t, a0, a1 = ops[0]
                    # 位置掩码：优先用该 op 类型对应的通道；BUILD 用 tower 通道的并集
                    if t == int(OperationType.BUILD_TOWER):
                        pos = pm[0:16].any(axis=0)
                    elif t == int(OperationType.DOWNGRADE_TOWER):
                        pos = pm[16]
                    elif t == int(OperationType.USE_LIGHTNING_STORM):
                        pos = pm[17]
                    else:
                        pos = np.zeros_like(pm[0], dtype=bool)
                    boards.append(board); stats_l.append(stats_v)
                    labels.append((t, a0, a1))
                    cmasks.append(cm); pcmasks.append(np.asarray(pos, dtype=bool))
                except Exception as exc:
                    print(f"  ⚠️ r{r} side{side} 记录失败 {type(exc).__name__}: {exc}", flush=True)
            for (t, a0, a1) in ops:
                try:
                    st.apply_operation_list(side, [Operation(OperationType(int(t)), int(a0), int(a1))])
                except Exception:
                    pass
        st.advance_round()
        if st.terminal:
            break

    hp_got = tuple(int(b.hp) for b in st.bases)
    ok = (hp_logged is None) or (hp_got == hp_logged)
    if not ok:
        print(f"  ✗ {log_path.name}: 重放不一致 got={hp_got} logged={hp_logged}", flush=True)
        return 0
    if not labels:
        return 0
    out = out_dir / f"{arm}_{log_path.stem}.npz"
    np.savez_compressed(
        out,
        board=np.stack(boards).astype(np.float16),
        stats=np.stack(stats_l).astype(np.float16),
        label=np.asarray(labels, dtype=np.int16),
        cmask=np.stack(cmasks),
        pmask=np.stack(pcmasks),
    )
    return len(labels)


def main() -> int:
    out_dir = REPO / "training_history/bc_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    feat = FeatureExtractor(max_actions=96)
    total_games = total_samples = 0
    for arm in ARMS:
        d = REPO / "training_history" / arm
        if not d.exists():
            print(f"  ⚠️ 缺少 {arm}")
            continue
        logs = sorted(d.glob("seed*_p*.log"))
        for lp in logs:
            if only and only not in lp.name:
                continue
            n = mine_one(arm, lp, feat, out_dir)
            if n:
                total_games += 1
                total_samples += n
            print(f"  [{arm}] {lp.name}: {n} 样本（累计 {total_samples}）", flush=True)
    print(f"\n  完成：{total_games} 局，{total_samples} 个决策样本 → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
