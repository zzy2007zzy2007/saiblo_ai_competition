#!/usr/bin/env python3
"""把 bridge 的 `state_dump.json`（含 ops_history）转成亚军计划检查器要的 `replay.json`。

## 目标格式（`load_replay_round_start`，sdk_lure_inspector.cpp:1205 要求）
```json
[ {"seed": 1234, "round": 0, "op0": [[t,a0,a1],...], "op1": [...]},
  {"seed": 1234, "round": 1, "op0": [...],       "op1": [...]},
  ... ]
```
检查器**从第 0 回合开始重放 op0/op1** 推进 `NativeSimulator` 到目标回合，
所以核心是每回合的操作序列；`round_state` 只在部分模式里用来校验。

## 输入
`state_dump.json`（由 `match_via_sdk.py` 在 `AZAI_DUMP_ROUND` 处写出）里：
  `ops_history[i]` = `[op0_tokens..., None, op1_tokens...]`
  其中每个 token 形如 `[op_type, arg0, arg1]`（`Operation.to_protocol_tokens()`）

## 用法
```
python code/my_ai/distill_top2/state_dump_to_replay.py \
    --dump state_dump.json --out out/replay_seed1234.json [--seed 1234] [--require-state]
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _op_obj(t, a0, a1=None):
    """把 `Operation.to_protocol_tokens()` 的 [type, arg0, (arg1)] 转成
    亚军 `parse_replay_operation`（sdk_lure_inspector.cpp:1019）要的**对象**格式：

        {"type": t, "id": a0, "args": a1, "pos": {"x": a0, "y": a1}}

    ⚠️ `to_protocol_tokens()` 的元素个数**不固定**（如降级只有 [type, id]），
    所以 a1 可省略。该函数按类型取不同字段：
      BuildTower / UseLightningStorm / UseEmpBlaster / UseDeflector /
      UseEmergencyEvasion                      -> 用 pos.x / pos.y = (a0, a1)
      UpgradeTower                             -> 用 id=a0 / args=a1
      DowngradeTower                           -> 用 id=a0
    所以**同时带上 id/args/pos** 对所有类型都安全（各类型只读自己那份）。
    """
    v = int(a0)
    w = int(a1) if a1 is not None else v
    return {"type": int(t), "id": v, "args": w, "pos": {"x": v, "y": w}}


def _split_history(entry):
    """ops_history 的一条 = [op0..., None, op1...] -> (op0, op1)（已转成对象格式）。"""
    if not isinstance(entry, (list, tuple)):
        raise ValueError(f"ops_history 条目不是 list: {type(entry)}")
    if None not in entry:
        return [_op_obj(*t) for t in entry], []
    sep = entry.index(None)
    op0 = [_op_obj(*t) for t in entry[:sep]]
    op1 = [_op_obj(*t) for t in entry[sep + 1:]]
    return op0, op1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="state_dump.json")
    ap.add_argument("--out", required=True, help="输出的 replay.json")
    ap.add_argument("--seed", type=int, default=0, help="0=从 dump 里读")
    ap.add_argument("--require-state", action="store_true",
                    help="要求每条都带 round_state（默认不带也允许）")
    a = ap.parse_args()

    d = json.loads(Path(a.dump).read_text(encoding="utf-8"))
    seed = a.seed or int(d.get("seed", 0) or 0)
    hist = d.get("ops_history") or []
    if not hist:
        print("[x] dump 里没有 ops_history —— 采集时是否设了 AZAI_DUMP_ROUND？")
        return 1

    state0 = d.get("state") or d.get("round_state") or d.get("snapshot")
    if state0 is None:
        # `_cpm.dump_state()` 直接把状态字段平铺在顶层（towers/ants/bases/coins/...）
        # ⇒ 整个 dump 去掉元字段后就是第 0 回合的 round_state
        state0 = {k: v for k, v in d.items()
                  if k not in ("seed", "snapshot_round", "ops_history")}
    out = []
    for i, entry in enumerate(hist):
        op0, op1 = _split_history(entry)
        rec = {"seed": seed, "round": i, "op0": op0, "op1": op1}
        if state0 is not None and i == 0:
            rec["round_state"] = state0      # 只有第 0 回合有快照；后续靠重放 ops 推进
        elif a.require_state:
            print(f"[warn] 第 {i} 回合缺 round_state（只有第 0 回合有快照）")
            return 1
        out.append(rec)

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n0 = sum(len(r["op0"]) for r in out)
    n1 = sum(len(r["op1"]) for r in out)
    print(f"[ok] {len(out)} 回合, seed={seed}, 操作总数 p0={n0} p1={n1} -> {p}")
    print("     下一步: sdk_lure_inspector 的 plan_rollouts 模式读这个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
