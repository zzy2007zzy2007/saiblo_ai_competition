"""零影响：`AZAI_CAND_DECODE` 不设（=legacy）时，**unified 候选生成与改 split 之前逐位相同**（2026-10-31）。

改 split 之前的那份 `unified_cand.py` 是 commit `3b306b5`（玻尔兹曼 + 完整回写）。
本探针把那份从 git 里取出来、`importlib` 单独加载成一个模块，在**同一进程**里对**同一批真实局面**
分别跑两个 `UnifiedCandidateGen`（参数逐字相同），比对：
  * 候选**顺序 + 内容**（`repr` 硬比）
  * 每条的阶段 1 分数（float64 原始字节 sha256）
  * `n_pool / 缓存命中 / 步数 / 形状 / 穷尽 / 重复类形状 / 同类型多op / 长度分布`
  * 决策内状态（`Q` 全部条目、`N`、`cursor` 的 t 值）

> ⚠️ 为什么必须"同一进程 + 同一批局面"逐个比：跨进程/跨局面的 `n_cand` 会随**局面**变
> （金币、塔数），拿不同局的数比会把"局面差异"误读成"代码差异"（本轮踩过：
> 拿 mellowmax 时代的日志去比玻尔兹曼后的代码，33 vs 34 的差其实是 Q 公式改的，不是 split）。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_decode_zero_impact.py --rounds 20 60 98 200 300
```
退出码 0 = 逐位相同。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_PRE_COMMIT = "3b306b5"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base_env() -> dict:
    txt = (REPO / "code" / "test_match" / "_inject_vs_runnerup.sh").read_text(encoding="utf-8")
    m = re.search(r'^BASE="(.*)"$', txt, re.M)
    out = {}
    for kv in m.group(1).split(";"):
        if kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def _digest(x) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="UNIFIED_CAND_FULL")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--rounds", type=int, nargs="+", default=[20, 60, 98, 200, 300])
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    # 取出 split 之前的那份源码
    src = subprocess.run(["git", "show", f"{_PRE_COMMIT}:code/my_ai/az_intent/unified_cand.py"],
                         cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")
    if src.returncode != 0 or "UnifiedCandidateGen" not in src.stdout:
        emit(f"!! 取不到 {_PRE_COMMIT} 的 unified_cand.py：{src.stderr[:300]}")
        return 3
    tmp = Path(tempfile.gettempdir()) / "_unified_cand_pre_split.py"
    tmp.write_text(src.stdout, encoding="utf-8")
    pre = _load("_unified_pre_split", tmp)
    import my_ai.az_intent.unified_cand as post
    emit(f"[dec0] pre  = {_PRE_COMMIT}:code/my_ai/az_intent/unified_cand.py  → {tmp}")
    emit(f"[dec0] post = 工作区当前版本（含 AZAI_CAND_DECODE 分支）")

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"
    os.environ["AZAI_CAND_MODE"] = "unified"
    os.environ["AZAI_CAND_DECODE"] = "legacy"      # ← 显式 legacy
    os.environ["AZAI_K"] = str(int(args.k))

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    rsc = _load("_rsc_dec0", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    want = set(int(x) for x in args.rounds)
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    snaps = []
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, st.clone()))
        for pl in (0, 1):
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
        st.advance_round()

    bridge = _load("az_bridge_ai_dec0", REPO / "code" / "test_match" / "az_bridge_ai.py")
    searcher, _m, _f = bridge.build_engine(args.seed, 0)
    me = int(args.player)

    def _mk(mod_cls):
        g = mod_cls(k=int(args.k), beta=3.0, ucb_c=1.0, pool_mult=1.5, attempts_mult=6,
                    force_hold=True, tau=1.0, steps_per_shape=12, steps_total=0, seed=0)
        g.rollout_fn = searcher.idle_rollout_values
        return g

    g_pre = _mk(pre.UnifiedCandidateGen)
    g_post = _mk(post.UnifiedCandidateGen)
    emit(f"[dec0] pre.decode_mode={getattr(g_pre, 'decode_mode', '<无此属性=改前版本>')}  "
         f"post.decode_mode={g_post.decode_mode}")

    n_bad = 0
    for (r, snap) in snaps:
        net_out = searcher.net_fn(snap.clone(), me)
        a = g_pre.generate(snap.clone(), me, net_out, k=int(args.k))
        b = g_post.generate(snap.clone(), me, net_out, k=int(args.k))
        sa, sb = a["stats"], b["stats"]
        same = (
            repr(a["bundles"]) == repr(b["bundles"])
            and _digest(sa["scores"]) == _digest(sb["scores"])
            and sa["n_pool"] == sb["n_pool"] and sa["n_dup"] == sb["n_dup"]
            and sa["steps"] == sb["steps"] and sa["shapes"] == sb["shapes"]
            and sa["dead_shapes"] == sb["dead_shapes"] and sa["n_rep"] == sb["n_rep"]
            and sa["n_multi_same"] == sb["n_multi_same"] and sa["sizes"] == sb["sizes"]
            and sa["q_entries"] == sb["q_entries"]
            and repr(g_pre.Q) == repr(g_post.Q)
            and repr(g_pre.N) == repr(g_post.N)
            and repr({k2: v["t"] for k2, v in g_pre.cursor.items()})
                == repr({k2: v["t"] for k2, v in g_post.cursor.items()})
        )
        n_bad += int(not same)
        emit(f"[dec0] r{r:<4} | 候选={sa['n_cand']:<3} 池={sa['n_pool']:<3} 命中={sa['n_dup']:<5} "
             f"步数={sa['steps']:<5} 形状={sa['shapes']:<4} 穷尽={sa['dead_shapes']:<3} "
             f"Q条目={sa['q_entries']:<5} sizes={sa['sizes']} "
             f"scores_sha={_digest(sa['scores'])} "
             f"| {'✓ 逐位相同' if same else '!! 不一致'}")
        if not same:
            emit(f"    !! post: 候选={sb['n_cand']} 池={sb['n_pool']} 命中={sb['n_dup']} "
                 f"步数={sb['steps']} 形状={sb['shapes']} Q条目={sb['q_entries']} "
                 f"sizes={sb['sizes']} scores_sha={_digest(sb['scores'])}")

    emit("")
    emit(f"[dec0] 小结：{len(snaps)} 个真实局面（pre={_PRE_COMMIT} vs 当前工作区 legacy）"
         f"不一致 = {n_bad}（必须 0）")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[dec0] 已写入 {p}")
    return 0 if n_bad == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
