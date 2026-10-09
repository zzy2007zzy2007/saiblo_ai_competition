"""**冻结基线** launcher（本 goal 的对手）：`grafted_sdk_A.pt` + 上一 goal 跑出 0.8016/0.8047 的那套配置。

用途：本 goal 要"打当前基线 ≥80%"。两侧都是我们自己的 bridge、而 ladder 给两个子进程发**同一套 env**
⇒ 基线侧必须在**本进程内覆盖** `AZAI_*`（子进程隔离，不影响候选侧）。

⚠️ 冻结要求（写进 `goal-2600a56a`）：本文件的配置与 `checkpoints/grafted_sdk_A.pt`
（sha256 8c21f44d73bcefb3fe7a31233934fde4a90fcb3415ee8d6f7780259fb4ac36ba）**不得改动**；
任何"把基线换弱"的做法算作弊。若确需改动，必须走机械门并重写预注册。
⚠️ **不覆盖** `AZAI_BAN_CLASSES` / `AZAI_BAN_OPS`（子博弈要双方都禁闪电，由父进程 env 传下来）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
TM = REPO / "code" / "test_match"
for _p in (REPO, TM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BASELINE_CKPT = REPO / "checkpoints" / "grafted_sdk_A.pt"
os.environ["AZAI_CKPT"] = str(BASELINE_CKPT)          # 冻结基线 ckpt
os.environ["AZAI_MODE"] = "pos-only"
os.environ["AZAI_POSPIN"] = "mc_quiet"
os.environ["AZAI_MC_EVERY"] = "5"
os.environ["AZAI_MC_ROOTONLY"] = "1"
os.environ["AZAI_MC_MARGIN"] = "0"
os.environ["AZAI_MC_UPG"] = "1"
os.environ["AZAI_MC_TOPK"] = "3"
os.environ["AZAI_MC_CELLS"] = "0"
os.environ["AZAI_MC_TOP2"] = "0"
os.environ["AZAI_MC_M4"] = "0"
os.environ["AZAI_MC_NODOWN"] = "0"
os.environ["AZAI_MC_NO_LIGHT"] = "0"
os.environ["AZAI_MC_SELFSPEND"] = "0"
os.environ["AZAI_MC_SELFSPEND_MIX"] = "1.0"
os.environ["AZAI_MC_BIDIR"] = "0"
os.environ["AZAI_MC_SCORE"] = ""
os.environ["AZAI_MC_HORIZON"] = "256"
os.environ["AZAI_ITERS"] = "256"
os.environ["AZAI_DEPTH"] = "4"
os.environ["AZAI_K"] = "24"
os.environ["AZAI_SAMPLE_MULT"] = "15"
os.environ["AZAI_SKIP1"] = "1"
os.environ["AZAI_TEMP"] = "1e-6"
os.environ["AZAI_TCLASS"] = "0.5"
os.environ["AZAI_TPOS"] = "1.0"
os.environ["AZAI_POSPRIOR"] = "off"
os.environ["AZAI_VERIFY"] = "1"

import az_bridge_ai  # noqa: E402

if __name__ == "__main__":
    print(f"[base_pkg] 冻结基线 ckpt={BASELINE_CKPT}", file=sys.stderr, flush=True)
    raise SystemExit(az_bridge_ai.main())
