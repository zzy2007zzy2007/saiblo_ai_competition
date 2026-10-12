"""**旧最好配置** launcher（S0：`posnet_A_k5_m32.pt` + pos-only + mc_quiet + 宽菜单）。

用途：给"**非自指读数**"当对手——让新的（蒸馏嫁接）agent 与**我们此前最好的子博弈 agent** 正面交手。
两侧都是我们自己的 bridge（`az_bridge_ai.py`），而 ladder 给两个子进程发的是**同一套 env** ⇒
所以旧配置必须在**本进程内覆盖** `AZAI_*`（子进程隔离，不影响对侧）。

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

# ── 旧最好（S0）配置：只覆盖需要不同的项，其余（含禁闪电开关）继承父进程 ──
os.environ["AZAI_CKPT"] = str(REPO / "checkpoints" / "posnet_A_k5_m32.pt")
os.environ["AZAI_MODE"] = "pos-only"
os.environ["AZAI_POSPIN"] = "mc_quiet"
os.environ["AZAI_MC_EVERY"] = "5"
os.environ["AZAI_MC_ROOTONLY"] = "1"
os.environ["AZAI_MC_MARGIN"] = "0"
os.environ["AZAI_MC_UPG"] = "1"
os.environ["AZAI_MC_TOPK"] = "3"
os.environ["AZAI_MC_CELLS"] = "0"
os.environ["AZAI_MC_NO_LIGHT"] = "0"
os.environ["AZAI_MC_SELFSPEND"] = "0"
os.environ["AZAI_MC_BIDIR"] = "0"
os.environ["AZAI_MC_SCORE"] = ""
os.environ["AZAI_SKIP1"] = "1"
os.environ["AZAI_TEMP"] = "1e-6"
os.environ["AZAI_TCLASS"] = "0.5"
os.environ["AZAI_TPOS"] = "1.0"

import az_bridge_ai  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(az_bridge_ai.main())
