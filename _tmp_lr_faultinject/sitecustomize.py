"""故障注入（**只用于验证** `_tmp_ladder.py --retry-invalid / --final-sweep`）：让前 K 个
**对局进程**（`match_via_sdk.py`）在启动瞬间 `os._exit(RC)`。

⚠️ 本目录与 `_tmp_faultinject/` **故意分开**：那个是 2026-10-28 的"AI 进程内 sleep"注入
（`FAULTINJECT_CLAIM_DIR`），被 `docs/match_retry_resume_20261028.md` 引用了，不能改语义。

为什么用 `sitecustomize` 而不是"换掉 AI 文件"：
  · 换 `az_bridge_ai.py` 会**污染正在跑的另一批真实对局**（它们共用同一个 bridge）；
  · 只换 `--ai0/--ai1` 又没法让"第 K+1 次"变成一局真对局（重试就永远救不回来）。

本文件只影响 `sys.argv` 里含 `match_via_sdk.py` 的进程（bridge AI 进程、ladder 驱动进程都不碰），
并用 `os.open(..., O_CREAT|O_EXCL)` 抢占编号文件（原子）做"前 K 次"的全局计数：
只有前 K 个对局进程会立刻退出，第 K+1 个开始一切正常 ⇒ 重试 / final-sweep 能把局救回来。

环境变量：
  LR_FAULT_DIR  抢占文件目录（必须已存在）
  LR_FAULT_K    让前 K 个对局进程立刻退出（默认 0 = 不注入）
  LR_FAULT_RC   退出码（默认 3）

用法（把本文件所在目录放进 PYTHONPATH，`sitecustomize` 才 import 得到）：
    PYTHONPATH="$PWD/_tmp_lr_faultinject" LR_FAULT_K=1 LR_FAULT_DIR=$PWD/_tmp_lr_claims \\
        python _tmp_ladder.py ...
"""
from __future__ import annotations

import os
import sys


def _maybe_kill() -> None:
    try:
        k = int(os.environ.get("LR_FAULT_K", "0"))
    except ValueError:
        return
    if k <= 0:
        return
    if not any("match_via_sdk.py" in a for a in sys.argv):
        return
    d = os.environ.get("LR_FAULT_DIR") or ""
    if not d or not os.path.isdir(d):
        return
    try:
        rc = int(os.environ.get("LR_FAULT_RC", "3"))
    except ValueError:
        rc = 3
    for i in range(1, k + 1):
        try:
            fd = os.open(os.path.join(d, "claim.%d" % i),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        except OSError:
            return
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        sys.stderr.write("[faultinject] 第 %d/%d 次对局进程被注入 ⇒ os._exit(%d)\n" % (i, k, rc))
        sys.stderr.flush()
        os._exit(rc)


_maybe_kill()
