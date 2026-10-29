"""假 AI（**看门狗验证专用**）：什么都不实现，直接 `sleep 600`。

用途：验证 `_tmp_ladder.py --watchdog-stall-min`（用户验证 4）。
- 它像正常 AI 一样被 `match_via_sdk.start_ai` 起来（`.py` ⇒ 用同一个 python 解释器），
  并从 stdin 收到 `"player seed"`；
- 之后既不回包也不退出 ⇒ 该局卡在 round 0，progress 文件与日志 mtime 都不再前进；
- 看门狗（阈值 60s）按 ladder 自己 Popen 的**那一个 PID** 精确 `taskkill /T /F` 掉这一局的进程树。

`LR_HANG_PIDFILE`（可选）：把自己的 PID 追写进去 ⇒ 万一看门狗没杀掉，验证脚本可以按**这些确切
PID** 补刀（绝不按镜像名/命令行通配符批量杀）。
"""
import os
import sys
import time

_pidfile = os.environ.get("LR_HANG_PIDFILE")
if _pidfile:
    try:
        with open(_pidfile, "a", encoding="utf-8") as fh:
            fh.write("%d\n" % os.getpid())
    except OSError:
        pass
sys.stderr.write("[hangai] 假 AI 启动 pid=%d：故意挂住 600s（看门狗验证用）\n" % os.getpid())
sys.stderr.flush()
time.sleep(600)
