"""一次性故障注入（**只用于测试**，不是产品代码）。

背景：`ab_search_modes.py` 把 AI 路径写死成 `code/test_match/az_bridge_ai.py`，所以"用一个
故意 sleep 的假 AI"没法直接换文件 —— **而换掉 az_bridge_ai.py 会污染正在跑的另一批对局**
（被驱动起来的 match_via_sdk 会重新加载它）。所以改成在**AI 进程内部**注入延迟：

    --flat-env="...;PYTHONPATH=<repo>/_tmp_faultinject;FAULTINJECT_CLAIM_DIR=<dir>;FAULTINJECT_SLEEP_S=120"

`match_via_sdk.start_ai` 会给 Python AI 设 `PYTHONPATH`（它自己会前置 Ant-Game，本目录排在后面），
于是 AI 进程启动时会 `import sitecustomize` ⇒ 这里执行。

"只坏第一局"靠 `os.makedirs(claim, exist_ok=False)` 的原子性：第一个进程抢到标记后 sleep，
后面的进程抢不到就正常跑。这样一局被看门狗杀掉、另一局不受影响，正好验证"只 kill 该局 PID 树"。
"""
import os
import time

_claim_dir = os.environ.get("FAULTINJECT_CLAIM_DIR")
if _claim_dir:
    try:
        os.makedirs(os.path.join(_claim_dir, "claimed"))
        _won = True
    except FileExistsError:
        _won = False
    except Exception:
        _won = False
    if _won:
        time.sleep(float(os.environ.get("FAULTINJECT_SLEEP_S", "120")))
