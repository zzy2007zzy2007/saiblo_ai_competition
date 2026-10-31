# 对局进度日志系统（`training_history/progress/`）现状调查 + 修复方案

> 调查日期：2026-10-31
> 范围：只调查 + 出方案，**未改任何代码**。
> 结论速览：真正写 `training_history/progress/` 的只有 **1 个库 + 4 个驱动**；`az_bridge_ai.py` **完全不写** progress 日志。

---

## §1 现状全景：谁在写、写什么、目录名怎么来

### 1.0 一句话版

| 角色 | 文件 | 写什么 |
|---|---|---|
| **进度库**（被 import） | `code/test_match/progress.py` | `az_progress_seed*.txt`（逐回合）+ `status.txt`（整批） |
| **对局驱动**（真正落盘 trace） | `code/test_match/match_via_sdk.py` | 逐回合 trace + 终局行（调 `progress.GameTrace`） |
| **ladder 驱动**（写 status） | `_tmp_ladder.py` | `status.txt`（含 **bug** 的"最近完成"） |
| **ladder 驱动**（旧版） | `_tmp_ladder_head.py` | `status.txt`（同 bug） |
| **AB 搜索驱动** | `code/test_match/ab_search_modes.py` | `status.txt` + 自己的 `_progress.json` |
| **AB 搜索驱动**（旧版） | `code/test_match/_tmp_ab_old.py` | `status.txt` |

**关键澄清（用户疑点）**：`az_bridge_ai.py` **一个字节的 progress 日志都不写**（全仓库 grep `progress|training_history|status.txt|az_progress` 在该文件 0 匹配）。它是被 `match_via_sdk.py` 拉起来的 **AI 进程**，只往 `match_results/ai?_az_bridge_ai_seed*.stderr.log` 写自己的 stderr；progress trace 是**裁判驱动 `match_via_sdk.py`** 写的。用户看到"桥接程序似乎会写"是因为 `az_bridge_ai.py` 每局都出现在对局里、且 stderr 文件名带 `az_bridge_ai`，容易误认。

另外两个程序写 **progress 文件**但**不在 `training_history/progress/` 下**（命名相似、易混淆，单独列出）：

| 程序 | 写什么 | 位置 |
|---|---|---|
| `code/my_ai/az_intent/az_selfplay.py:798` | `az_progress_seed{seed:05d}.txt`（5 位补零，格式不同） | 自对弈的 `--out-dir`（训练输出目录） |
| `code/my_ai/az_intent/collect_value_prior.py:95` | `vp_progress_seed%05d.txt` | `--progress-dir`（默认 out 的父目录） |

---

### 1.1 `code/test_match/progress.py` —— 进度库（唯一落盘逻辑）

| 项 | 详情 |
|---|---|
| **目录名规则** | `_ensure_dir()` L44-61：`name = f"{tag}_{stamp}" if stamp else str(tag)`；`stamp = os.environ.get("DSH_PROGRESS_DIR_STAMP", "").strip()`（L55-56） |
| **逐回合文件** | `GameTrace.__init__` L104：`self.path = _ensure_dir(tag) / f"az_progress_seed{seed}{suffix}.txt"` |
| **逐回合字段** | L121-123：`round=N HP0= HP1= G0= G1= P0=[BUILD(4,16)] P1=[...]`（P0 段补空格到 36 列） |
| **终局行** | `terminal()` L137：`terminal winner= hp=[x,y] rounds=N` |
| **status.txt** | `write_status()` L152-176：`tag= / total= / done= / W= / L= / 无效=` + `now= / elapsed= / eta=` + `inflight: N 局（回合…）` + 若干行 `最近完成: …`；原子写 `.tmp` + `os.replace`（L172-174） |
| **tag 默认** | 无默认（由调用方传入） |

### 1.2 `code/test_match/match_via_sdk.py` —— 对局驱动（trace 实际写入者）

| 项 | 详情 |
|---|---|
| **tag 来源** | L669：`_os.environ.get("DSH_PROGRESS_TAG", "match")` —— **默认 `"match"`** |
| **suffix 来源** | L671：`_os.environ.get("DSH_PROGRESS_SUFFIX", "")`（配对 `_p0`/`_p1` 防覆盖） |
| **seed 来源** | L670：`globals().get("_SEED_TOKEN") or seed`（token 如 `7r` 原样进文件名） |
| **逐回合写** | L766-776：`_TRACE.round(...)` |
| **终局写** | L849-855：`_TRACE.terminal(...)` |
| **stderr 命名** | L681-683：`DSH_PROGRESS_SUFFIX` + `MVS_STDERR_SUFFIX=1` 时拼进 `ai?_<label>_seed<N><suf>.stderr.log` |

它本身不写 `status.txt`；`status.txt` 由上层驱动调 `progress.write_status()` 写。

### 1.3 `_tmp_ladder.py` —— ladder 驱动（写 status.txt，**有 bug**）

| 项 | 详情 |
|---|---|
| **tag 来源** | L746：`_TAG = str(cfg.get("tag", "ladder"))`（`--tag=`，默认 `ladder`） |
| **传给对局** | L501：`env["DSH_PROGRESS_TAG"] = str(_TAG)` |
| **写 status.txt** | L566-613：每局完成时重扫 `match_results/ladder_logs/<tag>/*.log` 聚合 W/L + 调 `_pg.write_status(...)` |
| **"最近完成"** | L576-599（**bug 见 §2.2**） |
| **inflight** | L602：`(REPO / "training_history/progress" / _tag).glob("az_progress_seed*.txt")` 读末行 `round=` |
| **看门狗读路径** | L180-201 `_progress_candidates`：`PROGRESS_ROOT / str(tag)`（**不带 stamp**，见 §3 风险） |

### 1.4 `_tmp_ladder_head.py` —— ladder 驱动（旧版，**同 bug**）

| 项 | 详情 |
|---|---|
| **tag 来源** | L240：`_TAG = str(cfg.get("tag", "ladder"))` |
| **传给对局** | L111：`env["DSH_PROGRESS_TAG"] = str(_TAG)` |
| **写 status.txt** | L133-180（`_recent` bug 同 `_tmp_ladder.py`） |
| **inflight** | L169：同 L602 的 glob |

### 1.5 `code/test_match/ab_search_modes.py` —— AB 搜索驱动（**"最近完成"正确**）

| 项 | 详情 |
|---|---|
| **tag 来源** | L235：`TAG = out_dir.name`（`--out-dir` 的最后一段） |
| **传给对局** | L236：`env_base["DSH_PROGRESS_TAG"] = TAG` |
| **写 status.txt** | L342-367：`write_status(...)` |
| **"最近完成"** | L349-353：`_recent = [... for k, v in list(results.items())[-6:]]` —— 取 dict **插入序的最后 6 个**（= 完成序最后 6 个 = 最近完成） |
| **额外** | L106-160 `ProgressFile` 写 `<out-dir>/_progress.json`（**不在** training_history/progress 下） |

### 1.6 `code/test_match/_tmp_ab_old.py` —— AB 搜索驱动（旧版，**"最近完成"正确**）

| 项 | 详情 |
|---|---|
| **tag 来源** | L69：`TAG = out_dir.name` |
| **传给对局** | L70：`env_base["DSH_PROGRESS_TAG"] = TAG` |
| **写 status.txt** | L142-166 |
| **"最近完成"** | L148-152：`list(results.items())[-6:]`（同 1.5，正确） |

### 1.7 时间戳后缀（`DSH_PROGRESS_DIR_STAMP`）的完整链路

- **唯一设置点**：`code/run_logged.sh:61`：`export DSH_PROGRESS_DIR_STAMP="${DSH_PROGRESS_DIR_STAMP:-$TS}"`（`$TS` = 启动时刻 `date +%Y%m%d_%H%M%S`，L36）。
- **唯一消费点**：`progress.py:55-56`（拼进目录名）。
- 全仓库 grep 确认：`DSH_PROGRESS_DIR_STAMP` 只在这两处出现；**`code/run_logged.ps1`（PowerShell 等价版）不设它**。
- 结论：**只有"走 `bash code/run_logged.sh` 包一层"的实验**，目录名才带 `_YYYYMMDD_HHMMSS` 后缀；直接跑 `_tmp_ladder.py` / `ab_search_modes.py` / 各 `.sh` 的，一律**无后缀**。

### 1.8 目录名（tag）怎么来 —— 汇总

| 源程序 | tag 值 | 有无时间戳后缀 |
|---|---|---|
| `match_via_sdk.py`（直连） | `DSH_PROGRESS_TAG` 环境变量，默认 `"match"` | 有（若外层设了 stamp） |
| `_tmp_ladder.py` | `--tag=`，默认 `"ladder"` | 同上 |
| `_tmp_ladder_head.py` | `--tag=`，默认 `"ladder"` | 同上 |
| `ab_search_modes.py` / `_tmp_ab_old.py` | `--out-dir` 的最后一段 | 同上 |
| `_us_vs_external.sh` 等 | 脚本内 `TAG=...` → 传给 `_tmp_ladder.py --tag` | 同上 |
| `_value_ab_vs_rv4.sh` | **不设** `DSH_PROGRESS_TAG` → 落到默认 `"match"` | 同上 |
| `_filter_gate_matrix.sh` | 硬编码 `filter_gate_matrix` | 同上 |

> 实测目录佐证：`training_history/progress/` 下 150+ 个子目录，绝大多数**无后缀**（`NONHOLD_VS_CHAMPION_B`、`ABPRE`、`US_VS_CHAMPION_BASE`…），只有少数带 `_20261030_154735` / `_20261030_112513`（恰好是 2026-10-30 起走 `run_logged.sh` 的那几批）。

---

## §2 两个 bug 的精确定位

### §2.1 Bug ①：目录命名混乱，同名 tag 多批堆一起

- **位置**：`code/test_match/progress.py:55-56`
  ```python
  stamp = os.environ.get("DSH_PROGRESS_DIR_STAMP", "").strip()
  name = f"{tag}_{stamp}" if stamp else str(tag)
  ```
- **根因**：时间戳是 **opt-in** 的——只有外层 `code/run_logged.sh:61` 设了 `DSH_PROGRESS_DIR_STAMP` 才加后缀；直接跑（或走 `run_logged.ps1`）一律 `name = tag`。于是**同一 tag 的多次续跑、失败重启用同名、没设名字用默认名，全部堆进同一个 `<tag>/`**。
- **证据**：
  - `training_history/progress/NONHOLD_VS_CHAMPION_B/` 有 24 个 `az_progress_seed*.txt`，mtime 从 10-29 22:12 跨到 10-30 12:46（跨天两批堆一起）；而 `NONHOLD_VS_CHAMPION_B_20261030_112513/` 是走 `run_logged.sh` 的那批。
  - `ABPRE/`（2 文件）与 `ABPRE_20261030_154735/`（2 文件）并存，同一个 tag 两批被拆成两个目录。
- **为什么"每个目录都加时间戳"做不到**：见 §3.1，核心是"时间戳必须整批共享、且各驱动/消费者要能算出同一个目录名"，不能简单在 `_ensure_dir` 里 `time.strftime` 兜底（否则 match_via_sdk 每次建 GameTrace 和 write_status 会各算出一个不同时刻 → 同一个 tag 反而拆得更碎）。

### §2.2 Bug ②：`status.txt`"最近完成"其实是"最先完成"

- **位置 1**：`_tmp_ladder.py:576-599`（关键在 L577 的顺序遍历 + L597 的截断）
  ```python
  _recent = []                                   # L576
  for _tok in _tok_list:                          # L577：按 CLI token 顺序（seed 升序）
      ...
      if len(_recent) < 6:                        # L597：只保留【前 6 个】已完成的
          _recent.append(f"{_tok} {'胜' if _won else '负'} ...")
  ```
- **位置 2**：`_tmp_ladder_head.py:143-166`（同样的 L144 + L164）。
- **根因**：`_recent` 是**按 token 顺序（`_tok_list`）遍历、取前 6 个已完成局**，而不是按**完成时间**排序取最后 6 个。`_tok_list` 是 CLI 传入的 seed 升序（如 `7 7r 8 8r ...`），所以输出的是"token 号最小的 6 个已打完的局"。
- **证据**：`training_history/progress/NONHOLD_VS_CHAMPION_B/status.txt` 的"最近完成"列的是 `15 / 15r / 16 / 16r / 17 / 17r`；但该批 token 到 18/18r，且 17r 的 mtime 是 12:46:27（最晚）、18r 是 12:38、18 是 12:34——**真正最晚完成的 17r/18r/18 里，18 和 18r 被漏掉**，反而把更早的 15/15r 排进去了。
- **对比（正确实现）**：`ab_search_modes.py:349-353` 与 `_tmp_ab_old.py:148-152` 用 `list(results.items())[-6:]`——`results` 是各线程完成时 `with lock:` 写入的 dict，插入序 = 完成序，所以 `[-6:]` = 最近完成的 6 个。**bug 只在两个 ladder 脚本**。

---

## §3 修复方案

### §3.1 统一加时间戳后缀（Bug ①）

**核心思路**：把"目录名 = `<tag>_<stamp>`"变成**必加**，并让 stamp 由**批次驱动进程自己兜底生成一次**，整批共享。

分三步：

1. **让 stamp 有兜底默认值**：在 4 个批次驱动入口，启动时：
   ```python
   os.environ.setdefault("DSH_PROGRESS_DIR_STAMP", time.strftime("%Y%m%d_%H%M%S"))
   ```
   改哪里：`_tmp_ladder.py`（main 开头，L745 附近）、`_tmp_ladder_head.py`（main，L239 附近）、`ab_search_modes.py`（L235 附近）、`_tmp_ab_old.py`（L69 附近）。
   怎么生效：`run_one` 里 `env = os.environ.copy()`（`_tmp_ladder.py:497`、`ab_search_modes.py` 的 `env_base`）会把 stamp 带进每个 `match_via_sdk.py` 子进程 ⇒ 同批所有局落同一个目录。
   效果：外层 `run_logged.sh` 设了就沿用（整批一致）；没设就由驱动自己生成一次（同样整批一致）。

2. **补 `code/run_logged.ps1`**：在它里面也 `$env:DSH_PROGRESS_DIR_STAMP = ...`（当前没设，PowerShell 版跑的批次无后缀）。

3. **让 `progress.py` 的 `_ensure_dir` 兜底强制加 stamp**（可选，增强）：把 L55-56 改成"stamp 为空时用模块级缓存的当前时间"（**缓存一次**，不是每次调用重算）：
   ```python
   _DIR_STAMP = os.environ.get("DSH_PROGRESS_DIR_STAMP", "").strip() \
                or time.strftime("%Y%m%d_%H%M%S")   # 进程内只算一次
   ```
   这样即便有调用方忘了 setdefault，同一进程内也稳定。

- **影响面**：目录名从 `<tag>` → `<tag>_<stamp>`。以下**按 `<tag>` 精确拼路径的消费者**必须同步改为"用相同规则解析目录"（否则 glob 不到新目录）：
  - `_tmp_ladder.py:602`、`_tmp_ladder_head.py:169`、`ab_search_modes.py:355`、`_tmp_ab_old.py:154`（inflight glob）
  - `_tmp_ladder.py:187`（`_progress_candidates`，看门狗判卡死用）
  - `code/test_match/_ladder_watchdog.ps1:108`（`Get-ChildItem $progDir`，`$progDir` 按 `<Tag>` 拼）
  - 建议：在 `progress.py` 暴露一个 `progress_dir(tag) -> Path`（= `_ensure_dir` 的返回），让这些地方**调同一个函数**而不是各自手拼。
- **向后兼容**：
  - 对**新批次**：自动进 `<tag>_<stamp>`，问题解决。
  - 对**旧目录**（大量无后缀目录）：glob 改成"先查 `<tag>_<stamp>`，找不到再回退 `<tag>`"，或直接 `glob("<tag>_*")` 取最新。看门狗/分析工具（`probe_unified_kgeo_coin.py`、`probe_why_no_upgrade.py`、`audit_replay_batch.py` 用 `rglob` 的不受影响；用精确 `<tag>` glob 的要注意）。

### §3.2 修 status.txt"最近完成"（Bug ②）

- **改哪里**：`_tmp_ladder.py:576-599` 与 `_tmp_ladder_head.py:143-166`（两处同样的 `if len(_recent) < 6:` 截断）。
- **怎么改**：不要"遍历 token 顺序、取前 6"；改为"收集**每局完成时间 + 文本**，按完成时间降序取前 6"。
  - 完成时间可直接用该 token 日志的 **mtime**（`_lg.stat().st_mtime`），或本线程完成时刻。
  - 伪代码：
    ```python
    _recs = []  # (完成时间, 文本)
    for _tok in _tok_list:
        ... # 判有效、拼文本
        _recs.append((_lg.stat().st_mtime, 文本))
    _recent = [t for _, t in sorted(_recs, reverse=True)[:6]]
    ```
  - 另一个等价更小的改法：保留遍历，但**收集全部**后 `_recent = 全量[-6:]`——不过这只在"遍历顺序 == 完成顺序"时才对，而 ladder 是按 token 顺序遍历，所以**必须按 mtime 排序**（不能用 `[-6:]`）。
- **影响面**：只影响 `status.txt` 的"最近完成"几行文本，不动 `total/done/W/L/invalid` 的聚合逻辑，也不影响对局。`ab_search_modes.py` / `_tmp_ab_old.py` 的 `[-6:]` 是对的，不用改。
- **向后兼容**：完全兼容（纯输出内容修正）。

### §3.3 顺手发现的混乱点（建议一并处理，可单独拆任务）

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| 1 | `_value_ab_vs_rv4.sh` **不设 `DSH_PROGRESS_TAG`** ⇒ 74 个 trace 全落默认 `match/`，与其他默认 tag 的局混在一起 | `_value_ab_vs_rv4.sh`（`run_one` 里没 export TAG） | 加 `export DSH_PROGRESS_TAG="$TAG"`（TAG 默认 `AB_RV4`） |
| 2 | 直连 `match_via_sdk.py` 的 `.sh`（`_value_ab_vs_rv4.sh`、`_filter_gate_matrix.sh`、`_cand_mode_zero_impact_game.sh`、`_unified_cand_ab.sh` 等）**没有 ladder ⇒ 不写 status.txt**，这些批次没有整批汇总 | 上述脚本 | 要么接受，要么统一走 `_tmp_ladder.py` 驱动 |
| 3 | `_filter_gate_matrix.sh` 硬编码 `DSH_PROGRESS_TAG=filter_gate_matrix`，多臂靠 `DSH_PROGRESS_SUFFIX` 区分 | `_filter_gate_matrix.sh:37` | 可接受，但注意它直连（无 status） |
| 4 | `az_selfplay.py:798` / `collect_value_prior.py:95` 写 progress 文件但**命名/位置不同**（5 位补零、不在 training_history/progress 下） | 见 1.0 表 | 暂不动（迁移影响面大）；文档标注即可 |
| 5 | `_ladder_watchdog.ps1:25` 按文件名 `az_progress_seed<N>_<Tag>.txt` 匹配（带 `_<Tag>` 后缀的旧格式），与当前实际文件名（`az_progress_seed<N>[_p0|_p1|_<tag>].txt`）不完全一致 | `_ladder_watchdog.ps1` | 若 §3.1 改目录名，必须一并重写路径匹配 |

---

## §4 风险与建议

### 4.1 要不要统一成一个 writer？

**建议：不要合并成单一 writer 程序；只统一"公共逻辑"。**

理由：
- 4 个驱动职责不同：`_tmp_ladder.py` 有重试/看门狗/`final-sweep`（L162-176、L794-840），`ab_search_modes.py` 有 `_progress.json` + RLock + retry/resume（L243-256、L269-288），把它们合并风险大、收益低。
- 真正该统一的只有**三件公共逻辑**，下沉到 `progress.py`：
  1. 目录名解析（`progress_dir(tag)`，含 stamp 兜底，见 §3.1）；
  2. "最近完成"的排序（提供 `recent_from_logs(tokens, log_dir)` 帮助函数，按 mtime 排序）；
  3. 看门狗/消费者的目录 glob（复用同一函数）。
- 各驱动只调用这些函数，互不合并。

### 4.2 风险清单

| 风险 | 说明 | 缓解 |
|---|---|---|
| **改目录名影响在跑的长批次/看门狗** | `_tmp_ladder.py:187`、`_ladder_watchdog.ps1` 按 `<tag>` 精确拼路径，改名后 glob 不到新目录 | §3.1 的"回退扫描"：先 `<tag>_<stamp>`，找不到再 `<tag>`；或统一走 `progress_dir()` |
| **stamp 不一致导致更碎** | 若有人在 `_ensure_dir` 里每次 `time.strftime` 兜底，同 tag 每局各进一个目录 | stamp 必须"进程内只算一次 + 驱动入口 setdefault + 子进程共享 env"（§3.1 第 1、3 步） |
| **旧分析脚本 break** | 精确 `glob("<tag>")` 的脚本（`probe_*`、`audit_*` 用 `rglob` 的没事） | 先跑一遍 grep 受影响消费者，逐个加回退 |
| **status bug 修复范围** | 只改两个 ladder 脚本的 `_recent`，别顺手动 W/L 聚合 | 小步提交，单测对比 status.txt 文本 |

### 4.3 实施顺序建议（小步、可回滚）

1. **先修 Bug ②**（status.txt "最近完成"）：纯逻辑、零风险、立即可验证。
2. **再修 Bug ①**（时间戳必加）：先加"驱动入口 setdefault + progress_dir() 回退扫描"，确认看门狗/分析脚本兼容后再上"强制 stamp"。
3. **最后清默认 tag 漏设**（§3.3 #1）：给 `_value_ab_vs_rv4.sh` 补 tag，避免继续污染 `match/`。

---

## 附：本次调查涉及的关键文件与行号速查

| 文件 | 关键行 | 内容 |
|---|---|---|
| `code/test_match/progress.py` | 44-61 | `_ensure_dir`（目录名 + stamp） |
| 同上 | 104 / 121-123 / 137 | trace 文件名 / 逐回合行 / 终局行 |
| 同上 | 152-176 | `write_status`（含 `recent` 切片 L170） |
| `code/test_match/match_via_sdk.py` | 664-673 / 766-776 / 849-855 | trace 初始化 / 逐回合写 / 终局写 |
| `_tmp_ladder.py` | 501 / 746 / 566-613 | DSH_PROGRESS_TAG / tag 默认 / status 块 |
| 同上 | 576-599（bug L597） | "最近完成"（先完成） |
| `_tmp_ladder_head.py` | 111 / 240 / 133-180 | 同上 |
| 同上 | 143-166（bug L164） | "最近完成"（先完成） |
| `code/test_match/ab_search_modes.py` | 235-236 / 342-367 | tag / status（`recent` 正确 L349-353） |
| `code/test_match/_tmp_ab_old.py` | 69-70 / 142-166 | tag / status（`recent` 正确 L148-152） |
| `code/run_logged.sh` | 36 / 61 | TS / `DSH_PROGRESS_DIR_STAMP` 唯一设置点 |
| `code/my_ai/az_intent/az_selfplay.py` | 798 | 自对弈 progress（不在 progress/ 下） |
| `code/my_ai/az_intent/collect_value_prior.py` | 95 | 采集 progress（不在 progress/ 下） |

> 已实施（commit 1a8d63c）：Bug① 目录名统一加时间戳后缀 + 回退扫描（`progress_dir`/`progress_scan_dirs`）；Bug② status"最近完成"改为按日志 mtime 降序取前 6。
