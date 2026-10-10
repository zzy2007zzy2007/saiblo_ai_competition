# 工具/自动化踩坑清单（PowerShell + 本项目脚本）

> 真实日期 **2026-10-09**。本 goal 期间（第 60–69 轮）我在**自动化边角**上反复踩到同一类坑，
> 每次都靠"语法检查 / fail-fast / 机制闸门"接住。集中记下来，避免以后重复。

## 1. `powershell -File script.ps1` 的**参数传递**限制（踩了 3 次）

| 坑 | 现象 | 正确做法 |
|---|---|---|
| **哈希表参数** | `-ExtraEnv @{A='1'}` ⇒ `Cannot convert "System.Collections.Hashtable" from "System.String"`（被字符串化）⇒ 参数绑定失败，**脚本静默继续**、采集 0 局 | 让**脚本内部**从 `$env:` 读，或调用方在**父进程先设 `$env:X`**（子进程继承）。本项目里：驱动**不管理** `AZAI_DUMP_RAW_DIR/AZAI_AZ_DUMP/AZAI_MC_TAIL/_MIX` ⇒ 直接在父进程设即可 |
| **空字符串参数** | `-McScore ''` ⇒ `Missing an argument for parameter 'McScore'` ⇒ **整个臂瞬间报错** | 想用默认值就**省略**该参数（本项目驱动的 `-McScore` 默认就是空串） |
| **中文脚本必须 UTF-8 BOM** | 无 BOM 的 `.ps1` 里中文注释被按 ANSI 解析 ⇒ 解析器报一堆语法错 | 写文件后用 `[System.IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($true)))` 补 BOM，再 `Parser::ParseFile` 校验 |

（另：**`Tee-Object` 写出的日志是 UTF-16**，普通文本读取器会读不到 ⇒ 读日志要用能识别 BOM 的方式。）

## 2. **陈旧产物目录**会骗过守卫（踩了 2 次）

- 失败/被杀的 run 会留下同名目录（`match_results/ladder_logs/SMK_*`、`data/*_obs/<tag>/*.bin`）；
  只检查"目录是否存在/日志里有没有 RESULT"的守卫会被**旧文件**满足 ⇒ 提前放行 ⇒ 用尚不存在的 ckpt 起跑（随后被冒烟闸门拦下）。
- **做法**：① 守卫先等**真正的产物**（如 `checkpoints/xxx.pt`）；② 只读 **mtime 晚于本次开始** 的日志；
  ③ 陈旧目录**改名留证**（`*_STALE_<日期>_<时刻>`）而不是删除。

## 3. **tag 命名的隐式约定**（踩了 1 次）

- `_tmp_arm.ps1` 跑出来的 ladder tag 是 **`AB_<Name>_vs_<Opp>`**（不是裸 `<Name>`）；
  按裸名去 `--ladder-tag` 查日志会得到 **0 个标签 ⇒ ingest 0 个 pkl**。
- **做法**：向 ingest 传**实际 tag**；并用 fail-fast 断言"pkl 数 > 0"（本次正是它中止了空转）。

## 4. **成本判断要用 CPU 增量，不要看"有没有输出"**

- `joint + K=24 + SAMPLE_MULT=15` 时**单次决策 ~30 min**：日志缓冲导致"看起来没动"，
  而只过滤 ladder 进程的 CPU 会显示 0（它只是**在等对局**）。
- **做法**：采样**全部** python 进程的 `KernelModeTime+UserModeTime` 增量：bridge 满核 ⇒ 在算（只是慢）；全为 0 ⇒ 真卡住。

## 5. 本 goal 里**被纪律接住**的次数（值得记）

| 纪律 | 接住的问题 |
|---|---|
| `Parser::ParseFile` 语法检查 | BOM 缺失、`-ExtraEnv` 那行的语法错误 |
| 驱动**冒烟闸门**（建塔 ≤0.5/局 判退化） | 两次：空跑产出的"0 建塔+2 无效"、陈旧目录导致的不存在 ckpt |
| 流水线 **fail-fast**（采集未跑满/0 观测/0 pkl/无 ckpt） | 采集 0 局、ingest 0 pkl（tag 写错） |
| **预注册 + 控制臂** | 价值网"+3.9pp"在两段反向 ⇒ 判为不可靠（否则会采纳一个假结论） |
