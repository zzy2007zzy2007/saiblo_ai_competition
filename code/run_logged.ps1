# run_logged.ps1 —— `code/run_logged.sh` 的等价 PowerShell 版（**只换记录工具，不改任何被测量的量**）
#
# 为什么有它：2026-10-05 发现本机 `bash` 不可用 —— PATH 里的
# `...\WindowsApps\bash.exe` 只是 **WSL 壳**，而本机**没有装 WSL 发行版**（实测报
# "Windows Subsystem for Linux 尚未安装..."），PATH 里也没有 Git Bash ⇒ `run_logged.sh` 跑不了。
# 本脚本复刻它的**全部产物与格式**，让实验日志的历史可比性不受影响：
#   training_history/runs/<时间戳>_<名字>/{cmd.txt,meta.txt,output.log}
#   docs/experiment_log.md 末尾追加一条（result 需手动补）
# 并额外多写一个 env.txt（`AZAI_*/ANTWAR_*/PYTHON*/CUDA_*/OMP_*` 环境快照）——
# 因为本项目的"配置"很多走 **环境变量**，只记命令行不足以复现。
#
# 用法（环境变量请**调用前**设好，子进程会继承）:
#   code\run_logged.ps1 -Name B3_baseline_rulev4_128 `
#       -CommandLine 'D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_ladder.py --tag=B3_baseline_rulev4_128 --jobs=8 ...'
#
# ⚠️ 接口故意用**一整条命令行字符串**（和 bash 版的 `bash run_logged.sh NAME <命令...>` 同构）：
#    `powershell -File script.ps1 -A a b c` 会把带空格的参数拆散（实测踩到），
#    而单字符串没有这个问题。
#
# 编码：用 `cmd /c "<cmd> > output.log 2>&1"` 做重定向 —— **原生字节直落文件**，
# 不经过 PowerShell 的字符串管道（否则中文/emoji 会被重编码或直接抛 UnicodeEncodeError，
# 2026-09-22 就被 emoji 打断过整批汇总）。
# ⚠️ 本文件必须存成 **UTF-8 with BOM**：本机只有 Windows PowerShell 5.1，它读无 BOM 的
#    UTF-8 会按 ANSI 解 ⇒ 中文注释把语法搞坏（实测报 "Unexpected token '}'"）。
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$CommandLine
)

$ErrorActionPreference = 'Stop'
$Repo = 'D:/2026智能体大赛_新2'
Set-Location $Repo

$env:PYTHONIOENCODING = 'utf-8'   # 与 run_logged.sh 一致

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$humanTs = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$commit = (& git rev-parse --short HEAD 2>$null); if (-not $commit) { $commit = 'none' }
$porcelain = @(& git status --porcelain)
$dirtyN = @($porcelain | Where-Object { $_ -notmatch '^\?\?' }).Count
$untrackedN = @($porcelain | Where-Object { $_ -match '^\?\?' }).Count

$runDir = Join-Path $Repo "training_history/runs/${ts}_${Name}"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

# --- 复现用命令行（尽量与旧 bash 条目同形：`env A=1 B=2 <命令...>`）---
$envPrefix = (@(Get-ChildItem Env: | Where-Object { $_.Name -like 'AZAI_*' } |
    Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" })) -join ' '
$loggedCmd = if ($envPrefix) { "env $envPrefix $CommandLine" } else { $CommandLine }
Set-Content -Path (Join-Path $runDir 'cmd.txt') -Value $loggedCmd -Encoding utf8

@{ name = $Name; start = $humanTs; commit = $commit
   dirty_tracked_files = $dirtyN; untracked_files = $untrackedN } |
    ConvertTo-Json | Set-Content -Path (Join-Path $runDir 'meta.txt') -Encoding utf8

# 环境快照（只留名字像配置的，避免把整机环境写进仓库）
@(Get-ChildItem Env: | Where-Object { $_.Name -match '^(AZAI_|ANTWAR_|PYTHON|CUDA_|OMP_)' } |
    Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) |
    Set-Content -Path (Join-Path $runDir 'env.txt') -Encoding utf8

Write-Host "[run_logged] name=$Name  commit=$commit  dirty_tracked=$dirtyN  untracked=$untrackedN"
Write-Host "[run_logged] cmd: $loggedCmd"
Write-Host "[run_logged] log: $runDir/output.log"

$logPath = Join-Path $runDir 'output.log'
$t0 = Get-Date
$inner = "$CommandLine > `"$logPath`" 2>&1"
& cmd.exe /d /c $inner
$rc = $LASTEXITCODE
$dur = [int]((Get-Date) - $t0).TotalSeconds

Add-Content -Path (Join-Path $runDir 'meta.txt') -Value @(
    "end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "exit_code: $rc", "duration_s: $dur")

# ⚠️ 用**单引号** here-string + 占位符替换：双引号 here-string 里反引号是转义符，
# 会把 ``` 代码栅栏吃掉（实测踩到，日志里变成 "  `ash"）。
$entry = @'

## @HUMAN_TS@ — @NAME@

- **commit**: `@COMMIT@` (dirty: @DIRTY@ files)
- **exit**: @RC@，用时 @DUR@s
- **cmd**:
  ```bash
  @LOGGED_CMD@
  ```
- **output**: `training_history/runs/@TS@_@NAME@/output.log`
- **result**: _待填_
'@
$entry = $entry.Replace('@HUMAN_TS@', $humanTs).Replace('@NAME@', $Name).
    Replace('@COMMIT@', $commit).Replace('@DIRTY@', "$dirtyN").Replace('@RC@', "$rc").
    Replace('@DUR@', "$dur").Replace('@LOGGED_CMD@', $loggedCmd).Replace('@TS@', $ts)
Add-Content -Path (Join-Path $Repo 'docs/experiment_log.md') -Value $entry -Encoding utf8

Write-Host "[run_logged] exit=$rc  (${dur}s)  -> docs/experiment_log.md"
exit $rc
