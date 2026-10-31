# 阶梯实验看门狗：某局 progress 文件长时间不前进 ⇒ 报警（`-Kill` 时**按 PID 精确**杀掉该局进程树）
#
# 为什么需要（2026-10-29 的真实事故）：
#   `PIN_K2_VS_RU` 第一次起跑的 4 局在 round 40-45「卡」了 2 小时，最后落成 `verdict=timeout`。
#   实际根因是**整台机器进了 Modern Standby 待机**（系统事件 `2026/10/29 7:52:19 系统已从低功耗状态中恢复`；
#   对手自己的日志里 `elapsed_us` 涨到 235 s / 7551 s）。见 `docs/pin_k2_ab_20261028.md` §2.1。
#   ⇒ 根治手段是 `_keepawake.ps1`；本脚本是**第二道防线**：
#   即使还是卡了（真的死锁 / 管道写满 / 对手挂了），也能在 N 分钟内发现并只清掉**那一局**，
#   而不是让整批陪着等 30 分钟的 `MVS_TIMEOUT_SECONDS`（甚至整局 `LADDER_TIMEOUT`）。
#
# ⚠️ 安全设计（用户明确要求：**严禁按镜像名/通配符批量杀**）：
#   * 默认**只看不杀**（`-Kill` 才动手）；
#   * 杀的范围 = `<tag>` 那一局**自己的**进程树（`match_via_sdk.py <token>` 及其所有后代），
#     从 `ParentProcessId` 递归求得，**逐个 PID** `Stop-Process`；
#   * 动手前会打印完整 PID 清单；带 `-WhatIf` 可以只打印不执行；
#   * **不会**碰任何别的 tag 的进程（token 对不上就不在树里）。
#
# 用法：
#   # 只观察（安全，可常驻）
#   powershell -NoProfile -ExecutionPolicy Bypass -File code/test_match/_ladder_watchdog.ps1 `
#       -Tag PIN_K2_VS_RU -StallMinutes 10 -IntervalSeconds 120
#   # 到阈值就精确清掉那一局（会重启该 token 由 `--resume` 语义补跑）
#   ... -Tag PIN_K2_VS_RU -StallMinutes 10 -Kill
#
# 判据：`training_history/progress/<Tag>_<stamp>/az_progress_seed<N>[r][_p0|_p1].txt` 的
#   **LastWriteTime** 距今 > `StallMinutes` ⇒ 判该 seed 卡住（旧无后缀目录 `<Tag>` 也一并扫）。
#   用 mtime 而不是解析 `round=`：mtime 与内容格式无关，且 bridge 每回合都写，语义稳定。

param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [int]$StallMinutes = 10,
    [int]$IntervalSeconds = 120,
    [int]$MaxRounds = 0,                       # 0 = 常驻；>0 = 只看这么多轮就退出（便于冒烟）
    [switch]$Kill,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$progRoot = Join-Path $repo "training_history\progress"

function Get-Descendants {
    param([int[]]$RootIds)
    $all = Get-CimInstance Win32_Process
    $set = New-Object System.Collections.Generic.HashSet[int]
    foreach ($r in $RootIds) { [void]$set.Add($r) }
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($p in $all) {
            if ($set.Contains([int]$p.ParentProcessId) -and -not $set.Contains([int]$p.ProcessId)) {
                [void]$set.Add([int]$p.ProcessId); $changed = $true
            }
        }
    }
    # 只返回还活着的，按父→子排序（先杀子更干净，但 Stop-Process 对单个 PID 无所谓顺序）
    return $all | Where-Object { $set.Contains([int]$_.ProcessId) }
}

# token → `match_via_sdk.py` 的 PID：命令行**最后一个参数**就是 token（`_tmp_ladder.py` 拼的）。
#
# ⚠️ **必须再按 tag 认一次祖先**：token 名（`7`/`11r`/…）在各批次之间是**重名的**。
#   2026-10-29 实测同一台机器上同时跑 `US_VS_CHAMPION_BASE` 与 `PIN_K2_VS_RU` 两批，
#   都跑 token `11r` ⇒ 只按 token 匹配会把**别人的对局**算进来（`-Kill` 时甚至杀掉）。
#   判据：从 `match_via_sdk.py` 往上找祖先，祖先命令行里必须有 `--tag=<Tag>`
#   （`_tmp_ladder.py` 是这么被调的）。认不到就**跳过并打印原因**，绝不猜。
function Get-BridgeForToken {
    param([string]$Token, [string]$Tag)
    $all = Get-CimInstance Win32_Process
    $byPid = @{}
    foreach ($p in $all) { $byPid[[int]$p.ProcessId] = $p }
    $out = @()
    $cands = $all | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match 'match_via_sdk\.py' -and
        $_.CommandLine -match "(^|\s)$([regex]::Escape($Token))\s*$"
    }
    foreach ($c in $cands) {
        $cur = $c
        $owned = $false
        for ($i = 0; $i -lt 8 -and $cur; $i++) {          # 最多往上 8 层，防环
            $ppid = [int]$cur.ParentProcessId
            if (-not $byPid.ContainsKey($ppid)) { break }
            $cur = $byPid[$ppid]
            if ($cur.CommandLine -and $cur.CommandLine -match "--tag=$([regex]::Escape($Tag))(\s|$)") {
                $owned = $true
                break
            }
        }
        if ($owned) { $out += $c }
        else {
            Write-Output ("[wd]   (跳过 PID {0}：token={1} 但祖先链里没有 --tag={2} ⇒ 不是本批的局，不碰)" -f `
                $c.ProcessId, $Token, $Tag)
        }
    }
    return $out | Sort-Object ProcessId -Unique
}

Write-Output ("[wd] tag={0}  stall>{1} min  interval={2}s  kill={3}  whatif={4}  repo={5}" -f `
    $Tag, $StallMinutes, $IntervalSeconds, $Kill, $WhatIf, $repo)

$iter = 0
while ($true) {
    $iter++
    # 2026-10-31（Bug ① 修复）：进度目录名从 `<Tag>` 变成 `<Tag>_<stamp>`；
    #   每轮重扫，新旧两种都看（旧无后缀目录向后兼容）。
    $progDirs = @(Get-ChildItem $progRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $Tag -or $_.Name.StartsWith("${Tag}_") })
    if ($progDirs.Count -eq 0) {
        Write-Output ("[wd] {0}  progress 目录还不存在（还没开跑？）: {1}" -f (Get-Date -Format 'HH:mm:ss'), $progRoot)
    } else {
        $now = Get-Date
        foreach ($d in $progDirs) {
            foreach ($f in Get-ChildItem $d.FullName -Filter "az_progress_seed*.txt" -ErrorAction SilentlyContinue) {
                $idleMin = [math]::Round(($now - $f.LastWriteTime).TotalMinutes, 1)
                $seed = ($f.Name -replace '^az_progress_seed(\d+).*$', '$1')
                $last = (Get-Content $f.FullName -Tail 1 -ErrorAction SilentlyContinue)
                # ⚠️ 已终局的局**不更新** progress 文件 ⇒ mtime 会一直变老，会被误判成 STALL。
                #   bridge 终局时写的第一行是 `terminal …`，见到它就跳过（否则会对已完成的局狂报警）。
                if ($last -match '^\s*terminal') {
                    Write-Output ("[wd] {0} done  seed={1,-4} （已终局，不判卡死）" -f (Get-Date -Format 'HH:mm:ss'), $seed)
                    continue
                }
                $tagTxt = if ($idleMin -gt $StallMinutes) { 'STALL' } else { 'ok   ' }
                Write-Output ("[wd] {0} {1} seed={2,-4} idle={3,6} min  last={4}" -f `
                    (Get-Date -Format 'HH:mm:ss'), $tagTxt, $seed, $idleMin, $last)
                if ($idleMin -le $StallMinutes) { continue }
                # ── 卡住了：找到这一局的进程树 ──
                # progress 文件名里的 seed 是数字，但 token 可能是 `7` 或 `7r`（镜像）；
                # 两种都要试（同一 seed 只有一局在跑，除非两个 token 并发 —— 那就是两棵树，分别报）。
                foreach ($tok in @("$seed", "${seed}r")) {
                    $bridges = Get-BridgeForToken -Token $tok -Tag $Tag
                    if (-not $bridges) { continue }
                    $tree = Get-Descendants -RootIds ($bridges | ForEach-Object { [int]$_.ProcessId })
                    Write-Output ("[wd]   !! token={0} 疑似卡住 {1} min ⇒ 进程树 {2} 个:" -f `
                        $tok, $idleMin, ($tree | Measure-Object).Count)
                    $tree | Select-Object ProcessId, ParentProcessId, Name | ForEach-Object {
                        Write-Output ("[wd]      {0,-7} parent={1,-7} {2}" -f $_.ProcessId, $_.ParentProcessId, $_.Name)
                    }
                    if ($Kill -and -not $WhatIf) {
                        foreach ($p in $tree) {
                            try { Stop-Process -Id ([int]$p.ProcessId) -Force -ErrorAction Stop
                                  Write-Output ("[wd]      killed {0} ({1})" -f $p.ProcessId, $p.Name) }
                            catch { Write-Output ("[wd]      FAILED {0}: {1}" -f $p.ProcessId, $_.Exception.Message) }
                        }
                        Write-Output ("[wd]   ⇒ token={0} 已清掉；该局会由 --resume 语义在下一轮补跑" -f $tok)
                    } elseif ($Kill -and $WhatIf) {
                        Write-Output ("[wd]   (WhatIf) 本应清掉 token={0} 的 {1} 个进程" -f $tok, ($tree | Measure-Object).Count)
                    }
                }
            }
        }
    }
    if ($MaxRounds -gt 0 -and $iter -ge $MaxRounds) { Write-Output "[wd] 到 MaxRounds，退出"; break }
    Start-Sleep -Seconds $IntervalSeconds
}
