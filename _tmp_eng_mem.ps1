# _tmp_eng_mem.ps1 —— 复现 mc_light 的原生崩溃，同时**采样每个 AI 进程的内存**
#   背景：同一 seed-11 第 55 回合状态（hp=[46,47] coins=[95,63] towers=1）
#     * 8 局并发（M6_mc_128 / M6_retry8）⇒ 两次都 INVALID（崩在 round 55）
#     * 单局 + faulthandler ⇒ 跑过去了
#   ⇒ 假设：内存累积/耗尽（mc_light 每局几百次 clone + 前向）。本脚本同时抓
#     ① 原生崩溃的 faulthandler 栈（MVS_FAULTHANDLER=1）② 每个 AI 的 WorkingSet 曲线。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Minutes = 50,
    [int]$SampleSec = 15
)

Set-Location $Repo
$tag = 'ENG_mem8'
$csv = "$Repo/training_history/vprior/eng_mem8_samples.csv"
"t_sec,pid,workingset_mb,private_mb" | Set-Content -Encoding UTF8 $csv

$env:MVS_FAULTHANDLER = '1'
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'mc_light'; $env:AZAI_MC_HORIZON = '100'; $env:AZAI_MC_EVERY = '4'
$env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'

$cl = "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 11 11r 13 13r 14 14r 18 18r"
Write-Host "[eng] 启动 ladder $(Get-Date -Format 'HH:mm:ss')"
$job = Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',
    "& '$Repo/code/run_logged.ps1' -Name $tag -CommandLine `"$cl`"") -PassThru -WindowStyle Hidden

$t0 = Get-Date
$peak = 0
while (((Get-Date) - $t0).TotalMinutes -lt $Minutes) {
    Start-Sleep -Seconds $SampleSec
    $el = [int]((Get-Date) - $t0).TotalSeconds
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'az_bridge_ai') })
    $tot = 0
    foreach ($p in $procs) {
        $ws = [math]::Round($p.WorkingSetSize / 1MB, 1)
        $pv = [math]::Round($p.PrivatePageCount / 1MB, 1)
        "$el,$($p.ProcessId),$ws,$pv" | Add-Content -Encoding UTF8 $csv
        $tot += $ws
    }
    if ($tot -gt $peak) { $peak = $tot }
    $ladderAlive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') }).Count
    Write-Host ("[{0}] AI进程={1} 合计WS={2:N0} MB 峰值={3:N0} MB ladder={4}" -f (Get-Date -Format 'HH:mm:ss'), $procs.Count, $tot, $peak, $ladderAlive)
    if ($ladderAlive -eq 0 -and $el -gt 60) { Write-Host "[eng] ladder 已退出"; break }
}
Write-Host "[eng] 监视结束 $(Get-Date -Format 'HH:mm:ss') 峰值合计WS=$([math]::Round($peak)) MB"
$mem = Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
Write-Host ("[eng] 机器物理内存 = {0:N0} MB" -f ($mem / 1MB))
