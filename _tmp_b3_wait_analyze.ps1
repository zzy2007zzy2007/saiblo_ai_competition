# _tmp_b3_wait_analyze.ps1 —— 等 `B3` 基线跑完，**自动**跑判据统计并存盘（不做任何判断/不改文档）
#
# 为什么这样：单次前台命令上限 600s（实测），而基线要跑 ~5–6 h ⇒ 用"轮询式逐段等待"会
# 把一整轮对话拆成几十次空转。这里把**等待 + 机械的复算**交给一个后台作业：
#   * 它只做确定性的事（等进程退出 → 跑 analyze_paired.py → 存 txt/json）；
#   * **判读、记档、宣布成功仍然由我（agent）来做**，脚本不做任何结论。
# 产出（都在 ladder 日志目录里，便于独立验证者直接拿）：
#   match_results/ladder_logs/<tag>/_analysis.txt    判据统计的完整输出（UTF-8）
#   match_results/ladder_logs/<tag>/_analysis.json   同上的结构化版本
#   match_results/ladder_logs/<tag>/_analysis.meta   跑完时间 + 退出码
#
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File _tmp_b3_wait_analyze.ps1 [-Tag B3_baseline_rulev4_128]
param(
    [string]$Tag = 'B3_baseline_rulev4_128',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$TimeoutMin = 600
)

$Repo = 'D:/2026智能体大赛_新2'
Set-Location $Repo
$logDir = "match_results/ladder_logs/$Tag"
$deadline = (Get-Date).AddMinutes($TimeoutMin)

Write-Host "[wait] 等 ladder 退出（最多 $TimeoutMin min）…"
$sawRunning = $false
while ((Get-Date) -lt $deadline) {
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*_tmp_ladder.py*' })
    if ($running.Count -gt 0) { $sawRunning = $true }
    elseif ($sawRunning) { Write-Host "[wait] ladder 已退出 $(Get-Date -Format 'HH:mm:ss')"; break }
    else { Write-Host "[wait] 还没看到 ladder 进程，继续等…" }
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 20   # 让 ladder 把汇总刷完盘

Write-Host "[analyze] 开始 $(Get-Date -Format 'HH:mm:ss')"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag `
    --json="$logDir/_analysis.json" 2>&1 | Out-String
$rc = $LASTEXITCODE
[System.IO.File]::WriteAllText((Join-Path $Repo "$logDir/_analysis.txt"), $out,
    (New-Object System.Text.UTF8Encoding($false)))
"end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`nexit_code: $rc`ntag: $Tag" |
    Set-Content -Path "$logDir/_analysis.meta" -Encoding utf8
Write-Host "[analyze] 完成 rc=$rc -> $logDir/_analysis.txt"
Write-Host $out
