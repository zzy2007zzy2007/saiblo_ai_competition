# _tmp_m6b_repro.ps1 —— ① **M6b 原样重跑**（当前代码、`MC_NODOWN` 不设）+ ② **同代码 A1 对照**
#   目的：
#     * ① 检验 M6b 的 0.6641 在**当前代码**上是否复现（独立验证者限制②：M6b 跑在 ff9c2a8；
#          此后我加了 mc_taken/mc_fallback 埋点与 mc_no_downgrade 旋钮 ⇒ 需要同代码复核）；
#     * ② 同代码（当前 HEAD）上的 A1 对照，量化"代码漂移"对 A1 读数的影响。
#   顺序跑（机器纪律：一次只跑一个多进程批）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

function Wait-Idle([int]$min) {
    $deadline = (Get-Date).AddMinutes($min); $empty = 0
    while ((Get-Date) -lt $deadline) {
        $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
        if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
        Start-Sleep -Seconds 30
    }
    Start-Sleep -Seconds 10
}

Wait-Idle $WaitTimeoutMin
Write-Host "[repro] 开工 $(Get-Date -Format 'HH:mm:ss')"

$tokens = ((7..70) | ForEach-Object { "$_", "${_}r" }) -join ' '

# 公共：A1 底座
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_MENU3 = '0'; $env:AZAI_REL2ABS = '0'
Remove-Item Env:AZAI_MC_NODOWN -ErrorAction SilentlyContinue

# ① M6b 原样重跑
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'
& $runner -Name M6b_re128 -CommandLine "$Py -u _tmp_ladder.py --tag=M6b_re128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=M6b_re128 --seeds=7-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/M6b_re128/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out

# ② 同代码 A1 对照
Wait-Idle 30
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_MC_EVERY = '0'
& $runner -Name A1_head_128 -CommandLine "$Py -u _tmp_ladder.py --tag=A1_head_128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out2 = & $Py -u code/test_match/analyze_paired.py --tag=A1_head_128 --seeds=7-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/A1_head_128/_analysis.txt", $out2,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out2
Write-Host "[repro] 完成 $(Get-Date -Format 'HH:mm:ss')"
