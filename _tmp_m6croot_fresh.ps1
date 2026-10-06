# _tmp_m6croot_fresh.ps1 —— M6c-ROOT 的**未参与筛选的 seed 段复核**（seed 103..166 = 128 局）
#   目的（机械门的一部分）：判据 seed 7..70 上的 0.7344 是**首次跨过 0.70**；
#   用**另一段 seed**（103..166，与 7..70 不重叠、此前只用于 E2 的新段复核）做**样本外确认**。
#   对照：A1 在该段的读数（此前只跑过 103..166 的 A1 吗？——若没有，同批一起跑一个 A1 对照更干净）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 60
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
Write-Host "[rootfresh] 开工 $(Get-Date -Format 'HH:mm:ss')"
$tokens = ((103..166) | ForEach-Object { "$_", "${_}r" }) -join ' '

# ① 样本外：M6c-ROOT
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
& $runner -Name M6cROOT_fresh128 -CommandLine "$Py -u _tmp_ladder.py --tag=M6cROOT_fresh128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=M6cROOT_fresh128 --seeds=103-166 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/M6cROOT_fresh128/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out

# ② 同段 A1 对照（同代码、同一批 seed）
Wait-Idle 30
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_MC_EVERY = '0'; $env:AZAI_MC_ROOTONLY = '0'
& $runner -Name A1_fresh128 -CommandLine "$Py -u _tmp_ladder.py --tag=A1_fresh128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out2 = & $Py -u code/test_match/analyze_paired.py --tag=A1_fresh128 --seeds=103-166 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/A1_fresh128/_analysis.txt", $out2,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out2
Write-Host "[rootfresh] 完成 $(Get-Date -Format 'HH:mm:ss')"
