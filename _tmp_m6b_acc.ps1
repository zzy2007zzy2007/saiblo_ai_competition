# _tmp_m6b_acc.ps1 —— M6b（`pos_pin=mc_quiet`：**安静 rollout** 的 MC 类决策）的 128 局验收
#   关键：M6（贪心 rollout）会**原生崩溃**（32% INVALID）；M6b 只打候选那一手、之后谁都不出招地推进
#   ⇒ 在曾崩的 8 个 token 上**全出有效结果**（含 seed 11 先手，此前必崩）。
#   成本（修后）：每个决策只算一次 MC + `AZAI_MC_EVERY` 门控 + 截断 rollout。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m6b] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m6b] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'

$tokens = ((7..70) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'M6b_mcq_128'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-70 `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[m6b] 完成 $(Get-Date -Format 'HH:mm:ss')"
