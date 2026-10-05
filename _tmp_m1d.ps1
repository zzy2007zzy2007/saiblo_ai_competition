# _tmp_m1d.ps1 —— M1d：**全反频率权重**（power 1.0）+ masked 部署
#   理由：power 0.5（M1c）时类头学会了"闪电/HOLD"两个分支，但**花钱分支（0.8%）仍赢不了 argmax**。
#   加权到 **每类等权**（w ∝ 1/出现次数）等价于"让分类器在各分支间平衡召回" ⇒ 这才有可能把
#   「该花才花」的**条件结构**学进 argmax。代价：可能过度花钱（D 型崩法），故必须看机制读数。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m1d] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay|train_class_outcome') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m1d] 开工 $(Get-Date -Format 'HH:mm:ss')"

& $runner -Name M1d_train -CommandLine "$Py -u code/my_ai/az_intent/train_class_outcome.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_sp_reserve60_exec --epochs 20 --lr 1e-3 --beta 1000000 --class-weight auto --class-weight-power 1.0 --out training_history/vprior/posnet_M1d.pt"

$env:AZAI_CKPT = 'training_history/vprior/posnet_M1d.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'masked'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((7..14) | ForEach-Object { "$_", "${_}r" }) -join ' '   # 先跑 16 局（成本闸门：masked+多候选会慢）
$tag = 'M1d_masked_16'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-14 `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[m1d] 完成 $(Get-Date -Format 'HH:mm:ss')"
