# _tmp_ktau.ps1 —— kgeo 的 tau 小阶梯（预注册 §8）：35 / 50(对照) / 70，各 16 局 A1 配置快筛
#   部署必须带 AZAI_VALUE_TANH=0 + AZAI_REL2ABS=1.0（rel 头还原绝对价值）
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[ktau] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[ktau] 开工 $(Get-Date -Format 'HH:mm:ss')"

# ① 训练两个新臂
foreach ($tv in @(@('k35', '35'), @('k70', '70'))) {
    $n = $tv[0]; $tau = $tv[1]
    & $runner -Name "VT_$n" -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data training_history/vprior/data_pin002_100 --cache training_history/vprior/vcache_pin002 --label-mode rel --label-weight kgeo --tau $tau --epochs 8 --lr 1e-4 --freeze-bn --out training_history/vprior/posnet_V$n.pt"
}

# ② A1 配置 16 局快筛（三臂：k35 / k50 / k70）
$tokens = ((7..14) | ForEach-Object { "$_", "${_}r" }) -join ' '
$arms = @(
    @('VK35_16', 'training_history/vprior/posnet_Vk35.pt'),
    @('VK50c_16', 'training_history/vprior/posnet_Vk50.pt'),
    @('VK70_16', 'training_history/vprior/posnet_Vk70.pt')
)
foreach ($a in $arms) {
    $tag = $a[0]; $ck = $a[1]
    if (-not (Test-Path $ck)) { Write-Host "[ktau] 跳过 $tag（缺 $ck）"; continue }
    $env:AZAI_CKPT = $ck
    $env:AZAI_VALUE_TANH = '0'; $env:AZAI_REL2ABS = '1.0'; $env:AZAI_MENU3 = '0'
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
    $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
    $env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    Write-Host "[ktau] == $tag 开始 $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-14 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}
Write-Host "[ktau] 完成 $(Get-Date -Format 'HH:mm:ss')"
