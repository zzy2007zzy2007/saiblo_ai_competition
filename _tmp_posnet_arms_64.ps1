# _tmp_posnet_arms_64.ps1 —— 预注册 `docs/prereg_20261007_posnet_arms.md`：
#   位置网三臂单变量对照（只换 AZAI_CKPT 的 pos_state），各 64 局（固定段前缀 7..38）
#   P-A 的读数**已有**（M6cROOT_128 同前缀 = 0.6562）⇒ 本脚本只跑 P-A1、P-B 两臂。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 30
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[posnet] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|az_selfplay|train_') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}

$seeds = 7..38
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')

$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = '0'            # ⚠️ 与 P-A 对照必须一致：不用双向 rollout
$env:AZAI_SLOW_DECISION_SEC = '30'

foreach ($arm in @(@('P-A1', 'checkpoints/posnet_A1_km1_m32.pt'),
                   @('P-B',  'checkpoints/posnet_B_km1.pt'))) {
    $tag = "P_$($arm[0])_64"
    $env:AZAI_CKPT = $arm[1]
    Write-Host "[$tag] ckpt=$($arm[1])  seed 数=$($seeds.Count) 局数=64"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-38 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}
Write-Host "[posnet] 全部完成 $(Get-Date -Format 'HH:mm:ss')"
