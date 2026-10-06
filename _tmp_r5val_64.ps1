# _tmp_r5val_64.ps1 —— 预注册 `docs/prereg_20261007_value_retrain.md` 的价值头筛选：
#   用**新分布上重训过的价值头**（posnet_R5_val.pt，P0 已通过：val MSE 0.10145 → 0.09763）打 64 局（前缀 7..38）
#   对照 = M6cROOT_128 同前缀 0.6562；晋级线 0.7362。
#   ⚠️ 单变量已核：class_state / pos_state 逐张量相同，**只有 value_state 变**。
#   ⚠️ 换 ckpt = 机械门 ⇒ 若晋级须跑两段各 128 局验收 + 独立验证。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Tag = 'V_r5val_64',
    [string]$Ckpt = 'training_history/vprior/posnet_R5_val.pt'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
$seeds = 7..38
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')
Write-Host "[$Tag] ckpt=$Ckpt seed 数=$($seeds.Count)，局数=64"

$env:AZAI_CKPT = $Ckpt
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
$env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
$env:AZAI_SLOW_DECISION_SEC = '30'

& $runner -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag --seeds=7-38 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[$Tag] 完成 $(Get-Date -Format 'HH:mm:ss')"
