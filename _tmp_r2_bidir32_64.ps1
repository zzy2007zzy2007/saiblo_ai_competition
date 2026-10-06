# _tmp_r2_bidir32_64.ps1 —— rollout 阶梯 R2：**K=32 双向**，固定段前缀 seed 7..38（64 局）
#   预注册：docs/prereg_20261006_rollout_ladder.md（R1 已测：p̂=0.6562 = R0，未晋级；按阶梯继续 R2）
#   ⚠️ 成本：K=32 ⇒ 单决策约为 K=8 的 4×（K=8 实测 dec_max 17.5 s）⇒ 预期 dec_max ~60-70 s，
#      仍低于 bridge 读超时（MVS_TIMEOUT_SECONDS 默认 300 s）与慢阈值告警（AZAI_SLOW_DECISION_SEC=90）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Tag = 'R2_bidir32_64',
    [int]$K = 32
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

$seeds = 7..38
$tokens = (($seeds | ForEach-Object { "$_", "${_}r" }) -join ' ')
Write-Host "[$Tag] seed 数=$($seeds.Count)，局数=$(2*$seeds.Count)；K=$K"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = "$K"
$env:AZAI_SLOW_DECISION_SEC = '90'      # K=32 本来就更慢：把"慢决策"告警线相应抬高，但仍远低于 300 s 读超时

& $runner -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag --seeds=7-38 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[$Tag] 完成 $(Get-Date -Format 'HH:mm:ss')"
