# _tmp_eval_distill.ps1 —— 预注册 `docs/prereg_20261013_distill_k48_cells.md` 第 3 步：**判据尺度评估**
#   P1 门（先跑）：小批（seed 7..10，8 局）算"落点一致率"，必须 < 0.8
#   主判据：固定段 128 局 + 随机段 128 局（K=24；对照 = M6cROOT_128 / M6cROOT_rand128）
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Ckpt = 'training_history/vprior/posnet_dk48.pt',
    [string]$TagBase = 'DK48'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

function Set-Cfg {
    $env:AZAI_CKPT = $Ckpt
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'; $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'
    $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    $env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
    $env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
    $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
    $env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
    $env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
    $env:AZAI_SLOW_DECISION_SEC = '30'
}

# ---- P1 门：小批看落点是否变 ----
Set-Cfg
& $runner -Name "${TagBase}_gate" -CommandLine "$Py -u _tmp_ladder.py --tag=${TagBase}_gate --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 7 7r 8 8r 9 9r 10 10r"
Write-Host '[gate] 落点一致率（对照 vs 学生；应 < 0.8）:'
& $Py -u code/test_match/cmp_op_cells.py M6cROOT_128 "${TagBase}_gate" 7-10

# ---- 固定段 128 局 ----
Set-Cfg
$fixed = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
& $runner -Name "${TagBase}_fixed128" -CommandLine "$Py -u _tmp_ladder.py --tag=${TagBase}_fixed128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $fixed"
$o1 = & $Py -u code/test_match/analyze_paired.py --tag="${TagBase}_fixed128" --seeds=7-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/${TagBase}_fixed128/_analysis.txt", $o1, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $o1

# ---- 随机段 128 局 ----
Set-Cfg
$spec = (Get-Content -Raw -Encoding UTF8 "$Repo/training_history/vprior/_rand_seeds_spec.txt").Trim()
$rand = (($spec -split ',') | ForEach-Object { $_, "${_}r" }) -join ' '
& $runner -Name "${TagBase}_rand128" -CommandLine "$Py -u _tmp_ladder.py --tag=${TagBase}_rand128 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $rand"
$o2 = & $Py -u code/test_match/analyze_paired.py --tag="${TagBase}_rand128" --seeds="$spec" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/${TagBase}_rand128/_analysis.txt", $o2, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $o2
Write-Host "[eval $TagBase] done $(Get-Date -Format 'HH:mm:ss')"
