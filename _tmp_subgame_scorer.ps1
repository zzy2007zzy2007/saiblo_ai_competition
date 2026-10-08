# _tmp_subgame_scorer.ps1 —— 预注册 docs/prereg_20261018_subgame_scorer.md
#   禁闪电子博弈（双方都禁）里扫"MC 评分器"：S0 基准 / SELFSPEND 1,2,4 / MIX 0.25,0.5 / BIDIR 2,4
#   每臂固定段全段 128 局（7..70 双向），对手 = 官方 example（AI_BAN_LIGHTNING=1），我方掩码层禁闪电。
#   判据：只看胜率。机制读数只作观测。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo

function Set-Base {
    $env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'; $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'
    $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    $env:AZAI_TIMING_EVERY = '25'
    $env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
    $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_SCORE = ''
    $env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
    $env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
    $env:AZAI_MC_CELLS = '0'; $env:AZAI_MC_TOP2 = '0'; $env:AZAI_MC_M4 = '0'; $env:AZAI_MC_NODOWN = '0'
    $env:AZAI_MC_NO_LIGHT = '0'; $env:AZAI_SLOW_DECISION_SEC = '60'
    # === 子博弈：双方都禁闪电（我方掩码层；对手 example 侧过滤）===
    $env:AZAI_BAN_CLASSES = '17'; $env:AZAI_BAN_OPS = '21'; $env:AI_BAN_LIGHTNING = '1'
    # === 基准菜单（= E2 配置）===
    $env:AZAI_MC_UPG = '1'; $env:AZAI_MC_TOPK = '3'
    # === 默认评分器旋钮（各臂按需覆盖）===
    $env:AZAI_MC_SELFSPEND = '0'; $env:AZAI_MC_SELFSPEND_MIX = '1.0'
    $env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_ROLLOUT_CLASSES = '4'
}

function Report-Arm([string]$tag) {
    foreach ($seeds in @('7-70', '7-38', '39-70')) {
        $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=$seeds 2>&1 | Out-String
        if ($seeds -eq '7-70') {
            [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
                (New-Object System.Text.UTF8Encoding($false)))
        }
        $phat = ($out -split "`n" | Select-String -Pattern '配对胜率' | Select-Object -First 1)
        $bd   = ($out -split "`n" | Select-String -Pattern '完整配对数' | Select-Object -First 1)
        Write-Host ("  [report] {0} seeds={1} | {2} | {3}" -f $tag, $seeds,
            ($(if ($phat) { $phat.Line.Trim() } else { 'p_hat=?' })),
            ($(if ($bd) { $bd.Line.Trim() } else { '' })))
    }
    $mech = & $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1
    Write-Host ("  [mech] " + $mech)
}

function Run-Arm([string]$name, [string]$selfspend, [string]$mix, [string]$bidir, [string]$classes) {
    Set-Base
    $env:AZAI_MC_SELFSPEND = $selfspend
    $env:AZAI_MC_SELFSPEND_MIX = $mix
    $env:AZAI_MC_BIDIR = $bidir
    $env:AZAI_MC_ROLLOUT_CLASSES = $classes
    $tag = "SG_$name"
    Write-Host ("=== arm {0} (selfspend={1} mix={2} bidir={3} classes={4}) start {5}" -f `
        $name, $selfspend, $mix, $bidir, $classes, (Get-Date -Format 'HH:mm:ss'))
    $tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py $tokens"
    Report-Arm $tag
    Write-Host ("=== arm {0} done {1}" -f $name, (Get-Date -Format 'HH:mm:ss'))
}

Run-Arm 'S0_quiet'      '0' '1.0' '0' '4'
Run-Arm 'S1_spend1'     '1' '1.0' '0' '4'
Run-Arm 'S2_spend2'     '2' '1.0' '0' '4'
Run-Arm 'S4_spend4'     '4' '1.0' '0' '4'
Run-Arm 'M25_sp4'       '4' '0.25' '0' '4'
Run-Arm 'M50_sp4'       '4' '0.5' '0' '4'
Run-Arm 'D2_bidir2'     '0' '1.0' '2' '12'
Run-Arm 'D4_bidir4'     '0' '1.0' '4' '12'
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
