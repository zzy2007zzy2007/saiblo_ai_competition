# _tmp_c1_arm.ps1 —— 按预注册跑"配置阶梯"里的**任意一个臂**：闸门(1局) → 筛查 → 判据统计
#
# 依据：`docs/prereg_20261005_search_config_ladder.md` §2/§3/§4。
# 本脚本**只执行预注册里写死的东西**，不做任何选择或结论；读数由 agent 写进文档。
#
# 用法（在 B3 基线跑完之后、且没有别的 ladder 在跑时）:
#   powershell -NoProfile -ExecutionPolicy Bypass -File _tmp_c1_arm.ps1 -Arm a2
#   powershell -NoProfile -ExecutionPolicy Bypass -File _tmp_c1_arm.ps1 -Arm a1 -Seeds 7-70 -TagSuffix 128
#
# 预注册的臂定义（只列与钉死值的差异；ckpt 一律 `posnet_A_k5_m32`）:
#   a1 = 历史标准   : K=24 SAMPLE_MULT=15 MODE=pos-only SKIP1=1  （闸门上限 6 min/局）
#   a2 = 厚 joint   : K=24 SAMPLE_MULT=15 MODE=joint   SKIP1=1  （闸门上限 20 min/局）
param(
    [Parameter(Mandatory = $true)][ValidateSet('a1', 'a2')][string]$Arm,
    [string]$Seeds = '',            # 空 = 用预注册的筛查样本（a1: 7-22 / a2: 7-14）
    [string]$TagSuffix = '',        # 例如 '128' ⇒ tag = C1_<arm>_<suffix>
    [int]$Jobs = 8,
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)

Set-Location $Repo

# ---- 钉死协议（章程 §2），只改本臂的那几个变量 ----
$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_SKIP1 = '1'
switch ($Arm) {
    'a1' { $env:AZAI_MODE = 'pos-only'; $gateMax = 6;  $defSeeds = '7-22'; $expectGames = 32 }
    'a2' { $env:AZAI_MODE = 'joint';    $gateMax = 20; $defSeeds = '7-14'; $expectGames = 16 }
}
if (-not $Seeds) { $Seeds = $defSeeds }
$parts = $Seeds -split '-'
$seedList = [int]$parts[0]..[int]$parts[1]
$tokens = ($seedList | ForEach-Object { "$_", "${_}r" }) -join ' '
$expectGames = $seedList.Count * 2

$tag = "C1_${Arm}_screen$($Seeds -replace '-','_')"
if ($TagSuffix) { $tag = "C1_${Arm}_$TagSuffix" }
$gateTag = "C1_${Arm}_gate"
$runner = "$Repo/code/run_logged.ps1"
$py = $Py

function Latest-RunDir([string]$name) {
    @(Get-ChildItem "$Repo/training_history/runs" -Directory |
        Where-Object { $_.Name -like "*_$name" } | Sort-Object Name -Descending)[0].FullName
}

Write-Host "[c1-$Arm] MODE=$($env:AZAI_MODE) K=$($env:AZAI_K) SM=$($env:AZAI_SAMPLE_MULT) SKIP1=$($env:AZAI_SKIP1)"
Write-Host "[c1-$Arm] 闸门 tag=$gateTag  筛查 tag=$tag  局数=$expectGames  seeds=$Seeds  jobs=$Jobs"

# ---------- 1) 成本闸门：1 局 ----------
& $runner -Name $gateTag -CommandLine "$py -u _tmp_ladder.py --tag=$gateTag --jobs=1 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 7"
$gateLog = Get-Content -Encoding UTF8 "$(Latest-RunDir $gateTag)/output.log" -ErrorAction SilentlyContinue
$mins = $null
foreach ($ln in $gateLog) { if ($ln -match '^\[7\] rc=(-?\d+)\s+([\d.]+) min') { $mins = [double]$Matches[2] } }
$bad = @($gateLog | Select-String -Pattern 'winner=INVALID' -Quiet).Count -gt 0
Write-Host "[c1-$Arm] 闸门用时 = $mins min（上限 $gateMax）INVALID=$bad"
if ($null -eq $mins -or $mins -gt $gateMax -or $bad) {
    Write-Host "[c1-$Arm] ⛔ 闸门未过 ⇒ 按预注册 §3 停止，不跑筛查。"
    exit 0
}

# ---------- 2) 筛查 ----------
Write-Host "[c1-$Arm] 筛查开始 $(Get-Date -Format 'HH:mm:ss')"
& $runner -Name $tag -CommandLine "$py -u _tmp_ladder.py --tag=$tag --jobs=$Jobs --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"

# ---------- 3) 判据统计 ----------
$out = & $py -u code/test_match/analyze_paired.py --tag=$tag --seeds=$Seeds `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[c1-$Arm] 完成 $(Get-Date -Format 'HH:mm:ss')"
