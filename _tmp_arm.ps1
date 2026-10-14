# _tmp_arm.ps1 —— 单臂驱动：冒烟守卫 → 128 局 → 分析（供逐臂调度，避免一次拉起 7 臂）
#   用法: powershell -File _tmp_arm.ps1 -Name B_clsonly -Ckpt checkpoints/grafted_clsonly.pt `
#             -Mode pos-only -Pin mc_quiet -Tclass 0.5 -Opp example -Jobs 8
param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Ckpt,
    [string]$Mode = 'pos-only',
    [string]$Pin = 'mc_quiet',
    [string]$Tclass = '0.5',
    [string]$Tpos = '1.0',        # 位置采样温度（**除法**约定：exp(logits/t) ⇒ 越大越平；1000 ≈ 均匀 ⇒ '无先验'对照）
    [string]$Temp = '1e-6',          # 决策温度：1e-6 ⇒ 取 visits 的 argmax（现状）；1.0 ⇒ 按 visits 分布**采样**（= "教师作为分布" 的强度测量）
    [string]$Opp = 'example',            # 'example' 或 'baseline'
    [int]$Jobs = 8,
    [string]$K = '24',            # 搜索候选数（搜索超参扫描用）
    [string]$Skip1 = '1',         # 1 = 单候选短路（旧行为）；0 = **不短路**（路 C 的 C1 需要）
    [string]$ClassFeasMask = '0', # 1 = 启用 C1.1 类采样前的可行性掩蔽（默认关，旧行为逐位不变）
    [string]$VisitRank = '1',     # 名次实验（用户 2026-10-12 设计）：取 visit 排序第 r 名（1 = argmax = 旧行为 ✓）
    [hashtable]$ExtraEnv = @{},   # 额外/覆盖 AZAI_*（如 AZAI_DUMP_RAW_DIR / AZAI_AZ_DUMP / AZAI_MODE=joint）
    [string]$Iters = '256',       # MCTS 迭代
    [string]$Depth = '4',         # 束展开深度
    [string]$SampleMult = '15',
    [string]$SeedSpec = '7-70',   # 可只跑前缀筛查（如 7-38 = 64 局）
    [string]$McEvery = '5',       # MC 仲裁频率（MC 覆盖度扫描：1 = 每个决策都仲裁）
    [string]$McTopk = '3',        # 经济类菜单宽度
    [string]$McCells = '0',       # 菜单里加入"前 N 个格子"
    [string]$McHorizon = '256',   # 安静 rollout 视野
    [string]$McScore = '',        # 评分模式（'' = 安静 rollout；value_only/delta = 价值网）
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
# ⚠️ 2026-10-19 修：`param()` 默认值里的 `$PSScriptRoot` 在某些调用方式下是空的
#   ⇒ 导致 `$Repo` 为空、"$Repo/code/run_logged.ps1" 变成 "/code/..."（B 臂因此没跑成 128 局）
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
if (-not $Repo) { $Repo = (Get-Location).Path }
Set-Location $Repo
# ⚠️ 2026-10-12：子进程 stdout 统一 UTF-8（否则中文标签在控制台/日志里会乱码 ✗；数字不受影响但可读性差）
$env:PYTHONIOENCODING = 'utf-8'
# 2026-10-12 补：**控制台**输出编码也设成 UTF-8 —— 否则 `Select-String -Pattern '配对胜率'` 匹配不上
#   ⇒ 臂跑完只打印 [mech] 行、**分析行（p̂/CI）不显示**（数字没丢，可用 analyze_paired.py 重取 ✓）
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:AZAI_CKPT = $Ckpt
$env:AZAI_ITERS=$Iters; $env:AZAI_DEPTH=$Depth; $env:AZAI_K=$K; $env:AZAI_SAMPLE_MULT=$SampleMult
$env:AZAI_MODE=$Mode; $env:AZAI_SKIP1=$Skip1; $env:AZAI_TPOS=$Tpos; $env:AZAI_TEMP=$Temp; $env:AZAI_VISIT_RANK=$VisitRank
$env:AZAI_CLASS_FEAS_MASK=$ClassFeasMask
foreach ($k in $ExtraEnv.Keys) { Set-Item -Path ("Env:" + $k) -Value $ExtraEnv[$k]; Write-Host "[arm] ExtraEnv $k=$($ExtraEnv[$k])" }
$env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'; $env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='25'
$env:AZAI_POSPIN=$Pin; $env:AZAI_TCLASS=$Tclass
$env:AZAI_MC_HORIZON=$McHorizon; $env:AZAI_MC_EVERY=$McEvery; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'
$env:AZAI_MC_SCORE=$McScore; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_SELFSPEND_MIX='1.0'
$env:AZAI_MC_BIDIR='0'; $env:AZAI_MC_NO_LIGHT='0'; $env:AZAI_MC_CELLS=$McCells; $env:AZAI_MC_TOP2='0'
$env:AZAI_MC_M4='0'; $env:AZAI_MC_NODOWN='0'; $env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK=$McTopk
$env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'
$env:AZAI_FORCE_UPG='0'; $env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'
$env:AZAI_SLOW_DECISION_SEC='60'
$env:AZAI_BAN_CLASSES='17'; $env:AZAI_BAN_OPS='21'; $env:AI_BAN_LIGHTNING='1'

$oppPath = if ($Opp -eq 'example') { 'code/test_match/ex_pkg/main.py' } else { 'code/test_match/base_pkg/main.py' }
$tag = "AB_${Name}_vs_$Opp"
Write-Host ("=== {0} (ckpt={1} mode={2} pin={3} tclass={4} K={5} ITERS={6} DEPTH={7} seeds={8} opp={9} jobs={10}) smoke {11}" -f `
    $Name,$Ckpt,$Mode,$Pin,$Tclass,$K,$Iters,$Depth,$SeedSpec,$Opp,$Jobs,(Get-Date -Format 'HH:mm:ss'))
& $Py -u _tmp_ladder.py --tag="SMK_$tag" --jobs=2 --ai0=code/test_match/az_bridge_ai.py --ai1=$oppPath 7 7r 2>&1 | Out-Null
$line = & $Py -u code/test_match/phase0_report.py "SMK_$tag" 2>&1 | Select-Object -Skip 1 -First 1
$m = [regex]::Match($line, '^\S+\s+\d+\s+\S+\s+([\d.]+)/')
$b = if ($m.Success) { [double]$m.Groups[1].Value } else { -1.0 }
Write-Host ("  [smoke] 我方建塔/局 = {0}   ({1})" -f $b, $line.Trim())
if ($b -le 0.5) { Write-Host ("  [SKIP] {0} 冒烟即退化 ⇒ 不跑 128 局" -f $Name); exit 0 }
$lo, $hi = $SeedSpec.Split('-')[0], $SeedSpec.Split('-')[1]
$tokens = (( [int]$lo..[int]$hi | ForEach-Object { "$_", "${_}r" }) -join ' ')
$seedSpec = (( [int]$lo..[int]$hi) -join ',')
& "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=$Jobs --ai0=code/test_match/az_bridge_ai.py --ai1=$oppPath $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=$seedSpec 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
$out -split "`n" | Select-String -Pattern '配对胜率|完整配对数|有效性门槛|判词来源' | ForEach-Object { "  " + $_.Line.Trim() }
Write-Host ("  [mech] " + (& $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1))
Write-Host ("=== {0} done {1}" -f $Name,(Get-Date -Format 'HH:mm:ss'))
