# _tmp_collect_az.ps1 —— 采集 AZ 环（④）与 S2（⑤）**共用**的数据
#
#   设定：双方禁闪电（我方掩码层禁类 + 执行层禁 op；对侧用我们自己的 base_pkg）+
#         **冻结基线配置**（同一个 `grafted_sdk_A.pt`、pos-only + mc_quiet + 宽菜单）
#         ⇒ 同策略镜像自对弈（数据可用；配对标尺退化，这里不看胜负）。
#   seed 段：**训练专用**（默认 5001..5100；预注册 `prereg_20261019_goal2_az_loop.md` 写的是 5001..5400）
#         ⇒ 与判据段（7..70）/随机段/**未见段（501..564）全不重叠**；判据 seed 一律不入训。
#   两个 dump（都由 `code/test_match/az_bridge_ai.py` 写，默认关闭，这里打开）：
#         `AZAI_DUMP_RAW_DIR` -> obs_seed*_p*.bin（定长 20328 B/决策：board+stats+player+chosen+类掩码）
#         `AZAI_AZ_DUMP`      -> az_seed*_p*.jsonl（候选束 + 已解出标签 + 访问分布 + intent + root_value）
#   体量：~500 条决策/局 × 20328 B ≈ **10 MB/局** ⇒ 200 局 ≈ 2 GB（够用；不够就缩 seed 段）。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Lo = 5001,
    [int]$Hi = 5100,
    [int]$Jobs = 8,
    [string]$Tag = 'COLLECT_AZ',
    [string]$ObsDir = 'data/az_obs',
    [string]$AkDir = 'data/az_policy'
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo

# ── 冻结基线配置（与 goal 文本逐项一致；双方禁闪电）──
$env:AZAI_CKPT = 'checkpoints/grafted_sdk_A.pt'
$env:AZAI_ITERS='256'; $env:AZAI_DEPTH='4'; $env:AZAI_K='24'; $env:AZAI_SAMPLE_MULT='15'
$env:AZAI_MODE='pos-only'; $env:AZAI_POSPIN='mc_quiet'; $env:AZAI_TCLASS='0.5'
$env:AZAI_MC_EVERY='5'; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'
$env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK='3'; $env:AZAI_MC_CELLS='0'; $env:AZAI_MC_TOP2='0'
$env:AZAI_MC_HORIZON='256'; $env:AZAI_MC_SCORE=''; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_BIDIR='0'
$env:AZAI_SKIP1='1'; $env:AZAI_TPOS='1.0'; $env:AZAI_TEMP='1e-6'; $env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'
$env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'; $env:AZAI_FORCE_UPG='0'
$env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'; $env:AZAI_BAN_CLASSES='17'; $env:AZAI_BAN_OPS='21'
$env:AI_BAN_LIGHTNING='1'; $env:AZAI_SLOW_DECISION_SEC='60'
# ── 两种 dump（相对路径由桥接端按仓库根解析）──
$env:AZAI_DUMP_RAW_DIR = "$ObsDir/$Tag"
$env:AZAI_AZ_DUMP     = "$AkDir/$Tag"
$env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='250'

$tokens = (( [int]$Lo..[int]$Hi | ForEach-Object { "$_", "${_}r" }) -join ' ')
$n = 2 * ($Hi - $Lo + 1)
Write-Host ("=== {0}: seed {1}..{2} ⇒ {3} 局；dump -> {4} / {5}" -f $Tag, $Lo, $Hi, $n, $env:AZAI_DUMP_RAW_DIR, $env:AZAI_AZ_DUMP)
& "$Repo/code/run_logged.ps1" -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=$Jobs --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/base_pkg/main.py $tokens"
Write-Host ("=== 采集完成；观测文件数 = " + @(Get-ChildItem "$ObsDir/$Tag" -Filter '*.bin' -ErrorAction SilentlyContinue).Count +
            "；策略文件数 = " + @(Get-ChildItem "$AkDir/$Tag" -Filter '*.jsonl' -ErrorAction SilentlyContinue).Count)
Write-Host ("   下一步：python code/distill/convert_az_dump.py --az-dir $AkDir/$Tag --obs-dir $ObsDir/$Tag --out data/az_npz/$Tag")
