# 编译 dump_plan_prior.exe（把我们的状态 JSON 灌进亚军 generate_root_plans 取先验标签）
#
# 坑（见 docs/runnerup_build_note.md）：
#   1) MinGW 的 ld/汇编器不认中文路径 ⇒ 所有 -I、源文件、输出、TMP 都走纯 ASCII 路径。
#      → 在 D:\dsh_inc\ 下建 junction 指到中文目录。
#   2) 必须把 C:\mingw64\bin 前置到 PATH（D:\anaconda3\Library\mingw-w64\bin 的 DLL 版本不符，
#      运行时会 0xC0000139）。

$ErrorActionPreference = "Stop"

$Repo   = "D:\2026智能体大赛_新2_distill"
$SdkInc = Join-Path $Repo "其他版本ai\saiblo-30th-AI\Game1\antgame_cpp_sdk\include"
$V4Inc  = Join-Path $Repo "其他版本ai\saiblo-30th-AI\Game1\antgame_ai_cpp\cpp_lure_v4\include"
$GameInc= Join-Path $Repo "Ant-Game\game\include"
$Lib    = "D:\dsh_build_tmp\libantgame_cpp_sdk.a"
$SrcDir = Join-Path $Repo "code\my_ai\distill_top2\cpp_test"

# --- ASCII 别名的 junction ---
$IncRoot = "D:\dsh_inc"
New-Item -ItemType Directory -Force -Path $IncRoot | Out-Null
$links = @{ "sdk" = $SdkInc; "v4" = $V4Inc; "game" = $GameInc }
foreach ($name in $links.Keys) {
    $link = Join-Path $IncRoot $name
    if (-not (Test-Path $link)) {
        cmd /c mklink /J "$link" "$($links[$name])" | Out-Null
        Write-Host "[link] $link -> $($links[$name])"
    }
}
foreach ($name in $links.Keys) {
    $link = Join-Path $IncRoot $name
    if (-not (Test-Path (Join-Path $link "antgame_sdk")) -and
        -not (Test-Path (Join-Path $link "antgame_ai")) -and
        -not (Test-Path (Join-Path $link "json.hpp"))) {
        throw "junction 失效或内容缺失: $link"
    }
}

if (-not (Test-Path $Lib)) { throw "缺少静态库: $Lib（从主工作区 antgame_cpp_sdk/build 复制）" }

# --- 源文件拷到 ASCII 路径再编译 ---
$work = "D:\dsh_build_tmp"
New-Item -ItemType Directory -Force -Path $work | Out-Null
Copy-Item (Join-Path $SrcDir "dump_plan_prior.cpp") (Join-Path $work "dump_plan_prior.cpp") -Force

$env:TMP = $work
$env:TEMP = $work
$env:PATH = "C:\mingw64\bin;" + $env:PATH

$out = Join-Path $work "dump_plan_prior.exe"
Write-Host "[build] $out"
g++ -std=c++17 -O3 -DNDEBUG -w -pipe `
    "-I$IncRoot\v4" "-I$IncRoot\sdk" "-I$IncRoot\game" `
    (Join-Path $work "dump_plan_prior.cpp") $Lib -o $out
if ($LASTEXITCODE -ne 0) { throw "编译失败 exit=$LASTEXITCODE" }

Get-Item $out | Select-Object FullName, Length, LastWriteTime | Format-List
