# _tmp_awr_class.ps1 —— 计划 `docs/plan_20261015_explore_awr_class.md` 第 2-3 步
#   打标签（pinned）→ AWR 训练类头（只改 class_state）→ 逐张量核对 → 报告训练读数
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Data = 'training_history/vprior/data_explore_80',
    [string]$Out = 'training_history/vprior/posnet_awrcls.pt',
    [int]$Epochs = 10,
    [string]$Beta = '0.5'
)
Set-Location $Repo
$n = (Get-ChildItem $Data -Filter '*.pkl' -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "[awr] 数据 $Data : $n 个 pkl"
if ($n -lt 20) { Write-Host '[awr] 数据太少（<20 局），先别训'; exit 1 }

$exec = "$Data" + '_exec'
& $Py -u code/my_ai/az_intent/ingest_selfplay_classes.py --src $Data --dst $exec --label pinned
if ($LASTEXITCODE -ne 0) { Write-Host '[awr] 打标签失败'; exit 1 }

& "$Repo/code/run_logged.ps1" -Name awrcls_train -CommandLine "$Py -u code/my_ai/az_intent/train_class_outcome.py --ckpt checkpoints/posnet_A_k5_m32.pt --data $exec --out $Out --epochs $Epochs --beta $Beta --class-weight-power 0.5"

# --- 单变量核对 ---
@'
import sys, hashlib, torch
def flat(sd, prefix=""):
    out = {}
    for k, v in sd.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict): out.update(flat(v, kk + "."))
        else: out[kk] = v
    return out
def sub(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = flat(sd); res = {}
    for tag in ("class_state", "pos_state", "value_state"):
        ks = sorted(k for k in sd if k.startswith(tag)); h = hashlib.md5()
        for k in ks:
            h.update(k.encode()); v = sd[k]
            h.update(v.detach().cpu().numpy().tobytes() if hasattr(v, "detach") else repr(v).encode())
        res[tag] = h.hexdigest()[:12] if ks else "-"
    return res
a = sub("checkpoints/posnet_A_k5_m32.pt"); b = sub(sys.argv[1])
print("A(现用)   :", a); print("AWR(训练后):", b)
ok = (a["pos_state"] == b["pos_state"] and a["value_state"] == b["value_state"] and a["class_state"] != b["class_state"])
print("=> 只有 class_state 变？", "是 OK（单变量成立）" if ok else "**否**")
'@ | Set-Content -Encoding UTF8 _tmp_h20.py
& $Py -u _tmp_h20.py $Out
Remove-Item _tmp_h20.py
Write-Host "[awr] 产物 $Out"
