"""核对：T2 的 B 侧配置 与 A4（vs rule_v4 = 63-1-0 那个）是否逐字段相同。"""
import re
from pathlib import Path

# A4 的 env：来自 _ab_2x2.sh 的导出 + run_arm 里的 arm 级覆盖
a4 = """
AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24
AZAI_SAMPLE_MULT=15 AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet AZAI_MC_HORIZON=256
AZAI_MC_ROOTONLY=1 AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1 AZAI_SEARCH=flat
AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64 AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000
AZAI_MC_EVERY=1
"""
# T2 的 B 侧：_ab_t1t2.sh 的 BASE + ";AZAI_MC_EVERY=1;AZAI_MC_CELLS=0"
base = Path("code/test_match/_ab_t1t2.sh").read_text(encoding="utf-8", errors="replace")
m = re.search(r'^BASE="([^"]+)"', base, re.M)
t2_b = m.group(1).replace(";", " ") + " AZAI_MC_EVERY=1 AZAI_MC_CELLS=0"


def parse(text: str) -> dict:
    return dict(kv.split("=", 1) for kv in text.split() if "=" in kv)


a, b = parse(a4), parse(t2_b)
keys = sorted(set(a) | set(b))
print(f"{'KEY':<22}{'A4':<46}{'T2( B 侧 )'}")
diff = 0
for k in keys:
    va, vb = a.get(k, "(未设→默认)"), b.get(k, "(未设→默认)")
    if k == "AZAI_CKPT":
        va = Path(va).name if va != "(未设→默认)" else va
        vb = Path(vb).name if vb != "(未设→默认)" else vb
    mark = "" if va == vb else "   ← 不同"
    if va != vb:
        diff += 1
    print(f"{k:<22}{va:<46}{vb}{mark}")
print(f"\n不同的字段数 = {diff}")
print("⚠️ AZAI_MC_CELLS 在 A4 里是未设（默认 0），T2 显式写 0 ⇒ 等价" if diff == 1 and "AZAI_MC_CELLS" in a == False else "")
