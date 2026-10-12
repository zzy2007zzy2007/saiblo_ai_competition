#!/bin/bash
# =========================================================================
# 多专家串行训练脚本 v2
# 依次训练专家A→B→C→D，每个 50 代
# 启动前自动清理残留的 Python 进程
# =========================================================================

BASE_DIR="/d/2026智能体大赛_新2"
PYTHON="/d/anaconda3/envs/pytorch-gpu/python.exe"
# 每个专家 50 代，~8 分/代 → ~7 小时/专家
COMMON_OPTS="--pop-size 64 --games 16 --workers 10 --small --bc --bc-k 3 --save-every 10 --generations 50"
INIT="--load-bc distill_data_v2/model_all_bal.pt"

cd "$BASE_DIR"

# ── 启动前清理 ──────────────────────────────────────────────
echo "[$(date)] 清理残留 Python 进程..."
taskkill //F //IM python.exe 2>/dev/null
sleep 3
remaining=$(tasklist //NH //FI "IMAGENAME eq python.exe" 2>/dev/null | grep python | wc -l)
echo "[$(date)] 残留 Python 进程数: ${remaining}（应为 0）"

echo "============================================================"
echo "  多专家串行训练启动 at $(date)"
echo "============================================================"

run_expert() {
    local name="$1"
    local desc="$2"
    local classes="$3"
    local probs="$4"
    local out_dir="$5"
    shift 5

    echo ""
    echo "========================================================================"
    echo "  ${name} 开始训练 at $(date)"
    echo "  策略: ${desc}"
    echo "  动作类: ${classes}"
    echo "  目录: ${out_dir}"
    echo "========================================================================"

    PYTHONIOENCODING=utf-8 "$PYTHON" -u code/my_ai/es_train.py \
        $COMMON_OPTS $INIT \
        --allowed-classes $classes \
        --opp-probs $probs \
        --out-dir "$out_dir"

    local exit_code=$?
    echo "[$(date)] ${name} 结束, exit code=${exit_code}"
    return $exit_code
}

set +e  # 一个专家失败不影响后续

# ── 专家 A: Thunder + 基地升级 ──────────────────────────────
run_expert \
    "专家A" "Thunder + 基地升级" \
    "17 21 22 23" "0.1 0.4 0.5" \
    "training_history/expert_a_v2"
EXPERT_A_EXIT=$?

# ── 专家 B: 纯塔攻防 ────────────────────────────────────────
run_expert \
    "专家B" "纯塔攻防 (0-16 + HOLD)" \
    "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 23" \
    "0.3 0.4 0.3" \
    "training_history/expert_b"
EXPERT_B_EXIT=$?

# ── 专家 C: 辅助武器 ────────────────────────────────────────
run_expert \
    "专家C" "辅助武器 + 基地升级" \
    "18 19 20 21 22 23" "0.5 0.4 0.1" \
    "training_history/expert_c"
EXPERT_C_EXIT=$?

# ── 专家 D: 产兵运营 ────────────────────────────────────────
run_expert \
    "专家D" "产兵运营" \
    "4 13 14 15 16 23" "0.6 0.3 0.1" \
    "training_history/expert_d"
EXPERT_D_EXIT=$?

# ── 汇总 ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  全部训练完成 at $(date)"
echo "============================================================"
echo "  专家A exit=${EXPERT_A_EXIT}"
echo "  专家B exit=${EXPERT_B_EXIT}"
echo "  专家C exit=${EXPERT_C_EXIT}"
echo "  专家D exit=${EXPERT_D_EXIT}"
echo "============================================================"

echo "Completed at $(date)" > training_history/ALL_EXPERTS_DONE.txt
echo "A=${EXPERT_A_EXIT} B=${EXPERT_B_EXIT} C=${EXPERT_C_EXIT} D=${EXPERT_D_EXIT}" >> training_history/ALL_EXPERTS_DONE.txt
