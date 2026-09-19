#!/bin/bash
# ============================================================
# 二分种子方案验证实验 — 部署到 server9
# ============================================================
# 用法:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# 环境要求:
#   - server9 可 SSH 访问
#   - server9 有 Python 3.9+ 和 pip
# ============================================================

set -euo pipefail

# ── 配置（按需修改） ──────────────────────────────────────────
SERVER="server9"
SERVER_USER="${SERVER_USER:-$(whoami)}"
REMOTE_DIR="~/bisection_experiment"
EXPERIMENT_PRESET="default"   # quick | default | full
LOCAL_CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$LOCAL_CODE_DIR/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "二分种子验证实验 — 部署到 server9"
echo "============================================================"
echo "服务器:   ${SERVER_USER}@${SERVER}"
echo "远程目录: ${REMOTE_DIR}"
echo "预设:     ${EXPERIMENT_PRESET}"
echo "时间戳:   ${TIMESTAMP}"
echo "============================================================"

# ── Step 1: 打包实验代码 ──
echo ""
echo "[1/5] 打包实验代码..."

PACK_DIR="/tmp/bisection_experiment_${TIMESTAMP}"
rm -rf "${PACK_DIR}"
mkdir -p "${PACK_DIR}"

# 复制实验模块
cp -r "${LOCAL_CODE_DIR}"/*.py "${PACK_DIR}/"

# 创建独立的依赖文件
cat > "${PACK_DIR}/requirements.txt" << 'EOF'
trueskill>=0.4
numpy>=1.24
loguru>=0.7
scipy>=1.10
EOF

chmod +x "${PACK_DIR}/run_on_server.py"

# 打包
PACK_FILE="/tmp/bisection_experiment_${TIMESTAMP}.tar.gz"
tar czf "${PACK_FILE}" -C "$(dirname "${PACK_DIR}")" "$(basename "${PACK_DIR}")"
echo "打包完成: ${PACK_FILE} ($(du -h "${PACK_FILE}" | cut -f1))"

# ── Step 2: 上传到 server9 ──
echo ""
echo "[2/5] 上传到 server9..."
ssh "${SERVER_USER}@${SERVER}" "mkdir -p ${REMOTE_DIR}"
scp "${PACK_FILE}" "${SERVER_USER}@${SERVER}:${REMOTE_DIR}/"

# ── Step 3: 解压并安装 ──
echo ""
echo "[3/5] 远程解压..."
ssh "${SERVER_USER}@${SERVER}" "cd ${REMOTE_DIR} && tar xzf $(basename "${PACK_FILE}") && ls -la $(basename "${PACK_DIR}")"

# ── Step 4: 运行实验 ──
echo ""
echo "[4/5] 远程安装依赖并运行实验..."
ssh "${SERVER_USER}@${SERVER}" "cd ${REMOTE_DIR}/$(basename "${PACK_DIR}") && pip3 install -r requirements.txt --quiet 2>&1 | tail -3 && python3 run_on_server.py --preset ${EXPERIMENT_PRESET} --output-dir outputs 2>&1 | tee outputs/run_console.log"

# ── Step 5: 拉取结果 ──
echo ""
echo "[5/5] 拉取结果..."
REMOTE_OUTPUT="${REMOTE_DIR}/$(basename "${PACK_DIR}")/outputs"
LOCAL_RESULTS="${LOCAL_CODE_DIR}/results/server9_${TIMESTAMP}"
mkdir -p "${LOCAL_RESULTS}"
scp -r "${SERVER_USER}@${SERVER}:${REMOTE_OUTPUT}/"* "${LOCAL_RESULTS}/" 2>/dev/null || true

# 也保存到项目 outputs 目录
PROJECT_OUTPUT="${PROJECT_ROOT}/ppo_v10/outputs/bisection_experiment"
mkdir -p "${PROJECT_OUTPUT}"
cp -r "${LOCAL_RESULTS}/"* "${PROJECT_OUTPUT}/" 2>/dev/null || true

echo ""
echo "============================================================"
echo "部署完成!"
echo "本地结果: ${LOCAL_RESULTS}"
echo "项目结果: ${PROJECT_OUTPUT}"
echo ""
echo "查看汇总:"
echo "  cat ${LOCAL_RESULTS}/latest_summary.json | python3 -m json.tool"
echo "============================================================"

# 清理本地临时文件
rm -rf "${PACK_DIR}" "${PACK_FILE}"
