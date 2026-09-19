#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="server5"
MODEL="ppo_v1"

# 复制测试脚本到服务器
bash "$SCRIPT_DIR/tools/manager.sh" --server "$SERVER" --model "$MODEL" exec \
    "cat > /tmp/test_baseline_simple.py << 'PYTHON_EOF'
$(cat "$SCRIPT_DIR/test_baseline_simple.py")
_EOF_"

# 运行测试脚本
bash "$SCRIPT_DIR/tools/manager.sh" --server "$SERVER" --model "$MODEL" exec "
cd /tmp && /root/miniconda3/bin/python /tmp/test_baseline_simple.py
"
