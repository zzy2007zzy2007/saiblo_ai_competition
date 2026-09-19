# AntWar PPO 训练运维操作指南 v2.1

---

## ⚠️ 重要提示

本项目所有运维脚本均应通过 **`tools/manager.sh`** 统一调用，不要直接调用单独的脚本文件。

**错误用法**：
```bash
bash tools/deploy/start_training.sh server5 ppo_v1
```

**正确用法**：
```bash
bash tools/manager.sh --server server5 --model ppo_v1 train_start
```

使用 `tools/manager.sh` 的好处：
- 统一的命令格式和参数处理
- 更好的错误处理和日志记录
- 自动加载配置和依赖
- 支持更丰富的功能选项

---

## 1. 工具架构

```
tools/
├── manager.sh                    # 主管理入口（统一入口）
├── config/
│   └── server_config.sh        # 服务器配置
├── lib/
│   ├── config_helper.sh        # 配置管理
│   ├── ssh_helper.sh           # SSH 操作
│   ├── python_env.sh           # Python 环境
│   └── status_manager.sh       # 状态管理
├── deploy/
│   ├── deploy_code.sh          # 部署代码
│   ├── deploy_full.sh          # 完整部署
│   ├── start_training.sh       # 启动训练
│   ├── stop_training.sh        # 停止训练
│   └── resume_training.sh      # 恢复训练
├── monitor/
│   ├── check_training_status.sh  # 训练状态
│   ├── check_gpu.sh            # GPU 监控
│   ├── check_system.sh         # 系统监控
│   └── train_progress_check.sh  # 综合进度检查
├── file/
│   ├── upload_file.sh          # 上传文件
│   ├── download_file.sh        # 下载文件
│   └── sync_files.sh           # 同步文件
├── server/
│   ├── connect.sh              # 连接服务器
│   ├── check_status.sh         # 服务器状态
│   └── setup_cuda.sh           # CUDA 设置
└── operation_guide.md           # 本文档
```

---

## 2. 快速开始

### 2.1 常用命令速查表

| 功能 | 命令 |
|------|------|
| **启动训练** | `bash tools/manager.sh --server server5 --model ppo_v1 train_start --episodes 150000` |
| **停止训练** | `bash tools/manager.sh --server server5 --model ppo_v1 train_stop` |
| **查看状态** | `bash tools/manager.sh --server server5 --model ppo_v1 train_status` |
| **综合进度检查** | `bash tools/manager.sh --server server5 --model ppo_v1 train_progress_check` |
| **完整部署** | `bash tools/manager.sh --server server5 --model ppo_v1 deploy_full` |
| **部署代码** | `bash tools/manager.sh --server server5 --model ppo_v1 deploy_code` |
| **监控 GPU** | `bash tools/manager.sh --server server5 --model ppo_v1 monitor_gpu` |
| **监控系统** | `bash tools/manager.sh --server server5 --model ppo_v1 monitor_system` |
| **查看日志** | `bash tools/manager.sh --server server5 --model ppo_v1 exec "tail -f /path/to/train.log"` |
| **恢复训练** | `bash tools/manager.sh --server server5 --model ppo_v1 train_resume` |

### 2.2 服务器配置

服务器信息配置在 `tools/config/server_config.sh` 中：

```bash
# 服务器列表
SERVERS=("server1" "server2" "server3" "server4" "server5")

# SSH 连接信息
SERVER_IPS=("server1.example.com" ...)

# 模型目录基础路径
MODEL_BASE_DIR="/root/autodl-tmp/AntWar"
```

---

## 3. 训练管理

### 3.1 启动训练

**基本用法**：
```bash
bash tools/manager.sh --server <server> --model <model> train_start
```

**常用参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--episodes <num>` | 训练总轮次 | 10000 |
| `--batch-size <num>` | 批量大小 | 64 |
| `--n-envs <num>` | 并行环境数 | 1 |
| `--learning-rate <num>` | 学习率 | 0.0003 |
| `--gamma <num>` | 折扣因子 | 0.99 |
| `--lambda <num>` | GAE 参数 | 0.95 |

**示例**：
```bash
# 使用默认参数启动
bash tools/manager.sh --server server5 --model ppo_v1 train_start

# 指定训练轮次
bash tools/manager.sh --server server5 --model ppo_v1 train_start --episodes 150000

# 自定义学习率和批量大小
bash tools/manager.sh --server server5 --model ppo_v1 train_start --episodes 100000 --learning-rate 0.0005 --batch-size 128
```

### 3.2 停止训练

```bash
bash tools/manager.sh --server server5 --model ppo_v1 train_stop
```

### 3.3 查看训练状态

```bash
# 基本状态
bash tools/manager.sh --server server5 --model ppo_v1 train_status

# 详细模式
bash tools/manager.sh --server server5 --model ppo_v1 train_status -v
```

### 3.4 综合训练进度检查

```bash
bash tools/manager.sh --server server5 --model ppo_v1 train_progress_check
```

此命令会检查：
- 当前时间和训练时长
- Screen 会话状态
- 训练进程状态
- 训练指标（奖励、损失、步数）
- 综合训练状态
- 时间统计
- 系统资源采样
- GPU 状态
- 对战统计
- 异常记录

### 3.5 恢复训练

```bash
bash tools/manager.sh --server server5 --model ppo_v1 train_resume
```

### 3.6 查看训练日志

```bash
# 查看最近日志
bash tools/manager.sh --server server5 --model ppo_v1 exec "tail -100 /root/autodl-tmp/AntWar/ppo_v1/train.log"

# 实时跟踪日志
bash tools/manager.sh --server server5 --model ppo_v1 exec "tail -f /root/autodl-tmp/AntWar/ppo_v1/train.log"
```

---

## 4. 部署管理

### 4.1 完整部署

部署代码、SDK、配置文件到远程服务器：

```bash
bash tools/manager.sh --server server5 --model ppo_v1 deploy_full
```

**完整部署步骤**：
| 步骤 | 说明 |
|------|------|
| 1/4 | 部署 Ant-Game SDK 目录 |
| 2/4 | 部署 PPO 源码目录 |
| 3/4 | 部署 requirements.txt |
| 4/4 | 安装 Python 依赖 |

### 4.2 部署代码

仅部署代码文件（不包含依赖安装）：

```bash
bash tools/manager.sh --server server5 --model ppo_v1 deploy_code
```

---

## 5. 资源监控

### 5.1 GPU 监控

```bash
# 单次检查
bash tools/manager.sh --server server5 --model ppo_v1 monitor_gpu

# 持续监控（每秒刷新）
bash tools/manager.sh --server server5 --model ppo_v1 monitor_gpu --watch
```

**输出示例**：
```
==========================================
        GPU 监控 - server5
==========================================
时间: 2026-05-04 16:20:00

【GPU 0】NVIDIA GeForce RTX 5090
  使用率:     22%
  显存:       1,161 MB / 32,607 MB (3.6%)
  温度:       40°C
==========================================
```

### 5.2 监控系统

```bash
# 单次检查
bash tools/manager.sh --server server5 --model ppo_v1 monitor_system

# 持续监控
bash tools/manager.sh --server server5 --model ppo_v1 monitor_system --watch
```

**输出示例**：
```
==========================================
        系统资源监控 - server5
==========================================
时间: 2026-05-04 16:20:00

【CPU 信息】
  CPU 使用率:     7.3%
  核心数:         208

【内存信息】
  物理内存:       60 GB / 754 GB (8.0%)

【磁盘信息】
  根分区:         146 MB / 30 GB (1%)

【系统负载】
  1分钟:          5.73
  5分钟:          6.61
  15分钟:         7.60
==========================================
```

### 5.3 服务器状态

```bash
bash tools/manager.sh --server server5 server_status
```

检查项包括：
- SSH 连接状态
- GPU 状态
- 训练进程状态
- 磁盘空间
- Python 环境
- CUDA 版本
- 关键目录存在性

---

## 6. 文件管理

### 6.1 上传文件

```bash
# 上传单个文件
bash tools/manager.sh --server server5 file_upload ./train.py /root/autodl-tmp/AntWar/ppo_v1/ppo/train.py

# 上传多个文件
bash tools/manager.sh --server server5 file_upload ./config.yaml /root/autodl-tmp/AntWar/ppo_v1/config.yaml
```

### 6.2 下载文件

```bash
# 下载单个文件
bash tools/manager.sh --server server5 file_download /root/autodl-tmp/AntWar/ppo_v1/train.log ./train.log

# 下载检查点
bash tools/manager.sh --server server5 file_download /root/autodl-tmp/AntWar/ppo_v1/ppo/checkpoint_1000.pt ./checkpoint_1000.pt
```

### 6.3 执行远程命令

```bash
# 执行简单命令
bash tools/manager.sh --server server5 exec "ps aux | grep python"

# 执行复杂命令
bash tools/manager.sh --server server5 exec "cd /root/autodl-tmp/AntWar/ppo_v1 && ls -la ppo/outputs/"
```

### 6.4 文件同步

```bash
bash tools/manager.sh --server server5 file_sync ./src /root/autodl-tmp/AntWar/ppo_v1/src
```

---

## 7. 服务器管理

### 7.1 连接服务器

```bash
# SSH 交互式连接
bash tools/manager.sh --server server5 server_connect
```

### 7.2 设置 CUDA 环境

```bash
# 使用默认 CUDA 版本
bash tools/manager.sh --server server5 server_setup_cuda

# 指定 CUDA 版本
bash tools/manager.sh --server server5 server_setup_cuda 11.8
```

---

## 8. 训练配置说明

### 8.1 配置文件位置

训练配置位于：`ppo/src/ppo_antwar/configs/ppo_antwar.yaml`

### 8.2 主要配置项

| 配置节 | 说明 |
|--------|------|
| `ppo` | PPO 算法核心参数（学习率、gamma、clip 等） |
| `training` | 训练流程参数（总轮次、评估间隔等） |
| `selfplay` | Self-Play 参数（对手池、探索率等） |
| `league` | League 训练参数（衰减、最小对局数等） |
| `network` | 神经网络结构参数 |
| `env` | 环境配置 |
| `system` | 系统配置（设备、随机种子等） |

详细的参数说明请参阅：`ppo/docs/design/ppo_antwar_config.md`

---

## 9. 日志与检查点

### 9.1 日志目录结构

```
/root/autodl-tmp/AntWar/ppo_v1/
├── train.log                    # 训练主日志
├── ppo/
│   ├── checkpoint_*.pt         # 模型检查点
│   └── outputs/
│       └── {timestamp}/
│           ├── training/
│           │   ├── training_metrics.json    # 训练指标
│           │   ├── training_details.log     # 训练详情
│           │   └── training_start.log       # 启动信息
│           └── system/
│               └── system_metrics.json       # 系统监控
```

### 9.2 检查点说明

- **命名规则**: `checkpoint_{episode}.pt`
- **保存间隔**: 默认 5000 episodes
- **加载恢复**: 使用 `train_resume` 命令恢复训练

---

## 10. 常见问题排查

### 10.1 训练无法启动

**问题**：训练进程启动失败

**排查步骤**：
```bash
# 1. 检查 Python 环境
bash tools/manager.sh --server server5 exec "which python3 && python3 --version"

# 2. 检查依赖包
bash tools/manager.sh --server server5 exec "pip list | grep torch"

# 3. 检查 CUDA 可用性
bash tools/manager.sh --server server5 exec "python3 -c 'import torch; print(torch.cuda.is_available())'"

# 4. 检查磁盘空间
bash tools/manager.sh --server server5 monitor_system

# 5. 查看错误日志
bash tools/manager.sh --server server5 exec "cat /root/autodl-tmp/AntWar/ppo_v1/train.log | tail -100"
```

### 10.2 GPU 不可用

**问题**：`CUDA error: device not available`

**排查步骤**：
```bash
# 1. 检查 nvidia-smi
bash tools/manager.sh --server server5 monitor_gpu

# 2. 检查 CUDA 驱动
bash tools/manager.sh --server server5 exec "nvidia-smi"

# 3. 检查 PyTorch CUDA
bash tools/manager.sh --server server5 exec "python3 -c 'import torch; print(torch.version.cuda)'"
```

### 10.3 训练状态显示 stopped

**问题**：训练进程在运行但状态显示 stopped

**原因**：训练使用 nohup 启动，未使用 screen session

**验证方法**：
```bash
# 检查实际进程
bash tools/manager.sh --server server5 exec "ps aux | grep train.py"

# 检查日志
bash tools/manager.sh --server server5 exec "tail -20 /root/autodl-tmp/AntWar/ppo_v1/train.log"
```

### 10.4 内存不足

**问题**：训练过程内存占用过高

**排查**：
```bash
bash tools/manager.sh --server server5 monitor_system
```

观察内存使用率，如果持续高于 90%，考虑：
- 减小 `batch_size`
- 减小 `n_envs`

### 10.5 梯度爆炸

**问题**：日志中出现 `梯度范数过大` 警告

**处理**：
- 当前 `max_grad_norm` 设置为 1.0，梯度裁剪已启用
- 如果持续出现，检查学习率是否过高
- 建议降低学习率尝试

---

## 11. 附录

### 11.1 服务器访问信息

| 服务器 | 用途 | 路径 |
|--------|------|------|
| server5 | 主要训练服务器 | `/root/autodl-tmp/AntWar/ppo_v1` |

### 11.2 远程服务器 Python 环境

```bash
# Python 路径
/root/miniconda3/bin/python3

# 推荐的 PYTHONPATH 设置
/root/autodl-tmp/AntWar/ppo_v1/ppo/src:/root/autodl-tmp/AntWar/ppo_v1/Ant-Game
```

### 11.3 联系方式

如遇问题，请检查：
1. 训练日志：`/root/autodl-tmp/AntWar/{model}/train.log`
2. 系统监控数据：`/root/autodl-tmp/AntWar/{model}/ppo/outputs/{timestamp}/system/`
3. 训练指标：`/root/autodl-tmp/AntWar/{model}/ppo/outputs/{timestamp}/training/`

---

## 12. 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v2.1 | 2026-05-05 | 统一命令格式，去掉 category 前缀 |
| v2.0 | 2026-05-04 | 适配新部署脚本结构，添加完整部署流程 |
| v1.0 | 2026-01-01 | 初始版本 |
