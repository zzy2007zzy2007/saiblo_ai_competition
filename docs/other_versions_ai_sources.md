# 其他版本ai 的来源与取舍（2026-09-19 整理）

## 为什么写这个

在把仓库推送到 https://git.tsinghua.edu.cn/zengzy25/saiblo_ai_competition 时发现
`其他版本ai/` 目录实占 **1.37GB**，其中绝大部分是日志 / 二进制 / 构建产物 /
别人的仓库完整副本。为了既能保留参考代码、又不把仓库撑爆，做了如下取舍，
这里记录清楚，免得以后忘了哪个目录被忽略、为什么被忽略。

## 三个嵌入式仓库（整目录忽略，需要时重新 clone）

这三个子目录**本身就是别人仓库的完整 clone（内含 .git）**。git 遇到这种目录只会
记成一个 gitlink（伪子模块），既不能真正 clone 下来、内容也进不了本仓库，
所以直接整目录忽略；要恢复就按下面的 origin 重新 clone 到原路径：

| 本地路径 | origin | 抓取时的 HEAD | 备注 |
| --- | --- | --- | --- |
| `其他版本ai/saiblo-30th-AI/` | https://github.com/omegafantasy/saiblo-30th-AI.git | `eee5619` (2026-06-25, "update") | 抓取时工作区另有 2 处未提交改动，未保留 |
| `其他版本ai/Antwat_2/Ant-Game/` | https://github.com/LorenzLorentz/Ant-Game.git | `0a6bee4` (2026-04-19, "fix: mcts and training logic") | 即官方引擎；抓取时工作区另有 4 处未提交改动，未保留 |
| `其他版本ai/ant-war2-magica-v3/` | https://gitee.com/ylf11235/ant-war2-magica-v3.git | `f33534d` (2026-05-17, "first commit") | 抓取时工作区另有 6 处未提交改动，未保留 |

注意：这三个目录里各有若干**未提交的本地改动**（多为跑评测时留下的临时改动），
忽略之后这些改动不会进仓库，重新 clone 也拿不回来。如果哪天需要，先手动
`git -C <该目录> diff > 补丁文件` 存一份。

## 保留进仓库的部分（源码，约 9.8MB）

- `其他版本ai/rule_v4/`：自己写的 rule_v4（`ai_decisions.log` 单独忽略）
- `其他版本ai/Antwat_2/`：除 `Ant-Game/` 外的部分，含 `baselines/gen99/SDK` 等
- `其他版本ai/rule_v4_mcts_lightning/`：本来就已经在版本控制里

## 忽略规则（见仓库根 `.gitignore`）

```gitignore
# 嵌入式 clone：整目录
其他版本ai/saiblo-30th-AI/
其他版本ai/ant-war2-magica-v3/
其他版本ai/Antwat_2/Ant-Game/

# 构建产物 / 模型 / 数据集 / 日志
其他版本ai/**/*.o
其他版本ai/**/*.exe
其他版本ai/**/*.npz
其他版本ai/**/*.pt
其他版本ai/**/*.zip
其他版本ai/**/models/
其他版本ai/**/build/
其他版本ai/**/output/
其他版本ai/**/logs/
其他版本ai/*/ai_decisions.log
```

## 顺带记一笔：仓库外的那些大数据目录

2026-09-19 实测，下列目录合计约 **280GB**，全部加入 `.gitignore`：

`training_history/`（263GB，之前就已忽略）、`value_data_v2/`(2.8GB)、
`value_merged/`(1.5GB)、`match_results/`(910MB)、`distill_data_v2/`(444MB)、
`value_data/`(235MB)、`distill_data_biased/`(181MB)、`data/`(167MB)、
`submission/` 未跟踪部分(167MB)、`distill_data/`(150MB)、`score_pretrained_zscore/`(112MB) 等。

**所以千万别在这个仓库里直接跑 `git add -A` / `git add .`**（加之前先看
`git status --short`）。真要提交被忽略的东西，用 `git add -f <路径>`。
