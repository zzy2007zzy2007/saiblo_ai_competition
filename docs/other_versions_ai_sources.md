# 其他版本ai 的来源与取舍（2026-09-19 整理）

## 为什么写这个

在把仓库推送到 https://git.tsinghua.edu.cn/zengzy25/saiblo_ai_competition 时发现
`其他版本ai/` 目录实占 **1.37GB**，其中绝大部分是日志 / 二进制 / 构建产物 /
别人的仓库完整副本。为了既能保留参考代码、又不把仓库撑爆，做了如下取舍，
这里记录清楚，免得以后忘了哪个目录被忽略、为什么被忽略。

## 三个嵌入式仓库（2026-09-25 起已登记为真子模块）

这三个子目录**本身就是别人仓库的完整 clone（内含 .git）**。一开始是把它们整目录忽略掉
（因为 git 遇到这种目录只会记成一个 gitlink，既不能真正 clone 下来、内容也进不了本仓库），
2026-09-25 改成正式的**子模块**——和顶层 `Ant-Game` 一样：超级仓库里只存一个指针，
内容按 url 现拉，仓库体积不受影响。

| 本地路径 | url（写在 `.gitmodules`） | 钉住的提交 | 与上游关系 |
| --- | --- | --- | --- |
| `其他版本ai/saiblo-30th-AI/` | https://github.com/omegafantasy/saiblo-30th-AI.git | `eee5619` (2026-06-25, "update") | = 上游 `main` 尖端 ✅ |
| `其他版本ai/ant-war2-magica-v3/` | https://gitee.com/ylf11235/ant-war2-magica-v3.git | `f33534d` (2026-05-17, "first commit") | = 上游 `master` 尖端 ✅ |
| `其他版本ai/Antwat_2/Ant-Game/` | https://github.com/LorenzLorentz/Ant-Game.git | `0a6bee4` (2026-04-19, "fix: mcts and training logic") | = 上游 `main` 尖端 ✅ |

三个钉住的提交都确定在上游（都 fetch 核实过），所以 `git clone --recurse-submodules` 能正常初始化。
这点和顶层 `Ant-Game` 不同：那个钉的是 `ef9e653`，比上游多两个本地提交，上游根本没有，
所以当时只能把 pin 对齐到上游的 `0a6bee4`。

别人 clone 之后要初始化子模块：

```bash
git clone --recurse-submodules <url>        # 或者 clone 完再：
git submodule update --init --recursive
```

**注意**：这三个目录里各有若干**未提交的本地改动**（2 / 6 / 4 处，多是跑评测时留下的临时改动），
子模块指针不包含这些改动，只留在本地工作区。要留证据就先
`git -C <该目录> diff > 补丁文件` 存一份。

另外顶层 `Ant-Game` 的工作区 HEAD 是 `ef9e653`，而 pin 是 `0a6bee4`，所以 `git submodule status`
会显示 `+`（工作区与 pin 不一致）。**别随手跑 `git submodule update`** —— 它会把 `Ant-Game`
检出到 `0a6bee4`，把本地那两个 README/截图提交从工作区拿掉（分支 `main` 本身还在，没丢）。

## 保留进仓库的部分（源码，约 9.8MB）

- `其他版本ai/rule_v4/`：自己写的 rule_v4（`ai_decisions.log` 单独忽略）
- `其他版本ai/Antwat_2/`：除 `Ant-Game/` 外的部分，含 `baselines/gen99/SDK` 等
- `其他版本ai/rule_v4_mcts_lightning/`：本来就已经在版本控制里

## 忽略规则（见仓库根 `.gitignore`）

三个嵌入式 clone 已经改成子模块，所以**不再整目录忽略**；`.gitignore` 里剩下的是构建产物等：

```gitignore
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
