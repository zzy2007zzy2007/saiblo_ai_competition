我(用户)是一个大二学生，自学过机器学习，但是没有过专业的训练，有些想法不一定对,或者有些想法可能是我自己想的野路子，如果有错可以纠正一下，或者说我的想法也不一定靠谱
代码改完之后都用git保存一下, 直接 `git commit`，不要用 `-c user.name=agent -c user.email=agent@local` 之类的方式覆盖作者——那样提交在 GitHub/远程上会显示成识别不出来的作者。用仓库已配置的身份（`zzy2007zzy2007 <263590731+zzy2007zzy2007@users.noreply.github.com>`）。可以标注"这次是 agent 做的"，用 trailer 写在提交信息末尾，并且带上当前会话 id（从环境变量读，不要手写死）：`git commit -m "<信息>" --trailer "Co-authored-by: dsh-agent <agent@local>" --trailer "Session-Id: $env:DSH_SESSION_ID"`。几点注意：`--trailer` 可以给多次；bash 里变量写成 `$DSH_SESSION_ID`；subagent 子会话的 id 和父会话不同，各写各的；如果这个变量为空（不是在 DSH 会话里跑的环境）就把 `Session-Id` 那条去掉，不要写空值
如果是 subagent 提交的，再加一条 `Session-Parent: <父会话 id>`：父 id 不会出现在 `$env:` 里（没有 `DSH_PARENT_SESSION_ID` 这种变量），只有派活的父 agent 知道，所以要么父 agent 在任务描述里写明自己的 session id，要么从自己 prompt 里 harness 注入的那句 `Your parent agent id is ...` 抄。顶层会话没有父，不写这条；任务里没给父 id 就省略，不要自己编
想查某个会话（比如某次自动训练，或某个 subagent 干过的活）都改了什么：`git log --format='%h %s %(trailers:key=Session-Id,valueonly) %(trailers:key=Session-Parent,valueonly)'`
电脑有 gpu,conda activate pytorch-gpu 激活环境就能用
电脑cpu16核心，不要同时开多个多进程并行、总进程数大于16的多个程序。单个测试16进程可以。还有秉性数也别太少要不然太慢，8-12进程之间比较合适
游戏规则在Ant-Game\README.md
Ant-Game这是官方提供的代码仓库，有样例程序和游戏引擎
游戏引擎有好几个版本，比如官方的python引擎、c++引擎，和我们重新pybind包装的c++引擎，官方python引擎实现的有些问题。c++引擎是对的，但官方c++引擎似乎还有一个编译好的版本也有些毛病，以官方c++引擎源码为准
code\ 目录下写代码
docs\ 目录下写文档
写新代码或者对代码做一些比较大的修改前都先写一个计划文档
如果有一些重大的或者关键的发现和想法，也可以写一个文档记录
比较大或比较重要的实验（训练、对战测试、采集等）用 `bash code/run_logged.sh <实验名> <命令...>` 跑，它会自动记录 git commit/dirty、命令、输出日志并追加到 docs/experiment_log.md，跑完补上 result，这样方便复现。小的一次性命令不用
写程序的时候可以尽量多打印日志，还有日志原始输出最好输出到文件里
code\my_ai 这里放主要的ai实现和训练代码
code\test_match 这里放测试用的代码，比如测试ai的性能
把关键的 checkpoint 拷进checkpoints/，并且在 docs\checkpoints.md 记录相关信息
时间很充裕，小时级甚至更长的十小时级的测试或者训练是可以接受的
项目里可能有一些文档比如下一步计划看起来像近期的，但实际可能并不是，是历史中遗留的，用文档中信息的时候最好确认一下时间和来源，避免使用过期的信息
F:\2026智能体大赛_新2 这里是用来备份的，短期用不上的东西可以放这里面相同的路径下节省D盘空间
开subagent如果能指定模型建议用deepseel flash