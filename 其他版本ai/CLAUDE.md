这里放了其他版本的ai

rule_v4:我编的比较强的规则模型，比赛里14名，目前神经网络ai还没有能打过rule_v4的

ant-war2-magica-v3:这次比赛冠军的ai

saiblo-30th-AI\Game1:这次比赛亚军的ai，但似乎比冠军还强一些，不知道为什么。以及这个亚军在更早的几届比赛里连着拿过四五次冠军了。saiblo-30th-AI是冠军的ai的完整的仓库，里面包含两个游戏

AAArena:清华计算机系学生科协做的基准仓库（论文 arXiv:2610.12341），收录第24~30届+4个AI9老游戏共12个赛题、909份人类选手源码（只放出偶数名次且>8名的，前8名永远不给）。antwar2的C++后端和本地Ant-Game引擎逐字节相同。详细介绍见 AAArena说明.md

AAArena规则:12个游戏的 rules.md / decision_space.yaml / replay_format.md / replay_skill.md 已解包到 AAArena规则\ 下，直接翻不用解 tar.gz。注意 pacman/dorado/monecraft/lota 这4个 legacy_ai9 游戏只有规则摘要。

⚠️造塔价格:本地 Ant-Game\README.md 的造塔价格公式「15×2^i」是错的，正确的是 15×3^(n//2)（n为奇数再×2）。C++证据 game\src\coin.cpp:11，SDK证据 SDK\backend\constants.py:325（SDK 是对的，照 README 手算会高估成本、建塔偏保守）。详见 docs\antwar2规则勘误_造塔价格.md