# 长期任务的目标文本（goal objective）—— 乱码已还原

> 2026-10-05。DSH 里这个长任务的 **goal objective 被一次编码事故写成了乱码**：
> 原文是中文 UTF-8，被当成 **Windows CP936(GBK)** 解码。本文件保存**还原后的正确文本**；
> 以后每轮**以本文件的文本为准**，不要再去读活 goal 里那段乱码。
>
> 还原工具：`code/tools/decode_cp936_mojibake.py`（可复用；原理见下 §3）

## 1. 正确文本（= 现在活 goal objective 的内容）

按 `docs/task_ruleV4_70_charter.md`（本次任务章程：判据/度量协议/第一轮要做的活）与
`docs/goal_longrun_plan.md`（总章程：纪律与机械门）执行，结论写进 `docs/goal_register.md`。
目标：**vs `rule_v4` 的 128 局配对胜率 ≥70%**。每轮先读寄存器与任务章程、先写假设与判据再动手；
日常迭代 **32 局快读数**、验收才跑满 **128 局**；重要结论需过**无上下文独立验证**；
**判据变更 / 换 ckpt / 宣布成功一律走机械门**。**第一轮先测当前基线**。

（与章程一致处：判据 `≥70%` = `task_ruleV4_70_charter.md` §1；快读数 32 / 验收 128 = §1 表；
机械门三类 = §4 + `goal_longrun_plan.md` §5.3；第一轮的活 = §3。）

### 1.1 ⚠️ 当前用于替换活 goal 的文本（2026-10-05 更新：快读数 32 → 64 局）

> §1 是**还原记录**（与 §2 的乱码做过逐字符 round-trip 验证）——**不要改它**。
> 下面这块才是**要填进活 goal 的当前文本**；与 §1 只差"日常迭代快读数 32 局 → 64 局"这一处
> （用户 2026-10-05 定：时间充裕，慢一点没事）。

按 `docs/task_ruleV4_70_charter.md`（本次任务章程：判据/度量协议/第一轮要做的活）与
`docs/goal_longrun_plan.md`（总章程：纪律与机械门）执行，结论写进 `docs/goal_register.md`。
目标：**vs `rule_v4` 的 128 局配对胜率 ≥70%**。每轮先读寄存器与任务章程、先写假设与判据再动手；
日常迭代 **64 局快读数**、验收才跑满 **128 局**；重要结论需过**无上下文独立验证**；
**判据变更 / 换 ckpt / 宣布成功一律走机械门**。**第一轮先测当前基线**。

## 2. 乱码原文（存档，便于以后比对）

```
鎸?docs/task_ruleV4_70_charter.md锛堟湰娆′换鍔＄珷绋嬶細鍒ゆ嵁/搴﹂噺鍗忚/绗竴杞鑻ュ共鐨勬椿锛?docs/goal_longrun_plan.md锛堟€荤珷绋嬶細绾緥涓庢満姊伴棬锛夋墽琛岋紝缁撹鍐欒繘 docs/goal_register.md銆傜洰鏍囷細vs rule_v4 鐨?128 灞€閰嶅鑳滅巼 鈮?0%銆傛瘡杞厛璇诲瘎瀛樺櫒涓庝换鍔＄珷绋嬨€佸厛鍐欏亣璁句笌鍒ゆ嵁鍐嶅姩鎵嬶紱鏃ュ父杩唬 32 灞€蹇鏁般€侀獙鏀舵墠璺戞弧 128 灞€锛涢噸瑕佺粨璁洪渶杩囨棤涓婁笅鏂囩嫭绔嬮獙璇侊紱鍒ゆ嵁鍙樻洿 / 鎹?ckpt / 瀹ｅ竷鎴愬姛涓€寰嬭蛋鏈烘闂ㄣ€傜涓€杞厛娴嬪綋鍓嶅熀绾裤€?
```

## 3. 还原方法（可复核）

1. **原理**：乱码 = 原文的 UTF-8 字节被 CP936 解码。还原就是「按 CP936 编回字节 → 按 UTF-8 解码」。
2. **不能用 Python 的 `gbk`/`gb18030`**：
   - Windows CP936 把「未定义但格式合法」的字节对映到**私用区** `U+E0xx..`，Python 的 `gbk` 表没有这些项
     ⇒ `UnicodeEncodeError`；
   - Windows 把单字节 `0x80` 映成 `€`(U+20AC)，`gb18030` 却把 U+20AC 编成 `A2E3` ⇒ 字节错位、解码失败。
   ⇒ 必须用 **Windows 自己的转换表**（本工具用 `ctypes` 调 `MultiByteToWideChar(936, ...)`）。
3. **会丢字节**：CP936 遇到非法字节对时吐一个 ASCII `?`，它**吃掉两个字节**
   —— 一个 UTF-8 续字节 `0x80..0xBF`（补完当前字符）+ 一个 ASCII 字节 `<0x40`（空格/数字/标点）；
   字符串**末尾**的单字节残尾（如「。」的第 3 字节 `0x82`）只丢 **1** 个字节。
   这些字节**无法从乱码本身唯一确定**，只能给出候选、按语义定夺。
4. 本段乱码共 **328 字符 / 6 个 `?`**，6 处选择：

   | # | 乱码里的样子 | 补的字节 | 还原成 | 依据 |
   |---|---|---|---|---|
   | j0 | `鎸?docs/task_ruleV4_70_charter.md` | `0x89` + `0x20` | `按 docs/...` | 文首是「按 …… 执行」 |
   | j1 | `娲伙級涓?docs/goal_longrun_plan.md` | `0x8E` + `0x20` | `（活）与 docs/...` | 并列两份章程，必须是「与」 |
   | j2 | `鐨?128 灞€` | `0x84` + `0x20` | `的 128 局` | 「vs rule_v4 的 128 局配对胜率」 |
   | j3 | `鈮?0%` | `0xA5` + `0x37` | `≥70%` | 判据是 `≥70%`（文件名 `_70_`、章程 §1） |
   | j4 | `鎹?ckpt` | `0xA2` + `0x20` | `换 ckpt` | 寄存器／章程列出的机械门之一「换 ckpt」 |
   | j5 | 末尾 `绾裤€?` | `0x82`（不补 ASCII） | `基线。` | 句子结束要用「。」 |

   候选（按仓库语料频率）与为什么否掉，见工具 `--corpus docs/*.md` 输出的候选表：
   j0 `按/持/指/挑/挂`、j1 `不/一/个/上/为`、j2 `的/皮/皿…`、j3 `≈/≥/≠/≤`、j4 `据/换/损…`、j5 `。/、/」`。
5. **独立验证（逐字符）**：`还原文本 → UTF-8 编码 → CP936 解码` 必须**精确**得到乱码原文。
   实测 **True**（328/328 字符全等）⇒ 除上表 6 处外，每个字节都被独立钉死；
   上表 6 处是乱码的**唯一**自由度（任何「续字节+ASCII<0x40」都会被 CP936 重新吃成 `?`）。

## 4. 复现命令

```bash
# ① 从 DSH 会话状态里取出乱码（活 goal objective）
python - <<'PY'
import json, os
p = os.path.join(os.environ['DSH_HOME'], 'storages/session_projcache/sessions',
                 'session-5291b0fc-8d4c-404a-8e51-aa7bdcf78812.json')
obj = json.load(open(p, encoding='utf-8'))['record']['rows']['goal']['val']['current']['goal']['objective']
open('_tmp_moji.txt', 'w', encoding='utf-8').write(obj.split('\n\n')[0])   # 去掉用户追加的提问
PY

# ② 还原（--picks 就是 §3 表里的 6 处；不给 picks 会按语料频率猜并打印候选）
python code/tools/decode_cp936_mojibake.py --file _tmp_moji.txt --corpus "docs/*.md" \
  --picks "0=89:20,1=8e:20,2=84:20,3=a5:37,4=a2:20,5=82:none" --out _tmp_fixed.txt

# ③ 独立验证：应打印 round-trip 逐字符一致: True
python code/tools/decode_cp936_mojibake.py --verify _tmp_moji.txt _tmp_fixed.txt
```

## 5. 现状

- 活的 goal objective 已被替换为 §1 的文本（`update_goal` edit），本文件是它的存档副本。
- ⚠️ 事故本身出在 DSH 侧（中文写进 goal objective 的那条路径把 UTF-8 当 CP936 解），
  **本仓库没有改动 DSH**；若再出现同类乱码，直接用本工具 + §4 的命令还原即可。
