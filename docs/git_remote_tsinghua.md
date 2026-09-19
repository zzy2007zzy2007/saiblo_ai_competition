# 推到清华 GitLab 的配置与踩坑记录（2026-09-19）

远端仓库：https://git.tsinghua.edu.cn/zengzy25/saiblo_ai_competition

## 远端配置

```
origin   http://gitea:gitea123@localhost:3000/gitea/2026_ant_game.git   # 本地 gitea，原来就在用
tsinghua git@git.tsinghua.edu.cn:zengzy25/saiblo_ai_competition.git     # 新增，master 已设置 tracking
```

认证走 **SSH**（不是 HTTPS）：

- 密钥对：`~/.ssh/id_ed25519_tsinghua`（私钥，无口令）+ `.pub`（公钥已加到 git.tsinghua.edu.cn 账号）
- `~/.ssh/config` 里给 `git.tsinghua.edu.cn` 指定了 `IdentityFile` + `IdentitiesOnly yes`，所以直接
  `git push tsinghua` 就行，不用带参数
- 用 `ssh -T git@git.tsinghua.edu.cn` 自检，正常会回 `Welcome to GitLab, @zengzy25!`

## 为什么不能用 HTTPS

HTTPS 推送**被服务器端卡在 HTTP 413（请求体过大）**，实测：

| 尝试 | 结果 |
| --- | --- |
| 默认 chunked 传输 388MB | `error: RPC failed; HTTP 413`，写到约 190MB 时被拒 |
| `-c http.postBuffer=600000000`（单请求 + Content-Length） | 同样 413 |

原因不是总大小，而是**历史里单个提交就 400MB**：第 9 个提交 `2cd626f4` 一次性把
`training_history_20260630_*/` 那批检查点（约 400MB，逐提交统计见根目录
`git_push_size_profile.csv`）提交了进去，后来才加进 `.gitignore`。
单提交无法拆小，所以**分批推送也绕不过去**；SSH 不经过 nginx，才能一次推完。
（顺带：机器上用 git credential manager 存了 git.tsinghua.edu.cn 的 `glpat-` token，
git 读写权限是够的，但它没有 `api` scope，改不了项目设置、也没有绕过 413 的办法。）

## 推送记录

| 提交 | 内容 |
| --- | --- |
| `076d95e` | 全量历史，355 个提交 / 388MB（含上面那批历史检查点） |
| `ef15ee8` | 补 `.gitignore`（约 280GB 大数据目录）+ `其他版本ai` 源码入库 |

## 待办 / 已知问题

1. **默认分支还没改过来**：做写入权限自检时推过一个临时分支 `_authcheck`
   （当时仓库是空的，GitLab 把它设成了默认分支），现在删不掉，会报
   `remote: GitLab: The default branch of a project cannot be deleted.`。
   去 https://git.tsinghua.edu.cn/zengzy25/saiblo_ai_competition/-/settings/repository
   把 Default branch 改成 `master` 保存，然后 `git push tsinghua --delete _authcheck`。
2. **`.gitmodules` 改成 GitHub 上游了**：原来写的是 `./Ant-Game`（相对 URL），在这个
   仓库上会解析成 `git.tsinghua.edu.cn/zengzy25/Ant-Game`——那个项目不存在；而原来钉的
   提交 `ef9e653` GitHub 上也没有（它比 GitHub 的 main `0a6bee4` 多两个本地提交：
   "Update README and add image"、"Replace remote image with local screenshot"）。
   所以现在 `.gitmodules` 指向 `https://github.com/LorenzLorentz/Ant-Game.git`，
   并把 submodule 的 pin **对齐到 `0a6bee4`**，保证 `git clone --recurse-submodules` 能成功。
   代价：那两个本地提交、以及 `game/src/comm_judger.cpp` / `game/src/main.cpp` 里
   **用 `fwrite` 改的二进制安全输出**（绕开 Windows 下 `cout` 把 `0x0A` 翻译成 CRLF 的
   bug）都不在 pin 里，只留在本地工作区/gitea，换机器就没了。要长期保留的话得单独想办法
   （推到自己的 Ant-Game 项目当 fork，或者存成补丁文件）。
3. 工作区还有大量未提交改动（`code/`、`docs/`、根目录一堆 `_tmp_*`），这次**没有**一起提交；
   远端 master 是 `ef15ee8` 的状态。
