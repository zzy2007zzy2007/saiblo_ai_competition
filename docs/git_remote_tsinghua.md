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
| `59b618c` | 加 GitHub Actions 定时镜像工作流（见下节） |

## GitHub 自动镜像（Actions 每小时同步）

GitHub 镜像仓库：https://github.com/zzy2007zzy2007/saiblo_ai_competition （**public**）

GitHub 自己**没有**"定时拉取镜像"功能（GitLab 有 Pull Mirror，GitHub 只有一次性 Import），
所以用 GitHub Actions 的 `on: schedule` 定时触发，见
`.github/workflows/sync-from-gitlab.yml`：

```
git clone --mirror  https://<user>:<token>@git.tsinghua.edu.cn/...   # 拉清华
git push --mirror   https://x-access-token:<GITHUB_TOKEN>@github.com/...  # 推到 GitHub
```

配置齐了这些才能跑：

1. **清华侧**：项目 Settings → Repository → Deploy tokens，建一个
   **username = `oauth2`**、scope 只勾 **`read_repository`** 的 deploy token
   （仓库是私有的，必须给读权限；username 填 `oauth2` 才能对上 workflow 里默认的用户名，
   否则得另外配一个 `GITLAB_USER` secret）。
2. **GitHub 侧**：Settings → Secrets and variables → Actions → 新建 secret
   **`GITLAB_TOKEN`** = 上面那个 deploy token 的值。
3. **GitHub 侧**：Settings → Actions → General → Workflow permissions 改成
   **Read and write permissions**。默认的 read-only 会让 `push --mirror` 直接 403。
4. **workflow 文件必须两边都有**：`push --mirror` 会让 GitHub 完全对齐 GitLab，
   GitHub 上有、GitLab 上没有的文件下次同步就会被删掉（包括这个 workflow 自己）。

已验证：手动触发 run #1（`36107712370`），mirror job 1m4s 成功（clone 49s + push 12s），
之后 GitHub 与清华 GitLab 的 `master` 指向同一个 commit。

**注意**：

- cron 是 UTC 且不精确（高峰期会晚几分钟）；仓库 60 天无活动，定时工作流会被自动停用。
- `push --mirror` 会**删掉** GitHub 上 GitLab 里没有的分支/标签，所以别直接在 GitHub 上开分支干活。
- 想改同步频率就改 workflow 里的 `cron`；文件改了记得**同时**更新两边（或只改 GitLab 那份，
  等下一次同步覆盖 GitHub 那份——但下一次同步前 GitHub 上的旧 file 仍在生效）。


## 坑：Git for Windows 的 MSYS `ssh` 读不到 `~/.ssh`（中文用户名）

**症状**（在带 conda `(base)` 的那个 shell 里 `git push tsinghua master` 时报）：

```
Could not create directory '/c/Users/\324\370\327\323\321\322/.ssh' (No such file or directory).
Failed to add the host to the list of known hosts (...)
git@git.tsinghua.edu.cn: Permission denied (publickey).
```

**原因**：Git for Windows 自带的 MSYS 版 `ssh.exe`（`D:\Program Files\Git\usr\bin\ssh.exe`）
用 ANSI 代码页去取用户名，中文用户名 `曾子岩` 变成了字节串 `\324\370\327\323\321\322`
（GBK 的 曾=D4F8 子=D7D3 岩=D1D2），于是它认的 HOME 是 `/c/Users/<乱码>`——这个目录不存在，
所以它既读不到 `~/.ssh/config`（配 `IdentityFile` 的地方），也读不到 `known_hosts`，
只能去试默认密钥名（`~/.ssh/id_ed25519` 等，我们没这个文件）→ `Permission denied (publickey)`。
Windows 自带的 OpenSSH（`C:\Windows\System32\OpenSSH\ssh.exe`）没这个问题，能用中文用户名。
哪个 ssh 胜出取决于 PATH，所以**同一个命令在不同 shell 里表现不一样**（带 conda 的 shell 踩过）。

**已修**：在本仓库的 git config 里钉死 ssh 可执行文件和密钥，屏蔽 PATH 差异：

```
git config core.sshCommand '"C:/Windows/System32/OpenSSH/ssh.exe" -i "C:/Users/曾子岩/.ssh/id_ed25519_tsinghua" -o IdentitiesOnly=yes'
```

`core.sshCommand` 优先级高于 `GIT_SSH`（实测：即使强制 `GIT_SSH=MSYS ssh` 也照样成功）。
如果希望所有仓库都生效，把同一条命令加 `--global`。想验证身份：`ssh -T git@git.tsinghua.edu.cn`
应回 `Welcome to GitLab, @zengzy25!`。

顺带确认：`git.tsinghua.edu.cn` 的 ED25519 主机密钥指纹是
`SHA256:zYgtTfwQdCJCMA0GQGRq06ldQGNlNIxoKPDfFfW2kZc`，与首次连接时提示的一致，可以放心 `yes`。

## 待办 / 已知问题

1. ~~临时分支 `_authcheck`~~ **已解决**：做写入权限自检时推过它，当时仓库是空的，
   GitLab 把它设成了默认分支。默认分支改回 `master` 后，用命令行删它会被拒
   （`You can only delete protected branches using the web interface.`，它当年是默认分支
   所以被自动保护）。后来在网页上删掉了；GitHub 镜像那边也跟着消失了
   （`push --mirror` 会 prune 掉 GitLab 里没有的分支）。**教训：往空仓库做写入权限自检时，
   别用 `master`/`main` 之外的分支名——第一个推上去的分支会被设成默认分支。**
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
3. 工作区还有大量未提交改动（`code/`、`docs/`、根目录一堆 `_tmp_*`），推送时**没有**一起提交；
   同步远端就一条命令：`git push tsinghua master`（另一个会话还在同一工作区持续提交，
   所以远端会不断落后，想起来就推一次）。`gitea` 那个本地远端一直没推，比 master 落后更多。
   **GitHub 那边已经不用手动推了**：只要推到了清华 GitLab，Actions 每小时会自动镜像过去
   （想立刻同步就去 Actions 页面手动 Run workflow）。
