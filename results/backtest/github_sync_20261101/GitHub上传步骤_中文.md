# GitHub 原目录同步步骤（2026-11-01）

本次直接在现有仓库中提交源文件。当前本地分支为 `yae/backtest-handoff`，origin 为 `https://github.com/rosesnack/DSA5205_Project2.git`。没有自动运行 git add、commit、push、rebase 或合并，也没有新建另一个项目仓库。远端当前变化由下面的 fetch 检查。

## 本次目录安排

|位置|同步内容|
|---|---|
|README.md、两份原进度/交接说明|接入最新 SPY 结果、读取与校验入口|
|SPY增强策略_最终结果与交接_20261101.md|当前最终展示方案和完整结果索引|
|scripts/|六个既有新增实验/结果脚本，另加两个同步校验和安全复现入口|
|notebooks/06_spy_overlay_final_results.ipynb|有保存输出的结果阅读 notebook；01—05 不变|
|results/backtest/spy_overlay_validation_20261101/|验证期两门槛全部结果及逐日记录|
|results/backtest/spy_overlay_test_20261101/|测试期全部结果、分红再投SPY、年份/行业/成本诊断|
|results/backtest/spy_overlay_final_20261101/|最终选定 GBDT S1、15／7.5 bp，报告、表图及全部候选排序|
|results/backtest/github_sync_20261101/|精确上传路径、五份文档的改前字节快照与改后哈希、同步清单|
|.gitattributes|保留文件原字节，避免 CRLF/LF 自动转换破坏冻结哈希|
|.gitignore|补充排除本地根目录 ZIP 和交接打包目录|

不上传仓库外的 `交接打包_yae_20261101/` 或汇报 ZIP；不复制旧仓库目录到新文件夹；不重新训练或回测。全部详细结果选择直接同步，三组结果约 128 MB，最大单文件约 13.4 MB，不为了压缩上传量删掉完整账本。

## 1. 进入正确仓库并检查分支

在 PowerShell 中按顺序运行：

```powershell
Set-Location -LiteralPath 'D:\桌面文件\NUS_sem1\DSA5205\Project2\DSA5205_Project2-git'
git branch --show-current
git remote -v
git fetch origin
git status --short --branch
```

当前应在 `yae/backtest-handoff`。若不是这个分支，先核对，不要直接提交到别的分支。这里的 GitHub 同步指把新增文件推到已有分支，不是自动覆盖 main。

## 2. 只读核验本次内容

```powershell
python -B scripts/verify_spy_overlay_delivery.py
```

应显示 `passed: true`。该命令只用 Python 标准库，只读文件，不重训、不生成回测、不改实验结果。它允许五份明确登记的文档更新，同时严格检查原预测、交易代码、所有数值结果与输出。

旧实验生产脚本的整仓库冻结哈希保留为当时快照；文档已更新后，不用旧命令当作当前上传验证入口。当前如需另行复现，使用 `replay_spy_overlay_frozen.py` 写入仓库外的新空目录；上传时不要运行复现命令。

## 3. 按清单暂存，并看清暂存内容

```powershell
git add --pathspec-from-file=results/backtest/github_sync_20261101/upload_paths.txt
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check
```

`upload_paths.txt` 列的是当前仓库内的相对路径，Git 会读取 UTF-8 中文文件名。它涵盖本次新增代码、完整结果、notebook 和明确修改的说明文件，包含同步清单本身，不包含 ZIP、临时文件或环境目录。不使用笼统的 `git add .`。

`git diff --cached --check` 正常时没有输出。暂存后可再运行一次上面的只读验证。

## 4. 提交、同步已有分支并推送

```powershell
git commit -m "Add SPY-core backtests and final result handoff"
git pull --rebase origin yae/backtest-handoff
python -B scripts/verify_spy_overlay_delivery.py
git push origin yae/backtest-handoff
```

若 rebase 出现冲突，或之后校验失败，先停在该步处理，不要 force push。若服务器没有新增提交，pull 会提示已经是最新。推送失败时本地 commit 仍在，不必重新复制文件或重复生成结果。

## 5. 在 GitHub 确认看到的是正确分支

打开 [仓库](https://github.com/rosesnack/DSA5205_Project2)，切换到 `yae/backtest-handoff`，查看新的 commit、README 当前入口，以及上述 scripts/notebooks/results 路径。

`git push origin yae/backtest-handoff` 不会自动修改 main。如果组长希望内容进入 main，创建 `yae/backtest-handoff → main` 的 Pull Request，再由组内合并。可以使用本目录 [PR说明.md](PR说明.md) 的标题和正文。

## 最终方案口径

GBDT S1、15／7.5 bp 是已有 12 组中测试期成本后净年化最高者；测试净年化 22.72%，分红再投入 SPY 22.02%，最大回撤约 19.30%。这是看过测试结果后按用户要求作出的展示选择，报告和候选排序均已保留该说明。
