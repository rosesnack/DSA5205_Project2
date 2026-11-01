# spy_overlay_validation_20261101

验证期两组门槛、六组预测的完整结果。summary.csv 为 20 行比较表，experiment_10bp/ 和 experiment_15bp/ 保留逐日账本、持仓、决策与成交。原结果和 manifest 不改。

当前完整性校验在仓库根目录运行：

```powershell
python -B scripts/verify_spy_overlay_delivery.py
```

本目录参与 GitHub 原结构同步，包含源文件而非单独汇报 ZIP。
