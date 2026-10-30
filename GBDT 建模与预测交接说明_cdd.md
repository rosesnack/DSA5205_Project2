# GBDT 建模与预测交接说明

## 1. 交付内容

GBDT 已完成验证选参、按月训练、逐日预测和预测误差评价。回测直接读取现有预测 CSV，无需重新训练。

| 文件 | 用途 |
|---|---|
| [gbdt_predictions.csv](results/gbdt/gbdt_predictions.csv) | **回测接入的主预测表**，与 Elastic Net 使用相同的七列接口 |
| [gbdt_predictions_full.csv](results/gbdt/gbdt_predictions_full.csv) | 附带数据集划分、成交日期、实际收益和评分标记 |
| [model_panel.csv](data/processed/model_panel.csv) | 统一时间、收益标签和样本资格 |
| [04_gbdt.ipynb](notebooks/04_gbdt.ipynb) | 完整建模与评价代码 |
| [README.md](README.md) | 环境安装、运行顺序及结果文件说明 |

## 2. 预测接口与时间对应

| 字段 | 含义 |
|---|---|
| `date` | 信号日期 |
| `ticker` | 行业 ETF：XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY |
| `model` | 固定为 `gbdt` |
| `scheme` | S0、S1 或 S2 |
| `model_month` | 当月模型对应的月份，格式 YYYY-MM |
| `fit_cutoff` | 模型可使用历史信息的截止时刻，带时区 |
| `y_hat` | 行业减 SPY 的持有期现金收益预测，小数单位 |

信号在 `date` 当天纽约时间 **18:00** 形成，下一 SPY 交易日开盘进入，再下一交易日开盘退出。实际日期读取 `entry_date`、`exit_date`，按交易日历对应。

例如：2018-01-02 的信号对应 2018-01-03 开盘至 2018-01-04 开盘。`y_hat = 0.001` 表示预测行业相对 SPY 收益为 0.1%，即 10 bp；预测值已经包含减 SPY 的定义。

预测表以 `(date, ticker, model, scheme)` 唯一标识记录。三个 scheme 共 **59,346 行**，包含验证期和测试期：

| 区间 | 每个 scheme 的预测行数 | 每个 scheme 的可评分行数 |
|---|---:|---:|
| 验证：2018–2022 | 11,331 | 11,313 |
| 测试：2023–2026 年 9 月 | 8,451 | 8,433 |

统一评价使用 `split` 和 `common_scoreable`。验证期末跨区间标签及数据末端尚未实现的标签保留预测，但不进入对应区间的评分。

## 3. 模型与最终配置

每个行业单独训练，历史从 2008 年开始，按月扩展窗口重训，只使用拟合截止时刻已实现的标签。同一 scheme 的九个行业共享验证集选出的超参数，测试期间参数固定。

GBDT 使用学习率 0.03、最大深度 4。叶子数候选为 `[3, 5, 7]`，最小叶节点样本数候选为 `[10, 20, 50]`，L2 候选为 `[0, 1, 10]`，每个 scheme 共 27 个组合，三个 scheme 共 81 个。根据已有验证集的逐轮 MSE 选择轮数，最大 100 轮、patience 10、tol `1e-8`。

| Scheme | 特征方案 | 最大叶子数 | 最小叶节点样本数 | L2 | 最佳轮数 | 验证 R² | 测试 R² |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | 57 个自身行业及市场特征 | 5 | 20 | 10 | 23 | 0.002199 | 0.000261 |
| S1 | 97 个特征，加入其他行业信息 | 3 | 20 | 10 | 43 | 0.003587 | 0.000429 |
| S2 | 102 个特征，加入行业关系汇总 | 3 | 20 | 10 | 40 | 0.003552 | 0.000403 |

**S1 是验证集选出的主方案，S0/S1/S2 全部保留供统一对比。** 表中 R² 为相对历史均值基准的小数值。当前 GBDT 的测试 MSE 略优于 Elastic Net，交易净收益由统一回测计算。

## 4. 接入示例

从仓库根目录运行，将七列预测表按 `(date, ticker)` 连接共享面板：

```python
import pandas as pd

pred = pd.read_csv(
    'results/gbdt/gbdt_predictions.csv',
    parse_dates=['date'], float_precision='round_trip'
)
pred['fit_cutoff'] = pd.to_datetime(pred['fit_cutoff'], utc=True)
panel = pd.read_csv(
    'data/processed/model_panel.csv',
    parse_dates=['date', 'entry_date', 'exit_date'],
    float_precision='round_trip'
)
columns = [
    'date', 'ticker', 'split', 'entry_date', 'exit_date',
    'common_scoreable', 'sector_forward_cash_return',
    'spy_forward_cash_return', 'target_relative_return'
]
joined = pred.merge(panel[columns], on=['date', 'ticker'], how='left')
validation_scored = joined.loc[
    joined['split'].eq('validation') & joined.common_scoreable
].copy()
test_scored = joined.loc[
    joined['split'].eq('test') & joined.common_scoreable
].copy()
```

`joined` 保留全部信号及时间信息；后两张表用于统一评价。需要主方案时筛选 `scheme == 'S1'`，做三方案对比时分别构建组合。

## 5. 回测同学接下来完成

1. 接入 GBDT 与 Elastic Net 的预测，按日期、行业和 scheme 对齐。
2. 沿用小组统一的入场、退出、持仓延续、仓位限制、SPY 净额合并及成本规则；交易阈值等参数用验证期选择，固定后评价测试期。
3. 根据实际持仓及行业/SPY 两条腿收益计算策略收益，计入分红、交易成本和借券费用。
4. 输出各模型、各 scheme 的净收益曲线、年化收益、波动率、Sharpe、最大回撤、换手率与成本影响。

模型参数见 [selected_hyperparameters.csv](results/gbdt/selected_hyperparameters.csv)，预测指标见 [forecast_metrics.csv](results/gbdt/forecast_metrics.csv)，月度训练记录见 [fit_audit.csv](results/gbdt/fit_audit.csv)。
