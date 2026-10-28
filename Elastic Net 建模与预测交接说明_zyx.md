# Elastic Net 建模与预测交接说明（zyx → 下一棒）

更新日期：2026-10-27  
主要接收人：负责统一交易回测的同学（yae）；另一模型负责人也可据此对齐预测接口。

## 1. 本次交付到哪一步

zyx 负责的 Elastic Net 部分已经完成：读取第一棒的固定数据与特征、训练窗口内标准化、按月扩展窗口训练、验证期选择参数、测试期逐日预测，以及预测误差、系数、训练窗口和 XLF 专项诊断。

**下一棒请直接接收预测表，进入统一交易回测。** 本部分没有计算交易策略收益，没有选择交易阈值，也没有替代统一回测中的仓位、交易成本和借券费用规则。

当前 [03_elastic_net.ipynb](notebooks/03_elastic_net.ipynb) 共 10 节，21 个代码单元格均有执行记录，未发现错误输出。最近补充的 XLF 人工解读和主汇总指向属于文字更新，不改变已生成的预测值、锁定参数或模型。

### 优先阅读和使用的文件

| 文件 | 用途 |
|---|---|
| [notebooks/03_elastic_net.ipynb](notebooks/03_elastic_net.ipynb) | 完整建模过程、结果及第 9、10 节诊断 |
| [results/elastic_net/elastic_net_predictions.csv](results/elastic_net/elastic_net_predictions.csv) | **回测接入的主要预测表**，固定七列接口 |
| [results/elastic_net/elastic_net_predictions_full.csv](results/elastic_net/elastic_net_predictions_full.csv) | 含 split、参数、标签、基线、可评分标记和持有期日期的审计表 |
| [data/processed/model_panel.csv](data/processed/model_panel.csv) | 日期、真实收益、成交日期、标签可得时刻及评价资格的共享来源 |
| [data/processed/panel_spec.json](data/processed/panel_spec.json) | 预测目标、特征白名单与时间接口定义 |
| [results/elastic_net/report_summary.md](results/elastic_net/report_summary.md) | 正式模型结果概述，含第 10 节指向 |
| [README_elastic_net.md](README_elastic_net.md) | 环境、运行顺序及输出说明 |

文中路径均相对于仓库根目录，可直接在 GitHub 上查看。

## 2. 数据、预测目标和时间定义

使用第一棒已经准备好的 `model_panel.csv`、`model_inputs/` 和 `panel_spec.json`，没有重新下载行情、重新定义标签或改动上游特征。

行业为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY。每个行业分别拟合三套输入方案：

| 方案 | 特征数 | 信息范围 |
|---|---:|---|
| S0 | 57 | 本行业特征与共同市场状态 |
| S1 | 97 | S0 加其他行业历史信息 |
| S2 | 102 | S1 加行业关系汇总特征 |

预测目标为 `target_relative_return`，即行业 ETF 的持有期现金收益减去 SPY 同期现金收益。按共享接口：

- 信号在信号日纽约时间 18:00 形成。
- 进入时点是下一个 SPY 交易日开盘。
- 退出时点是第二个后续 SPY 交易日开盘。
- `date` 是**信号日期**，不是成交日期；成交和持有期请读取共享 panel 的 `entry_date`、`exit_date`、`entry_at` 等字段。
- `y_hat` 是小数收益单位。`0.001` 表示预测相对收益 0.1%，即 10 bp。它已经是行业减 SPY 的预测，不能再减一次 SPY。

本部分的预测误差评价使用上游现金收益标签。下一棒计算策略收益时仍须统一处理实际持仓、分红和成本，不能把预测标签直接当作最终策略净收益。

## 3. 正式 Elastic Net 的训练与选参规则

训练历史从 2008 年开始；验证期为 2018–2022 年；测试期为 2023 年至 2026 年 9 月，数据截止 2026-09-30。

主模型对每个行业单独拟合，并按月 expanding-window 重训。每个月的拟合截止时刻是该月首个交易日前一个 SPY 交易日的纽约时间 18:00。训练观测必须同时满足：

1. 信号日期不早于 2008-01-01。
2. `common_features_valid` 与 `label_valid` 为真。
3. `signal_at <= fit_cutoff`。
4. `label_available_at <= fit_cutoff`。

`StandardScaler` 只在该次合格历史训练样本上拟合，再用于当月预测；目标收益不标准化。测试期按月重训可以使用截至当时已经实现、标签已经成熟的测试期历史观测，但超参数保持验证期锁定值。

### 统一参数网格与锁定结果

主流程和第 9 节均使用：

```python
ALPHAS = [1e-5, 1e-4, 1e-3]
L1_RATIOS = [0.2, 0.4, 0.6, 0.8]  # rho
```

每个 scheme 有 12 对参数，三套 scheme 共 36 个组合。每个 scheme 内，九个行业分别训练，但共享一对通过验证选择的参数。选择标准是共同可评分验证样本上的 pooled MSE；精确相同时优先较大 alpha，再优先较大 rho。

| Scheme | 锁定 alpha | 锁定 rho | 验证 MSE | 验证可评分行数 |
|---|---:|---:|---:|---:|
| S0 | 0.001 | 0.8 | 0.0000992538214 | 11,313 |
| S1 | 0.001 | 0.8 | 0.0000992453384 | 11,313 |
| S2 | 0.001 | 0.8 | 0.0000992453384 | 11,313 |

**三个方案全部保留并交接，没有选择一个“总体获胜方案”。** S1 与 S2 的验证 MSE 差约为 2.03×10⁻¹⁹，属于数值上可忽略的差别，不应解释成 S2 明显更好。

主验证共 19,440 次月度行业拟合；锁定后的测试共 1,215 次拟合。训练资格与第一棒的月度表交叉核对，共 945 条“月份 × 行业”记录，训练行数及相关时间戳均匹配。数值不收敛时只增加迭代预算重试，不接受未收敛的结果，也不以更好的预测成绩作为重试标准。

## 4. 正式结果应该怎样理解

### 4.1 汇总预测表现

三个 scheme 的测试预测完全相同，包括末端尚不可评分的预测。测试期每个 scheme 的可评分样本均为 8,433 行：

| 指标 | S0 | S1 | S2 |
|---|---:|---:|---:|
| RMSE（小数收益） | 0.00956634 | 0.00956634 | 0.00956634 |
| MAE（小数收益） | 0.00693158 | 0.00693158 | 0.00693158 |
| OOS R²，相对历史均值 | −0.0473% | −0.0473% | −0.0473% |
| OOS R²，相对零预测 | −0.0273% | −0.0273% | −0.0273% |
| 方向准确率 | 50.8241% | 50.8241% | 50.8241% |

历史均值基线与模型使用同一批当月已经成熟的历史训练标签。这里的 R² 是 `1 − 模型 SSE / 基线 SSE`，不是普通拟合内 R²，也不是策略收益率。

锁定测试拟合的非零系数数目均值约 0.348，中位数为 0，范围为 0–3；87.41% 的拟合只有截距。跨行业或关系汇总特征在锁定测试拟合中的选择率为零。最终参数位于所搜索网格的较强正则化角点，因此本次预测大多接近历史均值预测。

可支持的结论是：**在本次线性模型、参数网格和样本期下，加入跨行业信息没有带来增量测试预测表现。** 不能据此认定跨行业信息在任何模型或任何时期都无效，也不能将微小误差差异解释成稳定交易价值。

### 4.2 XLF 专项诊断（第 10 节）

XLF 占各 scheme 非零“系数 × 拟合”出现次数的约 95.74%，并占各行业正 excess SSE 总和的约 99.96%。后一个比例是**所有相对基线表现较差行业的正超额损失中的份额**，不是 XLF 占全部预测误差的比例，也不表示其他行业都没有非零系数。

XLF 的测试 OOS R² 约为 −0.813%。其验证期结果如下，均使用共同 expanding 历史均值基线；参数沿用第 9 节按全部行业选定的结果，没有针对 XLF 重新选参：

| Scheme / 模型 | Expanding | Rolling 5y | Rolling 3y |
|---|---:|---:|---:|
| S0 / Elastic Net | −0.660% | −0.212% | −0.637% |
| S1 / Elastic Net | −0.567% | −0.267% | −0.778% |
| S2 / Elastic Net | −0.567% | −0.267% | −0.789% |
| S0 / Ridge | +0.065% | +0.001% | −0.276% |
| S1 / Ridge | −0.141% | −0.193% | −0.630% |
| S2 / Ridge | −0.140% | −0.190% | −0.633% |

XLF 在验证期就已经出现负的 Elastic Net R²，并非测试期才出现。S0 的五年窗口相对 expanding 有改善，但三年窗口又接近 expanding，**窗口越短并不单调越好**。Ridge 也只在 S0 的部分窗口下略为正，并非对所有特征方案均有效。

这些现象不能证明 2008–2009 金融危机就是表现较差的原因。本次没有删除早期数据，没有改动正式模型，也没有因为 XLF 的测试表现而更换参数或正式预测。

详细表格与图见 [xlf_diagnostics/](results/elastic_net/xlf_diagnostics/)。

### 4.3 第 9 节是验证期探索，不是另一套正式测试模型

第 9 节比较 expanding、rolling 5y、rolling 3y，并加入 Ridge 对照。共 144 个配置、77,760 次月度行业验证拟合，其中 Elastic Net 58,320 次、Ridge 19,440 次。

各方法在相同验证样本及共同历史均值基线下比较，同时报告行业等权选参、年度表现、系数稳定性与固定 alpha 下的 rho 敏感性。Ridge 使用 λ∈{0.01, 0.1, 1, 10}，对应 sklearn 的 `alpha = n_train * lambda`。

Ridge/expanding 在 pooled 验证选择下的整体 R² 约为 S0 +0.1133%、S1 +0.1288%、S2 +0.1307%。这些成绩在同一验证期内经过选择，含选择偏差；不能把它们当作独立测试证据。Ridge 通常系数稠密，其非零数目也不能直接视为稀疏特征选择效果。

**下一棒仍接收正式 expanding Elastic Net 的预测。第 9 节没有生成替代测试预测，也没有授权把某个验证期最优窗口直接升级为最终模型。**

## 5. 给回测同学的预测接口

主文件：`results/elastic_net/elastic_net_predictions.csv`。

| 列 | 含义 |
|---|---|
| `date` | 信号日期 |
| `ticker` | 行业 ETF |
| `model` | 固定为 `elastic_net` |
| `scheme` | S0、S1 或 S2 |
| `model_month` | 信号所在拟合月份，格式 YYYY-MM |
| `fit_cutoff` | 带时区的历史拟合截止时刻 |
| `y_hat` | 行业相对 SPY 的持有期收益预测，小数单位 |

导出表同时包含验证和测试预测，共 **59,346 行**；每个信号日、每个 scheme 包含九个行业。

| 区间 | 每个 scheme 的预测行数 | 每个 scheme 的可评分行数 |
|---|---:|---:|
| Validation | 11,331 | 11,313 |
| Test | 8,451 | 8,433 |

预测存在不代表标签可用于评价。验证期末跨分区标签和样本末端未成熟标签等情况，均应由共享 `common_scoreable` 与 `split` 控制。全样本最后两个信号日仍保留 54 行预测（三个 scheme × 九行业 × 两日期），但不计入已实现预测评价。

### 接入示例

从仓库根目录运行。下面仅演示连接和样本检查，不构造交易策略：

```python
from pathlib import Path
import pandas as pd

root = Path('.')
forecasts = pd.read_csv(
    root / 'results/elastic_net/elastic_net_predictions.csv',
    parse_dates=['date'], float_precision='round_trip'
)
forecasts['fit_cutoff'] = pd.to_datetime(forecasts['fit_cutoff'], utc=True)
panel = pd.read_csv(
    root / 'data/processed/model_panel.csv',
    parse_dates=['date', 'entry_date', 'exit_date'],
    float_precision='round_trip'
)

assert not forecasts.duplicated(['date', 'ticker', 'model', 'scheme']).any()
assert not panel.duplicated(['date', 'ticker']).any()
metadata = [
    'date', 'ticker', 'split', 'entry_date', 'exit_date', 'entry_at',
    'label_available_at', 'common_features_valid', 'common_scoreable',
    'target_relative_return', 'sector_forward_cash_return',
    'spy_forward_cash_return'
]
joined = forecasts.merge(
    panel[metadata], on=['date', 'ticker'], how='left',
    validate='many_to_one', indicator=True
)
assert joined['_merge'].eq('both').all()
joined = joined.drop(columns='_merge')
validation_scored = joined.loc[
    joined['split'].eq('validation') & joined['common_scoreable']
].copy()
test_scored = joined.loc[
    joined['split'].eq('test') & joined['common_scoreable']
].copy()
```

由于预测表保留三套 scheme，而 panel 每个 `(date, ticker)` 只有一行，连接时使用 `many_to_one`。跨模型比较时，应再与其他模型对齐共同日期、行业和评分资格；不要仅按文件行号拼接。

## 6. 下一棒具体接什么、做什么

1. **接入并核对样本。** 读取七列预测表，连接共享 panel；区分验证与测试，并核对九行业覆盖、重复键、时区和实际成交日期。
2. **按小组统一规则构造策略。** 在允许做多、做空或空仓的行业/SPY 配对框架下实现信号、进出场、持仓延续及组合净敞口。
3. **交易参数只用验证期选择。** 入场阈值及其他允许选择的交易参数，由统一回测负责人根据小组方案确定；测试结果只作锁定规则后的评价，不据此反复调整。
4. **按当前小组接口实现仓位与成本。** 当前 Notebook 的交接约定包括半阈值退出、最多四对、每条腿 ±12.5%、持仓延续、SPY 净额合并和 3 bp 主执行成本。阈值候选、成本计费口径、借券费用及其余实施细节请与统一回测方案核对；本建模文件没有替你实现或验证这些交易规则。
5. **与另一模型及基准统一比较。** 保留 Elastic Net 的 S0/S1/S2 标识，使用一致的标签、区间、成交假设和成本口径，与另一模型及 SPY 买入持有等小组基准比较。
6. **完成交易层面结果与组内汇总。** 输出净收益、年化收益、波动率、Sharpe、最大回撤、换手率、成本影响和收益曲线等小组要求的指标，并将预测层与交易层结论分开解释。

本次三个 scheme 的正式测试预测相同。因此，如果三者在相同样本、参数和确定性交易规则下得到不同测试仓位或收益，应优先排查连接、排序、缺失值处理或交易参数是否一致。

不要根据 XLF 的事后诊断直接删除 XLF、删除 2008–2009 数据或改用 rolling 窗口替换正式模型；如果小组另做敏感性实验，应单独标明实验性质和与正式结果的区别。

## 7. 复现与审计文件

环境：Python 3.12.10；完整依赖见 [requirements.txt](requirements.txt)。当前记录使用 NumPy 2.5.4、pandas 3.0.6、scikit-learn 1.9.1。

如需复现，从仓库根目录建立项目环境，再打开 Notebook：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

打开 `notebooks/03_elastic_net.ipynb`，选择上述环境对应的内核，按顺序执行全部单元格。第 10 节依赖第 1–9 节在同一内核中的对象，不能在空内核里直接运行。完整运行会重新写入结果目录；只接入回测时可直接使用交付 CSV，无需重训。

| 审计或诊断文件 | 核对内容 |
|---|---|
| `results/elastic_net/design_config.json` | 参数网格、预处理和选择规则、上游哈希 |
| `results/elastic_net/locked_selection.json` | 每个 scheme 的锁定参数与配置哈希 |
| `results/elastic_net/validation_grid.csv` | 全部 36 个正式验证组合 |
| `results/elastic_net/fit_audit.csv` | 月度拟合样本、标签成熟时间、收敛与稀疏性 |
| `results/elastic_net/monthly_eligibility_crosscheck.csv` | 与上游月度训练资格的交叉核对 |
| `results/elastic_net/monthly_coefficients.csv` | 锁定测试拟合的逐特征系数 |
| `results/elastic_net/sector_forecast_metrics.csv` | 各行业预测指标 |
| `results/elastic_net/loss_difference_uncertainty.csv` | 预测损失差异的近似 HAC 区间 |
| `results/elastic_net/run_manifest.json` | 运行环境、源数据哈希和主流程完成检查 |
| `results/elastic_net/exploratory_validation/` | 第 9 节配置、月度审计、验证比较及图表 |
| `results/elastic_net/xlf_diagnostics/` | 第 10 节行业损失、XLF 系数、窗口比较及图表 |

已经查看过的测试区间不会因重新执行 Notebook 而重新成为全新留出集。后续交接应保留现有验证/测试边界，区分复现检查、描述性诊断和新的模型选择。

## 8. 上传 GitHub 时随本说明一起交付

建议将本文件放在仓库根目录，随以下内容提交：

- `notebooks/03_elastic_net.ipynb`
- `README_elastic_net.md`
- `requirements.txt`
- `results/elastic_net/` 中本次正式结果、第 9 节和第 10 节诊断
- 现有上游数据与接口文件，保持与运行记录中的输入版本对应

不要把 `.venv/`、`.DS_Store`、`.ipynb_checkpoints/` 或本机临时文件作为交付材料。提交前确认 GitHub 上的 Notebook、预测表和结果目录属于同一套运行结果；不要将旧的部分运行结果与当前文件混用。

**交接范围：zyx 交付 Elastic Net 预测与预测层评价；下一棒完成统一交易回测及交易价值判断。本文件已经准备好供后续上传，不代表已提交或推送到 GitHub。**
