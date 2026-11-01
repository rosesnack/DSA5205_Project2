# yae：统一交易回测最终交接

交付日期：2026-10-31。第1—15项已完成，第16项可选扩展未执行。本交付整理现有代码、协议、账本、指标和图表，不重训、不重新选门槛、不重复已完成回测。

## 先看什么

1. 阅读本说明，了解结论、权威配置与使用边界。
2. 阅读[中文结果报告](results/backtest/reporting/report_中文.md)：两阶段完整指标、损益来源、成本敏感性和六张图。
3. 用[主结果表](results/backtest/reporting/main_metrics.csv)、[全部敏感性指标](results/backtest/reporting/sensitivity_metrics.csv)及 PNG/SVG 撰写团队报告或制作展示。
4. 需要追溯过程时，查[回测进度与说明](回测进度与说明_yae.md)、[文件清单](results/backtest/handoff/file_inventory.csv)和各阶段 manifest。

Notebook 入口为[05_portfolio_backtest.ipynb](notebooks/05_portfolio_backtest.ipynb)。现有17个代码单元及保存输出未改；第15项只更新其中的交接说明。17个单元此前已在同一Python进程顺序执行，含两张嵌入图；**尚未验证原生Jupyter内核执行**。

## 主要结果如何解释

在当前模型、数据和交易规则下，GBDT S1/S2具有有限的正现金超额表现，Elastic Net及GBDT S0的测试交易表现没有超过现金。S2未显示比S1更好的增量效果。结果适合支持“跨行业信息在GBDT中带来有限增益”的审慎结论，不足以宣称普遍有效、统计显著或可直接实盘盈利。

测试阶段为2023-01-04开盘至2026-09-30开盘，主成本结果如下。表内金额收益均已计费用及融资；Sharpe减去同区间现金收益，最大回撤采用负值口径。

| 策略或基准 | 年化净收益 | 年化净波动率 | 现金超额Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| Elastic Net S0 | 4.688% | 0.337% | -0.399 | -0.577% |
| Elastic Net S1 | 4.688% | 0.337% | -0.399 | -0.577% |
| Elastic Net S2 | 4.688% | 0.337% | -0.399 | -0.577% |
| GBDT S0 | 4.420% | 1.558% | -0.231 | -2.275% |
| GBDT S1 | 5.534% | 1.329% | 0.529 | -1.199% |
| GBDT S2 | 5.440% | 1.322% | 0.464 | -1.199% |
| 现金 | 4.808% | 0.186% | N/A | 0.000% |
| SPY固定份额买入持有 | 21.672% | 14.896% | 1.076 | -19.261% |

- GBDT S1/S2相对现金的年化收益差约0.73/0.63个百分点；不能把全部约5.5%的收益都归为模型贡献。其持仓区间比例约13%，较低波动和回撤也与大量时间持有现金有关。
- 在预先固定的8个成本/融资设置下，GBDT S1/S2测试期均保持正现金超额，Elastic Net三组和GBDT S0均为负。这只说明在这些假设内方向保持一致，没有选择最优成本，也未据此再次挑模型。
- SPY为满仓市场多头，配对策略风险敞口不同。SPY绝对收益较高是真实比较结果，但不是同风险条件下的胜负判断。
- 三组Elastic Net测试预测相同，交易账本也精确相同，符合建模同学的交接结果。GBDT S1稍优、S2没有额外优势，也与预测评价方向一致。这是同一批预测在交易端的延伸验证，不能称为独立重复实验。

验证期是2018-01-03开盘至2022-12-30开盘。GBDT三组年化净收益约3.13%—3.53%，现金约1.32%；完整验证结果同样保留，不与测试区间拼接成一条独立样本外业绩。

## 必须保留的评价边界

1. 验证期已经用于模型参数与公共门槛选择，其结果属于选择后的证据。
2. 原建模阶段已经查看测试预测评价；GBDT配置记录了 `Exploratory grid; previous project test results already known`。本次是固定规则下的测试期交易评价，不能称为从未接触的全盲留出集。
3. 第9项是独立补算的训练期前推诊断，未替换原正式模型或正式预测。
4. 不根据测试表现修改模型、门槛、主成本或删除弱结果。全部六组策略、两阶段及固定八个成本设置均保留。
5. 账务和确定性检查通过，说明当前快照的实现与记录一致，不证明预测有效、统计显著或未来收益。成本敏感性也不是全部市场风险的检验。
6. 2026年年度收益只到9月30日，年度热图不将这段收益再年化；整个阶段的CAGR按冻结的252交易日口径计算。

两位建模负责人的原说明：[Elastic Net / zyx](Elastic%20Net%20建模与预测交接说明_zyx.md)、[GBDT / cdd](GBDT%20建模与预测交接说明_cdd.md)。原始说明与原模型评价均未改写。

## 哪个文件说了算

| 内容 | 当前权威来源 | 说明 |
|---|---|---|
| 研究与交易规则 | [5205proj2.md](../5205proj2.md) | 位于仓库上一层，压缩包保留这一相对位置 |
| 公共门槛 | [selected_threshold.json](results/backtest/threshold_selection/selected_threshold.json) | 开仓15、退出7.5 bps，即0.0015/0.00075 |
| 冻结回测协议 | [protocol_config.json](results/backtest/phase_evaluation/protocol_config.json) | 阶段、资格、配对、执行、账务、费用、统计及证据边界 |
| 全部主结果 | [main_metrics.csv](results/backtest/reporting/main_metrics.csv) | 12条策略与4条投资基准路径，两阶段分别报告 |
| 全部固定场景 | [cost_scenarios.csv](results/backtest/reporting/cost_scenarios.csv)、[sensitivity_metrics.csv](results/backtest/reporting/sensitivity_metrics.csv) | 8个设置，共96行策略指标 |
| 第14项完成状态 | [reporting/run_manifest.json](results/backtest/reporting/run_manifest.json) | 指标、图表、报告与敏感性登记身份 |
| 第15项完成状态 | [handoff/run_manifest.json](results/backtest/handoff/run_manifest.json) | 本次整理、文件清单、原文件保护及检查 |
| 压缩包身份 | 包旁的 `archive_manifest.json` | 两个ZIP的SHA-256、大小、成员数和逐成员验证结论 |

**历史快照不能互相覆盖。**旧成本/诊断文件里的“门槛未选定”，以及旧manifest里的“第11—15项待完成”，均描述生成时的状态，保留原样。当前状态以对应新阶段的manifest和本说明为准。

## 已冻结的执行与评价口径

- 每阶段、每策略均从现金1和空仓开始，期末清仓，不跨阶段承接。验证为1,258个开盘/1,257段收益，测试为938个开盘/937段收益。
- 最多4组行业/SPY配对，每条腿按交易前NAV配置±12.5%，不放大闲置仓位。先保留/退出旧仓，新仓只从当日绝对预测前四名选择；并列使用固定ticker顺序，不用第五名补位。
- 每日恢复目标权重，无容忍带；SPY先合并，再按实际净份额差交易。空仓日期仍参与收益和风险统计。
- 主执行费单边3 bps；借券年化1%，按实际净空头起始金额和ACT/365计提；卖空抵押现金不返息。现金利息使用现有相邻开盘RF，不另除252/365。
- 使用观测open/dividends。分红归旧份额，空头支付分红；价格损益、分红、利息、费用均只入账一次，不额外叠加预测标签收益。
- SPY基准首日买固定小数份额，由本金1同时支付本金与入场费；分红留现金、不再投资；同一期末开盘出售并收费。现金基准按同区间RF复利累计。
- CAGR按终值和252/区间数计算。波动率为净收益样本标准差乘sqrt(252)；Sharpe使用同区间现金超额。最大回撤计入初始资本1及首日入场费。双边换手包含首尾，不除以二；持仓比例按区间起点判断。

完整细则与容差不在此重新定义，直接引用冻结协议与各阶段design_config。

## 文件位置与用途

继续沿用 `scripts/`、`notebooks/` 和 `results/backtest/`，没有移动或重命名既有成果。代码、配置和账本通过以下路径连接。

| 环节 | 代码 | 结果目录或文件 |
|---|---|---|
| 接收、时间、资格、预测基准 | `prepare_backtest_inputs.py`、`backtest_signal_policy.py`、`backtest_baselines.py` | `results/backtest/`根目录的已登记表与manifest |
| 配对、目标、净交易、账务、费用 | `backtest_pair_decisions.py`、`backtest_execution.py`、`backtest_ledger.py`、`backtest_costs.py` | 根目录预览为工程案例，不作最终业绩 |
| 独立训练诊断 | `backtest_training_oof.py`、`backtest_oof_parallel.py` | `training_oof/`，63,594条独立诊断预测 |
| 公共门槛选择 | `backtest_threshold_selection.py` | `threshold_selection/`，全部5候选×6策略保留 |
| 阶段评价 | `backtest_phase_evaluation.py` | `phase_evaluation/`，复用验证、新增测试 |
| 投资基准 | `backtest_investment_benchmarks.py` | `investment_benchmarks/`，两阶段SPY/现金完整账本 |
| 独立审计 | `backtest_consistency_audit.py` | `consistency_audit/`，1,049项检查与重建误差 |
| 指标、敏感性、图表、报告 | `backtest_reporting.py`、`backtest_report_figures.py` | `reporting/`，16条主路径、96行场景、6张PNG/SVG |
| 本次整理、校验与打包 | `backtest_handoff.py` | `handoff/`及仓库外两个ZIP |

以上脚本均位于仓库 `scripts/`；结果目录均相对 `results/backtest/`。

**完整账本不要找根目录的 `*_demo_*` 或 `cost_preview_*`。**正式主成本及敏感性明细的位置为：

| 路径 | 内容 |
|---|---|
| `threshold_selection/checkpoints/entry_15_{daily,assets,trades,returns,summary}.csv` | 正式验证主成本；第11项复用这套15bps账本，没有再生成一份验证副本 |
| `phase_evaluation/test_{daily,assets,trades,returns,summary}.csv` | 正式测试六组主成本 |
| `investment_benchmarks/benchmark_{daily,trades,returns,summary}.csv` | 两阶段SPY与现金四条路径 |
| `reporting/sensitivity/checkpoints/` | 14批替代设置，每批含daily/assets/trades/returns/metrics及JSON断点 |

账本关键字段：

| 表 | 键与主要字段 | 读取要点 |
|---|---|---|
| daily | `split, model, scheme, open_date`；`cash_pre/post, nav_pre/post, cash_interest, borrow_fee, execution_fee, dividend_cash, price_pnl` | `date`为信号日，`open_date`才是实际结算/成交开盘日；期初行无真实收益区间 |
| assets | 上述键加`ticker`；`current_shares, target_shares, delta_shares, open_price, target_weight` | 份额带符号；SPY已合并；费用按净变化计算 |
| trades | assets中实际非零交易 | 不能将未发生的配对名义腿另算费用 |
| returns | `interval_entry_date, open_date`；`holding_interval_return, interval_risk_free_return, cash_excess_return, held_any_pair` | 所有真实区间保留；首段分母处理已包含首次交易费 |
| metrics | `split, series_id`或`model, scheme, scenario_id`；收益、波动、Sharpe、回撤、换手、持仓比例 | CSV收益/波动为小数单位；表内百分数只用于展示，不能再按百分数读取CSV |

主路径账本以阶段/策略为键；敏感性读取时还须保留`scenario_id`。基准表使用`benchmark`区分SPY和cash，不套用行业配对字段。

## 如何查看与复核

**先校验保存结果，不执行历史生产命令。**在完整仓库或解压后的仓库目录中运行：

```powershell
python -B scripts\backtest_handoff.py --verify-only
```

本命令只用Python标准库，检查清单、文件集合和SHA-256，缺失或变化即报错；不导入模型或回测引擎，不重写结果。精简包会根据内置profile只验证其应有成员；需要强制检查完整包时可加 `--profile full`。没有profile的原工作目录默认检查完整快照。

若连同本机保存的两个ZIP一起检查：

```powershell
python -B scripts\backtest_handoff.py --verify-only --archives-dir "..\交接打包_yae_20261031"
```

也可直接用PowerShell查看ZIP的SHA-256，并与包旁 `archive_manifest.json` 比较：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "..\交接打包_yae_20261031\DSA5205_Project2_yae_full_20261031.zip"
```

现有Notebook 05在完整文件及原环境下会核对并读取已完成缓存。缺失或损坏文件应先恢复归档，不能通过删除manifest或运行旧阶段命令来“修复”已登记结果。README后半部分保留的是历史生产命令，它们可能写文件或触发昂贵拟合，不是本次推荐的阅读/校验入口。

原 `requirements.txt`保留建模交付时的版本声明，**不是本次回测的环境锁文件**。实际运行Python 3.11.5、numpy 1.24.3、pandas 2.0.3、matplotlib 3.7.2；训练诊断使用sklearn 1.3.0。本次[环境快照](results/backtest/handoff/environment_snapshot.json)读取已安装包的版本元数据，没有安装或调整依赖。未声称跨版本逐位复现。

## 压缩包与验证证据

压缩包位于项目工作目录下的 `交接打包_yae_20261031/`，在仓库外，避免把ZIP递归装入自身。两个包均保留“外层目录 / 5205proj2.md / DSA5205_Project2-main”的原相对关系。

| 文件 | 用途 |
|---|---|
| `DSA5205_Project2_yae_review_20261031.zip` | 给组员快速阅读：本说明、原交接、Notebook 05已保存输出、代码、配置、主要表格、六张PNG/SVG及校验清单；省略源数据与完整大账本，不能执行完整Notebook |
| `DSA5205_Project2_yae_full_20261031.zip` | 完整仓库快照加原规范，含原数据、01—04 Notebook、正式预测、全部历史结果、主成本和敏感性账本；排除环境、缓存和版本库内部文件 |
| `archive_manifest.json` | 两包SHA-256、大小、成员数及逐成员核验结论；位于包外，避免自身哈希循环 |

精简包的原仓库README仍含部分指向完整数据的链接；查看主结论请从本说明和中文报告开始。文件清单同时标明每个文件是否进入精简包；不是将精简包误称为完整复现数据。

`handoff/`保留整理前快照、环境记录、上游通过记录、逐文件SHA-256/字节数/用途/包归属、交接检查及当前完成manifest。第9—14项既有检查数分别为41、64、59、72、1,049、215；这些是各阶段接受检查，不能相加后当作独立实验次数。

本次对原数据、01—04 Notebook、正式预测和所有此前结果逐文件核对。只更新仓库README、进度说明和Notebook 05的说明文字，新增本交接说明、标准库打包脚本及handoff记录。17个代码单元与已有输出的JSON内容保持一致，原脚本全部保留。

第16项Ridge或九行业等权多头等扩展未执行；需要单独明确后才开展。当前可直接将已有主结果、敏感性结果与图表用于团队报告整合，不需要为了交接重新训练或再次选门槛。
