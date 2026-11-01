"""Finalize saved SPY-overlay results under the user's test-best selection rule.

Reads completed books only: no model fitting, trading replay or parameter search.
Historical protocols and manifests remain intact. Selection is explicitly ex post.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import nbformat

from backtest_spy_overlay import read_frame, save_frame, sha256, write_json, summary_row

VAL='results/backtest/spy_overlay_validation_20261101'
TEST='results/backtest/spy_overlay_test_20261101'
OUT='results/backtest/spy_overlay_final_20261101'
NOTE='SPY增强策略_最终结果与交接_20261101.md'
NOTEBOOK='notebooks/06_spy_overlay_final_results.ipynb'


def verify_sources(root):
    config=json.loads((root/TEST/'frozen_protocol.json').read_text(encoding='utf-8'))
    for name,expected in config['protected_hashes'].items():
        if sha256(root/name)!=expected: raise AssertionError(f'Historical protected file changed: {name}')
    manifest=json.loads((root/TEST/'all_steps_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['outputs'].items():
        if sha256(root/TEST/name)!=expected: raise AssertionError(f'Completed experiment changed: {name}')
    return len(config['protected_hashes'])


def choose(root,out):
    test=read_frame(root/TEST/'summary.csv')
    val=read_frame(root/VAL/'summary.csv')
    candidates=test.loc[test.kind.eq('spy_overlay')].copy()
    assert len(candidates)==12 and candidates.holding_intervals.nunique()==1
    candidates=candidates.sort_values(['annualized_net_return','series_id'],ascending=[False,True]).reset_index(drop=True)
    assert candidates.annualized_net_return.iloc[0]>candidates.annualized_net_return.iloc[1]
    assert candidates.series_id.tolist()==candidates.sort_values('ending_nav',ascending=False).series_id.tolist()
    candidates.insert(0,'test_return_rank',np.arange(1,len(candidates)+1))
    candidates['selected_for_final_presentation']=candidates.test_return_rank.eq(1)
    source=val.loc[val.kind.eq('spy_overlay'),['series_id','annualized_net_return','max_drawdown']].rename(
        columns={'annualized_net_return':'validation_annualized_net_return','max_drawdown':'validation_max_drawdown'})
    candidates=candidates.merge(source,on='series_id',validate='one_to_one')
    save_frame(out/'all_12_candidates_ranked.csv',candidates)
    selected=candidates.iloc[0]
    record=dict(selected_utc=datetime.now(timezone.utc).isoformat(),
        authority='User instruction: no further parameter changes; choose the best final test result',
        selection_criterion='Highest test-period net annualized return after execution, borrow and financing costs',
        candidate_universe='All 12 previously evaluated SPY-overlay model/scheme/threshold combinations',
        candidate_count=12,series_id=selected.series_id,model=selected.model,scheme=selected.scheme,
        entry_bps=float(selected.entry_bps),exit_bps=float(selected.exit_bps),
        test_net_annualized_return=float(selected.annualized_net_return),
        test_driven_selection=True,ex_post_selection=True,independent_test_validation_claim=False,
        new_parameter_values_tried=False,model_refitted=False,strategy_replayed=False,
        previous_labels='10/5 bp was the planned main comparison; 15/7.5 bp was the fixed reference. These were reporting roles, not performance rankings.',
        current_label='Selected final presentation strategy; all other results retained',
        source_summary_sha256=sha256(root/TEST/'summary.csv'))
    write_json(out/'selection_record.json',record)
    return selected,record


def selected_comparison(root,out,selected):
    rows,curves=[],[]
    for phase,base in [('validation',VAL),('test',TEST)]:
        saved=read_frame(root/base/'summary.csv')
        book=read_frame(root/base/f'experiment_{int(selected.entry_bps)}bp/daily.csv',parse_dates=['open_date'])
        book=book.loc[book.model.eq(selected.model)&book.scheme.eq(selected.scheme)].sort_values('open_date')
        ri=read_frame(root/TEST/('spy_reinvest_daily.csv' if phase=='test' else 'step2_diagnostics/validation_spy_reinvest_daily.csv'),parse_dates=['open_date']).sort_values('open_date')
        bench=read_frame(root/'results/backtest/investment_benchmarks/benchmark_daily.csv',parse_dates=['open_date'])
        bench=bench.loc[bench['split'].eq(phase)].copy(); bench['traded_notional']=bench.trade_notional
        spy=bench.loc[bench.benchmark.eq('spy_buy_hold')].sort_values('open_date')
        spy_returns=spy.iloc[1:].set_index('open_date').holding_interval_return
        cases=[(selected.series_id,book),("spy_buy_hold",spy),("spy_reinvest",ri),('cash',bench.loc[bench.benchmark.eq('cash')].sort_values('open_date'))]
        for name,g in cases:
            assert np.array_equal(g.open_date,ri.open_date)
            if name==selected.series_id:
                row=saved.loc[saved.series_id.eq(name)].iloc[0].to_dict()
            else:
                row,_=summary_row(g,series_id=name,kind='benchmark',model='',scheme='',entry_bps=np.nan,
                    spy_returns=spy_returns,spy_ending=float(spy.nav_post.iloc[-1]))
            active=g.holding_interval_return.iloc[1:].to_numpy()-ri.holding_interval_return.iloc[1:].to_numpy()
            te=float(np.std(active,ddof=1)*np.sqrt(252))
            row.update(split=phase,cumulative_relative_to_spy_reinvest=float(g.nav_post.iloc[-1]/ri.nav_post.iloc[-1]-1),
                annualized_net_return_difference_vs_spy_reinvest=float(row['annualized_net_return']-(ri.nav_post.iloc[-1]**(252/(len(ri)-1))-1)),
                annualized_tracking_error_vs_spy_reinvest=te,
                information_ratio_vs_spy_reinvest=float(np.mean(active)*252/te) if te>1e-12 else np.nan,
                final_presentation_role='selected' if name==selected.series_id else 'benchmark')
            rows.append(row)
            curves.append(g[['open_date','nav_post']].assign(split=phase,series_id=name))
    summary=pd.DataFrame(rows); curves=pd.concat(curves,ignore_index=True)
    save_frame(out/'selected_and_benchmarks.csv',summary); save_frame(out/'selected_nav_curves.csv',curves)
    for source,name in [('yearly_returns.csv','selected_yearly_returns.csv'),
        ('sector_contributions.csv','selected_sector_contributions.csv'),
        ('cost_and_wealth_decomposition.csv','selected_cost_decomposition.csv'),
        ('episode_concentration.csv','selected_episode_concentration.csv')]:
        frame=read_frame(root/TEST/'step2_diagnostics'/source)
        save_frame(out/name,frame.loc[frame.series_id.eq(selected.series_id)])
    return summary,curves


def figure(curves,selected,out):
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='row')
    for i,phase in enumerate(('validation','test')):
        data=curves.loc[curves['split'].eq(phase)]
        ri=data.loc[data.series_id.eq('spy_reinvest')].sort_values('open_date')
        for key,label,color,style in [(selected.series_id,f'Selected {selected.model.upper()} {selected.scheme} | {selected.entry_bps:g} / {selected.exit_bps:g} bp','#147d92','-'),
            ('spy_reinvest','SPY dividends reinvested','#25313d','--'),('spy_buy_hold','SPY fixed shares','#929292',':')]:
            g=data.loc[data.series_id.eq(key)].sort_values('open_date')
            axes[i,0].plot(g.open_date,g.nav_post,label=label,color=color,linestyle=style,linewidth=1.5)
            if key==selected.series_id:
                axes[i,1].plot(g.open_date,g.nav_post.to_numpy()/ri.nav_post.to_numpy()-1,color=color,linewidth=1.5)
        axes[i,1].axhline(0,color='#777777',linestyle='--',linewidth=1)
        axes[i,1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[i,0].set_title(f'{phase.title()}: net wealth',loc='left',weight='bold')
        axes[i,1].set_title(f'{phase.title()}: relative to reinvestment SPY',loc='left',weight='bold')
        for ax in axes[i]:
            ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        axes[i,0].set_ylabel('Initial capital = 1')
    fig.suptitle('Final presentation: GBDT S1 with SPY core | entry 15 bp, exit 7.5 bp',x=.06,ha='left',fontsize=14)
    fig.text(.06,.925,'Chosen ex post by highest test net CAGR among 12 candidates | All specified costs included | Periods start independently',color='#555555',fontsize=10)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(.02,.075,1,.91))
    fig.savefig(out/'selected_strategy.png',dpi=170);fig.savefig(out/'selected_strategy.svg');plt.close(fig)


def report(root,out,summary,selected,record):
    test=summary.loc[summary['split'].eq('test')].set_index('series_id')
    val=summary.loc[summary['split'].eq('validation')].set_index('series_id')
    own=test.loc[selected.series_id];ri=test.loc['spy_reinvest'];spy=test.loc['spy_buy_hold']
    lines=['# SPY 增强策略：最终结果与展示选择','',
        f'最终展示方案为 **{selected.model.upper()} {selected.scheme}，开仓 {selected.entry_bps:g} bp／退出 {selected.exit_bps:g} bp**，使用 100% SPY 基础仓位＋行业—SPY 多空调整。','',
        '选择口径按用户最新决定：在已完成的 12 个组合中，以扣除交易费、借券费及融资利息后的测试期净年化收益最高者作为最终展示方案。没有新增参数值、重训模型或重新回放交易。','',
        '## 主方案与对照组是什么意思','',
        '此前“10／5 bp 主方案”表示重点观察的实验版本；“15／7.5 bp 对照组”表示帮助比较降低门槛影响的固定参照。二者只是实验组织与展示角色，不是优劣评级，也不要求最后一定采用原来的主方案。','',
        '现按测试期结果选择最终展示方案，将此前标签保留在历史实验记录中。本报告及 selection_record.json 记录新的展示决定；其他 11 个组合完整保留。','',
        '## 测试期主要结果','',
        '区间 2023-01-04—2026-09-30，共 937 个持有区间，初始资金归一化为 1。','',
        '|指标|最终展示策略|SPY 分红再投入|SPY 固定股数|','|---|---:|---:|---:|',
        f'|净年化收益|{own.annualized_net_return:.3%}|{ri.annualized_net_return:.3%}|{spy.annualized_net_return:.3%}|',
        f'|累计净收益|{own.net_cumulative_return:.3%}|{ri.net_cumulative_return:.3%}|{spy.net_cumulative_return:.3%}|',
        f'|年化波动率|{own.annualized_net_volatility:.3%}|{ri.annualized_net_volatility:.3%}|{spy.annualized_net_volatility:.3%}|',
        f'|最大回撤|{own.max_drawdown:.3%}|{ri.max_drawdown:.3%}|{spy.max_drawdown:.3%}|','',
        f'相比分红再投入 SPY，最终展示策略的净年化收益高 **{own.annualized_net_return_difference_vs_spy_reinvest*100:.3f} 个百分点**，期末财富比高 **{own.cumulative_relative_to_spy_reinvest:.3%}**，年化跟踪误差为 **{own.annualized_tracking_error_vs_spy_reinvest:.3%}**，信息比率为 **{own.information_ratio_vs_spy_reinvest:.3f}**。这两个收益差指标的分母与复利定义不同，不能混用。','',
        '![最终展示策略：两阶段独立建账](selected_strategy.png)','',
        '## 持仓、换手与成本','',
        f'- 行业调整参与比例：{own.sector_participation:.3%}；其余时间继续持有 SPY，不再是全部现金。',
        f'- 行业平均持有时长：{own.mean_holding_intervals:.3f} 个开盘到开盘区间；每日检查信号，没有最低持有天数。',
        f'- 平均总敞口：{own.mean_gross_exposure:.3%}；最大调仓后总敞口：{own.max_gross_exposure:.3%}；平均空头敞口：{own.mean_short_exposure:.3%}。',
        f'- 年化双边换手：{own.annualized_two_sided_turnover:.3f} 倍，包括期初及期末。',
        f'- 累计交易费用：初始资金的 {own.total_execution_fee:.4%}；借券费：{own.total_borrow_fee:.4%}；融资利息：{own.total_financing_interest:.4%}；合计 {own.total_cost:.4%}。这些是整个测试区间累计成本，不是年化费率。','',
        '仓位、费用、融资与分红规则完全沿用既有实验：最多四个行业，初始每个行业 ±12.5%；行业空头超 25% 时统一缩放全部行业调整，SPY 权重为 1 减行业净权重。扣费后 NAV 配仓，调仓后总敞口上限 150%。','',
        'SPY 先合并净头寸后计算成交，单边交易费 3 bp；空头借券年化 1%，融资利率为当时可得的 DGS3MO＋1 个百分点；空头所得不返息，融资与借券分开按 ACT/365 计提。','',
        '## 验证期一并保留','',
        f'2018-01-03—2022-12-30 独立建账：所选同一组合净年化 {val.loc[selected.series_id,"annualized_net_return"]:.3%}；分红再投入 SPY 为 {val.loc["spy_reinvest","annualized_net_return"]:.3%}。验证期和测试期不拼接成新的独立样本外记录。逐年、行业、持仓段及成本来源见随附 CSV。','',
        '## 评价口径','',
        '**最终方案是看过测试结果后选出的本次 12 组中收益最高的一组。** 该选择遵照用户本次决定；因此测试收益属于事后选择后的探索性结果，不能称为未参与选择的独立盲测证据，也不据此声称统计显著或未来最优。','',
        '## 团队报告可直接采用的结论','',
        f'> 在本次探索性比较的 12 个 SPY 增强策略组合中，{selected.model.upper()} {selected.scheme} 配合 {selected.entry_bps:g}／{selected.exit_bps:g} bp 门槛取得最高的测试期成本后净年化收益，为 {own.annualized_net_return:.2%}。同期分红再投入 SPY 为 {ri.annualized_net_return:.2%}，对应年化收益增量约 {own.annualized_net_return_difference_vs_spy_reinvest*100:.2f} 个百分点，最大回撤为 {own.max_drawdown:.2%}。结果表明该组合在所研究区间取得了有限的成本后增强收益；全部候选组合均保留，最终展示方案依据已观察的测试表现选出。','',
        '## 交付文件','',
        '- selection_record.json：展示选择、候选范围和证据口径。',
        '- all_12_candidates_ranked.csv：全部候选按测试净年化排序，不隐去较弱结果。',
        '- selected_and_benchmarks.csv：验证、测试两阶段及现金、两种 SPY 基准。',
        '- selected_yearly_returns.csv / selected_sector_contributions.csv / selected_cost_decomposition.csv：已选组合的现有诊断。',
        '- selected_strategy.png / .svg：两阶段独立净值及相对 SPY 表现，可用于报告与展示。',
        '- report_English.md：可用于英文课程报告的简短结果段落。',
        '- 新增 notebook 06：只读取保存结果，不重新训练或交易回放。六个代码单元已在当前 Python 进程内顺序执行并保存输出；原生 Jupyter 内核启动未完成，因此不声称已通过原生内核验证。原 notebook 05 及旧交接文件继续作为历史快照。','']
    (out/'report_中文.md').write_text('\n'.join(lines),encoding='utf-8')
    english=(f'Among the twelve previously evaluated exploratory SPY-core sector-overlay configurations, '
        f'{selected.model.upper()} {selected.scheme} with {selected.entry_bps:g} bps entry and {selected.exit_bps:g} bps exit thresholds '
        f'achieved the highest test-period net CAGR. From 4 January 2023 to 30 September 2026, its net CAGR was '
        f'{own.annualized_net_return:.2%}, compared with {ri.annualized_net_return:.2%} for SPY with dividends reinvested '
        f'and {spy.annualized_net_return:.2%} for fixed-share SPY. The annualized return difference versus reinvestment SPY '
        f'was {own.annualized_net_return_difference_vs_spy_reinvest*100:.2f} percentage points, with '
        f'{own.cumulative_relative_to_spy_reinvest:.2%} higher terminal wealth, tracking error of '
        f'{own.annualized_tracking_error_vs_spy_reinvest:.2%}, an information ratio of '
        f'{own.information_ratio_vs_spy_reinvest:.2f}, and maximum drawdown of {own.max_drawdown:.2%}. '
        'All figures include the specified execution, stock-borrow and financing costs. '
        'The final presentation configuration was selected ex post using observed test performance, '
        'so these results are exploratory and are not an independent, untouched holdout estimate. '
        'All twelve candidate results are retained.\n')
    (out/'report_English.md').write_text(english,encoding='utf-8')
    root_note=f'''# SPY 增强策略：当前最终交接（2026-11-01）

当前展示方案为 **GBDT S1，15／7.5 bp**。按用户最新决定，以已完成 12 组中测试期成本后净年化收益最高者作最终展示；不再新增参数或实验。此前“10 bp 主方案／15 bp 参照”只是实验角色，历史文件保留。

先看 [最终中文报告]({OUT}/report_中文.md)、[全部候选排序]({OUT}/all_12_candidates_ranked.csv) 和 [两阶段指标]({OUT}/selected_and_benchmarks.csv)。测试净年化为 **{own.annualized_net_return:.2%}**，同期分红再投入 SPY 为 **{ri.annualized_net_return:.2%}**。这是一项看过测试结果后的展示选择，报告已标注探索性。

可供课程报告使用的 [英文段落]({OUT}/report_English.md)、[PNG 图]({OUT}/selected_strategy.png) 和 [SVG 图]({OUT}/selected_strategy.svg) 已保存。

新增结果阅读入口为 [06_spy_overlay_final_results.ipynb]({NOTEBOOK})，只读已保存结果。六个代码单元已在当前 Python 进程内顺序执行并保存输出，原生 Jupyter 内核尚未验证。原 README、notebook 05、2026-10-31 交接、验证期与测试期协议及结果均保留为原快照。当前新增决策以本说明和 [selection_record.json]({OUT}/selection_record.json) 为准。

汇报包只包含结果、图表、阅读 notebook、说明和相关脚本，不含完整原始数据与大账本，适合发给组员审阅，不能单独复现上游模型或完整回测。完整复现仍使用当前项目目录。
'''
    (root/NOTE).write_text(root_note,encoding='utf-8')


def notebook(root,out):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    nb=nbformat.v4.new_notebook()
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'))
    nb.cells=[nbformat.v4.new_markdown_cell(
        '# SPY 增强策略：最终结果阅读\n\n最终展示方案按已完成 12 组中的测试期净年化收益最高者选择。'
        '这是事后选择的探索性结果；本 notebook 只读取保存数据，不训练模型、不执行交易回放。'),
        nbformat.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in (Path.cwd(), Path.cwd().parent) if (p / 'results/backtest/spy_overlay_final_20261101').is_dir())
BASE = ROOT / 'results/backtest/spy_overlay_final_20261101'
selection = json.loads((BASE / 'selection_record.json').read_text(encoding='utf-8'))
assert selection['ex_post_selection'] and not selection['strategy_replayed']
display(Markdown(f"**最终展示：{selection['model'].upper()} {selection['scheme']}，{selection['entry_bps']:g}／{selection['exit_bps']:g} bp**"))
print('按已观察的测试期成本后净年化收益选择；没有新增参数或回放。')"""),
        nbformat.v4.new_code_cell("""summary = pd.read_csv(BASE / 'selected_and_benchmarks.csv')
fields = ['split','series_id','annualized_net_return','annualized_net_volatility','max_drawdown','cumulative_relative_to_spy_reinvest']
table = summary[fields].copy()
for col in fields[2:]: table[col] = table[col].map(lambda x: f'{x:.3%}')
display(table)
print('两阶段独立初始化；相对表现为期末财富比减一。')"""),
        nbformat.v4.new_code_cell("""ranked = pd.read_csv(BASE / 'all_12_candidates_ranked.csv')
assert len(ranked) == 12 and ranked.iloc[0]['series_id'] == selection['series_id']
fields = ['test_return_rank','series_id','annualized_net_return','validation_annualized_net_return','selected_for_final_presentation']
table = ranked[fields].copy()
for col in ['annualized_net_return','validation_annualized_net_return']:
    table[col] = table[col].map(lambda x: f'{x:.3%}')
display(table)
print('完整保留其他组合，不仅展示收益最高者。')"""),
        nbformat.v4.new_code_cell("display(Image(filename=str(BASE / 'selected_strategy.png')))"),
        nbformat.v4.new_code_cell("""yearly = pd.read_csv(BASE / 'selected_yearly_returns.csv')
table = yearly[['split','year','partial_year','net_period_return','spy_reinvest_period_return','relative_period_return']].copy()
for col in ['net_period_return','spy_reinvest_period_return','relative_period_return']:
    table[col] = table[col].map(lambda x: f'{x:.3%}')
display(table)
print('2026 年仅截至 9 月 30 日；此处不将部分年度收益再年化。')"""),
        nbformat.v4.new_code_cell("""selected = summary.loc[summary['series_id'].eq(selection['series_id'])].copy()
fields = ['split','sector_participation','mean_holding_intervals','mean_gross_exposure','max_gross_exposure',
          'annualized_two_sided_turnover','total_execution_fee','total_borrow_fee','total_financing_interest','total_cost']
display(selected[fields])
print('成本金额均以初始资金 1 为单位，累计金额不是年化费率。所有计算来自已保存账本。')""")]
    config=Config();config.HistoryManager.hist_file=':memory:'
    shell=InteractiveShell.instance(config=config)
    old_cwd=Path.cwd();os.chdir(root)
    try:
        for index,cell in enumerate((c for c in nb.cells if c.cell_type=='code'),start=1):
            with capture_output(stdout=True,stderr=True,display=True) as captured:
                result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec is not None or result.error_in_exec is not None:
                raise RuntimeError(f'Read-only notebook cell {index} failed: {result.error_before_exec or result.error_in_exec}')
            cell.execution_count=index;cell.outputs=[]
            if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for rich in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    finally:
        os.chdir(old_cwd)
    nb.metadata['verification']=dict(mode='Sequential IPython cells in the current Python process',
        native_jupyter_kernel_verified=False,note='Native kernel startup did not complete; no claim of native-kernel execution')
    nbformat.validate(nb)
    nbformat.write(nb,root/NOTEBOOK)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert len(cells)==6 and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    print('Notebook 06: six read-only cells executed sequentially in the current Python process; native kernel unverified',flush=True)
    return len(cells)


def package(root,out):
    files=[root/NOTE,root/NOTEBOOK,Path(__file__)]
    files.extend(p for p in out.iterdir() if p.is_file())
    for base in (VAL,TEST):
        files.extend(p for p in (root/base).iterdir() if p.is_file())
    diagnostics=root/TEST/'step2_diagnostics'
    files.extend(p for p in diagnostics.iterdir() if p.is_file() and not p.name.endswith('_daily_attribution.csv'))
    files.extend(root/'scripts'/name for name in ['backtest_spy_overlay.py','check_spy_overlay.py',
        'backtest_spy_overlay_test.py','check_spy_overlay_test.py','diagnose_spy_overlay.py'])
    files=sorted(set(files))
    payload={p.relative_to(root).as_posix():sha256(p) for p in files}
    marker=dict(profile='SPY-overlay final-results review package',self_contained_model_replay=False,
        notebook='notebooks/06_spy_overlay_final_results.ipynb',files=payload,
        note='Contains final results and summaries, not complete raw data or daily trade books. Notebook 06 reads packaged final results only; current-process execution verified, native Jupyter kernel unverified.')
    folder=root.parent/'交接打包_yae_20261101';folder.mkdir(exist_ok=True)
    archive=folder/'DSA5205_SPY增强_结果汇报包_20261101.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.writestr('review_package_manifest.json',json.dumps(marker,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in payload.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    detached=dict(archive_name=archive.name,sha256=sha256(archive),size_bytes=archive.stat().st_size,
        payload_files=len(payload),profile=marker['profile'],all_packaged_hashes_verified=True)
    write_json(folder/'archive_manifest.json',detached)
    return archive,detached


def run(root):
    root=root.resolve();out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    count=verify_sources(root)
    selected,record=choose(root,out)
    print(f"Final presentation selected: {selected.series_id}; no trading replay",flush=True)
    summary,curves=selected_comparison(root,out,selected)
    figure(curves,selected,out);report(root,out,summary,selected,record)
    executed=notebook(root,out)
    verify_sources(root)
    outputs={p.relative_to(root).as_posix():sha256(p) for p in out.iterdir() if p.is_file() and p.name!='final_manifest.json'}
    outputs.update({NOTE:sha256(root/NOTE),NOTEBOOK:sha256(root/NOTEBOOK)})
    write_json(out/'final_manifest.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        selected_series_id=selected.series_id,selection_is_ex_post=True,
        no_parameter_tuning=True,no_model_refitting=True,no_trading_replay=True,
        historical_protected_files=count,all_historical_inputs_outputs_unchanged=True,
        candidate_count=12,selected_benchmark_rows=len(summary),current_process_executed_cells=executed,native_jupyter_kernel_verified=False,
        code_sha256=sha256(Path(__file__)),outputs=outputs))
    archive,manifest=package(root,out)
    print(summary.loc[summary['split'].eq('test'),['series_id','annualized_net_return','max_drawdown',
        'cumulative_relative_to_spy_reinvest']].to_string(index=False),flush=True)
    print(f'PASS: {count} historical files and all completed experiment outputs unchanged',flush=True)
    print(f'Review ZIP: {archive} ({manifest["size_bytes"]} bytes, all {manifest["payload_files"]} payload hashes verified)',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().project_root)
