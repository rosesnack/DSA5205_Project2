"""Read-only attribution of frozen validation/test SPY overlay results.

No strategy replay or parameter changes. Adds a same-account SPY-reinvestment
comparator for validation, and decomposes actual ledger differences exactly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

from backtest_phase_evaluation import load_phase_inputs
from backtest_spy_overlay import TICKERS, EPS, holding_episodes, read_frame, save_frame, sha256, write_json
from backtest_spy_overlay_test import OUTPUT, replay_spy_reinvest, verify


def yearly_returns(book, reference, phase, series):
    book=book.sort_values('open_date').reset_index(drop=True)
    reference=reference.sort_values('open_date').reset_index(drop=True)
    if not np.array_equal(book.open_date,reference.open_date):
        raise AssertionError('Annual comparison dates differ')
    interval=book.iloc[1:].copy()
    interval['reference_return']=reference.holding_interval_return.iloc[1:].to_numpy()
    records=[]
    for year,group in interval.groupby(interval.open_date.dt.year):
        own=float(np.prod(1+group.holding_interval_return)-1)
        ref=float(np.prod(1+group.reference_return)-1)
        diff=group.holding_interval_return-group.reference_return
        te=float(diff.std(ddof=1)*np.sqrt(252))
        records.append(dict(split=phase,series_id=series,year=int(year),intervals=len(group),
            first_interval_exit=group.open_date.iloc[0],last_interval_exit=group.open_date.iloc[-1],
            partial_year=bool(year==2026),net_period_return=own,spy_reinvest_period_return=ref,
            relative_period_return=(1+own)/(1+ref)-1,return_difference=own-ref,
            annualized_tracking_error=te,information_ratio=float(diff.mean()*252/te) if te>EPS else np.nan))
    return records


def attribution(book, assets, benchmark):
    """Exact wealth-difference identity, in units of initial capital (not alpha).

    Each sector's paired contribution = old signed sector value *
    (sector total return - SPY total return). The residual SPY term captures
    different capital paths. Costs are kept at portfolio level, without
    inventing allocations of shared SPY trades or funding to individual sectors.
    """
    b=book.sort_values('open_date').reset_index(drop=True)
    ri=benchmark.sort_values('open_date').reset_index(drop=True)
    if not np.array_equal(b.open_date,ri.open_date):
        raise AssertionError('Attribution date mismatch')
    spy=assets.loc[assets.ticker.eq('SPY')].sort_values('open_date').reset_index(drop=True)
    if not np.array_equal(spy.open_date,b.open_date):
        raise AssertionError('Missing unique net SPY asset record')
    spy_interval=(spy.open_price-spy.previous_open+spy.observed_dividend_per_share)/spy.previous_open
    spy_interval.index=b.open_date
    sector=assets.loc[assets.ticker.isin(TICKERS)].copy()
    sector['sector_start_value']=sector.current_shares*sector.previous_open
    sector['spy_interval_total_return']=sector.open_date.map(spy_interval)
    sector['sector_price_dividend_pnl']=sector.price_pnl_old+sector.dividend_cash_old
    sector['equivalent_spy_pnl']=sector.sector_start_value*sector.spy_interval_total_return
    sector['paired_contribution']=sector.sector_price_dividend_pnl-sector.equivalent_spy_pnl
    sector['year']=sector.open_date.dt.year
    sector['direction']=np.sign(sector.current_shares).astype(int)
    grouped=sector.groupby('open_date')[['paired_contribution','equivalent_spy_pnl']].sum().reindex(b.open_date)
    daily=b[['open_date','nav_post','price_pnl','dividend_cash','cash_interest','execution_fee','borrow_fee','financing_interest']].copy()
    daily['paired_sector_contribution']=grouped.paired_contribution.to_numpy()
    daily['capital_path_spy_contribution']=(spy.price_pnl_old+spy.dividend_cash_old).to_numpy()+grouped.equivalent_spy_pnl.to_numpy()-ri.price_pnl.to_numpy()-ri.dividend_cash.to_numpy()
    daily['cash_interest_difference']=b.cash_interest.to_numpy()-ri.cash_interest.to_numpy()
    daily['execution_fee_difference']=b.execution_fee.to_numpy()-ri.execution_fee.to_numpy()
    daily['borrow_fee_difference']=b.borrow_fee.to_numpy()-ri.borrow_fee.to_numpy()
    daily['financing_interest_difference']=b.financing_interest.to_numpy()-ri.financing_interest.to_numpy()
    daily['wealth_difference']=b.nav_post.to_numpy()-ri.nav_post.to_numpy()
    daily['wealth_difference_change']=daily.wealth_difference.diff().fillna(daily.wealth_difference.iloc[0])
    daily['explained_difference_change']=(daily.paired_sector_contribution+daily.capital_path_spy_contribution
        +daily.cash_interest_difference-daily.execution_fee_difference-daily.borrow_fee_difference-daily.financing_interest_difference)
    daily['reconciliation_error']=daily.wealth_difference_change-daily.explained_difference_change
    if daily.reconciliation_error.abs().max()>EPS:
        raise AssertionError('Attribution does not reconstruct actual wealth difference')
    return daily,sector


def episode_results(assets, sector, series):
    records=holding_episodes(assets,series)
    result=[]
    for row in records:
        part=sector.loc[sector.ticker.eq(row['ticker']) & sector.open_date.gt(row['entry_open'])
            & sector.open_date.le(row['exit_open'])]
        result.append(row | dict(paired_contribution=float(part.paired_contribution.sum()),
            sector_price_dividend_pnl=float(part.sector_price_dividend_pnl.sum())))
    # Shared SPY/funding costs remain at portfolio level; no arbitrary episode allocation.
    frame=pd.DataFrame(result)
    if len(frame) and abs(frame.paired_contribution.sum()-sector.paired_contribution.sum())>EPS:
        raise AssertionError('Holding episodes omit or duplicate paired PnL')
    positive=frame.loc[frame.paired_contribution.gt(0),'paired_contribution'].sort_values(ascending=False) if len(frame) else pd.Series(dtype=float)
    total=float(positive.sum())
    concentration=dict(series_id=series,completed_episodes=len(frame),positive_episodes=len(positive),
        total_positive_paired_contribution=total,total_negative_paired_contribution=float(frame.loc[frame.paired_contribution.lt(0),'paired_contribution'].sum()) if len(frame) else 0.,
        top1_share_of_positive=float(positive.head(1).sum()/total) if total>EPS else np.nan,
        top5_share_of_positive=float(positive.head(5).sum()/total) if total>EPS else np.nan,
        top10_share_of_positive=float(positive.head(10).sum()/total) if total>EPS else np.nan)
    return frame,concentration


def load_experiments(root,phase):
    base=root/(OUTPUT if phase=='test' else 'results/backtest/spy_overlay_validation_20261101')
    for entry in (10,15):
        path=base/f'experiment_{entry}bp'
        book=read_frame(path/'daily.csv',parse_dates=['open_date'])
        assets=read_frame(path/'assets.csv',parse_dates=['open_date'])
        for (model,scheme),group in book.groupby(['model','scheme'],sort=True):
            selected=assets.model.eq(model)&assets.scheme.eq(scheme)
            yield entry,model,scheme,group,assets.loc[selected]


def run(root):
    root=root.resolve(); base=root/OUTPUT; out=base/'step2_diagnostics'
    if not (base/'step1_manifest.json').exists(): raise RuntimeError('Complete frozen test experiment first')
    config=json.loads((base/'frozen_protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((base/'step1_manifest.json').read_text(encoding='utf-8'))
    verify(root,config['protected_hashes'])
    for path,digest in manifest['outputs'].items():
        if sha256(base/path)!=digest: raise AssertionError(f'Step 1 artifact changed: {path}')
    out.mkdir(exist_ok=True)
    logic=validate_logic()
    save_frame(out/'logic_checks.csv',pd.DataFrame(logic))
    annual,sector_rows,year_sector_rows,cost_rows,episode_rows,concentrations,checks,events=[],[],[],[],[],[],[],[]
    for phase in ('validation','test'):
        if phase=='test':
            reference=read_frame(base/'spy_reinvest_daily.csv',parse_dates=['open_date'])
        else:
            (_,_,prices,rates,opens),_=load_phase_inputs(root,phase)
            reference,reference_assets=replay_spy_reinvest(prices,rates,opens,phase)
            save_frame(out/'validation_spy_reinvest_daily.csv',reference)
            save_frame(out/'validation_spy_reinvest_assets.csv',reference_assets)
        for entry,model,scheme,book,assets in load_experiments(root,phase):
            series=f'overlay_{entry}_{model}_{scheme}'
            daily,sector=attribution(book,assets,reference)
            metadata=dict(split=phase,series_id=series,entry_bps=entry,model=model,scheme=scheme)
            annual.extend(yearly_returns(book,reference,phase,series))
            event_book=book.sort_values('open_date').copy()
            directions=event_book.target_directions.fillna('')
            event_book['event_type']=np.where(directions.eq(directions.shift(1)),
                'same_directions_rebalance','sector_direction_set_changed')
            event_book.loc[event_book.is_initial_open,'event_type']='initial_build'
            event_book.loc[event_book.is_final_open,'event_type']='final_liquidation'
            for event,g in event_book.groupby('event_type'):
                events.append(metadata|dict(event_type=event,opening_records=len(g),
                    execution_fee=float(g.execution_fee.sum()),traded_notional=float(g.traded_notional.sum()),
                    share_of_execution_fee=float(g.execution_fee.sum()/book.execution_fee.sum())))
            for ticker,group in sector.groupby('ticker'):
                sector_rows.append(metadata|dict(ticker=ticker,paired_contribution=float(group.paired_contribution.sum()),
                    held_intervals=int(group.current_shares.ne(0).sum()),
                    sector_price_dividend_pnl=float(group.sector_price_dividend_pnl.sum()),equivalent_spy_pnl=float(group.equivalent_spy_pnl.sum())))
            for (year,ticker),group in sector.groupby(['year','ticker']):
                year_sector_rows.append(metadata|dict(year=int(year),ticker=ticker,paired_contribution=float(group.paired_contribution.sum())))
            ep,concentration=episode_results(assets,sector,series)
            episode_rows.append(ep.assign(split=phase,entry_bps=entry,model=model,scheme=scheme))
            concentrations.append(metadata|concentration)
            components=['paired_sector_contribution','capital_path_spy_contribution','cash_interest_difference',
                'execution_fee_difference','borrow_fee_difference','financing_interest_difference']
            cost_rows.append(metadata|{name:float(daily[name].sum()) for name in components}
                |dict(final_wealth_difference=float(daily.wealth_difference.iloc[-1]),
                    max_daily_reconciliation_error=float(daily.reconciliation_error.abs().max()),
                    all_explicit_cost=float(book.execution_fee.sum()+book.borrow_fee.sum()+book.financing_interest.sum())))
            save_frame(out/f'{phase}_{series}_daily_attribution.csv',daily)
            checks.append(dict(**metadata,passed=True,check='exact_daily_wealth_and_episode_contribution_reconciliation'))
    frames={'yearly_returns.csv':pd.DataFrame(annual),'sector_contributions.csv':pd.DataFrame(sector_rows),
        'year_sector_contributions.csv':pd.DataFrame(year_sector_rows),'cost_and_wealth_decomposition.csv':pd.DataFrame(cost_rows),
        'episode_contributions.csv':pd.concat(episode_rows,ignore_index=True),'episode_concentration.csv':pd.DataFrame(concentrations),
        'checks.csv':pd.DataFrame(checks),'execution_cost_events.csv':pd.DataFrame(events)}
    for name,frame in frames.items(): save_frame(out/name,frame)
    write_diagnostic_report(base,out,frames)
    draw_readable_figures(base,out)
    verify(root,config['protected_hashes'])
    write_json(out/'manifest.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha256(Path(__file__)),strategy_replays=0,parameter_changes=False,checks=len(checks),logic_checks=len(logic),
        protected_files_unchanged=True,outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(pd.DataFrame(cost_rows).loc[lambda x:x['split'].eq('test')].to_string(index=False),flush=True)
    print(f'PASS: {len(checks)} exact attribution reconciliations; {out}',flush=True)


def validate_logic():
    """Independent two-open examples: a 12.5% sector-over-SPY adjustment."""
    records=[]
    for direction,initial_fee,borrow,funding in [(1,0.,0.,0.),(-1,0.,0.,0.),(1,.0003,0.,0.),(1,0.,.0003,.0004)]:
        dates=pd.to_datetime(['2020-01-02','2020-01-03'])
        q_sector=.00125*direction; q_spy=.01-q_sector
        pnl=q_sector*2+q_spy
        book=pd.DataFrame(dict(open_date=dates,nav_post=[1-initial_fee,1+pnl-initial_fee-borrow-funding],
            price_pnl=[0.,pnl],dividend_cash=0.,cash_interest=0.,execution_fee=[initial_fee,0.],
            borrow_fee=[0.,borrow],financing_interest=[0.,funding]))
        benchmark=pd.DataFrame(dict(open_date=dates,nav_post=[1.,1.01],price_pnl=[0.,.01],
            dividend_cash=0.,cash_interest=0.,execution_fee=0.,borrow_fee=0.,financing_interest=0.))
        assets=[]
        for i,date in enumerate(dates):
            for ticker in [*TICKERS,'SPY']:
                q=q_sector if ticker==TICKERS[0] else q_spy if ticker=='SPY' else 0.
                change=2. if ticker==TICKERS[0] else 1. if ticker=='SPY' else 0.
                assets.append(dict(open_date=date,ticker=ticker,previous_open=100.,
                    open_price=100.+(change if i else 0.),observed_dividend_per_share=0.,
                    current_shares=q if i else 0.,target_shares=0. if i else q,
                    price_pnl_old=q*change if i else 0.,dividend_cash_old=0.))
        assets=pd.DataFrame(assets)
        daily,sector=attribution(book,assets,benchmark)
        expected=.00125*direction
        if not np.isclose(daily.paired_sector_contribution.sum(),expected,atol=1e-14,rtol=0):
            raise AssertionError('Hand sector-vs-SPY contribution failed')
        if abs(daily.capital_path_spy_contribution.sum())>1e-14:
            raise AssertionError('Equal capital must have zero capital-path contribution')
        if not np.isclose(daily.wealth_difference.iloc[-1],expected-initial_fee-borrow-funding,atol=1e-14,rtol=0):
            raise AssertionError('Hand after-cost wealth difference failed')
        ep,_=episode_results(assets,sector,'fixture')
        if len(ep)!=1 or abs(ep.paired_contribution.iloc[0]-expected)>1e-14:
            raise AssertionError('Hand episode mapping failed')
        records.append(dict(check=f'paired_direction_{direction}_entry_{initial_fee}_borrow_{borrow}_funding_{funding}',passed=True))
    return records


def write_diagnostic_report(base,out,frames):
    annual=frames['yearly_returns.csv']; sectors=frames['sector_contributions.csv']
    cost=frames['cost_and_wealth_decomposition.csv']; concentration=frames['episode_concentration.csv']
    lines=['# 第二步：年份、行业、持仓段与成本来源诊断','',
        '本步骤读取两阶段已完成的逐日账本，不重跑或修改行业策略。验证期仅新增分红再投入的纯 SPY 对照，原验证期所有文件保持不变。','',
        '## 测试期各年相对分红再投入 SPY 的表现','',
        '以下为每年实际持有区间的复合财富比减一；2026 年仅截至 9 月 30 日，不进行年度外推。跨年区间按退出开盘所在年份归属。','',
        '|组合|2023|2024|2025|2026截至9月|正增量年份数|','|---|---:|---:|---:|---:|---:|']
    main=annual.loc[annual['split'].eq('test') & annual.series_id.str.startswith('overlay_10_')]
    for series,g in main.groupby('series_id',sort=True):
        values=g.set_index('year').relative_period_return
        lines.append('|'+series+'|'+'|'.join(f'{values.get(y,np.nan):.2%}' for y in (2023,2024,2025,2026))
            +f'|{int(values.gt(0).sum())}/{len(values)}|')
    lines += ['','## 配对行业贡献与成本：测试期','',
        '金额单位均为初始资金 1。行业配对贡献衡量实际行业头寸相对等额 SPY 的价格及分红损益，包含多头与空头方向；这是组合成本前的贡献，不是各行业独立策略的净收益或统计 alpha。','',
        '精确恒等式：期末策略财富−再投SPY财富 = 行业配对贡献＋资本路径差异的SPY贡献＋现金利息差−交易费差−借券费差−融资利息差。资本路径项记录此前盈亏导致后续投入资本不同的复利影响。','',
        '|组合|行业配对贡献|资本路径SPY贡献|额外交易费|借券费|融资利息|最终财富差|','|---|---:|---:|---:|---:|---:|---:|']
    for row in cost.loc[cost['split'].eq('test')].itertuples(index=False):
        lines.append(f'|{row.series_id}|{row.paired_sector_contribution:.4%}|{row.capital_path_spy_contribution:.4%}|{row.execution_fee_difference:.4%}|{row.borrow_fee_difference:.4%}|{row.financing_interest_difference:.4%}|{row.final_wealth_difference:.4%}|')
    lines += ['','不能把实际成本简单加回净收益，当作重新运行过的“零成本策略”；取消成本会改变后续 NAV、持仓数量及复利路径。这里报告实际账本的损益分解。','',
        '## 主要行业与持仓段集中性','',
        '下面只描述 10 bp 主方案；完整 15 bp 参照和验证期分解均在 CSV 中。共享的 SPY 交易费用和融资成本保留在组合层面，不随意分摊为各行业或持仓段的“净利润”。','']
    for series,g in sectors.loc[sectors['split'].eq('test') & sectors.entry_bps.eq(10)].groupby('series_id',sort=True):
        ranked=g.loc[g.held_intervals.gt(0)].sort_values('paired_contribution',ascending=False)
        top=ranked.head(2); bottom=ranked.tail(2)
        c=concentration.loc[concentration['split'].eq('test') & concentration.series_id.eq(series)].iloc[0]
        best='、'.join(f'{r.ticker} {r.paired_contribution:+.3%}' for r in top.itertuples())
        worst='、'.join(f'{r.ticker} {r.paired_contribution:+.3%}' for r in bottom.itertuples())
        description=f'贡献较高 {best}；较低 {worst}' if len(ranked)>3 else '实际行业贡献 '+ '、'.join(f'{r.ticker} {r.paired_contribution:+.3%}' for r in ranked.itertuples())
        lines.append(f'- {series}：{description}。共 {c.completed_episodes} 段行业持仓，前五个正贡献持仓段占全部正贡献的 {c.top5_share_of_positive:.1%}。')
    lines += ['','前五持仓段占比的分母为所有正贡献持仓段之和，未减负贡献和组合费用；它只用于描述集中性，不是对总体净收益的归因比例。','',
        '## 交易费用主要来自哪种事件','',
        '按行业目标方向集合是否变化划分交易日，期初与期末单列。集合变化日也可能含少量目标权重恢复成交，因此本表不是将每一笔费用强行拆成“信号费”和“再平衡费”。','']
    events=frames['execution_cost_events.csv']
    for row in events.loc[events['split'].eq('test') & events.entry_bps.eq(10)
            & events.event_type.eq('same_directions_rebalance')].itertuples(index=False):
        lines.append(f'- {row.series_id}：行业方向集合未变的再平衡日，仅占全部交易费用的 {row.share_of_execution_fee:.2%}。')
    lines += ['','因此，单纯减少“方向没变时的小额再平衡”不是当前优先优化项；即便省下该类全部费用，也只能减少很小一部分实际交易费，且这里没有据此重跑新策略。','',
        '所有策略逐日财富差均按上式对账，行业贡献也与完整持仓段汇总对账。验证期和测试期分开统计，2026 年为部分年份。',
        '这些诊断用于提出研究问题，不能据此直接删掉表现差的行业，或在测试期上反复调参后再声称通过盲测。','']
    (out/'report_中文.md').write_text('\n'.join(lines),encoding='utf-8')


def draw_readable_figures(base,out):
    curves=read_frame(base/'nav_curves.csv',parse_dates=['open_date'])
    reference=curves.loc[curves.series_id.eq('spy_reinvest')].sort_values('open_date').nav_post.to_numpy()
    for relative in (False,True):
        fig,axes=plt.subplots(2,3,figsize=(15,8.5),sharex=True,sharey=True)
        for i,model in enumerate(('elastic_net','gbdt')):
            for j,scheme in enumerate(('S0','S1','S2')):
                ax=axes[i,j]
                series=[(f'overlay_10_{model}_{scheme}','Main 10 / 5 bp','#d56332','-'),
                    (f'overlay_15_{model}_{scheme}','Reference 15 / 7.5 bp','#147d92','-')]
                if not relative:
                    series=[('spy_buy_hold','SPY fixed shares','#555555',':'),
                        ('spy_reinvest','SPY reinvest dividends','#17252f','--')]+series
                else: ax.axhline(0,color='#777777',linewidth=1,linestyle='--')
                for key,label,color,style in series:
                    data=curves.loc[curves.series_id.eq(key)].sort_values('open_date')
                    values=data.nav_post.to_numpy()/reference-1 if relative else data.nav_post.to_numpy()
                    ax.plot(data.open_date,values,label=label,color=color,linestyle=style,linewidth=1.4)
                ax.set_title(f"{'Elastic Net' if i==0 else 'GBDT'} / {scheme}",loc='left',weight='bold')
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.grid(alpha=.18)
                if relative: ax.yaxis.set_major_formatter(PercentFormatter(1))
                if j==0: ax.set_ylabel('Relative wealth vs SPY reinvest' if relative else 'Net wealth (initial capital = 1)')
        title='Relative performance vs dividend-reinvestment SPY' if relative else 'Frozen test comparison: SPY core and sector adjustments'
        fig.suptitle(title,x=.06,ha='left',fontsize=15)
        fig.text(.06,.925,'2023-01-04 to 2026-09-30 | Exploratory, previously inspected test period | All specified costs included',color='#555555')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
        fig.tight_layout(rect=(.02,.075,1,.91))
        name='test_relative_to_spy_reinvest' if relative else 'test_nav_comparison'
        fig.savefig(out/(name+'.png'),dpi=170); fig.savefig(out/(name+'.svg'))
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-logic',action='store_true')
    args=parser.parse_args()
    if args.check_logic: print(pd.DataFrame(validate_logic()).to_string(index=False))
    else: run(args.project_root)
