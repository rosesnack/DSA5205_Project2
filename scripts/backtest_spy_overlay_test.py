"""Frozen exploratory test comparison; imports the unchanged validation engine."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_phase_evaluation import load_phase_inputs
from backtest_spy_overlay import (
    ASSETS, TICKERS, EPS, overlay_step, replay_overlay, validate_run, summary_row,
    read_frame, save_frame, sha256, write_json, markdown_table,
)

OUTPUT = "results/backtest/spy_overlay_test_20261101"
NEW_SCRIPTS = {"backtest_spy_overlay_test.py", "check_spy_overlay_test.py", "diagnose_spy_overlay.py"}


def snapshot(root):
    paths = [p for folder in ("scripts", "data", "results", "notebooks")
        for p in (root/folder).rglob("*") if p.is_file()
        and "__pycache__" not in p.parts and not p.is_relative_to(root/OUTPUT)
        and not (p.parent == root/"scripts" and p.name in NEW_SCRIPTS)]
    paths.extend(p for p in root.iterdir() if p.is_file())
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}


def verify(root, hashes):
    for path, expected in hashes.items():
        if sha256(root/path) != expected:
            raise AssertionError(f"Frozen file changed: {path}")


def freeze(root, out):
    config_path = out/"frozen_protocol.json"
    code = Path(__file__)
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["driver_sha256"] != sha256(code):
            raise AssertionError("Driver changed after freeze; do not mix versions")
        verify(root, config["protected_hashes"])
        return config
    hashes = snapshot(root)
    # Verify the previous validation result and its exact trading engine first.
    audit = json.loads((root/"results/backtest/spy_overlay_validation_20261101/final_audit.json").read_text(encoding="utf-8"))
    for name, expected in audit["source_hashes"].items():
        if sha256(root/"scripts"/name) != expected:
            raise AssertionError("Validation trading engine changed")
    config = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), phase="test",
        endpoints=["2023-01-04", "2026-09-30"], initial_nav=1.0,
        main_entry_exit_bps=[10, 5], fixed_reference_entry_exit_bps=[15, 7.5],
        models=["elastic_net", "gbdt"], schemes=["S0", "S1", "S2"],
        costs=dict(execution_bps=3.0, borrow_annual_rate=.01, financing_spread_annual=.01),
        trading_rules="Import unchanged backtest_spy_overlay.py; exact original decisions and post-fee NAV sizing",
        independence="Each threshold/model/scheme starts with capital 1 and zero positions; final open liquidates",
        report_all_combinations=True, model_refitting=False, threshold_search=False,
        benchmark_addition="Pure SPY with dividends reinvested at the ex-date observed open, same daily target/fees; original fixed-share SPY retained",
        comparisons="Original six 15bp dollar-neutral strategies, cash, SPY fixed shares, SPY reinvestment",
        evidence="Exploratory test-period comparison; test forecasts and prior strategy results were already viewed; not untouched holdout",
        decisions="No parameter/model selection from this test; diagnose first, document any future proposal separately",
        driver_sha256=sha256(code), protected_hashes=hashes,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
    )
    write_json(config_path, config)
    return config


def replay_spy_reinvest(prices, rates, opens, phase="test"):
    quotes = prices.loc[prices.ticker.eq("SPY")].set_index("date").reindex(opens)
    rate_map = {(r.entry_date, r.exit_date): r for r in rates.itertuples(index=False)}
    cash, shares, previous_nav = 1.0, np.zeros(10), 1.0
    previous_prices, previous_date = None, None
    rows, assets = [], []
    for i, (date, quote) in enumerate(quotes.iterrows()):
        price = np.full(10, float(quote.open))
        dividends = np.zeros(10); dividends[-1] = float(quote.dividends)
        initial, final = i == 0, i == len(opens)-1
        weights = np.zeros(10)
        if not final: weights[-1] = 1.0
        days, rf = 0, 0.0
        if not initial:
            rate = rate_map[(previous_date, date)]
            days, rf = (date-previous_date).days, float(rate.risk_free_return_act365)
            if (not rate.interval_valid or not rate.rate_known_at_entry or rate.holding_calendar_days != days
                    or not pd.Timestamp(rate.assumed_release_date) < previous_date):
                raise ValueError("Unavailable reinvestment benchmark interest interval")
        cash, new_shares, detail, row = overlay_step(cash, shares,
            price if initial else previous_prices, price, dividends, weights, rf=rf, calendar_days=days)
        if not np.isclose(row["nav_previous"], previous_nav, atol=EPS, rtol=0):
            raise AssertionError("Reinvestment benchmark previous NAV mismatch")
        metadata = dict(split=phase, benchmark="spy_reinvest", model="benchmark", scheme="spy_reinvest",
            open_date=date, interval_entry_date=previous_date, is_initial_open=initial, is_final_open=final,
            target_pairs=0, holding_return_denominator=np.nan if initial else 1.0 if i == 1 else previous_nav,
            holding_interval_return=np.nan if initial else row["nav_post"]/(1.0 if i == 1 else previous_nav)-1)
        rows.append(metadata | row)
        assets.extend(metadata | a for a in detail if a["ticker"] == "SPY")
        shares, previous_nav, previous_prices, previous_date = new_shares, row["nav_post"], price, date
    return pd.DataFrame(rows), pd.DataFrame(assets)


def references(root, phase="test"):
    if phase == "test":
        path = root/"results/backtest/phase_evaluation"
        old = read_frame(path/"test_daily.csv", parse_dates=["open_date"])
        assets = read_frame(path/"test_assets.csv", parse_dates=["open_date"])
    else:
        path = root/"results/backtest/threshold_selection/checkpoints"
        old = read_frame(path/"entry_15_daily.csv", parse_dates=["open_date"])
        assets = read_frame(path/"entry_15_assets.csv", parse_dates=["open_date"])
    bench = read_frame(root/"results/backtest/investment_benchmarks/benchmark_daily.csv", parse_dates=["open_date"])
    bench = bench.loc[bench["split"].eq(phase)].copy()
    bench["traded_notional"] = bench.trade_notional
    return old, assets, bench


def build_summary(root, experiments, reinvest, phase="test"):
    old, old_assets, bench = references(root, phase)
    spy = bench.loc[bench.benchmark.eq("spy_buy_hold")].sort_values("open_date")
    spy_returns = spy.iloc[1:].set_index("open_date").holding_interval_return
    ri = reinvest.sort_values("open_date")
    ri_returns = ri.iloc[1:].set_index("open_date").holding_interval_return
    rows, episodes, curves = [], [], []
    def append(book, series, kind, model="", scheme="", entry=np.nan, assets=None):
        row, ep = summary_row(book, series_id=series, kind=kind, model=model, scheme=scheme,
            entry_bps=entry, spy_returns=spy_returns, spy_ending=float(spy.nav_post.iloc[-1]), assets=assets)
        row["split"] = phase  # The imported historical summary helper labels validation.
        ordered = book.sort_values("open_date")
        delta = ordered.holding_interval_return.iloc[1:].to_numpy()-ri_returns.to_numpy()
        te = float(np.std(delta, ddof=1)*np.sqrt(252))
        row.update(cumulative_relative_to_spy_reinvest=float(ordered.nav_post.iloc[-1]/ri.nav_post.iloc[-1]-1),
            annualized_tracking_error_vs_spy_reinvest=te,
            information_ratio_vs_spy_reinvest=float(np.mean(delta)*252/te) if te > EPS else np.nan,
            role="main" if kind == "spy_overlay" and entry == 10 else "fixed_reference")
        rows.append(row); episodes.extend(ep)
        curves.append(ordered[["open_date", "nav_post"]].assign(series_id=series))
    for (model, scheme), group in old.groupby(["model", "scheme"], sort=True):
        selected = old_assets.model.eq(model) & old_assets.scheme.eq(scheme)
        append(group, f"original_{model}_{scheme}", "original", model, scheme, 15, old_assets.loc[selected])
    for name, group in bench.groupby("benchmark", sort=True):
        append(group, name, "benchmark")
    append(ri, "spy_reinvest", "benchmark")
    for entry, book, assets in experiments:
        for (model, scheme), group in book.groupby(["model", "scheme"], sort=True):
            selected = assets.model.eq(model) & assets.scheme.eq(scheme)
            append(group, f"overlay_{entry}_{model}_{scheme}", "spy_overlay", model, scheme, entry, assets.loc[selected])
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True), pd.DataFrame(episodes)


def draw_test_curves(curves, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for i, model in enumerate(("elastic_net", "gbdt")):
        for j, scheme in enumerate(("S0", "S1", "S2")):
            ax = axes[i,j]
            for series, label, color, style in [
                ("spy_buy_hold", "SPY fixed shares", "#1b2530", "-"),
                ("spy_reinvest", "SPY reinvest dividends", "#659478", "--"),
                ("cash", "Cash", "#888888", ":"),
                (f"original_{model}_{scheme}", "Original 15 / 7.5 bp", "#9374ab", "--"),
                (f"overlay_15_{model}_{scheme}", "Reference 15 / 7.5 bp", "#147d92", "-"),
                (f"overlay_10_{model}_{scheme}", "Main 10 / 5 bp", "#d56332", "-")]:
                data = curves.loc[curves.series_id.eq(series)].sort_values("open_date")
                ax.plot(pd.concat([data.open_date.iloc[:1],data.open_date]), np.r_[1.,data.nav_post],
                    label=label, color=color, linestyle=style, linewidth=1.5 if series.startswith("overlay") else 1.)
            ax.set_title(f"{'Elastic Net' if i==0 else 'GBDT'} / {scheme}", loc="left", weight="bold")
            ax.grid(alpha=.18)
            if j==0: ax.set_ylabel("Net wealth (initial capital = 1)")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle("Exploratory test: frozen SPY-core sector adjustments | main 10 / 5 bp, reference 15 / 7.5 bp",x=.06,ha="left",fontsize=14)
    fig.text(.06,.925,"2023-01-04 to 2026-09-30 | All execution, borrow and financing costs included | Test previously inspected",color="#555555")
    fig.tight_layout(rect=(.02,.10,1,.91))
    fig.savefig(out/"cumulative_nav.png",dpi=170); fig.savefig(out/"cumulative_nav.svg")
    plt.close(fig)


def write_report(summary, curves, episodes, out):
    save_frame(out/"summary.csv",summary)
    save_frame(out/"nav_curves.csv",curves)
    save_frame(out/"holding_episodes.csv",episodes)
    draw_test_curves(curves,out)
    overlay = summary.loc[summary.kind.eq("spy_overlay")]
    high=overlay.loc[overlay.entry_bps.eq(15)].set_index(["model","scheme"])
    low=overlay.loc[overlay.entry_bps.eq(10)].set_index(["model","scheme"])
    fields=["annualized_net_return","max_drawdown","cumulative_relative_to_spy",
        "cumulative_relative_to_spy_reinvest","mean_holding_intervals","sector_participation",
        "annualized_two_sided_turnover","total_execution_fee","total_borrow_fee",
        "total_financing_interest","total_cost"]
    paired=pd.DataFrame(index=high.index)
    for f in fields:
        paired[f+"_15bp"]=high[f]; paired[f+"_10bp"]=low[f]
        paired[f+"_delta_10_minus_15"]=low[f]-high[f]
    save_frame(out/"threshold_paired_comparison.csv",paired.reset_index())
    spy=summary.loc[summary.series_id.eq("spy_buy_hold")].iloc[0]
    ri=summary.loc[summary.series_id.eq("spy_reinvest")].iloc[0]
    lines=["# 第一步：冻结规则的测试期探索性比较","",
        "区间：2023-01-04—2026-09-30；10／5 bp 为主方案，15／7.5 bp 为固定参照。六组官方预测全部展示，不根据测试期排名改变门槛或选模型。","",
        "交易费仍为单边 3 bp，实际空头借券年化 1%，融资 DGS3MO＋1 个百分点，按 ACT/365。复用验证期原交易引擎，未修改已有代码和结果。","",
        f"SPY 固定股数买入持有净年化 {spy.annualized_net_return:.2%}，分红再投入 SPY 净年化 {ri.annualized_net_return:.2%}；原 SPY 分红留存现金的基准继续保留。","",
        "![测试期累计净值](cumulative_nav.png)","",markdown_table(summary),"",
        "成本列为整个区间累计实际金额/初始资金，年化换手为双边倍数/年。行业参与不含 SPY 基础仓位；平均持有时长按行业、方向的连续持仓段统计。完整 CSV 另外提供相对分红再投入 SPY 的累计表现、跟踪误差与信息比率。","",
        f"主方案超过固定股数 SPY 的组合数：{int(low.cumulative_relative_to_spy.gt(0).sum())}/6；超过分红再投入 SPY：{int(low.cumulative_relative_to_spy_reinvest.gt(0).sum())}/6。","",
        f"10 bp 相比 15 bp 净年化改善：{int((low.annualized_net_return>high.annualized_net_return).sum())}/6。这是固定方案比较，不据此事后选最优门槛。","",
        "## 与验证期并列观察","",
        "|组合|验证15净年化|验证10净年化|测试15净年化|测试10净年化|","|---|---:|---:|---:|---:|",
    ]
    root=out.parents[2]
    val=read_frame(root/"results/backtest/spy_overlay_validation_20261101/summary.csv")
    for model,scheme in high.index:
        v=val.loc[val.kind.eq("spy_overlay") & val.model.eq(model) & val.scheme.eq(scheme)].set_index("entry_bps")
        lines.append(f"|{model}/{scheme}|{v.loc[15,'annualized_net_return']:.2%}|{v.loc[10,'annualized_net_return']:.2%}|{high.loc[(model,scheme),'annualized_net_return']:.2%}|{low.loc[(model,scheme),'annualized_net_return']:.2%}|")
    lines += ["","测试期的预测、旧策略结果此前已被查看；本次为探索性复用，不是全新盲测。验证期和测试期独立初始化，2026 年仅到 9 月 30 日。","",
        "下一步只做年份、行业、持仓段与成本来源诊断；任何规则调整需另列为新实验，不能再把当前测试期当作未使用的验证证据。",""]
    (out/"report_中文.md").write_text("\n".join(lines),encoding="utf-8")
    table=summary[["series_id","annualized_net_return","max_drawdown","cumulative_relative_to_spy",
        "cumulative_relative_to_spy_reinvest","sector_participation","total_cost"]].copy()
    table.columns=["组合","净年化","最大回撤","相对SPY累计","相对分红再投SPY累计","行业参与","累计成本/初始资金"]
    for col in table.columns[1:]: table[col]=table[col].map(lambda v:f"{v:.2%}")
    content="<!doctype html><meta charset='utf-8'><title>测试期探索性结果</title><style>body{font:15px system-ui;margin:32px;color:#17232c}img{width:100%;max-width:1500px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style>"
    content+="<h1>冻结规则的测试期探索性比较</h1><p>10／5 bp 主方案；15／7.5 bp 固定参照。2023-01-04—2026-09-30。完整口径见 report_中文.md。</p><img src='cumulative_nav.png'>"+table.to_html(index=False,escape=True)
    (out/"report.html").write_text(content,encoding="utf-8")


def run(root):
    root=root.resolve(); out=root/OUTPUT
    out.mkdir(parents=True,exist_ok=True)
    config=freeze(root,out)
    if (out/"step1_manifest.json").exists():
        done=json.loads((out/"step1_manifest.json").read_text(encoding="utf-8"))
        for name,digest in done["outputs"].items():
            if sha256(out/name)!=digest: raise AssertionError(f"Completed output changed: {name}")
        print("Verified completed frozen test comparison; no replay",flush=True)
        return
    (signals,daily,prices,rates,opens),input_checks=load_phase_inputs(root,"test")
    reasons=read_frame(root/"results/backtest/daily_signal_status.csv",parse_dates=["date"],
        usecols=["date","model","scheme","reason"])
    daily=daily.merge(reasons.rename(columns={"reason":"signal_reason"}),on=["date","model","scheme"],validate="one_to_one")
    save_frame(out/"input_checks.csv",pd.DataFrame(input_checks))
    save_frame(out/"rate_alignment.csv",rates)
    print(f"Frozen before new test replay; {len(opens)} opens, {len(opens)-1} intervals",flush=True)
    experiments=[]
    for entry in (10,15):
        folder=out/f"experiment_{entry}bp"; folder.mkdir(exist_ok=True)
        book,assets,trades,decisions=replay_overlay(signals,daily,prices,rates,entry)
        checks=validate_run(book,assets,rates,opens,entry)
        for name,frame in (("daily.csv",book),("assets.csv",assets),("trades.csv",trades),
            ("decisions.csv",decisions),("checks.csv",checks)):
            save_frame(folder/name,frame)
        experiments.append((entry,book,assets))
        print(f"PASS {entry}bp: {len(checks)} runtime checks",flush=True)
    ri,ri_assets=replay_spy_reinvest(prices,rates,opens)
    save_frame(out/"spy_reinvest_daily.csv",ri); save_frame(out/"spy_reinvest_assets.csv",ri_assets)
    summary,curves,episodes=build_summary(root,experiments,ri)
    if len(summary)!=21 or summary.accounting_residual.abs().max()>EPS:
        raise AssertionError("Missing comparison row or summary accounting residual")
    # Reused original strategy metrics must agree with the established report.
    old=read_frame(root/"results/backtest/reporting/main_metrics.csv")
    old=old.loc[old['split'].eq('test') & old.kind.eq('strategy')]
    joined=summary.loc[summary.kind.eq('original')].merge(old,on=['model','scheme'],suffixes=('_now','_old'),validate='one_to_one')
    fields=['ending_nav','annualized_net_return','annualized_net_volatility','max_drawdown',
        'annualized_two_sided_turnover','total_execution_fee','total_borrow_fee']
    errors={f:float((joined[f+'_now']-joined[f+'_old']).abs().max()) for f in fields}
    if max(errors.values())>EPS: raise AssertionError(errors)
    write_report(summary,curves,episodes,out)
    verify(root,config['protected_hashes'])
    outputs={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='step1_manifest.json'}
    write_json(out/'step1_manifest.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Step 1 complete: fixed exploratory test comparison',opening_records=len(opens),holding_intervals=len(opens)-1,
        new_strategy_count=12,summary_rows=len(summary),runtime_checks=122,input_checks=len(input_checks),
        original_reference_errors=errors,protected_file_count=len(config['protected_hashes']),original_files_unchanged=True,
        trading_engine_unchanged=True,model_refitting=False,parameter_changes=False,
        primary_threshold_bps=10,reference_threshold_bps=15,outputs=outputs))
    print(summary.loc[summary.kind.ne('original'),['series_id','annualized_net_return','max_drawdown',
        'cumulative_relative_to_spy','cumulative_relative_to_spy_reinvest','sector_participation','total_cost']].to_string(index=False),flush=True)
    print(f"PASS: {len(config['protected_hashes'])} protected files unchanged; {out}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().project_root)
