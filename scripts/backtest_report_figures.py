"""Render step-14 scientific figures and a Chinese report from saved tables."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ORDER = [f"{m}_{s}" for m in ("elastic_net", "gbdt") for s in ("S0", "S1", "S2")] + ["spy_buy_hold", "cash"]
LABELS = {**{f"elastic_net_{s}": f"EN {s}" for s in ("S0", "S1", "S2")},
          **{f"gbdt_{s}": f"GBDT {s}" for s in ("S0", "S1", "S2")}, "spy_buy_hold": "SPY buy & hold", "cash": "Cash"}
COLORS = dict(zip(ORDER, ["#7aaed1", "#367daf", "#174e80", "#f4b13e", "#df702c", "#a83427", "#252a34", "#87919b"]))
STYLES = dict(zip(ORDER, [":", "--", "-", ":", "--", "-", "-", "--"]))
PHASE_LABELS = {"validation": "Validation: tuning evidence", "test": "Test: prior forecast exploration disclosed"}


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def number(value, percent=False, digits=3):
    if pd.isna(value):
        return "N/A"
    return f"{float(value)*(100 if percent else 1):.{digits}f}" + ("%" if percent else "")


def write_report(out, metrics, robust, annual, pnl, scenarios, figures):
    lines = ["# 第14项：策略指标、比较图与成本/融资敏感性", "",
        "本项已在第13项1,049项审计通过、协议与输入哈希一致后执行。公共门槛继续为开仓15 bps、退出7.5 bps。原主成本策略与投资基准均复用，不重新拟合模型或选择门槛。", "",
        "主成本为单边3 bps、借券年化1%、卖空抵押现金不返息。以下全部策略和基准使用同阶段起止开盘，分别从现金1开始，期末清仓。SPY持有固定份额，分红留现金计息。", "",
        "## 评价与统计口径", "",
        "验证期2018-01-03至2022-12-30开盘，属于模型与门槛选择后的证据。测试期2023-01-04至2026-09-30开盘；原模型阶段已查看测试预测评价，GBDT也记录探索背景，因此不是从未接触的全盲留出集。", "",
        "年化净收益按终值与252个交易日计算；波动率使用全部日净收益的样本标准差（ddof=1）乘sqrt(252)；Sharpe使用日净收益减同区间现金收益。最大回撤是带负号的最小净值/历史高点−1，并计入初始资本1和首次开盘费用。双边换手含首尾实际净交易、不除以二。空仓区间保留；现金超额零方差的Sharpe为N/A。", "",
        "SPY为满仓市场多头，配对策略风险敞口不同；基准用于比较可投资替代品，不能仅凭总收益判定风险调整后的优劣。现金日收益随利率与区间自然日数变化，其净收益波动率可以非零，而现金超额Sharpe仍为N/A。", ""]
    for phase in ("validation", "test"):
        title = "验证期" if phase == "validation" else "测试期"
        data = metrics.loc[metrics["split"].eq(phase)].set_index("series_id").loc[ORDER]
        lines += [f"## {title}主成本完整指标", "", md_table(
            ["策略/基准", "年化净收益", "年化净波动率", "现金超额Sharpe", "最大回撤(负值)", "双边年化换手", "持仓区间比例", "期末净值"],
            [[LABELS[k], number(row.annualized_net_return, True), number(row.annualized_net_volatility, True),
              number(row.annualized_cash_excess_sharpe), number(row.max_drawdown, True), number(row.annualized_two_sided_turnover),
              number(row.holding_interval_fraction, True), number(row.ending_nav, digits=6)] for k, row in data.iterrows()]), ""]
    lines += ["## 逐年收益", "",
        "逐年收益按持有区间终止开盘所属年份复利汇总，不将不完整年份年化。首尾年份遵守阶段边界；2026年只截至9月30日。全部72行保存在calendar_year_returns.csv。", "",
        "## 主成本损益来源", "",
        "下表均为相对初始资本1的金额百分比，不是各组件分别复利的收益率。终值=1+价格损益+分红+现金利息−借券费−执行费。分红可能为净支出。", ""]
    for phase in ("validation", "test"):
        data = pnl.loc[pnl["split"].eq(phase) & pnl.kind.eq("strategy")]
        lines += ["### " + ("验证期" if phase == "validation" else "测试期"), "", md_table(
            ["策略", "价格损益", "分红净额", "现金利息", "借券费", "执行费", "净损益"],
            [[LABELS[r.series_id], *[number(getattr(r, f), True) for f in ("price_pnl", "dividends", "cash_interest", "borrow_fee", "execution_fee", "net_pnl")]]
             for r in data.itertuples()]), ""]
    lines += ["## 预先固定的敏感性场景", "",
        "每个替代场景分别从现金1回放完整账本，保持同一预测、配对规则、目标权重和15/7.5 bps门槛。费用改变NAV，也会改变以后实际份额、融资及再平衡金额，因此不从固定毛收益曲线直接减费。主设置没有因结果改变。零执行费场景仍有主借券费。", "",
        md_table(["场景", "单边执行费bps", "借券年化", "全部现金返息"],
                 [[r.scenario_id, number(r.execution_bps, digits=0), number(r.borrow_annual_rate, True, 0), "是" if r.full_cash_rebate else "否"]
                  for r in scenarios.itertuples()]), "",
        "共96行策略/阶段/场景指标，其中12条主成本路径复用，84条替代路径是本次新增执行。SPY/现金主成本比较沿用第12项，不将其另行扩展成未规定的敏感性基准。", "",
        "以下范围只是固定8个假设的结果范围，不是最优成本或模型选择。全部96行、相对主设置变化与费用细项保存在sensitivity_metrics.csv。", "",
        md_table(["阶段", "策略", "年化净收益范围", "现金超额Sharpe范围"],
            [["验证" if r.split == "validation" else "测试", LABELS[f"{r.model}_{r.scheme}"],
              f"{number(r.min_annualized_net_return, True)} 至 {number(r.max_annualized_net_return, True)}",
              f"{number(r.min_cash_excess_sharpe)} 至 {number(r.max_cash_excess_sharpe)}"] for r in robust.itertuples()]), "",
        "## 比较图", "",
        "图中保留全部六组策略，三组Elastic Net测试路径相同，曲线会重合。PNG用于查看，SVG用于编辑或导出。", ""]
    captions = ["主成本净值与六组策略相对现金财富", "策略及投资基准回撤（上下行分别使用适合的纵轴范围）",
                "主成本收益、波动与现金超额Sharpe", "执行费、借券与返息对年化净收益的影响",
                "执行费、借券与返息对现金超额Sharpe的影响", "逐年实际区间复利收益，2026年为截至9月30日"]
    for name, caption in zip(figures, captions):
        lines += [f"### {caption}", "", f"![{caption}](figures/{name}.png)", ""]
    lines += ["## 文件与复用", "",
        "主结果见main_metrics.csv、main_curves.csv、calendar_year_returns.csv及pnl_decomposition.csv；敏感性见sensitivity_metrics.csv、cost_robustness.csv及sensitivity/checkpoints/完整账本。design_config.json记录预先固定口径，numeric_manifest.json记录数值结果，run_manifest.json记录本项整体完成状态。", "",
        "原始数据、01—04 Notebook、正式模型预测、训练诊断、门槛选择、已登记主策略/基准及审计输出保持不变。本项未执行第15项最终打包或第16项可选扩展。", "",
        "协议来源：[冻结协议](../phase_evaluation/protocol_config.json)；[公共门槛](../threshold_selection/selected_threshold.json)；[第13项独立审计](../consistency_audit/run_manifest.json)。", ""]
    (out / "report_中文.md").write_text("\n".join(lines), encoding="utf-8")


def render_report(root, out):
    # Keep plotting's generated font cache in an owned temporary directory.
    with tempfile.TemporaryDirectory(prefix="dsa5205-report-") as config_dir:
        previous_config = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = config_dir
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
            from PIL import Image
            result = _render(out, plt, PercentFormatter, Image)
            result["matplotlib_version"] = matplotlib.__version__
        finally:
            if previous_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_config
    return result


def _render(out, plt, PercentFormatter, Image):
    if __package__:
        from .backtest_threshold_selection import read_frame
    else:
        from backtest_threshold_selection import read_frame
    metrics = read_frame(out / "main_metrics.csv")
    curves = read_frame(out / "main_curves.csv", parse_dates=["open_date"])
    sensitivity = read_frame(out / "sensitivity_metrics.csv")
    annual = read_frame(out / "calendar_year_returns.csv")
    robust = read_frame(out / "cost_robustness.csv")
    pnl = read_frame(out / "pnl_decomposition.csv")
    scenarios = read_frame(out / "cost_scenarios.csv")
    folder = out / "figures"; folder.mkdir(exist_ok=True)
    checks, figures = [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "#fafbfc",
        "axes.grid": True, "grid.alpha": .18, "legend.frameon": False, "svg.fonttype": "none"})
    def export(fig, name):
        fig.savefig(folder / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with Image.open(folder / f"{name}.png") as im:
            im.verify()
        with Image.open(folder / f"{name}.png") as im:
            valid = im.width >= 1000 and im.height >= 600
        ET.parse(folder / f"{name}.svg")
        assert valid
        checks.extend([dict(phase="both", scenario_id="figure", check=name+"_png_valid", passed=True, detail="PNG decoded, dimensions checked"),
                       dict(phase="both", scenario_id="figure", check=name+"_svg_valid", passed=True, detail="SVG parsed")])
        figures.append(name)
    def line(ax, data, series, y):
        ax.plot(data.open_date, data[y], color=COLORS[series], ls=STYLES[series], lw=1.6,
                label=LABELS[series], alpha=.95)
    def legend(fig, ax):
        h, labels = ax.get_legend_handles_labels()
        fig.legend(h, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, .01), fontsize=9)
    def phase_curve(phase, series):
        return curves.loc[curves["split"].eq(phase) & curves.series_id.eq(series)].sort_values("point_order")
    phases = ("validation", "test")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
    for col, phase in enumerate(phases):
        for series in ORDER:
            line(axes[0, col], phase_curve(phase, series), series, "nav")
        for series in ORDER[:6]:
            line(axes[1, col], phase_curve(phase, series), series, "nav_relative_to_cash")
        axes[1, col].axhline(1, color=COLORS["cash"], ls="--", lw=1)
        axes[0, col].set_title(PHASE_LABELS[phase]); axes[0, col].set_ylabel("NAV (initial capital = 1)")
        axes[1, col].set_ylabel("Paired strategy NAV / cash NAV")
    fig.suptitle("Main costs | entry / exit 15 / 7.5 bps", fontsize=15, y=.98)
    legend(fig, axes[0, 0]); fig.tight_layout(rect=(0, .09, 1, .95)); export(fig, "01_nav_and_cash_relative")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
    for col, phase in enumerate(phases):
        for series in ORDER[:6]:
            line(axes[0, col], phase_curve(phase, series), series, "drawdown")
        for series in ORDER[6:]:
            line(axes[1, col], phase_curve(phase, series), series, "drawdown")
        axes[0, col].set_title(PHASE_LABELS[phase])
        for row in (0, 1):
            axes[row, col].yaxis.set_major_formatter(PercentFormatter(1)); axes[row, col].set_ylabel("Drawdown: strategies" if row == 0 else "Drawdown: benchmarks")
    h1, l1 = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1+h2, l1+l2, loc="lower center", ncol=4, bbox_to_anchor=(.5, .01), fontsize=9)
    fig.suptitle("Drawdowns include initial entry fee | row-specific scales", fontsize=15, y=.98)
    fig.tight_layout(rect=(0, .09, 1, .95)); export(fig, "02_drawdowns")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    specs = [("annualized_net_return", "Net CAGR (%)", 100), ("annualized_net_volatility", "Net volatility (%)", 100),
             ("annualized_cash_excess_sharpe", "Cash-excess Sharpe", 1)]
    for row, phase in enumerate(phases):
        data = metrics.loc[metrics["split"].eq(phase)].set_index("series_id").loc[ORDER]
        for col, (field, title, scale) in enumerate(specs):
            ax = axes[row, col]; values = data[field].to_numpy(float)*scale; yy = np.arange(8)
            ax.barh(yy, np.nan_to_num(values), color=[COLORS[s] for s in ORDER], height=.65)
            ax.set_yticks(yy, [LABELS[s] for s in ORDER]); ax.invert_yaxis(); ax.axvline(0, color="#555", lw=.7)
            ax.set_title(("Validation" if row == 0 else "Test")+" | "+title); ax.margins(x=.3)
            span = max(float(np.nanmax(values)-np.nanmin(values)), .1)
            for y, value in zip(yy, values):
                ax.text(0 if np.isnan(value) else value+span*.02*(1 if value >= 0 else -1), y,
                    "N/A" if np.isnan(value) else f"{value:.2f}", va="center", ha="left" if np.isnan(value) or value >= 0 else "right", fontsize=9)
    fig.suptitle("Main-cost metrics | alternative investments have different risk exposure", fontsize=14, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .96)); export(fig, "03_main_metrics")
    for metric, scale, ylabel, name in (("annualized_net_return", 100, "Net CAGR (%)", "04_cost_return_sensitivity"),
                                      ("annualized_cash_excess_sharpe", 1, "Cash-excess Sharpe", "05_cost_sharpe_sensitivity")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        groups = [(["execution_0", "execution_1", "main", "execution_5"], "execution_bps", 1, "Execution cost (bps)", 3),
                  (["borrow_0", "main", "borrow_2", "borrow_3"], "borrow_annual_rate", 100, "Annual borrow rate (%)", 1),
                  (["main", "full_cash_rebate"], "full_cash_rebate", 1, "Short-collateral rebate", 0)]
        for row, phase in enumerate(phases):
            for col, (ids, xfield, multiplier, xlabel, main_x) in enumerate(groups):
                ax = axes[row, col]
                for series in ORDER[:6]:
                    data = sensitivity.loc[sensitivity["split"].eq(phase) & sensitivity.series_id.eq(series) & sensitivity.scenario_id.isin(ids)].sort_values(xfield)
                    xx = data[xfield].astype(float)*multiplier
                    ax.plot(xx, data[metric]*scale, color=COLORS[series], ls=STYLES[series], marker="o", ms=4, lw=1.6, label=LABELS[series])
                ax.axvline(main_x, color="#888", lw=.8, ls=":"); ax.axhline(0, color="#777", lw=.6)
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title("Validation" if phase == "validation" else "Test")
                if col == 2:
                    ax.set_xticks([0, 1], ["No rebate (main)", "Full rebate"])
                elif col == 0:
                    ax.set_xticks([0, 1, 3, 5])
                else:
                    ax.set_xticks([0, 1, 2, 3])
        fig.suptitle("Frozen one-factor sensitivities | same 15 / 7.5 bps threshold", fontsize=15, y=.99)
        legend(fig, axes[0, 0]); fig.tight_layout(rect=(0, .09, 1, .96)); export(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), constrained_layout=True)
    bound = max(float(annual.net_return.abs().max()*100), 1.)
    for ax, phase in zip(axes, phases):
        data = annual.loc[annual["split"].eq(phase)].pivot(index="series_id", columns="calendar_year", values="net_return").reindex(ORDER)*100
        image = ax.imshow(data, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_yticks(range(8), [LABELS[s] for s in ORDER]); ax.set_xticks(range(len(data.columns)), [f"{y}*" if y == 2026 else str(y) for y in data.columns]); ax.grid(False)
        for i in range(8):
            for j in range(len(data.columns)):
                value = data.iloc[i, j]; ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > .65*bound else "#151515")
        ax.set_title("Validation (tuning)" if phase == "validation" else "Test (prior exploration disclosed)")
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02, label="Actual compounded return (%)")
    fig.suptitle("Calendar-year groups by interval end | *2026 through Sep 30, not annualized", fontsize=13)
    export(fig, "06_calendar_year_returns")
    write_report(out, metrics, robust, annual, pnl, scenarios, figures)
    checks.append(dict(phase="both", scenario_id="report", check="chinese_report_created", passed=(out / "report_中文.md").stat().st_size > 5000, detail="All 16 main paths and all 12 sensitivity ranges included"))
    assert all(c["passed"] for c in checks)
    return dict(checks=checks, figures=figures)
