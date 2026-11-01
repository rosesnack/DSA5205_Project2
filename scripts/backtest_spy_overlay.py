"""Validation-only exploratory SPY-core/sector-overlay experiment.

Uses delivered forecasts and the existing pair decision function. Original
engines, forecasts and results are read-only. No test replay or model fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_execution import pair_weights
from backtest_pair_decisions import decide_pairs
from backtest_threshold_selection import (
    TICKERS, load_validation_inputs, read_frame, save_frame, sha256, write_json,
)

ASSETS = (*TICKERS, "SPY")
OUTPUT = "results/backtest/spy_overlay_validation_20261101"
EPS = 1e-12


def overlay_weights(pairs, *, liquidate=False):
    raw = pair_weights(pairs, tickers=TICKERS)
    short = sum(max(-raw[t], 0.0) for t in TICKERS)
    scale = min(1.0, 0.25 / short) if short else 1.0
    sectors = [raw[t] * scale for t in TICKERS]
    weights = np.array([*sectors, 1.0 - sum(sectors)])
    if liquidate:
        weights[:] = 0.0
    return weights, short, scale


def overlay_step(cash, shares, previous_prices, prices, dividends, weights, *,
                 rf, calendar_days, execution_bps=3.0, borrow_rate=0.01,
                 funding_spread=0.01):
    """Settle old shares; fund fees inside NAV; trade unique net asset targets.

    Interval carry uses previous post-trade cash and previous-open actual shorts.
    Dividends arriving at this open do not earn preceding-interval interest.
    """
    shares, prices, dividends, weights = map(
        lambda x: np.asarray(x, dtype=float), (shares, prices, dividends, weights))
    previous_prices = np.asarray(previous_prices, dtype=float)
    if any(a.shape != (10,) for a in (shares, prices, dividends, weights, previous_prices)):
        raise ValueError("Expected ten unique net portfolio assets")
    if (not all(np.isfinite(a).all() for a in (shares, prices, dividends, weights, previous_prices))
            or min(prices.min(), previous_prices.min()) <= 0):
        raise ValueError("Missing/invalid observed execution price or dividend")
    if (not all(np.isfinite(v) for v in (cash, rf, execution_bps, borrow_rate, funding_spread))
            or rf <= -1 or min(execution_bps, borrow_rate, funding_spread) < 0
            or calendar_days < 0 or int(calendar_days) != calendar_days):
        raise ValueError("Invalid cash, carry or cost input")
    if calendar_days == 0 and (np.any(shares) or rf != 0):
        raise ValueError("Initial zero-day record cannot contain old shares or interest")
    flat = not np.any(weights)
    if not flat and (abs(weights.sum() - 1) > EPS or np.abs(weights).sum() > 1.5 + EPS):
        raise ValueError("Overlay targets must be net 100% and gross at most 150%")

    start_values, old_values = shares * previous_prices, shares * prices
    collateral = np.maximum(-start_values, 0).sum()
    interest_base = max(cash - collateral, 0.0)
    funding_base = max(collateral - cash, 0.0)
    funding_return = rf + funding_spread * calendar_days / 365
    interest = interest_base * rf
    financing = funding_base * funding_return
    borrow = collateral * borrow_rate * calendar_days / 365
    dividend_by_asset = shares * dividends
    pnl_by_asset = shares * (prices - previous_prices)
    dividend_cash, price_pnl = dividend_by_asset.sum(), pnl_by_asset.sum()
    nav_previous = cash + start_values.sum()
    cash_pre = cash + dividend_cash + interest - borrow - financing
    nav_pre = cash_pre + old_values.sum()
    if min(nav_previous, nav_pre) <= 0:
        raise ValueError("Nonpositive NAV")
    cost = execution_bps / 10000

    # Solve x + fee(x) = NAV_pre. The fee map is a contraction with
    # Lipschitz constant cost * gross <= 0.00045 under the frozen policy.
    target_nav = nav_pre
    for _ in range(30):
        updated = nav_pre - cost * np.abs(weights * target_nav - old_values).sum()
        if updated <= 0:
            raise ValueError("Insufficient NAV to fund transaction costs")
        if abs(updated - target_nav) <= 1e-15 * max(1.0, nav_pre):
            target_nav = updated
            break
        target_nav = updated
    else:
        raise AssertionError("Fee-funded target NAV did not converge")
    new_values = weights * target_nav
    new_shares = new_values / prices
    delta = new_shares - shares
    notional_by_asset = np.abs(delta) * prices
    fees_by_asset = notional_by_asset * cost
    fee = fees_by_asset.sum()
    cash_change = -(delta * prices).sum()
    cash_post = cash_pre + cash_change - fee
    nav_post = cash_post + new_values.sum()
    pre_error = nav_pre - (nav_previous + price_pnl + dividend_cash + interest - borrow - financing)
    post_error = nav_post - (nav_pre - fee)
    if max(abs(pre_error), abs(post_error), abs(nav_post - target_nav)) > EPS * max(1.0, nav_pre):
        raise AssertionError("Overlay accounting identity failed")
    if not flat and abs(cash_post) > EPS * max(1.0, nav_pre):
        raise AssertionError("Fee-funded net-100% targets must consume unrestricted book cash")
    gross_post = np.abs(new_values).sum() / nav_post
    if gross_post > 1.5 + EPS:
        raise AssertionError("Actual post-fee gross exceeds 150%")
    summary = dict(
        cash_prev=float(cash), nav_previous=float(nav_previous), holding_calendar_days=calendar_days,
        short_collateral_start=float(collateral), cash_interest_base=float(interest_base),
        financing_base=float(funding_base), interval_risk_free_return=float(rf),
        interval_financing_return=float(funding_return),
        financing_annual_rate=float(funding_return * 365 / calendar_days) if calendar_days else np.nan,
        dividend_cash=float(dividend_cash), price_pnl=float(price_pnl), cash_interest=float(interest),
        borrow_fee=float(borrow), financing_interest=float(financing), cash_pre=float(cash_pre),
        nav_pre=float(nav_pre), traded_notional=float(notional_by_asset.sum()),
        sector_traded_notional=float(notional_by_asset[:-1].sum()), spy_traded_notional=float(notional_by_asset[-1]),
        cash_change_before_costs=float(cash_change), execution_fee=float(fee), cash_post=float(cash_post),
        market_value_post=float(new_values.sum()), nav_post=float(nav_post),
        target_gross_weight=float(np.abs(weights).sum()), actual_gross_post=float(gross_post),
        actual_short_post=float(np.maximum(-new_values, 0).sum() / nav_post),
        actual_gross_pre=float(np.abs(old_values).sum() / nav_pre),
        pre_reconciliation_error=float(pre_error), post_reconciliation_error=float(post_error),
    )
    details = [dict(ticker=t, target_weight=float(weights[i]), open_price=float(prices[i]),
        previous_open=float(previous_prices[i]), observed_dividend_per_share=float(dividends[i]),
        current_shares=float(shares[i]), target_shares=float(new_shares[i]), delta_shares=float(delta[i]),
        trade_notional=float(notional_by_asset[i]), execution_fee=float(fees_by_asset[i]),
        market_value_post=float(new_values[i]), actual_weight_post=float(new_values[i] / nav_post),
        dividend_cash_old=float(dividend_by_asset[i]), price_pnl_old=float(pnl_by_asset[i]))
        for i, t in enumerate(ASSETS)]
    return float(cash_post), new_shares, details, summary


def replay_overlay(signals, daily, prices, rates, entry_bps):
    quote_map = {}
    for date, group in prices.groupby("date", sort=False):
        group = group.set_index("ticker").reindex(ASSETS)
        quote_map[date] = (group.open.to_numpy(float), group.dividends.to_numpy(float))
    rate_map = {(r.entry_date, r.exit_date): r for r in rates.itertuples(index=False)}
    predictions = {key: dict(zip(g.ticker, g.y_hat)) for key, g in
                   signals.groupby(["date", "model", "scheme"], sort=False)}
    rows, asset_rows, decision_rows = [], [], []
    for (model, scheme, split), group in daily.groupby(["model", "scheme", "split"], sort=True):
        cash, shares, active = 1.0, np.zeros(10), {}
        previous_date, previous_prices, previous_nav = None, None, 1.0
        intervals = 0
        for instruction in group.sort_values("date").to_dict("records"):
            if instruction["action"] == "NO_ORDER_OUTSIDE_PHASE":
                continue
            stamp = instruction["action_at"]
            if not instruction["signal_at"] < stamp:
                raise AssertionError("Signal must precede execution")
            date = stamp.tz_localize(None).normalize()
            if previous_date is not None and date <= previous_date:
                raise AssertionError("Execution dates must increase")
            quotes, dividends = quote_map[date]
            initial = previous_date is None
            days, rf = 0, 0.0
            rate_metadata = {}
            if not initial:
                rate = rate_map[(previous_date, date)]
                days = (date - previous_date).days
                if (not rate.interval_valid or not rate.rate_known_at_entry
                        or rate.holding_calendar_days != days
                        or not pd.Timestamp(rate.assumed_release_date) < previous_date):
                    raise ValueError("Unavailable or mismatched interval rate")
                rf = float(rate.risk_free_return_act365)
                if not np.isclose(rf, float(rate.annual_yield_pct) / 100 * days / 365, atol=1e-15, rtol=1e-12):
                    raise AssertionError("Annual yield / ACT365 interval mismatch")
                rate_metadata = dict(rate_observation_date=rate.observation_date,
                    assumed_release_date=rate.assumed_release_date, annual_yield_pct=rate.annual_yield_pct)
            values = predictions.get((instruction["date"], model, scheme), {})
            target, decisions, decision = decide_pairs(active, values, instruction,
                h_in=entry_bps / 10000, tickers=TICKERS)
            final = decision["effective_action"] == "FINAL_LIQUIDATION"
            weights, raw_short, scale = overlay_weights(target, liquidate=final)
            new_cash, new_shares, details, summary = overlay_step(cash, shares,
                quotes if initial else previous_prices, quotes, dividends, weights,
                rf=rf, calendar_days=days)
            if abs(summary["nav_previous"] - previous_nav) > EPS:
                raise AssertionError("Carried balances do not reconstruct preceding NAV")
            denominator = np.nan if initial else 1.0 if intervals == 0 else previous_nav
            ret = np.nan if initial else summary["nav_post"] / denominator - 1
            intervals += int(not initial)
            action = decision["effective_action"]
            if action == "CLOSE_ALL_TO_CASH":
                action = "RESTORE_SPY"
            metadata = dict(model=model, scheme=scheme, split=split, entry_bps=entry_bps,
                exit_bps=entry_bps / 2, date=instruction["date"], signal_at=instruction["signal_at"],
                action_at=stamp, open_date=date)
            rows.append(metadata | summary | rate_metadata | dict(
                interval_entry_date=previous_date, is_initial_open=initial, is_final_open=final,
                holding_return_denominator=denominator, holding_interval_return=ret,
                target_pairs=len(target), raw_sector_short_weight=raw_short, sector_scale=scale,
                effective_action=action, signal_reason=instruction["signal_reason"],
                target_directions="|".join(f"{t}:{target[t]:+d}" for t in TICKERS if t in target)))
            asset_rows.extend(metadata | d for d in details)
            decision_rows.extend(metadata | d for d in decisions)
            cash, shares, active = new_cash, new_shares, target
            previous_date, previous_prices, previous_nav = date, quotes, summary["nav_post"]
        if not final or np.any(shares):
            raise AssertionError("Phase must finish liquidated")
        print(f"  {entry_bps:g} bp {model}/{scheme}: final NAV {previous_nav:.6f}", flush=True)
    book, assets = pd.DataFrame(rows), pd.DataFrame(asset_rows)
    return book, assets, assets.loc[assets.delta_shares.ne(0)].copy(), pd.DataFrame(decision_rows)


def holding_episodes(assets, series_id):
    records = []
    for ticker, group in assets.loc[assets.ticker.isin(TICKERS)].groupby("ticker", sort=True):
        group = group.sort_values("open_date")
        old, start_index, start_date = 0, None, None
        for i, row in enumerate(group.itertuples(index=False)):
            direction = int(np.sign(row.target_shares))
            if direction == old:
                continue
            if old:
                records.append(dict(series_id=series_id, ticker=ticker, direction=old,
                    entry_open=start_date, exit_open=row.open_date,
                    holding_intervals=i-start_index, calendar_days=(row.open_date-start_date).days,
                    end_reason="phase_end" if i == len(group)-1 else "reversal" if direction else "exit"))
            old, start_index, start_date = direction, i, row.open_date
        if old:
            raise AssertionError("Unclosed sector holding episode")
    return records


def strategy_exposures(book, assets):
    by_date = assets.assign(_gross=assets.market_value_post.abs(),
        _short=(-assets.market_value_post).clip(lower=0)).groupby("open_date")[["_gross", "_short"]].sum()
    ordered = book.sort_values("open_date").copy()
    value = by_date.reindex(ordered.open_date).to_numpy()
    ordered["gross_post"] = value[:, 0] / ordered.nav_post.to_numpy()
    ordered["short_post"] = value[:, 1] / ordered.nav_post.to_numpy()
    return ordered


def summary_row(book, *, series_id, kind, model, scheme, entry_bps,
                spy_returns, spy_ending, assets=None):
    group = book.sort_values("open_date").reset_index(drop=True)
    interval = group.iloc[1:]
    returns = interval.holding_interval_return.to_numpy(float)
    if not np.array_equal(interval.open_date.to_numpy(), spy_returns.index.to_numpy()):
        raise AssertionError("SPY comparison date mismatch")
    active_returns = returns-spy_returns.to_numpy(float)
    n, end = len(returns), float(group.nav_post.iloc[-1])
    active_std = float(np.std(active_returns, ddof=1))
    te = active_std*np.sqrt(252)
    ir = float(np.mean(active_returns)*252/te) if active_std > EPS else np.nan
    wealth = np.r_[1.0, group.nav_post.to_numpy(float)]
    drawdown = wealth/np.maximum.accumulate(wealth)-1
    episodes = []
    if assets is not None:
        group = strategy_exposures(group, assets)
        episodes = holding_episodes(assets, series_id)
        participation = float(group.target_pairs.iloc[:-1].gt(0).mean())
    else:
        group["gross_post"] = group.market_value_post.abs()/group.nav_post
        group["short_post"] = 0.0
        participation = 0.0
    total_fee = float(group.execution_fee.sum())
    borrow = float(group.borrow_fee.sum())
    financing = float(group.financing_interest.sum()) if "financing_interest" in group else 0.0
    turnover = float((group.traded_notional/group.nav_pre).sum())
    gross = group.gross_post.iloc[:-1]
    short = group.short_post.iloc[:-1]
    row = dict(split="validation", series_id=series_id, kind=kind, model=model, scheme=scheme,
        entry_bps=entry_bps, exit_bps=entry_bps/2 if np.isfinite(entry_bps) else np.nan,
        first_open_date=group.open_date.iloc[0], last_open_date=group.open_date.iloc[-1],
        opening_records=len(group), holding_intervals=n, ending_nav=end,
        net_cumulative_return=end-1, annualized_net_return=end**(252/n)-1,
        annualized_net_volatility=float(np.std(returns, ddof=1)*np.sqrt(252)),
        max_drawdown=float(drawdown.min()), cumulative_relative_to_spy=end/spy_ending-1,
        cumulative_return_difference_vs_spy=end-spy_ending,
        annualized_tracking_error=te, information_ratio=ir,
        mean_gross_exposure=float(gross.mean()), max_gross_exposure=float(gross.max()),
        mean_short_exposure=float(short.mean()), mean_holding_intervals=float(np.mean(
            [r["holding_intervals"] for r in episodes])) if episodes else np.nan,
        mean_holding_calendar_days=float(np.mean([r["calendar_days"] for r in episodes])) if episodes else np.nan,
        completed_sector_episodes=len(episodes), sector_participation=participation,
        total_two_sided_turnover=turnover, annualized_two_sided_turnover=turnover*252/n,
        mean_daily_two_sided_turnover=turnover/n,
        total_execution_fee=total_fee, total_borrow_fee=borrow, total_financing_interest=financing,
        total_cost=total_fee+borrow+financing, total_cash_interest=float(group.cash_interest.sum()),
        total_dividend_cash=float(group.dividend_cash.sum()), total_price_pnl=float(group.price_pnl.sum()),
        accounting_residual=float(end-(1+group.price_pnl.sum()+group.dividend_cash.sum()
            +group.cash_interest.sum()-total_fee-borrow-financing)),
    )
    if "actual_gross_pre" in group:
        row["max_pretrade_gross_after_drift"] = float(group.actual_gross_pre.iloc[1:].max())
    return row, episodes


def load_references(root):
    path = root/"results/backtest/threshold_selection/checkpoints"
    old = read_frame(path/"entry_15_daily.csv", parse_dates=["date", "open_date"])
    assets = read_frame(path/"entry_15_assets.csv", parse_dates=["date", "open_date"])
    bench = read_frame(root/"results/backtest/investment_benchmarks/benchmark_daily.csv",
        parse_dates=["open_date"])
    bench = bench.loc[bench["split"].eq("validation")].copy()
    bench["traded_notional"] = bench.trade_notional
    spy = bench.loc[bench.benchmark.eq("spy_buy_hold")].sort_values("open_date")
    spy_returns = spy.iloc[1:].set_index("open_date").holding_interval_return
    return old, assets, bench, spy_returns, float(spy.nav_post.iloc[-1])


def build_summary(root, experiments):
    old, old_assets, bench, spy_returns, spy_end = load_references(root)
    rows, episodes, curves = [], [], []
    for model in ("elastic_net", "gbdt"):
        for scheme in ("S0", "S1", "S2"):
            selected = old.model.eq(model) & old.scheme.eq(scheme)
            selected_assets = old_assets.model.eq(model) & old_assets.scheme.eq(scheme)
            series = f"original_{model}_{scheme}"
            row, ep = summary_row(old.loc[selected], series_id=series, kind="original", model=model,
                scheme=scheme, entry_bps=15, spy_returns=spy_returns, spy_ending=spy_end,
                assets=old_assets.loc[selected_assets])
            rows.append(row); episodes.extend(ep)
            curves.append(old.loc[selected, ["open_date", "nav_post"]].assign(series_id=series))
    for name, group in bench.groupby("benchmark", sort=True):
        row, _ = summary_row(group, series_id=name, kind="benchmark", model="", scheme="",
            entry_bps=np.nan, spy_returns=spy_returns, spy_ending=spy_end)
        rows.append(row)
        curves.append(group[["open_date", "nav_post"]].assign(series_id=name))
    for entry, book, assets in experiments:
        for (model, scheme), group in book.groupby(["model", "scheme"], sort=True):
            selected = assets.model.eq(model) & assets.scheme.eq(scheme)
            series = f"overlay_{entry:g}_{model}_{scheme}"
            row, ep = summary_row(group, series_id=series, kind="spy_overlay", model=model, scheme=scheme,
                entry_bps=entry, spy_returns=spy_returns, spy_ending=spy_end, assets=assets.loc[selected])
            rows.append(row); episodes.extend(ep)
            curves.append(group[["open_date", "nav_post"]].assign(series_id=series))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True), pd.DataFrame(episodes)


def draw_curves(curves, destination):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    colors = {"15": "#147d92", "10": "#d56332", "original": "#7d65a8", "cash": "#777777", "spy": "#171f2a"}
    for i, model in enumerate(("elastic_net", "gbdt")):
        for j, scheme in enumerate(("S0", "S1", "S2")):
            ax = axes[i, j]
            ids = [("spy_buy_hold", "SPY buy & hold", colors["spy"], "-"),
                   ("cash", "Cash", colors["cash"], ":"),
                   (f"original_{model}_{scheme}", "Original 15 / 7.5 bp", colors["original"], "--"),
                   (f"overlay_15_{model}_{scheme}", "SPY core 15 / 7.5 bp", colors["15"], "-"),
                   (f"overlay_10_{model}_{scheme}", "SPY core 10 / 5 bp", colors["10"], "-")]
            for series, label, color, style in ids:
                data = curves.loc[curves.series_id.eq(series)].sort_values("open_date")
                if data.empty:
                    continue
                # Show pre-entry capital at the same first-open timestamp.
                dates = pd.concat([data.open_date.iloc[:1], data.open_date])
                ax.plot(dates, np.r_[1.0, data.nav_post], label=label, color=color,
                    linestyle=style, linewidth=1.6 if series.startswith("overlay") else 1.1)
            ax.set_title(f"{'Elastic Net' if i == 0 else 'GBDT'} / {scheme}", loc="left", weight="bold")
            ax.grid(alpha=.18)
            if j == 0:
                ax.set_ylabel("Net wealth (initial capital = 1)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .015))
    fig.suptitle("Validation: SPY core with sector adjustments | all execution, borrow and financing costs included",
        fontsize=15, x=.06, ha="left")
    fig.text(.06, .925, "2018-01-03 to 2022-12-30 | Exploratory comparison; validation previously used for tuning", color="#555555")
    fig.tight_layout(rect=(.02, .065, 1, .91))
    fig.savefig(destination/"cumulative_nav.png", dpi=170)
    fig.savefig(destination/"cumulative_nav.svg")
    plt.close(fig)


def validate_run(book, assets, rates, opens, entry):
    checks = []
    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(f"{name}: {detail}")
    check("six_complete_strategies", len(book.groupby(["model", "scheme"])) == 6, "All delivered forecasts retained")
    for (model, scheme), group in book.groupby(["model", "scheme"], sort=True):
        group = group.sort_values("open_date").reset_index(drop=True)
        a = assets.loc[assets.model.eq(model) & assets.scheme.eq(scheme)]
        prefix = f"{entry:g}_{model}_{scheme}_"
        check(prefix+"coverage", pd.DatetimeIndex(group.open_date).equals(opens), "No dropped missing-signal or cash dates")
        check(prefix+"reset_and_liquidation", group.cash_prev.iloc[0] == 1 and group.target_pairs.iloc[-1] == 0
            and a.loc[a.open_date.eq(opens[-1]), "target_shares"].eq(0).all(), "Independent initial capital and charged terminal exit")
        check(prefix+"return_product", np.isclose(np.prod(1+group.holding_interval_return.iloc[1:]),
            group.nav_post.iloc[-1], atol=1e-12, rtol=1e-12), "Initial/final costs included once")
        check(prefix+"target_and_actual_caps", group.target_pairs.le(4).all()
            and group.target_gross_weight.le(1.5+EPS).all() and group.actual_gross_post.le(1.5+EPS).all(),
            "Target and actual post-fee gross at most 150%")
        check(prefix+"carry_formulas", np.allclose(group.cash_interest_base,
            np.maximum(group.cash_prev-group.short_collateral_start, 0), atol=1e-12)
            and np.allclose(group.financing_base, np.maximum(group.short_collateral_start-group.cash_prev, 0), atol=1e-12)
            and np.allclose(group.financing_interest, group.financing_base*group.interval_financing_return, atol=1e-12)
            and np.allclose(group.borrow_fee, group.short_collateral_start*.01*group.holding_calendar_days/365, atol=1e-12),
            "Restricted proceeds, financing and borrow are separate")
        check(prefix+"funding_rate", np.allclose(group.interval_financing_return.iloc[1:],
            rates.risk_free_return_act365.to_numpy()+.01*rates.holding_calendar_days.to_numpy()/365, atol=1e-15),
            "Same known DGS3MO interval plus fixed 1 percentage point spread")
        check(prefix+"unique_net_trades", not a.duplicated(["open_date", "ticker"]).any()
            and np.allclose(a.execution_fee, np.abs(a.delta_shares)*a.open_price*.0003, atol=1e-14), "SPY netted before actual trade fee")
        guard = group.effective_action.eq("RESTORE_SPY")
        check(prefix+"incomplete_signal_fallback", group.loc[guard, "target_pairs"].eq(0).all()
            and group.loc[guard, "target_gross_weight"].eq(1).all(), "Missing inputs restore SPY, never omit the day")
        no_sector = group.target_pairs.eq(0) & ~group.is_final_open
        spy = a.loc[a.ticker.eq("SPY")].set_index("open_date")
        check(prefix+"no_signal_spy_core", np.allclose(spy.loc[group.loc[no_sector, "open_date"], "actual_weight_post"], 1, atol=1e-12),
            "No eligible sector still holds 100% SPY")
        check(prefix+"accounting", group.pre_reconciliation_error.abs().le(EPS).all()
            and group.post_reconciliation_error.abs().le(EPS).all(), "Daily settled balances reconcile")
    return pd.DataFrame(checks)


def protected_snapshot(root):
    # Protect every pre-existing non-cache file in the project, including all
    # original results, source forecasts, model code, documentation and notebooks.
    paths = [p for folder in ("scripts", "data", "results", "notebooks")
             for p in (root/folder).rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and not p.is_relative_to(root/OUTPUT)
             and p.name not in {"backtest_spy_overlay.py", "check_spy_overlay.py"}]
    paths.extend(p for p in root.iterdir() if p.is_file())
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}


def protocol(root, destination, snapshot):
    config = dict(
        scope="New exploratory strategy; validation only", phase="validation",
        phase_opens=["2018-01-03", "2022-12-30"], planned_entry_bps=[15, 10],
        exit_relation="entry / 2", initial_nav=1.0, annualization=252,
        forecasts="Original Elastic Net and GBDT S0/S1/S2; no refitting or replacement",
        signals="Original strict boundaries, retention, global top-four entry and tie/reversal rules",
        execution="Daily next-open target restoration; net unique SPY position; fractional shares; no minimum holding period",
        overlay=dict(base_spy=1.0, raw_sector_weight=.125, maximum_sectors=4, maximum_sector_short=.25,
            maximum_target_gross=1.5, scaling="Scale ALL sector weights by min(1, .25 / raw short sum)",
            spy_weight="1 - sum(final sector weights)", incomplete_signal="Next open restore SPY and record source reason"),
        costs=dict(execution_bps=3, borrow_annual_rate=.01, funding_spread_annual=.01,
            cash_interest="max(C-B,0) * known RF interval return",
            financing="max(B-C,0) * (known RF interval return + .01 * actual days/365)",
            borrow="Actual net short value at previous execution open * .01 * actual days/365",
            collateral="Short proceeds restricted and earn no rebate; financing and borrow charged separately"),
        nav_sizing="NEW fee-funded post-cost NAV: solve x + .0003*sum(abs(x*w-old_value)) = NAV_pre; original reference remains unchanged",
        reason_for_nav_sizing="Keeps actual post-trade gross <=150%, including fees; old NAV_pre sizing would slightly breach at cap",
        dividend="Old signed shares receive/pay ex-date dividends before new trades; raw opens, no total-return double count",
        rate_availability="Reuse original assumed H.15 release calendar; release date strictly before interval entry; latest prior available observation",
        rate_limit="Original downloaded FRED snapshot with assumed release timing, not a reconstructed real-time vintage database",
        missing_execution_quote="Stop; no forward fill or invented execution",
        benchmarks="Reuse same-date original 15/7.5 strategy, cash and fixed-share SPY with entry/exit fees; SPY dividends remain cash earning later RF",
        statistics=dict(participation="Intervals with nonzero sector holdings at START / all 1257 intervals; SPY core excluded",
            holding="Unweighted mean per completed ticker/direction episode; size rebalances do not reset; reversals close and reopen; final exits included",
            exposure="Simple mean over interval-start post-trade gross/short weights; maximum post-trade gross; drift maximum separate",
            turnover="Two-sided actual net asset trade notional / NAV_pre, summed and annualized 252/N; both ends included; no half factor",
            relative_spy="Ending wealth ratio - 1; cumulative return difference also supplied",
            tracking_error="Sample std of paired daily net return differences * sqrt(252)",
            information_ratio="Mean daily net return difference * 252 / tracking error; zero tracking error -> NA",
            daily_return="First real interval denominator is initial capital 1, retaining initial fee; all cash and missing-signal dates kept",
            drawdown="NAV/running max - 1 with initial pre-fee capital 1 included",
            cost_amounts="NAV units, initial capital = 1; equal to fraction of initial capital; scalable to any chosen initial amount"),
        evidence="Exploratory validation reuse; validation already used for model and old threshold tuning; no untouched holdout claim",
        test_executed=False, threshold_selection=False, model_training=False, cost_grid=False,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        protected_file_count=len(snapshot), protected_hashes=snapshot,
    )
    write_json(destination/"experiment_config.json", config)
    return config


def markdown_table(summary):
    fields = ["series_id", "annualized_net_return", "annualized_net_volatility", "max_drawdown",
        "cumulative_relative_to_spy", "annualized_tracking_error", "information_ratio",
        "mean_holding_intervals", "sector_participation", "annualized_two_sided_turnover",
        "mean_gross_exposure", "max_gross_exposure", "mean_short_exposure",
        "total_execution_fee", "total_borrow_fee", "total_financing_interest", "total_cost"]
    labels = ["组合", "净年化", "年化波动", "最大回撤", "相对SPY累计", "跟踪误差", "信息比率",
        "持有区间均值", "行业参与", "年化双边换手", "平均总敞口", "最大总敞口", "平均空头",
        "交易费/NAV₀", "借券费/NAV₀", "融资利息/NAV₀", "总成本/NAV₀"]
    pct = set(fields)-{"series_id", "information_ratio", "mean_holding_intervals", "annualized_two_sided_turnover"}
    lines = ["|"+"|".join(labels)+"|", "|"+"|".join(["---"]*len(labels))+"|"]
    for row in summary.to_dict("records"):
        cells = []
        for field in fields:
            value = row[field]
            if field == "series_id": cells.append(str(value))
            elif pd.isna(value): cells.append("—")
            elif field in pct: cells.append(f"{value:.2%}")
            else: cells.append(f"{value:.2f}")
        lines.append("|"+"|".join(cells)+"|")
    return "\n".join(lines)


def write_report(destination, summary, curves, episodes):
    save_frame(destination/"summary.csv", summary)
    save_frame(destination/"nav_curves.csv", curves)
    save_frame(destination/"holding_episodes.csv", episodes)
    draw_curves(curves, destination)
    table = markdown_table(summary)
    new = summary.loc[summary.kind.eq("spy_overlay")]
    lines = ["# SPY 基础持仓＋行业调整：验证期探索性实验", "",
        "验证期：2018-01-03 至 2022-12-30；每组 1,257 个开盘到开盘持有区间。原始结果保留，测试期未运行。", "",
        "复用六组官方预测及原排序、保留、退出和反向规则；主成本：单边 3 bp、借券年化 1%、融资 DGS3MO＋1 个百分点。", "",
        "新策略每天按扣费后 NAV 恢复目标权重，保证调仓后实际总敞口不超过 150%。原策略参照保留其原有扣费前 NAV 配仓口径。", "",
        "![累计净值](cumulative_nav.png)", "", table, "",
        "年化换手为双边倍数/年，含期初与期末；成本为累计实际金额除以初始资金。初始资金设为 1，若按 100 万元展示，成本列乘以 100 万即可。", "",
        "持有区间均值按每个行业连续同方向持仓段计算，调仓数量变化不重置；行业参与只计算行业调整，不包含基础 SPY。", "",
        "相对 SPY 累计表现采用期末财富比减一；信息比率基于逐日净收益差，不用年化复合收益差代替。", "",
        "SPY 买入持有为固定股数、分红留存现金并在后续区间计息；新策略因每日再平衡会将分红重新投入目标组合。两者均计入进出成本。", "",
        "验证期已用于既有模型和门槛调整，本次结果仍属于探索性验证期分析。利率沿用原数据的假设公布时间，不代表重新构建的实时历史版本。", "",
        f"本轮新增策略 {len(new)} 组；净年化收益高于 SPY 的组合数：{int(new.cumulative_relative_to_spy.gt(0).sum())}。", ""]
    if set(new.entry_bps) == {10, 15}:
        high = new.loc[new.entry_bps.eq(15)].set_index(["model", "scheme"])
        low = new.loc[new.entry_bps.eq(10)].set_index(["model", "scheme"])
        numeric = ["annualized_net_return", "max_drawdown", "cumulative_relative_to_spy", "mean_holding_intervals",
            "sector_participation", "annualized_two_sided_turnover", "total_execution_fee", "total_borrow_fee",
            "total_financing_interest", "total_cost", "mean_gross_exposure", "mean_short_exposure"]
        paired = pd.DataFrame(index=high.index)
        for field in numeric:
            paired[field+"_15bp"] = high[field]
            paired[field+"_10bp"] = low[field]
            paired[field+"_delta_10_minus_15"] = low[field]-high[field]
        save_frame(destination/"threshold_paired_comparison.csv", paired.reset_index())
        wins = int((low.annualized_net_return > high.annualized_net_return).sum())
        lines += [f"10／5 bp 相比 15／7.5 bp，净年化收益改善的组合数：{wins}/6。逐对变化详见 threshold_paired_comparison.csv。", ""]
        for model, scheme in high.index:
            a, b = high.loc[(model, scheme)], low.loc[(model, scheme)]
            lines += [f"- {model}/{scheme}：净年化 {a.annualized_net_return:.2%} → {b.annualized_net_return:.2%}；"
                f"行业参与 {a.sector_participation:.2%} → {b.sector_participation:.2%}；"
                f"平均持有 {a.mean_holding_intervals:.2f} → {b.mean_holding_intervals:.2f} 个区间；"
                f"年化双边换手 {a.annualized_two_sided_turnover:.2f} → {b.annualized_two_sided_turnover:.2f} 倍；"
                f"总成本/初始资金 {a.total_cost:.2%} → {b.total_cost:.2%}。"]
    (destination/"report_中文.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # A browser-friendly view of the same single table and same figure.
    display = summary.copy()
    display = display[["series_id", "annualized_net_return", "max_drawdown", "cumulative_relative_to_spy",
        "mean_holding_intervals", "sector_participation", "annualized_two_sided_turnover", "total_cost"]]
    html = "<!doctype html><meta charset='utf-8'><title>SPY overlay validation</title><style>body{font:15px system-ui;margin:32px;color:#17232c}img{width:100%;max-width:1500px}table{border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}</style>"
    html += "<h1>SPY 基础持仓＋行业调整：验证期探索性实验</h1><p>2018-01-03—2022-12-30；测试期未运行。完整指标和口径见同目录 report_中文.md 与 summary.csv。</p><img src='cumulative_nav.png'>"
    html += display.to_html(index=False, float_format=lambda v: f"{v:.5f}")
    (destination/"report.html").write_text(html, encoding="utf-8")


def run(root, entry):
    root = root.resolve()
    destination = root/OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = protected_snapshot(root)
    protocol(root, destination, snapshot)
    source_hash = sha256(Path(__file__))
    signals, daily, prices, rates, opens = load_validation_inputs(root)
    reasons = read_frame(root/"results/backtest/daily_signal_status.csv", parse_dates=["date"],
        usecols=["date", "model", "scheme", "reason"])
    daily = daily.merge(reasons.rename(columns={"reason": "signal_reason"}),
        on=["date", "model", "scheme"], validate="one_to_one")
    # Persist the exact rate alignment used, before any portfolio replay.
    save_frame(destination/"validation_rate_alignment.csv", rates)
    folder = destination/f"experiment_{entry:g}bp"
    folder.mkdir(exist_ok=True)
    print(f"Starting validation-only {entry:g}/{entry/2:g} bp experiment", flush=True)
    book, assets, trades, decisions = replay_overlay(signals, daily, prices, rates, entry)
    checks = validate_run(book, assets, rates, opens, entry)
    for name, frame in (("daily.csv", book), ("assets.csv", assets), ("trades.csv", trades),
                        ("decisions.csv", decisions), ("checks.csv", checks)):
        save_frame(folder/name, frame)
    summary, curves, episodes = build_summary(root, [(entry, book, assets)])
    if summary.accounting_residual.abs().max() > EPS:
        raise AssertionError("Summary PnL decomposition failed")
    write_report(folder, summary, curves, episodes)
    write_json(folder/"manifest.json", dict(entry_bps=entry, exit_bps=entry/2,
        completed_utc=datetime.now(timezone.utc).isoformat(), code_sha256=source_hash,
        check_count=len(checks), checks_passed=bool(checks.passed.all()),
        protected_hashes=snapshot, daily_rows=len(book), sector_asset_rows=len(assets),
        test_executed=False, model_retrained=False,
        outputs={p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}))
    experiments = []
    for candidate in (15, 10):
        path = destination/f"experiment_{candidate}bp"
        if not (path/"manifest.json").exists():
            continue
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        if manifest["code_sha256"] != source_hash or manifest["protected_hashes"] != snapshot:
            raise AssertionError("Existing experiment has different code or protected inputs; replay it first")
        for name, digest in manifest["outputs"].items():
            if sha256(path/name) != digest:
                raise AssertionError(f"Completed experiment output changed: {path/name}")
        experiments.append((candidate,
            read_frame(path/"daily.csv", parse_dates=["open_date"]),
            read_frame(path/"assets.csv", parse_dates=["open_date"])))
    summary, curves, episodes = build_summary(root, experiments)
    write_report(destination, summary, curves, episodes)
    for path, digest in snapshot.items():
        if sha256(root/path) != digest:
            raise AssertionError(f"Protected original changed: {path}")
    write_json(destination/"run_manifest.json", dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), phase="validation",
        completed_entry_bps=[e[0] for e in experiments], new_strategy_count=6*len(experiments),
        summary_rows=len(summary), protected_file_count=len(snapshot), original_files_unchanged=True,
        test_executed=False, model_retrained=False, threshold_selected=False,
        outputs={p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p.name != "run_manifest.json"}))
    print(f"PASS {len(checks)} checks; {len(snapshot)} original files unchanged; report: {destination}", flush=True)
    print(summary.loc[summary.kind.ne("original"), ["series_id", "annualized_net_return", "max_drawdown",
        "cumulative_relative_to_spy", "sector_participation", "total_cost"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--entry-bps", type=int, choices=[15, 10], required=True)
    args = parser.parse_args()
    run(args.project_root, args.entry_bps)
