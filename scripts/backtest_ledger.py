"""Open-to-open signed-share/cash/NAV ledger with explicit cost parameters."""

from __future__ import annotations

import math

if __package__:
    from .backtest_execution import plan_net_trades
    from .backtest_pair_decisions import decide_pairs
else:
    from backtest_execution import plan_net_trades
    from backtest_pair_decisions import decide_pairs


def ledger_step(cash_prev, shares_prev, previous_opens, opens, dividends, target_pairs, *,
                risk_free_return, calendar_days, execution_bps, borrow_annual_rate,
                full_cash_rebate, tickers):
    """Settle old positions, size from NAV_pre, then execute one net trade per asset.

    All rates are explicit. risk_free_return is the exported interval return and
    is never divided again. Borrow expense uses actual natural days / 365 and
    short market value at the interval start, after SPY netting. An initial open
    has zero days and no old shares, so newly purchased assets receive no dividend.
    """
    scalars = (cash_prev, risk_free_return, execution_bps, borrow_annual_rate)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("Cash and rate inputs must be finite")
    if risk_free_return <= -1 or execution_bps < 0 or borrow_annual_rate < 0:
        raise ValueError("Invalid return, execution fee or borrow rate")
    if isinstance(calendar_days, bool) or calendar_days < 0 or int(calendar_days) != calendar_days:
        raise ValueError("Holding days must be a nonnegative integer")
    if type(full_cash_rebate) is not bool:
        raise ValueError("full_cash_rebate must be a boolean")
    assets = [*tickers, "SPY"]
    if any(t not in assets or not math.isfinite(float(q)) for t, q in shares_prev.items()):
        raise ValueError("Invalid old share state")
    if calendar_days == 0 and any(shares_prev.values()):
        raise ValueError("Initial zero-day settlement cannot contain old shares")
    if calendar_days == 0 and risk_free_return != 0:
        raise ValueError("Initial zero-day settlement cannot earn interval interest")
    collateral = old_value_start = old_value_now = dividend_cash = price_pnl = 0.0
    old_details = {}
    for ticker in assets:
        q = float(shares_prev.get(ticker, 0.0))
        if q:
            old_open = float(previous_opens.get(ticker, float("nan")))
            current_open = float(opens.get(ticker, float("nan")))
            dividend = float(dividends.get(ticker, float("nan")))
            if not all(math.isfinite(value) for value in (old_open, current_open, dividend)) or min(old_open, current_open) <= 0:
                raise ValueError(f"Missing old-position open/dividend data for {ticker}")
            collateral += max(-q * old_open, 0.0)
            old_value_start += q * old_open
            old_value_now += q * current_open
            asset_dividend, asset_price_pnl = q * dividend, q * (current_open - old_open)
        else:
            old_open = float("nan")
            dividend = float(dividends.get(ticker, float("nan")))
            asset_dividend = asset_price_pnl = 0.0
        dividend_cash += asset_dividend
        price_pnl += asset_price_pnl
        old_details[ticker] = dict(previous_open=old_open, observed_dividend_per_share=dividend,
                                   dividend_cash_old=asset_dividend, price_pnl_old=asset_price_pnl)
    interest_base = float(cash_prev) if full_cash_rebate else max(float(cash_prev) - collateral, 0.0)
    interest = interest_base * risk_free_return
    borrow_fee = collateral * borrow_annual_rate * calendar_days / 365
    cash_pre = float(cash_prev) + dividend_cash + interest - borrow_fee
    nav_previous = float(cash_prev) + old_value_start
    nav_pre = cash_pre + old_value_now
    if nav_previous <= 0 or nav_pre <= 0:
        raise ValueError("Nonpositive NAV before execution")
    plans, plan_summary = plan_net_trades(target_pairs, shares_prev, opens, nav_pre, tickers=tickers)
    execution_fee = plan_summary["traded_notional"] * execution_bps / 10000
    cash_post = cash_pre + plan_summary["cash_change_before_costs"] - execution_fee
    shares_post = {row["ticker"]: row["target_shares"] for row in plans if row["target_shares"] != 0}
    market_value_post = sum(q * float(opens[ticker]) for ticker, q in shares_post.items())
    nav_post = cash_post + market_value_post
    if nav_post <= 0:
        raise ValueError("Nonpositive NAV after execution fees")
    expected_pre = nav_previous + price_pnl + dividend_cash + interest - borrow_fee
    pre_error, post_error = nav_pre - expected_pre, nav_post - (nav_pre - execution_fee)
    if not math.isclose(nav_pre, expected_pre, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Pre-trade NAV does not reconcile with price/dividend/financing changes")
    if not math.isclose(nav_post, nav_pre - execution_fee, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Net trading changed NAV beyond its execution fee")
    asset_rows = [row | old_details[row["ticker"]] | dict(
        market_value_post=row["target_shares"] * row["open_price"] if row["target_shares"] else 0.0,
        actual_weight_post=(row["target_shares"] * row["open_price"] / nav_post) if row["target_shares"] else 0.0,
        execution_fee=row["trade_notional"] * execution_bps / 10000,
        cash_change_after_execution_fee=row["cash_change_before_costs"] - row["trade_notional"] * execution_bps / 10000
    ) for row in plans]
    if not math.isclose(sum(row["cash_change_after_execution_fee"] for row in asset_rows),
                        cash_post - cash_pre, rel_tol=1e-12, abs_tol=1e-10):
        raise AssertionError("Per-asset principal/fee flows do not reconcile with post-trade cash")
    summary = dict(cash_prev=cash_prev, nav_previous=nav_previous, holding_calendar_days=calendar_days,
                   short_collateral_start=collateral, cash_interest_base=interest_base,
                   interval_risk_free_return=risk_free_return, dividend_cash=dividend_cash,
                   price_pnl=price_pnl, cash_interest=interest, borrow_fee=borrow_fee,
                   cash_pre=cash_pre, nav_pre=nav_pre, traded_notional=plan_summary["traded_notional"],
                   cash_change_before_costs=plan_summary["cash_change_before_costs"],
                   execution_fee=execution_fee, cash_post=cash_post, market_value_post=market_value_post,
                   nav_post=nav_post, pre_reconciliation_error=pre_error, post_reconciliation_error=post_error,
                   target_pairs=len(target_pairs), target_gross_weight=plan_summary["target_gross_weight"])
    return dict(cash=cash_post, shares=shares_post), asset_rows, summary


def replay_ledger(signals, daily, prices, rates, *, h_in, initial_nav, execution_bps,
                  borrow_annual_rate, full_cash_rebate, tickers, require_final_liquidation=False):
    """Replay actual signed shares from stored signal instructions, reset per phase.

    For holding returns, the first real interval starts at initial capital so the
    first entry fee is included once. The initial zero-length opening row has no
    holding return. Later intervals divide by previous post-trade NAV. Full return
    products therefore reproduce ending NAV / initial capital, including both ends.
    """
    import pandas as pd

    if not math.isfinite(initial_nav) or initial_nav <= 0:
        raise ValueError("initial_nav must be positive and finite")
    if signals.duplicated(["date", "ticker", "model", "scheme"]).any() or daily.duplicated(["date", "model", "scheme"]).any():
        raise ValueError("Duplicate signal or daily instruction keys")
    if prices.duplicated(["date", "ticker"]).any() or rates.duplicated(["entry_date", "exit_date"]).any():
        raise ValueError("Duplicate open/dividend or interest-interval keys")
    if any(not pd.api.types.is_bool_dtype(rates[field]) or rates[field].isna().any()
           for field in ("interval_valid", "rate_known_at_entry")):
        raise ValueError("Interest-interval validity/availability flags must be nonmissing booleans")
    quote_rows = prices.set_index(["date", "ticker"])
    rate_rows = rates.set_index(["entry_date", "exit_date"])
    prediction_rows = {key: dict(zip(group.ticker, group.y_hat)) for key, group in
                       signals.groupby(["date", "model", "scheme"], sort=False)}
    ledger_rows, asset_rows, trade_rows = [], [], []
    for (model, scheme, split), group in daily.groupby(["model", "scheme", "split"], sort=True):
        state = dict(cash=float(initial_nav), shares={})
        active, previous_quotes = {}, {}
        previous_date, previous_nav = None, float(initial_nav)
        return_product, holding_intervals = 1.0, 0
        last_action = None
        for instruction in group.sort_values("date").to_dict("records"):
            if instruction["action"] == "NO_ORDER_OUTSIDE_PHASE":
                continue
            if pd.isna(instruction["action_at"]):
                raise ValueError("A scheduled ledger action requires an execution timestamp")
            open_date = pd.Timestamp(instruction["action_at"]).tz_convert("America/New_York").tz_localize(None).normalize()
            if previous_date is not None and open_date <= previous_date:
                raise ValueError("Ledger opens must be strictly increasing within each phase")
            quotes, dividends = {}, {}
            for ticker in [*tickers, "SPY"]:
                if (open_date, ticker) in quote_rows.index:
                    quote = quote_rows.loc[(open_date, ticker)]
                    quotes[ticker], dividends[ticker] = float(quote.open), float(quote.dividends)
            days, rf = 0, 0.0
            if previous_date is not None:
                key = (previous_date, open_date)
                if key not in rate_rows.index:
                    raise ValueError(f"Missing exact adjacent open interval: {key}")
                rate = rate_rows.loc[key]
                days = (open_date - previous_date).days
                if not bool(rate.interval_valid) or not bool(rate.rate_known_at_entry) or int(rate.holding_calendar_days) != days:
                    raise ValueError("Invalid, unavailable or calendar-mismatched cash rate interval")
                rf = float(rate.risk_free_return_act365)
            values = prediction_rows.get((instruction["date"], model, scheme), {})
            target, _, decision = decide_pairs(active, values, instruction, h_in=h_in, tickers=tickers)
            next_state, assets, summary = ledger_step(
                state["cash"], state["shares"], previous_quotes, quotes, dividends, target,
                risk_free_return=rf, calendar_days=days, execution_bps=execution_bps,
                borrow_annual_rate=borrow_annual_rate, full_cash_rebate=full_cash_rebate, tickers=tickers)
            if not math.isclose(summary["nav_previous"], previous_nav, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError("Carried cash/shares do not reconstruct previous post-trade NAV")
            initial = previous_date is None
            denominator = float("nan") if initial else initial_nav if holding_intervals == 0 else previous_nav
            holding_return = float("nan") if initial else summary["nav_post"] / denominator - 1
            if not initial:
                holding_intervals += 1
                return_product *= 1 + holding_return
            metadata = dict(date=instruction["date"], model=model, scheme=scheme, split=split,
                            action_at=instruction["action_at"], open_date=open_date,
                            interval_entry_date=previous_date if previous_date is not None else pd.NaT)
            ledger_rows.append(metadata | summary | dict(effective_action=decision["effective_action"],
                is_initial_open=initial, initial_nav=initial_nav, holding_return_denominator=denominator,
                holding_interval_return=holding_return))
            asset_rows.extend(metadata | row for row in assets)
            trade_rows.extend(metadata | row for row in assets if row["delta_shares"] != 0)
            state, active, previous_quotes = next_state, target, quotes
            previous_date, previous_nav = open_date, summary["nav_post"]
            last_action = decision["effective_action"]
        if holding_intervals and not math.isclose(initial_nav * return_product, previous_nav, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("Holding return product omits an initial/final fee or repeats a cash flow")
        if require_final_liquidation and (last_action != "FINAL_LIQUIDATION" or state["shares"]):
            raise AssertionError("A full phase must end with scheduled liquidation and no shares")
    assets = pd.DataFrame(asset_rows)
    return pd.DataFrame(ledger_rows), assets, pd.DataFrame(trade_rows, columns=assets.columns)


def validate_ledger(tickers):
    """Hand-calculated accounting and replay cases, with no model parameter search."""
    import pandas as pd

    t = list(tickers)
    records, examples = [], []
    quotes = {t[0]: 100.0, t[1]: 50.0, "SPY": 200.0}
    zero_div = dict.fromkeys(quotes, 0.0)
    old = {t[0]: 125.0, "SPY": -62.5}
    defaults = dict(risk_free_return=0.0, calendar_days=1, execution_bps=0.0,
                    borrow_annual_rate=0.0, full_cash_rebate=False, tickers=tickers)

    def check(name, passed, detail):
        records.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name)

    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    def step(name, cash, shares, prev, current, div, pairs, **kwargs):
        original = (dict(shares), dict(prev), dict(current), dict(div), dict(pairs))
        state, assets, row = ledger_step(cash, shares, prev, current, div, pairs, **(defaults | kwargs))
        check(name + "_reconciles", close(row["nav_post"], row["nav_pre"] - row["execution_fee"])
              and (shares, prev, current, div, pairs) == original,
              "Self-financing NAV identity holds and input states are unchanged")
        examples.append(dict(example_case=name, example_only=True, **row))
        return state, assets, row

    entry, _, row = step("initial_entry", 100000.0, {}, {}, quotes, zero_div, {t[0]: 1}, calendar_days=0)
    check("short_proceeds_are_not_profit", row["cash_post"] == 100000 and row["nav_post"] == 100000
          and entry["shares"][t[0]] == 125 and entry["shares"]["SPY"] == -62.5,
          "Equal long/short principal cash flows preserve initial capital")
    paid_entry, _, row = step("initial_entry_fee", 100000.0, {}, {}, quotes, zero_div, {t[0]: 1},
                              calendar_days=0, execution_bps=3.0)
    check("entry_fee_and_pre_fee_sizing", row["execution_fee"] == 7.5 and row["nav_post"] == 99992.5
          and paid_entry["shares"] == old, "Size from NAV_pre, deduct the initial fee once afterward")
    _, _, row = step("no_buy_day_dividend", 100000.0, {}, {}, quotes, {t[0]: 1, "SPY": 2},
                      {t[0]: 1}, calendar_days=0)
    check("new_positions_receive_no_ex_day_dividend", row["dividend_cash"] == 0,
          "Dividends apply only to old shares, never newly purchased shares")
    _, _, row = step("long_ex_dividend_exit", 100000.0, old, quotes, quotes | {t[0]: 99},
                      zero_div | {t[0]: 1}, {})
    check("old_long_dividend_offsets_ex_price_drop", row["price_pnl"] == -125
          and row["dividend_cash"] == 125 and close(row["nav_post"], 100000),
          "Old long earns dividend even when sold at the ex-date open; no double counting")
    _, _, row = step("short_ex_dividend_exit", 100000.0, {t[0]: -125, "SPY": 62.5}, quotes,
                      quotes | {t[0]: 99}, zero_div | {t[0]: 1}, {})
    check("old_short_pays_dividend", row["price_pnl"] == 125 and row["dividend_cash"] == -125
          and close(row["nav_post"], 100000), "Short dividends are signed expenses")
    _, _, row = step("spy_dividend_expense", 100000.0, old, quotes, quotes, zero_div | {"SPY": 2}, {})
    check("short_spy_dividend_is_included", row["dividend_cash"] == -125 and close(row["nav_post"], 99875),
          "SPY short dividends are treated like any other short asset")
    _, _, main_interest = step("own_cash_interest", 100000.0, old, quotes, quotes, zero_div, {}, risk_free_return=0.001)
    check("collateral_excluded_from_main_interest", main_interest["short_collateral_start"] == 12500
          and main_interest["cash_interest_base"] == 87500 and main_interest["cash_interest"] == 87.5,
          "Use starting cash minus starting actual short market value; no second day-count division")
    _, _, rebate = step("full_cash_rebate", 100000.0, old, quotes, quotes, zero_div, {},
                         risk_free_return=0.001, full_cash_rebate=True)
    check("full_cash_rebate_not_double_interest", rebate["cash_interest"] == 100
          and close(rebate["cash_interest"] - main_interest["cash_interest"], 12.5),
          "Full rebate accrues on total book cash once")
    _, _, row = step("own_available_cash_floor", 1000.0, old, quotes, quotes, zero_div, {}, risk_free_return=0.001)
    check("negative_own_available_cash_does_not_earn_interest", row["cash_interest_base"] == 0
          and row["cash_interest"] == 0, "Main interest base is max(cash minus collateral, zero)")
    _, _, row = step("weekend_borrow", 100000.0, old, quotes, quotes | {"SPY": 220}, zero_div, {},
                      calendar_days=3, borrow_annual_rate=0.01)
    check("borrow_uses_starting_short_and_natural_days", close(row["borrow_fee"], 12500 * 0.01 * 3 / 365),
          "Borrow uses interval-start collateral, not the changed end-open short value")
    _, _, row = step("netted_short_collateral", 100000.0, {t[0]: 125, t[1]: -250}, quotes, quotes,
                      zero_div, {t[0]: 1, t[1]: -1}, risk_free_return=0.001, borrow_annual_rate=0.01)
    check("financing_uses_actual_netted_holdings", row["short_collateral_start"] == 12500,
          "Opposing SPY legs create no phantom collateral or borrow expense")
    _, _, row = step("fee_driven_small_rebalance", paid_entry["cash"], paid_entry["shares"], quotes,
                      quotes, zero_div, {t[0]: 1}, execution_bps=3.0)
    check("small_rebalance_fee_is_not_omitted", close(row["traded_notional"], 1.875)
          and close(row["execution_fee"], 0.0005625), "Retained pair rebalances after prior fee reduces NAV")
    final, _, row = step("final_liquidation_fee", paid_entry["cash"], paid_entry["shares"], quotes,
                          quotes, zero_div, {}, execution_bps=3.0)
    check("initial_and_final_fees_both_in_nav", not final["shares"] and close(final["cash"], 99985)
          and row["execution_fee"] == 7.5, "Round trip at unchanged prices loses exactly both 7.5 fees")
    cash, _, row = step("cash_compound_one", 1.0, {}, {}, {}, {}, {}, risk_free_return=0.001)
    cash, _, row = step("cash_compound_two", cash["cash"], {}, {}, {}, {}, {}, risk_free_return=0.002)
    check("cash_interest_compounds", close(cash["cash"], 1.001 * 1.002), "Interest uses last post-trade cash")
    _, _, row = step("valid_zero_rate", 1.0, {}, {}, {}, {}, {})
    check("zero_cash_rate_remains_valid", row["cash_interest"] == 0 and row["nav_post"] == 1,
          "Zero is a valid rate, not a missing-value sentinel")
    for name, shares, prev, current, div, extra in (
        ("missing_old_dividend", old, quotes, quotes, {}, {}),
        ("missing_old_open", old, {}, quotes, zero_div, {}),
        ("invalid_interval_rate", {}, {}, {}, {}, dict(risk_free_return=float("nan"))),
        ("negative_execution_fee", {}, {}, {}, {}, dict(execution_bps=-1)),
        ("negative_borrow_rate", {}, {}, {}, {}, dict(borrow_annual_rate=-0.01)),
        ("invalid_calendar_days", {}, {}, {}, {}, dict(calendar_days=-1)),
        ("zero_day_old_shares", old, quotes, quotes, zero_div, dict(calendar_days=0))):
        try:
            ledger_step(100000.0, shares, prev, current, div, {}, **(defaults | extra))
        except ValueError:
            check(name, True, "Incomplete accounting inputs/configuration rejected")
        else:
            check(name, False, "Invalid accounting input was not rejected")

    # End-of-validation round trip, followed by a separate initial test signal.
    specs = [("2022-12-28", "2022-12-29", "validation", "EVALUATE_SIGNAL", 0.002),
             ("2022-12-29", "2022-12-30", "validation", "FINAL_LIQUIDATION", 0.002),
             ("2023-01-03", "2023-01-04", "test", "EVALUATE_SIGNAL", 0.0007)]
    ds, ss, ps = [], [], []
    for day, execution_day, split, action, value in specs:
        day, execution_day = pd.Timestamp(day), pd.Timestamp(execution_day)
        ds.append(dict(date=day, model="fixture", scheme="S0", split=split,
                       signal_at=(day + pd.Timedelta(hours=18)).tz_localize("America/New_York"),
                       action_at=(execution_day + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York"),
                       action=action, force_flat=action == "FINAL_LIQUIDATION", signal_complete=True))
        ss.extend(dict(date=day, model="fixture", scheme="S0", ticker=ticker,
                       y_hat=value if ticker == t[0] else 0.0) for ticker in t)
        ps.extend(dict(date=execution_day, ticker=ticker, open=price, dividends=0.0) for ticker, price in quotes.items())
    rf = pd.DataFrame([dict(entry_date=pd.Timestamp("2022-12-29"), exit_date=pd.Timestamp("2022-12-30"),
                            interval_valid=True, rate_known_at_entry=True, holding_calendar_days=1,
                            risk_free_return_act365=0.0)])
    args = dict(h_in=0.001, initial_nav=1.0, execution_bps=3.0, borrow_annual_rate=0.0,
                full_cash_rebate=False, tickers=tickers)
    daily_fixture, signal_fixture, price_fixture = pd.DataFrame(ds), pd.DataFrame(ss), pd.DataFrame(ps)
    logs, holdings, trades = replay_ledger(signal_fixture, daily_fixture, price_fixture, rf, **args)
    val_end = logs.loc[logs["split"].eq("validation")].iloc[-1]
    first_test = logs.loc[logs["split"].eq("test")].iloc[0]
    check("return_product_includes_both_end_fees", close(val_end.nav_post, 0.99985)
          and close(val_end.holding_interval_return, -0.00015), "Initial entry fee is included in the first actual holding return")
    check("phase_cash_shares_and_pairs_reset", first_test.cash_prev == 1 and first_test.target_pairs == 0
          and first_test.nav_post == 1, "Test begins from independent unit cash, without validation holdings")
    check("first_open_has_no_holding_interval", logs.loc[logs.is_initial_open, "holding_interval_return"].isna().all(),
          "No zero-length synthetic daily return is inserted before first holding interval")
    check("final_trade_logged_once_per_asset", len(trades.loc[trades.open_date.eq(pd.Timestamp("2022-12-30"))]) == 2,
          "Final exit records the two actual net trades")
    poisoned = signal_fixture.assign(target_relative_return=-999, label_valid=False, common_scoreable=False)
    poison_logs, _, _ = replay_ledger(poisoned, daily_fixture, price_fixture, rf, **args)
    pd.testing.assert_frame_equal(logs, poison_logs, check_exact=True)
    check("future_outcomes_do_not_enter_ledger", True, "Future-label fields added to signals leave all ledger records unchanged")
    replay_ledger(signal_fixture.loc[signal_fixture.date.lt(pd.Timestamp("2023-01-01"))],
                  daily_fixture.loc[daily_fixture["split"].eq("validation")], price_fixture, rf,
                  require_final_liquidation=True, **args)
    check("full_phase_requires_final_liquidation", True, "Complete validation fixture exits all shares at its scheduled final open")
    for name, altered in (("missing_exact_rate_interval", rf.iloc[:0]),
                           ("rate_not_known_at_entry", rf.assign(rate_known_at_entry=False)),
                           ("mismatched_rate_day_count", rf.assign(holding_calendar_days=3))):
        try:
            replay_ledger(signal_fixture, daily_fixture, price_fixture, altered, **args)
        except ValueError:
            check(name, True, "Rate alignment/availability error stops the ledger instead of skipping or filling it")
        else:
            check(name, False, "Invalid interval was not rejected")
    try:
        replay_ledger(signal_fixture, daily_fixture, price_fixture, rf, require_final_liquidation=True, **args)
    except AssertionError:
        check("truncated_phase_rejected_as_formal_full_run", True,
              "The test fixture is deliberately incomplete and cannot be labeled a completed full phase")
    else:
        check("truncated_phase_rejected_as_formal_full_run", False, "Incomplete phase was accepted as a full run")
    return pd.DataFrame(records), pd.DataFrame(examples), logs
