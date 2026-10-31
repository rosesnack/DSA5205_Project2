"""Protocol-defined shared cost settings and one-factor sensitivity scenarios.

Values come from 5205proj2.md. These scenarios are comparisons, not parameters to
select by performance. Prediction inputs and the public entry threshold are shared.
"""

from __future__ import annotations

from types import MappingProxyType

PARAMETER_FIELDS = ("execution_bps", "borrow_annual_rate", "full_cash_rebate")
_MAIN = MappingProxyType(dict(execution_bps=3.0, borrow_annual_rate=0.01, full_cash_rebate=False))


def main_cost_parameters():
    return dict(_MAIN)


def scenario_records():
    """Main plus seven one-factor changes; return fresh records on every call."""
    records = [dict(scenario_id="main", variation="main", is_main=True, **main_cost_parameters())]
    for field, values, prefix, multiplier in (
        ("execution_bps", (0.0, 1.0, 5.0), "execution", 1),
        ("borrow_annual_rate", (0.0, 0.02, 0.03), "borrow", 100)):
        for value in values:
            parameters = main_cost_parameters() | {field: value}
            records.append(dict(scenario_id=f"{prefix}_{value * multiplier:g}", variation=field,
                                is_main=False, **parameters))
    records.append(dict(scenario_id="full_cash_rebate", variation="full_cash_rebate", is_main=False,
                        **(main_cost_parameters() | dict(full_cash_rebate=True))))
    return records


def cost_settings():
    return dict(version="5205-costs-v1", source="../5205proj2.md: sections 3.6 and 3.7",
                main=main_cost_parameters(), scenarios=scenario_records(),
                sensitivity_design="one factor at a time; main plus seven predefined comparisons",
                shared_across_models_and_schemes=True, scenario_selection_by_performance=False,
                selected_public_entry_bps=None,
                execution_basis="sum of actual absolute net share changes times observed opens, after SPY netting",
                borrow_basis="starting actual net short market value times annual rate times natural days / 365",
                main_interest_basis="max(previous post-trade cash minus starting short collateral, 0)",
                full_rebate_interest_basis="previous post-trade cash, counted once",
                cash_rate_basis="exported open-to-open interval return, with no additional day-count division",
                thresholds="same public threshold for every cost scenario; never reselected per scenario")


def run_cost_scenarios(signals, daily, prices, rates, *, h_in, initial_nav, tickers):
    """Replay each predefined cost scenario from independent initial capital.

    Small preview inputs are used by step 8. Formal full-phase comparisons can
    use the same function only after the public threshold/protocol are frozen.
    """
    import pandas as pd
    if __package__:
        from .backtest_ledger import replay_ledger
    else:
        from backtest_ledger import replay_ledger

    books, assets_all, trades_all, summaries = [], [], [], []
    for scenario in scenario_records():
        parameters = {field: scenario[field] for field in PARAMETER_FIELDS}
        book, assets, trades = replay_ledger(signals, daily, prices, rates, h_in=h_in,
            initial_nav=initial_nav, tickers=tickers, require_final_liquidation=False, **parameters)
        for frame in (book, assets, trades):
            for field, value in scenario.items():
                frame[field] = value
            frame["entry_bps"] = h_in * 10000
            frame["example_only"] = True
            frame["truncated_preview"] = True
            frame["threshold_selected"] = False
        books.append(book)
        assets_all.append(assets)
        trades_all.append(trades)
        for (model, scheme, split), group in book.groupby(["model", "scheme", "split"], sort=True):
            last = group.sort_values("open_date").iloc[-1]
            identity = initial_nav + group.price_pnl.sum() + group.dividend_cash.sum() + group.cash_interest.sum()
            identity -= group.borrow_fee.sum() + group.execution_fee.sum()
            summaries.append(dict(model=model, scheme=scheme, split=split, **scenario,
                initial_nav=initial_nav, entry_bps=h_in * 10000, opening_records=len(group),
                ending_open_date=last.open_date, preview_ending_nav=float(last.nav_post),
                sum_price_pnl=float(group.price_pnl.sum()), sum_dividend_cash=float(group.dividend_cash.sum()),
                sum_cash_interest=float(group.cash_interest.sum()), sum_borrow_fee=float(group.borrow_fee.sum()),
                sum_execution_fee=float(group.execution_fee.sum()), traded_notional=float(group.traded_notional.sum()),
                borrow_exposed_intervals=int((group.short_collateral_start.gt(0) & group.holding_calendar_days.gt(0)).sum()),
                accounting_residual=float(last.nav_post - identity),
                example_only=True, truncated_preview=True, threshold_selected=False))
    summary = pd.DataFrame(summaries)
    main = summary.loc[summary.is_main, ["model", "scheme", "split", "preview_ending_nav",
        "sum_cash_interest", "sum_borrow_fee", "sum_execution_fee"]]
    summary = summary.merge(main, on=["model", "scheme", "split"], how="left",
                             validate="many_to_one", suffixes=("", "_main"), sort=False)
    for field in ("preview_ending_nav", "sum_cash_interest", "sum_borrow_fee", "sum_execution_fee"):
        summary["delta_" + field + "_vs_main"] = summary[field] - summary[field + "_main"]
    return (pd.concat(books, ignore_index=True), pd.concat(assets_all, ignore_index=True),
            pd.concat(trades_all, ignore_index=True), summary)


def synthetic_cost_fixture(tickers):
    """Four opens with sustained shorts, a weekend and ex-dividend events."""
    import pandas as pd

    specifications = [
        ("2020-01-02", "2020-01-03", "EVALUATE_SIGNAL", 100.0, 200.0, 0.0, 0.0),
        ("2020-01-03", "2020-01-06", "EVALUATE_SIGNAL", 99.0, 198.0, 1.0, 2.0),
        ("2020-01-06", "2020-01-07", "EVALUATE_SIGNAL", 101.0, 201.0, 0.0, 0.0),
        ("2020-01-07", "2020-01-08", "FINAL_LIQUIDATION", 100.0, 200.0, 0.0, 0.0)]
    signals, daily, prices, rates = [], [], [], []
    previous = None
    for signal_day, execution_day, action, sector_open, spy_open, sector_div, spy_div in specifications:
        date, open_date = pd.Timestamp(signal_day), pd.Timestamp(execution_day)
        daily.append(dict(date=date, model="synthetic", scheme="COMMON", split="validation",
            signal_at=(date + pd.Timedelta(hours=18)).tz_localize("America/New_York"),
            action_at=(open_date + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York"),
            action=action, force_flat=action == "FINAL_LIQUIDATION", signal_complete=True))
        signals.extend(dict(date=date, model="synthetic", scheme="COMMON", ticker=ticker,
                            y_hat=0.002 if ticker == tickers[0] else 0.0) for ticker in tickers)
        for ticker in [*tickers, "SPY"]:
            price = spy_open if ticker == "SPY" else sector_open if ticker == tickers[0] else 100.0
            dividend = spy_div if ticker == "SPY" else sector_div if ticker == tickers[0] else 0.0
            prices.append(dict(date=open_date, ticker=ticker, open=price, dividends=dividend))
        if previous is not None:
            days = (open_date - previous).days
            rates.append(dict(entry_date=previous, exit_date=open_date, interval_valid=True,
                rate_known_at_entry=True, holding_calendar_days=days, risk_free_return_act365=0.0001 * days))
        previous = open_date
    return tuple(pd.DataFrame(rows) for rows in (signals, daily, prices, rates))


def validate_cost_scenarios(book, assets, summary):
    """Validate settings/wiring and nonzero financing with the independent fixture."""
    import numpy as np
    import pandas as pd

    records = []
    def check(name, passed, detail):
        records.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name)

    scenarios = scenario_records()
    main = main_cost_parameters()
    check("protocol_main_parameters", main == dict(execution_bps=3.0, borrow_annual_rate=0.01, full_cash_rebate=False),
          "Main settings are the supplied protocol, not a performance-selected scenario")
    check("eight_unique_predefined_scenarios", len(scenarios) == 8 and len({s["scenario_id"] for s in scenarios}) == 8,
          "Main plus seven predefined one-factor comparisons")
    check("one_factor_at_a_time", all(sum(s[field] != main[field] for field in PARAMETER_FIELDS)
          == (0 if s["is_main"] else 1) for s in scenarios), "No unrequested factorial grid or simultaneous parameter change")
    check("execution_values_match_protocol", {s["execution_bps"] for s in scenarios} == {0, 1, 3, 5},
          "Execution costs are 0/1/3/5 bps")
    check("borrow_values_match_protocol", {s["borrow_annual_rate"] for s in scenarios} == {0, 0.01, 0.02, 0.03},
          "Borrow rates are decimal annual 0%/1%/2%/3%")
    mutated = main_cost_parameters()
    mutated["execution_bps"] = 999
    check("main_settings_cannot_be_mutated_through_copy", main_cost_parameters() == main,
          "Callers receive independent settings dictionaries")
    check("public_threshold_remains_unselected", cost_settings()["selected_public_entry_bps"] is None
          and not cost_settings()["scenario_selection_by_performance"], "Cost scenarios do not select the trading threshold")
    check("same_entry_threshold_every_scenario", book.entry_bps.nunique() == 1,
          "Every scenario receives the same explicit entry threshold")
    check("execution_fee_parameter_wired", np.allclose(book.execution_fee,
          book.traded_notional * book.execution_bps / 10000, rtol=1e-12, atol=1e-14),
          "Each scene charges actual net trading amounts at its own explicit bps")
    check("borrow_parameter_and_day_count_wired", np.allclose(book.borrow_fee,
          book.short_collateral_start * book.borrow_annual_rate * book.holding_calendar_days / 365,
          rtol=1e-12, atol=1e-14), "Borrow uses starting actual shorts and natural days")
    expected_base = np.where(book.full_cash_rebate, book.cash_prev, np.maximum(book.cash_prev - book.short_collateral_start, 0))
    check("interest_policy_wired_once", np.allclose(book.cash_interest,
          expected_base * book.interval_risk_free_return, rtol=1e-12, atol=1e-14),
          "Interest policy changes the correct base without duplicate rate/day-count conversion")
    check("zero_execution_does_not_zero_other_costs", summary.loc[summary.scenario_id.eq("execution_0"),
          "sum_execution_fee"].eq(0).all() and summary.loc[summary.scenario_id.eq("execution_0"), "sum_borrow_fee"].gt(0).all(),
          "Zero execution cost retains the main borrow and interest assumptions")
    check("scenario_accounting_conserves_nav", summary.accounting_residual.abs().le(1e-12).all(),
          "Ending wealth reconciles with price/dividend/interest/borrow/execution components")
    base_weights = assets.loc[assets.is_main, ["model", "scheme", "split", "open_date", "ticker", "target_weight"]]
    weights = assets.merge(base_weights, on=["model", "scheme", "split", "open_date", "ticker"],
                            validate="many_to_one", suffixes=("", "_main"))
    check("logical_pair_rules_unchanged", weights.target_weight.eq(weights.target_weight_main).all(),
          "Cost changes do not retune entries, exits, rankings or logical pair targets")
    first = book.loc[book.open_date.eq(pd.Timestamp("2020-01-06"))].set_index("scenario_id")
    for scenario_id, annual in (("borrow_0", 0.0), ("main", 0.01), ("borrow_2", 0.02), ("borrow_3", 0.03)):
        check("hand_borrow_" + scenario_id, np.isclose(first.loc[scenario_id, "borrow_fee"],
              0.125 * annual * 3 / 365, rtol=1e-12, atol=1e-14),
              "First holding interval: short value 0.125, three natural days, explicit annual rate")
    check("hand_full_rebate_increment", np.isclose(first.loc["full_cash_rebate", "cash_interest"]
          - first.loc["main", "cash_interest"], 0.125 * 0.0003, rtol=1e-12, atol=1e-14),
          "Rebating collateral adds exactly collateral times the exported interval return")
    check("fixture_has_real_financing_exposure", summary.borrow_exposed_intervals.eq(3).all(),
          "Synthetic comparisons include three intervals with nonzero short collateral")
    check("fixture_includes_final_liquidation", book.groupby("scenario_id", sort=False).tail(1).target_pairs.eq(0).all(),
          "Every synthetic scenario includes a complete final exit")
    return pd.DataFrame(records)
