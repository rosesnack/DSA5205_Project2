"""Convert pair directions to net target shares and share-difference trade plans.

Call at a scheduled execution open after step 7 supplies settled NAV_pre and old
shares. No dividends, interest, borrow fees, execution fees or NAV are settled here.
"""

from __future__ import annotations

import math

LEG_WEIGHT = 0.125


def pair_weights(target_pairs, *, tickers):
    """Return nine sector weights and the combined SPY weight, without rescaling."""
    if len(tickers) != 9 or len(set(tickers)) != 9 or "SPY" in tickers:
        raise ValueError("Expected nine distinct sector tickers, excluding SPY")
    if len(target_pairs) > 4 or any(t not in tickers or isinstance(d, bool)
                                   or d not in (-1, 1) for t, d in target_pairs.items()):
        raise ValueError("Expected at most four sector pairs with direction +/-1")
    weights = dict.fromkeys([*tickers, "SPY"], 0.0)
    for ticker, direction in target_pairs.items():
        weights[ticker] = LEG_WEIGHT * direction
        weights["SPY"] -= LEG_WEIGHT * direction
    return weights


def plan_net_trades(target_pairs, current_shares, open_prices, nav_pre, *, tickers):
    """Plan fractional shares using observed opens and a caller-supplied NAV_pre.

    SPY is combined before share targets and actual trade notional are computed.
    There is no weight tolerance band or rounding of small trades. Quotes are
    mandatory for held/nonzero-target assets; unrelated zero positions need none.
    Do not call on NO_ORDER_OUTSIDE_PHASE: the scheduler must skip those signals.
    """
    if not math.isfinite(nav_pre) or nav_pre <= 0:
        raise ValueError("NAV_pre must be positive and finite")
    weights = pair_weights(target_pairs, tickers=tickers)
    if any(t not in weights or not math.isfinite(float(q)) for t, q in current_shares.items()):
        raise ValueError("Current shares must be finite and limited to the ten portfolio assets")
    rows = []
    for ticker, weight in weights.items():
        old = float(current_shares.get(ticker, 0.0))
        quote_required = bool(old != 0 or weight != 0)
        try:
            price = float(open_prices.get(ticker, float("nan")))
        except (ValueError, TypeError):
            price = float("nan")
        if quote_required and (not math.isfinite(price) or price <= 0):
            raise ValueError(f"Missing/invalid observed execution open for {ticker}")
        target_notional = nav_pre * weight
        target = target_notional / price if weight else 0.0
        delta = target - old
        traded = abs(delta) * price if delta else 0.0
        cash_change = -delta * price if delta else 0.0
        rows.append(dict(ticker=ticker, target_weight=weight, open_price=price,
                         current_shares=old, target_notional=target_notional,
                         target_shares=target, delta_shares=delta,
                         trade_side="BUY" if delta > 0 else "SELL" if delta < 0 else "NONE",
                         trade_notional=traded, cash_change_before_costs=cash_change,
                         quote_required=quote_required))
    gross = sum(abs(weight) for weight in weights.values())
    net = sum(weights.values())
    if gross > 1 or net != 0:
        raise AssertionError("Combined targets must be dollar-neutral and at most 100% gross")
    summary = dict(nav_pre=nav_pre, pair_count=len(target_pairs),
                   gross_weight_before_spy_netting=2 * LEG_WEIGHT * len(target_pairs),
                   target_gross_weight=gross, target_net_weight=net,
                   traded_notional=sum(row["trade_notional"] for row in rows),
                   traded_notional_over_nav=sum(row["trade_notional"] for row in rows) / nav_pre,
                   cash_change_before_costs=sum(row["cash_change_before_costs"] for row in rows),
                   nonzero_trade_assets=sum(row["delta_shares"] != 0 for row in rows))
    return rows, summary


def validate_net_trades(tickers):
    """Independent numerical cases for weights, quotes and net share differences."""
    import pandas as pd

    t = list(tickers)
    records, worked_rows = [], []
    prices = {t[0]: 100.0, t[1]: 50.0, t[2]: 80.0, t[3]: 125.0, "SPY": 200.0}

    def check(name, passed, detail):
        records.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name)

    def plan(name, pairs, old, quotes, nav):
        before = (dict(pairs), dict(old), dict(quotes))
        rows, summary = plan_net_trades(pairs, old, quotes, nav, tickers=tickers)
        check(name + "_inputs_unchanged", (pairs, old, quotes) == before,
              "Planning does not mutate target pairs, old shares or observed quotes")
        worked_rows.extend(dict(example_case=name, nav_pre=nav, example_only=True, **row) for row in rows)
        return {row["ticker"]: row for row in rows}, summary

    positive = pair_weights({t[0]: 1}, tickers=tickers)
    negative = pair_weights({t[0]: -1}, tickers=tickers)
    check("pair_direction_and_hedge", positive[t[0]] == 0.125 and positive["SPY"] == -0.125
          and negative[t[0]] == -0.125 and negative["SPY"] == 0.125,
          "Each sector and its opposite SPY leg have +/-12.5% weights")
    mixed = pair_weights({t[0]: 1, t[1]: -1}, tickers=tickers)
    check("opposite_spy_legs_cancel", mixed["SPY"] == 0 and sum(abs(v) for v in mixed.values()) == 0.25,
          "Two opposite pairs have 25% combined gross after SPY cancellation")
    four = pair_weights(dict.fromkeys(t[:4], 1), tickers=tickers)
    check("four_same_direction_pairs_cap_gross", four["SPY"] == -0.5 and sum(abs(v) for v in four.values()) == 1,
          "Four pairs have at most 100% target gross; dollar net equals zero")
    four_mixed = pair_weights(dict(zip(t[:4], [1, 1, -1, -1])), tickers=tickers)
    check("four_mixed_pairs_can_have_lower_gross", four_mixed["SPY"] == 0
          and sum(abs(v) for v in four_mixed.values()) == 0.5,
          "No rescaling after SPY netting reduces gross")

    entry, summary = plan("initial_entry", {t[0]: 1}, {}, prices, 100000.0)
    check("initial_shares_from_pretrade_nav", entry[t[0]]["target_shares"] == 125
          and entry["SPY"]["target_shares"] == -62.5 and summary["traded_notional"] == 25000,
          "NAV 100,000: buy 125 at 100 and short 62.5 SPY at 200")
    check("few_pairs_not_rescaled_or_short_proceeds_reinvested", summary["target_gross_weight"] == 0.25
          and entry[t[0]]["target_notional"] == 12500,
          "One pair retains 12.5% per leg; short cash does not increase the NAV basis")
    old = {t[0]: 125.0, "SPY": -62.5}
    unchanged, summary = plan("unchanged_prices_and_nav", {t[0]: 1}, old, prices, 100000.0)
    check("unchanged_shares_no_redundant_round_trip", summary["traded_notional"] == 0
          and summary["nonzero_trade_assets"] == 0, "No close-and-reopen when current and target shares agree")
    rebalance, summary = plan("retained_pair_rebalanced", {t[0]: 1}, old, prices | {t[0]: 110.0}, 101250.0)
    check("retained_pair_rebalanced_at_every_open", math.isclose(rebalance[t[0]]["target_shares"], 12656.25 / 110)
          and rebalance["SPY"]["target_shares"] == -63.28125
          and rebalance[t[0]]["delta_shares"] < 0 and rebalance["SPY"]["delta_shares"] < 0,
          "Same pair: new prices/NAV require actual small share adjustments")
    tiny, _ = plan_net_trades({t[0]: 1}, {t[0]: 125 + 1e-8, "SPY": -62.5}, prices, 100000.0, tickers=tickers)
    check("no_small_trade_tolerance_band", next(row for row in tiny if row["ticker"] == t[0])["trade_notional"] > 0,
          "A nonzero share difference is retained, without a minimum trade band")
    reverse, summary = plan("pair_reversal", {t[0]: -1}, old, prices, 100000.0)
    check("reversal_as_one_net_delta_per_asset", reverse[t[0]]["delta_shares"] == -250
          and reverse["SPY"]["delta_shares"] == 125 and summary["nonzero_trade_assets"] == 2
          and summary["traded_notional"] == 50000,
          "Long-to-short reversal is one signed share change per asset")
    rotation, summary = plan("sector_rotation_spy_unchanged", {t[1]: 1}, old, prices, 100000.0)
    check("spy_netted_before_trade_amounts", rotation["SPY"]["delta_shares"] == 0
          and summary["traded_notional"] == 25000 and summary["nonzero_trade_assets"] == 2,
          "Sector rotation does not charge fictitious SPY closing/opening notional")
    netted, summary = plan("opposite_pairs_close_old_spy", {t[0]: 1, t[1]: -1}, {"SPY": -10}, prices, 100000.0)
    check("net_zero_spy_target_still_closes_old_spy", netted["SPY"]["target_shares"] == 0
          and netted["SPY"]["delta_shares"] == 10, "An old SPY holding is closed when the combined target is zero")
    close, summary = plan("final_liquidation", {}, old, prices, 100000.0)
    check("final_and_missing_signal_liquidation", close[t[0]]["delta_shares"] == -125
          and close["SPY"]["delta_shares"] == 62.5 and all(row["target_shares"] == 0 for row in close.values())
          and summary["traded_notional"] == 25000, "Empty final targets close all signed old shares")
    cash, summary = plan_net_trades({}, {}, {}, 1.0, tickers=tickers)
    check("flat_cash_requires_no_irrelevant_quotes", all(not row["quote_required"] for row in cash)
          and summary["traded_notional"] == 0, "Unheld zero-target assets do not require execution quotes")
    fraction, _ = plan_net_trades({t[0]: 1}, {}, {t[0]: 3.0, "SPY": 7.0}, 101.0, tickers=tickers)
    fraction = {row["ticker"]: row for row in fraction}
    check("fractional_shares_not_rounded", fraction[t[0]]["target_shares"] == 12.625 / 3
          and fraction["SPY"]["target_shares"] == -12.625 / 7, "Fractional shares preserve exact percentage targets")
    for name, pairs, shares, quote, nav in (
        ("missing_required_quote", {t[0]: 1}, {}, {t[0]: 100}, 100000),
        ("zero_required_quote", {t[0]: 1}, {}, prices | {t[0]: 0}, 100000),
        ("nonfinite_required_quote", {t[0]: 1}, {}, prices | {t[0]: float("nan")}, 100000),
        ("old_holding_also_requires_quote", {}, {t[0]: 1}, {}, 100000),
        ("unknown_old_asset", {}, {"UNKNOWN": 1}, prices, 100000),
        ("nonfinite_old_shares", {}, {t[0]: float("nan")}, prices, 100000),
        ("invalid_nav", {}, {}, prices, 0),
        ("too_many_target_pairs", dict.fromkeys(t[:5], 1), {}, prices, 100000),
        ("invalid_target_direction", {t[0]: 2}, {}, prices, 100000)):
        try:
            plan_net_trades(pairs, shares, quote, nav, tickers=tickers)
        except ValueError:
            check(name, True, "Invalid shares, execution quote or target configuration rejected")
        else:
            check(name, False, "Invalid input was not rejected")
    return pd.DataFrame(records), pd.DataFrame(worked_rows)
