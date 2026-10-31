"""Step 13: independent ledger reconciliation and deterministic consistency audit.

Read the twelve selected strategy paths and four investment benchmark paths.
Do not refit models, replay complete portfolios, search thresholds, change costs
or overwrite registered results. Only isolated single-open/function checks call
the existing execution functions. All audit evidence has its own output folder.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

if __package__:
    from .backtest_execution import plan_net_trades
    from .backtest_ledger import ledger_step
    from .backtest_pair_decisions import decide_pairs
    from .backtest_signal_policy import build_trading_status, PHASE_OPENS, ASOF_PANEL_FIELDS, CALENDAR_FIELDS
    from .backtest_threshold_selection import TICKERS, MODELS, SCHEMES, read_frame, save_frame, sha256, verify_hashes, write_json
    from .backtest_investment_benchmarks import replay_benchmark
else:
    from backtest_execution import plan_net_trades
    from backtest_ledger import ledger_step
    from backtest_pair_decisions import decide_pairs
    from backtest_signal_policy import build_trading_status, PHASE_OPENS, ASOF_PANEL_FIELDS, CALENDAR_FIELDS
    from backtest_threshold_selection import TICKERS, MODELS, SCHEMES, read_frame, save_frame, sha256, verify_hashes, write_json
    from backtest_investment_benchmarks import replay_benchmark

ATOL, RTOL, TOTAL_ATOL = 1e-12, 1e-12, 1e-10
KEYS = ["model", "scheme", "open_date"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_dates(path):
    frame = read_frame(path)
    for field in ("date", "open_date", "interval_entry_date", "entry_date", "exit_date"):
        if field in frame:
            frame[field] = pd.to_datetime(frame[field])
    return frame


class Audit:
    def __init__(self):
        self.records = []

    def flag(self, category, phase, identity, name, passed, rows=1, detail=""):
        self.records.append(dict(category=category, phase=phase, identity=identity, check=name,
            passed=bool(passed), rows=int(rows), max_abs_error=None, atol=None, rtol=None, detail=detail))

    def close(self, category, phase, identity, name, actual, expected, atol=ATOL):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape:
            passed, error = False, None
        else:
            passed = np.isclose(a, b, atol=atol, rtol=RTOL, equal_nan=True).all()
            finite = np.isfinite(a) & np.isfinite(b)
            error = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
            passed = passed and not (np.isinf(a).any() or np.isinf(b).any())
        self.records.append(dict(category=category, phase=phase, identity=identity, check=name,
            passed=bool(passed), rows=int(a.size), max_abs_error=error, atol=atol, rtol=RTOL, detail=""))
        return error

    def frames(self, category, phase, identity, name, actual, expected, exact=True):
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=exact, atol=ATOL, rtol=RTOL)
        except AssertionError as exc:
            passed, detail = False, str(exc)[:700]
        else:
            passed, detail = True, "Exact value equality" if exact else "Within registered numeric tolerances"
        self.flag(category, phase, identity, name, passed, len(actual), detail)


def register_design(root, out):
    previous_path = root / "results/backtest/investment_benchmarks/run_manifest.json"
    previous = read_json(previous_path)
    assert all(c["passed"] for c in previous["checks"])
    verify_hashes(root, previous["source_hashes"])
    verify_hashes(root, previous["code_hashes"])
    verify_hashes(root, {x["path"]: x["sha256"] for x in previous["outputs"]})
    policy = dict(schema_version=1, scope="Step 13: twelve selected strategy paths and four investment benchmark paths",
        prerequisites="completed frozen step-11 phases and step-12 investment benchmarks; all recorded hashes intact",
        phase_opens={p: list(v) for p, v in PHASE_OPENS.items()}, entry_bps=15, exit_bps=7.5,
        accounting="independent arithmetic from source opens/dividends/RF and saved signed shares; not stored residual columns alone",
        state="initial cash 1/no shares; cash and shares carry exactly within each path; all final shares zero",
        rules="independent retain/exit/top-four reference; compare saved targets and existing decision function",
        eligibility="rebuild using original as-of panel/forecast/calendar fields; shuffle inputs and poison future scoring fields",
        determinism="all signal decisions under reversed dictionary order; selected single-open planner/ledger calls; identical-forecast paths; tiny benchmark order fixtures",
        case_selection="per strategy: first/last open, first held rebalance, reversal, net-zero SPY exposure, held dividend and short-collateral interval when present; no return-based selection",
        numeric_tolerance=dict(atol=ATOL, rtol=RTOL, phase_total_atol=TOTAL_ATOL,
            interpretation="independent floating-point reconstructions use tolerance; input-order checks require exact equality in this runtime"),
        summary_checks="NAV product, PnL sum, annualization, cash-excess Sharpe, double-sided turnover, start-of-interval holding fraction and saved benchmark joins",
        historical_candidates="all prior candidate outputs retain hashes; numeric audit focuses on the selected 15-bps formal paths",
        original_results_read_only=True, full_strategy_replay=False, full_benchmark_replay=False,
        model_retraining=False, threshold_reselection=False, cost_sensitivity=False, steps_14_to_16_executed=False)
    path = out / "design_config.json"
    if path.exists():
        saved = read_json(path)
        assert {k: v for k, v in saved.items() if k != "registered_utc"} == policy, "Audit design changed"
        policy = saved
    else:
        policy["registered_utc"] = utc_now()
        write_json(path, policy)
    protected = dict(previous["source_hashes"])
    protected.update({x["path"]: x["sha256"] for x in previous["outputs"]})
    protected[previous_path.relative_to(root).as_posix()] = sha256(previous_path)
    codes = dict(previous["code_hashes"])
    codes["scripts/backtest_consistency_audit.py"] = sha256(Path(__file__))
    config = dict(design_sha256=sha256(path), source_hashes=protected, code_hashes=codes,
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__)
    config_path = out / "run_config.json"
    if config_path.exists():
        assert read_json(config_path) == config, "Audit source/code/runtime changed; incompatible cache"
    else:
        write_json(config_path, config)
    return policy, config, sha256(config_path)


def audit_signal_inputs(root, audit):
    panel = read_frame(root / "data/processed/model_panel.csv", parse_dates=["date"], usecols=ASOF_PANEL_FIELDS)
    forecasts = pd.concat([read_frame(root / f"results/{m}/{m}_predictions.csv", parse_dates=["date"])
                          for m in MODELS], ignore_index=True)
    forecasts["fit_cutoff"] = pd.to_datetime(forecasts.fit_cutoff, utc=True).dt.tz_convert("America/New_York")
    calendar = read_frame(root / "results/backtest/signal_calendar.csv", parse_dates=["date", "entry_date"], usecols=CALENDAR_FIELDS)
    for field in ("signal_at", "entry_at"):
        calendar[field] = pd.to_datetime(calendar[field], utc=True).dt.tz_convert("America/New_York")
    kwargs = dict(models=MODELS, schemes=SCHEMES, tickers=TICKERS)
    signals, daily = build_trading_status(panel, forecasts, calendar, **kwargs)
    poisoned = panel.assign(common_scoreable=False, label_valid=False, label_within_split=False,
                            target_relative_return=999., label_available_at="2100-01-01")
    other_signals, other_daily = build_trading_status(poisoned.iloc[::-1], forecasts.iloc[::-1], calendar.iloc[::-1], **kwargs)
    audit.frames("signal_inputs", "both", "all", "future_fields_and_row_order_do_not_change_signals", other_signals, signals)
    audit.frames("signal_inputs", "both", "all", "future_fields_and_row_order_do_not_change_daily_actions", other_daily, daily)
    stored_signals = read_dates(root / "results/backtest/trading_signal_inputs.csv")
    stored_daily = read_dates(root / "results/backtest/daily_signal_status.csv")
    sf = ["date", "model", "scheme", "ticker", "split", "y_hat", "common_features_valid", "input_row_present",
          "prediction_present", "prediction_finite", "model_fit_available", "row_signal_ready", "signal_complete", "action", "force_flat", "reason"]
    df = ["date", "model", "scheme", "split", "sector_count", "ready_sectors", "signal_complete", "action", "force_flat", "reason"]
    audit.frames("signal_inputs", "both", "all", "rebuilt_sector_readiness_matches_registered_signals", signals[sf], stored_signals[sf])
    audit.frames("signal_inputs", "both", "all", "rebuilt_phase_guards_match_registered_actions", daily[df], stored_daily[df])
    for field in ("signal_at", "action_at"):
        audit.flag("signal_inputs", "both", "all", f"{field}_utc_matches", np.array_equal(
            pd.to_datetime(daily[field], utc=True).astype("int64"), pd.to_datetime(stored_daily[field], utc=True).astype("int64")), len(daily))
    return signals, daily


def reference_pairs(old, values, instruction):
    if instruction["force_flat"]:
        return {}
    if instruction["action"] == "NO_ORDER_OUTSIDE_PHASE":
        return dict(old)
    if not instruction["signal_complete"] or set(values) != set(TICKERS) or not all(math.isfinite(v) for v in values.values()):
        return {}
    retained = {t: d for t, d in old.items() if d*values[t] > 0.00075}
    ranked = sorted(TICKERS, key=lambda t: (-abs(values[t]), TICKERS.index(t)))[:4]
    entrants = [t for t in ranked if t not in retained and abs(values[t]) > 0.0015]
    return retained | {t: (1 if values[t] > 0 else -1) for t in entrants[:4-len(retained)]}


def audit_pair_rules(phase, book, assets, signals, daily, audit):
    lookup = {k: dict(zip(g.ticker, g.y_hat)) for k, g in signals.groupby(["date", "model", "scheme"], sort=False)}
    observed = {(m, s, d): dict(zip(g.ticker, g.target_weight)) for (m, s, d), g in assets.groupby(["model", "scheme", "date"], sort=False)}
    rows, reference_weights = [], []
    for (model, scheme), instructions in daily.loc[daily["split"].eq(phase)].groupby(["model", "scheme"], sort=True):
        old = {}; matches = []; decisions = []; permutations = []; unmutated = []
        for row in instructions.sort_values("date").to_dict("records"):
            key = (row["date"], model, scheme)
            values = lookup.get(key, {})
            target = reference_pairs(old, values, row)
            before = (dict(old), dict(values), dict(row))
            engine, _, _ = decide_pairs(old, values, row, h_in=0.0015, tickers=TICKERS)
            alternate, _, _ = decide_pairs(dict(reversed(list(old.items()))), dict(reversed(list(values.items()))),
                row | dict(common_scoreable=False, target_relative_return=-999., label_valid=False), h_in=0.0015, tickers=TICKERS)
            decisions.append(engine == target); permutations.append(alternate == engine)
            unmutated.append((old, values, row) == before)
            if row["action"] != "NO_ORDER_OUTSIDE_PHASE":
                weights = {t: 0.125*target.get(t, 0) for t in TICKERS}
                weights["SPY"] = -sum(weights.values())
                recorded = observed[(model, scheme, row["date"])]
                matches.append(recorded == weights)
                reference_weights.extend(dict(model=model, scheme=scheme, date=row["date"], ticker=t, reference_weight=w) for t, w in weights.items())
            old = target
        identity = f"{model}/{scheme}"
        for name, values in (("independent_targets_match_saved_weights", matches), ("engine_matches_independent_pair_reference", decisions),
                             ("all_decisions_exact_under_input_order_and_future_field_changes", permutations), ("decision_inputs_not_mutated", unmutated)):
            audit.flag("pair_rules", phase, identity, name, all(values), len(values))
        rows.append(dict(phase=phase, model=model, scheme=scheme, scheduled_opens=len(matches),
            independent_targets_match=all(matches), engine_reference_match=all(decisions),
            input_order_and_future_fields_match=all(permutations), inputs_unchanged=all(unmutated)))
    return pd.DataFrame(reference_weights), rows


def audit_strategy(phase, book, assets, trades, saved_returns, summary, quotes, rates, weights, audit):
    reports = []
    for (model, scheme), raw_book in book.groupby(["model", "scheme"], sort=True):
        identity = f"{model}/{scheme}"
        b = raw_book.sort_values("open_date").reset_index(drop=True)
        a = assets.loc[assets.model.eq(model) & assets.scheme.eq(scheme)].sort_values(["open_date", "ticker"]).reset_index(drop=True)
        saved = summary.loc[summary["split"].eq(phase) & summary.model.eq(model) & summary.scheme.eq(scheme)].iloc[0]
        audit.flag("strategy", phase, identity, "unique_keys_and_ten_assets_per_open", not b.open_date.duplicated().any()
            and not a.duplicated(["open_date", "ticker"]).any() and len(a) == len(b)*10, len(a))
        a = a.merge(quotes, left_on=["open_date", "ticker"], right_on=["quote_date", "ticker"], validate="many_to_one")
        a = a.merge(weights.loc[weights.model.eq(model) & weights.scheme.eq(scheme), ["date", "ticker", "reference_weight"]],
                    on=["date", "ticker"], validate="one_to_one")
        audit.flag("strategy", phase, identity, "complete_source_calendar_and_joins", len(a) == len(b)*10
            and list(b.open_date) == list(quotes.loc[quotes.ticker.eq("SPY"), "quote_date"]), len(b))
        a["expected_old_q"] = a.groupby("ticker", sort=False).target_shares.shift(fill_value=0)
        audit.close("strategy", phase, identity, "shares_carry_from_previous_targets", a.current_shares, a.expected_old_q)
        audit.close("strategy", phase, identity, "opens_match_source", a.open_price, a.raw_open)
        audit.close("strategy", phase, identity, "dividends_match_source", a.observed_dividend_per_share, a.raw_dividend)
        audit.close("strategy", phase, identity, "held_previous_opens_match_source", a.previous_open,
                    np.where(a.current_shares.ne(0), a.raw_previous_open, np.nan))
        audit.close("strategy", phase, identity, "weights_match_independent_signal_reference", a.target_weight, a.reference_weight, atol=0)
        a["price_pnl_calc"] = (a.current_shares*(a.raw_open-a.raw_previous_open)).fillna(0)
        a["dividend_calc"] = a.current_shares*a.raw_dividend
        a["old_market_now"] = a.current_shares*a.raw_open
        a["old_market_start"] = (a.current_shares*a.raw_previous_open).fillna(0)
        a["collateral_calc"] = np.maximum(-a.old_market_start, 0)
        a["market_post_calc"] = a.target_shares*a.raw_open
        a["delta_calc"] = a.target_shares-a.current_shares
        a["trade_calc"] = a.delta_calc.abs()*a.raw_open
        a["principal_calc"] = -a.delta_calc*a.raw_open
        a["fee_calc"] = a.trade_calc*0.0003
        for actual, expected in (("price_pnl_old", "price_pnl_calc"), ("dividend_cash_old", "dividend_calc"),
                                 ("market_value_post", "market_post_calc"), ("delta_shares", "delta_calc"),
                                 ("trade_notional", "trade_calc"), ("cash_change_before_costs", "principal_calc"), ("execution_fee", "fee_calc")):
            audit.close("strategy", phase, identity, f"asset_{actual}", a[actual], a[expected])
        audit.close("strategy", phase, identity, "asset_cash_after_fee", a.cash_change_after_execution_fee, a.principal_calc-a.fee_calc)
        post_nav = pd.Series(b.nav_post.to_numpy(), index=b.open_date)
        audit.close("strategy", phase, identity, "actual_post_fee_weights", a.actual_weight_post, a.market_post_calc/a.open_date.map(post_nav))
        pairs = a.loc[a.ticker.ne("SPY")].assign(active=lambda f: f.reference_weight.ne(0)).groupby("open_date").active.sum().reindex(b.open_date)
        gross = a.assign(gross=lambda f: f.reference_weight.abs()).groupby("open_date").gross.sum().reindex(b.open_date)
        audit.close("strategy", phase, identity, "pair_count_from_independent_weights", b.target_pairs.to_numpy(), pairs.to_numpy(), atol=0)
        audit.close("strategy", phase, identity, "gross_from_independent_netted_weights", b.target_gross_weight.to_numpy(), gross.to_numpy(), atol=0)
        totals = a.groupby("open_date", sort=True)[["price_pnl_calc", "dividend_calc", "old_market_now", "old_market_start",
            "collateral_calc", "market_post_calc", "trade_calc", "principal_calc", "fee_calc"]].sum().reindex(b.open_date).reset_index()
        rf = b[["interval_entry_date", "open_date"]].merge(rates, left_on=["interval_entry_date", "open_date"],
            right_on=["entry_date", "exit_date"], how="left", validate="many_to_one")
        rf_value = rf.risk_free_return_act365.fillna(0)
        day_count = (b.open_date-b.open_date.shift()).dt.days.fillna(0)
        interest = np.maximum(b.cash_prev-totals.collateral_calc, 0)*rf_value
        borrow = totals.collateral_calc*0.01*day_count/365
        cpre = b.cash_prev+totals.dividend_calc+interest-borrow
        npre = cpre+totals.old_market_now
        cpost = cpre+totals.principal_calc-totals.fee_calc
        npost = cpost+totals.market_post_calc
        cash_error = audit.close("strategy", phase, identity, "cash_carry", b.cash_prev, b.cash_post.shift(fill_value=1))
        for name, actual, expected in (
            ("nav_previous", b.nav_previous, b.cash_prev+totals.old_market_start),
            ("nav_carry", b.nav_previous, b.nav_post.shift(fill_value=1)),
            ("interval_rf", b.interval_risk_free_return, rf_value), ("calendar_days", b.holding_calendar_days, day_count),
            ("starting_short_collateral", b.short_collateral_start, totals.collateral_calc),
            ("price_pnl", b.price_pnl, totals.price_pnl_calc), ("dividends", b.dividend_cash, totals.dividend_calc),
            ("interest", b.cash_interest, interest), ("borrow", b.borrow_fee, borrow), ("cash_pre", b.cash_pre, cpre),
            ("nav_pre", b.nav_pre, npre), ("execution_fee", b.execution_fee, totals.fee_calc),
            ("traded_notional", b.traded_notional, totals.trade_calc), ("cash_post", b.cash_post, cpost), ("nav_post", b.nav_post, npost)):
            audit.close("strategy", phase, identity, name, actual, expected)
        nav_map = pd.Series(npre.to_numpy(), index=b.open_date)
        expected_target = a.reference_weight*a.open_date.map(nav_map)/a.raw_open
        share_error = audit.close("strategy", phase, identity, "targets_from_independently_settled_nav", a.target_shares, expected_target)
        denominator = b.nav_post.shift(); denominator.iloc[1] = 1.
        expected_returns = b.nav_post/denominator-1
        audit.close("strategy", phase, identity, "first_fee_included_denominator", b.holding_return_denominator, denominator)
        audit.close("strategy", phase, identity, "holding_returns", b.holding_interval_return, expected_returns)
        audit.flag("strategy", phase, identity, "available_all_rf_intervals", rf.interval_valid.iloc[1:].eq(True).all()
            and rf.rate_known_at_entry.iloc[1:].eq(True).all(), len(b)-1)
        audit.flag("strategy", phase, identity, "cash_one_empty_start_final_flat", b.cash_prev.iloc[0] == 1
            and a.loc[a.open_date.eq(b.open_date.iloc[0]), "current_shares"].eq(0).all()
            and a.loc[a.open_date.eq(b.open_date.iloc[-1]), "target_shares"].eq(0).all()
            and b.effective_action.iloc[-1] == "FINAL_LIQUIDATION", len(b))
        actual_trades = trades.loc[trades.model.eq(model) & trades.scheme.eq(scheme)].sort_values(["open_date", "ticker"]).reset_index(drop=True)
        trade_fields = ["open_date", "ticker", "current_shares", "target_shares", "delta_shares", "trade_notional", "execution_fee"]
        audit.frames("strategy", phase, identity, "trade_export_equals_nonzero_asset_changes", actual_trades[trade_fields],
                     a.loc[a.delta_shares.ne(0), trade_fields].reset_index(drop=True))
        rr = saved_returns.loc[saved_returns.model.eq(model) & saved_returns.scheme.eq(scheme)].sort_values("open_date").reset_index(drop=True)
        audit.close("strategy", phase, identity, "return_export", rr.holding_interval_return, expected_returns.iloc[1:])
        excess = expected_returns.iloc[1:].to_numpy()-rf_value.iloc[1:].to_numpy()
        audit.close("strategy", phase, identity, "cash_excess_export", rr.cash_excess_return, excess)
        held = b.target_pairs.shift().iloc[1:]
        std = float(np.std(excess, ddof=1))
        sharpe = float(np.sqrt(252)*np.mean(excess)/std) if std > 1e-12 else float("nan")
        net_from_pnl = 1+math.fsum(totals.price_pnl_calc)+math.fsum(totals.dividend_calc)+math.fsum(interest)-math.fsum(borrow)-math.fsum(totals.fee_calc)
        product = math.prod(1+x for x in expected_returns.iloc[1:])
        turn = math.fsum(totals.trade_calc/npre)*252/(len(b)-1)
        actual_summary = [saved.ending_nav, saved.mean_daily_cash_excess_return, saved.annualized_cash_excess_sharpe,
            saved.annualized_two_sided_turnover, saved.holding_interval_fraction, saved.annualized_net_return, saved.cash_ending_nav,
            saved.total_execution_fee, saved.total_borrow_fee, saved.total_cash_interest, saved.total_dividend_cash,
            saved.asset_trade_records, saved.opening_records, saved.holding_intervals, saved.net_cumulative_return,
            saved.mean_daily_net_return, saved.mean_daily_cash_return, saved.mean_pairs_held]
        expected_summary = [product, float(np.mean(excess)), sharpe, turn, float(held.gt(0).mean()),
            math.expm1(math.log(product)*252/(len(b)-1)), math.prod(1+x for x in rf_value.iloc[1:]),
            math.fsum(totals.fee_calc), math.fsum(borrow), math.fsum(interest), math.fsum(totals.dividend_calc),
            len(actual_trades), len(b), len(b)-1, product-1, float(expected_returns.iloc[1:].mean()),
            float(rf_value.iloc[1:].mean()), float(held.mean())]
        audit.close("strategy", phase, identity, "summary_metrics_recomputed", actual_summary, expected_summary)
        pnl_error = audit.close("strategy", phase, identity, "ending_nav_from_independent_pnl_sum", [b.nav_post.iloc[-1]], [net_from_pnl], atol=TOTAL_ATOL)
        reports.append(dict(phase=phase, model=model, scheme=scheme, opens=len(b), intervals=len(b)-1, asset_rows=len(a),
            trade_rows=len(actual_trades), saved_ending_nav=float(b.nav_post.iloc[-1]), reconstructed_nav_product=product,
            reconstructed_nav_pnl=net_from_pnl, cash_carry_max_error=cash_error, target_shares_max_error=share_error,
            independent_nav_max_error=float(np.max(np.abs(b.nav_post-npost))), pnl_ending_error=pnl_error))
    return reports


def audit_single_opens(phase, book, assets, quotes, audit):
    cases = []
    for (model, scheme), raw in book.groupby(["model", "scheme"], sort=True):
        b = raw.sort_values("open_date").reset_index(drop=True)
        a = assets.loc[assets.model.eq(model) & assets.scheme.eq(scheme)]
        weights = a.pivot(index="open_date", columns="ticker", values="target_weight").reindex(b.open_date)
        same = weights.eq(weights.shift()).all(axis=1).to_numpy()
        reversal_dates = set(a.loc[(a.current_shares*a.target_shares).lt(0), "open_date"])
        dividend_dates = set(a.loc[a.dividend_cash_old.ne(0), "open_date"])
        spy_zero = set(a.loc[a.ticker.eq("SPY") & a.target_weight.eq(0), "open_date"])
        conditions = {
            "initial": np.arange(len(b)) == 0, "final": np.arange(len(b)) == len(b)-1,
            "retained_rebalance": same & b.traded_notional.gt(0).to_numpy() & b.target_pairs.gt(0).to_numpy(),
            "reversal": b.open_date.isin(reversal_dates).to_numpy(),
            "spy_net_zero": b.open_date.isin(spy_zero).to_numpy() & b.target_pairs.gt(0).to_numpy(),
            "held_dividend": b.open_date.isin(dividend_dates).to_numpy(),
            "borrow_interval": (b.short_collateral_start.gt(0) & b.holding_calendar_days.gt(0)).to_numpy(),
        }
        selected = {}
        for reason, mask in conditions.items():
            indices = np.flatnonzero(mask)
            if len(indices):
                selected.setdefault(int(indices[0]), []).append(reason)
        for index, reasons in sorted(selected.items()):
            row = b.iloc[index]
            one = a.loc[a.open_date.eq(row.open_date)].merge(quotes,
                left_on=["open_date", "ticker"], right_on=["quote_date", "ticker"], validate="one_to_one")
            old = {x.ticker: float(x.current_shares) for x in one.itertuples() if x.current_shares != 0}
            target = {x.ticker: (1 if x.target_weight > 0 else -1) for x in one.itertuples() if x.ticker != "SPY" and x.target_weight != 0}
            opens = dict(zip(one.ticker, one.raw_open)); dividends = dict(zip(one.ticker, one.raw_dividend))
            previous = {x.ticker: float(x.raw_previous_open) for x in one.itertuples() if x.current_shares != 0}
            reverse = lambda value: dict(reversed(list(value.items())))
            plans, _ = plan_net_trades(target, old, opens, float(row.nav_pre), tickers=TICKERS)
            other_plans, _ = plan_net_trades(reverse(target), reverse(old), reverse(opens), float(row.nav_pre), tickers=TICKERS)
            case_id = f"{model}/{scheme}/{row.open_date.date()}"
            audit.frames("determinism", phase, case_id, "planner_exact_under_dictionary_order", pd.DataFrame(plans), pd.DataFrame(other_plans))
            expected_plan = one.set_index("ticker").loc[[x["ticker"] for x in plans]].reset_index()[list(plans[0])]
            audit.frames("determinism", phase, case_id, "planner_matches_saved_asset_plan", pd.DataFrame(plans), expected_plan, exact=False)
            kwargs = dict(risk_free_return=float(row.interval_risk_free_return), calendar_days=int(row.holding_calendar_days),
                execution_bps=3., borrow_annual_rate=.01, full_cash_rebate=False, tickers=TICKERS)
            state, rows, result = ledger_step(float(row.cash_prev), old, previous, opens, dividends, target, **kwargs)
            state2, rows2, result2 = ledger_step(float(row.cash_prev), reverse(old), reverse(previous), reverse(opens), reverse(dividends), reverse(target), **kwargs)
            audit.flag("determinism", phase, case_id, "ledger_state_exact_under_dictionary_order", state == state2 and result == result2)
            audit.frames("determinism", phase, case_id, "ledger_asset_rows_exact_under_dictionary_order", pd.DataFrame(rows), pd.DataFrame(rows2))
            error = audit.close("determinism", phase, case_id, "isolated_settlement_matches_saved_open", list(result.values()), [row[k] for k in result])
            cases.append(dict(phase=phase, model=model, scheme=scheme, open_date=row.open_date,
                selection_reasons=";".join(reasons), target_pairs=len(target), max_settlement_error=error))
    return cases


def audit_identical_predictions(phase, signals, book, assets, audit):
    groups = {}
    for (m, s), g in signals.loc[signals["split"].eq(phase)].groupby(["model", "scheme"]):
        groups[(m, s)] = g.sort_values(["date", "ticker"])[["date", "ticker", "y_hat", "action"]].reset_index(drop=True)
    identities = list(groups)
    pairs = []
    for i, left in enumerate(identities):
        for right in identities[i+1:]:
            if not groups[left].equals(groups[right]):
                continue
            pair = f"{left[0]}/{left[1]} = {right[0]}/{right[1]}"
            for name, frame, order in (("daily", book, ["open_date"]), ("assets", assets, ["open_date", "ticker"])):
                columns = [x for x in frame if x not in ("model", "scheme", "selection_role", "evaluation_role")]
                one = frame.loc[frame.model.eq(left[0]) & frame.scheme.eq(left[1])].sort_values(order)[columns].reset_index(drop=True)
                two = frame.loc[frame.model.eq(right[0]) & frame.scheme.eq(right[1])].sort_values(order)[columns].reset_index(drop=True)
                audit.frames("determinism", phase, pair, f"identical_predictions_give_exact_{name}", one, two)
            pairs.append(dict(phase=phase, left=f"{left[0]}/{left[1]}", right=f"{right[0]}/{right[1]}", matched=True))
    return pairs


def audit_benchmarks(root, book, quotes_by_phase, rates_by_phase, audit):
    folder = root / "results/backtest/investment_benchmarks"
    saved_summary = read_frame(folder / "benchmark_summary.csv")
    saved_returns = read_dates(folder / "benchmark_returns.csv")
    saved_trades = read_dates(folder / "benchmark_trades.csv")
    reports = []
    for (phase, benchmark), raw in book.groupby(["split", "benchmark"], sort=True):
        b = raw.sort_values("open_date").reset_index(drop=True)
        quotes = quotes_by_phase[phase].loc[lambda f: f.ticker.eq("SPY")].sort_values("quote_date")
        rates = rates_by_phase[phase].sort_values("entry_date")
        price, div = quotes.raw_open.to_numpy(), quotes.raw_dividend.to_numpy()
        rf = np.r_[0., rates.risk_free_return_act365.to_numpy()]
        n = len(b)-1
        q = 1/(price[0]*1.0003) if benchmark == "spy_buy_hold" else 0.
        targets = np.r_[np.repeat(q, n), 0.]
        old = np.r_[0., targets[:-1]]
        delta = targets-old
        fees = np.abs(delta)*price*.0003
        dividend = old*div
        pnl = np.r_[0., old[1:]*np.diff(price)]
        interest = np.maximum(b.cash_prev.to_numpy(), 0)*rf
        cpre = b.cash_prev.to_numpy()+dividend+interest
        cpost = cpre-delta*price-fees
        npre = cpre+old*price
        npost = cpost+targets*price
        for name, actual, expected in (
            ("source_open", b.open_price, price), ("source_dividend", b.observed_dividend_per_share, div),
            ("fixed_share_targets", b.shares_post, targets), ("old_shares", b.shares_prev, old),
            ("cash_carry", b.cash_prev, b.cash_post.shift(fill_value=1)), ("nav_carry", b.nav_previous, b.nav_post.shift(fill_value=1)),
            ("cash_interest", b.cash_interest, interest), ("dividends", b.dividend_cash, dividend), ("price_pnl", b.price_pnl, pnl),
            ("cash_pre", b.cash_pre, cpre), ("nav_pre", b.nav_pre, npre), ("delta_shares", b.delta_shares, delta),
            ("execution_fee", b.execution_fee, fees), ("cash_post", b.cash_post, cpost), ("nav_post", b.nav_post, npost),
            ("same_interval_rf", b.interval_risk_free_return, rf)):
            audit.close("benchmark", phase, benchmark, name, actual, expected)
        denominator = b.nav_post.shift(); denominator.iloc[1] = 1.
        expected_return = b.nav_post/denominator-1
        audit.close("benchmark", phase, benchmark, "fee_included_returns", b.holding_interval_return, expected_return)
        rr = saved_returns.loc[saved_returns["split"].eq(phase) & saved_returns.benchmark.eq(benchmark)].sort_values("open_date").reset_index(drop=True)
        audit.close("benchmark", phase, benchmark, "return_export", rr.holding_interval_return, expected_return.iloc[1:])
        excess = expected_return.iloc[1:].to_numpy()-rf[1:]
        audit.close("benchmark", phase, benchmark, "cash_excess_export", rr.cash_excess_return, excess)
        std = float(np.std(excess, ddof=1))
        sharpe = float(np.sqrt(252)*np.mean(excess)/std) if std > 1e-12 else float("nan")
        terminal_dividend, growth = 0., 1.
        for i in range(n, 0, -1):
            terminal_dividend += q*div[i]*growth
            growth *= 1+rf[i]
        terminal = q*price[-1]*.9997+terminal_dividend if benchmark == "spy_buy_hold" else math.prod(1+x for x in rf[1:])
        pnl_nav = 1+math.fsum(pnl)+math.fsum(dividend)+math.fsum(interest)-math.fsum(fees)
        saved = saved_summary.loc[saved_summary["split"].eq(phase) & saved_summary.benchmark.eq(benchmark)].iloc[0]
        audit.close("benchmark", phase, benchmark, "terminal_value_independent_formula", [b.nav_post.iloc[-1]], [terminal])
        audit.close("benchmark", phase, benchmark, "pnl_sum_reconstructs_nav", [b.nav_post.iloc[-1]], [pnl_nav], atol=TOTAL_ATOL)
        audit.close("benchmark", phase, benchmark, "summary_metrics", [saved.ending_nav, saved.annualized_net_return,
            saved.annualized_cash_excess_sharpe, saved.annualized_two_sided_turnover, saved.total_execution_fee],
            [terminal, math.expm1(math.log(terminal)*252/n), sharpe, math.fsum(np.abs(delta)*price/npre)*252/n, math.fsum(fees)])
        actual_trades = saved_trades.loc[saved_trades["split"].eq(phase) & saved_trades.benchmark.eq(benchmark)].reset_index(drop=True)
        audit.frames("benchmark", phase, benchmark, "trade_export", actual_trades,
                     b.loc[b.delta_shares.ne(0)].reset_index(drop=True))
        audit.flag("benchmark", phase, benchmark, "same_source_calendar_initial_cash_final_flat", list(b.open_date) == list(quotes.quote_date)
            and b.cash_prev.iloc[0] == 1 and b.shares_prev.iloc[0] == 0 and b.shares_post.iloc[-1] == 0, len(b))
        reports.append(dict(phase=phase, benchmark=benchmark, opens=len(b), intervals=n, saved_ending_nav=float(b.nav_post.iloc[-1]),
            independent_terminal_nav=terminal, independent_pnl_nav=pnl_nav, max_nav_error=float(np.max(np.abs(b.nav_post-npost))),
            terminal_error=abs(float(b.nav_post.iloc[-1])-terminal)))
    # A three-open fixture tests input-order determinism, not a full market replay.
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    quotes = pd.DataFrame(dict(date=dates, open=[100., 101., 102.], dividends=[10., 2., 3.]))
    rates = pd.DataFrame(dict(entry_date=dates[:-1], exit_date=dates[1:], holding_calendar_days=[1, 3],
        risk_free_return_act365=[.001, .003], interval_valid=[True, True], rate_known_at_entry=[True, True]))
    for benchmark in ("spy_buy_hold", "cash"):
        a = replay_benchmark(quotes, rates, phase="fixture", benchmark=benchmark, initial_nav=1., execution_bps=3.)
        b = replay_benchmark(quotes.iloc[::-1], rates.iloc[::-1], phase="fixture", benchmark=benchmark, initial_nav=1., execution_bps=3.)
        audit.frames("determinism", "fixture", benchmark, "three_open_benchmark_exact_under_input_row_order", a, b)
    return reports


def audit_comparison(root, summary, benchmarks, audit):
    comparison = read_frame(root / "results/backtest/investment_benchmarks/strategy_benchmark_comparison.csv")
    keys = ["split", "model", "scheme"]
    audit.frames("comparison", "both", "all", "original_twelve_strategy_rows_unchanged",
        comparison.sort_values(keys)[list(summary)].reset_index(drop=True), summary.sort_values(keys).reset_index(drop=True))
    for name in ("spy_buy_hold", "cash"):
        reference = benchmarks.loc[benchmarks.benchmark.eq(name)].set_index("split")
        for field in ("ending_nav", "net_cumulative_return", "annualized_net_return"):
            audit.close("comparison", "both", name, f"{field}_join", comparison[f"{name}_benchmark_{field}"], comparison["split"].map(reference[field]))
        audit.close("comparison", "both", name, "terminal_wealth_ratio", comparison[f"ending_relative_to_{name}_benchmark"],
            comparison.ending_nav/comparison["split"].map(reference.ending_nav)-1)
    cash = benchmarks.loc[benchmarks.benchmark.eq("cash")].set_index("split").ending_nav
    audit.close("comparison", "both", "cash", "all_strategy_cash_references_agree", summary.cash_ending_nav, summary["split"].map(cash))


def run_consistency_audit(project_root):
    root = Path(project_root).resolve()
    out = root / "results/backtest/consistency_audit"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    policy, config, identity = register_design(root, out)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["run_config_sha256"] == identity and manifest["status"] == "passed"
        verify_hashes(root, {o["path"]: o["sha256"] for o in manifest["outputs"]})
        print("Step 13: verified completed independent audit cache; no repeat audit or portfolio replay", flush=True)
        return manifest
    audit = Audit()
    signals, daily = audit_signal_inputs(root, audit)
    print("Step 13: reconstructed as-of eligibility; shuffled/poisoned-field checks recorded", flush=True)
    summary = read_frame(root / "results/backtest/phase_evaluation/phase_summary.csv")
    source_prices = read_frame(root / "data/interim/etf_prices_aligned.csv", parse_dates=["date"],
        usecols=["date", "ticker", "open", "dividends", "observed"])
    source_rates = read_frame(root / "data/interim/risk_free_open_intervals.csv", parse_dates=["entry_date", "exit_date"])
    strategy_reports, signal_reports, cases, identical_pairs = [], [], [], []
    quotes_by_phase, rates_by_phase, coverage = {}, {}, []
    for phase, (first, last) in PHASE_OPENS.items():
        prefix = "threshold_selection/checkpoints/entry_15_" if phase == "validation" else "phase_evaluation/test_"
        frames = {name: read_dates(root / f"results/backtest/{prefix}{name}.csv") for name in ("daily", "assets", "trades", "returns")}
        quotes = source_prices.loc[source_prices.date.between(first, last) & source_prices.ticker.isin([*TICKERS, "SPY"])].sort_values(["ticker", "date"]).copy()
        quotes["raw_previous_open"] = quotes.groupby("ticker").open.shift()
        quotes = quotes.rename(columns=dict(date="quote_date", open="raw_open", dividends="raw_dividend"))
        rates = source_rates.loc[source_rates.entry_date.ge(first) & source_rates.exit_date.le(last)].copy()
        quotes_by_phase[phase], rates_by_phase[phase] = quotes, rates
        audit.flag("source_data", phase, "all", "all_execution_quotes_observed", quotes.observed.eq(True).all(), len(quotes))
        weights, rule_rows = audit_pair_rules(phase, frames["daily"], frames["assets"], signals, daily, audit)
        signal_reports.extend(rule_rows)
        strategy_reports.extend(audit_strategy(phase, frames["daily"], frames["assets"], frames["trades"], frames["returns"], summary, quotes, rates, weights, audit))
        cases.extend(audit_single_opens(phase, frames["daily"], frames["assets"], quotes, audit))
        identical_pairs.extend(audit_identical_predictions(phase, signals, frames["daily"], frames["assets"], audit))
        coverage.append(dict(phase=phase, kind="strategies", paths=6, daily_rows=len(frames["daily"]), asset_rows=len(frames["assets"]),
            trade_rows=len(frames["trades"]), return_rows=len(frames["returns"]), source_prefix=prefix))
        print(f"Step 13: {phase} six strategy paths reconciled; no full portfolio replay", flush=True)
    benchmark_book = read_dates(root / "results/backtest/investment_benchmarks/benchmark_daily.csv")
    benchmark_reports = audit_benchmarks(root, benchmark_book, quotes_by_phase, rates_by_phase, audit)
    benchmark_summary = read_frame(root / "results/backtest/investment_benchmarks/benchmark_summary.csv")
    audit_comparison(root, summary, benchmark_summary, audit)
    for phase, group in benchmark_book.groupby("split"):
        coverage.append(dict(phase=phase, kind="benchmarks", paths=2, daily_rows=len(group), asset_rows=0,
            trade_rows=int(group.delta_shares.ne(0).sum()), return_rows=int((~group.is_initial_open).sum()), source_prefix="investment_benchmarks/benchmark_"))
    verify_hashes(root, config["source_hashes"])
    verify_hashes(root, config["code_hashes"])
    audit.flag("provenance", "both", "all", "all_registered_history_and_code_unchanged", True, len(config["source_hashes"]),
        "Original models, predictions, all candidate outputs, accepted ledgers and the nine existing code files retain their hashes")
    checks = pd.DataFrame(audit.records)
    for name, frame in (("audit_checks.csv", checks), ("strategy_reconciliation.csv", pd.DataFrame(strategy_reports)),
        ("benchmark_reconciliation.csv", pd.DataFrame(benchmark_reports)), ("signal_rule_checks.csv", pd.DataFrame(signal_reports)),
        ("determinism_cases.csv", pd.DataFrame(cases)), ("identical_prediction_pairs.csv", pd.DataFrame(identical_pairs)),
        ("determinism_checks.csv", checks.loc[checks.category.eq("determinism")]), ("coverage_inventory.csv", pd.DataFrame(coverage))):
        save_frame(out / name, frame)
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p)) for p in sorted(out.iterdir()) if p.is_file() and p != manifest_path]
    passed = bool(checks.passed.all())
    manifest = dict(schema_version=1, scope=policy["scope"], status="passed" if passed else "failed", completed_utc=utc_now(),
        design_registered_utc=policy["registered_utc"], run_config_sha256=identity,
        python_version=config["python_version"], numpy_version=config["numpy_version"], pandas_version=config["pandas_version"],
        strategy_paths=12, benchmark_paths=4, isolated_single_open_cases=len(cases), identical_prediction_path_pairs=len(identical_pairs),
        checks=audit.records, check_count=len(checks), failed_check_count=int((~checks.passed).sum()),
        protected_file_count=len(config["source_hashes"]), source_hashes=config["source_hashes"], code_hashes=config["code_hashes"],
        outputs=outputs, elapsed_seconds=time.perf_counter()-started, full_strategy_replay=False, full_benchmark_replay=False,
        model_retraining=False, threshold_reselection=False, cost_sensitivity=False,
        notes=["Reconciliation independently uses source prices/dividends/RF and saved state, not merely saved residual columns",
            "Only isolated single-open function calls and three-open synthetic benchmark fixtures are executed; no full market portfolio is replayed",
            "Exact determinism is checked only in the recorded local software environment; independent financial arithmetic uses registered tolerances",
            "Numerical audit covers selected formal paths; earlier threshold candidates are protected by hashes without recomputing their experiments",
            "Earlier manifests retain historical pending flags; validation selection and previously viewed test evidence boundaries remain unchanged"],
        not_yet_checked=["step 14: full reporting, comparison figures and frozen cost/financing sensitivities",
            "step 15: final package and handoff", "step 16: optional extensions only if separately requested"])
    write_json(manifest_path, manifest)
    if not passed:
        raise AssertionError(f"Audit failed: {manifest['failed_check_count']} checks; see audit_checks.csv")
    print(f"PASS: {len(checks)} independent audit checks; 12 strategy + 4 benchmark paths; {len(cases)} single-open cases", flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_consistency_audit(args.project_root)
