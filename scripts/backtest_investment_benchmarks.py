"""Step 12: SPY fixed-share buy-and-hold and cash investment benchmarks.

Use the frozen phase dates, observed opens/dividends, exported cash intervals and
main execution cost. Original strategies, forecasts and registered outputs remain
read-only. This standalone entry point preserves the step-11 wrapper's code hash.
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
    from .backtest_signal_policy import PHASE_OPENS
    from .backtest_threshold_selection import (
        SHARPE_STD_ZERO_ATOL, read_frame, save_frame, sha256, verify_hashes, write_json,
    )
else:
    from backtest_signal_policy import PHASE_OPENS
    from backtest_threshold_selection import (
        SHARPE_STD_ZERO_ATOL, read_frame, save_frame, sha256, verify_hashes, write_json,
    )

BENCHMARKS = ("spy_buy_hold", "cash")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check(records, phase, benchmark, name, passed, detail):
    records.append(dict(phase=phase, benchmark=benchmark, check=name, passed=bool(passed), detail=detail))
    if not passed:
        raise AssertionError(f"{phase}/{benchmark}: {name}")


def register_design(root, out):
    phase_out = root / "results/backtest/phase_evaluation"
    previous = read_json(phase_out / "run_manifest.json")
    assert previous["protocol_frozen"] and previous["test_strategy_evaluated"]
    assert all(c["passed"] for c in previous["checks"])
    verify_hashes(root, previous["source_hashes"])
    verify_hashes(root, previous["code_hashes"])
    verify_hashes(root, {r["path"]: r["sha256"] for r in previous["outputs"]})
    protocol = read_json(phase_out / "protocol_config.json")
    assert protocol["phase_opens"] == {k: list(v) for k, v in PHASE_OPENS.items()}
    assert protocol["main_cost_parameters"]["execution_bps"] == 3.0
    policy = dict(schema_version=1, scope="Step 12: prescribed investment benchmarks only",
        authority="../5205proj2.md: sections 3.7 and 4; frozen step-11 protocol",
        phase_protocol_sha256=sha256(phase_out / "protocol_config.json"),
        phase_opens=protocol["phase_opens"], benchmarks=list(BENCHMARKS), initial_nav=1.0,
        phase_reset="each benchmark and phase independently starts from cash 1 and no shares; final open liquidates",
        execution_bps=3.0, timezone="America/New_York", execution="09:30 observed open",
        prices="SPY observed open and dividends; no adjusted prices or repeated split adjustment",
        spy=dict(
            entry="fixed shares = initial capital / ((1 + execution_bps/10000) * first observed open)",
            funding="entry principal plus entry fee consume initial capital; no financing beyond floating-point roundoff",
            holding="fixed fractional shares until final open; no daily/monthly rebalancing",
            dividends="old shares times current dividend enter cash; initial ex-date purchases get no dividend",
            reinvest_dividends=False, cash_interest="max(previous cash, 0) times exported same-interval RF return",
            interest_timing="new dividends start earning interest in the next interval, not the interval just ended",
            liquidation="sell all fixed shares at final observed open; credit old-share dividend, interest and subtract exit fee"),
        cash=dict(interest="cash_previous plus cash_previous times exported interval RF return",
            fees=0, security_trades=0, no_extra_day_count_division=True),
        accounting="NAV = cash + shares * observed open; price PnL, signed dividends, cash interest and fees counted once",
        borrow="no short positions or borrowing fees in either benchmark",
        statistics=dict(annualization=252, daily_return="current NAV_post / previous NAV_post - 1; first real interval uses initial capital 1",
            initial_open="no zero-length daily return; initial entry fee included in first real interval",
            annualized_net_return="ending NAV ** (252 / holding_intervals) - 1",
            cash_excess="net daily return minus same-interval exported RF return",
            sharpe="sqrt(252) * mean(cash excess) / sample std(ddof=1); N/A if std <= zero tolerance",
            sharpe_std_zero_atol=SHARPE_STD_ZERO_ATOL,
            turnover="two-sided actual trade notional/NAV_pre summed and annualized by 252/holding_intervals; no half factor"),
        comparison="use saved step-11 strategies; identical endpoints/intervals and capital; no strategy replay",
        interpretation="SPY is fully invested market exposure; paired strategies have different exposure; these are investable alternatives, not risk-matched portfolios",
        evidence_boundary=protocol["evidence_boundary"], model_retraining=False, threshold_search=False,
        strategy_replay=False, sensitivity_replay=False, steps_13_to_16_executed=False)
    path = out / "design_config.json"
    if path.exists():
        saved = read_json(path)
        assert {k: v for k, v in saved.items() if k != "registered_utc"} == policy, "Benchmark design changed"
        policy = saved
    else:
        policy["registered_utc"] = utc_now()
        write_json(path, policy)
    protected = dict(previous["source_hashes"])
    protected.update({r["path"]: r["sha256"] for r in previous["outputs"]})
    protected["results/backtest/phase_evaluation/run_manifest.json"] = sha256(phase_out / "run_manifest.json")
    engines = dict(previous["code_hashes"])
    engines["scripts/backtest_investment_benchmarks.py"] = sha256(Path(__file__))
    config = dict(design_sha256=sha256(path), source_hashes=protected, code_hashes=engines,
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__)
    config_path = out / "run_config.json"
    if config_path.exists():
        assert read_json(config_path) == config, "Benchmark sources/code/runtime changed; incompatible cache"
    else:
        write_json(config_path, config)
    return policy, config, sha256(config_path), previous


def load_inputs(root, phase, expected):
    first, last = PHASE_OPENS[phase]
    quotes = read_frame(root / "data/interim/etf_prices_aligned.csv", parse_dates=["date"],
        usecols=["date", "ticker", "open", "dividends", "observed"])
    quotes = quotes.loc[quotes.ticker.eq("SPY") & quotes.date.between(first, last)].sort_values("date").reset_index(drop=True)
    rates = read_frame(root / "data/interim/risk_free_open_intervals.csv", parse_dates=["entry_date", "exit_date"])
    rates = rates.loc[rates.entry_date.ge(first) & rates.exit_date.le(last)].sort_values("entry_date").reset_index(drop=True)
    opens = pd.DatetimeIndex(quotes.date)
    records = []
    check(records, phase, "both", "same_execution_opens", opens.is_unique
        and len(opens) == expected["opening_records_per_strategy"] and str(opens[0].date()) == first
        and str(opens[-1].date()) == last, "Exact same phase opens as the saved strategies")
    check(records, phase, "both", "observed_spy_prices_and_dividends", quotes.observed.eq(True).all()
        and np.isfinite(quotes[["open", "dividends"]]).all().all() and quotes.open.gt(0).all(),
        "Only observed raw-unit execution fields; no adjusted prices or forward filling")
    check(records, phase, "both", "adjacent_available_cash_intervals",
        list(zip(rates.entry_date, rates.exit_date)) == list(zip(opens[:-1], opens[1:]))
        and rates.interval_valid.eq(True).all() and rates.rate_known_at_entry.eq(True).all()
        and np.isfinite(rates.risk_free_return_act365).all(), "All exact RF intervals already available at entry")
    check(records, phase, "both", "natural_day_metadata", np.array_equal(rates.holding_calendar_days.to_numpy(),
        (rates.exit_date-rates.entry_date).dt.days.to_numpy()), "RF is an interval return, not an annual yield to divide again")
    return quotes, rates, records


def replay_benchmark(quotes, rates, *, phase, benchmark, initial_nav, execution_bps):
    """Long-only fixed-share or cash ledger; deliberately not the pair allocator."""
    if benchmark not in BENCHMARKS or not math.isfinite(initial_nav) or initial_nav <= 0:
        raise ValueError("Invalid investment benchmark or initial NAV")
    if not math.isfinite(execution_bps) or execution_bps < 0:
        raise ValueError("Invalid execution cost")
    quotes = quotes.sort_values("date")
    if len(quotes) < 2 or quotes.date.duplicated().any() or rates.duplicated(["entry_date", "exit_date"]).any():
        raise ValueError("Require unique phase opens and rate intervals")
    rate_rows = rates.set_index(["entry_date", "exit_date"])
    cost = execution_bps/10000
    fixed_shares = initial_nav/((1+cost)*float(quotes.open.iloc[0])) if benchmark == "spy_buy_hold" else 0.0
    cash, shares, previous_open, previous_date, previous_nav = initial_nav, 0.0, None, None, initial_nav
    rows = []
    for index, quote in enumerate(quotes.itertuples(index=False)):
        date, price, dividend = quote.date, float(quote.open), float(quote.dividends)
        if not math.isfinite(price) or price <= 0 or not math.isfinite(dividend):
            raise ValueError("Invalid observed open or dividend")
        initial, final = index == 0, index == len(quotes)-1
        days, rf = 0, 0.0
        if not initial:
            key = (previous_date, date)
            if key not in rate_rows.index:
                raise ValueError("Missing adjacent cash interval")
            rate = rate_rows.loc[key]
            days, rf = (date-previous_date).days, float(rate.risk_free_return_act365)
            if not bool(rate.interval_valid) or not bool(rate.rate_known_at_entry) or rate.holding_calendar_days != days or not math.isfinite(rf) or rf <= -1:
                raise ValueError("Invalid or unavailable cash interval")
        price_pnl = 0.0 if initial else shares*(price-previous_open)
        dividend_cash = shares*dividend
        interest_base = max(cash, 0.0)
        interest = interest_base*rf
        cash_pre = cash+dividend_cash+interest
        nav_pre = cash_pre+shares*price
        target = 0.0 if final else fixed_shares
        delta = target-shares
        notional = abs(delta)*price
        fee = notional*cost
        cash_post = cash_pre-delta*price-fee
        nav_post = cash_post+target*price
        denominator = float("nan") if initial else initial_nav if index == 1 else previous_nav
        ret = float("nan") if initial else nav_post/denominator-1
        pre_error = nav_pre-(previous_nav+price_pnl+dividend_cash+interest)
        post_error = nav_post-(nav_pre-fee)
        if not math.isclose(pre_error, 0.0, abs_tol=1e-12) or not math.isclose(post_error, 0.0, abs_tol=1e-12):
            raise AssertionError("Benchmark NAV identity failed")
        if cash_post < -1e-14 or nav_post <= 0:
            raise AssertionError("Investment benchmark cannot borrow or have nonpositive NAV")
        rows.append(dict(split=phase, benchmark=benchmark, open_date=date, interval_entry_date=previous_date,
            is_initial_open=initial, is_final_open=final, open_price=price, observed_dividend_per_share=dividend,
            holding_calendar_days=days, interval_risk_free_return=rf, cash_prev=cash, shares_prev=shares,
            nav_previous=previous_nav, price_pnl=price_pnl, dividend_cash=dividend_cash,
            cash_interest_base=interest_base, cash_interest=interest, borrow_fee=0.0, cash_pre=cash_pre, nav_pre=nav_pre,
            fixed_shares=fixed_shares, shares_post=target, delta_shares=delta, trade_notional=notional,
            execution_fee=fee, cash_post=cash_post, market_value_post=target*price, nav_post=nav_post,
            holding_return_denominator=denominator, holding_interval_return=ret,
            pre_reconciliation_error=pre_error, post_reconciliation_error=post_error))
        cash, shares, previous_open, previous_date, previous_nav = cash_post, target, price, date, nav_post
    return pd.DataFrame(rows)


def summarize_benchmarks(book):
    summaries, intervals = [], []
    for (phase, benchmark), group in book.groupby(["split", "benchmark"], sort=True):
        group = group.sort_values("open_date").reset_index(drop=True)
        interval = group.loc[~group.is_initial_open].copy()
        interval["cash_excess_return"] = interval.holding_interval_return-interval.interval_risk_free_return
        std = float(interval.cash_excess_return.std(ddof=1))
        sharpe = float(np.sqrt(252)*interval.cash_excess_return.mean()/std) if std > SHARPE_STD_ZERO_ATOL else float("nan")
        last = group.iloc[-1]
        pnl_nav = 1+group.price_pnl.sum()+group.dividend_cash.sum()+group.cash_interest.sum()-group.execution_fee.sum()
        turnover = float((group.trade_notional/group.nav_pre).sum())
        summaries.append(dict(split=phase, benchmark=benchmark, opening_records=len(group), holding_intervals=len(interval),
            first_open_date=group.open_date.iloc[0], last_open_date=last.open_date, ending_nav=float(last.nav_post),
            net_cumulative_return=float(last.nav_post-1), annualized_net_return=float(np.expm1(np.log(last.nav_post)*252/len(interval))),
            mean_daily_net_return=float(interval.holding_interval_return.mean()),
            mean_daily_cash_excess_return=float(interval.cash_excess_return.mean()),
            mean_daily_cash_excess_bps=float(interval.cash_excess_return.mean()*10000), daily_cash_excess_std=std,
            annualized_cash_excess_sharpe=sharpe, annualized_two_sided_turnover=turnover*252/len(interval),
            asset_trade_records=int(group.delta_shares.ne(0).sum()), total_execution_fee=float(group.execution_fee.sum()),
            total_borrow_fee=float(group.borrow_fee.sum()), total_dividend_cash=float(group.dividend_cash.sum()),
            total_cash_interest=float(group.cash_interest.sum()), accounting_residual=float(last.nav_post-pnl_nav)))
        intervals.append(interval)
    return pd.DataFrame(summaries), pd.concat(intervals, ignore_index=True)


def validate_logic():
    """Hand-computed fixture separates dividend, interest, fee and cash timing."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    quotes = pd.DataFrame(dict(date=dates, open=[100., 101., 102.], dividends=[10., 2., 3.]))
    rates = pd.DataFrame(dict(entry_date=dates[:-1], exit_date=dates[1:], holding_calendar_days=[1, 3],
        risk_free_return_act365=[0.001, 0.003], interval_valid=[True, True], rate_known_at_entry=[True, True]))
    records = []
    spy = replay_benchmark(quotes, rates, phase="fixture", benchmark="spy_buy_hold", initial_nav=1., execution_bps=3.)
    cash = replay_benchmark(quotes, rates, phase="fixture", benchmark="cash", initial_nav=1., execution_bps=3.)
    q = 1/(100*1.0003)
    expected = q*(102+2+3+2*0.003)-q*102*0.0003
    close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
    def test(name, passed, detail):
        check(records, "fixture", "both", name, passed, detail)
    test("fee_funded_fixed_shares", close(spy.fixed_shares.iloc[0], q) and close(spy.cash_post.iloc[0], 0),
         "Entry principal plus fee exactly consume capital without leverage")
    test("initial_ex_date_no_dividend", spy.dividend_cash.iloc[0] == 0, "A new buyer does not receive the initial-date dividend")
    test("old_shares_get_dividend_even_at_exit", close(spy.dividend_cash.iloc[-1], q*3), "Final-open old shares get their dividend before sale")
    test("dividends_not_reinvested", close(spy.shares_post.iloc[1], q) and spy.delta_shares.iloc[1] == 0,
         "No mid-phase security trade despite price/dividend changes")
    test("interest_starts_after_dividend_credit", close(spy.cash_interest.iloc[1], 0)
         and close(spy.cash_interest.iloc[2], q*2*0.003), "The weekend RF return is used once, on prior cash only")
    test("closed_form_spy_nav", close(spy.nav_post.iloc[-1], expected), "Independent hand calculation includes both fees and dividend interest")
    test("both_end_fees", close(spy.execution_fee.iloc[0], q*100*0.0003)
         and close(spy.execution_fee.iloc[-1], q*102*0.0003), "Both actual entry and exit notionals are charged")
    test("cash_compounds_interval_returns", close(cash.nav_post.iloc[-1], 1.001*1.003)
         and cash.execution_fee.eq(0).all(), "Cash has no security fee or repeated day-count conversion")
    summary, returns = summarize_benchmarks(pd.concat([spy, cash], ignore_index=True))
    test("first_real_interval_keeps_entry_fee", close(np.prod(1+spy.holding_interval_return.iloc[1:]), expected)
         and spy.holding_return_denominator.iloc[1] == 1 and pd.isna(spy.holding_interval_return.iloc[0]),
         "No initial zero-length return or omitted entry cost")
    test("cash_zero_excess_sharpe_na", pd.isna(summary.loc[summary.benchmark.eq("cash"), "annualized_cash_excess_sharpe"].iloc[0])
         and returns.loc[returns.benchmark.eq("cash"), "cash_excess_return"].abs().le(1e-12).all(),
         "Floating-point subtraction cannot create a spurious cash Sharpe")
    for name, broken in (("missing_rate_rejected", rates.iloc[:1]),
                         ("unavailable_rate_rejected", rates.assign(rate_known_at_entry=False))):
        try:
            replay_benchmark(quotes, broken, phase="fixture", benchmark="cash", initial_nav=1., execution_bps=3.)
        except ValueError:
            passed = True
        else:
            passed = False
        test(name, passed, "Missing/unavailable rate data must stop rather than delete a return interval")
    return records


def validate_results(book, summary, phase_summary, rates_by_phase, phase_manifest):
    records = []
    for (phase, benchmark), group in book.groupby(["split", "benchmark"], sort=True):
        group = group.sort_values("open_date").reset_index(drop=True)
        row = summary.loc[summary["split"].eq(phase) & summary.benchmark.eq(benchmark)].iloc[0]
        interval = group.iloc[1:]
        first, last = group.iloc[0], group.iloc[-1]
        expected = phase_manifest["phases"][phase]
        def test(name, passed, detail):
            check(records, phase, benchmark, name, passed, detail)
        test("same_phase_coverage", len(group) == expected["opening_records_per_strategy"]
             and len(interval) == expected["holding_intervals_per_strategy"], "Same open and interval counts as all six strategies")
        test("cash_one_empty_start_final_flat", first.cash_prev == 1 and first.shares_prev == 0
             and first.nav_previous == 1 and last.shares_post == 0, "Each benchmark/phase resets and liquidates independently")
        test("cash_nonnegative_no_borrow", group.cash_post.ge(-1e-14).all()
             and group.shares_post.ge(0).all() and group.borrow_fee.eq(0).all(), "No short exposure or cash financing")
        test("return_product_and_end_fees", np.isclose(np.prod(1+interval.holding_interval_return), last.nav_post, atol=1e-12, rtol=1e-12)
             and group.holding_return_denominator.iloc[1] == 1 and pd.isna(first.holding_interval_return),
             "All initial/final costs enter the reconstructed return product")
        test("nav_identities", group.pre_reconciliation_error.abs().le(1e-12).all()
             and group.post_reconciliation_error.abs().le(1e-12).all() and abs(row.accounting_residual) <= 1e-10,
             "Cash, price PnL, dividends, interest and fees reconcile")
        test("prior_cash_interest", np.allclose(group.cash_interest, np.maximum(group.cash_prev, 0)*group.interval_risk_free_return,
             rtol=1e-12, atol=1e-14), "New dividends are not credited a preceding interval's interest")
        test("old_share_dividends", np.allclose(group.dividend_cash, group.shares_prev*group.observed_dividend_per_share,
             rtol=1e-12, atol=1e-14), "Old shares earn dividends once, including final-open entitlement")
        rates = rates_by_phase[phase]
        test("exact_adjacent_rf_keys", list(zip(interval.interval_entry_date, interval.open_date)) == list(zip(rates.entry_date, rates.exit_date))
             and np.array_equal(interval.interval_risk_free_return.to_numpy(), rates.risk_free_return_act365.to_numpy()),
             "Exact same open-to-open cash return keys and values")
        test("actual_trade_fee", np.allclose(group.execution_fee, group.delta_shares.abs()*group.open_price*0.0003,
             rtol=1e-12, atol=1e-14), "Actual security trades pay the frozen single-sided 3 bps")
        if benchmark == "spy_buy_hold":
            test("fixed_shares_no_reinvestment", group.shares_post.iloc[:-1].eq(first.fixed_shares).all()
                 and group.delta_shares.iloc[1:-1].eq(0).all(), "Fixed SPY shares until final sale; all dividends remain cash")
            test("entry_fee_funded", np.isclose(first.fixed_shares*first.open_price*1.0003, 1, atol=1e-14, rtol=1e-12)
                 and abs(first.cash_post) <= 1e-14, "Initial capital funds the security and fee, without borrowing")
            test("two_security_trades", row.asset_trade_records == 2 and first.execution_fee > 0 and last.execution_fee > 0,
                 "Only initial purchase and final liquidation")
        else:
            test("cash_no_security_flows", group.shares_post.eq(0).all() and group.trade_notional.eq(0).all()
                 and group.execution_fee.eq(0).all() and group.dividend_cash.eq(0).all(), "Cash does not receive equity dividends or trading fees")
            test("cash_matches_closed_form", np.isclose(last.nav_post, np.prod(1+rates.risk_free_return_act365), atol=1e-12, rtol=1e-12),
                 "Cash NAV matches independent RF compounding")
            references = phase_summary.loc[phase_summary["split"].eq(phase), "cash_ending_nav"]
            test("cash_matches_existing_strategy_reference", np.allclose(references, last.nav_post, atol=1e-12, rtol=1e-12),
                 "Formal cash benchmark agrees with every saved strategy's same-interval cash reference")
            test("cash_sharpe_na", pd.isna(row.annualized_cash_excess_sharpe), "No risk-premium Sharpe is reported for numerical-zero cash excess")
    return records


def run_investment_benchmarks(project_root):
    root = Path(project_root).resolve()
    out = root / "results/backtest/investment_benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    policy, config, identity, phase_manifest = register_design(root, out)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["run_config_sha256"] == identity
        verify_hashes(root, {r["path"]: r["sha256"] for r in manifest["outputs"]})
        print("Step 12: verified completed investment benchmark cache; no replay", flush=True)
        return manifest
    print("Step 12: benchmark definitions, fee-funded shares and input/code hashes recorded before replay", flush=True)
    records = validate_logic()
    save_frame(out / "logic_checks.csv", pd.DataFrame(records))
    books, rates_by_phase = [], {}
    for phase in PHASE_OPENS:
        quotes, rates, input_checks = load_inputs(root, phase, phase_manifest["phases"][phase])
        records.extend(input_checks)
        rates_by_phase[phase] = rates
        for benchmark in BENCHMARKS:
            books.append(replay_benchmark(quotes, rates, phase=phase, benchmark=benchmark,
                initial_nav=policy["initial_nav"], execution_bps=policy["execution_bps"]))
    book = pd.concat(books, ignore_index=True)
    summary, returns = summarize_benchmarks(book)
    phase_summary = read_frame(root / "results/backtest/phase_evaluation/phase_summary.csv")
    records.extend(validate_results(book, summary, phase_summary, rates_by_phase, phase_manifest))
    comparison = phase_summary.copy()
    for benchmark in BENCHMARKS:
        fields = ["split", "ending_nav", "net_cumulative_return", "annualized_net_return"]
        reference = summary.loc[summary.benchmark.eq(benchmark), fields].rename(columns={f: f"{benchmark}_benchmark_{f}" for f in fields if f != "split"})
        comparison = comparison.merge(reference, on="split", validate="many_to_one")
        comparison[f"ending_relative_to_{benchmark}_benchmark"] = comparison.ending_nav/comparison[f"{benchmark}_benchmark_ending_nav"]-1
    check(records, "both", "comparison", "all_twelve_saved_strategies", len(comparison) == 12
          and np.allclose(comparison.cash_ending_nav, comparison.cash_benchmark_ending_nav, rtol=1e-12, atol=1e-14),
          "All existing model/scheme/phase results retained; no strategy replay or reselection")
    verify_hashes(root, config["source_hashes"])
    verify_hashes(root, config["code_hashes"])
    check(records, "both", "comparison", "protected_history_and_code_unchanged", True,
          "All original inputs, registered strategy results and the eight step-11 code files keep their hashes")
    trades = book.loc[book.delta_shares.ne(0)].copy()
    for name, frame in (("benchmark_daily.csv", book), ("benchmark_returns.csv", returns),
                        ("benchmark_trades.csv", trades), ("benchmark_summary.csv", summary),
                        ("strategy_benchmark_comparison.csv", comparison), ("benchmark_checks.csv", pd.DataFrame(records))):
        save_frame(out / name, frame)
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p))
               for p in sorted(out.iterdir()) if p.is_file() and p != manifest_path]
    manifest = dict(schema_version=1, scope="Step 12: SPY buy-and-hold and cash investment benchmarks",
        completed_utc=utc_now(), design_registered_utc=policy["registered_utc"], run_config_sha256=identity,
        python_version=config["python_version"], numpy_version=config["numpy_version"], pandas_version=config["pandas_version"],
        phases={p: dict(opening_records_per_benchmark=phase_manifest["phases"][p]["opening_records_per_strategy"],
            holding_intervals_per_benchmark=phase_manifest["phases"][p]["holding_intervals_per_strategy"]) for p in PHASE_OPENS},
        benchmark_count=2, phase_benchmark_count=4, daily_ledger_rows=len(book), return_rows=len(returns),
        security_trade_rows=len(trades), strategy_comparison_rows=len(comparison),
        original_strategy_replay=False, model_retraining=False, threshold_reselection=False,
        sensitivity_scenarios_evaluated=False, protected_file_count=len(config["source_hashes"]), checks=records,
        source_hashes=config["source_hashes"], code_hashes=config["code_hashes"], outputs=outputs,
        elapsed_seconds=time.perf_counter()-started,
        notes=["SPY initial shares reserve the entry fee within capital 1; dividends remain cash and earn future-interval interest",
            "Cash benchmark exactly uses the same available RF intervals, with no artificial trading fee",
            "Original phase/threshold manifests keep their historical pending flags; this manifest owns step-12 completion",
            "Investment alternatives have different risk exposures; a higher SPY return alone does not establish risk-adjusted dominance",
            "Earlier test exploration and validation selection boundaries remain unchanged"],
        not_yet_checked=["step 13: broader accounting and deterministic consistency audit",
            "step 14: full reporting, comparison figures and frozen cost/financing sensitivities",
            "step 15: final package and handoff", "step 16: optional extensions only if separately requested"])
    write_json(manifest_path, manifest)
    print(f"PASS: {len(records)} benchmark checks; four complete phase/benchmark ledgers; original strategies not replayed", flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-logic", action="store_true")
    args = parser.parse_args()
    if args.check_logic:
        print(pd.DataFrame(validate_logic()).to_string(index=False))
    else:
        run_investment_benchmarks(args.project_root)
