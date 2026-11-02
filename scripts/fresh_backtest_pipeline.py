"""Fresh notebook 05/06 runs using current notebook 01-04 exports.

Only pure financial engines are reused. Historical backtest manifests, ledgers,
metrics and images are never inputs. Every 05 run gets an exclusive directory;
06 verifies that directory and recomputes every overlay, benchmark and figure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np
import pandas as pd

from backtest_baselines import build_baseline_tables
from backtest_costs import main_cost_parameters, scenario_records
from backtest_execution import validate_net_trades
from backtest_investment_benchmarks import replay_benchmark, validate_logic
from backtest_ledger import replay_ledger, validate_ledger
from backtest_pair_decisions import validate_pair_decisions
from backtest_reporting import path_statistics, strategy_statistics
from backtest_signal_policy import (
    PHASE_OPENS, FORECAST_FIELDS, build_trading_status,
    build_scoring_eligibility, validate_signal_policy,
)
from backtest_spy_overlay import replay_overlay, validate_run, summary_row
from backtest_spy_overlay_test import replay_spy_reinvest
from backtest_threshold_selection import (
    MODELS, SCHEMES, TICKERS, STRATEGIES, summarize_candidate,
    rank_candidates, validate_candidate, validate_selection_logic,
)

NY = "America/New_York"
ENTRY_CANDIDATES = (5, 10, 15, 20, 30)
OVERLAY_ENTRIES = (10, 15)
VERSION = "fresh-notebook-backtests-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "compresslevel": 1} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g", compression=compression)


def read(path, **kwargs):
    return pd.read_csv(path, low_memory=False, float_precision="round_trip", **kwargs)


def runtime():
    import sklearn
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)


def code_hashes(root):
    # Hash executable modules, never notebook output bytes/autosave metadata.
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / "scripts").glob("*.py"))}


def input_hashes(root):
    spec = read_json(root / "data/processed/panel_spec.json")
    files = ["data/processed/panel_spec.json", spec["shared_panel"],
             "data/interim/etf_prices_aligned.csv", "data/interim/risk_free_open_intervals.csv",
             "notebooks/panel_checks/monthly_training_eligibility.csv"]
    for model in MODELS:
        files.extend(f"results/{model}/{name}" for name in (
            f"{model}_predictions.csv", f"{model}_predictions_full.csv", "forecast_metrics.csv"))
    missing = [rel for rel in files if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError("Finish notebooks 01-04 before 05. Missing: " + ", ".join(missing))
    return {rel: sha(root / rel) for rel in files}


def verify_hashes(root, hashes):
    for rel, expected in hashes.items():
        path = root / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Input/code/result changed: {rel}. Finish 01-04, then start a NEW 05 run; do not mix runs.")


def protocol():
    return dict(version=VERSION, phase_opens=PHASE_OPENS, models=list(MODELS), schemes=list(SCHEMES),
        ticker_tie_order=list(TICKERS), initial_nav=1.0, max_pairs=4, sector_leg_weight=0.125,
        entry_candidates_bps=list(ENTRY_CANDIDATES), exit_fraction=0.5,
        threshold_selection="Validation only; equal mean of six strategies' daily net minus cash returns; exact ties prefer higher entry",
        main_costs=main_cost_parameters(), cost_scenarios=scenario_records(),
        overlay_entries_bps=list(OVERLAY_ENTRIES), overlay_net_weight=1.0, overlay_max_gross=1.5,
        overlay_borrow_annual_rate=0.01, overlay_funding_spread=0.01,
        overlay_selection="Highest TEST net CAGR among 12 combinations; exploratory ex-post selection, not independent holdout evidence",
        training_oof="Separate historical research diagnostic; not part of this portfolio reproduction. Candidate grid is registered directly before replay.",
        historical_backtest_files_used=False, cache_reuse=False)


def start_pair_run(root):
    root = Path(root).resolve()
    sources, codes = input_hashes(root), code_hashes(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out = root / "results/reproduction" / run_id
    out.mkdir(parents=True, exist_ok=False)
    config = dict(run_id=run_id, version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=runtime(), input_hashes=sources, code_hashes=codes, protocol=protocol())
    write_json(out / "run_config.json", config)
    print("NEW run:", out, "\nNo historical backtest results or caches will be loaded.", flush=True)
    return dict(root=root, out=out, config=config)


def prepare_current_inputs(ctx):
    root, out = ctx["root"], ctx["out"]
    verify_hashes(root, ctx["config"]["input_hashes"])
    spec = read_json(root / "data/processed/panel_spec.json")
    panel = read(root / spec["shared_panel"], parse_dates=["date", "entry_date", "exit_date"])
    for field in ("signal_at", "entry_at", "label_available_at"):
        panel[field] = pd.to_datetime(panel[field], utc=True).dt.tz_convert(NY)
    if panel.duplicated(["date", "ticker"]).any() or set(panel.ticker) != set(TICKERS):
        raise ValueError("Invalid panel keys or sector universe")
    for field in ("common_features_valid", "common_scoreable", "label_valid", "label_within_split"):
        if not pd.api.types.is_bool_dtype(panel[field]):
            raise ValueError(f"Expected boolean panel field: {field}")
    prices = read(root / "data/interim/etf_prices_aligned.csv", parse_dates=["date"])
    rates = read(root / "data/interim/risk_free_open_intervals.csv", parse_dates=["entry_date", "exit_date"])
    dates = pd.DatetimeIndex(prices.loc[prices.ticker.eq("SPY"), "date"].sort_values())
    if not dates.is_unique or not dates.equals(pd.DatetimeIndex(sorted(panel.date.unique()))):
        raise ValueError("Panel and SPY trading calendars differ")
    calendar = pd.DataFrame(dict(date=dates))
    calendar["entry_date"] = calendar.date.shift(-1)
    calendar["exit_date"] = calendar.date.shift(-2)
    calendar["signal_at"] = (calendar.date + pd.Timedelta(hours=18)).dt.tz_localize(NY)
    calendar["entry_at"] = (calendar.entry_date + pd.Timedelta(hours=9, minutes=30)).dt.tz_localize(NY)
    calendar["label_available_at"] = (calendar.exit_date + pd.Timedelta(hours=9, minutes=30)).dt.tz_localize(NY)
    calendar["split"] = "warmup"
    for start, end, phase in (("2008-01-01", "2017-12-31", "train"),
                              ("2018-01-01", "2022-12-31", "validation"),
                              ("2023-01-01", spec["data_cutoff"], "test")):
        calendar.loc[calendar.date.between(start, end), "split"] = phase
    mapping = calendar.set_index("date")
    for field in ("split", "entry_date", "exit_date", "signal_at", "entry_at", "label_available_at"):
        expected = panel.date.map(mapping[field])
        if not (panel[field].eq(expected) | (panel[field].isna() & expected.isna())).all():
            raise ValueError("Panel calendar mismatch: " + field)
    parts = []
    expected = panel.loc[panel["split"].isin(PHASE_OPENS) & panel.common_features_valid, ["date", "ticker"]]
    expected = expected.sort_values(["date", "ticker"]).reset_index(drop=True)
    for model in MODELS:
        frame = read(root / f"results/{model}/{model}_predictions.csv", parse_dates=["date"])
        if list(frame) != list(FORECAST_FIELDS) or not frame.model.eq(model).all():
            raise ValueError("Unexpected seven-column forecast interface: " + model)
        frame["fit_cutoff"] = pd.to_datetime(frame.fit_cutoff, utc=True).dt.tz_convert(NY)
        if frame.duplicated(["date", "ticker", "scheme"]).any() or set(frame.scheme) != set(SCHEMES) or not np.isfinite(frame.y_hat).all():
            raise ValueError("Duplicate, missing-scheme or nonfinite forecasts: " + model)
        for scheme in SCHEMES:
            actual = frame.loc[frame.scheme.eq(scheme), ["date", "ticker"]].sort_values(["date", "ticker"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        parts.append(frame)
    forecasts = pd.concat(parts, ignore_index=True)
    signals, daily = build_trading_status(panel, forecasts, calendar, models=MODELS, schemes=SCHEMES, tickers=TICKERS)
    scoring = build_scoring_eligibility(signals, panel)
    # Store only as-of whitelisted fields in the portfolio signal interface.
    signal_fields = ["date", "ticker", "model", "scheme", "split", "y_hat"]
    daily_fields = ["date", "model", "scheme", "split", "signal_at", "action_at", "action", "force_flat", "signal_complete", "reason"]
    ctx.update(panel=panel, calendar=calendar, forecasts=forecasts, signals=signals[signal_fields].copy(),
               daily=daily[daily_fields].copy(), prices=prices, rates=rates)
    save(out / "inputs/combined_predictions.csv", forecasts)
    save(out / "inputs/trading_signal_inputs.csv", ctx["signals"])
    save(out / "inputs/daily_signal_status.csv", ctx["daily"])
    save(out / "inputs/scoring_eligibility.csv.gz", scoring)
    coverage = forecasts.groupby(["model", "scheme"]).agg(rows=("y_hat", "size"), dates=("date", "nunique")).reset_index()
    save(out / "inputs/prediction_coverage.csv", coverage)
    print(f"Current forecasts: {len(forecasts):,}; daily instructions: {len(daily):,}", flush=True)
    return coverage


def recompute_baselines(ctx):
    root = ctx["root"]
    exports, metrics = {}, {}
    for model in MODELS:
        full = read(root / f"results/{model}/{model}_predictions_full.csv", parse_dates=["date"])
        full["fit_cutoff"] = pd.to_datetime(full.fit_cutoff, utc=True).dt.tz_convert(NY)
        exports[model] = full
        metrics[model] = read(root / f"results/{model}/forecast_metrics.csv")
    monthly = read(root / "notebooks/panel_checks/monthly_training_eligibility.csv", parse_dates=["latest_training_signal"])
    for field in ("fit_at", "latest_label_available_at"):
        monthly[field] = pd.to_datetime(monthly[field], utc=True).dt.tz_convert(NY)
    tables = build_baseline_tables(ctx["panel"], ctx["forecasts"], exports, monthly, metrics, fields=FORECAST_FIELDS)
    for name, frame in tables.items():
        save(ctx["out"] / "baselines" / name, frame)
    ctx["baseline_checks"] = tables["baseline_checks.csv"]
    return tables["prediction_comparison_metrics.csv"]


def run_engine_checks(ctx):
    kwargs = dict(models=MODELS, schemes=SCHEMES, tickers=TICKERS)
    groups = dict(signal=validate_signal_policy(ctx["panel"], ctx["forecasts"], ctx["calendar"], **kwargs),
                  pairs=validate_pair_decisions(TICKERS), selection=validate_selection_logic(),
                  benchmarks=pd.DataFrame(validate_logic()))
    groups["execution"] = validate_net_trades(TICKERS)[0]
    groups["ledger"] = validate_ledger(TICKERS)[0]
    rows = []
    for group, checks in groups.items():
        checks = pd.DataFrame(checks).copy()
        if not checks.passed.all():
            raise AssertionError("Engine checks failed: " + group)
        checks["category"] = group
        rows.append(checks)
    result = pd.concat(rows, ignore_index=True)
    save(ctx["out"] / "checks/engine_checks.csv", result)
    ctx["engine_checks"] = result
    print(f"PASS: {len(result)} freshly executed engine/fault checks", flush=True)
    return result.groupby("category").passed.agg(["size", "sum"])


def phase_inputs(ctx, phase):
    first, last = PHASE_OPENS[phase]
    signals = ctx["signals"].loc[ctx["signals"]["split"].eq(phase)].copy()
    daily = ctx["daily"].loc[ctx["daily"]["split"].eq(phase)].copy()
    prices = ctx["prices"].loc[ctx["prices"].date.between(first, last) & ctx["prices"].ticker.isin([*TICKERS, "SPY"])].copy()
    rates = ctx["rates"].loc[ctx["rates"].entry_date.ge(first) & ctx["rates"].exit_date.le(last)].sort_values("entry_date").reset_index(drop=True)
    opens = pd.DatetimeIndex(prices.loc[prices.ticker.eq("SPY"), "date"].sort_values())
    if not opens.is_unique or str(opens[0].date()) != first or str(opens[-1].date()) != last:
        raise ValueError("Wrong phase opens: " + phase)
    if len(prices) != len(opens) * 10 or prices.duplicated(["date", "ticker"]).any() or not prices.observed.all():
        raise ValueError("Require ten observed execution quotes on every phase open")
    if not np.isfinite(prices[["open", "dividends"]]).all().all() or not prices.open.gt(0).all():
        raise ValueError("Invalid observed quote or dividend")
    if list(zip(rates.entry_date, rates.exit_date)) != list(zip(opens[:-1], opens[1:])):
        raise ValueError("Rates must cover every adjacent opening interval")
    if not rates.interval_valid.all() or not rates.rate_known_at_entry.all() or not np.isfinite(rates.risk_free_return_act365).all():
        raise ValueError("Unavailable or nonfinite cash return")
    if not np.array_equal(rates.holding_calendar_days, (rates.exit_date - rates.entry_date).dt.days):
        raise ValueError("Wrong natural-day count")
    if set(zip(signals.model, signals.scheme)) != STRATEGIES or set(zip(daily.model, daily.scheme)) != STRATEGIES:
        raise ValueError("Each phase requires all six strategies")
    for _, group in daily.groupby(["model", "scheme"]):
        actual = group.loc[group.action.ne("NO_ORDER_OUTSIDE_PHASE"), "action_at"].dt.tz_localize(None).dt.normalize().sort_values()
        if not pd.DatetimeIndex(actual).equals(opens):
            raise ValueError("Scheduled instructions do not cover all phase opens")
    return signals, daily, prices, rates, opens


def audit_pair_book(book, assets, trades, inputs, parameters):
    """Independent cash-flow reconstruction from input quotes, rates and shares."""
    _, _, prices, rates, opens = inputs
    keys = ["model", "scheme", "open_date"]
    source_quotes = prices.sort_values(["ticker", "date"]).copy()
    source_quotes["source_previous_open"] = source_quotes.groupby("ticker").open.shift()
    quoted = assets.merge(source_quotes[["date", "ticker", "open", "dividends", "source_previous_open"]],
        left_on=["open_date", "ticker"], right_on=["date", "ticker"], validate="many_to_one")
    quoted["independent_dividend"] = quoted.current_shares * quoted.dividends
    quoted["independent_trade_cash"] = quoted.delta_shares * quoted.open
    quoted["independent_fee"] = quoted.delta_shares.abs() * quoted.open * parameters["execution_bps"] / 10000
    quoted["independent_market_value"] = quoted.target_shares * quoted.open
    quoted["independent_collateral"] = (-quoted.current_shares * quoted.source_previous_open.fillna(0)).clip(lower=0)
    quoted["independent_price_pnl"] = quoted.current_shares * (quoted.open - quoted.source_previous_open.fillna(quoted.open))
    totals = quoted.groupby(keys)[["independent_dividend", "independent_trade_cash", "independent_fee", "independent_market_value", "independent_collateral", "independent_price_pnl"]].sum()
    b = book.set_index(keys).join(totals)
    collateral = b.independent_collateral
    interest_base = b.cash_prev if parameters["full_cash_rebate"] else (b.cash_prev - collateral).clip(lower=0)
    interest = interest_base * b.interval_risk_free_return
    borrow = collateral * parameters["borrow_annual_rate"] * b.holding_calendar_days / 365
    cash = b.cash_prev + b.independent_dividend + interest - borrow - b.independent_trade_cash - b.independent_fee
    nav = cash + b.independent_market_value
    checks = []
    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))
        if not passed:
            raise AssertionError("Independent ledger check: " + name)
    close = lambda a, z: np.allclose(a, z, rtol=1e-12, atol=1e-12, equal_nan=True)
    check("source_execution_prices", np.array_equal(quoted.open_price, quoted.open))
    check("source_collateral_and_price_pnl", close(b.short_collateral_start, collateral) and close(b.price_pnl, b.independent_price_pnl))
    check("signed_old_share_dividends", close(b.dividend_cash, b.independent_dividend))
    check("net_trade_fees", close(b.execution_fee, b.independent_fee))
    check("cash_carry_and_borrow", close(b.cash_interest, interest) and close(b.borrow_fee, borrow))
    check("independent_cash_and_nav", close(b.cash_post, cash) and close(b.nav_post, nav))
    check("actual_trade_rows", len(trades) == int(assets.delta_shares.ne(0).sum()))
    check("fixed_pair_weights", assets.loc[assets.ticker.ne("SPY"), "target_weight"].abs().isin([0., .125]).all())
    check("net_dollar_neutral_weights", assets.groupby(keys).target_weight.sum().eq(0).all())
    sized = assets.merge(book[keys + ["nav_pre"]], on=keys, validate="many_to_one")
    check("daily_sizing_from_pretrade_nav", close(sized.target_shares * sized.open_price, sized.target_weight * sized.nav_pre))
    ordered_assets = assets.sort_values(["model", "scheme", "ticker", "open_date"])
    previous_targets = ordered_assets.groupby(["model", "scheme", "ticker"]).target_shares.shift().fillna(0)
    check("signed_share_continuity", close(ordered_assets.current_shares, previous_targets))
    aligned = book.loc[~book.is_initial_open].merge(rates[["entry_date", "exit_date", "risk_free_return_act365"]],
        left_on=["interval_entry_date", "open_date"], right_on=["entry_date", "exit_date"], validate="many_to_one")
    check("exact_cash_interval_alignment", len(aligned) == (len(opens)-1)*6 and np.array_equal(aligned.interval_risk_free_return, aligned.risk_free_return_act365))
    for (model, scheme), g in book.groupby(["model", "scheme"]):
        g = g.sort_values("open_date")
        final = assets.loc[assets.model.eq(model) & assets.scheme.eq(scheme) & assets.open_date.eq(opens[-1])]
        check(f"{model}_{scheme}_coverage_reset_liquidation", pd.DatetimeIndex(g.open_date).equals(opens)
            and g.cash_prev.iloc[0] == 1 and g.nav_previous.iloc[0] == 1
            and g.effective_action.iloc[-1] == "FINAL_LIQUIDATION" and final.target_shares.eq(0).all())
        check(f"{model}_{scheme}_return_product", close(np.prod(1+g.holding_interval_return.iloc[1:]), g.nav_post.iloc[-1]))
        check(f"{model}_{scheme}_nav_continuity", close(g.nav_previous.iloc[1:].to_numpy(), g.nav_post.iloc[:-1].to_numpy()))
    return pd.DataFrame(checks)


def pair_replay(inputs, entry, parameters):
    return replay_ledger(*inputs[:4], h_in=entry/10000, initial_nav=1., tickers=TICKERS,
                         require_final_liquidation=True, **parameters)


def save_books(folder, frames):
    for name, frame in zip(("daily", "assets", "trades"), frames):
        save(folder / f"{name}.csv.gz", frame)


def select_threshold(ctx):
    inputs = phase_inputs(ctx, "validation")
    summaries, best, selected_frames = [], None, None
    for entry in ENTRY_CANDIDATES:
        print(f"Validation threshold {entry}/{entry/2:g} bp: fresh six-strategy replay", flush=True)
        frames = pair_replay(inputs, entry, main_cost_parameters())
        summary, returns = summarize_candidate(*frames, entry)
        checks = validate_candidate(*frames, summary, returns, inputs[4], entry)
        save_books(ctx["out"] / f"thresholds/entry_{entry}", frames)
        save(ctx["out"] / f"thresholds/entry_{entry}/checks.csv", checks)
        summaries.append(summary)
        score = float(summary.mean_daily_cash_excess_return.mean())
        if best is None or (score, entry) > best:
            best, selected_frames = (score, entry), frames
    all_summary = pd.concat(summaries, ignore_index=True)
    scores = rank_candidates(all_summary)
    entry = int(scores.iloc[0].entry_bps)
    if entry != best[1]:
        raise AssertionError("Threshold selection disagrees with registered ranking")
    ctx.update(entry=entry, validation_frames=selected_frames)
    save(ctx["out"] / "thresholds/validation_strategy_summary.csv", all_summary)
    save(ctx["out"] / "thresholds/validation_candidate_scores.csv", scores)
    write_json(ctx["out"] / "thresholds/selected_threshold.json", dict(entry_bps=entry, exit_bps=entry/2,
               h_in=entry/10000, h_out=entry/20000, selection_split="validation", applies_to_all_six_strategies=True))
    print(f"Selected by THIS run: {entry}/{entry/2:g} bp", flush=True)
    return scores


def run_pair_phases(ctx):
    rows, audits = [], []
    ctx["pair_books"] = {}
    for phase in PHASE_OPENS:
        inputs = phase_inputs(ctx, phase)
        # Reuse only validation just computed above, never a previous run.
        frames = ctx.pop("validation_frames") if phase == "validation" else pair_replay(inputs, ctx["entry"], main_cost_parameters())
        audit = audit_pair_book(*frames, inputs, main_cost_parameters())
        audit["split"] = phase
        audits.append(audit)
        stats, _ = strategy_statistics(frames[0], scenario_records()[0])
        stats["entry_bps"], stats["exit_bps"] = ctx["entry"], ctx["entry"]/2
        rows.append(stats)
        save_books(ctx["out"] / "pair" / phase, frames)
        ctx["pair_books"][phase] = frames[0]
        print(f"PASS: {phase} six fresh ledgers; {len(audit)} independent checks", flush=True)
    result = pd.concat(rows, ignore_index=True)
    save(ctx["out"] / "pair/main_metrics.csv", result)
    save(ctx["out"] / "checks/pair_accounting.csv", pd.concat(audits, ignore_index=True))
    ctx["pair_metrics"] = result
    return result


def run_benchmarks(ctx):
    rows, books = [], []
    ctx["benchmark_books"] = {}
    for phase in PHASE_OPENS:
        _, _, prices, rates, _ = phase_inputs(ctx, phase)
        quotes = prices.loc[prices.ticker.eq("SPY")].sort_values("date")
        for name in ("spy_buy_hold", "cash"):
            book = replay_benchmark(quotes, rates, phase=phase, benchmark=name, initial_nav=1., execution_bps=3.)
            metrics, _, _ = path_statistics(book)
            if not np.isclose(book.nav_post.iloc[-1], np.prod(1+book.holding_interval_return.iloc[1:]), rtol=1e-12, atol=1e-12):
                raise AssertionError("Benchmark return product mismatch")
            rows.append(dict(split=phase, series_id=name, kind="benchmark", **metrics))
            books.append(book)
            ctx["benchmark_books"][(phase, name)] = book
    result = pd.DataFrame(rows)
    save(ctx["out"] / "benchmarks/summary.csv", result)
    save(ctx["out"] / "benchmarks/daily.csv.gz", pd.concat(books, ignore_index=True))
    ctx["benchmark_metrics"] = result
    return result


def run_sensitivities(ctx):
    rows = [ctx["pair_metrics"]]
    for scenario in scenario_records()[1:]:
        params = {k: scenario[k] for k in main_cost_parameters()}
        for phase in PHASE_OPENS:
            print("Fresh cost sensitivity:", scenario["scenario_id"], phase, flush=True)
            inputs = phase_inputs(ctx, phase)
            frames = pair_replay(inputs, ctx["entry"], params)
            audit = audit_pair_book(*frames, inputs, params)
            summary, _ = strategy_statistics(frames[0], scenario)
            rows.append(summary)
            folder = ctx["out"] / "sensitivities" / scenario["scenario_id"] / phase
            save_books(folder, frames)
            save(folder / "checks.csv", audit)
    result = pd.concat(rows, ignore_index=True)
    save(ctx["out"] / "sensitivities/metrics.csv", result)
    save(ctx["out"] / "sensitivities/scenarios.csv", pd.DataFrame(scenario_records()))
    ctx["sensitivity_metrics"] = result
    return result


def plot_curves(books, path, title, reference=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, book in books.items():
        b = book.sort_values("open_date")
        x = b.open_date
        axes[0].plot(x, b.nav_post, label=name, linewidth=1.3)
        wealth = np.r_[1., b.nav_post.to_numpy()]
        dd = wealth/np.maximum.accumulate(wealth)-1
        axes[1].plot(x, dd[1:], label=name, linewidth=1.1)
    axes[0].set(title=title, ylabel="Net asset value (initial capital 1)")
    axes[1].set(ylabel="Drawdown", xlabel="Execution open")
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def yearly_rows(book, series_id, phase):
    b = book.sort_values("open_date").iloc[1:].copy()
    rows = []
    for year, group in b.groupby(b.open_date.dt.year):
        rows.append(dict(split=phase, series_id=series_id, year=int(year), partial_year=int(year) == 2026,
                         net_period_return=float(np.prod(1+group.holding_interval_return)-1)))
    return rows


def output_hashes(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "complete.json" and "overlay" not in p.relative_to(folder).parts}


def finish_pair_run(ctx):
    yearly = []
    for phase in PHASE_OPENS:
        book = ctx["pair_books"][phase]
        curves = {f"{m}/{s}": g for (m, s), g in book.groupby(["model", "scheme"])}
        for name in ("spy_buy_hold", "cash"):
            curves[name] = ctx["benchmark_books"][(phase, name)]
        plot_curves(curves, ctx["out"] / f"figures/pair_{phase}.png", f"Fresh pair backtest: {phase}")
        for name, b in curves.items():
            yearly.extend(yearly_rows(b, name, phase))
    save(ctx["out"] / "pair/yearly_returns.csv", pd.DataFrame(yearly))
    verify_hashes(ctx["root"], ctx["config"]["input_hashes"])
    verify_hashes(ctx["root"], ctx["config"]["code_hashes"])
    record = dict(status="complete", run_id=ctx["config"]["run_id"], runtime=runtime(),
                  current_forecasts_used=True, historical_backtest_files_used=False,
                  threshold_selected_this_run=True, pair_phase_replay=True,
                  cost_sensitivities_replayed="sensitivity_metrics" in ctx,
                  outputs=output_hashes(ctx["out"]))
    write_json(ctx["out"] / "complete.json", record)
    write_json(ctx["root"] / "results/reproduction/latest_pair_run.json",
               dict(run_dir=ctx["out"].relative_to(ctx["root"]).as_posix(), run_id=record["run_id"]))
    print("05 COMPLETE:", ctx["out"], flush=True)
    return record


def connect_pair_run(root, run_dir=None):
    root = Path(root).resolve()
    if run_dir is None:
        pointer = root / "results/reproduction/latest_pair_run.json"
        if not pointer.is_file():
            raise FileNotFoundError("Run the NEW notebook 05 to completion before notebook 06; historical results are not a fallback.")
        run_dir = read_json(pointer)["run_dir"]
    out = (root / run_dir).resolve()
    if not out.is_relative_to(root / "results/reproduction"):
        raise ValueError("Expected a fresh run inside results/reproduction")
    config, complete = read_json(out / "run_config.json"), read_json(out / "complete.json")
    if complete["status"] != "complete" or config["version"] != VERSION or config["runtime"] != runtime():
        raise ValueError("Incomplete or incompatible 05 run; rerun 05 with the current environment")
    verify_hashes(root, config["input_hashes"])
    verify_hashes(root, config["code_hashes"])
    verify_hashes(out, complete["outputs"])
    ctx = dict(root=root, out=out, config=config)
    prepare_current_inputs_readonly(ctx)
    return ctx


def prepare_current_inputs_readonly(ctx):
    """Rehydrate only THIS completed 05 run's generated signal interface."""
    out = ctx["out"]
    ctx["signals"] = read(out / "inputs/trading_signal_inputs.csv", parse_dates=["date"])
    ctx["daily"] = read(out / "inputs/daily_signal_status.csv", parse_dates=["date"])
    for field in ("signal_at", "action_at"):
        ctx["daily"][field] = pd.to_datetime(ctx["daily"][field], utc=True).dt.tz_convert(NY)
    ctx["prices"] = read(ctx["root"] / "data/interim/etf_prices_aligned.csv", parse_dates=["date"])
    ctx["rates"] = read(ctx["root"] / "data/interim/risk_free_open_intervals.csv", parse_dates=["entry_date", "exit_date"])


def start_overlay_run(ctx):
    folder = ctx["out"] / "overlay" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    ctx.update(overlay_out=folder, overlay_summaries=[], overlay_selected_books={}, reinvest_books={}, overlay_checks=[])
    write_json(folder / "run_config.json", dict(pair_run_id=ctx["config"]["run_id"], runtime=runtime(),
        entries_bps=list(OVERLAY_ENTRIES), phase_opens=PHASE_OPENS, rules=protocol(), fresh_replay=True))
    print("NEW overlay run:", folder, flush=True)
    return folder


def run_overlay_phase(ctx, phase):
    inputs = phase_inputs(ctx, phase)
    signals, daily, prices, rates, opens = inputs
    daily = daily.rename(columns={"reason": "signal_reason"})
    reinvest, ri_assets = replay_spy_reinvest(prices, rates, opens, phase)
    ctx["reinvest_books"][phase] = reinvest
    save(ctx["overlay_out"] / phase / "spy_reinvest_daily.csv.gz", reinvest)
    save(ctx["overlay_out"] / phase / "spy_reinvest_assets.csv.gz", ri_assets)
    ref = reinvest.iloc[1:].set_index("open_date").holding_interval_return
    ending = float(reinvest.nav_post.iloc[-1])
    rows = []
    for entry in OVERLAY_ENTRIES:
        print(f"Fresh overlay: {phase}, {entry}/{entry/2:g} bp, six strategies", flush=True)
        book, assets, trades, decisions = replay_overlay(signals, daily, prices, rates, entry)
        checks = validate_run(book, assets, rates, opens, entry)
        folder = ctx["overlay_out"] / phase / f"entry_{entry}"
        save_books(folder, (book, assets, trades))
        save(folder / "decisions.csv.gz", decisions)
        save(folder / "checks.csv", checks)
        for (model, scheme), b in book.groupby(["model", "scheme"]):
            a = assets.loc[assets.model.eq(model) & assets.scheme.eq(scheme)]
            series = f"overlay_{entry}_{model}_{scheme}"
            row, _ = summary_row(b, series_id=series, kind="spy_overlay", model=model, scheme=scheme,
                entry_bps=entry, spy_returns=ref, spy_ending=ending, assets=a)
            row["split"] = phase
            row["cumulative_relative_to_spy_reinvest"] = row.pop("cumulative_relative_to_spy")
            row["cumulative_return_difference_vs_spy_reinvest"] = row.pop("cumulative_return_difference_vs_spy")
            row["annualized_tracking_error_vs_spy_reinvest"] = row.pop("annualized_tracking_error")
            row["information_ratio_vs_spy_reinvest"] = row.pop("information_ratio")
            row["relative_performance_benchmark"] = "spy_reinvest"
            if abs(row["accounting_residual"]) > 1e-10:
                raise AssertionError("Overlay aggregate PnL reconciliation failed")
            rows.append(row)
            ctx["overlay_selected_books"][(phase, series)] = b.copy()
        checks["split"], checks["entry_bps"] = phase, entry
        ctx["overlay_checks"].append(checks)
    result = pd.DataFrame(rows)
    ctx["overlay_summaries"].append(result)
    save(ctx["overlay_out"] / phase / "summary.csv", result)
    return result


def finish_overlay_run(ctx):
    out = ctx["overlay_out"]
    metrics = pd.concat(ctx["overlay_summaries"], ignore_index=True)
    if len(metrics) != 24 or set(metrics["split"]) != set(PHASE_OPENS):
        raise ValueError("Complete both validation and test: 12 combinations per phase")
    test = metrics.loc[metrics["split"].eq("test")].copy()
    ranked = test.sort_values(["annualized_net_return", "series_id"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "test_return_rank", np.arange(1, 13))
    ranked["selected_for_final_presentation"] = ranked.test_return_rank.eq(1)
    val = metrics.loc[metrics["split"].eq("validation"), ["series_id", "annualized_net_return"]]
    ranked = ranked.merge(val.rename(columns={"annualized_net_return": "validation_annualized_net_return"}), on="series_id", validate="one_to_one")
    selected = ranked.iloc[0]
    selection = dict(series_id=selected.series_id, model=selected.model, scheme=selected.scheme,
        entry_bps=int(selected.entry_bps), exit_bps=float(selected.exit_bps),
        ex_post_selection=True, selection_criterion="Highest TEST net CAGR; exact ties use series_id",
        strategy_replayed=True, historical_backtest_files_used=False, pair_run_id=ctx["config"]["run_id"])
    comparison, yearly = [], []
    for phase in PHASE_OPENS:
        ri = ctx["reinvest_books"][phase]
        ri_end = float(ri.nav_post.iloc[-1])
        b = ctx["overlay_selected_books"][(phase, selected.series_id)]
        curves = {selected.series_id: b, "spy_reinvest": ri}
        row = metrics.loc[metrics["split"].eq(phase) & metrics.series_id.eq(selected.series_id)].iloc[0].to_dict()
        comparison.append(row)
        stats, _, _ = path_statistics(ri)
        comparison.append(dict(split=phase, series_id="spy_reinvest", kind="benchmark", cumulative_relative_to_spy_reinvest=0., **stats))
        for name in ("spy_buy_hold", "cash"):
            benchmarks = read(ctx["out"] / "benchmarks/daily.csv.gz", parse_dates=["open_date"])
            bench = benchmarks.loc[benchmarks["split"].eq(phase) & benchmarks.benchmark.eq(name)]
            stats, _, _ = path_statistics(bench)
            comparison.append(dict(split=phase, series_id=name, kind="benchmark",
                cumulative_relative_to_spy_reinvest=stats["ending_nav"]/ri_end-1, **stats))
            curves[name] = bench
        plot_curves(curves, out / f"selected_{phase}.png", f"SPY overlay: {phase} (selection is ex post)")
        for name, curve in curves.items():
            yearly.extend(yearly_rows(curve, name, phase))
    summary = pd.DataFrame(comparison)
    yearly = pd.DataFrame(yearly)
    save(out / "all_metrics.csv", metrics)
    save(out / "all_12_candidates_ranked.csv", ranked)
    save(out / "selected_and_benchmarks.csv", summary)
    save(out / "yearly_returns.csv", yearly)
    save(out / "checks.csv", pd.concat(ctx["overlay_checks"], ignore_index=True))
    write_json(out / "selection_record.json", selection)
    report = (f"# Fresh SPY overlay results\n\nPair run: {selection['pair_run_id']}\n\n"
        f"Selected presentation: {selected.series_id}; TEST net CAGR {selected.annualized_net_return:.3%}.\n\n"
        "All 12 candidates were recomputed in both phases from current predictions. Selection uses test results "
        "and is exploratory, not independent holdout evidence. 2026 annual rows end on September 30.\n\n"
        "Relative performance, tracking error and information ratio columns ending in "
        "`_vs_spy_reinvest` use dividend-reinvestment SPY over matching holding intervals.\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    verify_hashes(ctx["root"], ctx["config"]["input_hashes"])
    verify_hashes(ctx["root"], ctx["config"]["code_hashes"])
    # Verify the parent run again: 06 must not alter accepted 05 outputs.
    verify_hashes(ctx["out"], read_json(ctx["out"] / "complete.json")["outputs"])
    outputs = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "complete.json"}
    write_json(out / "complete.json", dict(status="complete", strategy_replayed=True,
        historical_backtest_files_used=False, pair_run_id=selection["pair_run_id"], outputs=outputs))
    print("06 COMPLETE:", out, "\nSelected from freshly calculated results:", selected.series_id, flush=True)
    return summary, ranked, yearly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=["pair", "overlay", "all"], required=True)
    parser.add_argument("--skip-cost-sensitivities", action="store_true", help="Explicit shorter run; never loads old sensitivities")
    args = parser.parse_args()
    if args.stage in ("pair", "all"):
        ctx = start_pair_run(args.project_root)
        prepare_current_inputs(ctx)
        recompute_baselines(ctx)
        run_engine_checks(ctx)
        select_threshold(ctx)
        run_pair_phases(ctx)
        run_benchmarks(ctx)
        if not args.skip_cost_sensitivities:
            run_sensitivities(ctx)
        finish_pair_run(ctx)
    if args.stage in ("overlay", "all"):
        ctx = connect_pair_run(args.project_root)
        start_overlay_run(ctx)
        for phase in PHASE_OPENS:
            run_overlay_phase(ctx, phase)
        finish_overlay_run(ctx)


if __name__ == "__main__":
    main()
