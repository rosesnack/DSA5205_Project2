"""Freeze the complete protocol, reuse validation and evaluate test (step 11).

No fitting, threshold search, investment benchmark replay or cost sensitivity is
performed. Completed historical outputs are read-only. A compatible completed
phase evaluation is hash-verified and loaded without another portfolio replay.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

if __package__:
    from .backtest_costs import main_cost_parameters, scenario_records
    from .backtest_ledger import replay_ledger
    from .backtest_signal_policy import PHASE_OPENS
    from .backtest_threshold_selection import (
        ENGINE_FILES, MODELS, SCHEMES, STRATEGIES, TICKERS, SHARPE_STD_ZERO_ATOL,
        read_frame, save_frame, sha256, summarize_candidate, verify_hashes, write_json,
    )
else:
    from backtest_costs import main_cost_parameters, scenario_records
    from backtest_ledger import replay_ledger
    from backtest_signal_policy import PHASE_OPENS
    from backtest_threshold_selection import (
        ENGINE_FILES, MODELS, SCHEMES, STRATEGIES, TICKERS, SHARPE_STD_ZERO_ATOL,
        read_frame, save_frame, sha256, summarize_candidate, verify_hashes, write_json,
    )

PHASE_SIGNALS = {"validation": ("2018-01-01", "2022-12-31"), "test": ("2023-01-01", "2026-09-30")}
DAILY_FIELDS = ["date", "model", "scheme", "split", "signal_at", "action_at", "action", "force_flat", "signal_complete"]
SIGNAL_FIELDS = ["date", "ticker", "model", "scheme", "split", "y_hat"]
DATE_FIELDS = ("date", "open_date", "interval_entry_date", "first_open_date", "last_open_date")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_record(records, phase, name, passed, detail):
    records.append(dict(phase=phase, check=name, passed=bool(passed), detail=detail))
    if not passed:
        raise AssertionError(f"{phase}: {name}")


def register_protocol(root, out):
    """Write the immutable protocol and run identity before any test replay."""
    selection_out = root / "results/backtest/threshold_selection"
    previous = read_json(selection_out / "run_manifest.json")
    assert all(c["passed"] for c in previous["checks"])
    verify_hashes(root, previous["source_hashes"])
    verify_hashes(root, previous["code_hashes"])
    verify_hashes(root, {r["path"]: r["sha256"] for r in previous["outputs"]})
    selected = read_json(selection_out / "selected_threshold.json")
    assert selected["selection_split"] == "validation" and selected["applies_to_all_six_strategies"]
    assert (selected["h_in"], selected["h_out"]) == (0.0015, 0.00075)
    assert selected["selected_public_entry_bps"] == selected["h_in"] * 10000
    assert selected["selected_public_exit_bps"] == selected["h_out"] * 10000
    assert selected["main_cost_parameters"] == main_cost_parameters()
    policy = dict(
        schema_version=1, status="frozen", scope="Step 11: main-cost six-strategy phase evaluation",
        authority="../5205proj2.md: sections 2-5; existing signal, pair, execution, ledger and cost engines",
        models=list(MODELS), schemes=list(SCHEMES), ticker_tie_order=list(TICKERS),
        model_contract=dict(
            inputs="original seven-column official Elastic Net and GBDT predictions; no diagnostic replacement",
            feature_definition="data/processed/panel_spec.json and existing common validity flags",
            configurations=["results/elastic_net/design_config.json", "results/elastic_net/locked_selection.json",
                            "results/gbdt/model_config.json", "results/gbdt/selected_hyperparameters.csv"],
            fitting="sector-specific monthly expanding fits from 2008; only labels mature by fit cutoff; existing preprocessing and locked hyperparameters",
            implementation="consume the delivered monthly predictions; step 11 does not refit any model"),
        phase_opens={k: list(v) for k, v in PHASE_OPENS.items()}, initial_nav=1.0,
        phase_reset="each model/scheme/phase starts from cash 1 and no shares/pairs; no cross-phase carry",
        timezone="America/New_York", signal_time="18:00", execution_time="next SPY trading day 09:30 observed open",
        phase_classification="signal-day split; final open liquidates only; end-date guard independent of label availability",
        last_complete_test_signal="2026-09-28", final_test_signal_handling="2026-09-29 schedules liquidation; 2026-09-30 no order; no October prices",
        eligibility=dict(
            trading="known common_features_valid, finite nine-sector predictions, correct model month and available fit cutoff",
            scoring="common_scoreable is for forecast scoring only; future targets/label flags never enter replay",
            missing="incomplete nine-sector signal clears all pairs at next scheduled open; execution quote errors stop the run"),
        threshold=dict(source="results/backtest/threshold_selection/selected_threshold.json",
            source_sha256=sha256(selection_out / "selected_threshold.json"), h_in=selected["h_in"], h_out=selected["h_out"],
            entry_bps=15, exit_bps=7.5, comparisons="strict > / <; equality fails", shared_across_all_six=True,
            selection_objective=selected["objective"], selection_tie_rule=selected["tie_rule"], reselection_allowed=False),
        pairs=dict(maximum=4, retain_then_exit_then_fill=True,
            retain="old pairs meeting the directional exit bound remain even outside today's top four",
            entry="only global top four absolute predictions, fixed ticker tie order; no rank-five backfill",
            reversal="requires full opposite entry eligibility; execute final net share change once"),
        execution=dict(leg_weight=0.125, nav_basis="settled NAV_pre", idle_capacity_rescaled=False,
            daily_target_restoration=True, tolerance_band=0, fractional_shares=True,
            spy="sum opposite SPY legs before sizing actual net shares and fees",
            quote_fields=["open", "dividends"], adjusted_prices_used=False, repeat_split_adjustment=False),
        main_cost_parameters=main_cost_parameters(),
        accounting=dict(
            order="old signed-share dividends, interval interest and borrow; NAV_pre; net share trades; execution fee once; NAV_post",
            nav="cash plus signed shares times observed open; short proceeds carry an equal security liability",
            dividend="old shares only at ex-date; long receives, short pays; new ex-date buys receive none",
            execution_fee="sum(abs(delta_shares)*open) * 3/10000, including initial/final and small rebalancing trades",
            interest="max(previous cash - interval-start actual net short collateral, 0) * exported interval RF return",
            borrow="interval-start actual net short collateral * 0.01 * natural calendar days/365",
            no_double_count="NAV changes already include prices, dividends and costs; never add target returns or adjusted total returns"),
        statistics=dict(
            annualization=252, initial_open="no synthetic zero-length return; first actual interval denominator is initial capital 1",
            daily_return="current NAV_post / preceding NAV_post - 1, except first real interval uses initial capital",
            cash_excess="daily net return minus the same exported open-to-open RF return; all cash days retained",
            cumulative_net_return="ending NAV / initial capital - 1",
            annualized_net_return="(ending NAV / initial capital) ** (252 / number of holding intervals) - 1",
            volatility_for_step14="sample standard deviation (ddof=1) of daily net returns times sqrt(252)",
            sharpe="sqrt(252) * mean(daily cash excess) / sample std(ddof=1)", sharpe_std_zero_atol=SHARPE_STD_ZERO_ATOL,
            zero_trade="retain all cash intervals, report zero turnover/holding; zero excess std gives N/A Sharpe",
            drawdown_for_step14="minimum NAV/running maximum NAV - 1; ordered open NAV_post path includes initial pre-fee capital 1",
            turnover="sum(actual two-sided net asset trade notional/NAV_pre) *252/holding_intervals; include both ends; no half factor",
            holding_fraction="all intervals as denominator; position at interval START determines whether held"),
        deferred=dict(
            investment_benchmarks_step12="SPY fixed shares at same opens, dividends to interest-bearing cash, same entry/exit fees; cash compounds existing RF intervals",
            cost_scenarios_step14=scenario_records(), cost_comparison="one factor at a time, same 15/7.5 bps; independent ledgers; never select best costs",
            execution="steps 12-16 are not executed by this stage; broader audit, report and figures remain later"),
        evidence_boundary=dict(
            validation="already used for model and threshold selection; reused evaluation is tuning evidence",
            test="fixed-rule test-period strategy evaluation, not an untouched blind holdout",
            prior_test_predictions_and_metrics_seen=True,
            training_oof="retrospective training-only signal-scale diagnostic; not an alternative official model",
            no_test_driven_changes=True, all_six_results_retained=True,
            freeze_timing="frozen now before step-11 test portfolio replay; not a retrospective claim of freezing before earlier model exploration"),
        validation_replay=False, test_replay=True, model_retraining=False, threshold_search=False,
        sensitivity_replay=False, investment_benchmark_replay=False,
        historical_snapshots="old selected_threshold.json and unselected engineering/OOF flags are unchanged stage-specific snapshots",
    )
    protocol_path = out / "protocol_config.json"
    if protocol_path.exists():
        saved = read_json(protocol_path)
        assert {k: v for k, v in saved.items() if k != "frozen_utc"} == policy, "Frozen phase protocol changed"
        policy = saved
    else:
        policy["frozen_utc"] = utc_now()
        write_json(protocol_path, policy)
    engines = {f"scripts/{name}": sha256(root / "scripts" / name) for name in
               (*ENGINE_FILES, "backtest_phase_evaluation.py", "prepare_backtest_inputs.py")}
    config_path = out / "run_config.json"
    if config_path.exists():
        protected = read_json(config_path)["source_hashes"]
        verify_hashes(root, protected)
    else:
        protected = dict(previous["source_hashes"])
        # Preserve all already registered backtest results, including engineering
        # previews and OOF checkpoints, rather than regenerating their manifests.
        for path in sorted((root / "results/backtest").rglob("*")):
            if path.is_file() and not path.is_relative_to(out):
                protected[path.relative_to(root).as_posix()] = sha256(path)
    config = dict(protocol_sha256=sha256(protocol_path), source_hashes=protected, engine_hashes=engines,
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__)
    if config_path.exists():
        assert read_json(config_path) == config, "Phase source/engine/runtime changed; incompatible cache"
    else:
        write_json(config_path, config)
    return policy, config, sha256(config_path)


def load_phase_inputs(root, phase):
    """Read only signal-time columns into the replay; verify original y_hat."""
    daily = read_frame(root / "results/backtest/daily_signal_status.csv", parse_dates=["date"], usecols=DAILY_FIELDS)
    daily = daily.loc[daily["split"].eq(phase)].copy()
    for field in ("signal_at", "action_at"):
        daily[field] = pd.to_datetime(daily[field], utc=True).dt.tz_convert("America/New_York")
    signals = read_frame(root / "results/backtest/trading_signal_inputs.csv", parse_dates=["date"], usecols=SIGNAL_FIELDS)
    signals = signals.loc[signals["split"].eq(phase)].copy()
    first, last = PHASE_OPENS[phase]
    prices = read_frame(root / "data/interim/etf_prices_aligned.csv", parse_dates=["date"],
        usecols=["date", "ticker", "open", "dividends", "observed"])
    prices = prices.loc[prices.date.between(first, last) & prices.ticker.isin([*TICKERS, "SPY"])].copy()
    rates = read_frame(root / "data/interim/risk_free_open_intervals.csv", parse_dates=["entry_date", "exit_date"])
    rates = rates.loc[rates.entry_date.ge(first) & rates.exit_date.le(last)].copy()
    opens = pd.DatetimeIndex(prices.loc[prices.ticker.eq("SPY"), "date"].sort_values())
    checks = []
    def check(name, passed, detail):
        check_record(checks, phase, name, passed, detail)
    check("fixed_phase_opens", opens.is_unique and opens[0] == pd.Timestamp(first) and opens[-1] == pd.Timestamp(last),
          "Exact protocol endpoints, not inferred from future labels")
    check("six_original_signal_groups", set(zip(daily.model, daily.scheme)) == STRATEGIES
          and set(zip(signals.model, signals.scheme)) == STRATEGIES, "All six models/schemes, no performance selection")
    check("signal_day_split", signals.date.between(*PHASE_SIGNALS[phase]).all()
          and daily.date.between(*PHASE_SIGNALS[phase]).all(), "Respect the existing signal-day split")
    for _, group in daily.groupby(["model", "scheme"]):
        dates = group.loc[group.action.ne("NO_ORDER_OUTSIDE_PHASE"), "action_at"].dt.tz_localize(None).dt.normalize()
        assert pd.DatetimeIndex(dates.sort_values()).equals(opens)
    check("scheduled_actions_cover_each_open", True, "Exactly one action at every phase open for each strategy")
    check("observed_execution_quotes", len(prices) == len(opens)*10 and prices.observed.eq(True).all()
          and np.isfinite(prices[["open", "dividends"]]).all().all() and prices.open.gt(0).all(),
          "Ten observed open/dividend quotes per execution date; no adjusted prices or fills")
    expected_keys = list(zip(opens[:-1], opens[1:]))
    actual_keys = list(zip(rates.sort_values("entry_date").entry_date, rates.sort_values("entry_date").exit_date))
    check("exact_available_cash_intervals", actual_keys == expected_keys and rates.interval_valid.eq(True).all()
          and rates.rate_known_at_entry.eq(True).all() and np.isfinite(rates.risk_free_return_act365).all(),
          "All adjacent phase opens have exported cash returns available at entry")
    keys = ["date", "ticker", "model", "scheme"]
    official = []
    for model in MODELS:
        frame = read_frame(root / f"results/{model}/{model}_predictions.csv", parse_dates=["date"], usecols=keys + ["y_hat"])
        official.append(frame.loc[frame.date.between(*PHASE_SIGNALS[phase])])
    original = pd.concat(official, ignore_index=True).sort_values(keys).reset_index(drop=True)
    supplied = signals[keys + ["y_hat"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, supplied, check_exact=True)
    check("official_predictions_exact", True, "Trading y_hat exactly equals the original delivered forecast, with no refitting")
    check("no_future_scoring_fields", set(signals) == set(SIGNAL_FIELDS) and set(daily) == set(DAILY_FIELDS),
          "Replay receives no common_scoreable, future target, label_valid or label_within_split")
    return (signals, daily, prices, rates, opens), checks


def read_cached_frame(path):
    frame = read_frame(path)
    for field in DATE_FIELDS:
        if field in frame:
            frame[field] = pd.to_datetime(frame[field])
    return frame


def reuse_validation(root, out, identity, selected_entry):
    source = root / "results/backtest/threshold_selection"
    checkpoint_path = source / f"checkpoints/entry_{selected_entry:g}.json"
    checkpoint = read_json(checkpoint_path)
    assert checkpoint["identity"] == read_json(source / "run_manifest.json")["run_config_sha256"]
    frames = {}
    for item in checkpoint["outputs"]:
        path = source / item["path"]
        assert sha256(path) == item["sha256"]
        frames[item["name"]] = read_cached_frame(path)
        assert len(frames[item["name"]]) == item["rows"]
    summary, returns = summarize_candidate(frames["daily"], frames["assets"], frames["trades"], selected_entry)
    for actual, stored in ((summary, frames["summary"]), (returns, frames["returns"])):
        pd.testing.assert_frame_equal(actual, stored, check_dtype=False, rtol=1e-12, atol=1e-14)
    record = dict(run_config_sha256=identity, phase="validation", replayed=False,
        source_checkpoint=checkpoint_path.relative_to(root).as_posix(), checkpoint_sha256=sha256(checkpoint_path),
        source_outputs=checkpoint["outputs"], source_base=source.relative_to(root).as_posix(),
        entry_bps=selected_entry, exit_bps=selected_entry/2, initial_nav=1.0, main_cost_parameters=main_cost_parameters(),
        equivalence="same original forecasts, phase dates, selected threshold, ledger engines, cash reset and final liquidation",
        summary_and_returns_recomputed_from_saved_ledger=True,
        role="validation tuning evidence; reusing the ledger does not create independent performance evidence")
    target = out / "validation_reuse.json"
    if target.exists():
        assert read_json(target) == record
    else:
        write_json(target, record)
    return frames


def test_checkpoint(root, out, identity, frames=None, started_utc=None):
    path = out / "test_checkpoint.json"
    if frames is None:
        if not path.exists():
            return None
        record = read_json(path)
        assert record["run_config_sha256"] == identity
        result = {}
        for item in record["outputs"]:
            source = root / item["path"]
            assert sha256(source) == item["sha256"]
            result[item["name"]] = read_cached_frame(source)
            assert len(result[item["name"]]) == item["rows"]
        return result
    outputs = []
    for name, frame in frames.items():
        target = out / f"test_{name}.csv"
        save_frame(target, frame)
        outputs.append(dict(name=name, path=target.relative_to(root).as_posix(), rows=len(frame), sha256=sha256(target)))
    write_json(path, dict(run_config_sha256=identity, replay_started_utc=started_utc, replay_completed_utc=utc_now(), outputs=outputs))
    return frames


def validate_phase(phase, frames, inputs):
    """Checks needed to accept this phase run; broader step-13 audit is separate."""
    book, assets, trades, summary, returns = (frames[n] for n in ("daily", "assets", "trades", "summary", "returns"))
    signals, daily, prices, rates, opens = inputs
    checks = []
    def check(name, passed, detail):
        check_record(checks, phase, name, passed, detail)
    check("phase_only_six_strategies", book["split"].eq(phase).all() and len(summary) == 6
          and set(zip(summary.model, summary.scheme)) == STRATEGIES, "Keep all six results in the prescribed phase")
    check("complete_open_and_interval_counts", len(book) == len(opens)*6 and len(assets) == len(opens)*60
          and len(returns) == (len(opens)-1)*6 and summary.holding_intervals.eq(len(opens)-1).all(),
          "All execution opens and cash-only intervals retained")
    first = book.groupby(["model", "scheme"]).head(1)
    last = book.groupby(["model", "scheme"]).tail(1)
    check("independent_cash_and_no_old_shares", first.cash_prev.eq(1).all() and first.nav_previous.eq(1).all()
          and first.short_collateral_start.eq(0).all() and first.holding_calendar_days.eq(0).all()
          and assets.loc[assets.open_date.eq(opens[0]), "current_shares"].eq(0).all(),
          "No prior-phase NAV, pair or share state enters any strategy")
    check("scheduled_final_liquidation", last.effective_action.eq("FINAL_LIQUIDATION").all()
          and last.target_pairs.eq(0).all() and assets.loc[assets.open_date.eq(opens[-1]), "target_shares"].eq(0).all(),
          "Final open liquidates all actual shares, with no new position")
    for (model, scheme), group in book.groupby(["model", "scheme"]):
        group = group.sort_values("open_date")
        subset = returns.loc[returns.model.eq(model) & returns.scheme.eq(scheme)].sort_values("open_date")
        assert pd.DatetimeIndex(group.open_date).equals(opens)
        assert list(zip(subset.interval_entry_date, subset.open_date)) == list(zip(opens[:-1], opens[1:]))
        assert np.isclose(np.prod(1+subset.holding_interval_return), group.nav_post.iloc[-1], rtol=1e-12, atol=1e-12)
        assert pd.isna(group.holding_interval_return.iloc[0]) and group.holding_return_denominator.iloc[1] == 1.
        assert np.allclose(group.nav_previous.iloc[1:], group.nav_post.iloc[:-1], rtol=1e-12, atol=1e-14)
    check("common_adjacent_intervals_and_fee_inclusion", True,
          "Return product reconstructs NAV from capital 1; initial/final costs retained; no zero-length return")
    check("cash_excess_definition", np.array_equal(returns.cash_excess_return.to_numpy(),
          (returns.holding_interval_return-returns.interval_risk_free_return).to_numpy()),
          "Arithmetic same-interval cash subtraction")
    aligned = returns.merge(rates[["entry_date", "exit_date", "risk_free_return_act365"]],
        left_on=["interval_entry_date", "open_date"], right_on=["entry_date", "exit_date"], validate="many_to_one")
    check("cash_rates_match_original_intervals", len(aligned) == len(returns)
          and np.array_equal(aligned.interval_risk_free_return, aligned.risk_free_return_act365),
          "No rate shifting, extra annualization or missing interval")
    check("nav_and_pnl_reconcile", book.pre_reconciliation_error.abs().le(1e-12).all()
          and book.post_reconciliation_error.abs().le(1e-12).all() and summary.accounting_residual.abs().le(1e-10).all(),
          "Phase acceptance check on existing cash/price/dividend/financing NAV identities")
    check("pair_capacity_and_gross", book.target_pairs.between(0, 4).all() and book.target_gross_weight.between(0, 1).all(),
          "At most four unscaled pairs; netted gross no greater than one")
    sectors = assets.loc[assets.ticker.ne("SPY")]
    check("fixed_sector_leg_weights", sectors.target_weight.abs().isin([0.0, 0.125]).all(),
          "Every sector leg is zero or 12.5% of NAV_pre; idle capacity is not rescaled")
    sized = assets.merge(book[["model", "scheme", "open_date", "nav_pre"]],
        on=["model", "scheme", "open_date"], validate="many_to_one")
    check("daily_target_shares", np.allclose(sized.target_shares*sized.open_price,
          sized.target_weight*sized.nav_pre, rtol=1e-12, atol=1e-14), "Daily sizing uses NAV_pre without a tolerance band")
    check("spy_netted_dollar_neutral_targets", assets.groupby(["model", "scheme", "open_date"]).target_weight.sum().eq(0).all(),
          "The single SPY target offsets the combined sector targets before actual trades")
    nonzero = assets.loc[assets.delta_shares.ne(0)]
    check("actual_net_trade_fees", len(nonzero) == len(trades) and np.allclose(assets.execution_fee,
          assets.delta_shares.abs()*assets.open_price*3/10000, rtol=1e-12, atol=1e-14),
          "Charge the actual net asset share difference once, after SPY netting")
    asset_totals = assets.groupby(["model", "scheme", "open_date"])[["execution_fee", "market_value_post", "dividend_cash_old"]].sum()
    totals = book.set_index(["model", "scheme", "open_date"]).join(asset_totals, rsuffix="_assets")
    check("asset_totals_match_portfolio", np.allclose(totals.execution_fee, totals.execution_fee_assets, atol=1e-14, rtol=1e-12)
          and np.allclose(totals.market_value_post, totals.market_value_post_assets, atol=1e-14, rtol=1e-12)
          and np.allclose(totals.dividend_cash, totals.dividend_cash_old, atol=1e-14, rtol=1e-12),
          "Net asset fees, market values and signed dividends reconcile with portfolio totals")
    check("collateral_cash_interest", np.allclose(book.cash_interest,
          np.maximum(book.cash_prev-book.short_collateral_start, 0)*book.interval_risk_free_return, rtol=1e-12, atol=1e-14),
          "Only available own cash earns the exported interval return")
    check("natural_day_borrow", np.allclose(book.borrow_fee,
          book.short_collateral_start*0.01*book.holding_calendar_days/365, rtol=1e-12, atol=1e-14),
          "Actual net starting short collateral, annual 1%, natural days/365")
    quoted = assets.merge(prices[["date", "ticker", "open", "dividends"]], left_on=["open_date", "ticker"],
        right_on=["date", "ticker"], validate="many_to_one", suffixes=("", "_quote"))
    check("observed_prices_and_signed_dividends", len(quoted) == len(assets)
          and np.array_equal(quoted.open_price, quoted.open)
          and np.allclose(quoted.dividend_cash_old, quoted.current_shares*quoted.dividends, rtol=1e-12, atol=1e-14),
          "Use observed prices and old signed shares; no adjusted-return or new-buy dividend double count")
    check("fixed_public_threshold", book.entry_bps.eq(15).all() and book.exit_bps.eq(7.5).all()
          and book.scenario_id.eq("main").all(), "Only the selected public threshold and main cost case")
    check("zero_trade_reporting", all(r.holding_interval_fraction == 0 and r.annualized_two_sided_turnover == 0
          and pd.isna(r.annualized_cash_excess_sharpe) for r in summary.loc[summary.asset_trade_records.eq(0)].itertuples()),
          "Cash-only strategies remain valid results with N/A zero-variance Sharpe")
    incomplete = daily.loc[daily.force_flat & daily.action.ne("NO_ORDER_OUTSIDE_PHASE")]
    joined = book.merge(incomplete[["date", "model", "scheme"]], on=["date", "model", "scheme"])
    check("force_flat_instructions_respected", joined.target_pairs.eq(0).all(),
          "Incomplete signals and phase-end guards result in zero pairs")
    return checks


def run_phase_evaluation(project_root):
    root = Path(project_root).resolve()
    out = root / "results/backtest/phase_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    policy, config, identity = register_protocol(root, out)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["run_config_sha256"] == identity
        verify_hashes(root, {r["path"]: r["sha256"] for r in manifest["outputs"]})
        print("Step 11: verified frozen protocol and completed phase cache; no replay", flush=True)
        return manifest
    print("Step 11: complete protocol and runtime/source/engine hashes frozen before test replay", flush=True)
    checks, frames_by_phase, inputs_by_phase = [], {}, {}
    for phase in ("validation", "test"):
        inputs, input_checks = load_phase_inputs(root, phase)
        inputs_by_phase[phase] = inputs
        checks.extend(input_checks)
        if phase == "validation":
            frames = reuse_validation(root, out, identity, policy["threshold"]["entry_bps"])
            print("Validation: verified and reused complete 15-bps ledger; no replay", flush=True)
        else:
            frames = test_checkpoint(root, out, identity)
            if frames is None:
                replay_started = utc_now()
                book, assets, trades = replay_ledger(*inputs[:4], h_in=policy["threshold"]["h_in"],
                    initial_nav=policy["initial_nav"], tickers=TICKERS, require_final_liquidation=True,
                    **policy["main_cost_parameters"])
                summary, returns = summarize_candidate(book, assets, trades, policy["threshold"]["entry_bps"])
                for frame in (book, assets, trades):
                    frame["entry_bps"], frame["exit_bps"] = 15, 7.5
                    frame["scenario_id"], frame["evaluation_role"] = "main", "fixed_protocol_test_evaluation"
                    frame["example_only"], frame["phase_complete"] = False, True
                frames = test_checkpoint(root, out, identity, dict(daily=book, assets=assets, trades=trades,
                    summary=summary, returns=returns), replay_started)
                print(f"Test: six fixed-rule ledgers complete | elapsed {time.perf_counter()-started:.1f}s", flush=True)
            else:
                print("Test: verified saved step-11 checkpoint; no replay", flush=True)
        checks.extend(validate_phase(phase, frames, inputs))
        frames_by_phase[phase] = frames
    test_book = frames_by_phase["test"]["daily"]
    en_signals = inputs_by_phase["test"][0].loc[lambda f: f.model.eq("elastic_net")]
    reference_signal, reference_book = None, None
    for scheme in SCHEMES:
        values = en_signals.loc[en_signals.scheme.eq(scheme), ["date", "ticker", "y_hat"]].sort_values(["date", "ticker"]).reset_index(drop=True)
        ledger = test_book.loc[test_book.model.eq("elastic_net") & test_book.scheme.eq(scheme)].drop(columns="scheme").reset_index(drop=True)
        if reference_signal is not None:
            pd.testing.assert_frame_equal(reference_signal, values, check_exact=True)
            pd.testing.assert_frame_equal(reference_book, ledger, check_exact=True)
        reference_signal, reference_book = values, ledger
    check_record(checks, "cross_phase", "identical_en_test_predictions_give_identical_ledgers", True,
        "The delivered EN S0/S1/S2 test predictions are identical, and produce identical deterministic cash/NAV ledgers")
    checkpoint = read_json(out / "test_checkpoint.json")
    check_record(checks, "cross_phase", "protocol_frozen_before_test_replay",
        pd.Timestamp(policy["frozen_utc"]) <= pd.Timestamp(checkpoint["replay_started_utc"]),
        "Freeze timestamp precedes the first step-11 test portfolio replay; earlier model exploration is disclosed")
    verify_hashes(root, config["source_hashes"])
    verify_hashes(root, config["engine_hashes"])
    check_record(checks, "cross_phase", "protected_inputs_old_results_and_engines_unchanged", True,
        "All registered old outputs, data, official predictions, original notebooks and existing engines retain their hashes")
    summaries = []
    for phase, frames in frames_by_phase.items():
        summary = frames["summary"].copy()
        summary["annualized_net_return"] = np.expm1(np.log(summary.ending_nav)*252/summary.holding_intervals)
        summary["evaluation_role"] = "validation_tuning_evidence" if phase == "validation" else "fixed_protocol_test_evaluation"
        summary["ledger_replayed_in_step11"] = phase == "test"
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    save_frame(out / "phase_summary.csv", combined)
    save_frame(out / "evaluation_checks.csv", pd.DataFrame(checks))
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p))
               for p in sorted(out.iterdir()) if p.is_file() and p != manifest_path]
    manifest = dict(schema_version=1, scope="Step 11: frozen protocol, reused validation and fixed-rule test evaluation",
        completed_utc=utc_now(), protocol_frozen_utc=policy["frozen_utc"], protocol_frozen=True,
        run_config_sha256=identity, protocol_sha256=config["protocol_sha256"],
        python_version=config["python_version"], numpy_version=config["numpy_version"], pandas_version=config["pandas_version"],
        selected_public_entry_bps=15, selected_public_exit_bps=7.5, main_cost_parameters=main_cost_parameters(),
        phases={phase: dict(opening_records_per_strategy=len(inputs_by_phase[phase][4]),
            holding_intervals_per_strategy=len(inputs_by_phase[phase][4])-1,
            first_open_date=PHASE_OPENS[phase][0], last_open_date=PHASE_OPENS[phase][1],
            daily_ledger_rows=len(frames["daily"]), asset_ledger_rows=len(frames["assets"]),
            asset_trade_rows=len(frames["trades"]), return_rows=len(frames["returns"]),
            replayed_in_step11=phase == "test") for phase, frames in frames_by_phase.items()},
        validation_replayed=False, test_strategy_evaluated=True, model_retraining_performed=False,
        threshold_reselection_performed=False, sensitivity_scenarios_evaluated=False, investment_benchmarks_evaluated=False,
        checks=checks, protected_file_count=len(config["source_hashes"]), source_hashes=config["source_hashes"],
        code_hashes=config["engine_hashes"], outputs=outputs, elapsed_seconds=time.perf_counter()-started,
        notes=["Validation is selected/tuning evidence, even though phase accounts reset independently",
            "Test forecast metrics were previously viewed; GBDT has disclosed exploratory provenance; this is not an untouched blind holdout",
            "The existing selected threshold file and all earlier manifests remain historical snapshots; this protocol owns the current freeze status",
            "Step-11 phase acceptance checks do not mark the broader step-13 audit complete",
            "Basic same-interval cash-excess fields are reused for evaluation; no step-12 investment benchmark artifact is produced"],
        not_yet_checked=["step 12: SPY buy-and-hold and cash investment benchmarks",
            "step 13: broader accounting and deterministic consistency audit",
            "step 14: full strategy reporting, comparison figures and frozen cost/financing sensitivities",
            "step 15: final package and handoff", "step 16: optional extensions only if separately requested"])
    write_json(manifest_path, manifest)
    print(f"PASS: {len(checks)} phase-evaluation checks; validation reused; all six test results retained", flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_phase_evaluation(args.project_root)
