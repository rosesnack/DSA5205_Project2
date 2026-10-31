"""Select one public threshold on official validation forecasts only (step 10).

The model predictions, pair rules, signed-share ledger and main cost policy are
read-only inputs. Five pre-registered candidates each start independently from
cash 1 and finish flat. No test-period or cost-sensitivity replay occurs here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

if __package__:
    from .backtest_costs import main_cost_parameters, synthetic_cost_fixture
    from .backtest_ledger import replay_ledger
    from .backtest_signal_policy import PHASE_OPENS
else:
    from backtest_costs import main_cost_parameters, synthetic_cost_fixture
    from backtest_ledger import replay_ledger
    from backtest_signal_policy import PHASE_OPENS

MODELS = ("elastic_net", "gbdt")
SCHEMES = ("S0", "S1", "S2")
TICKERS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
STRATEGIES = {(m, s) for m in MODELS for s in SCHEMES}
ENGINE_FILES = ("backtest_signal_policy.py", "backtest_pair_decisions.py", "backtest_execution.py",
                "backtest_ledger.py", "backtest_costs.py", "backtest_threshold_selection.py")
SHARPE_STD_ZERO_ATOL = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_frame(path, frame):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", float_format="%.17g")
    temp.replace(path)


def read_frame(path, **kwargs):
    return pd.read_csv(path, low_memory=False, float_precision="round_trip", **kwargs)


def verify_hashes(root, hashes):
    for path, expected in hashes.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Threshold-selection input changed: {path}")


def register_design(root, out):
    previous = json.loads((root / "results/backtest/training_oof_manifest.json").read_text(encoding="utf-8"))
    assert all(c["passed"] for c in previous["checks"])
    protected = dict(previous["source_hashes"])
    protected.update({r["path"]: r["sha256"] for r in previous["outputs"]})
    verify_hashes(root, protected)
    grid_path = root / "results/backtest/training_oof/threshold_candidates.csv"
    grid = read_frame(grid_path)
    assert grid.entry_bps.tolist() == [5, 10, 15, 20, 30]
    assert grid.exit_bps.eq(grid.entry_bps / 2).all() and not grid.selected.any()
    policy = dict(schema_version=1, source="../5205proj2.md: section 3.4, 3.6-3.7",
        selection_split="validation", signal_period="2018-2022", phase_opens=list(PHASE_OPENS["validation"]),
        entry_candidates_bps=grid.entry_bps.tolist(), exit_fraction=0.5,
        candidate_source_sha256=sha256(grid_path), initial_nav=1.0, main_cost_parameters=main_cost_parameters(),
        objective="equal-weight mean of six strategy mean daily (net portfolio holding return minus same-interval cash return)",
        daily_return="NAV_post(current open)/NAV_post(previous open)-1; first real interval denominator is initial capital, including opening fee",
        return_intervals="every adjacent validation open, including cash/flat intervals; initial zero-length opening row not scored",
        cash_excess="arithmetic subtraction of exported ACT/365 open-interval cash return, not relative-to-SPY target or geometric ratio",
        tie_rule="exact equality of unrounded float objectives; choose larger entry_bps; never round before ranking",
        sharpe="sqrt(252)*mean(daily cash excess)/sample std(ddof=1); N/A when std <= numerical-zero tolerance",
        sharpe_std_zero_atol=SHARPE_STD_ZERO_ATOL, annualization=252,
        turnover="two-sided actual net asset trade notional / NAV_pre, including initial/final trades; annualized sum *252/holding_intervals; no half factor",
        holding_fraction="fraction of all holding intervals whose START-open target pair count is positive",
        no_trade_policy="retain cash days; zero turnover/holding; zero excess-variance Sharpe is N/A; never lower grid to force trading",
        official_validation_predictions_used=True, training_diagnostic_forecasts_used=False,
        test_strategy_evaluated=False, sensitivity_scenarios_evaluated=False,
        protocol_freeze_step=11)
    path = out / "design_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert {k: v for k, v in saved.items() if k != "registered_utc"} == policy, "Registered threshold policy changed"
        policy = saved
    else:
        policy["registered_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, policy)
    engines = {f"scripts/{name}": sha256(root / "scripts" / name) for name in ENGINE_FILES}
    run_config = dict(design_sha256=sha256(path), source_hashes=protected, engine_hashes=engines,
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__)
    config_path = out / "run_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == run_config, "Selection source/engine/runtime changed; incompatible cache"
    else:
        write_json(config_path, run_config)
    return policy, run_config, sha256(config_path)


def load_validation_inputs(root):
    daily = read_frame(root / "results/backtest/daily_signal_status.csv", parse_dates=["date"])
    daily = daily.loc[daily["split"].eq("validation")].copy()
    for field in ("signal_at", "action_at"):
        daily[field] = pd.to_datetime(daily[field], utc=True).dt.tz_convert("America/New_York")
    # Only signal-time fields enter the replay, not target labels or scoring masks.
    daily = daily[["date", "model", "scheme", "split", "signal_at", "action_at",
                   "action", "force_flat", "signal_complete"]]
    signals = read_frame(root / "results/backtest/trading_signal_inputs.csv", parse_dates=["date"],
        usecols=["date", "ticker", "model", "scheme", "split", "y_hat"])
    signals = signals.loc[signals["split"].eq("validation")].copy()
    first, last = PHASE_OPENS["validation"]
    prices = read_frame(root / "data/interim/etf_prices_aligned.csv", parse_dates=["date"],
        usecols=["date", "ticker", "open", "dividends"])
    prices = prices.loc[prices.date.between(first, last) & prices.ticker.isin([*TICKERS, "SPY"])].copy()
    rates = read_frame(root / "data/interim/risk_free_open_intervals.csv", parse_dates=["entry_date", "exit_date"])
    rates = rates.loc[rates.entry_date.ge(first) & rates.exit_date.le(last)].copy()
    opens = pd.DatetimeIndex(prices.loc[prices.ticker.eq("SPY"), "date"].sort_values())
    assert opens.is_unique and opens[0] == pd.Timestamp(first) and opens[-1] == pd.Timestamp(last)
    assert set(zip(daily.model, daily.scheme)) == STRATEGIES
    assert set(zip(signals.model, signals.scheme)) == STRATEGIES
    assert signals.date.between("2018-01-01", "2022-12-31").all()
    assert daily.date.between("2018-01-01", "2022-12-31").all()
    for _, group in daily.groupby(["model", "scheme"]):
        action_dates = group.loc[group.action.ne("NO_ORDER_OUTSIDE_PHASE"), "action_at"].dt.tz_localize(None).dt.normalize()
        assert pd.DatetimeIndex(action_dates.sort_values()).equals(opens)
    keys = ["date", "ticker", "model", "scheme"]
    original = []
    for model in MODELS:
        frame = read_frame(root / f"results/{model}/{model}_predictions.csv", parse_dates=["date"], usecols=keys + ["y_hat"])
        original.append(frame.loc[frame.date.between("2018-01-01", "2022-12-31")])
    original = pd.concat(original, ignore_index=True).sort_values(keys).reset_index(drop=True)
    supplied = signals[keys + ["y_hat"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, supplied, check_exact=True)
    return signals, daily, prices, rates, opens


def summarize_candidate(book, assets, trades, entry_bps):
    """Keep every real holding interval; aggregate equally within each strategy."""
    rows, returns = [], []
    for (model, scheme, split), group in book.groupby(["model", "scheme", "split"], sort=True):
        group = group.sort_values("open_date").reset_index(drop=True)
        interval = group.loc[~group.is_initial_open].copy()
        if len(interval) < 2:
            raise ValueError("At least two holding intervals required")
        interval["cash_excess_return"] = interval.holding_interval_return - interval.interval_risk_free_return
        interval["pairs_held_at_interval_start"] = group.target_pairs.shift(1).loc[interval.index].to_numpy()
        interval["held_any_pair"] = interval.pairs_held_at_interval_start.gt(0)
        excess = interval.cash_excess_return.to_numpy(float)
        if not np.isfinite(excess).all():
            raise ValueError("Every holding interval must have finite excess return")
        std = float(np.std(excess, ddof=1))
        sharpe = float(np.sqrt(252) * excess.mean() / std) if std > SHARPE_STD_ZERO_ATOL else float("nan")
        which = trades.model.eq(model) & trades.scheme.eq(scheme) & trades["split"].eq(split)
        asset_trades = trades.loc[which]
        norm_turnover = float((group.traded_notional / group.nav_pre).sum())
        cash_nav = 1.0
        for rf in interval.interval_risk_free_return:
            cash_nav += cash_nav * rf
        last = group.iloc[-1]
        identity = 1 + group.price_pnl.sum() + group.dividend_cash.sum() + group.cash_interest.sum()
        identity -= group.borrow_fee.sum() + group.execution_fee.sum()
        rows.append(dict(entry_bps=entry_bps, exit_bps=entry_bps/2, model=model, scheme=scheme, split=split,
            opening_records=len(group), holding_intervals=len(interval),
            first_open_date=group.open_date.iloc[0], last_open_date=last.open_date,
            mean_daily_net_return=float(interval.holding_interval_return.mean()),
            mean_daily_cash_return=float(interval.interval_risk_free_return.mean()),
            mean_daily_cash_excess_return=float(excess.mean()), mean_daily_cash_excess_bps=float(excess.mean()*10000),
            daily_cash_excess_std=std, annualized_cash_excess_sharpe=sharpe,
            holding_interval_fraction=float(interval.held_any_pair.mean()),
            mean_pairs_held=float(interval.pairs_held_at_interval_start.mean()),
            active_holding_intervals=int(interval.held_any_pair.sum()),
            cash_only_holding_intervals=int((~interval.held_any_pair).sum()),
            normalized_two_sided_turnover_sum=norm_turnover,
            mean_daily_two_sided_turnover=norm_turnover/len(interval),
            annualized_two_sided_turnover=norm_turnover*252/len(interval),
            asset_trade_records=len(asset_trades), traded_opens=int(group.traded_notional.gt(0).sum()),
            total_traded_notional=float(group.traded_notional.sum()),
            total_execution_fee=float(group.execution_fee.sum()), total_borrow_fee=float(group.borrow_fee.sum()),
            total_cash_interest=float(group.cash_interest.sum()), total_dividend_cash=float(group.dividend_cash.sum()),
            ending_nav=float(last.nav_post), net_cumulative_return=float(last.nav_post-1),
            cash_ending_nav=cash_nav, ending_relative_to_cash=float(last.nav_post/cash_nav-1),
            accounting_residual=float(last.nav_post-identity)))
        fields = ["date", "model", "scheme", "split", "open_date", "interval_entry_date",
                  "nav_post", "holding_interval_return", "interval_risk_free_return", "cash_excess_return",
                  "pairs_held_at_interval_start", "held_any_pair"]
        interval["entry_bps"], interval["exit_bps"] = entry_bps, entry_bps/2
        returns.append(interval[["entry_bps", "exit_bps", *fields]])
    return pd.DataFrame(rows), pd.concat(returns, ignore_index=True)


def rank_candidates(summary):
    if summary.duplicated(["entry_bps", "model", "scheme"]).any():
        raise ValueError("Duplicate candidate/strategy summary")
    for _, group in summary.groupby("entry_bps"):
        if set(zip(group.model, group.scheme)) != STRATEGIES:
            raise ValueError("Every candidate requires all six strategies")
        if not np.isfinite(group.mean_daily_cash_excess_return).all():
            raise ValueError("Nonfinite candidate objective")
        if group.holding_intervals.nunique() != 1:
            raise ValueError("Unequal holding-interval count within candidate")
    scores = summary.groupby("entry_bps", as_index=False).agg(
        strategy_count=("model", "size"), holding_intervals_per_strategy=("holding_intervals", "first"),
        objective_mean_daily_cash_excess_return=("mean_daily_cash_excess_return", "mean"),
        mean_holding_interval_fraction=("holding_interval_fraction", "mean"),
        mean_annualized_two_sided_turnover=("annualized_two_sided_turnover", "mean"),
        total_asset_trade_records=("asset_trade_records", "sum"))
    if scores.holding_intervals_per_strategy.nunique() != 1:
        raise ValueError("Unequal interval count across candidates")
    scores["objective_mean_daily_cash_excess_bps"] = scores.objective_mean_daily_cash_excess_return * 10000
    scores = scores.sort_values(["objective_mean_daily_cash_excess_return", "entry_bps"], ascending=[False, False]).reset_index(drop=True)
    scores["selection_rank"] = np.arange(1, len(scores)+1)
    scores["selected"] = scores.selection_rank.eq(1)
    return scores


def validate_selection_logic():
    records = []
    def check(name, passed, detail):
        records.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name)
    signals, daily, prices, rates = synthetic_cost_fixture(TICKERS)
    signals["y_hat"] = 0.0
    book, assets, trades = replay_ledger(signals, daily, prices, rates, h_in=.003,
        initial_nav=1., tickers=TICKERS, require_final_liquidation=True, **main_cost_parameters())
    summary, returns = summarize_candidate(book, assets, trades, 30)
    row = summary.iloc[0]
    check("cash_fixture_keeps_all_intervals", len(returns) == 3 and row.holding_intervals == 3,
          "Zero signals retain weekend/cash holding intervals")
    check("cash_fixture_no_trades_or_positions", trades.empty and row.holding_interval_fraction == 0
          and row.annualized_two_sided_turnover == 0, "Zero trading is an admissible result")
    check("cash_fixture_zero_excess_sharpe_na", returns.cash_excess_return.abs().le(1e-12).all()
          and pd.isna(row.annualized_cash_excess_sharpe), "Floating-point cash subtraction cannot produce a spurious Sharpe")
    def sample(entries, objectives):
        return pd.DataFrame([dict(entry_bps=h, model=m, scheme=s, mean_daily_cash_excess_return=value,
            holding_intervals=10, holding_interval_fraction=0., annualized_two_sided_turnover=0., asset_trade_records=0)
            for h, value in zip(entries, objectives) for m, s in sorted(STRATEGIES)])
    check("exact_tie_chooses_higher_threshold", rank_candidates(sample([5, 10, 30], [0., 0., 0.])).entry_bps.iloc[0] == 30,
          "A completely flat/cash grid ties in favor of the largest candidate")
    check("no_rounding_before_ranking", rank_candidates(sample([5, 30], [.1+1e-14, .1])).entry_bps.iloc[0] == 5,
          "Distinct unrounded objectives are not treated as ties")
    mixed = sample([5], [0.])
    mixed["mean_daily_cash_excess_return"] = [-.01, 0., 0., 0., 0., .004]
    check("six_strategies_equal_weight_including_zeros", np.isclose(
        rank_candidates(mixed).objective_mean_daily_cash_excess_return.iloc[0], -.001, rtol=0, atol=1e-18),
        "Weak and zero-trade strategies are not omitted from the public objective")
    for name, broken in [("reject_missing_strategy", mixed.iloc[:-1]),
                         ("reject_duplicate_strategy", pd.concat([mixed, mixed.iloc[:1]]))]:
        try:
            rank_candidates(broken)
        except ValueError:
            passed = True
        else:
            passed = False
        check(name, passed, "Invalid candidate comparison must stop")
    return pd.DataFrame(records)


def validate_candidate(book, assets, trades, summary, returns, opens, entry_bps):
    checks = []
    def check(name, passed, detail):
        checks.append(dict(entry_bps=entry_bps, check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(f"{entry_bps} bps: {name}")
    check("validation_only", book["split"].eq("validation").all()
          and book.date.between(pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-31")).all(),
          "No training diagnostic or test-period portfolio return")
    check("six_complete_strategies", set(zip(summary.model, summary.scheme)) == STRATEGIES and len(summary) == 6,
          "Same candidate applied to all six strategies")
    check("all_opens_and_intervals", len(book) == len(opens)*6 and len(returns) == (len(opens)-1)*6
          and summary.holding_intervals.eq(len(opens)-1).all(), "Cash dates cannot disappear")
    check("initial_capital_and_final_flat", book.groupby(["model", "scheme"]).head(1).nav_previous.eq(1).all()
          and book.groupby(["model", "scheme"]).tail(1).effective_action.eq("FINAL_LIQUIDATION").all()
          and book.groupby(["model", "scheme"]).tail(1).target_pairs.eq(0).all(),
          "Every strategy/candidate resets independently and closes on the scheduled final open")
    interval_keys = None
    for (model, scheme), group in book.groupby(["model", "scheme"]):
        ordered = group.sort_values("open_date")
        assert pd.DatetimeIndex(ordered.open_date).equals(opens)
        subset = returns.loc[returns.model.eq(model) & returns.scheme.eq(scheme)].sort_values("open_date")
        keys = list(zip(subset.interval_entry_date, subset.open_date))
        if interval_keys is None:
            interval_keys = keys
        assert keys == interval_keys
        assert np.isclose(np.prod(1+subset.holding_interval_return), ordered.nav_post.iloc[-1], rtol=1e-12, atol=1e-12)
        assert ordered.holding_return_denominator.iloc[1] == 1.
        final_assets = assets.loc[assets.model.eq(model) & assets.scheme.eq(scheme) & assets.open_date.eq(opens[-1])]
        assert final_assets.target_shares.eq(0).all()
    check("identical_return_keys_and_fee_inclusion", True, "Every return product reconstructs ending NAV including opening and liquidation fees")
    check("cash_excess_definition", np.array_equal(returns.cash_excess_return.to_numpy(),
        (returns.holding_interval_return-returns.interval_risk_free_return).to_numpy()),
        "Subtract the same exported cash interval return from each net return")
    check("nav_and_pnl_reconcile", book.pre_reconciliation_error.abs().le(1e-12).all()
          and book.post_reconciliation_error.abs().le(1e-12).all()
          and summary.accounting_residual.abs().le(1e-10).all(), "Full-phase cash/price/dividend/cost totals reconcile")
    check("at_most_four_pairs", book.target_pairs.between(0, 4).all()
          and book.target_gross_weight.between(0, 1).all(), "Existing pair and weight rules remain active")
    nonzero = assets.loc[assets.delta_shares.ne(0)]
    check("actual_net_trade_fees", np.allclose(nonzero.execution_fee,
        nonzero.delta_shares.abs()*nonzero.open_price*main_cost_parameters()["execution_bps"]/10000,
        rtol=1e-12, atol=1e-14) and len(nonzero) == len(trades), "One cost per actual net asset trade after SPY netting")
    check("zero_trade_reporting", all(
        r.holding_interval_fraction == 0 and r.annualized_two_sided_turnover == 0
        and pd.isna(r.annualized_cash_excess_sharpe)
        for r in summary.loc[summary.asset_trade_records.eq(0)].itertuples()),
        "Admissible cash-only cases retain N/A Sharpe and zero turnover/holding")
    return pd.DataFrame(checks)


def candidate_checkpoint(out, entry_bps, identity, frames=None):
    folder = out / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = folder / f"entry_{entry_bps:g}.json"
    if frames is None:
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["identity"] == identity
        result = {}
        for item in record["outputs"]:
            source = out / item["path"]
            assert sha256(source) == item["sha256"]
            frame = read_frame(source)
            for field in ("date", "open_date", "interval_entry_date", "first_open_date", "last_open_date"):
                if field in frame:
                    frame[field] = pd.to_datetime(frame[field])
            result[item["name"]] = frame
        return result
    records = []
    for name, frame in frames.items():
        target = folder / f"entry_{entry_bps:g}_{name}.csv"
        save_frame(target, frame)
        records.append(dict(name=name, path=target.relative_to(out).as_posix(), rows=len(frame), sha256=sha256(target)))
    write_json(path, dict(identity=identity, completed_utc=datetime.now(timezone.utc).isoformat(), outputs=records))
    return frames


def run_threshold_selection(project_root):
    root = Path(project_root).resolve()
    out = root / "results/backtest/threshold_selection"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    policy, config, identity = register_design(root, out)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_config_sha256"] == identity
        for item in manifest["outputs"]:
            assert sha256(root / item["path"]) == item["sha256"], f"Completed selection output changed: {item['path']}"
        print("Step 10: verified completed validation selection cache; no replay", flush=True)
        return manifest
    print("Step 10: design, main costs and fixed candidates recorded before replay", flush=True)
    signals, daily, prices, rates, opens = load_validation_inputs(root)
    logic_checks = validate_selection_logic()
    save_frame(out / "selection_logic_checks.csv", logic_checks)
    summaries, return_chunks, candidate_checks = [], [], []
    for entry in policy["entry_candidates_bps"]:
        saved = candidate_checkpoint(out, entry, identity)
        if saved is None:
            book, assets, trades = replay_ledger(signals, daily, prices, rates,
                h_in=entry/10000, initial_nav=1., tickers=TICKERS,
                require_final_liquidation=True, **policy["main_cost_parameters"])
            summary, returns = summarize_candidate(book, assets, trades, entry)
            for frame in (book, assets, trades):
                frame["entry_bps"], frame["exit_bps"] = entry, entry/2
                frame["scenario_id"], frame["selection_role"] = "main", "validation_threshold_search"
                frame["example_only"], frame["phase_complete"] = False, True
            saved = candidate_checkpoint(out, entry, identity, dict(
                daily=book, assets=assets, trades=trades, summary=summary, returns=returns))
        checks = validate_candidate(saved["daily"], saved["assets"], saved["trades"],
            saved["summary"], saved["returns"], opens, entry)
        summaries.append(saved["summary"])
        return_chunks.append(saved["returns"])
        candidate_checks.append(checks)
        print(f"Validation candidate {entry:g}/{entry/2:g} bps: six full replays complete | elapsed {time.perf_counter()-started:.1f}s", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    returns = pd.concat(return_chunks, ignore_index=True)
    scores = rank_candidates(summary)
    chosen = int(scores.entry_bps.iloc[0])
    verify_hashes(root, config["source_hashes"])
    verify_hashes(root, config["engine_hashes"])
    final_checks = pd.DataFrame([
        dict(check="official_validation_predictions_exact", passed=True,
             detail="Signal-time y_hat exactly matches the two original official prediction files for 2018-2022"),
        dict(check="fixed_five_candidates_all_six_strategies", passed=len(summary) == 30 and set(scores.entry_bps) == {5,10,15,20,30},
             detail="One public grid, with no per-model threshold selection"),
        dict(check="equal_interval_counts_across_all_candidates", passed=summary.holding_intervals.nunique() == 1,
             detail="All 30 strategies/candidates score identical-length validation intervals"),
        dict(check="selected_by_unrounded_registered_objective", passed=chosen == int(scores.loc[scores.selected, "entry_bps"].iloc[0])
             and scores.objective_mean_daily_cash_excess_return.iloc[0] == scores.objective_mean_daily_cash_excess_return.max(),
             detail="Maximum equal-weight mean daily cash excess; exact ties resolved by larger threshold"),
        dict(check="protected_sources_and_engine_unchanged", passed=True, detail="All source/engine hashes verified after replays"),
        dict(check="validation_only_no_sensitivity_search", passed=returns["split"].eq("validation").all()
             and returns.open_date.between(opens[0], opens[-1]).all(), detail="No test portfolio or alternative cost scenario evaluated")])
    checks = pd.concat([logic_checks, *candidate_checks, final_checks], ignore_index=True)
    assert checks.passed.all()
    summary["selected_public_threshold"] = summary.entry_bps.eq(chosen)
    for name, frame in [("validation_strategy_summary.csv", summary), ("validation_candidate_scores.csv", scores),
                        ("validation_daily_returns.csv", returns), ("selection_checks.csv", checks)]:
        save_frame(out / name, frame)
    selected = dict(schema_version=1, selection_split="validation", selected_public_entry_bps=chosen,
        selected_public_exit_bps=chosen/2, h_in=chosen/10000, h_out=chosen/20000,
        applies_to_all_six_strategies=True, model_predictions="original official validation/test forecasts; no diagnostic refit substitution",
        main_cost_parameters=policy["main_cost_parameters"], objective=policy["objective"], tie_rule=policy["tie_rule"],
        winning_mean_daily_cash_excess_return=float(scores.objective_mean_daily_cash_excess_return.iloc[0]),
        design_sha256=config["design_sha256"], run_config_sha256=identity,
        candidate_scores_sha256=sha256(out / "validation_candidate_scores.csv"),
        strategy_summary_sha256=sha256(out / "validation_strategy_summary.csv"),
        test_strategy_evaluated=False, protocol_freeze_pending=True)
    selected_path = out / "selected_threshold.json"
    if selected_path.exists():
        assert json.loads(selected_path.read_text(encoding="utf-8")) == selected, "Previously selected threshold changed"
    else:
        write_json(selected_path, selected)
    files = [p for p in out.rglob("*") if p.is_file() and p != manifest_path]
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p)) for p in sorted(files)]
    manifest = dict(schema_version=1, scope="Step 10: complete validation-only public threshold selection",
        completed_utc=datetime.now(timezone.utc).isoformat(), run_config_sha256=identity,
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__,
        selected_public_entry_bps=chosen, selected_public_exit_bps=chosen/2,
        threshold_selection_performed=True, candidate_count=len(scores), strategy_candidate_count=len(summary),
        candidate_entry_bps=policy["entry_candidates_bps"], main_cost_parameters=policy["main_cost_parameters"],
        official_validation_signal_rows=len(signals), opening_records_per_strategy=len(opens),
        holding_intervals_per_strategy=len(opens)-1, daily_return_rows=len(returns),
        first_open_date=str(opens[0].date()), last_open_date=str(opens[-1].date()),
        training_diagnostic_forecasts_used=False, model_retraining_performed=False,
        test_strategy_evaluated=False, sensitivity_scenarios_evaluated=False, protocol_freeze_pending=True,
        checks=checks.to_dict("records"), source_hashes=config["source_hashes"], code_hashes=config["engine_hashes"],
        outputs=outputs, elapsed_seconds=time.perf_counter()-started,
        notes=["Selection results are validation tuning evidence, not a new untouched performance estimate",
               "Daily net returns include initial/final fees and all cash days; cash excess subtracts the same-interval exported RF return",
               "The previous 10-bps engineering example was provisional and did not determine this result",
               "All original model, forecast, training-OOF and protocol files remain unchanged",
               "Checkpoint CSVs retain full per-candidate daily/share/trade ledgers; completed cache is verified without replay"],
        not_yet_checked=["freeze the full backtest protocol and use the selected public threshold for independent phase evaluation (step 11)",
                         "SPY/cash investment benchmarks, full reporting and frozen cost sensitivities (steps 12-14)"])
    # Concatenated check tables have a non-applicable entry_bps for logic checks.
    manifest["checks"] = [{k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
                          for row in manifest["checks"]]
    write_json(manifest_path, manifest)
    print(f"PASS: {len(checks)} selection checks; public entry/exit {chosen:g}/{chosen/2:g} bps; test not evaluated", flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-logic", action="store_true")
    args = parser.parse_args()
    if args.check_logic:
        print(validate_selection_logic().to_string(index=False))
    else:
        run_threshold_selection(args.project_root)
