"""Step 14: full metrics, comparison figures and frozen cost sensitivities.

Reuse all accepted main-cost strategy/benchmark books. Replay only the seven
previously specified alternative cost cases, preserving complete checkpoints.
No model fit, threshold search or performance-based cost selection is performed.
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
    from .backtest_costs import scenario_records, PARAMETER_FIELDS
    from .backtest_ledger import replay_ledger
    from .backtest_phase_evaluation import load_phase_inputs, read_cached_frame
    from .backtest_threshold_selection import TICKERS, STRATEGIES, read_frame, save_frame, sha256, verify_hashes, write_json
else:
    from backtest_costs import scenario_records, PARAMETER_FIELDS
    from backtest_ledger import replay_ledger
    from backtest_phase_evaluation import load_phase_inputs, read_cached_frame
    from backtest_threshold_selection import TICKERS, STRATEGIES, read_frame, save_frame, sha256, verify_hashes, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check(records, phase, scenario, name, passed, detail=""):
    records.append(dict(phase=phase, scenario_id=scenario, check=name, passed=bool(passed), detail=detail))
    if not passed:
        raise AssertionError(f"{phase}/{scenario}: {name}")


def register_design(root, out):
    prior_path = root / "results/backtest/consistency_audit/run_manifest.json"
    prior = read_json(prior_path)
    assert prior["status"] == "passed" and all(c["passed"] for c in prior["checks"])
    verify_hashes(root, prior["source_hashes"])
    verify_hashes(root, prior["code_hashes"])
    verify_hashes(root, {x["path"]: x["sha256"] for x in prior["outputs"]})
    protocol = read_json(root / "results/backtest/phase_evaluation/protocol_config.json")
    assert protocol["status"] == "frozen" and protocol["threshold"]["h_in"] == .0015
    scenarios = scenario_records()
    assert scenarios == protocol["deferred"]["cost_scenarios_step14"]
    policy = dict(schema_version=1, scope="Step 14: full metrics, plots and eight prespecified cost cases",
        phase_protocol_sha256=sha256(root / "results/backtest/phase_evaluation/protocol_config.json"),
        phase_opens=protocol["phase_opens"], initial_nav=1., h_in=.0015, h_out=.00075,
        threshold_source_sha256=protocol["threshold"]["source_sha256"], scenarios=scenarios,
        main_reused=True, alternative_cost_cases=7, alternative_strategy_paths=84,
        replay="each alternative cost case independently resets cash/shares by strategy and phase; require final liquidation",
        engine="existing replay_ledger; scenario_records supplies settings; engineering run_cost_scenarios wrapper is not used because its flags describe previews",
        statistics=protocol["statistics"], full_metrics="net total/CAGR, sample annualized net volatility, cash-excess Sharpe, signed maximum drawdown, two-sided turnover, holding fraction and fees/financing",
        drawdown="include capital 1 before initial entry fee plus every open NAV_post; same-date initial pre/post points retained",
        annual_returns="compound actual intervals grouped by ending-open year; no annualization; phase boundaries apply and 2026 ends September 30",
        sensitivity="same forecasts, pair decisions and threshold; NAV-dependent shares/cash/fees are replayed, not subtracted from a fixed gross curve",
        sensitivity_comparison="cash-excess comparison uses unchanged cash RF; SPY investment benchmark is shown only for the main-cost comparison",
        reporting="all six strategies and both main investment benchmarks, each phase separately; no omission of weak results",
        evidence_boundary=protocol["evidence_boundary"], selection_by_sensitivity_performance=False,
        original_results_read_only=True, model_retraining=False, threshold_reselection=False, steps_15_to_16_executed=False)
    path = out / "design_config.json"
    if path.exists():
        saved = read_json(path)
        assert {k: v for k, v in saved.items() if k != "registered_utc"} == policy, "Reporting design changed"
        policy = saved
    else:
        policy["registered_utc"] = utc_now()
        write_json(path, policy)
    protected = dict(prior["source_hashes"])
    protected.update({x["path"]: x["sha256"] for x in prior["outputs"]})
    protected[prior_path.relative_to(root).as_posix()] = sha256(prior_path)
    codes = dict(prior["code_hashes"])
    codes["scripts/backtest_reporting.py"] = sha256(Path(__file__))
    config = dict(design_sha256=sha256(path), source_hashes=protected, code_hashes=codes,
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__)
    path = out / "run_config.json"
    if path.exists():
        assert read_json(path) == config, "Numeric reporting source/code/runtime changed; incompatible cache"
    else:
        write_json(path, config)
    return policy, config, sha256(path)


def path_statistics(group):
    """Frozen metric definitions, including the initial fee in drawdown and return."""
    b = group.sort_values("open_date").reset_index(drop=True)
    interval = b.loc[~b.is_initial_open].copy()
    n = len(interval)
    if n < 2:
        raise ValueError("At least two holding intervals required")
    net = interval.holding_interval_return.to_numpy(float)
    rf = interval.interval_risk_free_return.to_numpy(float)
    excess = net-rf
    std = float(np.std(excess, ddof=1))
    nav = np.r_[1., b.nav_post.to_numpy(float)]
    drawdown = nav/np.maximum.accumulate(nav)-1
    trade_column = "traded_notional" if "traded_notional" in b else "trade_notional"
    if "target_pairs" in b:
        held = b.target_pairs.shift().iloc[1:].gt(0)
        mean_pairs = float(b.target_pairs.shift().iloc[1:].mean())
    else:
        held = b.shares_post.shift().iloc[1:].ne(0)
        mean_pairs = float("nan")
    turnover = float((b[trade_column]/b.nav_pre).sum())*252/n
    ending = float(b.nav_post.iloc[-1])
    cash_nav = float(np.prod(1+rf))
    result = dict(opening_records=len(b), holding_intervals=n, first_open_date=b.open_date.iloc[0],
        last_open_date=b.open_date.iloc[-1], ending_nav=ending, net_cumulative_return=ending-1,
        annualized_net_return=float(np.expm1(np.log(ending)*252/n)),
        annualized_net_volatility=float(np.std(net, ddof=1)*np.sqrt(252)),
        mean_daily_cash_excess_return=float(excess.mean()), mean_daily_cash_excess_bps=float(excess.mean()*10000),
        annualized_cash_excess_sharpe=float(np.sqrt(252)*excess.mean()/std) if std > 1e-12 else float("nan"),
        max_drawdown=float(drawdown.min()), annualized_two_sided_turnover=turnover,
        holding_interval_fraction=float(held.mean()), mean_pairs_held=mean_pairs,
        total_execution_fee=float(b.execution_fee.sum()), total_borrow_fee=float(b.borrow_fee.sum()),
        total_dividend_cash=float(b.dividend_cash.sum()), total_cash_interest=float(b.cash_interest.sum()),
        total_price_pnl=float(b.price_pnl.sum()), cash_ending_nav=cash_nav, ending_relative_to_cash=ending/cash_nav-1)
    interval["cash_excess_return"] = excess
    return result, interval, drawdown


def metric_logic_checks():
    records = []
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    b = pd.DataFrame(dict(open_date=dates, is_initial_open=[True, False, False], nav_post=[.99, .995, 1.],
        holding_interval_return=[np.nan, -.005, 1/.995-1], interval_risk_free_return=[0., 0., 0.],
        traded_notional=[.5, 0., .5], nav_pre=[1., .995, 1.], target_pairs=[1, 1, 0],
        execution_fee=[.01, 0., 0.], borrow_fee=[0., 0., 0.], dividend_cash=[0., 0., 0.],
        cash_interest=[0., 0., 0.], price_pnl=[0., .005, .005]))
    metric, interval, dd = path_statistics(b)
    check(records, "fixture", "metric", "initial_fee_drawdown_precedes_first_return", np.isclose(metric["max_drawdown"], -.01)
          and len(dd) == 4, "Using only real-interval cumulative returns would miss the initial -1% drawdown")
    check(records, "fixture", "metric", "no_initial_zero_length_return", metric["holding_intervals"] == 2
          and np.isclose(np.prod(1+interval.holding_interval_return), 1.))
    check(records, "fixture", "metric", "sample_net_volatility", np.isclose(metric["annualized_net_volatility"],
          np.std([-.005, 1/.995-1], ddof=1)*np.sqrt(252)))
    check(records, "fixture", "metric", "holding_uses_interval_start", metric["holding_interval_fraction"] == 1.)
    check(records, "fixture", "metric", "two_sided_turnover_keeps_both_ends", np.isclose(metric["annualized_two_sided_turnover"], 126.))
    cash = b.assign(nav_post=1., holding_interval_return=[np.nan, 0., 0.], target_pairs=0,
        execution_fee=0., traded_notional=0., price_pnl=0.)
    m, _, _ = path_statistics(cash)
    check(records, "fixture", "metric", "zero_variance_cash_sharpe_na", pd.isna(m["annualized_cash_excess_sharpe"])
          and m["max_drawdown"] == 0 and m["annualized_net_volatility"] == 0)
    return records


def strategy_statistics(book, scenario):
    rows, intervals = [], []
    for (model, scheme, phase), group in book.groupby(["model", "scheme", "split"], sort=True):
        metric, interval, _ = path_statistics(group)
        identity = dict(split=phase, kind="strategy", series_id=f"{model}_{scheme}", model=model, scheme=scheme,
                        scenario_id=scenario["scenario_id"], **{k: scenario[k] for k in PARAMETER_FIELDS})
        rows.append(identity | metric)
        for key, value in identity.items():
            interval[key] = value
        intervals.append(interval)
    return pd.DataFrame(rows), pd.concat(intervals, ignore_index=True)


def validate_scenario(frames, main, inputs, scenario, checks):
    book, assets, trades = (frames[n] for n in ("daily", "assets", "trades"))
    phase = book["split"].iloc[0]; sid = scenario["scenario_id"]; opens = inputs[4]
    def test(name, passed, detail=""):
        check(checks, phase, sid, name, passed, detail)
    test("six_complete_phase_paths", set(zip(book.model, book.scheme)) == STRATEGIES
         and len(book) == len(opens)*6 and len(assets) == len(book)*10)
    test("cash_reset_and_final_liquidation", book.groupby(["model", "scheme"]).head(1).cash_prev.eq(1).all()
         and book.groupby(["model", "scheme"]).head(1).nav_previous.eq(1).all()
         and book.groupby(["model", "scheme"]).tail(1).effective_action.eq("FINAL_LIQUIDATION").all()
         and assets.loc[assets.open_date.eq(opens[-1]), "target_shares"].eq(0).all())
    weights = ["model", "scheme", "open_date", "ticker", "target_weight"]
    pd.testing.assert_frame_equal(assets[weights].sort_values(weights[:-1]).reset_index(drop=True),
        main["assets"][weights].sort_values(weights[:-1]).reset_index(drop=True), check_exact=True)
    test("same_pairs_and_weights_as_main", True, "Only NAV-dependent share amounts change; no threshold or pair-rule retuning")
    test("actual_net_execution_fee", np.allclose(assets.execution_fee, assets.delta_shares.abs()*assets.open_price*scenario["execution_bps"]/10000,
         rtol=1e-12, atol=1e-14) and len(trades) == int(assets.delta_shares.ne(0).sum()))
    base = book.cash_prev if scenario["full_cash_rebate"] else np.maximum(book.cash_prev-book.short_collateral_start, 0)
    test("scenario_cash_interest", np.allclose(book.cash_interest, base*book.interval_risk_free_return, rtol=1e-12, atol=1e-14))
    test("scenario_natural_day_borrow", np.allclose(book.borrow_fee,
         book.short_collateral_start*scenario["borrow_annual_rate"]*book.holding_calendar_days/365, rtol=1e-12, atol=1e-14))
    test("nav_identities", book.pre_reconciliation_error.abs().le(1e-12).all() and book.post_reconciliation_error.abs().le(1e-12).all())
    all_products, all_pnl = [], []
    for _, group in book.groupby(["model", "scheme"]):
        group = group.sort_values("open_date")
        assert pd.DatetimeIndex(group.open_date).equals(opens)
        ret = group.holding_interval_return.iloc[1:]
        all_products.append(np.isclose(np.prod(1+ret), group.nav_post.iloc[-1], rtol=1e-12, atol=1e-12))
        end = 1+group.price_pnl.sum()+group.dividend_cash.sum()+group.cash_interest.sum()-group.borrow_fee.sum()-group.execution_fee.sum()
        all_pnl.append(np.isclose(end, group.nav_post.iloc[-1], rtol=1e-12, atol=1e-10))
    test("all_intervals_and_initial_final_fees", all(all_products) and len(frames["returns"]) == (len(opens)-1)*6)
    test("full_phase_pnl_reconstruction", all(all_pnl))
    test("frozen_threshold_and_only_current_scenario", book.entry_bps.eq(15).all() and book.exit_bps.eq(7.5).all()
         and book.scenario_id.eq(sid).all())


def scenario_checkpoint(root, out, phase, scenario, identity, frames=None):
    folder = out / "sensitivity/checkpoints"; folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{phase}_{scenario['scenario_id']}"
    path = folder / f"{prefix}.json"
    if frames is None:
        if not path.exists():
            return None
        record = read_json(path)
        assert record["run_config_sha256"] == identity and record["scenario"] == scenario
        result = {}
        for item in record["outputs"]:
            source = root / item["path"]
            assert sha256(source) == item["sha256"]
            result[item["name"]] = read_cached_frame(source)
            assert len(result[item["name"]]) == item["rows"]
        return result
    outputs = []
    for name, frame in frames.items():
        target = folder / f"{prefix}_{name}.csv"; save_frame(target, frame)
        outputs.append(dict(name=name, path=target.relative_to(root).as_posix(), rows=len(frame), sha256=sha256(target)))
    write_json(path, dict(run_config_sha256=identity, scenario=scenario, completed_utc=utc_now(), outputs=outputs))
    return frames


def main_frames(root, phase):
    prefix = "threshold_selection/checkpoints/entry_15_" if phase == "validation" else "phase_evaluation/test_"
    frames = {name: read_cached_frame(root / f"results/backtest/{prefix}{name}.csv") for name in ("daily", "assets", "trades")}
    return frames, prefix


def main_details(book, identity):
    metric, intervals, dd = path_statistics(book)
    b = book.sort_values("open_date").reset_index(drop=True)
    curve = pd.DataFrame(dict(open_date=[b.open_date.iloc[0], *b.open_date], point_order=np.arange(len(b)+1),
        nav=np.r_[1., b.nav_post], drawdown=dd, point_role=["initial_capital", *(["post_open"]*len(b))]))
    cash_nav = [1., 1.]
    for rf in intervals.interval_risk_free_return:
        cash_nav.append(cash_nav[-1]*(1+rf))
    curve["cash_reference_nav"] = cash_nav
    curve["nav_relative_to_cash"] = curve.nav/curve.cash_reference_nav
    years = []
    for year, group in intervals.groupby(intervals.open_date.dt.year):
        years.append(identity | dict(calendar_year=int(year), holding_intervals=len(group),
            first_interval_entry=group.interval_entry_date.iloc[0], last_interval_exit=group.open_date.iloc[-1],
            net_return=float(np.prod(1+group.holding_interval_return)-1), cash_return=float(np.prod(1+group.interval_risk_free_return)-1),
            mean_daily_cash_excess_bps=float(group.cash_excess_return.mean()*10000),
            boundary_note="phase boundary applies" if year in (b.open_date.iloc[0].year, b.open_date.iloc[-1].year) else "internal year"))
    for key, value in identity.items():
        curve[key] = value
    decomposition = identity | dict(price_pnl=metric["total_price_pnl"], dividends=metric["total_dividend_cash"],
        cash_interest=metric["total_cash_interest"], borrow_fee=metric["total_borrow_fee"], execution_fee=metric["total_execution_fee"],
        ending_nav=metric["ending_nav"], net_pnl=metric["ending_nav"]-1)
    return identity | metric, curve, years, decomposition


def run_numeric_reporting(root):
    root = Path(root).resolve(); out = root / "results/backtest/reporting"; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); policy, config, identity = register_design(root, out)
    path = out / "numeric_manifest.json"
    if path.exists():
        manifest = read_json(path)
        assert manifest["run_config_sha256"] == identity
        verify_hashes(root, {x["path"]: x["sha256"] for x in manifest["outputs"]})
        print("Step 14: verified completed numeric reporting cache; no cost replay", flush=True)
        return manifest
    checks = metric_logic_checks(); save_frame(out / "logic_checks.csv", pd.DataFrame(checks))
    save_frame(out / "cost_scenarios.csv", pd.DataFrame(policy["scenarios"]))
    main_rows, curves, annual, pnl, sensitivity, reuse = [], [], [], [], [], []
    accepted = read_frame(root / "results/backtest/phase_evaluation/phase_summary.csv")
    main_scenario = policy["scenarios"][0]
    for phase in policy["phase_opens"]:
        inputs, input_checks = load_phase_inputs(root, phase)
        checks.extend(dict(phase=phase, scenario_id="inputs", check=x["check"], passed=x["passed"], detail=x["detail"]) for x in input_checks)
        main, prefix = main_frames(root, phase)
        main_metrics, main_returns = strategy_statistics(main["daily"], main_scenario)
        main.update(metrics=main_metrics, returns=main_returns)
        sensitivity.append(main_metrics)
        validate_scenario(main, main, inputs, main_scenario, checks)
        for (model, scheme), group in main["daily"].groupby(["model", "scheme"], sort=True):
            ident = dict(split=phase, kind="strategy", series_id=f"{model}_{scheme}", model=model, scheme=scheme)
            row, curve, years, decomposition = main_details(group, ident)
            main_rows.append(row); curves.append(curve); annual.extend(years); pnl.append(decomposition)
            original = accepted.loc[accepted["split"].eq(phase) & accepted.model.eq(model) & accepted.scheme.eq(scheme)].iloc[0]
            fields = ["ending_nav", "annualized_net_return", "annualized_cash_excess_sharpe", "annualized_two_sided_turnover", "holding_interval_fraction"]
            check(checks, phase, "main", f"main_metrics_match_prior_{model}_{scheme}", np.allclose([row[f] for f in fields],
                [original[f] for f in fields], rtol=1e-12, atol=1e-12, equal_nan=True))
        reuse.append(dict(phase=phase, source_prefix=prefix, original_main_replayed=False, daily_rows=len(main["daily"]), asset_rows=len(main["assets"])))
        print(f"Step 14: {phase} main six paths reused; full metrics calculated", flush=True)
        for scenario in policy["scenarios"][1:]:
            frames = scenario_checkpoint(root, out, phase, scenario, identity)
            if frames is None:
                book, assets, trades = replay_ledger(*inputs[:4], h_in=policy["h_in"], initial_nav=1., tickers=TICKERS,
                    require_final_liquidation=True, **{k: scenario[k] for k in PARAMETER_FIELDS})
                metrics, returns = strategy_statistics(book, scenario)
                for frame in (book, assets, trades):
                    frame["entry_bps"], frame["exit_bps"] = 15, 7.5
                    for k in ("scenario_id", *PARAMETER_FIELDS):
                        frame[k] = scenario[k]
                    frame["example_only"], frame["phase_complete"] = False, True
                frames = scenario_checkpoint(root, out, phase, scenario, identity,
                    dict(daily=book, assets=assets, trades=trades, returns=returns, metrics=metrics))
                action = "replayed"
            else:
                action = "verified checkpoint"
            validate_scenario(frames, main, inputs, scenario, checks)
            sensitivity.append(frames["metrics"])
            print(f"Step 14: {phase}/{scenario['scenario_id']} six paths {action} | elapsed {time.perf_counter()-started:.1f}s", flush=True)
    benchmark_book = read_cached_frame(root / "results/backtest/investment_benchmarks/benchmark_daily.csv")
    benchmark_prior = read_frame(root / "results/backtest/investment_benchmarks/benchmark_summary.csv")
    for (phase, benchmark), group in benchmark_book.groupby(["split", "benchmark"], sort=True):
        ident = dict(split=phase, kind="benchmark", series_id=benchmark, model="", scheme="")
        row, curve, years, decomposition = main_details(group, ident)
        original = benchmark_prior.loc[benchmark_prior["split"].eq(phase) & benchmark_prior.benchmark.eq(benchmark)].iloc[0]
        fields = ["ending_nav", "annualized_net_return", "annualized_cash_excess_sharpe", "annualized_two_sided_turnover"]
        check(checks, phase, "benchmark", f"metrics_match_prior_{benchmark}", np.allclose([row[f] for f in fields],
              [original[f] for f in fields], rtol=1e-12, atol=1e-12, equal_nan=True))
        main_rows.append(row); curves.append(curve); annual.extend(years); pnl.append(decomposition)
    full = pd.concat(sensitivity, ignore_index=True)
    base = full.loc[full.scenario_id.eq("main"), ["split", "model", "scheme", "annualized_net_return", "annualized_cash_excess_sharpe", "ending_nav"]]
    full = full.merge(base, on=["split", "model", "scheme"], validate="many_to_one", suffixes=("", "_main"))
    for field in ("annualized_net_return", "annualized_cash_excess_sharpe", "ending_nav"):
        full[f"delta_{field}_vs_main"] = full[field]-full[field+"_main"]
    robust = full.groupby(["split", "model", "scheme"], as_index=False).agg(
        scenarios=("scenario_id", "size"), min_annualized_net_return=("annualized_net_return", "min"), max_annualized_net_return=("annualized_net_return", "max"),
        min_cash_excess_sharpe=("annualized_cash_excess_sharpe", "min"), max_cash_excess_sharpe=("annualized_cash_excess_sharpe", "max"))
    check(checks, "both", "all", "all_ninety_six_prespecified_strategy_cases", len(full) == 96 and robust.scenarios.eq(8).all())
    check(checks, "both", "all", "all_sixteen_main_paths_reported", len(main_rows) == 16 and len(annual) == 72)
    check(checks, "both", "all", "one_factor_settings_match_frozen_policy", scenario_records() == policy["scenarios"])
    verify_hashes(root, config["source_hashes"]); verify_hashes(root, config["code_hashes"])
    check(checks, "both", "all", "all_previous_registered_inputs_results_and_code_unchanged", True)
    write_json(out / "main_reuse.json", dict(run_config_sha256=identity, main_strategy_sources=reuse,
        benchmark_source="results/backtest/investment_benchmarks/benchmark_daily.csv", original_main_replayed=False))
    for name, frame in (("main_metrics.csv", pd.DataFrame(main_rows)), ("main_curves.csv", pd.concat(curves, ignore_index=True)),
        ("calendar_year_returns.csv", pd.DataFrame(annual)), ("pnl_decomposition.csv", pd.DataFrame(pnl)),
        ("sensitivity_metrics.csv", full), ("cost_robustness.csv", robust), ("numeric_checks.csv", pd.DataFrame(checks))):
        save_frame(out / name, frame)
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p)) for p in sorted(out.rglob("*")) if p.is_file() and p != path]
    manifest = dict(schema_version=1, scope=policy["scope"], status="passed", completed_utc=utc_now(),
        design_registered_utc=policy["registered_utc"], run_config_sha256=identity, checks=checks,
        protected_file_count=len(config["source_hashes"]), source_hashes=config["source_hashes"], code_hashes=config["code_hashes"],
        python_version=config["python_version"], numpy_version=config["numpy_version"], pandas_version=config["pandas_version"],
        main_paths_reused=16, sensitivity_strategy_paths=96, new_alternative_strategy_paths=84, new_phase_scenario_batches=14,
        model_retraining=False, threshold_reselection=False, main_strategy_replay=False, cost_settings_selected_by_performance=False,
        outputs=outputs, elapsed_seconds=time.perf_counter()-started,
        notes=["Zero execution fee still has the main 1% annual borrow and collateral policy",
            "All sensitivity scenarios independently replay NAV-dependent shares and financing; no fixed gross-curve cost subtraction",
            "Eight scenario extrema describe a fixed set of assumptions and do not select a new main model or cost setting",
            "Validation is tuning evidence; earlier test prediction evaluation and exploratory provenance remain disclosed"])
    write_json(path, manifest)
    print(f"PASS: {len(checks)} numeric reporting checks; 16 main paths reused and 84 new alternative paths completed", flush=True)
    return manifest


def run_reporting(project_root, numeric_only=False):
    root = Path(project_root).resolve(); numeric = run_numeric_reporting(root)
    if numeric_only:
        return numeric
    out = root / "results/backtest/reporting"; path = out / "run_manifest.json"
    renderer = root / "scripts/backtest_report_figures.py"
    if path.exists():
        manifest = read_json(path)
        assert manifest["run_config_sha256"] == numeric["run_config_sha256"]
        assert manifest["code_hashes"]["scripts/backtest_report_figures.py"] == sha256(renderer)
        verify_hashes(root, {x["path"]: x["sha256"] for x in manifest["outputs"]})
        print("Step 14: verified complete tables, figures and report cache; no replay or rendering", flush=True)
        return manifest
    if __package__:
        from .backtest_report_figures import render_report
    else:
        from backtest_report_figures import render_report
    rendered = render_report(root, out)
    manifest = dict(numeric)
    manifest.update(completed_utc=utc_now(), code_hashes=numeric["code_hashes"] | {"scripts/backtest_report_figures.py": sha256(renderer)},
        checks=numeric["checks"]+rendered["checks"], figures=rendered["figures"], matplotlib_version=rendered["matplotlib_version"],
        report="results/backtest/reporting/report_中文.md", numeric_manifest_sha256=sha256(out / "numeric_manifest.json"),
        not_yet_checked=["step 15: final package and handoff", "step 16: optional extensions only if separately requested"])
    manifest["outputs"] = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p)) for p in sorted(out.rglob("*")) if p.is_file() and p != path]
    write_json(path, manifest)
    print(f"PASS: {len(manifest['checks'])} total step-14 checks; {len(rendered['figures'])} comparison figures and Chinese report saved", flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--check-logic", action="store_true")
    args = parser.parse_args()
    if args.check_logic:
        print(pd.DataFrame(metric_logic_checks()).to_string(index=False))
    else:
        run_reporting(args.project_root, numeric_only=args.numeric_only)
