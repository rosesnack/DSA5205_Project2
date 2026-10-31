"""Prepare handed-over forecasts and run explicitly selected backtest stages.

Run from the project root: python scripts/prepare_backtest_inputs.py
The default forecast intake uses only Python's standard library and preserves
forecast text exactly. Add --stage panel for the pandas-based panel/time checks,
following the conventions in notebooks 02-04. Add --stage eligibility for the
step-3 signal/scoring separation and missing-signal policy checks. Add --stage
baselines to also receive and verify the two forecast baselines (step 4). Add
--stage decisions to validate pair decisions and create an explicitly provisional
five-signal-day validation example at 10 bps (step 5). Add --stage execution for
target weights, net share trade planning and normalized target examples (step 6).
Add --stage ledger to verify cash/share/NAV accounting and replay the same limited
engineering preview (step 7), with explicit documented cost parameters.
Add --stage costs to export the shared protocol cost policy and verify all eight
prespecified one-factor comparisons using market and synthetic engineering inputs.
Add --stage oof for the separate training-only forward diagnostic (step 9),
reusing hash-checked monthly checkpoints and leaving official forecasts intact.
Add --stage thresholds for validation-only public threshold selection (step 10),
using the original official forecasts and main costs; requires completed step 9.
Add --stage phases to freeze the complete protocol, reuse the selected validation
ledger and evaluate all six official test strategies under main costs (step 11).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


FIELDS = ("date", "ticker", "model", "scheme", "model_month", "fit_cutoff", "y_hat")
MODELS = ("elastic_net", "gbdt")
SCHEMES = ("S0", "S1", "S2")
TICKERS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_forecasts(project_root: Path) -> tuple[list[dict[str, str]], dict]:
    """Load all six strategies; reject incomplete or inconsistent handoffs.

    This checks the forecast interface only. Panel eligibility, execution-date
    alignment and historical fitting cutoffs require the next integration step.
    """
    project_root = project_root.resolve()
    rows: list[dict[str, str]] = []
    sources = []
    seen = set()
    coverage = defaultdict(set)
    dates_by_strategy = defaultdict(set)
    model_keys = defaultdict(set)
    parsed_dates = {}
    parsed_cutoffs = {}

    for model in MODELS:
        path = project_root / "results" / model / f"{model}_predictions.csv"
        before_hash = sha256(path)
        source_rows = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{path.name}: expected exactly these columns: {FIELDS}")
            for line_number, row in enumerate(reader, start=2):
                context = f"{path.name}:{line_number}"
                if None in row or any(row[field] in (None, "") for field in FIELDS):
                    raise ValueError(f"{context}: missing values or malformed CSV row")
                if row["model"] != model:
                    raise ValueError(f"{context}: unexpected model {row['model']!r}")
                if row["scheme"] not in SCHEMES or row["ticker"] not in TICKERS:
                    raise ValueError(f"{context}: unexpected scheme or ticker")

                signal_date = row["date"]
                if signal_date not in parsed_dates:
                    parsed = date.fromisoformat(signal_date)
                    if parsed.isoformat() != signal_date:
                        raise ValueError(f"{context}: date must be YYYY-MM-DD")
                    parsed_dates[signal_date] = parsed
                if row["model_month"] != signal_date[:7]:
                    raise ValueError(f"{context}: model_month differs from signal month")
                cutoff_text = row["fit_cutoff"]
                if cutoff_text not in parsed_cutoffs:
                    cutoff = datetime.fromisoformat(cutoff_text)
                    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                        raise ValueError(f"{context}: fit_cutoff must include a timezone")
                    parsed_cutoffs[cutoff_text] = cutoff
                if not math.isfinite(float(row["y_hat"])):
                    raise ValueError(f"{context}: y_hat is not finite")

                key = tuple(row[field] for field in ("date", "ticker", "model", "scheme"))
                if key in seen:
                    raise ValueError(f"{context}: duplicate forecast key {key}")
                seen.add(key)
                strategy = (model, row["scheme"])
                coverage[(*strategy, signal_date)].add(row["ticker"])
                dates_by_strategy[strategy].add(signal_date)
                model_keys[model].add((signal_date, row["ticker"], row["scheme"]))
                rows.append(row)
                source_rows += 1

        if sha256(path) != before_hash:
            raise ValueError(f"{path.name}: source changed while being read")
        if not source_rows:
            raise ValueError(f"{path.name}: no forecast rows")
        sources.append({
            "path": path.relative_to(project_root).as_posix(),
            "sha256": before_hash,
            "bytes": path.stat().st_size,
            "rows": source_rows,
        })

    required_strategies = {(model, scheme) for model in MODELS for scheme in SCHEMES}
    if set(dates_by_strategy) != required_strategies:
        raise ValueError("The handoff must contain all six model/scheme strategies")
    for key, tickers in coverage.items():
        if tickers != set(TICKERS):
            raise ValueError(f"{key}: incomplete nine-sector coverage")
    reference_dates = dates_by_strategy[(MODELS[0], SCHEMES[0])]
    if any(dates != reference_dates for dates in dates_by_strategy.values()):
        raise ValueError("The six strategies cover different signal dates")
    if model_keys[MODELS[0]] != model_keys[MODELS[1]]:
        raise ValueError("Model exports have different date/ticker/scheme keys")

    ticker_order = {ticker: index for index, ticker in enumerate(TICKERS)}
    rows.sort(key=lambda row: (
        row["date"], row["model"], row["scheme"], ticker_order[row["ticker"]]
    ))
    summary = []
    for model in MODELS:
        for scheme in SCHEMES:
            dates = dates_by_strategy[(model, scheme)]
            summary.append({
                "model": model,
                "scheme": scheme,
                "prediction_rows": len(dates) * len(TICKERS),
                "signal_dates": len(dates),
                "sectors_per_date": len(TICKERS),
                "first_signal_date": min(dates),
                "last_signal_date": max(dates),
            })

    audit = {
        "schema_version": 1,
        "scope": "Step 1: forecast handoff intake only; no strategy backtest",
        "sources": sources,
        "strategy_count": len(summary),
        "combined_prediction_rows": len(rows),
        "strategies": summary,
        "checks_passed": [
            "exact seven-column forecast schema",
            "expected model, scheme and sector identifiers",
            "unique date/ticker/model/scheme keys",
            "ISO signal dates and matching model_month",
            "timezone-aware fit_cutoff syntax",
            "finite numeric y_hat values",
            "all nine sectors on every exported strategy/date",
            "identical signal-date coverage across all six strategies",
            "identical date/ticker/scheme keys across both models",
            "input hashes unchanged during reading",
        ],
        "not_yet_checked": [
            "shared-panel joins and expected eligible dates",
            "fit_cutoff chronology and New York execution timestamps",
            "signal-day versus execution-day mapping",
            "trading eligibility and scoring eligibility",
            "price, dividend and risk-free interval alignment",
        ],
    }
    return rows, audit


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(project_root: Path) -> dict:
    project_root = project_root.resolve()
    rows, audit = load_forecasts(project_root)
    output = project_root / "results" / "backtest"
    output.mkdir(parents=True, exist_ok=True)
    forecasts_path = output / "combined_predictions.csv"
    coverage_path = output / "prediction_coverage.csv"
    write_csv(forecasts_path, FIELDS, rows)
    write_csv(coverage_path, tuple(audit["strategies"][0]), audit["strategies"])

    # Check the serialized artifact, including exact forecast and timestamp text.
    with forecasts_path.open("r", encoding="utf-8", newline="") as stream:
        restored = list(csv.DictReader(stream))
    if restored != rows:
        raise ValueError("Combined forecast export changed the source field values")
    audit["checks_passed"].append("exact seven-field text roundtrip after export")
    for source in audit["sources"]:
        if sha256(project_root / source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed during export: {source['path']}")
    audit["checks_passed"].append("source forecast hashes unchanged after export")
    audit["created_utc"] = datetime.now(timezone.utc).isoformat()
    audit["python_version"] = platform.python_version()
    audit["outputs"] = [
        {"path": path.relative_to(project_root).as_posix(), "sha256": sha256(path)}
        for path in (forecasts_path, coverage_path)
    ]
    (output / "input_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def save_frame(frame, name, output: Path, root: Path) -> dict:
    """Use notebook 02's date/time and round-trip float serialization policy."""
    import pandas as pd

    out = frame.copy()
    for field in out:
        if isinstance(out[field].dtype, pd.DatetimeTZDtype):
            out[field] = out[field].astype(str).replace("NaT", None)
        elif pd.api.types.is_datetime64_any_dtype(out[field]):
            out[field] = out[field].dt.strftime("%Y-%m-%d")
    path = output / name
    out.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
    return dict(path=path.relative_to(root).as_posix(), rows=len(frame), sha256=sha256(path))


def prepare_panel_inputs(project_root: Path) -> dict:
    """Join the current intake to shared metadata and verify the time interface.

    Run prepare_inputs first. Labels and eligibility flags are retained for audit,
    without filtering trading dates or constructing positions in this step.
    """
    import pandas as pd

    root = project_root.resolve()
    output = root / "results" / "backtest"
    intake = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    spec_path = root / "data/processed/panel_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    panel_path = root / spec["shared_panel"]
    calendar_path = root / "data/interim/etf_prices_aligned.csv"
    monthly_path = root / "notebooks/panel_checks/monthly_training_eligibility.csv"
    combined_path = output / "combined_predictions.csv"
    reference_paths = [root / "notebooks" / name for name in (
        "02_features_panel_data.ipynb", "03_elastic_net.ipynb", "04_gbdt.ipynb"
    )]
    source_paths = [spec_path, panel_path, calendar_path, monthly_path, combined_path]
    source_paths += [root / record["path"] for record in intake["sources"]]
    source_paths += reference_paths
    hashes = {path.relative_to(root).as_posix(): sha256(path) for path in source_paths}
    checks = []

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    def equal_with_missing(left, right):
        return (left.eq(right) | (left.isna() & right.isna())).all()

    recorded = intake["sources"] + intake["outputs"]
    check("current_intake_hashes", all(
        hashes[record["path"]] == record["sha256"]
        for record in recorded if record["path"] in hashes
    ), "The combined forecasts and both sources match the step-1 manifest")

    # Match the read/join/timezone conventions used by notebooks 02-04.
    ny = "America/New_York"
    keys = ["date", "ticker"]
    time_fields = ["signal_at", "entry_at", "label_available_at"]
    panel = pd.read_csv(panel_path, parse_dates=["date", "entry_date", "exit_date"],
                        low_memory=False, float_precision="round_trip")
    check("shared_panel_schema", panel.columns.tolist() == spec["shared_panel_columns"],
          "The shared panel retains the schema declared by panel_spec.json")
    for field in time_fields:
        panel[field] = pd.to_datetime(panel[field], utc=True).dt.tz_convert(ny)
    check("unique_panel_keys", not panel.duplicated(keys).any(), "One row per date/ticker")
    check("panel_boolean_flags", all(pd.api.types.is_bool_dtype(panel[field]) for field in (
        "common_features_valid", "common_scoreable", "label_valid", "label_within_split"
    )), "Flags are booleans, not nonempty strings")
    check("panel_sector_ids", set(panel.ticker) == set(TICKERS), "The same nine sectors")

    prices = pd.read_csv(calendar_path, usecols=["date", "ticker"], parse_dates=["date"])
    spy_dates = prices.loc[prices.ticker.eq("SPY"), "date"].sort_values()
    calendar = pd.DatetimeIndex(spy_dates)
    check("shared_spy_calendar", calendar.is_unique and not calendar.hasnans
          and pd.DatetimeIndex(sorted(panel.date.unique())).equals(calendar),
          f"Panel dates match the {len(calendar):,}-date aligned SPY calendar")
    check("rectangular_panel", panel.groupby("date").ticker.nunique().eq(len(TICKERS)).all(),
          "All nine sectors are retained on every calendar date")
    check("data_cutoff", calendar[-1] == pd.Timestamp(spec["data_cutoff"]),
          "Calendar ends at the declared data cutoff")

    # Reconstruct calendar mappings, independently of any future label values.
    signal_calendar = pd.DataFrame({"date": calendar})
    signal_calendar["split"] = "warmup"
    for start, end, split in (("2008-01-01", "2017-12-31", "train"),
                              ("2018-01-01", "2022-12-31", "validation"),
                              ("2023-01-01", spec["data_cutoff"], "test")):
        signal_calendar.loc[signal_calendar.date.between(start, end), "split"] = split
    signal_calendar["entry_date"] = signal_calendar.date.shift(-1)
    signal_calendar["exit_date"] = signal_calendar.date.shift(-2)
    signal_calendar["signal_at"] = (
        signal_calendar.date + pd.Timedelta(hours=18)).dt.tz_localize(ny)
    for field, date_field in (("entry_at", "entry_date"),
                              ("label_available_at", "exit_date")):
        signal_calendar[field] = (signal_calendar[date_field]
                                 + pd.Timedelta(hours=9, minutes=30)).dt.tz_localize(ny)
    by_date = signal_calendar.set_index("date")
    for field in ["split", "entry_date", "exit_date"] + time_fields:
        expected = panel.date.map(by_date[field])
        check(f"calendar_mapping_{field}", equal_with_missing(panel[field], expected),
              "Matches next/second-next SPY date and New York local clock, including missing endpoints")

    forecasts = pd.read_csv(combined_path, parse_dates=["date"], low_memory=False,
                            float_precision="round_trip")
    forecasts["fit_cutoff"] = pd.to_datetime(forecasts.fit_cutoff, utc=True).dt.tz_convert(ny)
    joined = forecasts.merge(panel, on=keys, how="left", validate="many_to_one",
                             indicator=True, sort=False)
    check("many_to_one_panel_join", len(joined) == len(forecasts)
          and joined["_merge"].eq("both").all(), f"All {len(joined):,} forecasts match the shared panel")
    joined = joined.drop(columns="_merge")
    expected_panel = panel.loc[panel["split"].isin(["validation", "test"])
                               & panel.common_features_valid].copy()
    expected_keys = pd.MultiIndex.from_frame(expected_panel[keys])
    for (model, scheme), group in joined.groupby(["model", "scheme"], sort=False):
        actual_keys = pd.MultiIndex.from_frame(group[keys])
        check(f"eligible_coverage_{model}_{scheme}", actual_keys.is_unique
              and len(actual_keys) == len(expected_keys)
              and actual_keys.difference(expected_keys).empty
              and expected_keys.difference(actual_keys).empty,
              "Exactly all common_features_valid validation/test keys; no scoreability filtering")

    schedule_rows = []
    first_month = expected_panel.date.min().to_period("M")
    last_month = pd.Timestamp(spec["data_cutoff"]).to_period("M")
    for month in pd.period_range(first_month, last_month, freq="M"):
        first = calendar[calendar.to_period("M") == month][0]
        position = calendar.get_loc(first)
        check(f"prior_date_{month}", position > 0, "Prior SPY date exists for monthly fit")
        cutoff = (calendar[position - 1] + pd.Timedelta(hours=18)).tz_localize(ny)
        schedule_rows.append(dict(model_month=str(month), first_signal_date=first,
                                  expected_fit_cutoff=cutoff))
    schedule = pd.DataFrame(schedule_rows).set_index("model_month")
    expected_cutoff = joined.model_month.map(schedule.expected_fit_cutoff)
    check("monthly_fit_cutoffs", joined.fit_cutoff.eq(expected_cutoff).all(),
          "Every model/scheme uses prior SPY trading day 18:00 for its signal month")
    check("cutoff_before_signal", joined.fit_cutoff.lt(joined.signal_at).all(),
          "Historical fitting cutoff precedes every forecast signal")
    known_entry = joined.entry_at.notna()
    check("signal_before_entry", joined.loc[known_entry, "signal_at"].lt(
        joined.loc[known_entry, "entry_at"]).all(), "Signals precede all known entry opens")
    known_exit = joined.label_available_at.notna()
    check("entry_before_label_available", joined.loc[known_exit, "entry_at"].lt(
        joined.loc[known_exit, "label_available_at"]).all(), "Labels mature at the later open")

    # Cross-check schedule and coverage against the existing notebook-02 audit.
    monthly = pd.read_csv(monthly_path, parse_dates=["first_signal_date"],
                          float_precision="round_trip")
    monthly["fit_at"] = pd.to_datetime(monthly.fit_at, utc=True).dt.tz_convert(ny)
    expected_panel["prediction_month"] = expected_panel.date.dt.strftime("%Y-%m")
    counts = expected_panel.groupby(["prediction_month", "ticker"], as_index=False).agg(
        prediction_rows_actual=("date", "size"), scoreable_rows_actual=("common_scoreable", "sum"))
    crosscheck = monthly.merge(counts, on=["prediction_month", "ticker"], how="outer",
                               validate="one_to_one", indicator=True)
    check("upstream_monthly_coverage", crosscheck["_merge"].eq("both").all()
          and crosscheck.prediction_rows_common.eq(crosscheck.prediction_rows_actual).all()
          and crosscheck.scoreable_rows_common.eq(crosscheck.scoreable_rows_actual).all(),
          f"All {len(crosscheck):,} upstream month/sector prediction and scoring counts agree")
    check("upstream_monthly_schedule", monthly.fit_at.eq(
        monthly.prediction_month.map(schedule.expected_fit_cutoff)).all()
        and monthly.first_signal_date.eq(monthly.prediction_month.map(schedule.first_signal_date)).all(),
          "Monthly cutoff and first signal dates agree with the notebook-02 audit")

    summary = joined.groupby(["model", "scheme", "split"], as_index=False, sort=False).agg(
        prediction_rows=("date", "size"), signal_dates=("date", "nunique"),
        scoreable_rows=("common_scoreable", "sum"),
        missing_entry_rows=("entry_date", lambda values: values.isna().sum()),
        missing_exit_rows=("exit_date", lambda values: values.isna().sum()),
        first_signal_date=("date", "min"), last_signal_date=("date", "max"))
    summary["unscoreable_rows"] = summary.prediction_rows - summary.scoreable_rows
    monthly_checks = joined.groupby(["model", "scheme", "model_month"],
                                     as_index=False, sort=False).agg(
        fit_cutoff=("fit_cutoff", "first"), prediction_rows=("date", "size"),
        first_signal_date=("date", "min"), last_signal_date=("date", "max"))
    monthly_checks["expected_fit_cutoff"] = monthly_checks.model_month.map(schedule.expected_fit_cutoff)
    monthly_checks["cutoff_matches"] = monthly_checks.fit_cutoff.eq(monthly_checks.expected_fit_cutoff)
    examples = joined.loc[joined.model.eq("elastic_net") & joined.scheme.eq("S0")
                          & joined.ticker.eq("XLB") & joined.date.isin(pd.to_datetime([
                              "2018-01-02", "2022-12-28", "2022-12-29", "2022-12-30",
                              "2023-01-03", "2025-10-31", "2026-03-06",
                              "2026-09-28", "2026-09-29", "2026-09-30"
                          ])), ["date", "split", "fit_cutoff", "signal_at", "entry_date", "entry_at",
                                "exit_date", "label_available_at", "common_features_valid",
                                "label_valid", "label_within_split", "common_scoreable"]].copy()

    outputs = [save_frame(frame, name, output, root) for frame, name in (
        (joined, "predictions_with_panel.csv"), (summary, "panel_alignment_summary.csv"),
        (monthly_checks, "monthly_time_checks.csv"), (signal_calendar, "signal_calendar.csv"),
        (examples, "time_alignment_examples.csv"))]
    restored = pd.read_csv(output / "predictions_with_panel.csv", parse_dates=["date"],
                           low_memory=False, float_precision="round_trip")
    for field in FIELDS:
        if field == "fit_cutoff":
            restored[field] = pd.to_datetime(restored[field], utc=True).dt.tz_convert(ny)
        pd.testing.assert_series_equal(restored[field], joined[field], check_exact=True)
    check("forecast_value_roundtrip", True, "All seven forecast columns survive joined CSV export exactly")
    check("source_hashes_unchanged", all(sha256(root / path) == digest for path, digest in hashes.items()),
          "Data, source forecasts, combined intake and reference notebooks are unchanged")
    audit = dict(schema_version=1, scope="Step 2: shared-panel and time-interface alignment only",
                 created_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                 pandas_version=pd.__version__, timezone=ny, panel_rows=len(panel),
                 calendar_dates=len(calendar), joined_prediction_rows=len(joined),
                 monthly_strategy_checks=len(monthly_checks), upstream_month_sector_checks=len(crosscheck),
                 source_hashes=hashes, checks=checks, outputs=outputs,
                 not_yet_checked=["trading eligibility and missing-signal actions",
                                  "execution prices, dividends and risk-free interval alignment",
                                  "positions, trading thresholds, costs and portfolio returns"])
    (output / "panel_alignment_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_signal_inputs(project_root: Path) -> dict:
    """Step 3: emit independent trading/scoring states and validate missing-data guards."""
    import pandas as pd
    if __package__:
        from . import backtest_signal_policy as policy
    else:
        import backtest_signal_policy as policy

    root = project_root.resolve()
    output = root / "results/backtest"
    previous = json.loads((output / "panel_alignment_manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_hashes"])
    protected.update({record["path"]: record["sha256"] for record in previous["outputs"]})
    protected["notebooks/01_download_clean_data.ipynb"] = sha256(root / "notebooks/01_download_clean_data.ipynb")
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Step-2 input/output changed; rerun the earlier checks: {path}")
    spec = json.loads((root / "data/processed/panel_spec.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(root / spec["shared_panel"], parse_dates=["date"],
                        low_memory=False, float_precision="round_trip")
    forecasts = pd.read_csv(output / "combined_predictions.csv", parse_dates=["date"],
                            low_memory=False, float_precision="round_trip")
    calendar = pd.read_csv(output / "signal_calendar.csv", parse_dates=["date", "entry_date"],
                           low_memory=False, float_precision="round_trip")
    forecasts["fit_cutoff"] = pd.to_datetime(forecasts.fit_cutoff, utc=True).dt.tz_convert("America/New_York")
    for field in ["signal_at", "entry_at"]:
        calendar[field] = pd.to_datetime(calendar[field], utc=True).dt.tz_convert("America/New_York")
    kwargs = dict(models=MODELS, schemes=SCHEMES, tickers=TICKERS)
    signals, daily = policy.build_trading_status(panel, forecasts, calendar, **kwargs)
    scoring = policy.build_scoring_eligibility(signals, panel)
    checks = policy.validate_signal_policy(panel, forecasts, calendar, **kwargs)
    if not checks.passed.all():
        raise AssertionError("Signal-policy checks failed")
    summary = daily.groupby(["model", "scheme", "split"], as_index=False).agg(
        signal_days=("date", "size"), complete_signal_days=("signal_complete", "sum"),
        evaluate_signal_days=("can_evaluate_signal", "sum"),
        missing_signal_close_days=("action", lambda value: value.eq("CLOSE_ALL_TO_CASH").sum()),
        scheduled_liquidation_days=("action", lambda value: value.eq("FINAL_LIQUIDATION").sum()),
        outside_phase_days=("action", lambda value: value.eq("NO_ORDER_OUTSIDE_PHASE").sum()))
    score_counts = scoring.groupby(["model", "scheme", "split"], as_index=False).agg(
        panel_scoreable_rows=("scoreable_by_panel", "sum"), score_eligible_rows=("score_eligible", "sum"))
    summary = summary.merge(score_counts, on=["model", "scheme", "split"], validate="one_to_one")
    examples = daily.loc[daily.date.isin(pd.to_datetime([
        "2020-03-24", "2022-12-28", "2022-12-29", "2022-12-30",
        "2026-09-28", "2026-09-29", "2026-09-30"
    ]))].copy()
    outputs = [save_frame(frame, name, output, root) for frame, name in (
        (signals, "trading_signal_inputs.csv"), (daily, "daily_signal_status.csv"),
        (scoring, "scoring_eligibility.csv"), (summary, "eligibility_summary.csv"),
        (examples, "signal_status_examples.csv"), (checks, "signal_policy_checks.csv"))]
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Protected input changed during step 3: {path}")
    audit = dict(schema_version=1, scope="Step 3: scoring and signal-time trading eligibility",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(), pandas_version=pd.__version__,
                 sector_signal_rows=len(signals), daily_strategy_rows=len(daily),
                 score_eligible_rows=int(scoring.score_eligible.sum()),
                 action_counts={str(key): int(value) for key, value in daily.action.value_counts().items()},
                 phase_opens=policy.PHASE_OPENS, trading_panel_fields=policy.ASOF_PANEL_FIELDS,
                 trading_forecast_fields=policy.FORECAST_FIELDS, trading_calendar_fields=policy.CALENDAR_FIELDS,
                 checks=checks.to_dict("records"), source_hashes=protected,
                 code_hashes={path.relative_to(root).as_posix(): sha256(path) for path in (
                     Path(__file__).resolve(), Path(policy.__file__).resolve())}, outputs=outputs,
                 notes=["Fault cases use in-memory copies, never modified source CSVs",
                        "CLOSE_ALL_TO_CASH and FINAL_LIQUIDATION request empty target pairs at action_at",
                        "EVALUATE_SIGNAL means input readiness, not an actual entry or holding decision",
                        "Scoring flags and targets are excluded from the trading-policy input whitelist"],
                 not_yet_checked=["prediction/investment baselines", "threshold and ranking position logic",
                                  "actual trades, prices, dividends, financing, costs and returns"])
    (output / "signal_policy_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_baseline_inputs(project_root: Path) -> dict:
    """Step 4: verify/reuse delivered means and evaluate the two forecast baselines."""
    import pandas as pd
    if __package__:
        from . import backtest_baselines as baselines
    else:
        import backtest_baselines as baselines

    root = project_root.resolve()
    output = root / "results/backtest"
    previous = json.loads((output / "signal_policy_manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_hashes"])
    protected.update({record["path"]: record["sha256"] for record in previous["outputs"]})
    added_paths = [root / "results" / model / name for model in MODELS for name in (
        f"{model}_predictions_full.csv", "forecast_metrics.csv")]
    protected.update({path.relative_to(root).as_posix(): sha256(path) for path in added_paths})
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Baseline source changed; rerun earlier input checks: {path}")
    spec = json.loads((root / "data/processed/panel_spec.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(root / spec["shared_panel"], parse_dates=["date"], low_memory=False,
                        float_precision="round_trip")
    for field in ("signal_at", "label_available_at"):
        panel[field] = pd.to_datetime(panel[field], utc=True).dt.tz_convert("America/New_York")
    main_forecasts = pd.read_csv(output / "combined_predictions.csv", parse_dates=["date"],
                                 low_memory=False, float_precision="round_trip")
    main_forecasts["fit_cutoff"] = pd.to_datetime(main_forecasts.fit_cutoff, utc=True).dt.tz_convert("America/New_York")
    full_exports, recorded_metrics = {}, {}
    for model in MODELS:
        full = pd.read_csv(root / "results" / model / f"{model}_predictions_full.csv",
                           parse_dates=["date"], low_memory=False, float_precision="round_trip")
        full["fit_cutoff"] = pd.to_datetime(full.fit_cutoff, utc=True).dt.tz_convert("America/New_York")
        full_exports[model] = full
        recorded_metrics[model] = pd.read_csv(root / "results" / model / "forecast_metrics.csv",
                                               low_memory=False, float_precision="round_trip")
    monthly = pd.read_csv(root / "notebooks/panel_checks/monthly_training_eligibility.csv",
                          parse_dates=["latest_training_signal"], low_memory=False,
                          float_precision="round_trip")
    for field in ("fit_at", "latest_label_available_at"):
        monthly[field] = pd.to_datetime(monthly[field], utc=True).dt.tz_convert("America/New_York")
    tables = baselines.build_baseline_tables(panel, main_forecasts, full_exports, monthly,
                                              recorded_metrics, fields=FIELDS)
    predictions = tables["baseline_predictions.csv"]
    outputs = [save_frame(predictions, "baseline_predictions.csv", output, root)]
    restored = pd.read_csv(output / "baseline_predictions.csv", parse_dates=["date"],
                           low_memory=False, float_precision="round_trip")
    restored["fit_cutoff"] = pd.to_datetime(restored.fit_cutoff, utc=True).dt.tz_convert("America/New_York")
    pd.testing.assert_frame_equal(restored, predictions, check_exact=True)
    tables["baseline_checks.csv"] = pd.concat([tables["baseline_checks.csv"], pd.DataFrame([
        dict(check="baseline_export_roundtrip", passed=True,
             detail="All seven baseline forecast columns survive CSV export exactly")])], ignore_index=True)
    outputs += [save_frame(frame, name, output, root) for name, frame in tables.items()
                if name != "baseline_predictions.csv"]
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Protected baseline source changed: {path}")
    audit = dict(schema_version=1, scope="Step 4: zero and historical-mean prediction baselines",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(), pandas_version=pd.__version__,
                 baseline_prediction_rows=len(predictions),
                 monthly_training_records=len(tables["monthly_baseline_audit.csv"]),
                 baseline_scheme=baselines.BASELINE_SCHEME,
                 canonical_mean_source="results/elastic_net/elastic_net_predictions_full.csv: historical_mean",
                 mean_verification="Independent mature-label monthly recomputation and GBDT full-export cross-check",
                 baseline_roles={"zero": "Prediction evaluation only; do not construct arbitrary tied portfolios",
                                 "historical_mean": "Prediction benchmark; optional future backtest under the same public rules"},
                 checks=tables["baseline_checks.csv"].to_dict("records"), source_hashes=protected,
                 code_hashes={path.relative_to(root).as_posix(): sha256(path) for path in (
                     Path(__file__).resolve(), Path(baselines.__file__).resolve())}, outputs=outputs,
                 notes=["COMMON shares one baseline across all three feature schemes; no duplicated scheme portfolios",
                        "Baseline fit_cutoff/model_month are common schedule anchors; zero requires no estimation",
                        "All error metrics are forecast metrics, not trading returns",
                        "Zero-forecast direction accuracy is N/A because it expresses no directional view",
                        "Main model forecasts/configurations and their original metrics remain unchanged"],
                 not_yet_checked=["daily pair ranking, persistence, entries and exits",
                                  "target weights, net trades, costs, financing and portfolio returns",
                                  "SPY buy-and-hold and cash investment benchmarks (step 12)"])
    (output / "baseline_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_pair_decisions(project_root: Path) -> dict:
    """Step 5: parameterized pair rules and a small engineering integration example."""
    import pandas as pd
    if __package__:
        from . import backtest_pair_decisions as decisions
    else:
        import backtest_pair_decisions as decisions

    root = project_root.resolve()
    output = root / "results/backtest"
    previous = json.loads((output / "baseline_manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_hashes"])
    protected.update({record["path"]: record["sha256"] for record in previous["outputs"]})
    protected["../5205proj2.md"] = sha256(root.parent / "5205proj2.md")
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Pair-decision source changed: {path}")
    checks = decisions.validate_pair_decisions(TICKERS)
    signals = pd.read_csv(output / "trading_signal_inputs.csv",
                          usecols=["date", "ticker", "model", "scheme", "y_hat"],
                          parse_dates=["date"], low_memory=False, float_precision="round_trip")
    daily = pd.read_csv(output / "daily_signal_status.csv", parse_dates=["date"],
                        low_memory=False, float_precision="round_trip")
    for field in ("signal_at", "action_at"):
        daily[field] = pd.to_datetime(daily[field], utc=True).dt.tz_convert("America/New_York")
    # Fixed technical example: no test data, PnL or threshold search is used.
    demo_dates = sorted(daily.loc[daily["split"].eq("validation"), "date"].unique())[:5]
    demo_daily = daily.loc[daily["split"].eq("validation") & daily.date.isin(demo_dates)]
    demo_signals = signals.loc[signals.date.isin(demo_dates)]
    entry_bps = 10
    day_targets, sector_targets = decisions.replay_pair_targets(
        demo_signals, demo_daily, h_in=entry_bps / 10000, tickers=TICKERS)
    for frame in (day_targets, sector_targets):
        frame["example_only"] = True
        frame["threshold_selected"] = False
    if len(day_targets) != 5 * len(MODELS) * len(SCHEMES) or len(sector_targets) != len(day_targets) * len(TICKERS):
        raise AssertionError("Pair-decision example lost strategy/date/sector rows")
    outputs = [save_frame(frame, name, output, root) for frame, name in (
        (checks, "pair_decision_checks.csv"), (day_targets, "pair_decision_demo_daily.csv"),
        (sector_targets, "pair_decision_demo_sectors.csv"))]
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Protected input changed during pair-rule checks: {path}")
    audit = dict(schema_version=1, scope="Step 5: pair decision rules; engineering example only",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(), pandas_version=pd.__version__,
                 example_only=True, threshold_selection_performed=False, example_entry_bps=entry_bps,
                 example_exit_bps=entry_bps / 2,
                 example_signal_dates=[pd.Timestamp(value).strftime("%Y-%m-%d") for value in demo_dates],
                 daily_example_rows=len(day_targets), sector_example_rows=len(sector_targets),
                 rules=dict(max_pairs=4, exit_threshold_ratio=0.5, inequalities="strict > / <",
                            new_entry_candidates="global top four absolute predictions only",
                            tie_order=list(TICKERS), preserve_old_outside_top_four=True,
                            reversal="exit plus full opposite-entry conditions, returned as one final target",
                            status_source="step-3 daily_signal_status.csv",
                            prediction_source="existing Elastic Net/GBDT y_hat, read from step-3 signal inputs"),
                 checks=checks.to_dict("records"), source_hashes=protected,
                 code_hashes={path.relative_to(root).as_posix(): sha256(path) for path in (
                     Path(__file__).resolve(), Path(decisions.__file__).resolve(),
                     root / "scripts/backtest_signal_policy.py")}, outputs=outputs,
                 notes=["No new user decision or model training is needed; rules come from 5205proj2.md",
                        "The core decide_pairs API requires an explicit h_in; no final threshold is assumed",
                        "10 bps is a fixed integration example, not a validation-selected threshold",
                        "The example covers only the first five validation signals; no test-period decision replay",
                        "Logical pair targets are not actual shares, weights, transactions or returns"],
                 not_yet_checked=["target leg weights, SPY netting and net trades (step 6)",
                                  "cash/NAV ledger, dividends, execution and financing costs",
                                  "public threshold selection from validation results (step 10)"])
    (output / "pair_decision_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_execution_inputs(project_root: Path) -> dict:
    """Step 6: share-difference planner plus normalized targets, without a NAV ledger."""
    import pandas as pd
    if __package__:
        from . import backtest_execution as execution
    else:
        import backtest_execution as execution

    root = project_root.resolve()
    output = root / "results/backtest"
    previous = json.loads((output / "pair_decision_manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_hashes"])
    protected.update({record["path"]: record["sha256"] for record in previous["outputs"]})
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Execution-planner input changed: {path}")
    checks, worked = execution.validate_net_trades(TICKERS)
    daily = pd.read_csv(output / "pair_decision_demo_daily.csv", parse_dates=["date"],
                        low_memory=False, float_precision="round_trip")
    daily["action_at"] = pd.to_datetime(daily.action_at, utc=True).dt.tz_convert("America/New_York")
    sectors = pd.read_csv(output / "pair_decision_demo_sectors.csv", parse_dates=["date"],
                          low_memory=False, float_precision="round_trip")
    prices = pd.read_csv(root / "data/interim/etf_prices_aligned.csv",
                         usecols=["date", "ticker", "open"], parse_dates=["date"],
                         low_memory=False, float_precision="round_trip")
    prices = prices.loc[prices.ticker.isin([*TICKERS, "SPY"])]
    if prices.duplicated(["date", "ticker"]).any():
        raise AssertionError("Duplicate execution-date/ticker open quotes")
    quote_lookup = prices.set_index(["date", "ticker"])["open"].to_dict()
    target_lookup = {key: dict(zip(group.ticker, group.target_direction)) for key, group in
                     sectors.groupby(["date", "model", "scheme"], sort=False)}
    target_rows, summary_rows = [], []
    for record in daily.to_dict("records"):
        if not record["order_scheduled"]:
            # No reconciliation/trade request for phase-excluded signals.
            continue
        key = (record["date"], record["model"], record["scheme"])
        directions = target_lookup[key]
        if set(directions) != set(TICKERS):
            raise AssertionError("Incomplete sector directions in the decision example")
        pairs = {ticker: int(direction) for ticker, direction in directions.items() if direction}
        open_date = record["action_at"].tz_localize(None).normalize()
        quotes = {ticker: quote_lookup.get((open_date, ticker), float("nan")) for ticker in [*TICKERS, "SPY"]}
        # Unit-NAV sizing is a target-only identity, not a historical share replay.
        rows, summary = execution.plan_net_trades(pairs, {}, quotes, 1.0, tickers=TICKERS)
        metadata = {field: record[field] for field in ("date", "model", "scheme", "split", "action_at")}
        summary_rows.append(metadata | dict(pair_count=summary["pair_count"],
            gross_weight_before_spy_netting=summary["gross_weight_before_spy_netting"],
            target_gross_weight=summary["target_gross_weight"], target_net_weight=summary["target_net_weight"],
            example_only=True, normalized_target_only=True, threshold_selected=False))
        target_rows.extend(metadata | dict(ticker=row["ticker"], target_weight=row["target_weight"],
            observed_open=row["open_price"], target_shares_per_nav=row["target_shares"],
            example_only=True, normalized_target_only=True, threshold_selected=False) for row in rows)
    targets, summary = pd.DataFrame(target_rows), pd.DataFrame(summary_rows)
    if len(summary) != previous["daily_example_rows"] or len(targets) != len(summary) * (len(TICKERS) + 1):
        raise AssertionError("Normalized target example lost asset or strategy/date rows")
    outputs = [save_frame(frame, name, output, root) for frame, name in (
        (checks, "execution_checks.csv"), (worked, "net_trade_worked_examples.csv"),
        (summary, "execution_target_demo_daily.csv"), (targets, "execution_target_demo_assets.csv"))]
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Protected input changed during execution checks: {path}")
    audit = dict(schema_version=1, scope="Step 6: net target share/trade planner; no cash/NAV replay",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(), pandas_version=pd.__version__,
                 leg_weight=execution.LEG_WEIGHT, fractional_shares=True, weight_tolerance_band=None,
                 price_field="data/interim/etf_prices_aligned.csv: observed open only",
                 source_signal_dates=previous["example_signal_dates"],
                 example_entry_bps=previous["example_entry_bps"], threshold_selection_performed=False,
                 normalized_target_example_only=True, actual_nav_ledger_implemented=False,
                 daily_target_examples=len(summary), asset_target_examples=len(targets),
                 worked_share_plan_rows=len(worked),
                 checks=checks.to_dict("records"), source_hashes=protected,
                 code_hashes={path.relative_to(root).as_posix(): sha256(path) for path in (
                     Path(__file__).resolve(), Path(execution.__file__).resolve())}, outputs=outputs,
                 notes=["Actual share planning requires caller-supplied settled NAV_pre and current signed shares",
                        "Target dollars use NAV_pre before new trading fees; fees never change the sizing basis here",
                        "SPY is netted before target shares and actual traded notional are computed",
                        "No rounding or minimum trade band; fractional shares are supported",
                        "Invalid required opens raise an error; prices are never filled or adjusted again",
                        "Normalized examples show weights and shares per unit NAV only, not actual historical holdings",
                        "Worked net trade cases use explicit hypothetical NAV/old shares and are marked as examples"],
                 not_yet_checked=["cash/shares/dividend/interest/borrow-fee settlement and NAV ledger (step 7)",
                                  "execution cost and financing assumptions in portfolio results (step 8)",
                                  "public threshold selection and formal portfolio runs"])
    (output / "execution_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_ledger_inputs(project_root: Path) -> dict:
    """Step 7: general accounting engine and a marked five-open engineering preview."""
    import pandas as pd
    if __package__:
        from . import backtest_ledger as ledger
        from .backtest_costs import main_cost_parameters
    else:
        import backtest_ledger as ledger
        from backtest_costs import main_cost_parameters

    root = project_root.resolve()
    output = root / "results/backtest"
    previous = json.loads((output / "execution_manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_hashes"])
    protected.update({record["path"]: record["sha256"] for record in previous["outputs"]})
    rf_path = "data/interim/risk_free_open_intervals.csv"
    protected[rf_path] = sha256(root / rf_path)
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Ledger source changed: {path}")
    checks, worked, fixture = ledger.validate_ledger(TICKERS)
    signals = pd.read_csv(output / "trading_signal_inputs.csv",
                          usecols=["date", "ticker", "model", "scheme", "y_hat"],
                          parse_dates=["date"], low_memory=False, float_precision="round_trip")
    daily = pd.read_csv(output / "daily_signal_status.csv", parse_dates=["date"],
                        low_memory=False, float_precision="round_trip")
    for field in ("signal_at", "action_at"):
        daily[field] = pd.to_datetime(daily[field], utc=True).dt.tz_convert("America/New_York")
    prices = pd.read_csv(root / "data/interim/etf_prices_aligned.csv",
                         usecols=["date", "ticker", "open", "dividends"], parse_dates=["date"],
                         low_memory=False, float_precision="round_trip")
    prices = prices.loc[prices.ticker.isin([*TICKERS, "SPY"])]
    rates = pd.read_csv(root / rf_path, parse_dates=["entry_date", "exit_date"],
                        low_memory=False, float_precision="round_trip")
    dates = pd.to_datetime(previous["source_signal_dates"])
    daily = daily.loc[daily["split"].eq("validation") & daily.date.isin(dates)]
    signals = signals.loc[signals.date.isin(dates)]
    # Documented main values are explicit engineering parameters. Step 8 will
    # formalize their configuration and sensitivity comparisons, without tuning.
    assumptions = main_cost_parameters()
    logs, assets, trades = ledger.replay_ledger(signals, daily, prices, rates,
        h_in=previous["example_entry_bps"] / 10000, initial_nav=1.0, tickers=TICKERS,
        require_final_liquidation=False, **assumptions)
    if len(logs) != previous["daily_target_examples"] or len(assets) != len(logs) * (len(TICKERS) + 1):
        raise AssertionError("Ledger preview lost strategy/date/asset rows")
    rate_alignment = logs.loc[~logs.is_initial_open, ["model", "scheme", "split", "interval_entry_date", "open_date",
        "cash_interest_base", "interval_risk_free_return", "cash_interest", "borrow_fee"]].merge(
            rates, left_on=["interval_entry_date", "open_date"], right_on=["entry_date", "exit_date"],
            how="left", validate="many_to_one")
    for frame in (logs, assets, trades, rate_alignment):
        frame["example_only"] = True
        frame["truncated_preview"] = True
        frame["threshold_selected"] = False
    outputs = [save_frame(frame, name, output, root) for frame, name in (
        (checks, "ledger_checks.csv"), (worked, "ledger_worked_examples.csv"),
        (fixture, "ledger_replay_fixture.csv"), (logs, "ledger_demo_daily.csv"),
        (assets, "ledger_demo_assets.csv"), (trades, "ledger_demo_trades.csv"),
        (rate_alignment, "ledger_rate_alignment.csv"))]
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Protected input changed during ledger verification: {path}")
    audit = dict(schema_version=1, scope="Step 7: open-to-open cash/share/NAV accounting",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(), pandas_version=pd.__version__,
                 example_only=True, truncated_preview=True, formal_phase_evaluation=False,
                 threshold_selection_performed=False, initial_nav=1.0,
                 example_entry_bps=previous["example_entry_bps"], engineering_cost_parameters=assumptions,
                 source_signal_dates=previous["source_signal_dates"], actual_share_ledger_implemented=True,
                 daily_ledger_rows=len(logs), asset_ledger_rows=len(assets), actual_nonzero_trade_rows=len(trades),
                 matched_rate_intervals=len(rate_alignment),
                 max_pre_reconciliation_error=float(logs.pre_reconciliation_error.abs().max()),
                 max_post_reconciliation_error=float(logs.post_reconciliation_error.abs().max()),
                 checks=checks.to_dict("records"), source_hashes=protected,
                 code_hashes={path.relative_to(root).as_posix(): sha256(path) for path in (
                     Path(__file__).resolve(), Path(ledger.__file__).resolve(),
                     root / "scripts/backtest_execution.py", root / "scripts/backtest_pair_decisions.py",
                     root / "scripts/backtest_signal_policy.py", root / "scripts/backtest_costs.py")}, outputs=outputs,
                 notes=["Signal-time decisions and open execution remain separate; labels never enter accounting",
                        "Dividends are accrued on old signed shares before any new target trade",
                        "Interest uses last post-trade cash and starting net short collateral",
                        "Borrow fees use starting net short value times natural days / 365",
                        "Exported risk-free interval returns are not divided a second time",
                        "Target shares use settled NAV_pre; execution fees reduce cash once afterward",
                        "First holding return includes initial entry cost; no zero-length initial return is inserted",
                        "Only the first five validation signals are replayed; this is not a complete phase result"],
                 not_yet_checked=["formal cost/financing configuration and sensitivity comparisons (step 8)",
                                  "training-period OOF signal-scale checks and public threshold selection (steps 9-10)",
                                  "frozen full validation/test portfolio evaluation and investment benchmarks"])
    (output / "ledger_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_cost_inputs(project_root: Path) -> dict:
    """Step 8: shared cost policy and eight fixed-threshold engineering comparisons."""
    import numpy as np
    import pandas as pd
    if __package__:
        from . import backtest_costs as costs
    else:
        import backtest_costs as costs

    root = project_root.resolve()
    output = root / "results/backtest"
    previous = json.loads((output / "ledger_manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_hashes"])
    protected.update({record["path"]: record["sha256"] for record in previous["outputs"]})
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Cost comparison input changed: {path}")
    # Record the protocol settings and all comparisons before any scenario replay.
    settings = costs.cost_settings()
    settings_path = output / "cost_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scenario_table = pd.DataFrame(costs.scenario_records())
    scenario_table["borrow_annual_pct"] = scenario_table.borrow_annual_rate * 100
    scenario_record = save_frame(scenario_table, "cost_scenarios.csv", output, root)
    settings_hash = sha256(settings_path)
    signals = pd.read_csv(output / "trading_signal_inputs.csv",
                          usecols=["date", "ticker", "model", "scheme", "y_hat"],
                          parse_dates=["date"], low_memory=False, float_precision="round_trip")
    daily = pd.read_csv(output / "daily_signal_status.csv", parse_dates=["date"],
                        low_memory=False, float_precision="round_trip")
    for field in ("signal_at", "action_at"):
        daily[field] = pd.to_datetime(daily[field], utc=True).dt.tz_convert("America/New_York")
    prices = pd.read_csv(root / "data/interim/etf_prices_aligned.csv",
                         usecols=["date", "ticker", "open", "dividends"], parse_dates=["date"],
                         low_memory=False, float_precision="round_trip")
    prices = prices.loc[prices.ticker.isin([*TICKERS, "SPY"])]
    rates = pd.read_csv(root / "data/interim/risk_free_open_intervals.csv",
                        parse_dates=["entry_date", "exit_date"], low_memory=False, float_precision="round_trip")
    dates = pd.to_datetime(previous["source_signal_dates"])
    daily = daily.loc[daily["split"].eq("validation") & daily.date.isin(dates)]
    signals = signals.loc[signals.date.isin(dates)]
    common = dict(h_in=previous["example_entry_bps"] / 10000, initial_nav=previous["initial_nav"], tickers=TICKERS)
    book, assets, trades, summary = costs.run_cost_scenarios(signals, daily, prices, rates, **common)
    synthetic = costs.synthetic_cost_fixture(TICKERS)
    synthetic_book, synthetic_assets, _, synthetic_summary = costs.run_cost_scenarios(*synthetic, **common)
    checks = costs.validate_cost_scenarios(synthetic_book, synthetic_assets, synthetic_summary)
    additional = []

    def check(name, passed, detail):
        additional.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name)

    reference = pd.read_csv(output / "ledger_demo_daily.csv", parse_dates=["date", "open_date"],
                            low_memory=False, float_precision="round_trip")
    keys = ["date", "model", "scheme", "split", "open_date"]
    numeric_fields = ["cash_prev", "nav_pre", "nav_post", "execution_fee", "borrow_fee", "cash_interest", "traded_notional"]
    matched = book.loc[book.is_main].merge(reference[keys + numeric_fields], on=keys, how="outer",
        validate="one_to_one", indicator=True, suffixes=("", "_step7"))
    check("main_case_reproduces_step7_exactly", matched["_merge"].eq("both").all() and all(
        np.array_equal(matched[field].to_numpy(), matched[field + "_step7"].to_numpy()) for field in numeric_fields),
        "Centralized main settings reproduce every existing preview cash/NAV/fee value exactly")
    check("all_eight_cases_for_six_strategies", len(summary) == 8 * len(MODELS) * len(SCHEMES)
          and summary.groupby("scenario_id").size().eq(6).all(), "All strategies share the same eight comparison settings")
    check("fixed_threshold_and_market_window", book.entry_bps.nunique() == 1 and book["split"].eq("validation").all()
          and set(book.date) == set(pd.to_datetime(previous["source_signal_dates"])),
          "Same five validation signals, initial capital and threshold; no test-period replay")
    check("preview_accounting_reconciles", summary.accounting_residual.abs().le(1e-12).all(),
          "Every preview case accounts for all price/dividend/interest/borrow/execution changes")
    asset_fees = assets.groupby(["scenario_id", "model", "scheme", "split", "open_date"], as_index=False).execution_fee.sum()
    fee_match = book.merge(asset_fees, on=["scenario_id", "model", "scheme", "split", "open_date"],
                           validate="one_to_one", suffixes=("", "_assets"))
    check("asset_fees_sum_to_portfolio_fees", np.allclose(fee_match.execution_fee, fee_match.execution_fee_assets,
          rtol=1e-12, atol=1e-14), "Fees use actual net asset trades with no duplicated SPY costs")
    weight_keys = ["model", "scheme", "split", "open_date", "ticker"]
    targets = assets.merge(assets.loc[assets.is_main, weight_keys + ["target_weight"]], on=weight_keys,
                            validate="many_to_one", suffixes=("", "_main"))
    check("market_pair_targets_not_retuned", targets.target_weight.eq(targets.target_weight_main).all(),
          "Same entry/exit and pair directions across real-data cost comparisons; NAV-dependent shares may differ")
    check("policy_snapshot_unchanged_by_outcomes", sha256(settings_path) == settings_hash
          and settings == costs.cost_settings(), "Configuration remains the protocol-defined main policy after comparison runs")
    checks = pd.concat([checks, pd.DataFrame(additional)], ignore_index=True)
    for frame in (book, assets, trades, summary):
        frame["input_source"] = "market_engineering_preview"
    for frame in (synthetic_book, synthetic_summary):
        frame["input_source"] = "synthetic_financing_fixture"
    outputs = [dict(path=settings_path.relative_to(root).as_posix(), sha256=settings_hash), scenario_record]
    outputs += [save_frame(frame, name, output, root) for frame, name in (
        (checks, "cost_checks.csv"), (summary, "cost_preview_summary.csv"),
        (book, "cost_preview_daily.csv"), (assets, "cost_preview_assets.csv"),
        (trades, "cost_preview_trades.csv"), (synthetic_summary, "cost_synthetic_summary.csv"),
        (synthetic_book, "cost_synthetic_daily.csv"))]
    for path, expected in protected.items():
        if sha256(root / path) != expected:
            raise AssertionError(f"Protected source changed during cost comparisons: {path}")
    audit = dict(schema_version=1, scope="Step 8: centralized protocol cost policy and engineering sensitivity checks",
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(), pandas_version=pd.__version__,
                 main_cost_parameters=costs.main_cost_parameters(), scenario_count=len(scenario_table),
                 settings_written_before_replay=True, settings_selected_by_performance=False,
                 example_entry_bps=previous["example_entry_bps"], threshold_selection_performed=False,
                 example_only=True, truncated_preview=True, formal_phase_evaluation=False,
                 market_preview_summary_rows=len(summary), market_preview_daily_rows=len(book),
                 market_preview_asset_rows=len(assets), market_preview_trade_rows=len(trades),
                 synthetic_summary_rows=len(synthetic_summary), synthetic_daily_rows=len(synthetic_book),
                 market_financing_exposed_intervals=int(summary.borrow_exposed_intervals.sum()),
                 max_accounting_residual=float(summary.accounting_residual.abs().max()),
                 checks=checks.to_dict("records"), source_hashes=protected,
                 code_hashes={path.relative_to(root).as_posix(): sha256(path) for path in (
                     Path(__file__).resolve(), Path(costs.__file__).resolve(), root / "scripts/backtest_ledger.py")},
                 outputs=outputs, notes=[
                     "One code-owned main cost policy is shared by all model/scheme strategies",
                     "Sensitivity changes one provided cost/financing assumption at a time; main case is not selected by NAV",
                     "Every case starts independently and replays actual shares; fees are not merely subtracted from fixed gross returns",
                     "The same threshold and forecasts are used; only NAV-based share quantities may change with costs",
                     "Preview financing exposure is recorded separately; nonzero borrow/rebate effects are additionally verified on the synthetic fixture",
                     "Synthetic and market preview tables are engineering records, not full validation/test strategy evidence"],
                 not_yet_checked=["training-period OOF signal-scale material and recorded threshold grid (step 9)",
                                  "public threshold selection using validation only (step 10)",
                                  "frozen full-phase cost/financing comparisons and strategy reporting"])
    (output / "cost_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_training_oof(project_root: Path) -> dict:
    """Step 9: training-only calibration/OOF and fixed threshold-grid record."""
    import numpy as np
    import pandas as pd
    if __package__:
        from .backtest_oof_parallel import run_training_oof
    else:
        from backtest_oof_parallel import run_training_oof
    root = project_root.resolve()
    output = root / "results/backtest"
    audit = run_training_oof(root)
    # Include every calendar date, including dates with zero eligible sectors.
    # The inner scale tables deliberately contain only actual valid forecasts.
    panel = pd.read_csv(root / "data/processed/model_panel.csv", parse_dates=["date"],
                        usecols=["date", "ticker", "common_features_valid"],
                        low_memory=False, float_precision="round_trip")
    panel = panel.loc[panel.date.between("2013-01-01", "2017-12-31")]
    calendar = panel.groupby("date", as_index=False).common_features_valid.sum().rename(
        columns={"common_features_valid": "valid_sectors"})
    calendar["complete_nine"] = calendar.valid_sectors.eq(len(TICKERS))
    calendar["no_valid_sector"] = calendar.valid_sectors.eq(0)
    calendar["partial_sector_day"] = calendar.valid_sectors.between(1, len(TICKERS) - 1)
    forecasts = pd.read_csv(output / "training_oof/training_oof_predictions.csv", parse_dates=["date"],
                            low_memory=False, float_precision="round_trip")
    coverage_checks = []
    for (model, scheme), group in forecasts.groupby(["model", "scheme"]):
        actual = group.groupby("date").size().reindex(calendar.date, fill_value=0).to_numpy()
        passed = np.array_equal(actual, calendar.valid_sectors.to_numpy())
        if not passed:
            raise AssertionError(f"Training OOF full-calendar coverage mismatch: {model}/{scheme}")
        coverage_checks.append(dict(check=f"full_calendar_coverage_{model}_{scheme}", passed=True,
                                    detail="All calendar dates checked, including zero-valid-sector dates"))
    record = save_frame(calendar, "training_oof_calendar_coverage.csv", output, root)
    audit = dict(audit)
    audit.update(calendar_dates=len(calendar), zero_valid_sector_dates=int(calendar.no_valid_sector.sum()),
                 partial_sector_dates=int(calendar.partial_sector_day.sum()),
                 checks=audit["checks"] + coverage_checks,
                 outputs=audit["outputs"] + [record, dict(path="results/backtest/training_oof/run_manifest.json",
                    sha256=sha256(output / "training_oof/run_manifest.json"))],
                 code_hashes={**audit["code_hashes"], "scripts/prepare_backtest_inputs.py": sha256(Path(__file__)),
                    "scripts/backtest_oof_parallel.py": sha256(root / "scripts/backtest_oof_parallel.py")})
    (output / "training_oof_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_threshold_selection(project_root: Path) -> dict:
    """Step 10: select one public threshold from official validation forecasts."""
    if __package__:
        from .backtest_threshold_selection import run_threshold_selection
    else:
        from backtest_threshold_selection import run_threshold_selection
    return run_threshold_selection(project_root)


def prepare_phase_evaluation(project_root: Path) -> dict:
    """Step 11: freeze rules, reuse validation and evaluate the test phase."""
    if __package__:
        from .backtest_phase_evaluation import run_phase_evaluation
    else:
        from backtest_phase_evaluation import run_phase_evaluation
    return run_phase_evaluation(project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--stage", choices=("forecasts", "panel", "eligibility", "baselines", "decisions", "execution", "ledger", "costs", "oof", "thresholds", "phases"), default="forecasts",
                        help="forecasts: 1; panel: 1-2; eligibility: 1-3; baselines: 1-4; decisions: 1-5; execution: 1-6; ledger: 1-7; costs: 1-8; oof: step 9; thresholds: step 10; phases: step 11, requires completed step 10")
    args = parser.parse_args()
    if args.stage == "phases":
        audit = prepare_phase_evaluation(args.project_root)
        print(f"PASS: {len(audit['checks'])} phase checks; frozen public entry/exit "
              f"{audit['selected_public_entry_bps']:g}/{audit['selected_public_exit_bps']:g} bps")
        print("Validation reused; test evaluated. Steps 12-16 remain separate.")
        return
    if args.stage == "thresholds":
        audit = prepare_threshold_selection(args.project_root)
        print(f"Public entry/exit: {audit['selected_public_entry_bps']:g}/{audit['selected_public_exit_bps']:g} bps")
        print("Validation-only selection complete; protocol freezing and test evaluation remain later steps.")
        return
    if args.stage == "oof":
        audit = prepare_training_oof(args.project_root)
        print(f"PASS: {audit['prediction_rows']:,} training-only OOF forecasts; "
              f"{len(audit['checks'])} checks; candidates {audit['threshold_candidates_bps']} bps")
        print("Official forecasts preserved; final public threshold remains unselected (step 10).")
        return
    audit = prepare_inputs(args.project_root)
    print(f"PASS: {audit['strategy_count']} strategies; "
          f"{audit['combined_prediction_rows']:,} forecast rows")
    for strategy in audit["strategies"]:
        print(f"  {strategy['model']:11} {strategy['scheme']}: "
              f"{strategy['prediction_rows']:,} rows, "
              f"{strategy['signal_dates']:,} signal dates, "
              f"{strategy['first_signal_date']} to {strategy['last_signal_date']}")
    print("Output: results/backtest/ (forecasts, coverage and input manifest)")
    if args.stage in ("panel", "eligibility", "baselines", "decisions", "execution", "ledger", "costs"):
        panel_audit = prepare_panel_inputs(args.project_root)
        print(f"PASS: {panel_audit['joined_prediction_rows']:,} joined forecasts; "
              f"{panel_audit['monthly_strategy_checks']:,} monthly strategy checks; "
              f"{panel_audit['upstream_month_sector_checks']:,} upstream month/sector checks")
        print("Panel/time outputs: results/backtest/ (see panel_alignment_manifest.json)")
    if args.stage in ("eligibility", "baselines", "decisions", "execution", "ledger", "costs"):
        signal_audit = prepare_signal_inputs(args.project_root)
        print(f"PASS: {signal_audit['daily_strategy_rows']:,} daily strategy states; "
              f"{len(signal_audit['checks'])} signal-policy checks")
        print("Actions:", signal_audit["action_counts"])
    if args.stage in ("baselines", "decisions", "execution", "ledger", "costs"):
        baseline_audit = prepare_baseline_inputs(args.project_root)
        print(f"PASS: {baseline_audit['baseline_prediction_rows']:,} baseline forecasts; "
              f"{baseline_audit['monthly_training_records']:,} monthly training records; "
              f"{len(baseline_audit['checks'])} baseline checks")
    if args.stage in ("decisions", "execution", "ledger", "costs"):
        pair_audit = prepare_pair_decisions(args.project_root)
        print(f"PASS: {len(pair_audit['checks'])} pair-rule checks; "
              f"{pair_audit['daily_example_rows']} daily engineering-example targets")
        print("Example only: 10 bps on the first five validation signals; final threshold remains unselected.")
    if args.stage in ("execution", "ledger", "costs"):
        execution_audit = prepare_execution_inputs(args.project_root)
        print(f"PASS: {len(execution_audit['checks'])} weight/net-trade checks; "
              f"{execution_audit['asset_target_examples']} normalized asset targets")
        print("Actual NAV/cash accounting and portfolio returns remain pending.")
    if args.stage in ("ledger", "costs"):
        ledger_audit = prepare_ledger_inputs(args.project_root)
        print(f"PASS: {len(ledger_audit['checks'])} ledger checks; "
              f"{ledger_audit['daily_ledger_rows']} engineering ledger opens; "
              f"{ledger_audit['matched_rate_intervals']} matched cash intervals")
    if args.stage == "costs":
        cost_audit = prepare_cost_inputs(args.project_root)
        print(f"PASS: {cost_audit['scenario_count']} predefined cost scenarios; "
              f"{len(cost_audit['checks'])} cost checks; "
              f"{cost_audit['market_preview_summary_rows']} preview comparison rows")
    print("No main-model retraining, final threshold selection or formal strategy evaluation performed.")


if __name__ == "__main__":
    main()
