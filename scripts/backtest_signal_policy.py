"""Signal-time eligibility and close-all instructions; no portfolio accounting."""

from __future__ import annotations

ASOF_PANEL_FIELDS = ["date", "ticker", "common_features_valid"]
FORECAST_FIELDS = ["date", "ticker", "model", "scheme", "model_month", "fit_cutoff", "y_hat"]
CALENDAR_FIELDS = ["date", "split", "signal_at", "entry_date", "entry_at"]
PHASE_OPENS = {
    "validation": ("2018-01-03", "2022-12-30"),
    "test": ("2023-01-04", "2026-09-30"),
}


def build_trading_status(panel, forecasts, calendar, *, models, schemes, tickers):
    """Return sector signals and daily instructions using only whitelisted fields.

    Inputs use parsed dates/New York timestamps, as in notebook 05. Reindex onto
    every phase signal date, strategy and sector so missing exports cannot delete
    a cash day. Phase boundaries are fixed protocol dates, never inferred from
    label validity or the availability of future price endpoints.
    """
    import numpy as np
    import pandas as pd

    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate panel date/ticker keys")
    if forecasts.duplicated(["date", "ticker", "model", "scheme"]).any():
        raise ValueError("Duplicate forecast keys")
    calendar = calendar.loc[calendar["split"].isin(PHASE_OPENS), CALENDAR_FIELDS].copy()
    if calendar.date.duplicated().any():
        raise ValueError("Duplicate calendar dates")
    universe = pd.MultiIndex.from_product(
        [models, schemes, tickers], names=["model", "scheme", "ticker"]
    ).to_frame(index=False)
    signals = calendar.merge(universe, how="cross")
    signals = signals.merge(panel[ASOF_PANEL_FIELDS], on=["date", "ticker"], how="left",
                            validate="many_to_one", indicator="_input")
    signals["input_row_present"] = signals.pop("_input").eq("both")
    signals["common_features_valid"] = signals.common_features_valid.fillna(False).astype(bool)
    signals = signals.merge(forecasts[FORECAST_FIELDS],
                            on=["date", "ticker", "model", "scheme"], how="left",
                            validate="one_to_one", indicator="_forecast")
    signals["prediction_present"] = signals.pop("_forecast").eq("both")
    signals["y_hat"] = pd.to_numeric(signals.y_hat, errors="coerce")
    signals["prediction_finite"] = np.isfinite(signals.y_hat)
    signals["model_fit_available"] = (signals.fit_cutoff.notna()
                                       & signals.fit_cutoff.lt(signals.signal_at)
                                       & signals.model_month.eq(signals.date.dt.strftime("%Y-%m")))
    signals["row_signal_ready"] = (signals.input_row_present & signals.common_features_valid
                                     & signals.prediction_present & signals.prediction_finite
                                     & signals.model_fit_available)
    problems = {
        "missing_input": ~signals.input_row_present,
        "invalid_features": signals.input_row_present & ~signals.common_features_valid,
        "missing_prediction": ~signals.prediction_present,
        "nonfinite_prediction": signals.prediction_present & ~signals.prediction_finite,
        "invalid_model_time": signals.prediction_present & ~signals.model_fit_available,
    }
    group_keys = ["date", "model", "scheme"]
    groups = signals.groupby(group_keys, sort=False)
    daily = groups.agg(split=("split", "first"), signal_at=("signal_at", "first"),
                       entry_date=("entry_date", "first"), entry_at=("entry_at", "first"),
                       sector_count=("ticker", "size"), ready_sectors=("row_signal_ready", "sum"),
                       signal_complete=("row_signal_ready", "all")).reset_index()
    daily["reason"] = ""
    for reason, mask in problems.items():
        # Ticker order is inherited from the fixed universe, not alphabetic ties.
        detail = (signals.loc[mask].groupby(group_keys, sort=False).ticker
                  .agg(lambda values: ",".join(values)).rename(reason))
        daily = daily.merge(detail, on=group_keys, how="left", validate="one_to_one")
        present = daily[reason].notna()
        daily.loc[present, "reason"] += reason + ":" + daily.loc[present, reason] + ";"
        daily[reason + "_sectors"] = daily[reason].fillna("").map(
            lambda value: len(value.split(",")) if value else 0)
        daily = daily.drop(columns=reason)
    daily["reason"] = daily.reason.str.rstrip(";").replace("", "OK")

    ny = "America/New_York"
    daily["phase_start_at"] = daily["split"].map({
        phase: (pd.Timestamp(opens[0]) + pd.Timedelta(hours=9, minutes=30)).tz_localize(ny)
        for phase, opens in PHASE_OPENS.items()})
    daily["phase_end_at"] = daily["split"].map({
        phase: (pd.Timestamp(opens[1]) + pd.Timedelta(hours=9, minutes=30)).tz_localize(ny)
        for phase, opens in PHASE_OPENS.items()})
    # The last data date is outside this phase's order schedule by prior design.
    after_end = daily.date.ge(daily.phase_end_at.dt.tz_localize(None).dt.normalize())
    final_open = ~after_end & daily.entry_at.eq(daily.phase_end_at)
    regular = (~after_end & ~final_open & daily.entry_at.ge(daily.phase_start_at)
               & daily.entry_at.lt(daily.phase_end_at))
    if not (after_end | final_open | regular).all():
        raise ValueError("Missing or inconsistent next-open calendar inside a scheduled phase")
    daily["action"] = "NO_ORDER_OUTSIDE_PHASE"
    daily.loc[regular & daily.signal_complete, "action"] = "EVALUATE_SIGNAL"
    daily.loc[regular & ~daily.signal_complete, "action"] = "CLOSE_ALL_TO_CASH"
    daily.loc[final_open, "action"] = "FINAL_LIQUIDATION"
    daily.loc[final_open, "reason"] = "scheduled_phase_end;" + daily.loc[final_open, "reason"]
    daily.loc[after_end, "reason"] = "outside_phase_order_schedule;" + daily.loc[after_end, "reason"]
    daily["force_flat"] = daily.action.isin(["CLOSE_ALL_TO_CASH", "FINAL_LIQUIDATION"])
    daily["can_evaluate_signal"] = daily.action.eq("EVALUATE_SIGNAL")
    daily["action_at"] = daily.entry_at.where(~after_end)
    instruction_fields = group_keys + ["signal_complete", "action", "action_at", "force_flat",
                                       "can_evaluate_signal", "reason"]
    signals = signals.merge(daily[instruction_fields], on=group_keys, how="left",
                            validate="many_to_one")
    signals = signals.sort_values(group_keys + ["ticker"]).reset_index(drop=True)
    daily = daily.sort_values(group_keys).reset_index(drop=True)
    return signals, daily


def build_scoring_eligibility(signals, panel):
    """Use the shared scoring mask; daily trading instructions do not filter it."""
    import numpy as np
    import pandas as pd

    fields = ["date", "ticker", "common_scoreable", "label_valid", "label_within_split",
              "target_relative_return"]
    scoring = signals[["date", "ticker", "model", "scheme", "split", "prediction_finite"]].merge(
        panel[fields], on=["date", "ticker"], how="left", validate="many_to_one")
    scoring["scoreable_by_panel"] = scoring.common_scoreable.fillna(False).astype(bool)
    scoring["target_available"] = np.isfinite(pd.to_numeric(scoring.target_relative_return, errors="coerce"))
    scoring["score_eligible"] = (scoring.scoreable_by_panel & scoring.prediction_finite
                                  & scoring.target_available)
    return scoring


def guard_existing_pairs(active_pairs, instruction):
    """Return a target-pair guard; execution/fees are implemented in later steps.

    EVALUATE_SIGNAL leaves the current pairs available for the later threshold
    logic. Close-all and final liquidation request an empty target pair set.
    """
    return {} if instruction["force_flat"] else dict(active_pairs)


def validate_signal_policy(panel, forecasts, calendar, *, models, schemes, tickers):
    """Small in-memory fault cases; never alter handoff inputs or run a backtest."""
    import numpy as np
    import pandas as pd

    dates = pd.to_datetime(["2020-03-24", "2022-12-28", "2022-12-29", "2022-12-30",
                            "2026-09-28", "2026-09-29", "2026-09-30"])
    p = panel.loc[panel.date.isin(dates)].copy()
    f = forecasts.loc[forecasts.date.isin(dates)].copy()
    c = calendar.loc[calendar.date.isin(dates)].copy()
    kwargs = dict(models=models, schemes=schemes, tickers=tickers)
    base_signals, base_daily = build_trading_status(p, f, c, **kwargs)
    records = []

    def check(name, passed, detail):
        records.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    def unchanged(left, right):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        return True

    # Removing all future fields is stronger than checking column names alone.
    minimal_signals, minimal_daily = build_trading_status(p[ASOF_PANEL_FIELDS], f, c, **kwargs)
    check("future_fields_can_be_removed", unchanged(base_signals, minimal_signals)
          and unchanged(base_daily, minimal_daily), "Trading policy runs with no target, label, exit or scoring columns")
    poisoned = p.copy()
    for field in set(p.columns) - set(ASOF_PANEL_FIELDS):
        poisoned[field] = "future_value_deliberately_changed"
    poisoned_signals, poisoned_daily = build_trading_status(poisoned, f, c, **kwargs)
    check("future_values_do_not_change_actions", unchanged(base_signals, poisoned_signals)
          and unchanged(base_daily, poisoned_daily), "All non-whitelisted panel fields changed; signals/actions identical")

    target = (f.date.eq(pd.Timestamp("2020-03-24")) & f.model.eq(models[-1])
              & f.scheme.eq(schemes[1]))
    one = target & f.ticker.eq("XLF")
    original_pairs = {"XLB": 1, "XLE": -1}

    def verify_close(case, altered_panel, altered_forecasts, expected_missing):
        signals, daily = build_trading_status(altered_panel, altered_forecasts, c, **kwargs)
        affected = daily.loc[daily.date.eq(pd.Timestamp("2020-03-24"))
                             & daily.model.eq(models[-1]) & daily.scheme.eq(schemes[1])].iloc[0]
        check(case, affected.action == "CLOSE_ALL_TO_CASH" and not affected.can_evaluate_signal
              and affected.force_flat and affected.action_at == pd.Timestamp("2020-03-25 09:30", tz="America/New_York")
              and affected.missing_prediction_sectors == expected_missing
              and guard_existing_pairs(original_pairs, affected) == {}
              and original_pairs == {"XLB": 1, "XLE": -1}
              and len(signals) == len(base_signals),
              "Affected strategy requests next-open flat targets; date/universe retained; original pair state unchanged")
        return daily

    missing_one = verify_close("one_missing_prediction", p, f.loc[~one], 1)
    other = ~(base_daily.date.eq(pd.Timestamp("2020-03-24"))
              & base_daily.model.eq(models[-1]) & base_daily.scheme.eq(schemes[1]))
    check("other_strategies_unaffected", unchanged(base_daily.loc[other].reset_index(drop=True),
          missing_one.loc[other].reset_index(drop=True)), "A single missing model forecast does not close other strategies")
    verify_close("whole_strategy_day_missing", p, f.loc[~target], len(tickers))
    for name, value in (("nan", np.nan), ("positive_infinity", np.inf), ("negative_infinity", -np.inf)):
        nonfinite = f.copy()
        nonfinite.loc[one, "y_hat"] = value
        verify_close(f"nonfinite_prediction_{name}", p, nonfinite, 0)
    future_fit = f.copy()
    future_fit.loc[one, "fit_cutoff"] = pd.Timestamp("2020-03-24 18:01", tz="America/New_York")
    verify_close("fit_after_signal", p, future_fit, 0)
    bad_features = p.copy()
    bad_features.loc[bad_features.date.eq(pd.Timestamp("2020-03-24"))
                     & bad_features.ticker.eq("XLF"), "common_features_valid"] = False
    bad_daily = verify_close("invalid_feature_flag", bad_features, f, 0)
    check("shared_input_failure_closes_six_strategies", bad_daily.loc[
        bad_daily.date.eq(pd.Timestamp("2020-03-24")), "force_flat"].all(),
        "Common-feature failure applies to every strategy on that date")
    missing_panel = p.loc[~(p.date.eq(pd.Timestamp("2020-03-24")) & p.ticker.eq("XLF"))]
    verify_close("missing_input_row", missing_panel, f, 0)

    label_missing = p.copy()
    label_missing.loc[label_missing.date.eq(pd.Timestamp("2020-03-24")),
                      ["common_scoreable", "label_valid", "label_within_split"]] = False
    label_missing.loc[label_missing.date.eq(pd.Timestamp("2020-03-24")), "target_relative_return"] = np.nan
    same_signals, same_daily = build_trading_status(label_missing, f, c, **kwargs)
    scoring = build_scoring_eligibility(same_signals, label_missing)
    check("missing_label_blocks_scoring_only", unchanged(base_daily, same_daily)
          and not scoring.loc[scoring.date.eq(pd.Timestamp("2020-03-24")), "score_eligible"].any(),
          "An interior missing future label changes scoring only; trade policy unchanged")
    expected = {"2022-12-28": "EVALUATE_SIGNAL", "2022-12-29": "FINAL_LIQUIDATION",
                "2022-12-30": "NO_ORDER_OUTSIDE_PHASE", "2026-09-28": "EVALUATE_SIGNAL",
                "2026-09-29": "FINAL_LIQUIDATION", "2026-09-30": "NO_ORDER_OUTSIDE_PHASE"}
    check("scheduled_phase_end_actions", all(base_daily.loc[base_daily.date.eq(pd.Timestamp(day)),
          "action"].eq(action).all() for day, action in expected.items()),
          "Prespecified dates control final liquidation and no-order days without future labels")
    return pd.DataFrame(records)
