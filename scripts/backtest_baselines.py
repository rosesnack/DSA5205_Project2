"""Zero/expanding historical-mean forecast baselines; no trading or model tuning."""

from __future__ import annotations

BASELINE_MODELS = ("zero", "historical_mean")
BASELINE_SCHEME = "COMMON"


def training_mask(info, cutoff):
    """Same mature-label training rule as notebook 03, including <= boundaries."""
    import pandas as pd

    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("fit_cutoff must be timezone-aware")
    return (info.date.ge(pd.Timestamp("2008-01-01")) & info.common_features_valid
            & info.label_valid & info.signal_at.le(cutoff)
            & info.label_available_at.le(cutoff))


def forecast_metrics(frame, keys):
    """Forecast errors and OOS R-squared on the same scored observations."""
    import numpy as np
    import pandas as pd

    rows = []
    for key, group in frame.groupby(keys, sort=True):
        y = group.target_relative_return.to_numpy(float)
        prediction = group.y_hat.to_numpy(float)
        mean = group.historical_mean.to_numpy(float)
        sse = np.square(y - prediction).sum()
        mean_sse = np.square(y - mean).sum()
        zero_sse = np.square(y).sum()
        # A zero forecast expresses no directional view.
        direction = np.nan if np.equal(prediction, 0).all() else np.mean(np.sign(prediction) == np.sign(y))
        rows.append(dict(zip(keys, key)) | dict(
            n=len(group), MSE=float(sse / len(group)), RMSE=float(np.sqrt(sse / len(group))),
            MAE=float(np.mean(np.abs(y - prediction))),
            R2_OOS_mean=float(1 - sse / mean_sse) if mean_sse else np.nan,
            R2_OOS_zero=float(1 - sse / zero_sse) if zero_sse else np.nan,
            direction_accuracy=float(direction)))
    return pd.DataFrame(rows)


def build_baseline_tables(panel, main_forecasts, full_exports, upstream_monthly, recorded_metrics, *, fields):
    """Reuse delivered historical means after independent sample/mean verification."""
    import hashlib
    import numpy as np
    import pandas as pd

    checks = []

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    keys = ["date", "ticker"]
    forecast_keys = keys + ["model", "scheme"]
    # Establish agreement of both full exports with their seven-column handoffs.
    monthly_exports = {}
    for model, full in full_exports.items():
        clean = main_forecasts.loc[main_forecasts.model.eq(model)]
        matched = clean.merge(full[list(fields) + ["historical_mean"]], on=forecast_keys,
                              how="outer", suffixes=("_clean", "_full"),
                              validate="one_to_one", indicator=True)
        check(f"full_export_keys_{model}", matched["_merge"].eq("both").all(),
              "Full export and seven-column handoff contain exactly the same forecast keys")
        check(f"full_export_values_{model}", all(matched[field + "_clean"].eq(
            matched[field + "_full"]).all() for field in ("model_month", "fit_cutoff", "y_hat")),
            "Full export corresponds to the same predictions and fitting timestamps")
        check(f"one_mean_per_month_sector_{model}", full.groupby(
            ["model_month", "ticker"]).historical_mean.nunique().eq(1).all()
            and np.isfinite(full.historical_mean).all(),
            "The delivered mean is constant across days/schemes within each monthly sector fit")
        monthly_exports[model] = full[["model_month", "ticker", "historical_mean"]].drop_duplicates()

    anchor = full_exports["elastic_net"].loc[lambda values: values.scheme.eq("S0"),
        ["date", "ticker", "model_month", "fit_cutoff", "historical_mean"]].copy()
    check("canonical_keys_unique", not anchor.duplicated(keys).any(), "One common baseline per date/sector")
    schedule = anchor[["model_month", "fit_cutoff"]].drop_duplicates()
    check("one_cutoff_per_month", not schedule.model_month.duplicated().any(), "All baseline sectors share the model schedule")
    audit_rows = []
    for ticker, info in panel.groupby("ticker", sort=True):
        info = info.sort_values("date").reset_index(drop=True)
        for month, cutoff in schedule.sort_values("model_month").itertuples(index=False, name=None):
            train = info.loc[training_mask(info, cutoff)]
            if train.empty or not np.isfinite(train.target_relative_return).all():
                raise AssertionError(f"Invalid historical-mean training labels: {month}/{ticker}")
            key_text = "\n".join(train.date.dt.strftime("%Y-%m-%d") + "," + train.ticker)
            audit_rows.append(dict(model_month=month, ticker=ticker, fit_cutoff=cutoff,
                training_rows=len(train), first_training_signal=train.date.min(),
                latest_training_signal=train.date.max(),
                latest_label_available_at=train.label_available_at.max(),
                eligible_cross_split_rows=int((~train.label_within_split).sum()),
                computed_mean=float(train.target_relative_return.to_numpy(float).mean()),
                training_keys_sha256=hashlib.sha256(key_text.encode("utf-8")).hexdigest()))
    audit = pd.DataFrame(audit_rows).sort_values(["model_month", "ticker"]).reset_index(drop=True)
    upstream = upstream_monthly.rename(columns={"prediction_month": "model_month"})
    crosscheck = audit.merge(upstream, on=["model_month", "ticker"], how="outer",
                             validate="one_to_one", indicator=True, suffixes=("", "_upstream"))
    check("upstream_training_keys", crosscheck["_merge"].eq("both").all(),
          f"All {len(audit):,} month/sector training records match the upstream schedule")
    check("upstream_training_sample_counts", crosscheck.training_rows.eq(crosscheck.training_rows_common).all()
          and crosscheck.eligible_cross_split_rows.eq(crosscheck.eligible_cross_split_rows_upstream).all(),
          "Training and mature cross-partition sample counts match notebook 02")
    check("upstream_training_times", crosscheck.fit_cutoff.eq(crosscheck.fit_at).all()
          and crosscheck.latest_training_signal.eq(crosscheck.latest_training_signal_upstream).all()
          and crosscheck.latest_label_available_at.eq(crosscheck.latest_label_available_at_upstream).all(),
          "Monthly fitting and latest included signal/label times match notebook 02")
    check("training_labels_mature", audit.latest_label_available_at.le(audit.fit_cutoff).all(),
          "Every training label was known by the monthly cutoff")
    for model, monthly in monthly_exports.items():
        name = model + "_export_mean"
        audit = audit.merge(monthly.rename(columns={"historical_mean": name}),
                            on=["model_month", "ticker"], how="left", validate="one_to_one")
        audit[model + "_abs_difference"] = (audit.computed_mean - audit[name]).abs()
        check(f"recomputed_mean_matches_{model}", np.isclose(
            audit.computed_mean, audit[name], rtol=1e-12, atol=1e-16).all(),
            f"Recomputed from mature training labels; max absolute difference={audit[model + '_abs_difference'].max():.3g}")
    check("exported_means_agree", np.isclose(audit.elastic_net_export_mean,
          audit.gbdt_export_mean, rtol=1e-12, atol=1e-16).all(), "Both model families delivered the same monthly mean baseline")

    parts = []
    for model in BASELINE_MODELS:
        frame = anchor.copy()
        frame["model"], frame["scheme"] = model, BASELINE_SCHEME
        frame["y_hat"] = 0.0 if model == "zero" else frame.historical_mean
        parts.append(frame[list(fields)])
    predictions = pd.concat(parts, ignore_index=True).sort_values(forecast_keys).reset_index(drop=True)
    check("baseline_keys_and_zero_values", not predictions.duplicated(forecast_keys).any()
          and predictions.loc[predictions.model.eq("zero"), "y_hat"].eq(0).all(),
          "Two nonduplicated common baselines; every zero forecast equals zero")

    all_forecasts = pd.concat([main_forecasts, predictions], ignore_index=True)
    comparison = all_forecasts.merge(panel[keys + ["split", "common_scoreable", "target_relative_return"]],
                                     on=keys, how="left", validate="many_to_one")
    comparison = comparison.merge(anchor[keys + ["historical_mean"]], on=keys, how="left",
                                   validate="many_to_one")
    scored = comparison.loc[comparison.common_scoreable].copy()
    for split, group in scored.groupby("split"):
        samples = [set(map(tuple, part[keys].to_numpy())) for _, part in group.groupby(["model", "scheme"])]
        check(f"identical_scoring_samples_{split}", all(sample == samples[0] for sample in samples)
              and np.isfinite(group[["y_hat", "historical_mean", "target_relative_return"]]).all().all(),
              "All six model strategies and both baselines share exactly the same scoring keys")
    metrics = forecast_metrics(scored, ["model", "scheme", "split"])
    baseline_metrics = metrics.loc[metrics.model.isin(BASELINE_MODELS)].reset_index(drop=True)
    sector_metrics = forecast_metrics(scored.loc[scored.model.isin(BASELINE_MODELS)],
                                      ["model", "scheme", "split", "ticker"])
    for model, original in recorded_metrics.items():
        observed = metrics.loc[metrics.model.eq(model)].merge(original, on=["scheme", "split"],
                    validate="one_to_one", suffixes=("", "_recorded"))
        names = ["n", "RMSE", "MAE", "R2_OOS_mean", "R2_OOS_zero", "direction_accuracy"]
        check(f"reproduce_recorded_metrics_{model}", len(observed) == len(original) and all(
            np.isclose(observed[name], observed[name + "_recorded"], rtol=1e-12, atol=1e-12).all()
            for name in names), "Existing main-model error metrics reproduced without changing forecasts")

    # Independent small fixture: mean(0.01, 0.03)=0.02. Other rows fail different
    # eligibility constraints; the first included label matures exactly at cutoff.
    cutoff = pd.Timestamp("2020-01-31 18:00", tz="America/New_York")
    fixture = pd.DataFrame(dict(
        date=pd.to_datetime(["2008-01-02", "2008-01-03", "2007-12-31", "2008-01-04", "2008-01-05", "2008-01-06", "2008-01-07"]),
        common_features_valid=[True, True, True, False, True, True, True],
        label_valid=[True, True, True, True, False, True, True],
        signal_at=[cutoff - pd.Timedelta(days=1)] * 6 + [cutoff + pd.Timedelta(minutes=1)],
        label_available_at=[cutoff, cutoff - pd.Timedelta(days=1), cutoff, cutoff, cutoff,
                            cutoff + pd.Timedelta(minutes=1), cutoff],
        target_relative_return=[0.01, 0.03, 99.0, 98.0, 97.0, 96.0, 95.0]))
    fixture_mask = training_mask(fixture, cutoff)
    check("hand_calculated_training_fixture", fixture_mask.tolist() == [True, True, False, False, False, False, False]
          and np.isclose(fixture.loc[fixture_mask, "target_relative_return"].mean(), 0.02),
          "Hand-calculated mean verifies warmup, invalid features/labels, future signal and maturity boundaries")
    poisoned = fixture.copy()
    poisoned.loc[~fixture_mask, "target_relative_return"] = -1e9
    check("unavailable_labels_do_not_change_mean", fixture.loc[fixture_mask, "target_relative_return"].mean()
          == poisoned.loc[training_mask(poisoned, cutoff), "target_relative_return"].mean(),
          "Changing all ineligible/future targets leaves the historical mean unchanged")
    poisoned["common_scoreable"] = False
    poisoned["label_within_split"] = False
    check("scoring_flags_do_not_filter_training", training_mask(poisoned, cutoff).equals(fixture_mask),
          "Mature historical samples can cross evaluation partitions; scoring flags are not training filters")
    return {"baseline_predictions.csv": predictions, "monthly_baseline_audit.csv": audit,
            "baseline_forecast_metrics.csv": baseline_metrics, "prediction_comparison_metrics.csv": metrics,
            "sector_baseline_metrics.csv": sector_metrics, "baseline_checks.csv": pd.DataFrame(checks)}
