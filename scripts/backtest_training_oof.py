"""Training-only forward forecasts for signal-scale diagnostics (step 9).

The 2012 calibration year selects diagnostic parameters; 2013-2017 is forecast
with frozen parameters and monthly expanding fits. Official model files are
read-only. No portfolio return, public threshold selection or test evaluation
is performed here. Monthly hash-checked checkpoints support interrupted runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
from itertools import product
import json
import os
from pathlib import Path
import platform
from threading import Lock
import time
import warnings

# Set before importing sklearn; also constrain loaded numerical runtimes below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import threadpoolctl
from threadpoolctl import threadpool_limits

SEED = 5205
TICKERS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
SCHEMES = ("S0", "S1", "S2")
MODELS = ("elastic_net", "gbdt")
EN_GRID = list(product([1e-5, 1e-4, 1e-3], [.2, .4, .6, .8]))
GB_GRID = list(product([3, 5, 7], [10, 20, 50], [0., 1., 10.]))
GB_FIXED = dict(loss="squared_error", learning_rate=.03, max_depth=4,
                max_bins=127, early_stopping=False, max_features=1.,
                categorical_features=None, random_state=SEED)
PARAMS = ["scheme", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_iter"]
SELECTION_CUTOFF = pd.Timestamp("2012-12-31 18:00", tz="America/New_York")
FIELDS = ["date", "ticker", "model", "scheme", "model_month", "fit_cutoff", "y_hat"]
EN_WARNING_LOCK = Lock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def csv_write(path, frame):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", float_format="%.17g")
    temp.replace(path)


def read_csv(path, **kwargs):
    return pd.read_csv(path, low_memory=False, float_precision="round_trip", **kwargs)


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, threadpoolctl=threadpoolctl.__version__)


def effective_gb_fixed():
    fixed = GB_FIXED.copy()
    if "max_features" not in inspect.signature(HistGradientBoostingRegressor).parameters:
        # sklearn <1.4 always considers every allowed feature. 1.0 requests the
        # same all-feature setting; do not silently map any sampling fraction.
        assert fixed.pop("max_features") == 1.0
    return fixed


def source_hashes(root):
    previous = json.loads((root / "results/backtest/cost_manifest.json").read_text(encoding="utf-8"))
    hashes = dict(previous["source_hashes"])
    hashes.update({r["path"]: r["sha256"] for r in previous["outputs"]})
    paths = list((root / "data").rglob("*"))
    for folder in ("elastic_net", "gbdt"):
        paths += list((root / "results" / folder).glob("*.json"))
        paths += list((root / "results" / folder).glob("*hyperparameters*.csv"))
    for path in paths:
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            current = sha256(path)
            if relative in hashes and hashes[relative] != current:
                raise AssertionError(f"Upstream source changed: {relative}")
            hashes[relative] = current
    verify_sources(root, hashes)
    return hashes


def verify_sources(root, hashes):
    for relative, expected in hashes.items():
        if sha256(root / relative) != expected:
            raise AssertionError(f"Protected source changed: {relative}")


def register_plan(root, out, workers):
    """Write candidate grid and design before fitting; refuse mismatched reuse."""
    sources = source_hashes(root)
    plan = dict(schema_version=1, purpose="training-only signal-scale diagnostic, separate from official forecasts",
        seed=SEED, runtime=runtime(), workers=workers, numerical_threads=1,
        initial_history="2008-2011", calibration_year=2012, forecast_period="2013-2017",
        selection_cutoff=str(SELECTION_CUTOFF), min_training_rows=1000,
        expanding_training="date >= 2008-01-01, common_features_valid, label_valid, signal_at <= fit_cutoff, label_available_at <= fit_cutoff",
        fit_cutoff="prior SPY trading day 18:00 America/New_York before first monthly signal",
        calibration_scoring="common_scoreable AND label_available_at <= selection_cutoff; pooled MSE over all nine sectors",
        preprocessing="Elastic Net StandardScaler fit on each mature monthly training subset only; GBDT binning fit on that same subset",
        parameter_sharing="one frozen diagnostic configuration per model and S0/S1/S2; nine independent sector fits",
        elastic_net=dict(grid=[dict(alpha=a, l1_ratio=r) for a, r in EN_GRID],
            fixed=dict(fit_intercept=True, max_iter=10000, tol=1e-6, selection="cyclic", random_state=SEED, precompute=True),
            numerical_max_iter_retries=[10000, 100000, 1000000], tie="larger alpha, then larger l1_ratio"),
        gbdt=dict(grid=[dict(max_leaf_nodes=a, min_samples_leaf=b, l2_regularization=c) for a, b, c in GB_GRID],
            requested_fixed=GB_FIXED, effective_fixed=effective_gb_fixed(), max_rounds=100, patience=10, tolerance=1e-8,
            tie="fewer leaves, fewer rounds, larger L2, larger minimum leaf size"),
        compatibility_note="sklearn <1.4 has no max_features argument and uses all features; omit only the requested 1.0 argument. Cross-version bitwise reproduction of official fits is not claimed.",
        compatibility_reference="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
        candidate_entry_bps=[5, 10, 15, 20, 30], candidate_exit_bps=[2.5, 5, 7.5, 10, 15],
        candidate_source="5205proj2.md, section 3.4; fixed before these new fits and before full validation strategy evaluation",
        grid_changed_by_diagnostics=False, public_threshold_selected=False, selected_public_entry_bps=None,
        diagnostic_summaries="absolute forecasts in bps; strict threshold exceedance; complete-nine-sector daily top-four ranks; no portfolio returns",
        missing_features="skip only invalid rows/months; no imputation; complete-nine-sector dates only for cross-sectional diagnostics",
        source_hashes=sources, code_sha256=sha256(Path(__file__)))
    path = out / "design_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "registered_utc"} != plan:
            raise AssertionError("Diagnostic design/source/runtime changed; do not reuse incompatible checkpoints")
        plan = saved
    else:
        plan["registered_utc"] = datetime.now(timezone.utc).isoformat()
        json_write(path, plan)
    grid = pd.DataFrame(dict(entry_bps=plan["candidate_entry_bps"], exit_bps=plan["candidate_exit_bps"]))
    grid["selected"] = False
    grid_path = out / "threshold_candidates.csv"
    if grid_path.exists():
        pd.testing.assert_frame_equal(read_csv(grid_path), grid, check_dtype=False)
    else:
        csv_write(grid_path, grid)
    return plan, sha256(path)


def load_data(root):
    spec = json.loads((root / "data/processed/panel_spec.json").read_text(encoding="utf-8"))
    panel = read_csv(root / spec["shared_panel"], parse_dates=["date"])
    for col in ("signal_at", "label_available_at"):
        panel[col] = pd.to_datetime(panel[col], utc=True).dt.tz_convert("America/New_York")
    assert not panel.duplicated(["date", "ticker"]).any()
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    schedule = {}
    for period in pd.period_range("2012-01", "2017-12", freq="M"):
        first = calendar[calendar.to_period("M") == period][0]
        prior = calendar[calendar.get_loc(first) - 1]
        schedule[str(period)] = (prior + pd.Timedelta(hours=18)).tz_localize("America/New_York")
    data = {}
    for ticker in TICKERS:
        info = panel.loc[panel.ticker.eq(ticker) & panel.date.lt("2018-01-01")].sort_values("date").reset_index(drop=True)
        info["month"] = info.date.dt.strftime("%Y-%m")
        xs = {}
        for scheme in SCHEMES:
            frame = read_csv(root / spec["input_files"][ticker][scheme], parse_dates=["date"])
            columns = spec["predictors_by_target"][ticker][scheme]
            assert list(frame.columns) == ["date", "ticker"] + columns
            assert len(columns) == spec["predictor_counts"][scheme]
            assert not set(columns).intersection(spec["columns_not_predictors"])
            frame = frame.loc[frame.date.lt("2018-01-01")].reset_index(drop=True)
            assert frame[["date", "ticker"]].equals(info[["date", "ticker"]])
            xs[scheme] = frame[columns].to_numpy(float)
            assert np.isfinite(xs[scheme][info.common_features_valid]).all()
        data[ticker] = (info, xs)
    return data, schedule


def training_mask(info, cutoff):
    return (info.date.ge("2008-01-01") & info.common_features_valid & info.label_valid
            & info.signal_at.le(cutoff) & info.label_available_at.le(cutoff)).to_numpy()


def month_inputs(data, schedule, month):
    """Reuse only the current month's data, not all monthly train matrices."""
    prepared, skipped = {}, []
    cutoff = schedule[month]
    for ticker in TICKERS:
        info, xs = data[ticker]
        train = training_mask(info, cutoff)
        pred = (info.month.eq(month) & info.common_features_valid).to_numpy()
        if not pred.any():
            skipped.append(dict(model_month=month, ticker=ticker, fit_cutoff=cutoff,
                                reason="no common_features_valid prediction rows"))
            continue
        assert train.sum() >= 1000
        assert info.loc[train, "label_available_at"].max() <= cutoff
        assert info.loc[pred, "signal_at"].min() > cutoff
        y = info.loc[train, "target_relative_return"].to_numpy(float)
        assert np.isfinite(y).all()
        meta = info.loc[pred, ["date", "ticker", "signal_at", "label_available_at", "common_scoreable", "target_relative_return"]].copy()
        score = (meta.common_scoreable & meta.label_available_at.le(SELECTION_CUTOFF)).to_numpy()
        base = dict(model_month=month, ticker=ticker, fit_cutoff=cutoff, training_rows=int(train.sum()),
            first_training_date=str(info.loc[train, "date"].min().date()),
            last_training_date=str(info.loc[train, "date"].max().date()),
            latest_training_signal=str(info.loc[train, "signal_at"].max()),
            latest_training_label=str(info.loc[train, "label_available_at"].max()),
            first_prediction_signal=str(meta.signal_at.min()), last_prediction_signal=str(meta.signal_at.max()),
            prediction_rows=len(meta), calibration_scoring_rows=int(score.sum()),
            latest_calibration_scoring_label=str(meta.loc[score, "label_available_at"].max()) if score.any() else None)
        for scheme in SCHEMES:
            raw_train, raw_pred = xs[scheme][train], xs[scheme][pred]
            scaler = StandardScaler()
            scaled_train = np.asfortranarray(scaler.fit_transform(raw_train))
            scaled_pred = scaler.transform(raw_pred)
            assert int(scaler.n_samples_seen_) == int(train.sum())
            # Auditable fingerprint of fitted preprocessing, not future data.
            scale_hash = hashlib.sha256(scaler.mean_.tobytes() + scaler.scale_.tobytes()).hexdigest()
            prepared[ticker, scheme] = (raw_train, raw_pred, scaled_train, scaled_pred, y, meta, score,
                dict(base, scheme=scheme, feature_count=raw_train.shape[1], scaler_training_rows=int(scaler.n_samples_seen_), scaler_sha256=scale_hash))
    return prepared, skipped


def fit_en(x, y, alpha, l1_ratio):
    for retry, limit in enumerate((10000, 100000, 1000000)):
        fitted = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
            max_iter=limit, tol=1e-6, selection="cyclic", random_state=SEED, precompute=True)
        # Warning filters are process-global in this Python runtime. Serialize
        # EN fits so one worker cannot hide another worker's convergence warning.
        with EN_WARNING_LOCK, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted.fit(x, y)
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            return fitted, dict(n_iter=int(fitted.n_iter_), max_iter_used=limit,
                convergence_retry_count=retry, dual_gap=float(fitted.dual_gap_), nonzero=int(np.count_nonzero(fitted.coef_)))
    raise RuntimeError("Elastic Net did not converge; diagnostic fit rejected")


def calibration_job(task):
    model, key, parameters, prepared = task
    raw_x, raw_p, scaled_x, scaled_p, y, meta, score, base = prepared[key]
    actual = meta.loc[score, "target_relative_return"].to_numpy(float)
    assert len(actual) > 0
    if model == "elastic_net":
        a, r = parameters
        fit, detail = fit_en(scaled_x, y, a, r)
        sse = float(np.square(actual - fit.predict(scaled_p[score])).sum())
        loss = [dict(scheme=key[1], alpha=a, l1_ratio=r, n=len(actual), sse=sse)]
        audit = dict(base, model=model, alpha=a, l1_ratio=r, **detail)
    else:
        leaves, minimum, l2 = parameters
        fit = HistGradientBoostingRegressor(**effective_gb_fixed(), max_leaf_nodes=leaves,
            min_samples_leaf=minimum, l2_regularization=l2, max_iter=100).fit(raw_x, y)
        loss = [dict(scheme=key[1], max_leaf_nodes=leaves, min_samples_leaf=minimum,
            l2_regularization=l2, max_iter=i, n=len(actual), sse=float(np.square(actual - p).sum()))
            for i, p in enumerate(fit.staged_predict(raw_p[score]), 1)]
        audit = dict(base, model=model, max_leaf_nodes=leaves, min_samples_leaf=minimum,
            l2_regularization=l2, max_iter=100, fitted_rounds=int(fit.n_iter_))
        # No fitted StandardScaler is part of a GBDT model.
        for field in ("scaler_training_rows", "scaler_sha256"):
            audit.pop(field)
    return loss, audit


def forecast_job(task):
    model, key, config, prepared = task
    raw_x, raw_p, scaled_x, scaled_p, y, meta, _, base = prepared[key]
    if model == "elastic_net":
        fit, detail = fit_en(scaled_x, y, **config)
        prediction = fit.predict(scaled_p)
        audit = dict(base, model=model, **config, **detail)
    else:
        fit = HistGradientBoostingRegressor(**effective_gb_fixed(), **config).fit(raw_x, y)
        prediction = fit.predict(raw_p)
        audit = dict(base, model=model, **config, fitted_rounds=int(fit.n_iter_))
        for field in ("scaler_training_rows", "scaler_sha256"):
            audit.pop(field)
    assert np.isfinite(prediction).all()
    frame = meta[["date", "ticker"]].copy()
    frame["model"], frame["scheme"] = model, key[1]
    frame["model_month"], frame["fit_cutoff"] = base["model_month"], base["fit_cutoff"]
    frame["y_hat"] = prediction
    # Future labels are deliberately not exported into the diagnostic interface.
    return frame[FIELDS], audit


def checkpoint(out, phase, month, identity, frames=None):
    folder = out / "checkpoints"
    folder.mkdir(exist_ok=True)
    manifest = folder / f"{phase}_{month}.json"
    if frames is None:
        if not manifest.exists():
            return None
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        assert saved["identity"] == identity, "Checkpoint identity mismatch"
        for item in saved["outputs"]:
            assert sha256(out / item["path"]) == item["sha256"], "Checkpoint hash mismatch"
        return {item["name"]: read_csv(out / item["path"]) for item in saved["outputs"]}
    records = []
    for name, frame in frames.items():
        path = folder / f"{phase}_{month}_{name}.csv"
        csv_write(path, frame)
        records.append(dict(name=name, path=path.relative_to(out).as_posix(), rows=len(frame), sha256=sha256(path)))
    json_write(manifest, dict(identity=identity, completed_utc=datetime.now(timezone.utc).isoformat(), outputs=records))
    return frames


def select_parameters(out, en_losses, gb_losses):
    en = en_losses.groupby(["scheme", "alpha", "l1_ratio"], as_index=False)[["n", "sse"]].sum()
    en["calibration_mse"] = en.sse / en.n
    en = en.sort_values(["scheme", "calibration_mse", "alpha", "l1_ratio"], ascending=[True, True, False, False])
    en_selected = en.groupby("scheme", sort=False).head(1).reset_index(drop=True)
    gb = gb_losses.groupby(PARAMS, as_index=False)[["n", "sse"]].sum()
    gb["calibration_mse"] = gb.sse / gb.n
    choices = []
    for _, curve in gb.groupby(PARAMS[:-1]):
        curve = curve.sort_values("max_iter")
        reference, stale = np.inf, 0
        for row in curve.itertuples():
            if row.calibration_mse < reference - 1e-8:
                reference, stale = row.calibration_mse, 0
            else:
                stale += 1
            stop = row.max_iter
            if stale >= 10:
                break
        best = curve.loc[curve.max_iter.le(stop)].sort_values(["calibration_mse", "max_iter"]).iloc[0].to_dict()
        best["stopping_round"] = int(stop)
        choices.append(best)
    ranking = ["scheme", "calibration_mse", "max_leaf_nodes", "max_iter", "l2_regularization", "min_samples_leaf"]
    search = pd.DataFrame(choices).sort_values(ranking, ascending=[True, True, True, True, False, False])
    gb_selected = search.groupby("scheme", sort=False).head(1).reset_index(drop=True)
    for name, frame in [("elastic_net_calibration_grid.csv", en), ("gbdt_calibration_round_curve.csv", gb),
                        ("gbdt_calibration_grid.csv", search), ("elastic_net_selected_diagnostic.csv", en_selected),
                        ("gbdt_selected_diagnostic.csv", gb_selected)]:
        csv_write(out / name, frame)
    configs = dict(elastic_net={r.scheme: dict(alpha=float(r.alpha), l1_ratio=float(r.l1_ratio)) for r in en_selected.itertuples()},
        gbdt={r.scheme: dict(max_leaf_nodes=int(r.max_leaf_nodes), min_samples_leaf=int(r.min_samples_leaf),
            l2_regularization=float(r.l2_regularization), max_iter=int(r.max_iter)) for r in gb_selected.itertuples()})
    lock = dict(selected=configs, selection_cutoff=str(SELECTION_CUTOFF),
                frozen_before_2013_forecasts=True, official_models_replaced=False)
    path = out / "selected_diagnostic_config.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == lock
    else:
        json_write(path, lock)
    return configs, sha256(path)


def signal_summaries(predictions, out):
    quantiles, exceedance, ranks, day_rows = [], [], [], []
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions.date)
    for (model, scheme), group in predictions.groupby(["model", "scheme"]):
        for scope, rows in [("all_valid_sector_rows", group), *[(str(y), g) for y, g in group.groupby(group.date.dt.year)]]:
            bps = rows.y_hat.abs().to_numpy() * 10000
            record = dict(model=model, scheme=scheme, scope=scope, rows=len(rows), dates=rows.date.nunique(),
                          mean_abs_bps=float(bps.mean()), max_abs_bps=float(bps.max()), zero_fraction=float((bps == 0).mean()))
            record.update({f"p{q:g}_abs_bps": float(np.percentile(bps, q)) for q in (25, 50, 75, 90, 95, 99)})
            quantiles.append(record)
            for threshold in (5, 10, 15, 20, 30):
                exceedance.append(dict(model=model, scheme=scheme, scope=scope, entry_bps=threshold,
                    rows=len(rows), exceeding_rows=int((bps > threshold).sum()), fraction=float((bps > threshold).mean())))
        for date, day in group.groupby("date"):
            complete = len(day) == 9 and set(day.ticker) == set(TICKERS)
            row = dict(date=date, model=model, scheme=scheme, valid_sectors=len(day), complete_nine=complete)
            if complete:
                day = day.assign(abs_bps=day.y_hat.abs() * 10000,
                                 ticker_order=day.ticker.map({t: i for i, t in enumerate(TICKERS)}))
                day = day.sort_values(["abs_bps", "ticker_order"], ascending=[False, True]).head(4).copy()
                day["rank"] = np.arange(1, 5)
                ranks.extend(day[["date", "ticker", "model", "scheme", "rank", "y_hat", "abs_bps"]].to_dict("records"))
                for threshold in (5, 10, 15, 20, 30):
                    row[f"top4_above_{threshold}bps"] = int(day.abs_bps.gt(threshold).sum())
            day_rows.append(row)
    tables = {"signal_quantiles.csv": pd.DataFrame(quantiles), "threshold_exceedance.csv": pd.DataFrame(exceedance),
              "complete_day_top4.csv": pd.DataFrame(ranks), "daily_signal_coverage.csv": pd.DataFrame(day_rows)}
    for name, frame in tables.items():
        csv_write(out / name, frame)
    return tables


def validate_results(data, schedule, predictions, fit_audit, calibration_audit, skipped, plan, configs, tables):
    checks = []
    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name)
    check("unique_seven_column_forecasts", list(predictions.columns) == FIELDS and not predictions.duplicated(FIELDS[:4]).any(), "Separate interface, no label fields")
    dates = pd.to_datetime(predictions.date)
    check("only_2013_2017_diagnostic_signals", dates.between("2013-01-01", "2017-12-31").all(), "No validation/test signals used as diagnostic forecasts")
    check("all_six_strategies", len(predictions.groupby(["model", "scheme"])) == 6, "No model/specification selected or removed")
    expected = set()
    for ticker, (info, _) in data.items():
        for date in info.loc[info.common_features_valid & info.date.ge("2013-01-01"), "date"]:
            expected.add((str(date.date()), ticker))
    for model in MODELS:
        for scheme in SCHEMES:
            actual = predictions.loc[predictions.model.eq(model) & predictions.scheme.eq(scheme)]
            check(f"exact_valid_coverage_{model}_{scheme}", set(zip(actual.date.astype(str).str[:10], actual.ticker)) == expected,
                  "Exactly valid common-feature rows; no imputed XLF or discarded valid individual-sector rows")
    for phase, audit in [("calibration", calibration_audit), ("oof", fit_audit)]:
        cutoffs = pd.to_datetime(audit.fit_cutoff, utc=True)
        check(f"{phase}_mature_labels_only", (pd.to_datetime(audit.latest_training_label, utc=True) <= cutoffs).all(), "Every fitted target available by monthly cutoff")
        check(f"{phase}_past_signals_only", (pd.to_datetime(audit.latest_training_signal, utc=True) <= cutoffs).all(), "Every fitted feature row existed by cutoff")
        check(f"{phase}_future_prediction_rows", (pd.to_datetime(audit.first_prediction_signal, utc=True) > cutoffs).all(), "Forecasts strictly after fit cutoff")
        check(f"{phase}_training_since_2008", pd.to_datetime(audit.first_training_date).ge("2008-01-01").all() and audit.training_rows.ge(1000).all(), "Same minimum history and start as original fits")
        en = audit.loc[audit.model.eq("elastic_net")]
        check(f"{phase}_training_only_scaler", en.scaler_training_rows.eq(en.training_rows).all(), "Scaler sample count equals eligible training count")
        check(f"{phase}_en_convergence", en.n_iter.le(en.max_iter_used).all() and np.isfinite(en.dual_gap).all(), "Convergence warnings rejected or numerically retried")
    latest_score = pd.to_datetime(calibration_audit.latest_calibration_scoring_label, utc=True)
    check("calibration_labels_known_before_2013", latest_score.le(SELECTION_CUTOFF.tz_convert("UTC")).all(), "Late-December labels not yet mature at selection are excluded")
    expected_scores = sum(int((i.common_features_valid & i.common_scoreable & i.date.dt.year.eq(2012) & i.label_available_at.le(SELECTION_CUTOFF)).sum()) for i, _ in data.values())
    for model in MODELS:
        selected = read_csv(Path(plan["_out"]) / f"{model}_selected_diagnostic.csv")
        check(f"{model}_identical_pooled_scoring_rows", selected.n.eq(expected_scores).all(), "Selected candidates score identical mature 2012 rows across nine sectors")
        for scheme in SCHEMES:
            subset = fit_audit.loc[fit_audit.model.eq(model) & fit_audit.scheme.eq(scheme)]
            check(f"frozen_config_{model}_{scheme}", all(subset[k].eq(v).all() for k, v in configs[model][scheme].items()), "All 2013-2017 fits use frozen 2012-only parameters")
    for date, ticker, cutoff in predictions[["date", "ticker", "fit_cutoff"]].drop_duplicates().itertuples(index=False, name=None):
        assert pd.Timestamp(cutoff) == schedule[str(date)[:7]]
    check("exact_monthly_cutoffs", True, "Every prediction cutoff equals prior SPY trading day 18:00 NY")
    coverage = tables["daily_signal_coverage.csv"]
    top = tables["complete_day_top4.csv"]
    check("top4_only_complete_nine_sector_dates", len(top) == int(coverage.complete_nine.sum()) * 4, "Partial dates retained separately and never used for nine-sector ranking")
    check("threshold_grid_not_selected", plan["candidate_entry_bps"] == [5, 10, 15, 20, 30] and not plan["public_threshold_selected"], "Fixed protocol candidates; no portfolio-return evaluation")
    check("invalid_months_explicit", len(skipped) > 0 and skipped.ticker.eq("XLF").any(), "No valid-row month silently filled")
    empty_months = [month for month in schedule if month >= "2013-01" and not any(
        (info.month.eq(month) & info.common_features_valid).any() for info, _ in data.values())]
    check("empty_months_have_no_fits_or_forecasts", all(
        not predictions.model_month.eq(month).any() and not fit_audit.model_month.eq(month).any()
        and set(skipped.loc[skipped.model_month.eq(month), "ticker"]) == set(TICKERS)
        for month in empty_months), "Wholly invalid months create header-only checkpoints and nine skip records")
    return pd.DataFrame(checks)


def run_training_oof(project_root, workers=4):
    root = Path(project_root).resolve()
    out = root / "results/backtest/training_oof"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan, identity = register_plan(root, out, workers)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["design_sha256"] == identity
        for item in manifest["outputs"]:
            assert sha256(root / item["path"]) == item["sha256"], f"Completed output changed: {item['path']}"
        print("Step 9: verified completed diagnostic cache; no refit", flush=True)
        return manifest
    print("Step 9: design and threshold grid registered before fits;", runtime(), flush=True)
    data, schedule = load_data(root)
    en_chunks, gb_chunks, calibration_audits, oof_chunks, oof_audits, skipped_chunks = [], [], [], [], [], []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for month in [m for m in schedule if m.startswith("2012")]:
            saved = checkpoint(out, "calibration", month, identity)
            if saved is None:
                prepared, skipped = month_inputs(data, schedule, month)
                tasks = [(model, key, param, prepared) for model, grid in [("elastic_net", EN_GRID), ("gbdt", GB_GRID)]
                         for key in prepared for param in grid]
                results = list(pool.map(calibration_job, tasks))
                losses = {m: [] for m in MODELS}
                audits = []
                for task, (rows, audit) in zip(tasks, results):
                    losses[task[0]].extend(rows)
                    audits.append(audit)
                en = pd.DataFrame(losses["elastic_net"]).groupby(["scheme", "alpha", "l1_ratio"], as_index=False)[["n", "sse"]].sum()
                gb = pd.DataFrame(losses["gbdt"]).groupby(PARAMS, as_index=False)[["n", "sse"]].sum()
                en["model_month"], gb["model_month"] = month, month
                saved = checkpoint(out, "calibration", month, identity,
                    dict(elastic_net_losses=en, gbdt_losses=gb, fit_audit=pd.DataFrame(audits)))
            en_chunks.append(saved["elastic_net_losses"])
            gb_chunks.append(saved["gbdt_losses"])
            calibration_audits.append(saved["fit_audit"])
            print(f"Calibration {month}: complete | elapsed {time.perf_counter()-started:.1f}s", flush=True)
        en_losses, gb_losses = pd.concat(en_chunks, ignore_index=True), pd.concat(gb_chunks, ignore_index=True)
        configs, selected_hash = select_parameters(out, en_losses, gb_losses)
        print("Frozen diagnostic configs:", json.dumps(configs), flush=True)
        forecast_identity = identity + ":" + selected_hash
        for month in [m for m in schedule if m >= "2013-01"]:
            saved = checkpoint(out, "oof", month, forecast_identity)
            if saved is None:
                prepared, skipped = month_inputs(data, schedule, month)
                tasks = [(model, key, configs[model][key[1]], prepared) for model in MODELS for key in prepared]
                results = list(pool.map(forecast_job, tasks))
                saved = checkpoint(out, "oof", month, forecast_identity, dict(
                    predictions=pd.concat([r[0] for r in results], ignore_index=True)
                        if results else pd.DataFrame(columns=FIELDS),
                    fit_audit=pd.DataFrame([r[1] for r in results])
                        if results else pd.DataFrame(columns=calibration_audits[0].columns),
                    skipped=pd.DataFrame(skipped, columns=["model_month", "ticker", "fit_cutoff", "reason"])))
            oof_chunks.append(saved["predictions"])
            oof_audits.append(saved["fit_audit"])
            skipped_chunks.append(saved["skipped"])
            print(f"OOF {month}: complete | elapsed {time.perf_counter()-started:.1f}s", flush=True)
    predictions = pd.concat(oof_chunks, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions.date).dt.strftime("%Y-%m-%d")
    predictions = predictions.sort_values(["model", "scheme", "date", "ticker"]).reset_index(drop=True)
    fit_audit = pd.concat(oof_audits, ignore_index=True)
    calibration_audit = pd.concat(calibration_audits, ignore_index=True)
    skipped = pd.concat(skipped_chunks, ignore_index=True)
    tables = signal_summaries(predictions, out)
    checks = validate_results(data, schedule, predictions, fit_audit, calibration_audit, skipped, dict(plan, _out=str(out)), configs, tables)
    for name, frame in [("training_oof_predictions.csv", predictions), ("oof_fit_audit.csv", fit_audit),
                        ("calibration_fit_audit.csv", calibration_audit), ("skipped_sector_months.csv", skipped),
                        ("elastic_net_calibration_monthly_losses.csv", en_losses), ("gbdt_calibration_monthly_losses.csv", gb_losses),
                        ("oof_checks.csv", checks)]:
        csv_write(out / name, frame)
    verify_sources(root, plan["source_hashes"])
    output_paths = [*out.iterdir(), *(out / "provenance").glob("*")]
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p))
               for p in sorted(output_paths) if p.is_file() and p != manifest_path]
    manifest = dict(schema_version=1, scope="Step 9: completed training-only forward signal-scale diagnostics",
        completed_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime(), design_sha256=identity,
        selected_diagnostic_config_sha256=selected_hash, selected_diagnostic_configs=configs,
        calibration_fit_count=len(calibration_audit), oof_fit_count=len(fit_audit), prediction_rows=len(predictions),
        first_signal_date=predictions.date.min(), last_signal_date=predictions.date.max(),
        skipped_sector_months=len(skipped), complete_nine_sector_days_per_strategy=int(tables["daily_signal_coverage.csv"].groupby(["model", "scheme"]).complete_nine.sum().iloc[0]),
        valid_sector_rows_per_strategy=len(predictions) // 6, checks=checks.to_dict("records"),
        threshold_candidates_bps=plan["candidate_entry_bps"], selected_public_entry_bps=None,
        threshold_selection_performed=False, portfolio_returns_computed=False, official_predictions_overwritten=False,
        source_hashes=plan["source_hashes"], code_hashes={"scripts/backtest_training_oof.py": sha256(Path(__file__))},
        outputs=outputs, elapsed_seconds=time.perf_counter()-started,
        notes=["2012 is a calibration year, not included in the final OOF scale summary",
               "Each monthly fit uses only labels mature at its cutoff; calibration scores also mature before 2013",
               "Parameters selected by 2018-2022 official validation are never imported into diagnostic fitting",
               "Feature definitions and candidate grids were already designed; this is a retrospective training diagnostic, not an untouched historical holdout",
               "Actual local software versions are recorded; cross-version bitwise equivalence to official models is not claimed",
               "Grid unchanged regardless of diagnostic exceedance; no profitability claim follows from exceeding a threshold"],
        not_yet_checked=["public threshold selection on validation under main costs (step 10)",
                         "protocol freeze and full validation/test strategy evaluation (steps 11-14)"])
    json_write(manifest_path, manifest)
    print(f"PASS: {len(checks)} OOF checks; {len(predictions):,} diagnostic forecasts; source hashes unchanged", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run_training_oof(args.project_root, args.workers)
