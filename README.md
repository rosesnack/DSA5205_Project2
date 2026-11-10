# DSA5205 Project 2: Sector Relative-Return Prediction

This project studies whether cross-sector information improves forecasts of sector ETF returns relative to SPY. Elastic Net and gradient boosted decision trees (GBDT) are evaluated on a common sample. The portfolio analysis compares sector–SPY pairs and a SPY-core overlay after trading costs.

Download `video-group12.mp4` from the **Assets** section of the [Group 12 presentation release](https://github.com/rosesnack/DSA5205_Project2/releases/tag/submission-video). Use the repository's `main` branch for the source code.

## Reproduction

Use Python 3.12. Run the setup commands from the repository root.

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name dsa5205 --display-name "Python (DSA5205)"
.venv/bin/python -m notebook
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name dsa5205 --display-name "Python (DSA5205)"
.\.venv\Scripts\python.exe -m notebook
```

Select the **Python (DSA5205)** kernel. Run the notebooks from top to bottom in this order.

| Notebook | Work performed | Main outputs |
|---|---|---|
| [01_download_clean_data.ipynb](notebooks/01_download_clean_data.ipynb) | Verify the frozen source files and align prices, returns and Treasury yields. | Cleaned tables in `data/interim/` and acquisition and cleaning checks. |
| [02_features_panel_data.ipynb](notebooks/02_features_panel_data.ipynb) | Build the features and prediction targets, then assemble the model panels and eligibility records. | `data/processed/model_panel.csv`, `panel_spec.json` and 27 sector-by-feature-set inputs. |
| [03_elastic_net.ipynb](notebooks/03_elastic_net.ipynb) | Fit and validate Elastic Net, then score and export its forecasts. | Predictions, validation scores and diagnostics in `results/elastic_net/`. |
| [04_gbdt.ipynb](notebooks/04_gbdt.ipynb) | Fit and validate GBDT, then evaluate its forecasts alongside Elastic Net. | Predictions, validation scores and diagnostics in `results/gbdt/`. |
| [05_portfolio_backtest.ipynb](notebooks/05_portfolio_backtest.ipynb) | Recompute the forecasting baselines, select the pair threshold on validation data and evaluate the portfolios, benchmarks and cost scenarios. | Fresh pair run with metrics, ledgers, accounting checks and figures in `results/reproduction/`. |
| [06_spy_overlay_final_results.ipynb](notebooks/06_spy_overlay_final_results.ipynb) | Verify the completed pair run and evaluate all 12 SPY-core candidates with dividend-reinvestment SPY as a benchmark. | Fresh overlay run with candidate rankings, benchmark comparisons and figures. |

Keep the [47 source snapshots](data/raw/) and [download manifest](notebooks/download_manifest.json) together. Notebook 01 verifies their SHA-256 hashes before reuse and regenerates the cleaned tables. A new download can reflect provider revisions. Finish both forecast notebooks before starting 05, and finish 05 before starting 06.

The daily sample runs from 2006 through September 2026. At 18:00 New York time the models forecast the next-open-to-following-open cash return of each sector ETF relative to SPY. The nested feature sets S0, S1 and S2 contain 57, 97 and 102 predictors per sector. The initial training period is 2008–2017, validation is 2018–2022 and testing is 2023–September 2026. Each monthly fit uses only labels available at its cutoff.

After notebooks 01–04 have completed, the portfolio stages can also be run from the repository root.

```bash
.venv/bin/python -B scripts/fresh_backtest_pipeline.py --stage all
.venv/bin/python -B -m unittest discover -s scripts -p test_fresh_backtest_pipeline.py -v
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `.venv/bin/python`. The second command runs the portfolio integration checks on a small fixture.

## Report artifact correspondence

The report identifies the saved experiment snapshot at commit `bc7b92f`. These tables, figures and records provide the evidence for the reported results. Fresh runs generate a separate set of outputs.

| Report evidence | Saved artifacts |
|---|---|
| Source checks and cleaning | [Download manifest](notebooks/download_manifest.json), [source checks](notebooks/quality_checks/) and [cleaning checks](notebooks/cleaning_checks/). |
| Features, targets and sample eligibility | [Panel specification](data/processed/panel_spec.json), [model panel](data/processed/model_panel.csv), [feature checks](notebooks/feature_checks/) and [panel checks](notebooks/panel_checks/). |
| Elastic Net selection, forecasts and diagnostics | [Elastic Net results](results/elastic_net/), including `selected_hyperparameters.csv`, `validation_grid.csv`, `elastic_net_predictions.csv`, `forecast_metrics.csv` and `fit_audit.csv`. |
| Supplementary Elastic Net comparisons | [Validation experiments](results/elastic_net/exploratory_validation/) and [XLF diagnostics](results/elastic_net/xlf_diagnostics/). |
| GBDT selection, forecasts and diagnostics | [GBDT results](results/gbdt/), including `model_config.json`, `validation_round_curve.csv`, `gbdt_predictions.csv`, `forecast_metrics.csv` and `fit_audit.csv`. |
| Dollar-neutral threshold choice and results | [Selected threshold](results/backtest/threshold_selection/selected_threshold.json), [validation scores](results/backtest/threshold_selection/validation_candidate_scores.csv), [portfolio metrics](results/backtest/reporting/main_metrics.csv) and [cost sensitivities](results/backtest/reporting/sensitivity_metrics.csv). |
| SPY-core experiments and final comparison | [Validation results](results/backtest/spy_overlay_validation_20261101/), [test results](results/backtest/spy_overlay_test_20261101/) and [final comparison](results/backtest/spy_overlay_final_20261101/). |

The forecast exports use the following seven columns.

```text
date, ticker, model, scheme, model_month, fit_cutoff, y_hat
```

`y_hat` is the predicted sector-minus-SPY cash return in decimal units. The `*_predictions_full.csv` files include realized targets, eligibility flags and holding-period dates for the portfolio analysis.

The archived SPY-core presentation uses GBDT S1 with 15 bp entry and 7.5 bp exit thresholds. It earned 22.723% net annualized return in testing, compared with 22.019% for dividend-reinvestment SPY. The choice was made after comparing the test returns of all 12 candidates, so this result is exploratory. The selection and cost details are recorded in [the final comparison](results/backtest/spy_overlay_final_20261101/README.md).

## Fresh portfolio outputs

Each 05 run writes to `results/reproduction/<run_id>/`. The file `results/reproduction/latest_pair_run.json` points to the latest completed run. Notebook 06 checks the parent run's inputs, code and output hashes before writing a new child run in `overlay/<overlay_run_id>/`.

| Run | Files to inspect |
|---|---|
| Pair run | `run_config.json` and `complete.json` record the settings and completion. `thresholds/validation_candidate_scores.csv` records threshold selection. `pair/main_metrics.csv`, `benchmarks/`, `sensitivities/metrics.csv` and `figures/` contain the evaluation. |
| Overlay run | `all_12_candidates_ranked.csv`, `selected_and_benchmarks.csv`, `selection_record.json` and `selected_{validation,test}.png` summarize the comparison. `checks.csv` and `complete.json` record validation and completion. |

If upstream inputs or scripts change, run 05 again before 06.
