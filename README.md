# DSA5205 Project 2: Sector Relative-Return Prediction

This project studies whether cross-sector information improves forecasts of sector ETF returns relative to SPY. It provides a complete workflow for data acquisition, cleaning, feature construction, monthly model fitting, forecast evaluation and prediction export. Elastic Net supplies a regularized linear benchmark; Gradient Boosted Decision Trees (GBDT) capture nonlinear relationships and feature interactions.

## Project structure

| Path | Contents |
|---|---|
| [01_download_clean_data.ipynb](notebooks/01_download_clean_data.ipynb) | Source acquisition, calendar alignment, cash returns and data-quality checks |
| [02_features_panel_data.ipynb](notebooks/02_features_panel_data.ipynb) | Features, prediction targets, sample eligibility and model input files |
| [03_elastic_net.ipynb](notebooks/03_elastic_net.ipynb) | Elastic Net validation search, monthly forecasts, evaluation and coefficient diagnostics |
| [04_gbdt.ipynb](notebooks/04_gbdt.ipynb) | GBDT validation search with early stopping, monthly forecasts, evaluation and group permutation importance |
| `data/raw/` | 47 source snapshots |
| `data/interim/` | Aligned prices, returns, rates, features and targets |
| `data/processed/` | Shared sample panel, 27 feature tables and their specification |
| `notebooks/*_checks/` | Quality reports, data dictionaries and processing manifests |
| `results/elastic_net/` | Elastic Net predictions, selected parameters, metrics, diagnostics and figures |
| `results/gbdt/` | GBDT predictions, validation curves, selected parameters, metrics and figures |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |

## Installation and execution

Use Python 3.12. Create a local environment and install dependencies from the repository root.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Register the environment as a Jupyter kernel, then execute the notebooks in order:

```bash
python -m ipykernel install --sys-prefix --name dsa5205 --display-name "Python (DSA5205)"
cd notebooks
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsa5205 --ExecutePreprocessor.timeout=-1 01_download_clean_data.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsa5205 --ExecutePreprocessor.timeout=-1 02_features_panel_data.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsa5205 --ExecutePreprocessor.timeout=-1 03_elastic_net.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsa5205 --ExecutePreprocessor.timeout=-1 04_gbdt.ipynb
```

For interactive execution, run `python -m notebook` from the `notebooks/` directory, open a notebook and select **Python (DSA5205)**. Execute its cells from top to bottom. Notebook 04 reads the Elastic Net prediction export produced by notebook 03 for model comparisons.

The supplied `data/raw/` snapshots and `notebooks/download_manifest.json` define the submitted dataset. Notebook 01 verifies and reuses these snapshots; notebooks 02–04 work from local files. To rerun only the models, use the supplied processed data and execute notebooks 03 and 04. Editing the parameter cell and rerunning notebook 04 updates its selection tables, forecasts, metrics and figures.

The intermediate feature table is stored as `data/interim/sector_features.csv.gz`. Notebook 02 regenerates the uncompressed CSV; the 27 final model-input tables are ordinary CSV files.

## Data and prediction target

The data period is **2006-01-01 to 2026-09-30**. The nine target ETFs are:

| Ticker | Sector |
|---|---|
| XLB | Materials |
| XLE | Energy |
| XLF | Financials |
| XLI | Industrials |
| XLK | Technology |
| XLP | Consumer Staples |
| XLU | Utilities |
| XLV | Health Care |
| XLY | Consumer Discretionary |

ETF observations come from Yahoo Finance, VIX from Cboe and the three-month Treasury yield from FRED DGS3MO. SPY defines the trading calendar and return benchmark. TLT, IEF, GLD, HYG and LQD supply auxiliary asset information.

Signals are observed at **18:00 America/New_York**. A signal on trading day $t$ predicts the sector's next-open-to-following-open cash return relative to SPY:

$$
y_{i,t}=R^{\mathrm{cash}}_{i,t+1\rightarrow t+2}
       -R^{\mathrm{cash}}_{\mathrm{SPY},t+1\rightarrow t+2}.
$$

Both endpoints refer to market opens. Cash returns include dividends accruing on the ex-dividend date. The cleaning stage handles corporate actions, excludes unresolved XLF distribution effects and preserves missing ETF prices. Returns and forecasts are stored in decimal units.

## Features and model inputs

Each sector is evaluated under three feature schemes:

| Scheme | Predictors | Information included |
|---|---:|---|
| S0 | 57 | Own-sector returns, momentum, volatility, liquidity, SPY, market state and auxiliary assets |
| S1 | 97 | S0 plus 40 return features from the other eight sectors |
| S2 | 102 | S1 plus five sector ranking, dispersion and correlation summaries |

[panel_spec.json](data/processed/panel_spec.json) lists the input paths, ordered predictor names, target definition and sample rules. [model_panel.csv](data/processed/model_panel.csv) contains the shared targets, timestamps, split labels and eligibility flags. Predictor values are stored separately in `data/processed/model_inputs/{ticker}_{scheme}.csv`; their keys are `(date, ticker)`.

Detailed definitions are in [feature_catalog.csv](notebooks/feature_checks/feature_catalog.csv), [metadata_dictionary.csv](notebooks/feature_checks/metadata_dictionary.csv) and [reason_codes.csv](notebooks/feature_checks/reason_codes.csv).

## Training and validation

Models are fitted separately for each sector and scheme using monthly expanding windows beginning in **2008**. The fitting cutoff is 18:00 New York time on the SPY trading day preceding each forecast month. Training includes observations whose features and target labels are available by that cutoff.

| Period | Role |
|---|---|
| 2008–2017 | Initial training history |
| 2018–2022 | Monthly expanding-window validation and parameter selection |
| 2023–September 2026 | Monthly expanding-window test evaluation |

Each scheme selects one parameter configuration shared across its nine sector models by pooled validation MSE. Selected parameters remain fixed during test evaluation; the training sample expands as additional labels become available. The common scoring mask gives each scheme **11,313 validation observations** and **8,433 test observations**.

### Elastic Net

Elastic Net fits a linear model with L1 and L2 penalties:

$$
\min_{b,\beta}\frac{1}{2n}\lVert y-b-X\beta\rVert_2^2
+\alpha\rho\lVert\beta\rVert_1
+\frac{\alpha(1-\rho)}{2}\lVert\beta\rVert_2^2.
$$

Predictors are standardized using each monthly training sample. The intercept is unpenalized and the target remains in its original units.

The search uses `alpha = [1e-5, 1e-4, 1e-3]` and `l1_ratio = [0.2, 0.4, 0.6, 0.8]`: 12 configurations per scheme. Exact ties prefer larger alpha, then larger L1 ratio. All three schemes select **alpha = 0.001** and **l1_ratio = 0.8**.

Notebook 03 also examines coefficient sparsity and stability, sector performance, rolling losses, rolling three- and five-year training windows, and Ridge comparisons on the validation period.

### GBDT with early stopping

Histogram-based gradient boosting adds regression trees sequentially to fit residual structure:

$$
F_0(x)=\overline y_{\mathrm{train}},\qquad
F_m(x)=F_{m-1}(x)+\eta h_m(x).
$$

The search uses `max_leaf_nodes = [3, 5, 7]` and `l2_regularization = [0, 1, 10]`: nine configurations per scheme. Fixed settings are learning rate **0.03**, maximum depth **4**, minimum leaf size **20**, maximum histogram bins **127** and random seed **5205**.

Boosting rounds are selected using the pooled 2018–2022 validation MSE curve, with **100 maximum rounds**, **patience 10** and **tolerance 1e-8**. Monthly models first compute their complete boosting paths. The selection code scans each pooled curve until ten rounds pass without a significant improvement and retains the lowest-MSE round visited. It then selects the best leaf-count/L2 configuration for each scheme. Exact ties prefer fewer leaves, fewer rounds and larger L2.

This uses the existing chronological validation period. The estimator's internal early stopping is disabled, and the selected model is refitted on all eligible historical observations each month. Forecasts are exported under the model name `gbdt`.

| Scheme | Maximum leaves | L2 penalty | Selected rounds | Patience stopping round |
|---|---:|---:|---:|---:|
| S0 | 5 | 10 | 23 | 33 |
| S1 | 3 | 10 | 43 | 48 |
| S2 | 3 | 10 | 40 | 46 |

S1 is the validation-selected primary GBDT scheme. Group permutation importance measures the test MSE increase after jointly shuffling each feature group within a month, averaged over three permutations and aggregated across sectors and months.

## Forecast results

Evaluation reports MSE, RMSE, MAE, direction accuracy and out-of-sample $R^2$. The primary $R^2$ compares squared forecast errors with those of each monthly training sample's historical mean:

$$
R^2_{\mathrm{OOS}}=1-
\frac{\sum_{i,t}(y_{i,t}-\widehat y_{i,t})^2}
{\sum_{i,t}(y_{i,t}-\overline y_{i,\mathrm{train}(t)})^2}.
$$

The submitted results are shown below. **All values are decimals, not percentages.**

| Model | Scheme | Validation $R^2$ | Test $R^2$ | Test RMSE | Test direction accuracy |
|---|---|---:|---:|---:|---:|
| Elastic Net | S0 | -0.000691 | -0.000473 | 0.009566 | 0.508241 |
| Elastic Net | S1 | -0.000605 | -0.000473 | 0.009566 | 0.508241 |
| Elastic Net | S2 | -0.000605 | -0.000473 | 0.009566 | 0.508241 |
| GBDT | S0 | 0.002199 | 0.000261 | 0.009563 | 0.503735 |
| GBDT | S1 | 0.003587 | 0.000429 | 0.009562 | 0.497569 |
| GBDT | S2 | 0.003552 | 0.000403 | 0.009562 | 0.496383 |

GBDT achieves small positive test improvements over the historical-mean benchmark, with S1 giving the lowest aggregate test MSE. S2's additional relationship features do not improve on S1. Elastic Net produces highly sparse models and identical test forecasts across the three schemes. GBDT's test loss improvements over the historical mean and Elastic Net have approximate 95% Newey–West intervals that include zero; performance also varies across years and sectors.

![Validation and test model comparison](results/gbdt/01_model_comparison.png)

## Outputs

| Output | Contents |
|---|---|
| `results/{elastic_net,gbdt}/selected_hyperparameters.csv` | Validation-selected parameters for each scheme |
| `results/{elastic_net,gbdt}/validation_grid.csv` | Candidate validation scores |
| `results/{elastic_net,gbdt}/forecast_metrics.csv` | Aggregate validation and test metrics |
| `results/{elastic_net,gbdt}/sector_forecast_metrics.csv` | Sector-level forecast metrics |
| `results/{elastic_net,gbdt}/fit_audit.csv` | Monthly training samples, timestamps and fit details |
| `results/{elastic_net,gbdt}/loss_difference_uncertainty.csv` | Paired forecast-loss differences and uncertainty intervals |
| `results/elastic_net/run_manifest.json` | Environment, input hashes and execution records |
| `results/elastic_net/monthly_coefficients.csv` | Monthly Elastic Net coefficients |
| `results/gbdt/model_config.json` | GBDT search settings and selected configurations |
| `results/gbdt/validation_round_curve.csv` | Pooled validation loss at every boosting round |
| `results/gbdt/validation_monthly_losses.csv` | Monthly validation losses by configuration and round |
| `results/gbdt/model_comparison_metrics.csv` | Elastic Net and GBDT metrics on common samples |
| `results/gbdt/annual_forecast_metrics.csv` | Annual GBDT performance |
| `results/gbdt/group_permutation_importance.csv` | Test importance of each feature group |

The forecast exports are [elastic_net_predictions.csv](results/elastic_net/elastic_net_predictions.csv) and [gbdt_predictions.csv](results/gbdt/gbdt_predictions.csv). Both use the same seven-column interface:

```text
date, ticker, model, scheme, model_month, fit_cutoff, y_hat
```

`y_hat` is the predicted sector-minus-SPY cash return in decimal units. The corresponding `*_predictions_full.csv` files also contain realized targets, scoring flags, holding-period dates and historical-mean forecasts. These exports provide inputs for downstream portfolio construction and trading evaluation.

The GBDT figures cover model comparison (`01`), sector performance (`02`), annual performance (`03`) and feature-group importance (`04`). Elastic Net figures cover parameter selection, forecast accuracy, rolling losses and coefficient diagnostics.
