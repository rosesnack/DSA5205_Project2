# Elastic Net contribution — notebook 03

`notebooks/03_elastic_net.ipynb` implements the complete forecast pipeline and prediction diagnostics for S0/S1/S2. Both the main model and Section 9 use:

```python
ALPHAS = [1e-5, 1e-4, 1e-3]
L1_RATIOS = [0.2, 0.4, 0.6, 0.8] 
```

The main validation search has 12 parameter pairs per scheme, 36 scheme/parameter combinations and 19,440 monthly-sector fits. Choose one pair per scheme by pooled 2018–2022 validation MSE; exact ties prefer larger alpha then larger rho. Retain all three schemes. Monthly expanding training fits StandardScaler only on eligible historical observations with mature labels. Test evaluation covers 2023–September 2026 with locked hyperparameters (1,215 monthly-sector fits).

Section 9 uses the same four rho values for expanding, rolling 5-year and rolling 3-year windows, alongside four Ridge penalties. There are 144 configurations and 77,760 monthly-sector validation fits, including 58,320 Elastic Net fits. All windows use identical scoring keys and the common expanding-mean baseline. These validation-selected comparisons are descriptive; they do not replace the main locked test models.

## Run

Use Python 3.12.10 and the dependencies in `requirements_elastic_net.txt`. Open the notebook from the repository root or `notebooks/`, select the project Python kernel, and run all cells in order. The delivered notebook contains the executed outputs. The run manifest records the execution environment, input hashes, grid and reproduction provenance.

## Outputs

All current results are in `results/elastic_net/`:

- `validation_grid.csv`, `selected_hyperparameters.csv`, `design_config.json`, `locked_selection.json`: complete search and selected parameters.
- `fit_audit.csv`, `monthly_eligibility_crosscheck.csv`, `monthly_coefficients.csv`: monthly training, label maturity, numerical convergence and coefficients.
- `forecast_metrics.csv`, `sector_forecast_metrics.csv`, rolling losses, uncertainty and coefficient diagnostics, plus six main figures.
- `elastic_net_predictions.csv`: clean handoff with columns `date,ticker,model,scheme,model_month,fit_cutoff,y_hat`.
- `elastic_net_predictions_full.csv`: forecasts with split, parameters, targets, historical baseline, scoring eligibility and holding dates.
- `report_summary.md`, `run_manifest.json`: computed findings and integrity checks.
- `exploratory_validation/`: Section 9's 144-configuration validation comparison, fit audits, coefficient stability and six figures, including four-ratio sensitivity curves.

`y_hat` is a decimal sector-minus-SPY cash-return forecast. The last two signal dates retain forecasts but are unscored. Portfolio construction, trading thresholds and unified cost-aware backtesting belong to yae.
