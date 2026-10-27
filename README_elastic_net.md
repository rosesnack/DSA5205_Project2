# 03 · Elastic Net: sector relative-return prediction

**Contribution scope:** Elastic Net modelling, time-ordered validation, forecast evaluation, diagnostics and forecast export.

This notebook continues from the frozen outputs of notebooks 01–02 and implements the complete Elastic Net forecasting pipeline for the three pre-specified feature schemes S0, S1 and S2. It ends at model forecasts, statistical prediction diagnostics and export to `yae`. Portfolio construction, trading-threshold selection, position persistence, SPY netting and cost-aware backtesting belong to `yae`'s unified trading pipeline and are not performed here.

## Economic motivation

The predictive question is whether sector-specific information and additional cross-sector information improve forecasts of next-period sector-minus-SPY returns. Because overlapping return windows, common market conditions and related sector movements generate correlated predictors, Elastic Net is used to combine L1 sparse selection with L2 coefficient stabilization.

For standardized predictors, sklearn Elastic Net minimizes

\[
\frac{1}{2n}\|y-b-X\beta\|_2^2
+
\alpha\rho\|\beta\|_1
+
\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2,
\]

where \(\rho\) is `l1_ratio`. The intercept is unpenalized and the target remains in decimal-return units. This is a predictive framework rather than a causal identification strategy.

## Feature schemes

Three pre-specified schemes are retained throughout:

- **S0:** 57 own-sector and shared market-state predictors.
- **S1:** 97 predictors, adding information from other sectors.
- **S2:** 102 predictors, adding relationship-summary features.

The objective is not to select one overall winning scheme. Instead, each scheme receives its own validation-selected Elastic Net hyperparameters so that their subsequent forecast performance can be compared on identical observations.

## Hyperparameter selection

Both the main expanding-window model and the Elastic Net component of Section 9 use the same grid:

\[
\alpha \in \{10^{-5},10^{-4},10^{-3}\},
\qquad
\rho \in \{0.2,0.4,0.6,0.8\}.
\]

This gives 12 parameter pairs per scheme and 36 scheme/parameter combinations in the main validation search.

For each scheme, one shared \((\alpha,\rho)\) pair is selected across its nine independently fitted sector models using pooled MSE over the 2018–2022 validation period. Exact numerical ties prefer the larger alpha and then the larger L1 ratio. The selected parameters are then locked before the 2023–September 2026 test evaluation.

The current validation search selects

- S0: `alpha = 0.001`, `rho = 0.8`
- S1: `alpha = 0.001`, `rho = 0.8`
- S2: `alpha = 0.001`, `rho = 0.8`

The selected specification lies at the strong-regularization boundary of the preset grid and produces highly sparse fitted models. This boundary behaviour is therefore reported explicitly as a limitation of the current specification; absence of selected cross-sector coefficients should be interpreted conditional on this regularization design rather than as universal evidence that cross-sector information has no predictive value.

## Time ordering and leakage control

Training begins in 2008. Validation covers 2018–2022 and test evaluation covers 2023–September 2026.

Models are refitted monthly using an expanding historical window. Each monthly fit occurs at the previous SPY trading day's 18:00 New York timestamp. Both the predictor observation and its target label must be available by the fitting cutoff. StandardScaler is fitted only on the eligible historical training observations for that fit, and the target is not standardized.

No random cross-validation, full-sample scaling, future-label use or test-based hyperparameter selection is introduced.

## Diagnostics and interpretation

Prediction evaluation includes aggregate and sector-level RMSE, MAE, OOS \(R^2\), direction accuracy, rolling losses, coefficient sparsity and stability, and approximate uncertainty diagnostics.

The validation-selected test models are highly sparse, with most monthly-sector fits containing only an intercept. S0, S1 and S2 consequently produce identical test forecasts under the selected specification. Sector-level diagnostics are therefore important for interpreting the aggregate result, including the concentration of negative OOS performance in XLF.

These diagnostics describe the fitted models and sample period. They are not used to retune the locked test models.

Section 9 provides additional **validation-only** diagnostics across expanding, rolling 5-year and rolling 3-year windows and includes Ridge comparisons. These comparisons are descriptive and do not replace the main locked Elastic Net test models.

## Reproduction note

The final notebook uses a unified four-value L1-ratio grid throughout the main model and Section 9. Earlier development runs had already inspected the test period before this final unified rerun. Therefore, the 2023–September 2026 test interval should not be described as a newly untouched holdout after the grid refinement.

The final run records its environment, source-file hashes, selected parameters, fit counts, integrity checks and reproduction provenance in `results/elastic_net/run_manifest.json`.
