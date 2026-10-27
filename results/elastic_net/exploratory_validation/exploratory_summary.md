# Exploratory validation findings

Selection and comparison use 2018-2022 validation only. The Elastic Net grid is identical to the main pipeline.

144 candidate specifications, 77,760 monthly-sector fits. All use identical validation scoring keys.

Primary criterion: pooled MSE within model/window/scheme. Sector-equal selection is sensitivity analysis.

R2 values below use the SAME expanding-history mean baseline for every window.

All displayed winners are selected on these validation dates; their apparent advantages are selection-biased.

- S0: expanding Elastic Net R2=-0.0691%; lowest explored validation MSE: Ridge/expanding, lambda=10, R2=0.1133%; MSE reduction vs expanding Elastic Net=0.1823%. This is descriptive, not a promoted final model.

- S1: expanding Elastic Net R2=-0.0605%; lowest explored validation MSE: Ridge/expanding, lambda=10, R2=0.1288%; MSE reduction vs expanding Elastic Net=0.1892%. This is descriptive, not a promoted final model.

- S2: expanding Elastic Net R2=-0.0605%; lowest explored validation MSE: Ridge/expanding, lambda=10, R2=0.1307%; MSE reduction vs expanding Elastic Net=0.1911%. This is descriptive, not a promoted final model.

- Selection criteria agree in 18/18 family/window/scheme comparisons.

- Ridge nonzero counts do not demonstrate sparse selection or causal effects.

- Inspect annual and sector tables before interpreting aggregate improvements; avoid cherry-picking years/sectors.

- Section 9 leaves the main locked parameters and the main test/forecast exports unchanged.

- These validation comparisons do not promote an alternative model for the test period.
