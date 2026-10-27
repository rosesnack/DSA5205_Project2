
### Elastic Net results and interpretation

- All three pre-specified schemes are retained; there is no overall winner. The common search uses rho ∈ [0.2, 0.4, 0.6, 0.8] and alpha ∈ [1e-05, 0.0001, 0.001]. Validation-selected parameters: S0: alpha=0.001, rho=0.8; S1: alpha=0.001, rho=0.8; S2: alpha=0.001, rho=0.8.
- Validation-selected MSEs lie within 8.48e-09 overall; the S1–S2 MSE gap is 2.03e-19. Interpret very small MSE differences with the paired losses and numerical precision in mind.
- Test R² vs mature historical mean: S0 -0.047%, S1 -0.047%, S2 -0.047%. Test forecasts are exactly identical across all schemes: **True** (including the final unscored dates).
- Locked test fits have mean nonzero coefficient count 0.348, median 0, range 0–3; 87.41% are intercept-only. Extra cross-sector/relationship feature selection rate is 0.00% at most across the relevant groups.
- For sector-level interpretation, see Section 10 (XLF concentration diagnostic), which reports coefficient-selection concentration, sector-level test losses and XLF's validation performance across training windows. These descriptive checks do not retune the locked test model.
- The validation-selected pair (alpha=1e-3, rho=0.8) is the upper corner of the pre-specified grid for every scheme, and 87.41% of locked test fits are intercept-only. The grid was fixed before any forecast was generated and is not revised after observing test results. The forecasts should therefore be read as close to the historical-mean forecast, and the absence of cross-sector coefficients is a statement about this grid, not about cross-sector information in general.
- Test forecasts coincide across S0/S1/S2 under the validation-selected models. This is a result about this design and period, not a general claim that cross-sector effects never exist. Sector, rolling-loss, coefficient and uncertainty diagnostics are descriptive and are not used to retune the model.
- All 945 monthly-sector eligibility records match upstream. Final checks confirm unique forecasts, all three schemes, nine sectors per signal date, finite predictions, mature training labels, complete eligible key coverage and all 54 terminal unscored forecasts.
- Deliver the clean seven-column forecasts and full audit file to yae for unified validation/backtesting. No independently specified portfolio metrics are part of this contribution.

