# PR title

Add SPY-core backtests and final result handoff

# PR body

The cash-dominated pair portfolio left most capital outside the equity market. This extension adds a 100% SPY core with sector-relative long/short adjustments, a 150% target gross cap, net SPY execution, restricted short collateral and explicit financing costs.

It includes the completed 15/7.5 bp and 10/5 bp validation/test comparisons, all twelve candidate configurations, full saved ledgers, a dividend-reinvestment SPY comparator, and year/sector/holding-episode/cost attribution. Results remain under the existing scripts, notebooks and results/backtest folders. The final presentation configuration is GBDT S1 at 15/7.5 bp: test net CAGR 22.723% versus 22.019% for reinvestment SPY, with maximum drawdown -19.303%. It was selected ex post by observed test net CAGR; the evidence is explicitly exploratory and all alternatives are retained.

No forecasts were replaced, no models retrained and no further parameter values evaluated in the synchronization preparation. README and handoff notes now point to the current results; the five documentation changes retain exact pre-edit snapshots. Existing trading engines, notebooks 01-05 and all original numerical results remain unchanged.

Validation: `python -B scripts/verify_spy_overlay_delivery.py` checks the delivery and historical source/output hashes without replay. Notebook 06 has six saved read-only code-cell outputs verified sequentially in Python; native Jupyter kernel execution is not claimed. ZIP handoff archives are excluded from this GitHub delivery.
