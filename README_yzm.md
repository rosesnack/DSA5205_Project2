# DSA5205 Project 2: Data Acquisition and Feature Preparation

This package prepares daily data for testing whether cross-sector information improves predictions of sector ETF returns relative to SPY. Notebook 01 acquires and cleans public data; notebook 02 constructs features, prediction targets and model inputs. Model fitting and portfolio evaluation are separate stages.

## Installation and execution

Use Python 3.12. The package was developed with Python 3.12.10, and dependency versions are pinned in `requirements.txt`.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name dsa5205proj2-data --display-name "Python (DSA5205 Data)"
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsa5205proj2-data --ExecutePreprocessor.timeout=900 notebooks/01_download_clean_data.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsa5205proj2-data --ExecutePreprocessor.timeout=900 notebooks/02_features_panel_data.ipynb
```

A clean Python 3.12.10 environment reproduced all 37 derived data files exactly from the 47 supplied raw snapshots.

Execute the notebooks in this order. The explicit kernel name selects the environment installed above. Notebook 01 creates the cleaned tables and their manifest; notebook 02 requires those outputs. Both notebooks create missing output directories and regenerate quality reports.

## Reproduction routes

**Frozen snapshot:** retain all 47 files in `data/raw/` and the matching `notebooks/download_manifest.json`, then run the commands above. Notebook 01 verifies checksums and reuses these immutable inputs. Notebook 02 runs offline. This route reproduces the submitted dataset without requiring another market-data download.

**Fresh acquisition:** use a separate copy with an empty `data/raw/` directory and no download manifest, then execute both notebooks. Internet access is required. Provider revisions or rate limits can change coverage, prices and distributions. Several checks describe the frozen snapshot, so changed observations require review and may fail those checks. Preserve the submitted snapshot for comparison.

## Files

| Path | Purpose |
|---|---|
| `notebooks/01_download_clean_data.ipynb` | Downloads, verifies, cleans and aligns source observations |
| `notebooks/02_features_panel_data.ipynb` | Constructs features, labels and sector-specific model inputs |
| `data/raw/` | 47 unchanged provider snapshots |
| `data/interim/` | Aligned prices, cash returns, rates, features, targets and validity flags |
| `data/processed/model_panel.csv` | Shared targets, timestamps and eligibility metadata |
| `data/processed/model_inputs/` | 27 predictor CSVs: nine sectors × three specifications |
| `data/processed/panel_spec.json` | Ordered predictor lists and file interfaces |
| `notebooks/*_checks/` | Quality reports, manifests and field definitions |

The repository stores `data/interim/sector_features.csv.gz` because the uncompressed CSV exceeds GitHub's 100 MiB file limit. Compression is lossless. Notebook 02 always regenerates the complete `sector_features.csv`; the 27 final model-input files remain ordinary CSVs. To inspect the supplied compressed copy directly, use `pd.read_csv("data/interim/sector_features.csv.gz")`.

## Research definitions

The requested period is 2006-01-01 through 2026-09-30. Prediction targets are XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY. Sources are Yahoo Finance through `yfinance`, Cboe VIX and FRED DGS3MO. SPY provides the trading calendar and return reference; TLT, IEF, GLD, HYG and LQD provide auxiliary information.

Signals are observed at 18:00 New York time. The target is the sector's next-open-to-following-open cash return minus SPY's corresponding return. Dividends accrue on the ex-dividend date. Adjusted prices are diagnostic inputs only. Unverified XLF distribution effects are excluded explicitly; ETF prices are not forward-filled.

Each sector has 57 S0 predictors, 97 S1 predictors including other sectors, and 102 S2 predictors including relationship summaries. Join predictor files to `model_panel.csv` on `(date, ticker)`. Consult `notebooks/feature_checks/feature_catalog.csv`, `metadata_dictionary.csv` and `reason_codes.csv` for definitions.

Training begins in 2008; validation covers 2018–2022 and testing covers 2023–2026-09. Monthly expanding-window training must respect label availability. Use the notebook's shared training filter and common evaluation samples. Fit preprocessing only on eligible historical training observations. The final two signal dates retain features but lack complete future labels.
