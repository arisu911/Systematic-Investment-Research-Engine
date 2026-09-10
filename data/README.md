# Data Storage Directory

This directory stores:
- `cache/`: High-speed local Parquet files for downloaded market and macroeconomic series.
- `demo/`: Deterministic synthetic and pre-bundled benchmark datasets for offline and Streamlit Cloud execution.
- User-supplied CSV or Parquet files for custom backtesting.

## Custom Data Format
To load custom local CSV files, name the file with the asset ticker (e.g. `1155.KL.csv` or `SPY.csv`) containing the standard headers:
`Date,Open,High,Low,Close,Volume`
with Date as index formatted `YYYY-MM-DD`.
