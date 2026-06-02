# Data

> **No proprietary financial data is distributed with this repository.**
> The original market data used in the paper was obtained from commercial /
> licensed vendors and **cannot be redistributed**. This directory ships only a
> small **synthetic example** that documents the expected schema, plus the policy
> below for supplying your own data.

PRIME is designed to run **out-of-the-box without any real data**: when a panel is
missing — or does not cover the configured train/valid/test windows — the pipeline
automatically synthesizes a **market-aware mock panel** that preserves realistic
cross-sectional and regime structure (see `generate_market_mock_panel` in
[`prime/data_loader.py`](../prime/data_loader.py)).

---

## Three ways to provide data

| # | Source | What happens |
|---|--------|--------------|
| **1** | **Nothing** (default) | The pipeline auto-generates mock data under `data/<market>/panel/` and proceeds. Great for trying the code and reproducing the *mechanics* of every experiment. |
| **2** | **Example panels** (`data/example/`) | Small synthetic panels you can inspect or load directly to verify the schema and column semantics. |
| **3** | **Your own panel** | Place a real panel at `data/<market>/panel/panel_data_complete.parquet` (or `.csv`), or point elsewhere via `--data_root /abs/path` / the `PRIME_DATA_ROOT` env var. |

```bash
# Example: use a custom data root (real or mock data lands here)
export PRIME_DATA_ROOT=/abs/path/to/data
python prime/main.py --mode train --market_profile csi500 --asset_type stock

# Or per-run:
python prime/main.py --mode train --data_root /abs/path/to/data/china
```

### Expected layout

```text
<data_root>/
└── <market>/                       # china, us, japan, europe, china_etf, …
    └── panel/
        ├── panel_data_complete.parquet   # stocks  (preferred; .csv also accepted)
        └── etf_panel_complete.parquet    # ETFs
```

Stock loaders try, in order: `panel_data_complete.parquet` → `panel_data_complete.csv`
→ `panel_data.parquet` → `panel_data.csv`. ETF loaders try
`etf_panel_complete.parquet` → `etf_panel_complete.csv`.

---

## Example dataset (`data/example/`)

| File | Rows | Description |
|------|------|-------------|
| `csi500_stock_example.csv` / `.parquet` | 8 tickers × ~30 trading days | Synthetic **stock** panel (78 columns) |
| `csi500_etf_example.csv` | 6 tickers × ~30 trading days | Synthetic **ETF** panel (101 columns) |
| `SCHEMA_stock.txt`, `SCHEMA_etf.txt` | — | Full column lists |

These were produced by PRIME's own mock generator and are **purely synthetic** — any
resemblance to a real security is coincidental. They are for schema illustration and
smoke-testing only, **not** for evaluating model performance.

```python
import pandas as pd
df = pd.read_parquet("data/example/csi500_stock_example.parquet")
print(df.shape)                  # (rows, 78)
print(sorted(df["ts_code"].unique()))
```

---

## Panel schema (a long/tidy daily panel)

One row per **(ticker, trading day)**. Required keys and the main field groups:

| Group | Columns (abridged) |
|-------|--------------------|
| **Keys** | `trade_date`, `ts_code`, `industry` |
| **OHLCV / price** | `open`, `high`, `low`, `close`, `pre_close`, `change`, `pct_chg`, `volume`, `amount`, `turnover`, `adj_factor`, `adj_close` |
| **Valuation** | `pe_ttm`, `pb`, `turnover_rate`, `total_mv`, `circ_mv` |
| **Macro** | `cpi_yoy`, `ppi_yoy`, `pmi`, `lpr_1y`, `real_rate_proxy`, `market_ret`, `benchmark_return`, `market_volatility`, `market_turnover`, `market_liquidity_change` |
| **Bull features** | `momentum_5d/10d/20d`, `rsi_6/14`, `north_net_flow`, `main_net_inflow`, `roe_growth`, `revenue_growth`, `profit_ratio`, `price_vs_ma20`, `volume_ratio`, `chip_support`, `winner_rate`, … |
| **Bear features** | `pe_rank`, `pb_rank`, `bias_20d/60d`, `trapped_ratio`, `dist_to_resistance`, `cost_pressure`, `avg_cost_deviation`, `debt_ratio`, `goodwill_risk` |
| **Friction** | `turnover_20d_avg`, `volatility_20d/60d`, `amplitude`, `asr`, `chip_concentration_change`, `vol_price_divergence` |
| **ETF-specific** | `unit_nav`, `accum_nav`, `premium_abs`, `fund_flow_5d/20d`, `nav_growth_rate`, `excess_return_20d`, `alpha_60d`, `tracking_error_20d`, … |

The loader (`harmonize_panel_schema` in
[`prime/market_profiles.py`](../prime/market_profiles.py)) is tolerant: it renames
common aliases (`symbol`→`ts_code`, `date`→`trade_date`, …) and **derives missing
columns** where possible (e.g. `pct_chg`, `adj_close`, RSI, valuation ranks,
market-aggregate macro fields). You therefore do not need every column above —
provide what you have and the pipeline fills the rest.

See `SCHEMA_stock.txt` / `SCHEMA_etf.txt` for the complete column list used by the
example panels.

---

## Reproducing with real data

The paper uses **CSI 500** and **S&P 500** constituents (with Nikkei 225 / STOXX 600
for cross-geography studies). To reproduce on real data, assemble a daily panel with
the schema above from your own licensed data provider and place it at the expected
path. The model, training, and backtest code are identical for real and synthetic
data — only the panel contents differ.
