# Bitcoin Volatility Analysis: NVRV and DVOL Dataset

## Project Overview

Data collection and analysis pipeline for Bitcoin volatility research using Network Value to Realized Value (NVRV) and Deribit Volatility Index (DVOL) metrics.

## Status: Data Collection Complete ✅

### Datasets
**NVRV On-Chain Data** (`bitcoin_nvrv_hourly_20251015.csv`)
- Records: 39,864 hourly observations
- Coverage: 2021-03-24 to 2025-10-14
- Metrics: Price, Market Cap, Realized Cap, NVRV, NUPL, Volume, TX Count

**DVOL Volatility Data** (`bitcoin_dvol_hourly_complete.csv`)  
- Records: 38,789 hourly observations
- Coverage: 2021-03-24 to 2025-10-15
- Metrics: DVOL Open, High, Low, Close

**Alignment:** 39,984 overlapping hours for econometric analysis

## NVRV Methodology

Academic formula validated against peer-reviewed literature:
```
NVRV = (Market Cap - Realized Cap) / Realized Cap
```
Net premium format optimized for econometric analysis.

## Next Steps

1. Data preprocessing and feature engineering
2. Volatility correlation analysis
3. Econometric modeling (GARCH, VAR)
4. Academic publication preparation

## File Structure
```
├── README.md
├── scripts/
│   └── data_collection/
│       └── researchbitcoin_data.py
└── data/
    └── raw/
        └── bitcoin_nvrv_dataset_20251015.csv
```

## Usage

### Environment Setup
```bash
export RESEARCH_BITCOIN_API_TOKEN="your_api_token"
```

### Data Collection
```bash
python scripts/data_collection/researchbitcoin_data.py
```

## Academic References

The NVRV metric selection is based on academic literature demonstrating its superiority over traditional MVRV for Bitcoin analysis:
- Enhanced correlation with Bitcoin price movements
- Superior behavioral sentiment proxy
- Improved volatility modeling capabilities

## Technical Notes

- All timestamps in UTC
- CSV format with pandas-compatible datetime indexing
- Data validation performed on collection
- Missing values handled through dropna() operations