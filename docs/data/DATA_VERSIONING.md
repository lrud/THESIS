# Data Versioning Guide

## Dataset Versions

| Version | Date | Records | Date Range | Location |
|---------|------|---------|------------|----------|
| **v1.0** | 2025-10-15 | 37,951 | 2021-04-23 to 2025-10-14 | `data/archive/v1.0_2025-10-15/` |
| **v1.1** | 2025-12-29 | 39,472 | 2021-04-23 to 2025-12-28 | `data/archive/v1.1_2025-12-29/` |

**Change:** +1,521 records (+4%), extending dataset by 75 days

---

## v1.0 → v1.1 Changes

### New Data Pulled
| Source | Records | Date Range | Notes |
|--------|---------|------------|-------|
| DVOL (Deribit) | 1,543 | Oct 15 - Dec 29, 2025 | Full historical access |
| On-chain (Research Bitcoin) | 1,795 | Oct 15 - Dec 28, 2025 | 6/7 metrics (see issues) |

### API Issues Resolved
1. **SSL Error:** `beta.thebitcoinresearcher.net` → `api.thebitcoinresearcher.net`
2. **Base URL updated in:** `scripts/data_collection/researchbitcoin_data.py`

### Known Issues
| Metric | v1.0 | v1.1 | Workaround |
|--------|------|------|------------|
| volume_usd | ✅ Available | ❌ API endpoint changed (422 error) | Use `network_activity` (tx_count) as proxy |

**Impact:** `transaction_volume` column is NaN in v1.1 features. Models using this feature need retraining without it.

---

## File Structure

```
data/
├── archive/
│   ├── v1.0_2025-10-15/
│   │   ├── bitcoin_dvol_hourly_complete.csv
│   │   ├── bitcoin_nvrv_hourly_20251015.csv
│   │   └── bitcoin_lstm_features.csv
│   └── v1.1_2025-12-29/
│       ├── bitcoin_dvol_hourly_complete.csv
│       ├── bitcoin_nvrv_hourly.csv
│       └── bitcoin_lstm_features.csv
├── prepared/
│   └── bitcoin_lstm_features_v1.1.csv
└── raw/ (v1.0 original files unchanged)
```

---

## Feature Comparison

| Feature | v1.0 | v1.1 | Notes |
|---------|------|------|-------|
| dvol | ✅ | ✅ | dvol_close from Deribit |
| dvol_lag_1d | ✅ | ✅ | 24-hour lag |
| dvol_lag_7d | ✅ | ✅ | 168-hour lag |
| dvol_lag_30d | ✅ | ✅ | 720-hour lag |
| transaction_volume | ✅ | ❌ NaN | API endpoint changed |
| network_activity | ✅ | ✅ | tx_count (from Research Bitcoin) |
| nvrv | ✅ | ✅ | Calculated from market_cap / realized_cap |
| dvol_rv_spread | ⚠️ BUG | ✅ | **V1.0 BUG:** Does not match standard formula (correlation 0.05). **V1.1:** DVOL - realized_volatility(price_returns, 30-day, annualized). See Data Integrity section below. |

---

## Model Recommendations

| Model Type | Recommended Data | Reason |
|------------|------------------|--------|
| Existing trained models | v1.0 | volume_usd not available in v1.1 |
| New model training | v1.1 without transaction_volume | Extended date range |
| Feature engineering | Use network_activity only | tx_count is reliable proxy |

---

## Data Collection Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| DVOL incremental pull | Deribit volatility data | `deribit_data_collector/btc_volatility_collector.py` |
| On-chain incremental pull | Research Bitcoin metrics | `scripts/data_collection/researchbitcoin_data.py` |
| Full incremental pipeline | Orchestrates both pulls | `scripts/data_collection/pull_incremental_data.py` |

---

## API Notes

**Deribit DVOL:**
- Endpoint: `get_volatility_index_data`
- Resolution: 3600 (hourly)
- Pagination: Supported via continuation token
- Historical access: Full

**Research Bitcoin:**
- Base URL: `https://api.thebitcoinresearcher.net/v2`
- Token lifetime: 90 days
- Renew at: https://api.researchbitcoin.net/token
- Resolution: h1 (hourly)

---

## Future Updates

To extend the dataset further:

1. Run incremental pull script:
```bash
python scripts/data_collection/pull_incremental_data.py
```

2. Merge new data with v1.1 to create v1.2

3. Re-engineer features with updated data

4. Update this document with new version info

---

## Data Integrity Findings

### V1.0 dvol_rv_spread Bug Discovery

During the v1.1 data integrity validation, a critical bug was discovered in the v1.0 `dvol_rv_spread` calculation.

#### Investigation Summary

**Initial Finding:** After merging v1.0 and v1.1 datasets, a comparison revealed that all overlapping features matched EXCEPT for `dvol_rv_spread` (37,951 differences in the overlapping period).

**Search for Original Formula:**
- Searched entire repository for `dvol_rv_spread` calculation script - **NOT FOUND**
- Searched for `realized_volatility`, `rv_spread`, `rolling_std(720)` - only found in documentation
- Checked deprecated/ folder - contains model loaders, not feature creators
- Checked scripts/data_collection/ - contains data collectors, not feature mergers
- Checked git history - **no commits** for deleted data preparation or feature creation scripts
- All references to `bitcoin_lstm_features.csv` show it being **loaded/used**, never **created**

**Conclusion:** The original feature creation script was run outside of version control and no longer exists.

#### Formula Validation

**Standard Economics Formula:**
```
Volatility Risk Premium (VRP) = Implied Volatility - Realized Volatility

Where:
- Implied Volatility = DVOL (Deribit 30-day implied volatility index)
- Realized Volatility = std(price_returns, 30-day) * sqrt(8760) * 100
  - 30-day window = 720 hours
  - sqrt(8760) = annualization factor (hourly to annual)
  - Multiplied by 100 for percentage
```

**Test Results (Correlation with Standard Formula):**

| Dataset | Correlation | Interpretation |
|---------|-------------|----------------|
| **V1.0 dvol_rv_spread** | **0.0485** | **INCORRECT** - No correlation with standard formula |
| **V1.1 dvol_rv_spread** | **0.9905** | **CORRECT** - Near-perfect match with standard formula |

#### Statistical Evidence

**V1.0 (Incorrect):**
- Mean: 21.46%
- Std: 38.52%
- Range: -294.35% to +92.17%
- **Issue:** Values do not represent the economics VRP concept

**V1.1 (Correct):**
- Mean: 6.49%
- Std: 10.74%
- Range: -36.37% to +68.29%
- **Interpretation:** Correctly represents the volatility risk premium (options market implied vol minus realized vol)

#### Impact on Models

| Aspect | Impact |
|--------|--------|
| Models trained with v1.0 | Used incorrect `dvol_rv_spread` values |
| Feature importance | May be artificially inflated or deflated |
| Predictions | Potential bias in volatility risk premium estimation |
| Recommendation | Retrain models with v1.1 data for correct VRP feature |

#### Correct Formula Implementation

```python
# Standard economics VRP calculation (used in v1.1)
df_merge['price_return'] = df_merge['price'].pct_change()
df_merge['rv_30d'] = df_merge['price_return'].rolling(window=720, min_periods=1).std() * np.sqrt(8760) * 100
df_merge['dvol_rv_spread'] = df_merge['dvol'] - df_merge['rv_30d']
```

Where:
- `rv_30d`: 30-day realized volatility from Bitcoin price returns, annualized
- `dvol_rv_spread`: Volatility Risk Premium (positive = options overprice risk, negative = options underprice risk)
