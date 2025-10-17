# Historical Options Open Interest Investigation

**Date:** October 16, 2025  
**Status:** Complete

---

## Key Findings

**Deribit API Limitation:**
- Expired option contracts are permanently deleted from API
- Earliest available contracts: December 26, 2024
- Historical data gap: April 2021 - May 2025 (4+ years)

**Available Data:**
- May 22 - Oct 16, 2025: 3,535 records (147 days, 99.97% continuity)
- Real-time collection: Active from Oct 16, 2025 forward

**Alternative Sources Tested:**
- Deribit History API (history.deribit.com): Has expired contracts but NO open_interest field
- BarendPotijk/deribit_historical_trades: Wraps History API, no OI data
- DeribitHistory (PyPI): Package does not exist
- OKX: Similar deletion policy (~1 week history)
- Bybit: API access blocked (403)
- Binance: Geo-blocked (451)
- CME/CBOE: Available via paid subscription ($500-5,000/month)
- Databento: US venues only, no crypto exchanges
- Tardis.dev: Has OI data but cost prohibitive for thesis budget
- OptionsDX: Commercial vendor (~$40-50 for 2-3 years) ⭐ Most viable option

**Current Data Files:**
- `data/processed/options_oi_clean.csv` - Real-time snapshots
- `data/processed/options_oi_hourly.csv` - 147-day bootstrap (3,535 records)

**Latest Market Snapshot (Oct 16, 2025):**
- Total OI: 439,530 BTC
- Put/Call Ratio: 0.657
- Active contracts: 713 of 800

---

## Viable Options Summary

**Option 1: OptionsDX** (~$40-50)
- Deribit historical data with OPEN_INTEREST field
- Sample data validated (btc_sample.csv)
- 2-3 years coverage possible
- Most cost-effective commercial solution

**Option 2: Current Data Only** ($0)
- Use May 2025 - present (5+ months)
- Document limitation in methodology
- Still publishable with proper documentation
- Growing dataset (real-time collection active)

**Option 3: CME Data** ($500-5,000/month)
- Different market than Deribit (low liquidity)
- Mismatch with DVOL target variable
- Not recommended for thesis
