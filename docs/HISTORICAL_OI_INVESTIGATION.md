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
- OKX: Similar deletion policy (~1 week history)
- Bybit: API access blocked (403)
- Binance: Geo-blocked (451)
- CME/CBOE: Available via paid subscription ($500-5,000/month)
- OptionsDX: Commercial historical data vendor (~$40-50 for 2-3 years)

---

**Alternative Sources Tested:**
- OKX: Similar deletion policy (~1 week history)
- Bybit: API access blocked (403)
- Binance: Geo-blocked (451)
- CME/CBOE: Available via paid subscription ($500-5,000/month)
- OptionsDX: Commercial historical data vendor (~$40-50 for 2-3 years)

**Current Data Files:**
- `data/processed/options_oi_clean.csv` - Real-time snapshots
- `data/processed/options_oi_hourly.csv` - 147-day bootstrap (3,535 records)

**Latest Market Snapshot (Oct 16, 2025):**
- Total OI: 439,530 BTC
- Put/Call Ratio: 0.657
- Active contracts: 713 of 800
