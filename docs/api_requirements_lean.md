# Data Sources

## API Data Sources for Bitcoin DVOL LSTM Thesis

## Primary Data Source: Coin Metrics API (Community Tier)

**Base URL**: `https://api.coinmetrics.io/v4`
**Rate Limits**: 10 requests per 6 seconds sliding window
**Cost**: Free community tier
**Period**: March 24, 2021 - October 15, 2025 (aligned with DVOL inception)

### 1. Deribit API - Bitcoin DVOL Data

**Endpoint**: `https://www.deribit.com/api/v2/public/get_volatility_index_data`
**Data Description**: Bitcoin implied volatility index (DVOL) with hourly granularity
**Period**: March 24, 2021 - October 15, 2025
**Observations**: 38,789 hourly records (97.01% coverage)
**File Location**: `/data/raw/bitcoin_dvol_hourly_complete.csv`
**Cost**: Free

### 2. Coin Metrics API - Bitcoin Price Data

**Endpoint**: `/timeseries/market-candles`
**Market**: `coinbase-btc-usd-spot`
**Data Description**: Hourly OHLC price data for realized volatility calculation
**Frequency**: 1 hour
**Expected Records**: ~38,000+ hourly observations
**File Location**: `/data/raw/btc_price_hourly_coinmetrics.csv`

### 3. Coin Metrics API - On-Chain Metrics

**Endpoint**: `/timeseries/asset-metrics`
**Asset**: `btc`
**Frequency**: 1 day
**Metrics**:
- `TxTfrValUSD`: Transaction Volume USD (daily)
- `AdrActCnt`: Active Addresses Count (daily) 
- `CapMVRVCur`: Market Value to Realized Value ratio (NVRV proxy)
- `NVTAdj`: Network Value to Transactions (additional metric)
**Expected Records**: ~1,700+ daily observations
**File Location**: `/data/raw/btc_onchain_daily_coinmetrics.csv`

### 4. Calculated Features

**Realized Volatility**: Calculated from hourly price returns using 24-hour rolling window
**DVOL-RV Spread**: Difference between DVOL and Realized Volatility
**File Location**: `/data/raw/btc_realized_volatility_daily_coinmetrics.csv`

## Complete 5-Feature LSTM Model

1. **Lagged DVOL** (independent variable) - Deribit API
2. **BTC Price/Realized Volatility** - Coin Metrics API  
3. **Transaction Volume USD** - Coin Metrics API
4. **Active Addresses Count** - Coin Metrics API
5. **NVRV/MVRV** - Coin Metrics API
6. **DVOL-RV Spread** - Calculated feature