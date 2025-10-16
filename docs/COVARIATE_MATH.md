# Covariate Mathematical Formulas

## DVOL (Implied Volatility)
$$\text{DVOL} = \text{Deribit 30-day Bitcoin IV}$$
- Source: Research Bitcoin API
- Lags: DVOL_{t-24h}, DVOL_{t-7d}, DVOL_{t-30d}

## Network Metrics

**Transaction Volume (USD)**
$$\text{TxVolume}_t = \sum_{transactions} \text{value}(t)$$

**Network Activity (Active Addresses)**
$$\text{NetActivity}_t = \text{active addresses at time } t$$

**NVRV Ratio (Network Value / Realized Value)**
$$\text{NVRV}_t = \frac{\text{Market Cap}_t}{\text{Realized Cap}_t}$$
- Market Cap: Current price × circulating supply
- Realized Cap: Sum of all BTC at price last moved

## DVOL-RV Spread
$$\text{DVOL-RV Spread}_t = \text{DVOL}_t - \text{Realized Volatility}_t$$

Where Realized Volatility is calculated from hourly returns:
$$\text{RV}_t = \sqrt{\sum_{i=1}^{n} r_i^2}$$

## Open Interest (Corrected Calculation)

**Direct Aggregation:**
$$\text{OI}_{\text{total,BTC}} = \sum_{i=1}^{n} \text{open\_interest}_i$$

Where:
- Deribit API returns `open_interest` field = OI already in BTC
- For BTC options: each contract = 1 BTC notional
- Formula: $\text{num\_contracts} \times 1 \text{ BTC} = \text{OI in BTC}$
- We simply sum all option OI values (already converted by API)

**By Type:**
$$\text{Call OI}_t = \sum_{\text{strikes}} \text{open\_interest}(\text{call}_i)$$
$$\text{Put OI}_t = \sum_{\text{strikes}} \text{open\_interest}(\text{put}_i)$$

**Ratios:**
$$\text{Total OI}_t = \text{Call OI}_t + \text{Put OI}_t$$
$$\text{Put/Call Ratio}_t = \frac{\text{Put OI}_t}{\text{Call OI}_t}$$

## Regression Specification

**DVOL as Dependent Variable:**
$$\text{DVOL}_t = \beta_0 + \beta_1(\text{OI}_t) + \beta_2(\text{TxVolume}_t) + \beta_3(\text{NetActivity}_t) + \beta_4(\text{NVRV}_t) + \beta_5(\text{DVOL-RV}_t) + \epsilon_t$$

## Time Aggregation
All metrics: **Hourly UTC aggregation**

## Data Validation
- No missing values after merge
- Inner join on timestamp (hour-level precision)
- Continuity: 99.65% (max gap: 1 day)
