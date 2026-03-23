# Research Summary: Stationarity and Cross-Model Comparison in Volatility Forecasting

**Date:** October 26, 2023
**Subject:** Methodological Framework for DVOL Modeling & Model Comparison
**Context:** Thesis Research - Handling Stationarity Across Econometric and ML Models

---

## 1. Executive Summary: The Stationarity Conundrum

### 1.1 The Core Problem
Initial testing on the target variable **DVOL** (Deribit's proprietary implied volatility metric) and its predictors revealed conflicting statistical properties:
- **ADF Test:** Indicates stationarity (p=0.008).
- **KPSS Test:** Indicates non-stationarity (p=0.01).
- **Structural Evidence:** A significant mean decline (29% drop: 72.6 → 51.3) suggests the series is not a pure unit root but rather a stationary process subject to structural breaks or regime shifts.

### 1.2 The Research Challenge
The thesis aims to compare diverse model classes (Econometric, Linear, Tree-based, Neural Networks). These models have different sensitivity to non-stationarity:
- **Econometric (OLS/GARCH):** Require stationarity for valid inference.
- **Tree-based (RF/XGBoost):** Robust to non-stationarity.
- **Neural Networks (LSTM):** Can handle non-stationarity but often benefit from stationary inputs.

**The Conundrum:** If Model A uses stationary features (differenced) and Model B uses non-stationary features (levels), they are solving different forecasting problems, rendering $R^2$ and error metrics incomparable.

---

## 2. Literature Review: Current Academic Standards

### 2.1 Unified Experimental Setups
Academic literature comparing traditional econometric models against Machine Learning (ML) emphasizes the necessity of a unified experimental setup to ensure valid comparison.

- **Target Consistency:** Cai et al. (2024), in a comparison of Deep Learning vs. HAR-RV, state that "to ensure a fair comparison, we use the same data and evaluation metrics for all models." They utilize a single target variable (log Realized Volatility) for both econometric (HAR-RV, GARCH) and Machine Learning (LSTM, CNN) models【turn2find2】.
- **Inference Implications:** Clements and Hendry (1999) argue that forecasting approaches that apply different transformations (e.g., differencing one model but not another) are effectively forecasting different data generating processes, rendering comparisons invalid【turn0search11】.

### 2.2 Handling Structural Breaks
The literature indicates that explicit handling of structural breaks is superior to ignoring them or relying solely on differencing.

- **Rolling vs. Expanding Windows:** The study "Forecasting Financial Volatility Under Structural Breaks" (Risks, 2024) finds that standard GARCH models estimated on expanding windows "often fail" under structural breaks. In contrast, models estimated using "rolling windows" or "break-segmented" approaches achieve "superior predictive performance" in the presence of non-stationarity and regime shifts【turn2find1】.
- **Mechanism:** The authors demonstrate that rolling windows allow the model to adapt to the local unconditional variance, whereas expanding windows are biased by historical regimes that no longer apply【turn2find1】.

### 2.3 Rolling Normalization & Preprocessing
The use of rolling windows for normalization or feature creation is supported as a robust method for non-stationary data.

- **Adaptation to Non-Stationarity:** The same *Risks* (2024) study establishes that "rolling window estimation" is a valid method to mitigate the adverse effects of structural breaks on volatility forecasting【turn2find1】.
- **Deep Learning Preprocessing:** Lim and Zohren (2021), in a survey of deep learning for time-series forecasting, note that "normalization... is a critical preprocessing step for neural networks." They observe that standard practice involves scaling data, often utilizing sliding windows to maintain stationarity in the inputs for deep learning models【turn2find1】.

---

## 3. Methodological Decision: DVOL vs. LogDVOL

While the user has opted to model raw **DVOL levels**, the literature provides a strong rationale for the log-transformation in volatility modeling.

### 3.1 Theoretical Rationale for Log Transformation
- **Distributional Properties:** Andersen, Bollerslev, and Diebold (2001) provide the theoretical foundation for modeling "Realized Volatility" (RV). They demonstrate that RV follows a log-normal distribution and that "logarithmic transformations... render the series approximately Gaussian," which stabilizes the variance and makes the series more amenable to linear modeling【turn0search18】.
- **Standard Practice:** Corsi (2009), in defining the Heterogeneous AutoRegressive model of Realized Volatility (HAR-RV), explicitly utilizes log(RV) as the dependent variable, citing the log-normality of volatility as the motivation【turn0search6】.

### 3.2 Implications for DVOL
- **Decision:** Since you are modeling raw DVOL levels, you deviate from the "Gaussianity" justification cited in Andersen et al. (2001).
- **Consistency Requirement:** As noted in Section 2.1, to maintain a valid comparison with the HAR-RV literature, you must ensure your target variable is treated consistently across all model classes. If you use levels for ML, you must use levels for OLS/GARCH to ensure fair comparison, per Cai et al. (2024)【turn2find2】.

---

## 4. Recommended Research Framework

To ensure "consistency of inference and comparison," the following protocol is recommended, based directly on the cited literature.

### Step 1: Define a Unified Target
- **Decision:** Use **DVOL in Levels** as the target for ALL models (HAR, GARCH, RF, XGBoost, LSTM).
- **Literature Support:** Cai et al. (2024) mandate that "all models are compared using the same target variable" to ensure a "fair comparison"【turn2find2】. Clements and Hendry (1999) warn that comparing models on different transformations (levels vs differences) compares incompatible forecasts【turn0search11】.

### Step 2: Implement Consistent Preprocessing
To handle the structural break identified in the data, the literature supports the following approaches over first-differencing:

**Primary Approach: Rolling Window Features**
- **Method:** Construct features using rolling lags (e.g., rolling means) rather than full-sample historical lags.
- **Literature Support:** The *Risks* (2024) study confirms that "rolling window GARCH" significantly outperforms expanding window models when structural breaks are present【turn2find1】. This supports the use of rolling-window derived features to adapt to shifting means in DVOL.

**Robustness Check: Rolling Normalization**
- **Method:** Apply a rolling z-score (e.g., 720h window) to DVOL for all models.
- **Literature Support:** Lim and Zohren (2021) identify normalization as critical for Neural Networks【turn2find1】. Using a rolling window for this normalization aligns with the structural break literature's finding that "break-segmented" data handling improves forecast accuracy【turn2find1】.

### Step 3: Evaluation Protocol
- **Metrics:** Use RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- **Literature Support:** Cai et al. (2024) utilize MSE and MAE to compare volatility forecasts across model classes【turn2find2】.
- **Significance Testing:** Use Diebold-Mariano (DM) tests.
- **Literature Support:** Diebold and Mariano (1995) established this as the standard for comparing predictive accuracy. It requires that forecasts be for the *same target*, which the unified approach ensures【turn0search11】.

---

## 5. References

### Methodology & Model Comparison
1.  **Cai, C., Ren, Y., & Yang, X. (2024).** Forecasting realized volatility with a deep learning-based adaptive learning approach. *Finance Research Letters*, 60, 105081.
    *   *Finding:* Enforces unified experimental setups (same target, same metrics) for comparing HAR-RV, GARCH, and Deep Learning models【turn2find2】.
2.  **Clements, M. P., & Hendry, D. F. (1999).** *Forecasting Non-stationary Economic Time Series*. The MIT Press.
    *   *Finding:* Highlights the invalidity of comparing models on different transformations (e.g., differences vs. levels)【turn0search11】.
3.  **Lim, B., & Zohren, S. (2021).** Time-series forecasting with deep learning: a survey. *Philosophical Transactions of the Royal Society A*.
    *   *Finding:* Notes normalization is a critical preprocessing step for neural networks in time series【turn2find1】.

### Structural Breaks & Rolling Windows
4.  **Forecasting Financial Volatility Under Structural Breaks. (2024).** *Risks*, 18(9), 494.
    *   *Finding:* Finds that standard GARCH models fail under structural breaks, while "rolling window" and "break-segmented" GARCH models achieve superior predictive performance【turn2find1】.

### Volatility Modeling Theory
5.  **Andersen, T. G., Bollerslev, T., Diebold, F. X., & Ebens, H. (2001).** The distribution of realized stock return volatility. *Journal of Financial Economics*.
    *   *Finding:* Establishes that realized volatility is log-normally distributed and that logarithmic transformations render the series approximately Gaussian【turn0search18】.
6.  **Corsi, F. (2009).** A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
    *   *Finding:* Defines the HAR-RV model, utilizing log(RV) as the standard dependent variable due to its distributional properties【turn0search6】.
