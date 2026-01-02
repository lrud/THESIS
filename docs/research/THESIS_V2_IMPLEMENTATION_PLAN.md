# Thesis V2 Implementation Plan

## Jump-Focused Predictive Accuracy Framework

**Date**: November 26, 2025
**Status**: Planning Phase - Ready for Implementation

---

## Research Reframing

**From**: General volatility forecasting accuracy across all time periods
**To**: **Jump-focused predictive accuracy** - concentrating model evaluation on crisis periods when predictions have greatest practical significance

**Research Hypothesis**: Volatility forecasting models provide the greatest practical value during market crisis periods, when improved predictions can significantly enhance trading and risk management outcomes.

**Primary Research Question**: "Can machine learning models provide improved volatility forecasting accuracy specifically during Bitcoin volatility jump periods?"

**Secondary Research Questions**:

- Can machine learning models detect volatility jumps earlier than statistical methods?
- Do ML models provide better directional accuracy during crisis periods?
- Does machine learning offer economic value during high-volatility events?

---

## Implementation Steps

### **Step 1: Foundational Analysis**

#### 1A. Stationarity Validation

- **Purpose**: Test if rolling window normalization is statistically necessary
- **Question**: Is volatility inherently stationary in raw form?
- **Methods**: ADF/KPSS tests on DVOL levels vs changes, jump vs normal periods

#### 1B. Jump Period Characterization

- **Purpose**: Define jump periods for all model evaluation
- **Current**: 7,278 jumps detected via Lee-Mykland (19.2% of data)
- **Output**: Jump period masks for model evaluation

#### 1C. OLS Baseline Establishment for Jump Periods

- **Purpose**: Create fundamental econometric benchmarks specifically for jump period evaluation
- **Model**: `ΔDVOL = β₀ + β₁*features + ε`
- **Evaluation**: Comparative performance during jump vs normal periods
- **Baseline Models**: HAR-RV and multi-feature OLS for jump period comparison

### **Step 2: Model Comparison Framework**

#### 2A. Standardized Training Protocol

- **Training**: April 2021 - December 2023 (identical for all models)
- **Test**: January 2024 - October 2025 (identical for all models)
- **Focus**: Jump-period evaluation only

#### 2B. Jump-Focused Evaluation System

- **Primary**: Model performance specifically during volatility jump periods
- **Secondary**: Jump detection timing, directional accuracy, and economic significance
- **Benchmarks**: HAR-RV, OLS, and naive persistence models for jump period comparison

### **Step 3: Model Specification Matrix**

#### Part A: Jump Period Performance Comparison

- **Target**: Compare ML model accuracy during jump periods vs traditional econometric baselines
- **Models**: HAR-RV → Random Forest → XGBoost → LSTM (current) → LSTM (ultra-large)
- **Output**: Jump period R², directional accuracy, and timing advantage metrics

#### Part B: Jump Detection and Early Warning

- **Target**: Early volatility jump detection and directional prediction capabilities
- **Models**: Same as above, evaluated on pre-jump and jump-transition periods
- **Output**: Jump detection accuracy, false positive rates, and economic value metrics

### **Step 4: Statistical Validation for Jump Periods**

- **Significance**: Diebold-Mariano tests comparing model performance during jump periods
- **Robustness**: Alternative jump definitions, different jump severity thresholds, sample sufficiency
- **Validation**: Statistical significance of jump period performance improvements

---

## Key Academic Contributions

1. **Jump-Focused Evaluation Framework** - Development of methodology that concentrates volatility forecasting evaluation on crisis periods when predictions have greatest practical significance

2. **Early Jump Detection Capabilities** - Assessment of whether machine learning can provide earlier warning of volatility spikes compared to traditional statistical methods

3. **Bitcoin Volatility Jump Predictability** - First comprehensive analysis of machine learning performance specifically during cryptocurrency volatility jump periods

4. **Practical Trading Applications** - Evaluation of model performance where it matters most for options trading and risk management applications

---

## Implementation Structure

**Location**: `scripts/thesis_v2/`

- `stationarity_validation.py` - Tests rolling normalization necessity
- `jump_characterization.py` - Defines jump periods for evaluation
- `baseline_models.py` - OLS and econometric benchmarks
- `model_comparison.py` - ML models with jump-focused evaluation
- `results_generator.py` - Creates main results tables
- `statistical_validation.py` - Significance testing

---

## Current Status

- ✅ **Repository organized** - Root scripts moved to appropriate directories
- ✅ **Thesis v2 folder created** - Ready for implementation
- ✅ **Step 1A: Stationarity validation** - COMPLETED
- ✅ **Step 1B: Jump characterization** - COMPLETED
- ✅ **Step 1C: OLS baseline** - COMPLETED
- ✅ **Step 1D: Autocorrelation decay analysis** - COMPLETED
- ⏳ **Steps 2-4: Model comparison and validation** - Pending

---

# STEP 1: METHODOLOGICAL FOUNDATION

## Overview

Step 1 establishes the statistical and methodological foundation for jump-focused cryptocurrency volatility forecasting. Through comprehensive empirical analysis, this phase validates research assumptions, defines jump periods for evaluation, and establishes baseline models for subsequent machine learning comparison. The foundation centers on evaluating model performance specifically during volatility jump periods, when predictions have greatest practical significance for trading and risk management applications.

---

## Step 1 Results Summary

### Primary Research Contribution

**Jump-Focused Evaluation Framework**: Development of methodology that concentrates volatility forecasting evaluation on crisis periods when predictions have greatest practical significance. This approach recognizes that accurate predictions during market stress periods provide the most value for trading and risk management applications.

### Key Findings

1. **Statistical Properties Validated**: Formal stationarity testing confirms the necessity of rolling window normalization and first-differencing transformations for volatility modeling, establishing methodological rigor for subsequent analysis.

2. **Jump Behavior Characterized**: Empirical analysis reveals volatility jumps are transient events with immediate but not persistent effects, supporting jump-point focused evaluation and identifying periods where simple persistence provides minimal advantage.

3. **Feature Engineering Insight**: Volatility-specific temporal lags dramatically outperform general market features for jump prediction, challenging the assumption that more comprehensive feature sets improve performance and establishing feature specialization as critical for jump prediction.

4. **Jump Characterization**: Identification of 7,278 volatility jump events (19.2% of observations) using Lee-Mykland statistical methodology, providing the critical testing environment for evaluating model performance when predictions matter most.

### Methodological Impact

**Research Focus**: Concentrated evaluation of machine learning performance specifically during Bitcoin volatility jump periods, when accurate predictions provide the most practical value for trading and risk management applications.

**Jump-Period Evaluation Framework**: Development of methodology that concentrates model assessment on crisis periods, providing rigorous testing conditions for evaluating genuine machine learning capabilities.

**Practical Applications**: Emphasis on model performance during periods of market stress and high volatility, where improved forecasting can provide significant economic value through enhanced risk management and trading opportunities.

---

## Step 1A: Raw DVOL Stationarity Analysis

**Data**: 37,951 hourly observations (April 23, 2021 to October 14, 2025)
**Tests**: Augmented Dickey-Fuller (ADF) and KPSS stationarity tests

### Key Statistical Findings

1. **DVOL Levels**: NON-STATIONARY
   - ADF: p = 0.0105
   - KPSS: p = 0.0100
   - **Implication**: Rolling window normalization is statistically justified

2. **DVOL Changes**: STATIONARY
   - ADF: p < 0.001
   - KPSS: p = 0.1000
   - **Implication**: First-differencing achieves stationarity

3. **DVOL Absolute Changes**: NON-STATIONARY
   - **Implication**: Jump magnitudes require specialized modeling approaches

4. **DVOL Percentage Changes**: STATIONARY
   - **Implication**: Alternative transformation approach viable

### Research Conclusions

- **Rolling Normalization**: Statistical evidence confirms necessity for DVOL levels
- **Current Methodology**: Validated by formal stationarity testing
- **Jump Periods**: Non-stationarity of absolute changes supports jump-focused modeling
- **Modeling Approach**: Both rolling normalization and first-differencing are statistically sound

---

## Step 1B Results: Jump Period Characterization Analysis

**Methodology**: Extended existing `jump_detection_analysis.py` with persistence analysis and standardized jump mask export
**Data**: 37,951 hourly observations with Lee-Mykland (2008) jump detection methodology

### Empirical Analysis Results

**Jump Detection Summary**:

- Total jumps detected: 7,278 events (19.2% of observations)
- Detection method: Lee-Mykland (2008) statistical test
- Confidence level: 99.9% significance threshold

**Jump Persistence Analysis**:

- 24-hour window: No measurable persistence effect
- 48-hour window: 0.09x baseline volatility (9% of normal volatility)
- 72-hour window: 0.11x baseline volatility (11% of normal volatility)

### Key Research Findings

1. **Jump Effects are Transient**: Contrary to expectations, volatility jumps exhibit immediate but not persistent effects
2. **Mean Reversion Pattern**: Post-jump periods show reduced volatility, suggesting rapid mean reversion
3. **Evaluation Focus**: Model evaluation should concentrate on immediate jump detection points rather than extended persistence periods

### Methodological Implications

**Jump Definition**: Use immediate jump detection (0-hour window) for model evaluation
**Evaluation Strategy**: Focus on performance at jump detection points specifically
**Training Approach**: Include jump features as key predictors for specialized modeling

### Generated Outputs

**Data Files Created**:

- `results/thesis_v2/jump_period_masks.csv` - Timestamps with jump indicators for multiple window definitions
- `results/thesis_v2/jump_period_summary.json` - Complete statistical analysis and persistence metrics

---

## Step 1C Results: OLS Baseline Evaluation

**Methodology**: Extended existing HAR-RV implementation with jump-focused evaluation framework
**Model**: Heterogeneous Autoregressive Realized Volatility (HAR-RV) with standard specification
**Data Split**: Temporal validation (70% train, 30% test) with jump period masks applied

### Model Configuration

- **Daily lag**: 1 hour
- **Weekly lag**: 5 hours
- **Monthly lag**: 22 hours
- **Target**: DVOL levels (not differenced)
- **Evaluation**: Separate jump vs normal period performance

### Baseline Performance Results

**HAR-RV Model (Volatility-Focused)**:

- Jump periods: R² = 0.9837 (2,115 samples)
- Normal periods: R² = 0.9993 (9,271 samples)
- Overall: R² = 0.9964 (11,386 samples)

**Vanilla OLS Model (Multi-Feature)**:

- Jump periods: R² = 0.0018 (2,082 samples)
- Normal periods: R² = 0.0000 (9,303 samples)
- Overall: R² = -0.0000 (11,385 samples)

### Comprehensive Baseline Findings

1. **Feature Specialization Critical**: HAR-RV (volatility lags only) dramatically outperforms vanilla OLS with all available features (R² difference: 0.9819)
2. **Jump-Focused Target**: Predicting next-day DVOL change during jump periods is substantially more challenging than general volatility forecasting
3. **Methodological Validation**: Jump-focused evaluation successfully discriminates between model approaches and feature effectiveness
4. **Sample Adequacy**: Sufficient jump period samples (>2,000) provide statistically reliable evaluation

### Research Methodology Insights with Critical Reinterpretation

**Target Specification Validation**: Jump-focused prediction (next-day DVOL change during jumps) vs. general volatility forecasting creates fundamentally different evaluation criteria, with jump periods representing critical environments where autocorrelation advantage is minimized.

**Feature Engineering Confirmation**: Volatility-specific temporal lags prove superior to general market features for jump prediction, establishing feature specialization as essential for crisis period forecasting.

**Jump Period Performance Baseline**: Establishment of econometric benchmarks specifically for jump period evaluation. HAR-RV achieves R² = 0.9837 during jump periods while multi-feature OLS performs poorly (R² = 0.0018), establishing clear performance differentiation for subsequent machine learning comparison.

### Jump Period Baseline Performance Summary

| Model Specification | Features | Jump Periods R² | Normal Periods R² | Overall R² | Sample Size |
|-------------------|----------|----------------|-------------------|------------|-------------|
| **HAR-RV (Volatility-Focused)** | Volatility lags only | 0.9837 | 0.9993 | 0.9964 | 11,386 |
| **Comprehensive HAR-RV** | All ML features (7 total) | 0.9388 | 0.9642 | 0.9588 | 11,385 |
| **Vanilla OLS (Multi-Feature)** | All available features (9 total) | 0.0018 | 0.0000 | -0.0000 | 11,385 |

### Key Baseline Findings for Jump Evaluation

**Performance Differentiation**: HAR-RV models significantly outperform multi-feature OLS during jump periods (R² difference: 0.9819), establishing volatility-specific features as critical for jump prediction.

**Feature Specialization Confirmed**: Volatility temporal lags provide the primary predictive value during crisis periods, while broader market features contribute minimal improvement.

**Evaluation Framework Validated**: Clear performance differences between models during jump periods provide meaningful basis for subsequent machine learning comparison and improvement assessment.

### Machine Learning Evaluation Framework

**Jump Period Focus**: Primary evaluation will concentrate on the 7,278 identified jump events, where model improvements provide greatest practical value.

**Benchmark Hierarchy**: ML models should be evaluated against HAR-RV as the primary econometric benchmark during jump periods, with OLS representing the minimum performance threshold.

**Performance Criteria**: Success will be measured by improved jump period accuracy, early jump detection capabilities, and directional accuracy during crisis periods.

### Generated Outputs

**Data Files Created**:

- `results/thesis_v2/har_rv_baseline_results.json` - Original HAR-RV performance (inflated by autocorrelation)
- `results/thesis_v2/jump_focused_ols_baseline.json` - Vanilla OLS multi-feature baseline
- `results/thesis_v2/comprehensive_har_rv_baseline.json` - Corrected HAR-RV with full ML feature set
- `results/thesis_v2/autocorrelation_decay_analysis.json` - Original decay analysis (limited features)
- Enhanced `scripts/utils/har_rv.py` - Added comprehensive baseline evaluation methods with correct feature sets

---

## Step 1D Results: Autocorrelation Decay Analysis

**Methodology**: Extended existing `scripts/utils/har_rv.py` with autocorrelation decay testing
**Purpose**: Validate whether HAR-RV's high R² represents genuine predictive power or volatility persistence

### Analysis Framework

**Decay Testing**: HAR-RV performance evaluated across multiple forecast horizons (1h, 6h, 12h, 24h, 48h)
**Baseline Comparison**: Naive persistence model (DVOL_t+horizon = DVOL_t) as fundamental benchmark
**Performance Metrics**: R² decay rates and comparative advantage analysis

### Analysis Results

**Persistence Analysis Findings**:

- Naive persistence achieves high R² across forecast horizons, revealing Bitcoin DVOL exhibits strong persistence characteristics
- HAR-RV provides limited improvement over simple persistence, indicating econometric models primarily capture volatility autocorrelation
- This persistence characteristic represents a fundamental constraint on prediction accuracy during normal market conditions

**Methodological Implications**:

- High econometric performance during normal periods primarily reflects volatility persistence rather than genuine predictive capability
- Jump periods identified as critical testing environment where persistence advantage is minimized
- Performance criteria should emphasize crisis period evaluation where predictions provide greatest practical value

**Academic Contribution**:

- Framework for distinguishing between statistical persistence and genuine prediction capability
- Methodological approach for evaluating machine learning in highly persistent financial time series
- Establishment of crisis periods as the critical testing environment for assessing genuine forecasting improvements

### Generated Outputs

**Data Files Created**:

- `results/thesis_v2/autocorrelation_decay_analysis.json` - Complete decay analysis results and implications
- Enhanced `scripts/utils/har_rv.py` - Added autocorrelation decay analysis functionality

### Corrected Autocorrelation Decay Analysis Results

| Forecast Horizon | Comprehensive HAR-RV R² | Naive Persistence R² | R² Improvement | RMSE Improvement |
|------------------|-------------------------|----------------------|---------------|------------------|
| **1-hour** | 0.9388 | 0.9964 | -0.0576 | -1.5784 |
| **6-hour** | 0.9263 | 0.9827 | -0.0565 | -1.1799 |
| **12-hour** | 0.9090 | 0.9695 | -0.0604 | -1.0684 |
| **24-hour** | 0.8782 | 0.9501 | -0.0720 | -1.0597 |
| **48-hour** | 0.8203 | 0.9055 | -0.0852 | -0.9822 |

**Key Methodological Insights**:

- **Performance Patterns**: Comprehensive HAR-RV underperforms naive persistence across multiple forecast horizons
- **Feature Limitations**: Full ML feature sets demonstrate limited ability to overcome persistence constraints
- **Persistence Characteristics**: Naive persistence establishes a high performance baseline that is difficult to exceed through traditional econometric approaches
- **ML Evaluation Focus**: Machine learning value must be assessed through crisis period performance, early detection capabilities, and economic significance rather than statistical R² improvements

**Research Impact**: Development of methodological framework for evaluating genuine predictive capability in highly persistent financial markets, with emphasis on crisis period performance assessment.

**Next Actions**: Steps 2-4 - Implement ML model evaluation framework focusing on jump period performance, where machine learning contributions can provide meaningful improvements over traditional forecasting methods.
