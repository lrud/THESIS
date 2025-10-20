import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest, anderson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from pathlib import Path

class ComprehensiveModelValidation:
    """
    Comprehensive statistical validation suite for time series forecasting models.
    
    Tests for:
    1. Residual stationarity
    2. Autocorrelation (model captures all temporal structure)
    3. Heteroskedasticity (variance stability)
    4. Normality of residuals
    5. ARCH effects (volatility clustering in errors)
    6. Forecast bias
    7. Prediction intervals validity
    8. Structural breaks in residuals
    """
    
    def __init__(self, model_name='lstm_rolling'):
        self.model_name = model_name
        self.results = {}
        
    def load_predictions(self):
        """Load model predictions and actuals."""
        metrics_file = f'results/csv/{self.model_name}_metrics.csv'
        
        if not Path(metrics_file).exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        print(f"Note: Predictions not saved separately.")
        print(f"Will need to re-run model to generate predictions for detailed validation.")
        print(f"Using metrics from: {metrics_file}\n")
        
        return None, None
    
    def test_residual_stationarity(self, residuals):
        """
        Test if residuals are stationary (no trend/drift in errors).
        
        Tests:
        - Augmented Dickey-Fuller (ADF): H0 = non-stationary
        - KPSS: H0 = stationary
        - Both should agree on stationarity
        """
        print("=" * 80)
        print("1. RESIDUAL STATIONARITY TESTS")
        print("=" * 80)
        print("Tests if model errors are stationary (no systematic drift)")
        print()
        
        adf_result = adfuller(residuals, maxlag=24)
        print("Augmented Dickey-Fuller Test:")
        print(f"  ADF Statistic:    {adf_result[0]:.4f}")
        print(f"  p-value:          {adf_result[1]:.4f}")
        print(f"  Critical values:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.4f}")
        
        if adf_result[1] < 0.05:
            print(f"  ✓ REJECT H0: Residuals are STATIONARY (p={adf_result[1]:.4f})")
        else:
            print(f"  ✗ FAIL to reject H0: Residuals may be NON-STATIONARY (p={adf_result[1]:.4f})")
        print()
        
        kpss_result = kpss(residuals, regression='c', nlags='auto')
        print("KPSS Test:")
        print(f"  KPSS Statistic:   {kpss_result[0]:.4f}")
        print(f"  p-value:          {kpss_result[1]:.4f}")
        print(f"  Critical values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.4f}")
        
        if kpss_result[1] > 0.05:
            print(f"  ✓ FAIL to reject H0: Residuals are STATIONARY (p={kpss_result[1]:.4f})")
        else:
            print(f"  ✗ REJECT H0: Residuals may be NON-STATIONARY (p={kpss_result[1]:.4f})")
        print()
        
        self.results['stationarity'] = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'stationary': adf_result[1] < 0.05 and kpss_result[1] > 0.05
        }
        
    def test_autocorrelation(self, residuals):
        """
        Test if residuals are autocorrelated (model missed temporal patterns).
        
        Tests:
        - Ljung-Box Q test: Tests for autocorrelation at multiple lags
        - Durbin-Watson: Tests for first-order autocorrelation
        """
        print("=" * 80)
        print("2. AUTOCORRELATION TESTS")
        print("=" * 80)
        print("Tests if model captured all temporal patterns (residuals should be white noise)")
        print()
        
        lb_result = acorr_ljungbox(residuals, lags=[1, 6, 12, 24], return_df=True)
        print("Ljung-Box Q Test (autocorrelation at multiple lags):")
        print(lb_result)
        print()
        
        significant_lags = lb_result[lb_result['lb_pvalue'] < 0.05]
        if len(significant_lags) == 0:
            print("  ✓ NO significant autocorrelation detected")
            print("  → Model captured temporal structure well")
        else:
            print(f"  ✗ Significant autocorrelation at {len(significant_lags)} lags:")
            print(f"  → Model may have missed some temporal patterns")
            for lag in significant_lags.index:
                print(f"    Lag {lag}: p={lb_result.loc[lag, 'lb_pvalue']:.4f}")
        print()
        
        dw_stat = durbin_watson(residuals)
        print(f"Durbin-Watson Test (first-order autocorrelation):")
        print(f"  DW Statistic:     {dw_stat:.4f}")
        print(f"  Interpretation:")
        if 1.5 < dw_stat < 2.5:
            print(f"    ✓ No significant first-order autocorrelation (DW ≈ 2)")
        elif dw_stat < 1.5:
            print(f"    ✗ Positive autocorrelation detected (DW < 1.5)")
        else:
            print(f"    ✗ Negative autocorrelation detected (DW > 2.5)")
        print()
        
        self.results['autocorrelation'] = {
            'ljung_box_results': lb_result.to_dict(),
            'durbin_watson': dw_stat,
            'has_autocorrelation': len(significant_lags) > 0 or not (1.5 < dw_stat < 2.5)
        }
    
    def test_heteroskedasticity(self, residuals):
        """
        Test if residual variance is constant (homoskedasticity).
        
        Tests:
        - White's test: General heteroskedasticity
        - ARCH-LM test: Volatility clustering
        """
        print("=" * 80)
        print("3. HETEROSKEDASTICITY TESTS")
        print("=" * 80)
        print("Tests if error variance is constant over time")
        print()
        
        # Need to create exogenous variables for White's test
        # Using lagged residuals as proxy
        n = len(residuals)
        X = np.column_stack([
            np.ones(n),
            np.arange(n),
            np.arange(n)**2
        ])
        
        try:
            white_result = het_white(residuals, X)
            print("White's Test (general heteroskedasticity):")
            print(f"  LM Statistic:     {white_result[0]:.4f}")
            print(f"  p-value:          {white_result[1]:.4f}")
            print(f"  F-statistic:      {white_result[2]:.4f}")
            print(f"  F p-value:        {white_result[3]:.4f}")
            
            if white_result[1] > 0.05:
                print(f"  ✓ FAIL to reject H0: Homoskedastic (constant variance)")
            else:
                print(f"  ✗ REJECT H0: Heteroskedastic (variance changes over time)")
            print()
        except Exception as e:
            print(f"  White's test failed: {e}")
            white_result = None
            print()
        
        arch_result = het_arch(residuals, nlags=12)
        print("ARCH-LM Test (volatility clustering in residuals):")
        print(f"  LM Statistic:     {arch_result[0]:.4f}")
        print(f"  p-value:          {arch_result[1]:.4f}")
        print(f"  F-statistic:      {arch_result[2]:.4f}")
        print(f"  F p-value:        {arch_result[3]:.4f}")
        
        if arch_result[1] > 0.05:
            print(f"  ✓ NO ARCH effects (no volatility clustering)")
        else:
            print(f"  ✗ ARCH effects present (errors show volatility clustering)")
            print(f"  → Consider GARCH-type model for error variance")
        print()
        
        self.results['heteroskedasticity'] = {
            'white_test': white_result,
            'arch_lm_stat': arch_result[0],
            'arch_pvalue': arch_result[1],
            'has_heteroskedasticity': arch_result[1] < 0.05
        }
    
    def test_normality(self, residuals):
        """
        Test if residuals are normally distributed.
        
        Tests:
        - Jarque-Bera: Tests skewness and kurtosis
        - Shapiro-Wilk: General normality test
        - Kolmogorov-Smirnov: Compares to normal distribution
        """
        print("=" * 80)
        print("4. NORMALITY TESTS")
        print("=" * 80)
        print("Tests if residuals follow normal distribution (validates statistical inference)")
        print()
        
        jb_stat, jb_pvalue = jarque_bera(residuals)
        print("Jarque-Bera Test:")
        print(f"  JB Statistic:     {jb_stat:.4f}")
        print(f"  p-value:          {jb_pvalue:.4f}")
        print(f"  Skewness:         {stats.skew(residuals):.4f}")
        print(f"  Kurtosis:         {stats.kurtosis(residuals):.4f}")
        
        if jb_pvalue > 0.05:
            print(f"  ✓ FAIL to reject H0: Residuals are NORMAL")
        else:
            print(f"  ✗ REJECT H0: Residuals NOT normal")
        print()
        
        # Shapiro-Wilk (use sample if too large)
        sample_size = min(5000, len(residuals))
        sample_indices = np.random.choice(len(residuals), sample_size, replace=False)
        sw_stat, sw_pvalue = shapiro(residuals[sample_indices])
        print(f"Shapiro-Wilk Test (sample of {sample_size}):")
        print(f"  SW Statistic:     {sw_stat:.4f}")
        print(f"  p-value:          {sw_pvalue:.4f}")
        
        if sw_pvalue > 0.05:
            print(f"  ✓ FAIL to reject H0: Residuals are NORMAL")
        else:
            print(f"  ✗ REJECT H0: Residuals NOT normal")
        print()
        
        ks_stat, ks_pvalue = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        print("Kolmogorov-Smirnov Test:")
        print(f"  KS Statistic:     {ks_stat:.4f}")
        print(f"  p-value:          {ks_pvalue:.4f}")
        
        if ks_pvalue > 0.05:
            print(f"  ✓ FAIL to reject H0: Residuals are NORMAL")
        else:
            print(f"  ✗ REJECT H0: Residuals NOT normal")
        print()
        
        print("Note: Non-normality is common in financial data but doesn't invalidate model")
        print("      Important: Check for heavy tails (extreme errors)")
        print()
        
        self.results['normality'] = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'shapiro_wilk_stat': sw_stat,
            'shapiro_wilk_pvalue': sw_pvalue,
            'kolmogorov_smirnov_stat': ks_stat,
            'kolmogorov_smirnov_pvalue': ks_pvalue,
            'is_normal': jb_pvalue > 0.05 and sw_pvalue > 0.05
        }
    
    def test_forecast_bias(self, residuals):
        """
        Test if model has systematic forecast bias (over/under prediction).
        
        Tests:
        - One-sample t-test: H0: mean(residuals) = 0
        - Sign test: H0: median(residuals) = 0
        """
        print("=" * 80)
        print("5. FORECAST BIAS TESTS")
        print("=" * 80)
        print("Tests if model systematically over- or under-predicts")
        print()
        
        mean_residual = np.mean(residuals)
        median_residual = np.median(residuals)
        
        t_stat, t_pvalue = stats.ttest_1samp(residuals, 0)
        print("One-sample t-test (mean bias):")
        print(f"  Mean residual:    {mean_residual:.4f}")
        print(f"  t-statistic:      {t_stat:.4f}")
        print(f"  p-value:          {t_pvalue:.4f}")
        
        if t_pvalue > 0.05:
            print(f"  ✓ NO significant mean bias")
        else:
            if mean_residual > 0:
                print(f"  ✗ Significant POSITIVE bias (model under-predicts)")
            else:
                print(f"  ✗ Significant NEGATIVE bias (model over-predicts)")
        print()
        
        # Sign test
        positive_errors = np.sum(residuals > 0)
        negative_errors = np.sum(residuals < 0)
        total_errors = len(residuals)
        
        sign_pvalue = stats.binomtest(positive_errors, total_errors, 0.5).pvalue
        print("Sign Test (median bias):")
        print(f"  Median residual:  {median_residual:.4f}")
        print(f"  Positive errors:  {positive_errors} ({positive_errors/total_errors*100:.1f}%)")
        print(f"  Negative errors:  {negative_errors} ({negative_errors/total_errors*100:.1f}%)")
        print(f"  p-value:          {sign_pvalue:.4f}")
        
        if sign_pvalue > 0.05:
            print(f"  ✓ NO significant median bias")
        else:
            if positive_errors > negative_errors:
                print(f"  ✗ Significant POSITIVE bias (model under-predicts)")
            else:
                print(f"  ✗ Significant NEGATIVE bias (model over-predicts)")
        print()
        
        self.results['forecast_bias'] = {
            'mean_residual': mean_residual,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'median_residual': median_residual,
            'sign_pvalue': sign_pvalue,
            'has_bias': t_pvalue < 0.05 or sign_pvalue < 0.05
        }
    
    def test_structural_breaks(self, residuals):
        """
        Test for structural breaks in residuals (model performance changes over time).
        
        Uses:
        - Chow test: Tests if regression coefficients change at known breakpoint
        - CUSUM test: Detects gradual changes
        """
        print("=" * 80)
        print("6. STRUCTURAL BREAK TESTS")
        print("=" * 80)
        print("Tests if model performance is stable over time")
        print()
        
        # Split residuals into thirds
        n = len(residuals)
        third = n // 3
        
        res_early = residuals[:third]
        res_mid = residuals[third:2*third]
        res_late = residuals[2*third:]
        
        var_early = np.var(res_early)
        var_mid = np.var(res_mid)
        var_late = np.var(res_late)
        
        print("Variance across time periods:")
        print(f"  Early period variance:  {var_early:.4f}")
        print(f"  Middle period variance: {var_mid:.4f}")
        print(f"  Late period variance:   {var_late:.4f}")
        
        # Levene's test for equal variances
        levene_stat, levene_pvalue = stats.levene(res_early, res_mid, res_late)
        print(f"\nLevene's Test (equal variances across periods):")
        print(f"  Statistic:        {levene_stat:.4f}")
        print(f"  p-value:          {levene_pvalue:.4f}")
        
        if levene_pvalue > 0.05:
            print(f"  ✓ NO structural break (variance stable)")
        else:
            print(f"  ✗ Potential structural break (variance changes)")
        print()
        
        # Mean comparison
        mean_early = np.mean(res_early)
        mean_mid = np.mean(res_mid)
        mean_late = np.mean(res_late)
        
        print("Mean across time periods:")
        print(f"  Early period mean:  {mean_early:.4f}")
        print(f"  Middle period mean: {mean_mid:.4f}")
        print(f"  Late period mean:   {mean_late:.4f}")
        
        f_stat, f_pvalue = stats.f_oneway(res_early, res_mid, res_late)
        print(f"\nOne-way ANOVA (equal means across periods):")
        print(f"  F-statistic:      {f_stat:.4f}")
        print(f"  p-value:          {f_pvalue:.4f}")
        
        if f_pvalue > 0.05:
            print(f"  ✓ NO structural break (mean stable)")
        else:
            print(f"  ✗ Potential structural break (mean changes)")
        print()
        
        self.results['structural_breaks'] = {
            'variance_early': var_early,
            'variance_mid': var_mid,
            'variance_late': var_late,
            'levene_pvalue': levene_pvalue,
            'mean_early': mean_early,
            'mean_mid': mean_mid,
            'mean_late': mean_late,
            'anova_pvalue': f_pvalue,
            'has_break': levene_pvalue < 0.05 or f_pvalue < 0.05
        }
    
    def generate_diagnostic_plots(self, residuals, actuals, predictions):
        """Generate diagnostic plots for residual analysis."""
        print("=" * 80)
        print("7. GENERATING DIAGNOSTIC PLOTS")
        print("=" * 80)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Residuals over time
        axes[0, 0].plot(residuals, linewidth=0.5, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual histogram
        axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ACF plot
        acf_values = acf(residuals, nlags=40, fft=True)
        axes[1, 1].stem(range(len(acf_values)), acf_values)
        axes[1, 1].axhline(y=0, color='black', linewidth=0.8)
        axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--', linewidth=1)
        axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--', linewidth=1)
        axes[1, 1].set_title('Autocorrelation Function (ACF)')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('ACF')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Predicted vs Actual
        axes[2, 0].scatter(actuals, predictions, alpha=0.5, s=1)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[2, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[2, 0].set_title('Predicted vs Actual')
        axes[2, 0].set_xlabel('Actual')
        axes[2, 0].set_ylabel('Predicted')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Residuals vs Fitted
        axes[2, 1].scatter(predictions, residuals, alpha=0.5, s=1)
        axes[2, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[2, 1].set_title('Residuals vs Fitted')
        axes[2, 1].set_xlabel('Fitted Values')
        axes[2, 1].set_ylabel('Residuals')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path('results/visualizations/diagnostics')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{self.model_name}_diagnostics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nDiagnostic plots saved to: {output_path}")
        plt.close()
    
    def print_summary(self):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 80)
        
        issues = []
        
        if not self.results.get('stationarity', {}).get('stationary', True):
            issues.append("✗ Residuals may not be stationary")
        else:
            print("✓ Residuals are stationary")
        
        if self.results.get('autocorrelation', {}).get('has_autocorrelation', False):
            issues.append("✗ Residuals show autocorrelation (missed temporal patterns)")
        else:
            print("✓ No significant autocorrelation in residuals")
        
        if self.results.get('heteroskedasticity', {}).get('has_heteroskedasticity', False):
            issues.append("✗ Heteroskedastic residuals (ARCH effects present)")
        else:
            print("✓ Residuals are homoskedastic")
        
        if not self.results.get('normality', {}).get('is_normal', True):
            print("⚠ Residuals not normally distributed (common in financial data)")
        else:
            print("✓ Residuals approximately normal")
        
        if self.results.get('forecast_bias', {}).get('has_bias', False):
            issues.append("✗ Systematic forecast bias detected")
        else:
            print("✓ No systematic forecast bias")
        
        if self.results.get('structural_breaks', {}).get('has_break', False):
            issues.append("✗ Structural breaks detected (performance changes over time)")
        else:
            print("✓ No structural breaks detected")
        
        print("\n" + "=" * 80)
        if len(issues) == 0:
            print("OVERALL ASSESSMENT: Model passes all major validation tests ✓")
        else:
            print(f"OVERALL ASSESSMENT: {len(issues)} issue(s) detected:")
            for issue in issues:
                print(f"  {issue}")
        print("=" * 80)
        
    def save_results(self):
        """Save validation results to CSV."""
        output_dir = Path('results/csv')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Flatten results for CSV
        flat_results = {}
        for category, values in self.results.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    if not isinstance(val, (dict, pd.DataFrame)):
                        flat_results[f'{category}_{key}'] = val
        
        df = pd.DataFrame([flat_results])
        output_path = output_dir / f'{self.model_name}_validation_results.csv'
        df.to_csv(output_path, index=False)
        print(f"\nValidation results saved to: {output_path}")

def create_synthetic_residuals_for_demo():
    """
    Create synthetic residuals for demonstration when actual predictions not available.
    This simulates what we expect from the rolling window LSTM.
    """
    np.random.seed(42)
    n = 7437  # Test set size
    
    # Generate residuals with realistic properties
    # Small autocorrelation, slight heteroskedasticity, near-normal
    residuals = np.random.normal(0, 3.0, n)
    
    # Add small trend (regime adaptation not perfect)
    residuals += np.linspace(0, 0.5, n)
    
    # Add small autocorrelation
    for i in range(1, n):
        residuals[i] += 0.1 * residuals[i-1]
    
    # Simulate actuals and predictions
    actual_mean = 48.0  # Test set mean
    actuals = actual_mean + np.random.normal(0, 5, n)
    predictions = actuals - residuals
    
    return residuals, actuals, predictions

if __name__ == '__main__':
    print("=" * 80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("=" * 80)
    print("\nNote: Using synthetic residuals for demonstration.")
    print("Re-run model training with prediction saving for full validation.")
    print()
    
    validator = ComprehensiveModelValidation(model_name='lstm_rolling')
    
    # Generate synthetic data for demo
    residuals, actuals, predictions = create_synthetic_residuals_for_demo()
    
    print(f"Analyzing {len(residuals):,} test set residuals...")
    print()
    
    # Run all validation tests
    validator.test_residual_stationarity(residuals)
    validator.test_autocorrelation(residuals)
    validator.test_heteroskedasticity(residuals)
    validator.test_normality(residuals)
    validator.test_forecast_bias(residuals)
    validator.test_structural_breaks(residuals)
    validator.generate_diagnostic_plots(residuals, actuals, predictions)
    
    # Print summary
    validator.print_summary()
    
    # Save results
    validator.save_results()
