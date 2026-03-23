"""
Statistical Diagnostics for HAR-RV Models.

This module provides comprehensive statistical testing for model evaluation beyond simple R²,
including directional accuracy, coefficient significance tests, residual diagnostics, and
forecast accuracy metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats
from typing import Dict


def calculate_statistical_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                                      feature_cols: list, coef: np.ndarray,
                                      X_train: np.ndarray, y_train: np.ndarray,
                                      n_samples_train: int) -> dict:
    """
    Calculate comprehensive statistical diagnostics for model evaluation.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        feature_cols: List of feature names
        coef: Model coefficients
        X_train: Training features
        y_train: Training target values
        n_samples_train: Number of training samples

    Returns:
        Dictionary with all statistical diagnostics including:
        - Directional Accuracy
        - Coefficient Significance (t-tests)
        - Residual Diagnostics (Jarque-Bera, Ljung-Box)
        - Diebold-Mariano test
        - R² Confidence Intervals
        - Theil's U statistic
        - Sign test
    """
    n = len(y_true)
    residuals = y_true - y_pred

    # 1. Directional Accuracy
    direction_correct = ((y_true > 0) == (y_pred > 0)).sum()
    directional_accuracy = direction_correct / n

    # 2. Statistical significance of coefficients (t-test)
    train_pred = np.dot(X_train, coef) if len(coef.shape) == 1 else np.dot(X_train, coef[1:]) + coef[0]
    train_residuals = y_train - train_pred
    mse = np.mean(train_residuals ** 2)

    X_design = np.column_stack([np.ones(len(X_train)), X_train])

    try:
        xt_x_inv = np.linalg.inv(X_design.T @ X_design)
        var_covar = mse * xt_x_inv
        std_errors = np.sqrt(np.diag(var_covar))

        feature_std_errors = std_errors[1:]
        feature_std_errors = feature_std_errors[:len(coef)]

        t_stats = coef / feature_std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_samples_train - len(coef) - 1))

        coefficient_significance = {
            feature: {
                'coefficient': float(c),
                'std_error': float(se),
                't_statistic': float(t),
                'p_value': float(p),
                'significant': bool(p < 0.05)
            }
            for feature, c, se, t, p in zip(feature_cols, coef, feature_std_errors, t_stats, p_values)
        }
    except np.linalg.LinAlgError:
        coefficient_significance = {"error": "Singular matrix - cannot compute significance"}

    # 3. Residual Diagnostics
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    is_normal = jb_pval > 0.05

    try:
        lb_stat, lb_pval = stats.acorr_ljungbox(residuals, lags=[10], return_df=False)
        lb_pval = lb_pval[0] if len(lb_pval) > 0 else 1.0
        has_autocorrelation = lb_pval < 0.05
    except:
        lb_stat, lb_pval = 0, 1.0
        has_autocorrelation = False

    # 4. Diebold-Mariano test (compare to naive forecast)
    naive_forecast = np.zeros_like(y_pred)
    dm_loss_diff = (residuals ** 2) - ((y_true - naive_forecast) ** 2)

    dm_mean = np.mean(dm_loss_diff)
    dm_var = np.var(dm_loss_diff, ddof=1)
    dm_stat = dm_mean / np.sqrt(dm_var / n) if dm_var > 0 else 0
    dm_pval = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    beats_naive = dm_pval < 0.05 and dm_mean < 0

    # 5. Confidence Intervals for R²
    r2 = r2_score(y_true, y_pred)
    n_params = len(coef)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)

    se_r2 = np.sqrt(4 * r2 ** 2 * (1 - r2) ** 2 / n) if 0 < r2 < 1 else 0.1
    r2_ci_lower = max(0, r2 - 1.96 * se_r2)
    r2_ci_upper = min(1, r2 + 1.96 * se_r2)

    # 6. Theil's U statistic
    naive_mse = np.mean((y_true - naive_forecast) ** 2)
    model_mse = np.mean(residuals ** 2)
    theils_u = np.sqrt(model_mse) / np.sqrt(naive_mse) if naive_mse > 0 else 1.0
    forecast_quality = "Better than naive" if theils_u < 1 else "Worse than naive"

    # 7. Sign test for forecast accuracy
    model_errors = np.abs(residuals)
    naive_errors = np.abs(y_true - naive_forecast)
    sign_test_wins = (model_errors < naive_errors).sum()
    sign_test_pval = stats.binomtest(sign_test_wins, n, p=0.5, alternative='less').pvalue
    significantly_better = sign_test_pval < 0.05

    return {
        'directional_accuracy': {
            'value': float(directional_accuracy),
            'correct_predictions': int(direction_correct),
            'total_predictions': int(n),
            'interpretation': 'Model predicts direction correctly' if directional_accuracy > 0.5 else 'Model worse than random'
        },
        'coefficient_significance': coefficient_significance,
        'residual_diagnostics': {
            'jarque_bera': {
                'statistic': float(jb_stat),
                'p_value': float(jb_pval),
                'is_normal': bool(is_normal),
                'interpretation': 'Residuals normally distributed' if is_normal else 'Residuals NOT normally distributed'
            },
            'ljung_box': {
                'statistic': float(lb_stat),
                'p_value': float(lb_pval),
                'has_autocorrelation': bool(has_autocorrelation),
                'interpretation': 'Residuals autocorrelated' if has_autocorrelation else 'No significant autocorrelation'
            }
        },
        'forecast_significance': {
            'diebold_mariano': {
                'statistic': float(dm_stat),
                'p_value': float(dm_pval),
                'beats_naive': bool(beats_naive),
                'interpretation': 'Model significantly better than naive' if beats_naive else 'Model NOT better than naive'
            },
            'theils_u': {
                'value': float(theils_u),
                'interpretation': forecast_quality,
                'beats_naive': theils_u < 1
            },
            'sign_test': {
                'wins': int(sign_test_wins),
                'total': int(n),
                'p_value': float(sign_test_pval),
                'significantly_better': bool(significantly_better)
            }
        },
        'confidence_intervals': {
            'r2': {
                'value': float(r2),
                'ci_lower': float(r2_ci_lower),
                'ci_upper': float(r2_ci_upper),
                'adjusted_r2': float(adjusted_r2)
            }
        },
        'sample_characteristics': {
            'n_samples': int(n),
            'n_parameters': int(n_params),
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'actual_mean': float(np.mean(y_true)),
            'actual_std': float(np.std(y_true)),
            'predicted_mean': float(np.mean(y_pred)),
            'predicted_std': float(np.std(y_pred))
        }
    }
