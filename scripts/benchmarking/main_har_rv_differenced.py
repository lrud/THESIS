"""
Main script for HAR-RV with DIFFERENCED target variable.
Usage: python scripts/benchmarking/main_har_rv_differenced.py
"""

import os
import numpy as np
import pandas as pd

from models.har_rv_differenced import HARRVDifferenced, HARRVDiffConfig, create_har_rv_differenced
from utils.data_loader_har import prepare_har_rv_data
from utils.evaluator_har import calculate_metrics, plot_predictions_comparison, plot_scatter_comparison, save_metrics_to_csv


def main():
    print(f"\n{'='*80}")
    print(f"HAR-RV MODEL WITH DIFFERENCED TARGET")
    print(f"{'='*80}")
    print(f"\nReference: Corsi (2009) + First Differences")
    print(f"Target: Δdvol_t = dvol_t - dvol_{{t-1}}")
    print(f"Model: Δdvol_t+24h = β₀ + β_d·DVOL_t + β_w·DVOL_t^(week) + β_m·DVOL_t^(month) + ε_t")
    print(f"{'='*80}\n")
    
    FORECAST_HORIZON = 24
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/visualizations/har_rv_diff', exist_ok=True)
    os.makedirs('results/csv', exist_ok=True)
    
    print("Step 1: Loading and preparing data...")
    data_dict = prepare_har_rv_data(
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )
    
    print(f"\nStep 2: Creating HAR-RV (Differenced) model...")
    config = HARRVDiffConfig(
        daily_lag=1,
        weekly_lag=5,
        monthly_lag=22,
        forecast_horizon=FORECAST_HORIZON
    )
    model = HARRVDifferenced(config)
    print(f"  {model}")
    
    print(f"\nStep 3: Training on DIFFERENCED target ({len(data_dict['rv_train']):,} samples)...")
    model.fit(data_dict['rv_train'])
    
    print(f"\nStep 4: Generating predictions...")
    
    print("  Training set...")
    y_diff_pred_train, y_recon_pred_train = model.predict(data_dict['rv_train'])
    _, y_diff_true_train, y_prev_train = model._create_har_features_diff(data_dict['rv_train'])
    y_true_train = y_prev_train + y_diff_true_train
    
    print("  Validation set...")
    y_diff_pred_val_full, y_recon_pred_val_full = model.predict(data_dict['rv_val'])
    _, y_diff_true_val_full, y_prev_val_full = model._create_har_features_diff(data_dict['rv_val'])
    y_true_val_full = y_prev_val_full + y_diff_true_val_full
    
    print("  Test set...")
    y_diff_pred_test_full, y_recon_pred_test_full = model.predict(data_dict['rv_test'])
    _, y_diff_true_test_full, y_prev_test_full = model._create_har_features_diff(data_dict['rv_test'])
    y_true_test_full = y_prev_test_full + y_diff_true_test_full
    
    split_indices = data_dict['split_indices']
    val_start = split_indices['train_end'] - config.monthly_lag - FORECAST_HORIZON
    val_end = split_indices['val_end'] - config.monthly_lag - FORECAST_HORIZON
    
    y_pred_val = y_recon_pred_val_full[val_start:val_end]
    y_true_val = y_true_val_full[val_start:val_end]
    y_diff_pred_val = y_diff_pred_val_full[val_start:val_end]
    y_diff_true_val = y_diff_true_val_full[val_start:val_end]
    
    test_start = split_indices['val_end'] - config.monthly_lag - FORECAST_HORIZON
    y_pred_test = y_recon_pred_test_full[test_start:]
    y_true_test = y_true_test_full[test_start:]
    y_diff_pred_test = y_diff_pred_test_full[test_start:]
    y_diff_true_test = y_diff_true_test_full[test_start:]
    
    print(f"\n  Shapes: Train={len(y_recon_pred_train)}, Val={len(y_pred_val)}, Test={len(y_pred_test)}")
    
    print(f"\nStep 5: Evaluating performance...")
    
    print("\n  A) Metrics on DIFFERENCED scale (Δdvol):")
    metrics_diff_train = calculate_metrics(y_diff_true_train, y_diff_pred_train)
    metrics_diff_val = calculate_metrics(y_diff_true_val, y_diff_pred_val)
    metrics_diff_test = calculate_metrics(y_diff_true_test, y_diff_pred_test)
    
    print(f"    Train - R²: {metrics_diff_train['R²']:.4f}, MAPE: {metrics_diff_train['MAPE']:.2f}%")
    print(f"    Val   - R²: {metrics_diff_val['R²']:.4f}, MAPE: {metrics_diff_val['MAPE']:.2f}%")
    print(f"    Test  - R²: {metrics_diff_test['R²']:.4f}, MAPE: {metrics_diff_test['MAPE']:.2f}%")
    
    print("\n  B) Metrics on RECONSTRUCTED scale (absolute DVOL):")
    metrics_recon_train = calculate_metrics(y_true_train, y_recon_pred_train)
    metrics_recon_val = calculate_metrics(y_true_val, y_pred_val)
    metrics_recon_test = calculate_metrics(y_true_test, y_pred_test)
    
    print(f"    Train - R²: {metrics_recon_train['R²']:.4f}, MAPE: {metrics_recon_train['MAPE']:.2f}%")
    print(f"    Val   - R²: {metrics_recon_val['R²']:.4f}, MAPE: {metrics_recon_val['MAPE']:.2f}%")
    print(f"    Test  - R²: {metrics_recon_test['R²']:.4f}, MAPE: {metrics_recon_test['MAPE']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"TRAINING SET RESULTS (Reconstructed Scale)")
    print(f"{'='*80}")
    for metric, value in metrics_recon_train.items():
        print(f"  {metric:<25}: {value:>10.4f}")
    
    print(f"\n{'='*80}")
    print(f"VALIDATION SET RESULTS (Reconstructed Scale)")
    print(f"{'='*80}")
    for metric, value in metrics_recon_val.items():
        print(f"  {metric:<25}: {value:>10.4f}")
    
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS (Reconstructed Scale)")
    print(f"{'='*80}")
    for metric, value in metrics_recon_test.items():
        print(f"  {metric:<25}: {value:>10.4f}")
    
    print(f"\nStep 6: Saving results...")
    
    metrics_dict = {
        'HAR-RV-Diff_Train': metrics_recon_train,
        'HAR-RV-Diff_Val': metrics_recon_val,
        'HAR-RV-Diff_Test': metrics_recon_test
    }
    save_metrics_to_csv(metrics_dict, 'results/csv/har_rv_differenced_metrics.csv')
    
    coef_df = pd.DataFrame([model.get_coefficients()])
    coef_df.to_csv('results/csv/har_rv_differenced_coefficients.csv', index=False)
    print(f"Coefficients saved to: results/csv/har_rv_differenced_coefficients.csv")
    
    print(f"\nStep 7: Creating visualizations...")
    plot_predictions_comparison(
        y_true_train,
        {'HAR-RV-Diff': y_recon_pred_train},
        dataset_name='Training Set (Reconstructed)',
        save_path='results/visualizations/har_rv_diff/predictions_train.png',
        max_samples=500
    )
    
    plot_predictions_comparison(
        y_true_val,
        {'HAR-RV-Diff': y_pred_val},
        dataset_name='Validation Set (Reconstructed)',
        save_path='results/visualizations/har_rv_diff/predictions_val.png',
        max_samples=500
    )
    
    plot_predictions_comparison(
        y_true_test,
        {'HAR-RV-Diff': y_pred_test},
        dataset_name='Test Set (Reconstructed)',
        save_path='results/visualizations/har_rv_diff/predictions_test.png',
        max_samples=500
    )
    
    plot_scatter_comparison(
        y_true_test,
        {'HAR-RV-Diff': y_pred_test},
        dataset_name='Test Set (Reconstructed)',
        save_path='results/visualizations/har_rv_diff/scatter_test.png'
    )
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nHAR-RV (Differenced) model training complete")
    print(f"\nTest Set Performance (Reconstructed DVOL):")
    print(f"  R²:                    {metrics_recon_test['R²']:.4f}")
    print(f"  MAPE:                  {metrics_recon_test['MAPE']:.2f}%")
    print(f"  Directional Accuracy:  {metrics_recon_test['Directional_Accuracy_%']:.2f}%")
    print(f"\nTest Set Performance (Differenced Δdvol):")
    print(f"  R²:                    {metrics_diff_test['R²']:.4f}")
    print(f"  MAPE:                  {metrics_diff_test['MAPE']:.2f}%")
    print(f"\nResults saved to:")
    print(f"  Metrics:        results/csv/har_rv_differenced_metrics.csv")
    print(f"  Coefficients:   results/csv/har_rv_differenced_coefficients.csv")
    print(f"  Visualizations: results/visualizations/har_rv_diff/*.png")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
