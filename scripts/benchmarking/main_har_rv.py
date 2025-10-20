"""
Main script for training and evaluating HAR-RV model.
Usage: python scripts/benchmarking/main_har_rv.py
"""

import os
import numpy as np
from pathlib import Path

from models.har_rv_model import HARRV, HARRVConfig, create_har_rv_model
from utils.data_loader_har import prepare_har_rv_data, extract_predictions_for_split, get_aligned_actuals
from utils.evaluator_har import (
    calculate_metrics, 
    print_metrics_comparison,
    plot_predictions_comparison,
    plot_scatter_comparison,
    save_metrics_to_csv
)


def main():
    print(f"\n{'='*80}")
    print(f"HAR-RV MODEL BENCHMARK")
    print(f"{'='*80}")
    print(f"\nReference: Corsi (2009)")
    print(f"Model: DVOL_t+24h = β₀ + β_d·DVOL_t + β_w·DVOL_t^(week) + β_m·DVOL_t^(month) + ε_t")
    print(f"{'='*80}\n")
    
    FORECAST_HORIZON = 24
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/visualizations/har_rv', exist_ok=True)
    os.makedirs('results/csv', exist_ok=True)
    
    print("Step 1: Loading and preparing data...")
    data_dict = prepare_har_rv_data(
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )
    
    print(f"\nStep 2: Creating HAR-RV model...")
    config = HARRVConfig(
        daily_lag=1,
        weekly_lag=5,
        monthly_lag=22,
        forecast_horizon=FORECAST_HORIZON
    )
    model = HARRV(config)
    print(f"  {model}")
    
    print(f"\nStep 3: Training model on {len(data_dict['rv_train']):,} samples...")
    model.fit(data_dict['rv_train'])
    
    print(f"\nStep 4: Generating predictions...")
    
    print("  Training set...")
    y_pred_train_full = model.predict(data_dict['rv_train'])
    y_true_train = get_aligned_actuals(
        data_dict['rv_train'],
        forecast_horizon=FORECAST_HORIZON,
        monthly_lag=config.monthly_lag
    )
    
    print("  Validation set...")
    y_pred_val_full = model.predict(data_dict['rv_val'])
    y_true_val = get_aligned_actuals(
        data_dict['rv_val'],
        forecast_horizon=FORECAST_HORIZON,
        monthly_lag=config.monthly_lag
    )
    
    print("  Test set...")
    y_pred_test_full = model.predict(data_dict['rv_test'])
    y_true_test = get_aligned_actuals(
        data_dict['rv_test'],
        forecast_horizon=FORECAST_HORIZON,
        monthly_lag=config.monthly_lag
    )
    
    split_indices = data_dict['split_indices']
    val_start = split_indices['train_end'] - config.monthly_lag - FORECAST_HORIZON + 1
    val_end = split_indices['val_end'] - config.monthly_lag - FORECAST_HORIZON + 1
    y_pred_val = y_pred_val_full[val_start:val_end]
    y_true_val_split = y_true_val[val_start:val_end]
    
    test_start = split_indices['val_end'] - config.monthly_lag - FORECAST_HORIZON + 1
    y_pred_test = y_pred_test_full[test_start:]
    y_true_test_split = y_true_test[test_start:]
    
    print(f"\n  Shapes: Train={len(y_pred_train_full)}, Val={len(y_pred_val)}, Test={len(y_pred_test)}")
    
    print(f"\nStep 5: Evaluating performance...")
    metrics_train = calculate_metrics(y_true_train, y_pred_train_full)
    metrics_val = calculate_metrics(y_true_val_split, y_pred_val)
    metrics_test = calculate_metrics(y_true_test_split, y_pred_test)
    
    print(f"\n{'='*80}")
    print(f"TRAINING SET RESULTS")
    print(f"{'='*80}")
    for metric, value in metrics_train.items():
        print(f"  {metric:<25}: {value:>10.4f}")
    
    print(f"\n{'='*80}")
    print(f"VALIDATION SET RESULTS")
    print(f"{'='*80}")
    for metric, value in metrics_val.items():
        print(f"  {metric:<25}: {value:>10.4f}")
    
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    for metric, value in metrics_test.items():
        print(f"  {metric:<25}: {value:>10.4f}")
    
    print(f"\nStep 6: Saving results...")
    metrics_dict = {
        'HAR-RV_Train': metrics_train,
        'HAR-RV_Val': metrics_val,
        'HAR-RV_Test': metrics_test
    }
    save_metrics_to_csv(metrics_dict, 'results/csv/har_rv_metrics.csv')
    
    import pandas as pd
    coef_df = pd.DataFrame([model.get_coefficients()])
    coef_df.to_csv('results/csv/har_rv_coefficients.csv', index=False)
    print(f"Coefficients saved to: results/csv/har_rv_coefficients.csv")
    
    print(f"\nStep 7: Creating visualizations...")
    plot_predictions_comparison(
        y_true_train,
        {'HAR-RV': y_pred_train_full},
        dataset_name='Training Set',
        save_path='results/visualizations/har_rv/predictions_train.png',
        max_samples=500
    )
    
    plot_predictions_comparison(
        y_true_val_split,
        {'HAR-RV': y_pred_val},
        dataset_name='Validation Set',
        save_path='results/visualizations/har_rv/predictions_val.png',
        max_samples=500
    )
    
    plot_predictions_comparison(
        y_true_test_split,
        {'HAR-RV': y_pred_test},
        dataset_name='Test Set',
        save_path='results/visualizations/har_rv/predictions_test.png',
        max_samples=500
    )
    
    plot_scatter_comparison(
        y_true_test_split,
        {'HAR-RV': y_pred_test},
        dataset_name='Test Set',
        save_path='results/visualizations/har_rv/scatter_test.png'
    )
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nHAR-RV model training complete")
    print(f"\nTest Set Performance:")
    print(f"  R²:                    {metrics_test['R²']:.4f}")
    print(f"  MAPE:                  {metrics_test['MAPE']:.2f}%")
    print(f"  Directional Accuracy:  {metrics_test['Directional_Accuracy_%']:.2f}%")
    print(f"\nResults saved to:")
    print(f"  Metrics:        results/csv/har_rv_metrics.csv")
    print(f"  Coefficients:   results/csv/har_rv_coefficients.csv")
    print(f"  Visualizations: results/visualizations/har_rv/*.png")
    print(f"\nNext: Run comparison script for LSTM vs HAR-RV analysis")
    print(f"  python scripts/benchmarking/compare_models.py")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
