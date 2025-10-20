"""
Main script for training and evaluating naive baseline models.
Usage: python scripts/benchmarking/main_naive_baselines.py
"""

import os
import numpy as np
import pandas as pd

from models.naive_models import create_naive_models, get_aligned_actuals_naive
from utils.data_loader_har import prepare_har_rv_data
from utils.evaluator_har import calculate_metrics, plot_predictions_comparison, plot_scatter_comparison, save_metrics_to_csv


def main():
    print(f"\n{'='*80}")
    print(f"NAIVE BASELINE MODELS")
    print(f"{'='*80}")
    print(f"\nSimple forecasting models as reality checks for complex approaches")
    print(f"Models: Persistence, Drift, Mean, MA(5), MA(22)")
    print(f"{'='*80}\n")
    
    FORECAST_HORIZON = 24
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    
    os.makedirs('../../models', exist_ok=True)
    os.makedirs('../../results/visualizations/naive', exist_ok=True)
    os.makedirs('../../results/csv', exist_ok=True)
    
    print("Step 1: Loading data...")
    data_dict = prepare_har_rv_data(
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )
    
    print(f"\nStep 2: Creating naive baseline models...")
    models = create_naive_models(forecast_horizon=FORECAST_HORIZON)
    
    for name, model in models.items():
        print(f"  - {name}: {model}")
    
    print(f"\nStep 3: Training models (where applicable)...")
    for name, model in models.items():
        if hasattr(model, 'fit') and name in ['Drift', 'Mean']:
            model.fit(data_dict['rv_train'])
            if name == 'Drift':
                print(f"  {name}: drift = {model.get_drift():.6f}")
            elif name == 'Mean':
                print(f"  {name}: mean = {model.get_mean():.2f}")
    
    print(f"\nStep 4: Generating predictions...")
    
    split_indices = data_dict['split_indices']
    
    results = {}
    
    for name, model in models.items():
        print(f"  {name}...")
        
        pred_train = model.predict(data_dict['rv_train'])
        y_true_train = get_aligned_actuals_naive(data_dict['rv_train'], FORECAST_HORIZON)
        
        pred_val_full = model.predict(data_dict['rv_val'])
        y_true_val_full = get_aligned_actuals_naive(data_dict['rv_val'], FORECAST_HORIZON)
        
        pred_test_full = model.predict(data_dict['rv_test'])
        y_true_test_full = get_aligned_actuals_naive(data_dict['rv_test'], FORECAST_HORIZON)
        
        val_start = split_indices['train_end'] - FORECAST_HORIZON
        val_end = split_indices['val_end'] - FORECAST_HORIZON
        pred_val = pred_val_full[val_start:val_end]
        y_true_val = y_true_val_full[val_start:val_end]
        
        test_start = split_indices['val_end'] - FORECAST_HORIZON
        pred_test = pred_test_full[test_start:]
        y_true_test = y_true_test_full[test_start:]
        
        metrics_train = calculate_metrics(y_true_train, pred_train)
        metrics_val = calculate_metrics(y_true_val, pred_val)
        metrics_test = calculate_metrics(y_true_test, pred_test)
        
        results[name] = {
            'train': (y_true_train, pred_train, metrics_train),
            'val': (y_true_val, pred_val, metrics_val),
            'test': (y_true_test, pred_test, metrics_test)
        }
    
    print(f"\nStep 5: Displaying results...")
    
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS (All Naive Models)")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<15} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'Dir%':>8}")
    print("-" * 80)
    
    for name in models.keys():
        metrics = results[name]['test'][2]
        print(f"{name:<15} {metrics['R²']:>8.4f} {metrics['RMSE']:>8.2f} "
              f"{metrics['MAE']:>8.2f} {metrics['MAPE']:>7.2f}% {metrics['Directional_Accuracy_%']:>7.2f}%")
    
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS BY MODEL")
    print(f"{'='*80}")
    
    for name, model in models.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        
        metrics_train = results[name]['train'][2]
        metrics_val = results[name]['val'][2]
        metrics_test = results[name]['test'][2]
        
        print(f"Train - R²: {metrics_train['R²']:.4f}, MAPE: {metrics_train['MAPE']:.2f}%, Dir: {metrics_train['Directional_Accuracy_%']:.2f}%")
        print(f"Val   - R²: {metrics_val['R²']:.4f}, MAPE: {metrics_val['MAPE']:.2f}%, Dir: {metrics_val['Directional_Accuracy_%']:.2f}%")
        print(f"Test  - R²: {metrics_test['R²']:.4f}, MAPE: {metrics_test['MAPE']:.2f}%, Dir: {metrics_test['Directional_Accuracy_%']:.2f}%")
    
    print(f"\nStep 6: Saving results...")
    
    metrics_dict = {}
    for name in models.keys():
        metrics_dict[f'Naive_{name}_Train'] = results[name]['train'][2]
        metrics_dict[f'Naive_{name}_Val'] = results[name]['val'][2]
        metrics_dict[f'Naive_{name}_Test'] = results[name]['test'][2]
    
    save_metrics_to_csv(metrics_dict, '../../results/csv/naive_baselines_metrics.csv')
    
    print(f"\nStep 7: Creating visualizations...")
    
    for name in models.keys():
        y_true_test, pred_test, _ = results[name]['test']
        
        plot_predictions_comparison(
            y_true_test,
            {f'Naive-{name}': pred_test},
            dataset_name=f'Test Set - {name}',
            save_path=f'../../results/visualizations/naive/predictions_{name.lower()}.png',
            max_samples=500
        )
        
        plot_scatter_comparison(
            y_true_test,
            {f'Naive-{name}': pred_test},
            dataset_name=f'Test Set - {name}',
            save_path=f'../../results/visualizations/naive/scatter_{name.lower()}.png'
        )
    
    all_predictions = {}
    y_true_test = None
    for name in ['Persistence', 'Drift', 'MA22']:
        y_true_test, pred_test, _ = results[name]['test']
        all_predictions[f'Naive-{name}'] = pred_test
    
    plot_predictions_comparison(
        y_true_test,
        all_predictions,
        dataset_name='Test Set - Naive Models Comparison',
        save_path='../../results/visualizations/naive/comparison_all.png',
        max_samples=500
    )
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nBest Naive Model (by Test R²):")
    
    best_name = max(models.keys(), key=lambda x: results[x]['test'][2]['R²'])
    best_metrics = results[best_name]['test'][2]
    
    print(f"  Model: {best_name}")
    print(f"  R²:    {best_metrics['R²']:.4f}")
    print(f"  MAPE:  {best_metrics['MAPE']:.2f}%")
    print(f"  Dir:   {best_metrics['Directional_Accuracy_%']:.2f}%")
    
    print(f"\nResults saved to:")
    print(f"  Metrics:        results/csv/naive_baselines_metrics.csv")
    print(f"  Visualizations: results/visualizations/naive/*.png")
    
    print(f"\n{'='*80}")
    print(f"REALITY CHECK")
    print(f"{'='*80}")
    print(f"\nCompare with complex models:")
    print(f"  HAR-RV (Absolute):   R² = 0.9649")
    print(f"  LSTM (Differenced):  R² = 0.9970")
    print(f"  HAR-RV (Differenced): R² = 0.9970")
    print(f"  Best Naive ({best_name}): R² = {best_metrics['R²']:.4f}")
    
    if best_metrics['R²'] > 0.99:
        print(f"\nWARNING: Naive baseline matches complex models!")
        print(f"Complex models may not provide meaningful forecasting value.")
    elif best_metrics['R²'] > 0.95:
        print(f"\nCAUTION: Naive baseline performs very well.")
        print(f"Complex models show modest improvement.")
    else:
        print(f"\nGOOD: Complex models substantially outperform naive baselines.")
        print(f"Complexity is justified.")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
