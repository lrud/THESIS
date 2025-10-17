"""
Main script for LSTM training with DIFFERENCED target variable.

This implements the fix for non-stationarity identified in baseline model.
Target is transformed to first differences: Δdvol_t = dvol_t - dvol_{t-1}

Usage:
    python scripts/modeling/main_differenced.py
"""

import os
import torch
from torch.utils.data import DataLoader

from data_loader_differenced import prepare_data_differenced, reconstruct_from_diff
from model import DVOLDataset, create_model, count_parameters
from trainer import train_model
from evaluator import evaluate_model, plot_training_history, plot_predictions


def main():
    """Main training and evaluation pipeline for differenced LSTM."""
    # Configuration
    MODEL_SAVE_PATH = 'models/lstm_differenced_best.pth'
    
    # Hyperparameters
    SEQUENCE_LENGTH = 24
    FORECAST_HORIZON = 24
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOP_PATIENCE = 15
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"LSTM TRAINING WITH DIFFERENCED TARGET (Fix for Non-Stationarity)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"{'='*80}\n")
    
    # Step 1: Load and prepare data with differencing
    print("📥 Loading and preparing data with first-difference transformation...")
    data_dict = prepare_data_differenced(
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON
    )
    
    # Step 2: Create datasets and loaders
    print("\n🔄 Creating data loaders...")
    train_dataset = DVOLDataset(data_dict['X_train'], data_dict['y_diff_train'])
    val_dataset = DVOLDataset(data_dict['X_val'], data_dict['y_diff_val'])
    test_dataset = DVOLDataset(data_dict['X_test'], data_dict['y_diff_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Step 3: Create model
    print("\n🏗️  Creating model...")
    model = create_model(
        input_size=data_dict['X_train'].shape[2],
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        l2_reg=1e-4,
        device=device
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Step 4: Train model
    print("\n🚀 Starting training on differenced target (Δdvol)...\n")
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        early_stop_patience=EARLY_STOP_PATIENCE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Step 5: Load best model
    print("\n📂 Loading best model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Step 6: Get predictions on differenced scale
    print("\n📊 Getting predictions on differenced scale...")
    model.eval()
    
    def get_predictions_differenced(loader):
        """Get predictions on differenced scale."""
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.numpy())
        return np.concatenate(actuals), np.concatenate(predictions)
    
    import numpy as np
    
    # Get differenced predictions
    y_diff_train_actual, y_diff_train_pred = get_predictions_differenced(train_loader)
    y_diff_val_actual, y_diff_val_pred = get_predictions_differenced(val_loader)
    y_diff_test_actual, y_diff_test_pred = get_predictions_differenced(test_loader)
    
    # Step 7: Inverse transform differences
    print("\n🔄 Inverse transforming differences to original scale...")
    scaler = data_dict['scaler_y_diff']
    
    y_diff_train_pred_orig = scaler.inverse_transform(y_diff_train_pred)
    y_diff_val_pred_orig = scaler.inverse_transform(y_diff_val_pred)
    y_diff_test_pred_orig = scaler.inverse_transform(y_diff_test_pred)
    
    # Step 8: Reconstruct original DVOL values
    print("🔧 Reconstructing original DVOL values...")
    
    y_train_reconstructed = reconstruct_from_diff(
        y_diff_train_pred_orig, 
        data_dict['y_prev_train'][:len(y_diff_train_pred_orig)]
    )
    y_val_reconstructed = reconstruct_from_diff(
        y_diff_val_pred_orig,
        data_dict['y_prev_val'][:len(y_diff_val_pred_orig)]
    )
    y_test_reconstructed = reconstruct_from_diff(
        y_diff_test_pred_orig,
        data_dict['y_prev_test'][:len(y_diff_test_pred_orig)]
    )
    
    # Get actual values (reconstruct from differences)
    y_diff_train_actual_orig = scaler.inverse_transform(y_diff_train_actual)
    y_diff_val_actual_orig = scaler.inverse_transform(y_diff_val_actual)
    y_diff_test_actual_orig = scaler.inverse_transform(y_diff_test_actual)
    
    y_train_actual = reconstruct_from_diff(
        y_diff_train_actual_orig,
        data_dict['y_prev_train'][:len(y_diff_train_actual_orig)]
    )
    y_val_actual = reconstruct_from_diff(
        y_diff_val_actual_orig,
        data_dict['y_prev_val'][:len(y_diff_val_actual_orig)]
    )
    y_test_actual = reconstruct_from_diff(
        y_diff_test_actual_orig,
        data_dict['y_prev_test'][:len(y_diff_test_actual_orig)]
    )
    
    # Step 9: Evaluate on reconstructed values
    print("\n📊 Evaluating model on reconstructed DVOL values...\n")
    
    from evaluator import calculate_metrics
    
    train_metrics = calculate_metrics(y_train_actual, y_train_reconstructed)
    val_metrics = calculate_metrics(y_val_actual, y_val_reconstructed)
    test_metrics = calculate_metrics(y_test_actual, y_test_reconstructed)
    
    print(f"{'='*80}")
    print(f"Train Set Evaluation (Reconstructed DVOL)")
    print(f"{'='*80}")
    for metric_name, value in train_metrics.items():
        print(f"{metric_name:25s}: {value:10.4f}")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print(f"Validation Set Evaluation (Reconstructed DVOL)")
    print(f"{'='*80}")
    for metric_name, value in val_metrics.items():
        print(f"{metric_name:25s}: {value:10.4f}")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print(f"Test Set Evaluation (Reconstructed DVOL)")
    print(f"{'='*80}")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name:25s}: {value:10.4f}")
    print(f"{'='*80}\n")
    
    # Step 10: Generate plots
    print("\n📈 Generating visualizations...")
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/csv', exist_ok=True)
    
    plot_training_history(history, 'results/visualizations/lstm_differenced_training_history.png')
    plot_predictions(y_train_actual, y_train_reconstructed, 'Train (Differenced)', 
                    'results/visualizations/lstm_differenced_train_predictions.png')
    plot_predictions(y_val_actual, y_val_reconstructed, 'Validation (Differenced)', 
                    'results/visualizations/lstm_differenced_val_predictions.png')
    plot_predictions(y_test_actual, y_test_reconstructed, 'Test (Differenced)', 
                    'results/visualizations/lstm_differenced_test_predictions.png')
    
    # Step 11: Save metrics to CSV
    print("\n💾 Saving metrics to CSV...")
    import pandas as pd
    
    metrics_df = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
        'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']],
        'R²': [train_metrics['R²'], val_metrics['R²'], test_metrics['R²']],
        'Directional_Accuracy_%': [
            train_metrics['Directional_Accuracy_%'],
            val_metrics['Directional_Accuracy_%'],
            test_metrics['Directional_Accuracy_%']
        ]
    })
    metrics_df.to_csv('results/csv/lstm_differenced_metrics.csv', index=False)
    print("✅ Saved: results/csv/lstm_differenced_metrics.csv")
    
    print("\n✅ Pipeline complete!")
    print(f"\n📁 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📁 Visualizations saved to: results/visualizations/")
    print(f"📁 Metrics saved to: results/csv/")
    
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*80}")
    print("Baseline (absolute DVOL) had:")
    print("  - Test R²: -5.92 (catastrophic)")
    print("  - Test Directional Accuracy: 2.16%")
    print("  - Straight-line predictions")
    print()
    print("Differenced model has:")
    print(f"  - Test R²: {test_metrics['R²']:.2f}")
    print(f"  - Test Directional Accuracy: {test_metrics['Directional_Accuracy_%']:.2f}%")
    print("  - (Check visualizations for prediction quality)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
