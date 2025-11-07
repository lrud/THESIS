"""
Main script for LSTM baseline training.

Usage:
    python scripts/modeling/main.py
"""

import os
import torch
from torch.utils.data import DataLoader

from data_loader import prepare_data
from model import DVOLDataset, create_model, count_parameters
from trainer import train_model
from evaluator import evaluate_model, plot_training_history, plot_predictions


def main():
    """Main training and evaluation pipeline."""
    # Configuration
    MODEL_SAVE_PATH = 'models/lstm_baseline_best.pth'
    
    # Hyperparameters
    SEQUENCE_LENGTH = 24
    FORECAST_HORIZON = 24
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOP_PATIENCE = 15
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"{'='*60}\n")
    
    # Step 1: Load and prepare data
    print("📥 Loading and preparing data...")
    data_dict = prepare_data(
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON
    )
    
    # Step 2: Create datasets and loaders
    print(" Creating data loaders...")
    train_dataset = DVOLDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = DVOLDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = DVOLDataset(data_dict['X_test'], data_dict['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Step 3: Create model
    print("\n🏗  Creating model...")
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
    print("\n🚀 Starting training...\n")
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
    print("\n Loading best model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Step 6: Evaluate on all datasets
    print("\n📊 Evaluating model...\n")
    
    # Train set
    train_metrics, y_train_true, y_train_pred = evaluate_model(
        model, train_loader, data_dict['scaler_y'], device, 'Train'
    )
    
    # Validation set
    val_metrics, y_val_true, y_val_pred = evaluate_model(
        model, val_loader, data_dict['scaler_y'], device, 'Validation'
    )
    
    # Test set
    test_metrics, y_test_true, y_test_pred = evaluate_model(
        model, test_loader, data_dict['scaler_y'], device, 'Test'
    )
    
    # Step 7: Generate plots
    print("\n Generating visualizations...")
    os.makedirs('results/visualizations/lstm', exist_ok=True)
    os.makedirs('results/csv', exist_ok=True)
    
    plot_training_history(history, 'results/visualizations/lstm/training_history.png')
    plot_predictions(y_train_true, y_train_pred, 'Train', 'results/visualizations/lstm/train_predictions.png')
    plot_predictions(y_val_true, y_val_pred, 'Validation', 'results/visualizations/lstm/val_predictions.png')
    plot_predictions(y_test_true, y_test_pred, 'Test', 'results/visualizations/lstm/test_predictions.png')
    
    print("\n Pipeline complete!")
    print(f"\n Model saved to: {MODEL_SAVE_PATH}")
    print(f" Visualizations saved to: results/visualizations/lstm/")
    print(f" CSV outputs saved to: results/csv/")


if __name__ == '__main__':
    main()
