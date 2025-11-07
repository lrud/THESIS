#!/usr/bin/env python3
"""
Differenced LSTM Trainer
========================

Trainer for differenced LSTM models.

Author: Claude Code Assistant
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add paths for imports
sys.path.append('scripts/modeling')
sys.path.append('scripts')

from model import create_model
from modeling.data_loader_differenced import prepare_data_differenced
from modeling.trainer import train_model


def train_differenced(config, save_prefix='cli', results_dir='results/cli_training'):
    """Train differenced LSTM model."""
    print(f"\n{'='*80}")
    print(f"TRAINING DIFFERENCED LSTM")
    print(f"{'='*80}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading differenced data...")
    train_loader, val_loader, test_loader, test_dataset = prepare_data_differenced(
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon'],
        batch_size=config['batch_size'],
        train_ratio=0.6,
        val_ratio=0.2
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    input_size = train_loader.dataset.X.shape[2]
    model = create_model(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=device
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Input features: {input_size}")

    # Train using existing trainer
    print(f"\nStarting training for up to {config['epochs']} epochs...")
    start_time = time.time()

    model_path = f'models/{save_prefix}_differenced_best.pth'

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        device=device,
        early_stop_patience=config['early_stop_patience'],
        model_save_path=model_path
    )

    training_time = time.time() - start_time

    # Load best model for evaluation
    model.load_state_dict(torch.load(model_path))
    print(f"\nEvaluating best model...")

    # Simple evaluation
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(y_batch.numpy())

    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets).flatten()

    # Reconstruct from differences (simplified)
    # Note: This is a basic reconstruction - full implementation would need proper inverse differencing
    if hasattr(test_dataset, 'mean_dvol'):
        test_pred_reconstructed = test_predictions + test_dataset.mean_dvol
        test_target_reconstructed = test_targets + test_dataset.mean_dvol
    else:
        # If mean is not available, just report differenced metrics
        test_pred_reconstructed = test_predictions
        test_target_reconstructed = test_targets

    # Calculate basic metrics
    mse = np.mean((test_target_reconstructed - test_pred_reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_target_reconstructed - test_pred_reconstructed))

    # Note: MAPE and R² for differenced data can be misleading
    # This is expected to achieve high R² (~0.997) but represents a trivial solution
    if np.any(test_target_reconstructed != 0):
        mape = np.mean(np.abs((test_target_reconstructed - test_pred_reconstructed) / test_target_reconstructed)) * 100
        ss_res = np.sum((test_target_reconstructed - test_pred_reconstructed) ** 2)
        ss_tot = np.sum((test_target_reconstructed - np.mean(test_target_reconstructed)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        mape = float('inf')
        r2 = 0

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'Note': 'High R² expected but represents trivial solution (persistence)'
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")

    print(f"\nTest Performance:")
    for metric, value in metrics.items():
        if metric != 'Note':
            print(f"  {metric}: {value:.4f}")
    print(f"  {metrics['Note']}")

    print(f"\n⚠️  WARNING: Differenced models typically achieve high R² but represent trivial solutions")
    print(f"    equivalent to naive persistence (predicting minimal change).")

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare results for saving
    results = {
        'model_type': 'differenced',
        'config': config,
        'training_time_minutes': training_time / 60,
        'best_val_loss': min(history['val_loss']),
        'timestamp': datetime.now().isoformat(),
        'evaluation': metrics,
        'history': history,
        'model_path': model_path,
        'parameters': param_count,
        'warning': 'Differenced models often represent trivial solutions'
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results_serializable = convert_numpy(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f'{save_prefix}_differenced_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n✅ Training completed successfully!")
    print(f"📁 Results saved: {results_file}")
    print(f"🏋️ Model saved: {model_path}")
    print('='*80)

    return results