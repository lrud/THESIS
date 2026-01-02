#!/usr/bin/env python3
"""
Rolling Window LSTM Trainer
===========================

Trainer for rolling window LSTM models.

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

from model import LSTM_DVOL
from modeling.data_loader_rolling import RollingWindowDataLoader
# Import directly from utils to avoid import chain issues
try:
    from utils.metrics import calculate_metrics
except ImportError:
    try:
        from scripts.utils.metrics import calculate_metrics
    except ImportError:
        # Fallback: define inline if import fails
        def calculate_metrics(y_true, y_pred):
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')
            if len(y_true) > 1:
                true_direction = np.sign(y_true[1:] - y_true[:-1])
                pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
                direction_correct = np.sum(true_direction == pred_direction)
                directional_accuracy = (direction_correct / (len(y_true) - 1)) * 100
            else:
                directional_accuracy = 0.0
            return {
                'RMSE': float(rmse) if not np.isnan(rmse) else np.nan,
                'MAE': float(mae) if not np.isnan(mae) else np.nan,
                'MAPE': float(mape) if not np.isnan(mape) else np.nan,
                'R²': float(r2) if not np.isnan(r2) else np.nan,
                'Directional_Accuracy_%': float(directional_accuracy)
            }


def train_rolling(config, save_prefix='cli', results_dir='results/cli_training'):
    """Train rolling window LSTM model."""
    print(f"\n{'='*80}")
    print(f"TRAINING ROLLING WINDOW LSTM")
    print(f"{'='*80}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)

    # Device setup with ROCm 7 support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice Setup:")
    print(f"  Device: {device}")

    use_multi_gpu = False
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  ROCm GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # Multi-GPU setup using DataParallel
        if num_gpus > 1 and config.get('use_multi_gpu', False):
            print(f"  Using DataParallel for multi-GPU training ({num_gpus} GPUs)")
            use_multi_gpu = True
        else:
            print(f"  Using single GPU training")
    else:
        print(f"  Using CPU training")

    # Load data - Use v1.1 complete with corrected dvol_rv_spread AND full transaction_volume
    print("Loading data with rolling window normalization...")
    data_path = 'data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv'
    print(f"  Data path: {data_path}")
    loader = RollingWindowDataLoader(
        data_path=data_path,
        sequence_length=config['sequence_length'],
        forecast_horizon=24,
        rolling_window=config['rolling_window'],
        batch_size=config['batch_size'],
        train_ratio=0.6,
        val_ratio=0.2
    )
    loader.prepare_data()

    # Show data statistics
    split_stats = loader.get_split_stats()
    print(f"\nData statistics:")
    for split_name, stats in split_stats.items():
        print(f"  {split_name.upper()}: rolling_mean={stats['rolling_mean_avg']:.2f}, "
              f"rolling_std={stats['rolling_std_avg']:.2f}")

    # Create model
    model = LSTM_DVOL(
        input_size=7,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # Setup device and multi-GPU
    if use_multi_gpu:
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        print(f"  Model wrapped with DataParallel")
        effective_batch_size = config['batch_size'] * num_gpus
        print(f"  Effective batch size: {config['batch_size']} x {num_gpus} = {effective_batch_size}")
    else:
        model = model.to(device)
        effective_batch_size = config['batch_size']

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Training setup with learning rate adjustment for multi-GPU
    base_lr = config['learning_rate']
    if use_multi_gpu:
        # Conservative LR for multi-GPU stability
        actual_lr = base_lr * 0.5
        print(f"  Learning rate: {base_lr} -> {actual_lr} (multi-GPU scaling)")
    else:
        actual_lr = base_lr
        print(f"  Learning rate: {actual_lr}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

    print(f"\nStarting training for up to {config['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []

        for X_batch, y_batch in loader.train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # L2 regularization
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2) ** 2
            loss += 1e-5 * l2_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in loader.val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model (handle DataParallel wrapper)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_path = f'models/{save_prefix}_rolling_best.pth'
            torch.save(model_to_save.state_dict(), model_path)
        else:
            patience_counter += 1

        # Progress reporting
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= config['early_stop_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    # Load best model and evaluate (handle DataParallel wrapper)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(model_path))
    print(f"\nEvaluating best model...")

    # Evaluation - collect predictions with their corresponding rolling stats
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader.test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(y_batch.numpy())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    # Convert back to original scale using the test rolling stats
    # The rolling stats are per-sample (each time point has its own mean/std)
    test_rolling_mean = loader.test_rolling_stats['mean'].flatten()
    test_rolling_std = loader.test_rolling_stats['std'].flatten()

    # Inverse transform: original = normalized * std + mean
    test_pred_orig = test_predictions.flatten() * test_rolling_std + test_rolling_mean
    test_target_orig = test_targets.flatten() * test_rolling_std + test_rolling_mean

    # Calculate metrics
    metrics = calculate_metrics(test_target_orig, test_pred_orig)

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")

    print(f"\nTest Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare results for saving
    results = {
        'model_type': 'rolling',
        'config': config,
        'training_time_minutes': training_time / 60,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
        'evaluation': metrics,
        'history': history,
        'model_path': model_path,
        'parameters': param_count
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
    results_file = results_path / f'{save_prefix}_rolling_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n✅ Training completed successfully!")
    print(f"📁 Results saved: {results_file}")
    print(f"🏋️ Model saved: {model_path}")
    print('='*80)

    return results