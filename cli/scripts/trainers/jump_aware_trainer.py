#!/usr/bin/env python3
"""
Jump-Aware LSTM Trainer
======================

Trainer for jump-aware LSTM models with weighted loss.

Author: Claude Code Assistant
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add paths for imports
sys.path.append('scripts/modeling')
sys.path.append('scripts')
sys.path.append('src/core')

from model import LSTM_DVOL
from data_loader_jump_aware import create_jump_aware_dataloaders


def weighted_mse_loss(predictions, targets, weights):
    """Weighted MSE loss for jump-aware training."""
    mse = (predictions - targets) ** 2
    weighted_mse = mse * weights
    return weighted_mse.mean()


def evaluate_model(model, test_loader, test_dataset):
    """Evaluate trained model and return metrics."""
    model.eval()

    all_preds = []
    all_targets = []
    all_weights = []
    all_stats = []

    with torch.no_grad():
        for X_batch, y_batch, w_batch, stats_batch in test_loader:
            X_batch = X_batch.to(model.device if hasattr(model, 'device') else 'cuda')
            predictions = model(X_batch)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_weights.append(w_batch.cpu().numpy())
            all_stats.append(stats_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    stats = np.concatenate(all_stats, axis=0)

    # Convert back to original scale
    preds_orig = test_dataset.inverse_transform_target(preds, stats)
    targets_orig = test_dataset.inverse_transform_target(targets, stats)

    # Calculate metrics
    is_jump = weights.flatten() > 1.0

    def calculate_metrics(y_true, y_pred):
        """Calculate standard metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Directional accuracy
        direction_correct = np.mean(
            np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])
        ) * 100 if len(y_pred) > 1 else 50

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Direction_%': direction_correct
        }

    overall_metrics = calculate_metrics(targets_orig.flatten(), preds_orig.flatten())
    normal_metrics = calculate_metrics(
        targets_orig.flatten()[~is_jump],
        preds_orig.flatten()[~is_jump]
    )
    jump_metrics = calculate_metrics(
        targets_orig.flatten()[is_jump],
        preds_orig.flatten()[is_jump]
    )

    return {
        'overall': overall_metrics,
        'normal': normal_metrics,
        'jump': jump_metrics,
        'jump_samples': int(is_jump.sum()),
        'normal_samples': int((~is_jump).sum())
    }


def setup_device_and_model(config, input_size=11):
    """Setup device and model with optional distributed training support."""
    # Device setup with ROCm 7 support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice Setup:")
    print(f"  Device: {device}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  ROCm GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # Multi-GPU setup using DataParallel (simpler than DDP, works well for 2 GPUs)
        if num_gpus > 1 and config.get('use_multi_gpu', False):
            print(f"  Using DataParallel for multi-GPU training ({num_gpus} GPUs)")
            use_multi_gpu = True
        else:
            print(f"  Using single GPU training")
            use_multi_gpu = False
    else:
        print(f"  Using CPU training")
        use_multi_gpu = False

    # Create model
    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # Setup device and multi-GPU
    if use_multi_gpu and torch.cuda.is_available():
        model = model.to(device)
        model = nn.DataParallel(model)
        print(f"  Model wrapped with DataParallel")
        # Adjust batch size for multi-GPU
        effective_batch_size = config['batch_size'] * num_gpus
        print(f"  Effective batch size: {config['batch_size']} x {num_gpus} = {effective_batch_size}")
    else:
        model = model.to(device)
        effective_batch_size = config['batch_size']

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    return model, device, effective_batch_size, param_count


def train_jump_aware(config, save_prefix='cli', results_dir='results/cli_training'):
    """Train jump-aware LSTM model with optional multi-GPU support."""
    print(f"\n{'='*80}")
    print(f"TRAINING JUMP-AWARE LSTM")
    print(f"{'='*80}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)

    # Setup device and model with multi-GPU support
    model, device, effective_batch_size, param_count = setup_device_and_model(config, input_size=11)

    # Load data with effective batch size for multi-GPU
    print("Loading data...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_jump_aware_dataloaders(
        sequence_length=config['sequence_length'],
        window_size=config['window_size'],
        batch_size=config['batch_size']  # Original batch size, DataParallel handles distribution
    )

    # Store device for evaluation
    if hasattr(model, 'module'):
        model.module.device = device
    else:
        model.device = device

    # Training setup with learning rate adjustment for multi-GPU
    base_lr = config['learning_rate']
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        # For DataParallel, use a more conservative approach:
        # 1. Reduce learning rate for multi-GPU stability
        # 2. Add weight decay for regularization
        # 3. Use gradient clipping
        scaled_lr = base_lr * 0.5  # More conservative LR for multi-GPU
        print(f"  Multi-GPU training: reduced LR {base_lr} -> {scaled_lr}")
        print(f"  Conservative settings for DataParallel stability")
    else:
        scaled_lr = base_lr

    optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=1e-5)  # Add weight decay for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

    # Initialize model path outside the loop
    model_path = f'models/{save_prefix}_jump_aware_best.pth'

    print(f"\nStarting training for up to {config['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []

        for X_batch, y_batch, w_batch, _ in train_loader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = weighted_mse_loss(predictions, y_batch, w_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch, w_batch, _ in val_loader:
                X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
                predictions = model(X_batch)
                loss = weighted_mse_loss(predictions, y_batch, w_batch)
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
            # Save best model - handle DataParallel properly
            if hasattr(model, 'module'):
                # DataParallel model - save the underlying model
                torch.save(model.module.state_dict(), model_path)
            else:
                # Regular model
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        # Enhanced real-time progress reporting
        if (epoch + 1) % 2 == 0 or epoch == 0:  # Report every 2 epochs
            elapsed = time.time() - start_time
            remaining = (config['epochs'] - epoch - 1) * (elapsed / (epoch + 1))

            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed/60:.1f}m | "
                  f"ETA: {remaining/60:.1f}m")

            # Save progress log to file for real-time monitoring
            with open(f'results/logs/current_training.log', 'a') as f:
                log_msg = (f"{datetime.now().isoformat()} | "
                        f"Epoch {epoch+1:3d}/{config['epochs']} | "
                        f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"Patience: {patience_counter}/{config['patience']}\n"
)
                f.write(log_msg)

        if patience_counter >= config['patience']:
            print(f"🏁 Early stopping at epoch {epoch+1} (patience: {patience_counter}/{config['patience']})")
            break

    training_time = time.time() - start_time

    # Load best model and evaluate - handle DataParallel properly
    if hasattr(model, 'module'):
        # DataParallel model - load into underlying model
        model.module.load_state_dict(torch.load(model_path))
    else:
        # Regular model
        model.load_state_dict(torch.load(model_path))
    print(f"\nEvaluating best model...")
    evaluation_results = evaluate_model(model, test_loader, test_ds)

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")

    print(f"\nOverall Performance ({evaluation_results['normal_samples'] + evaluation_results['jump_samples']} samples):")
    for metric, value in evaluation_results['overall'].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nNormal Periods ({evaluation_results['normal_samples']} samples):")
    for metric, value in evaluation_results['normal'].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nJump Periods ({evaluation_results['jump_samples']} samples):")
    for metric, value in evaluation_results['jump'].items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare results for saving

    results = {
        'model_type': 'jump_aware',
        'config': config,
        'training_time_minutes': training_time / 60,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
        'evaluation': evaluation_results,
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
    results_file = results_path / f'{save_prefix}_jump_aware_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n✅ Training completed successfully!")
    print(f"📁 Results saved: {results_file}")
    print(f"🏋️ Model saved: {model_path}")
    print('='*80)

    return results