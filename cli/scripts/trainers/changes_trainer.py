#!/usr/bin/env python3
"""
Differenced Target LSTM Trainer
==============================

Trainer for LSTM models predicting next-period DVOL changes (Δdvol).
Enables direct comparison to XGBoost Spec D benchmarks.

Target: Δdvol = dvol_{t+1} - dvol_t (stationary by construction)
Features: 7 core features or Spec D features (9 features)

Author: Claude Code Assistant
Date: December 30, 2025
Purpose: Option C - Train both LSTM variants (levels + changes)
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
sys.path.append('..')  # Go up to cli/
sys.path.append('../..')  # Go up to project root

from model import LSTM_DVOL
from modeling.data_loader_changes import create_changes_dataloaders
# Import metrics - try both paths for flexibility
try:
    from utils.metrics import calculate_metrics
except ImportError:
    from scripts.utils.metrics import calculate_metrics


def setup_device_and_model(config, input_size):
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


def train_changes(config, save_prefix='cli', results_dir='results/cli_training',
                  use_spec_d=False, jump_aware=False):
    """
    Train differenced target LSTM model.

    Args:
        config: Training configuration
        save_prefix: Prefix for saved model files
        results_dir: Directory for results
        use_spec_d: If True, use Spec D features (XGBoost match)
        jump_aware: If True, evaluate separately on jump vs normal periods
    """
    model_suffix = 'spec_d' if use_spec_d else 'changes'
    if jump_aware:
        model_suffix += '_jump_aware'

    print(f"\n{'='*80}")
    print(f"TRAINING DIFFERENCED TARGET LSTM ({model_suffix.upper()})")
    print(f"{'='*80}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"Features: {'Spec D (9 features)' if use_spec_d else 'Core (7 features)'}")
    print(f"Jump-Aware: {jump_aware}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)

    # Setup device and model with multi-GPU support
    input_size = 9 if use_spec_d else 7
    model, device, effective_batch_size, param_count = setup_device_and_model(config, input_size)

    # Store device for evaluation
    if hasattr(model, 'module'):
        model.module.device = device
    else:
        model.device = device

    # Load data
    print("\nLoading data...")
    data_path = 'data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv'
    train_loader, val_loader, test_loader, test_dataset = create_changes_dataloaders(
        data_path=data_path,
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size'],
        train_ratio=0.7,
        val_ratio=0.15,
        jump_aware=jump_aware,
        use_spec_d=use_spec_d
    )

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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=1e-5)
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

        if jump_aware:
            # X_batch, y_batch, jump_indicator
            for batch in train_loader:
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
        else:
            # X_batch, y_batch
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            if jump_aware:
                for batch in val_loader:
                    X_batch = batch[0].to(device)
                    y_batch = batch[1].to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_losses.append(loss.item())
            else:
                for X_batch, y_batch in val_loader:
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
            # Save best model
            model_path = f'models/{save_prefix}_{model_suffix}_best.pth'
            # Handle DataParallel model saving
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        # Progress reporting
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= config.get('patience', 15):
            print(f"Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    # Load best model and evaluate
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
    print(f"\nEvaluating best model...")

    # Evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    test_jumps = []

    with torch.no_grad():
        if jump_aware:
            for X_batch, y_batch, jump_batch in test_loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch)
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend(y_batch.numpy())
                test_jumps.extend(jump_batch.numpy())
        else:
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch)
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend(y_batch.numpy())

    test_predictions = np.array(test_predictions).flatten()
    test_targets = np.array(test_targets).flatten()
    test_jumps = np.array(test_jumps).flatten() if jump_aware else None

    # Calculate metrics
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Overall metrics
    metrics = calculate_metrics(test_targets, test_predictions)
    print(f"\nOverall Test Performance ({len(test_targets):,} samples):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    # Jump period metrics (if jump_aware)
    jump_metrics = None
    if jump_aware and test_jumps is not None:
        jump_mask = test_jumps == 1
        normal_mask = test_jumps == 0

        if jump_mask.sum() > 0:
            jump_preds = test_predictions[jump_mask]
            jump_targets = test_targets[jump_mask]
            jump_metrics = calculate_metrics(jump_targets, jump_preds)

            print(f"\nJump Periods ({jump_mask.sum():,} samples, {jump_mask.sum()/len(test_jumps)*100:.1f}%):")
            for metric, value in jump_metrics.items():
                print(f"  {metric}: {value}")

        if normal_mask.sum() > 0:
            normal_preds = test_predictions[normal_mask]
            normal_targets = test_targets[normal_mask]
            normal_metrics = calculate_metrics(normal_targets, normal_preds)

            print(f"\nNormal Periods ({normal_mask.sum():,} samples, {normal_mask.sum()/len(test_jumps)*100:.1f}%):")
            for metric, value in normal_metrics.items():
                print(f"  {metric}: {value}")

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    results = {
        'model_type': f'differenced_{model_suffix}',
        'config': config,
        'use_spec_d': use_spec_d,
        'jump_aware': jump_aware,
        'training_time_minutes': training_time / 60,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
        'evaluation': metrics,
        'evaluation_jump': jump_metrics,
        'history': history,
        'model_path': model_path,
        'parameters': param_count,
        'data_path': data_path
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
    results_file = results_path / f'{save_prefix}_{model_suffix}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n✅ Training completed successfully!")
    print(f"📁 Results saved: {results_file}")
    print(f"🏋️ Model saved: {model_path}")
    print('='*80)

    return results


if __name__ == "__main__":
    # Test the trainer
    config = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'sequence_length': 24,
        'patience': 5,
        'use_multi_gpu': False
    }

    print("Testing differenced trainer with core features...")
    train_changes(config, save_prefix='test', use_spec_d=False, jump_aware=False)

    print("\n" + "="*80 + "\n")

    print("Testing differenced trainer with Spec D features...")
    train_changes(config, save_prefix='test', use_spec_d=True, jump_aware=False)
