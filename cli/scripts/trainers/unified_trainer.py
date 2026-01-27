"""Unified trainer for all LSTM model types."""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.append('scripts/modeling')
sys.path.append('cli/config')

from model import LSTM_DVOL
from feature_configs import get_feature_config
from data_loader_unified import create_unified_dataloaders


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

    preds_orig = test_dataset.inverse_transform_target(preds, stats)
    targets_orig = test_dataset.inverse_transform_target(targets, stats)

    is_jump = weights.flatten() > 1.0

    def calculate_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        direction_correct = np.mean(
            np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])
        ) * 100 if len(y_pred) > 1 else 50
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2, 'Direction_%': direction_correct}

    overall_metrics = calculate_metrics(targets_orig.flatten(), preds_orig.flatten())

    if is_jump.sum() > 0 and (~is_jump).sum() > 0:
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

    return {'overall': overall_metrics, 'jump_samples': int(is_jump.sum()), 'normal_samples': int((~is_jump).sum())}


def setup_device_and_model(config, input_size):
    """Setup device and model with optional distributed training support."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and config.get('use_multi_gpu', False):
            use_multi_gpu = True
        else:
            use_multi_gpu = False
    else:
        use_multi_gpu = False

    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    if use_multi_gpu and torch.cuda.is_available():
        model = model.to(device)
        model = nn.DataParallel(model)
        effective_batch_size = config['batch_size'] * num_gpus
    else:
        model = model.to(device)
        effective_batch_size = config['batch_size']

    param_count = sum(p.numel() for p in model.parameters())

    return model, device, effective_batch_size, param_count, use_multi_gpu


def train_unified(model_type, config, save_prefix='cli', results_dir='results/cli_training'):
    """Train any LSTM model type using unified framework."""
    feature_config = get_feature_config(model_type)
    input_size = feature_config['input_size']
    use_weighting = feature_config['use_sample_weighting']

    model, device, effective_batch_size, param_count, use_multi_gpu = setup_device_and_model(config, input_size)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_unified_dataloaders(
        data_path='data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv',
        feature_set=model_type,
        sequence_length=config['sequence_length'],
        window_size=config['window_size'],
        batch_size=config['batch_size']
    )

    if hasattr(model, 'module'):
        model.module.device = device
    else:
        model.device = device

    base_lr = config['learning_rate']
    if use_multi_gpu:
        scaled_lr = base_lr * 0.5
    else:
        scaled_lr = base_lr

    optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

    model_path = f'models/{save_prefix}_{model_type}_best.pth'

    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []

        for X_batch, y_batch, w_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if use_weighting:
                w_batch = w_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)

            if use_weighting:
                loss = weighted_mse_loss(predictions, y_batch, w_batch)
            else:
                loss = nn.MSELoss()(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch, w_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if use_weighting:
                    w_batch = w_batch.to(device)

                predictions = model(X_batch)

                if use_weighting:
                    loss = weighted_mse_loss(predictions, y_batch, w_batch)
                else:
                    loss = nn.MSELoss()(predictions, y_batch)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 2 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed/60:.1f}m")

            with open(f'results/logs/current_training.log', 'a') as f:
                f.write(f"{datetime.now().isoformat()} | Epoch {epoch+1:3d}/{config['epochs']} | "
                       f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                       f"Patience: {patience_counter}/{config['patience']}\n")

        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    if hasattr(model, 'module'):
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

    evaluation_results = evaluate_model(model, test_loader, test_ds)

    print(f"\n{'='*80}")
    print(f"RESULTS - {model_type.upper()}")
    print(f"{'='*80}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Model parameters: {param_count:,}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"\nOverall Performance:")
    for metric, value in evaluation_results['overall'].items():
        print(f"  {metric}: {value:.4f}")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    results = {
        'model_type': model_type,
        'config': config,
        'training_time_minutes': training_time / 60,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
        'evaluation': evaluation_results,
        'history': history,
        'model_path': model_path,
        'parameters': param_count
    }

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f'{save_prefix}_{model_type}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Results saved: {results_file}")
    print('='*80)

    return results
