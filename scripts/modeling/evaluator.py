"""
Evaluation metrics and visualization for LSTM model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Directional accuracy
    direction_correct = np.sum(np.sign(y_true[1:] - y_true[:-1]) == 
                               np.sign(y_pred[1:] - y_pred[:-1]))
    directional_accuracy = (direction_correct / (len(y_true) - 1)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'Directional_Accuracy_%': directional_accuracy
    }


def get_predictions(model, data_loader, device):
    """Get predictions from model."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    return actuals, predictions


def plot_training_history(history, save_path='results/visualizations/training_history.png'):
    """Plot training and validation loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training History', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Learning rate plot
    ax2.plot(epochs, history['learning_rate'], color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training history saved to {save_path}")


def plot_predictions(y_true, y_pred, dataset_name, save_path=None):
    """Plot predictions vs actuals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series plot
    indices = range(len(y_true))
    ax1.plot(indices, y_true, label='Actual', alpha=0.7, linewidth=1.5)
    ax1.plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('DVOL', fontsize=12)
    ax1.set_title(f'{dataset_name} - Predictions vs Actuals', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual DVOL', fontsize=12)
    ax2.set_ylabel('Predicted DVOL', fontsize=12)
    ax2.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 {dataset_name} predictions saved to {save_path}")
    else:
        plt.show()


def evaluate_model(model, test_loader, scaler_y, device, dataset_name='Test'):
    """Complete evaluation pipeline."""
    # Get predictions
    y_true, y_pred = get_predictions(model, test_loader, device)
    
    # Inverse transform
    y_true_orig = scaler_y.inverse_transform(y_true)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_orig, y_pred_orig)
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Evaluation")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:25s}: {value:10.4f}")
    print(f"{'='*60}\n")
    
    return metrics, y_true_orig, y_pred_orig
