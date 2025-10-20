import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from model import LSTM_DVOL
from data_loader_jump_aware import create_jump_aware_dataloaders


class JumpAwareTrainer:
    """
    Train LSTM with jump-aware weighted loss.
    
    Key features:
    1. Weighted MSE loss (2x weight for jump periods)
    2. Separate metrics for normal vs. jump forecasting
    3. Jump-specific early stopping
    """
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def weighted_mse_loss(self, predictions, targets, weights):
        """
        MSE loss with sample weights.
        
        Jump periods get 2x weight → model pays more attention to crises.
        """
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        return weighted_mse.mean()
    
    def train_epoch(self, train_loader, optimizer):
        """Train one epoch with weighted loss."""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch, w_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            
            optimizer.zero_grad()
            
            predictions = self.model(X_batch)
            
            loss = self.weighted_mse_loss(predictions, y_batch, w_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader, dataset, split_name='val'):
        """
        Evaluate with separate metrics for normal vs. jump periods.
        
        Returns dict with overall and decomposed metrics.
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_weights = []
        all_stats = []
        
        with torch.no_grad():
            for X_batch, y_batch, w_batch, stats_batch in data_loader:
                X_batch = X_batch.to(self.device)
                
                predictions = self.model(X_batch)
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                all_weights.append(w_batch.cpu().numpy())
                all_stats.append(stats_batch.cpu().numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        weights = np.concatenate(all_weights, axis=0)
        stats = np.concatenate(all_stats, axis=0)
        
        preds_orig = dataset.inverse_transform_target(preds, stats)
        targets_orig = dataset.inverse_transform_target(targets, stats)
        
        is_jump = weights.flatten() > 1.0
        
        overall_metrics = self._compute_metrics(
            preds_orig.flatten(), 
            targets_orig.flatten(),
            prefix='overall'
        )
        
        if is_jump.sum() > 0:
            normal_metrics = self._compute_metrics(
                preds_orig[~is_jump].flatten(),
                targets_orig[~is_jump].flatten(),
                prefix='normal'
            )
        else:
            normal_metrics = {}
        
        if is_jump.sum() > 0:
            jump_metrics = self._compute_metrics(
                preds_orig[is_jump].flatten(),
                targets_orig[is_jump].flatten(),
                prefix='jump'
            )
        else:
            jump_metrics = {}
        
        weighted_loss = self.weighted_mse_loss(
            torch.FloatTensor(preds),
            torch.FloatTensor(targets),
            torch.FloatTensor(weights)
        ).item()
        
        metrics = {
            **overall_metrics,
            **normal_metrics,
            **jump_metrics,
            'weighted_loss': weighted_loss,
            'n_samples': len(preds),
            'n_jumps': int(is_jump.sum()),
            'jump_pct': float(is_jump.mean() * 100)
        }
        
        return metrics, preds_orig, targets_orig, is_jump
    
    def _compute_metrics(self, preds, targets, prefix=''):
        """Compute regression metrics with prefix."""
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        
        mape = np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100
        
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        directions_correct = np.sum(
            np.sign(preds[1:] - targets[:-1]) == np.sign(targets[1:] - targets[:-1])
        )
        directional_accuracy = directions_correct / (len(preds) - 1) * 100
        
        prefix_str = f"{prefix}_" if prefix else ""
        
        return {
            f'{prefix_str}rmse': float(rmse),
            f'{prefix_str}mae': float(mae),
            f'{prefix_str}mape': float(mape),
            f'{prefix_str}r2': float(r2),
            f'{prefix_str}directional_accuracy': float(directional_accuracy)
        }


def train_jump_aware_lstm(
    input_size=11,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    learning_rate=0.001,
    num_epochs=50,
    patience=10,
    batch_size=32,
    device='cuda'
):
    """
    Main training function for jump-aware LSTM.
    
    Innovation: Weighted loss + jump-specific metrics
    """
    print("\n" + "=" * 80)
    print("TRAINING JUMP-AWARE LSTM FOR BITCOIN DVOL FORECASTING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: LSTM (input={input_size}, hidden={hidden_size}, layers={num_layers})")
    print(f"Jump handling: Weighted loss (2x for jumps) + separate metrics")
    print("=" * 80)
    print()
    
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_jump_aware_dataloaders(
        batch_size=batch_size
    )
    
    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    trainer = JumpAwareTrainer(model, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_overall_r2': [],
        'val_normal_r2': [],
        'val_jump_r2': []
    }
    
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer)
        
        val_metrics, _, _, _ = trainer.evaluate(val_loader, val_ds, 'val')
        val_loss = val_metrics['weighted_loss']
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_overall_r2'].append(val_metrics.get('overall_r2', 0))
        history['val_normal_r2'].append(val_metrics.get('normal_r2', 0))
        history['val_jump_r2'].append(val_metrics.get('jump_r2', 0))
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val R²: {val_metrics.get('overall_r2', 0):.4f} | "
              f"Normal R²: {val_metrics.get('normal_r2', 0):.4f} | "
              f"Jump R²: {val_metrics.get('jump_r2', 0):.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("=" * 80)
    print()
    
    model.load_state_dict(best_model_state)
    
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    train_metrics, _, _, _ = trainer.evaluate(train_loader, train_ds, 'train')
    val_metrics, _, _, _ = trainer.evaluate(val_loader, val_ds, 'val')
    test_metrics, test_preds, test_targets, test_is_jump = trainer.evaluate(
        test_loader, test_ds, 'test'
    )
    
    def print_metrics(metrics, split_name):
        print(f"\n{split_name.upper()} SET:")
        print(f"  Samples: {metrics['n_samples']:,} ({metrics['n_jumps']:,} jumps = {metrics['jump_pct']:.1f}%)")
        print(f"  Overall: R²={metrics.get('overall_r2', 0):.4f}, RMSE={metrics.get('overall_rmse', 0):.2f}, "
              f"MAE={metrics.get('overall_mae', 0):.2f}, MAPE={metrics.get('overall_mape', 0):.2f}%, "
              f"Dir={metrics.get('overall_directional_accuracy', 0):.1f}%")
        if 'normal_r2' in metrics:
            print(f"  Normal:  R²={metrics.get('normal_r2', 0):.4f}, RMSE={metrics.get('normal_rmse', 0):.2f}, "
                  f"MAE={metrics.get('normal_mae', 0):.2f}, MAPE={metrics.get('normal_mape', 0):.2f}%, "
                  f"Dir={metrics.get('normal_directional_accuracy', 0):.1f}%")
        if 'jump_r2' in metrics:
            print(f"  Jump:    R²={metrics.get('jump_r2', 0):.4f}, RMSE={metrics.get('jump_rmse', 0):.2f}, "
                  f"MAE={metrics.get('jump_mae', 0):.2f}, MAPE={metrics.get('jump_mape', 0):.2f}%, "
                  f"Dir={metrics.get('jump_directional_accuracy', 0):.1f}%")
    
    print_metrics(train_metrics, 'train')
    print_metrics(val_metrics, 'val')
    print_metrics(test_metrics, 'test')
    
    print("\n" + "=" * 80)
    print()
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    torch.save(best_model_state, 'models/lstm_jump_aware_best.pth')
    print(f"Model saved: models/lstm_jump_aware_best.pth")
    
    metrics_df = pd.DataFrame({
        'split': ['train', 'val', 'test'],
        'overall_r2': [train_metrics['overall_r2'], val_metrics['overall_r2'], test_metrics['overall_r2']],
        'overall_rmse': [train_metrics['overall_rmse'], val_metrics['overall_rmse'], test_metrics['overall_rmse']],
        'overall_mae': [train_metrics['overall_mae'], val_metrics['overall_mae'], test_metrics['overall_mae']],
        'overall_mape': [train_metrics['overall_mape'], val_metrics['overall_mape'], test_metrics['overall_mape']],
        'overall_directional_accuracy': [train_metrics['overall_directional_accuracy'], val_metrics['overall_directional_accuracy'], test_metrics['overall_directional_accuracy']],
        'normal_r2': [train_metrics.get('normal_r2', np.nan), val_metrics.get('normal_r2', np.nan), test_metrics.get('normal_r2', np.nan)],
        'normal_rmse': [train_metrics.get('normal_rmse', np.nan), val_metrics.get('normal_rmse', np.nan), test_metrics.get('normal_rmse', np.nan)],
        'normal_directional_accuracy': [train_metrics.get('normal_directional_accuracy', np.nan), val_metrics.get('normal_directional_accuracy', np.nan), test_metrics.get('normal_directional_accuracy', np.nan)],
        'jump_r2': [train_metrics.get('jump_r2', np.nan), val_metrics.get('jump_r2', np.nan), test_metrics.get('jump_r2', np.nan)],
        'jump_rmse': [train_metrics.get('jump_rmse', np.nan), val_metrics.get('jump_rmse', np.nan), test_metrics.get('jump_rmse', np.nan)],
        'jump_directional_accuracy': [train_metrics.get('jump_directional_accuracy', np.nan), val_metrics.get('jump_directional_accuracy', np.nan), test_metrics.get('jump_directional_accuracy', np.nan)],
        'n_jumps': [train_metrics['n_jumps'], val_metrics['n_jumps'], test_metrics['n_jumps']]
    })
    
    csv_dir = Path('results/csv')
    csv_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(csv_dir / 'lstm_jump_aware_metrics.csv', index=False)
    print(f"Metrics saved: results/csv/lstm_jump_aware_metrics.csv")
    
    visualize_results(history, test_preds, test_targets, test_is_jump)
    
    return model, history, test_metrics


def visualize_results(history, test_preds, test_targets, test_is_jump):
    """Create diagnostic visualizations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Training History: Weighted Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['val_overall_r2'], label='Overall', linewidth=2)
    ax2.plot(history['val_normal_r2'], label='Normal', linewidth=2, linestyle='--')
    ax2.plot(history['val_jump_r2'], label='Jump', linewidth=2, linestyle=':')
    ax2.set_title('Validation R² by Period Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    sample_idx = np.arange(len(test_preds))
    ax3.plot(sample_idx, test_targets, label='Actual', alpha=0.7, linewidth=1)
    ax3.plot(sample_idx, test_preds, label='Predicted', alpha=0.7, linewidth=1)
    jump_idx = sample_idx[test_is_jump.flatten()]
    ax3.scatter(jump_idx, test_targets[test_is_jump.flatten()], 
                color='red', s=20, alpha=0.5, label='Jump Events', zorder=5)
    ax3.set_title('Test Set: Predictions vs Actual', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('DVOL')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2])
    residuals = test_targets.flatten() - test_preds.flatten()
    ax4.scatter(test_preds, residuals, alpha=0.5, s=10)
    ax4.scatter(test_preds[test_is_jump.flatten()], residuals[test_is_jump.flatten()],
                color='red', alpha=0.7, s=15, label='Jump Events')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted DVOL')
    ax4.set_ylabel('Residual')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 0])
    normal_residuals = residuals[~test_is_jump.flatten()]
    jump_residuals = residuals[test_is_jump.flatten()]
    ax5.hist(normal_residuals, bins=50, alpha=0.7, label='Normal', density=True)
    ax5.hist(jump_residuals, bins=30, alpha=0.7, label='Jump', density=True)
    ax5.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Residual')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(test_targets[~test_is_jump.flatten()], 
                test_preds[~test_is_jump.flatten()],
                alpha=0.5, s=10, label='Normal')
    ax6.scatter(test_targets[test_is_jump.flatten()], 
                test_preds[test_is_jump.flatten()],
                alpha=0.7, s=15, color='red', label='Jump')
    min_val = min(test_targets.min(), test_preds.min())
    max_val = max(test_targets.max(), test_preds.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    ax6.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Actual DVOL')
    ax6.set_ylabel('Predicted DVOL')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 2])
    errors = np.abs(residuals)
    normal_errors = errors[~test_is_jump.flatten()]
    jump_errors = errors[test_is_jump.flatten()]
    data_to_plot = [normal_errors, jump_errors]
    bp = ax7.boxplot(data_to_plot, labels=['Normal', 'Jump'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax7.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Absolute Error')
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Jump-Aware LSTM Diagnostics', fontsize=16, fontweight='bold', y=0.995)
    
    viz_dir = Path('results/visualizations/diagnostics')
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = viz_dir / 'lstm_jump_aware_diagnostics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualizations saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, history, test_metrics = train_jump_aware_lstm(
        input_size=11,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        num_epochs=50,
        patience=10,
        batch_size=32,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("JUMP-AWARE LSTM TRAINING COMPLETE")
    print("=" * 80)
    print(f"Overall Test R²: {test_metrics['overall_r2']:.4f}")
    print(f"Normal Period R²: {test_metrics.get('normal_r2', 0):.4f}")
    print(f"Jump Period R²: {test_metrics.get('jump_r2', 0):.4f}")
    print(f"Test MAPE: {test_metrics['overall_mape']:.2f}%")
    print("=" * 80)
