import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append('scripts/modeling')

from data_loader_rolling import RollingWindowDataLoader
from model import LSTM_DVOL
from trainer import train_model
from evaluator import calculate_metrics

def main():
    print("=" * 80)
    print("LSTM with ROLLING WINDOW NORMALIZATION")
    print("=" * 80)
    print("\nThis approach:")
    print("  - Normalizes using local statistics (30-day windows)")
    print("  - Adapts to regime changes (mean shift 69→47)")
    print("  - Preserves feature-target relationships")
    print("  - Avoids trivial solution (predict no change)")
    print()
    
    data_path = 'data/processed/bitcoin_lstm_features.csv'
    
    print("Loading data with rolling window normalization...")
    loader = RollingWindowDataLoader(
        data_path=data_path,
        sequence_length=24,
        forecast_horizon=24,
        rolling_window=720,  # 30 days
        batch_size=32,
        train_ratio=0.6,
        val_ratio=0.2
    )
    loader.prepare_data()
    
    split_stats = loader.get_split_stats()
    print("\nSplit statistics (rolling window adaptation):")
    for split_name, stats in split_stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Normalized target: mean={stats['normalized_mean']:.4f}, std={stats['normalized_std']:.4f}")
        print(f"  Rolling mean (avg): {stats['rolling_mean_avg']:.2f}")
        print(f"  Rolling std (avg):  {stats['rolling_std_avg']:.2f}")
    
    print("\nKey insight:")
    print(f"  Train rolling mean: {split_stats['train']['rolling_mean_avg']:.2f}")
    print(f"  Test rolling mean:  {split_stats['test']['rolling_mean_avg']:.2f}")
    print(f"  → Rolling normalization adapts to regime shift")
    print(f"  → Normalized target remains stationary (mean≈0, std≈1)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"ROCm GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Using GPU 0 for training (distributed training not implemented)")
    
    model = LSTM_DVOL(
        input_size=7,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTraining LSTM with rolling window normalization...")
    print("Expected performance:")
    print("  - R² = 0.75-0.85 (realistic, not trivial)")
    print("  - RMSE = 3-5 (absolute DVOL units)")
    print("  - Directional accuracy = 55-60% (genuine skill)")
    print()
    
    train_model(
        model=model,
        train_loader=loader.train_loader,
        val_loader=loader.val_loader,
        epochs=200,
        lr=1e-4,
        device=device,
        early_stop_patience=15,
        model_save_path='models/lstm_rolling_best.pth'
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    model.eval()
    
    def evaluate_split(data_loader, split_name, split_type='test'):
        predictions_norm = []
        actuals_norm = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                predictions_norm.extend(y_pred.cpu().numpy())
                actuals_norm.extend(y_batch.numpy())
        
        predictions_norm = np.array(predictions_norm).flatten()
        actuals_norm = np.array(actuals_norm).flatten()
        
        predictions_orig = loader.inverse_transform_target(predictions_norm, split=split_type)
        actuals_orig = loader.inverse_transform_target(actuals_norm, split=split_type)
        
        metrics = calculate_metrics(actuals_orig, predictions_orig)
        
        print(f"\n{split_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return metrics, actuals_orig, predictions_orig
    
    train_metrics, train_actual, train_pred = evaluate_split(loader.train_loader, "TRAIN", 'train')
    val_metrics, val_actual, val_pred = evaluate_split(loader.val_loader, "VALIDATION", 'val')
    test_metrics, test_actual, test_pred = evaluate_split(loader.test_loader, "TEST", 'test')
    
    results_df = pd.DataFrame({
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
    
    output_dir = Path('results/csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'lstm_rolling_metrics.csv', index=False)
    print(f"\nMetrics saved to {output_dir}/lstm_rolling_metrics.csv")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH TRIVIAL SOLUTION")
    print("=" * 80)
    print("\nExpected results:")
    print("  LSTM (Differenced): R²=0.997, Dir=51.7% → TRIVIAL (=naive persistence)")
    print("  LSTM (Rolling):     R²=0.75-0.85, Dir=55-60% → GENUINE forecasting")
    print("\nLower R² is GOOD:")
    print("  - High R² on differenced data = learned autocorrelation (trivial)")
    print("  - Moderate R² on absolute data = learned feature relationships (genuine)")
    print("=" * 80)

if __name__ == '__main__':
    main()
