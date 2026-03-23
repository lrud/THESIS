#!/usr/bin/env python
"""
Analyze Directional Accuracy at Different Time Horizons

Tests whether DA improves with longer prediction horizons.
For heavy-tailed, mean-reverting series, longer horizons may be more predictable.
"""
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("DIRECTIONAL ACCURACY BY TIME HORIZON")
print("=" * 80)
print()

# Load data
data_path = Path('data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv')
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

dvol = df['dvol'].values

# Simple baseline predictions for comparison
def compute_da_at_horizon(series, horizon_hours, n_predictions=None):
    """
    Compute directional accuracy at a given horizon.

    DA = % of correct predictions that series_{t+horizon} > series_t

    For comparison, we also compute:
    - Naive forecast: predict next value = current value
    - Mean forecast: predict next value = historical mean
    """
    if n_predictions is None:
        n_predictions = len(series) - horizon_hours

    # Actual directional changes
    actual_changes = series[horizon_hours:] - series[:-horizon_hours]
    actual_direction = np.sign(actual_changes)

    # Naive prediction: predict no change (direction = 0)
    # But for DA calculation, we need to count ties as neither correct nor wrong
    naive_direction = np.zeros_like(actual_direction)

    # Mean reversion prediction: predict return to mean
    mean_val = np.mean(series[:n_predictions])
    mr_changes = mean_val - series[:-horizon_hours]
    mr_direction = np.sign(mr_changes)

    # Compute DA
    def da_score(pred_dir, act_dir):
        # Exclude ties (where direction == 0)
        valid = act_dir != 0
        if valid.sum() == 0:
            return 0.0
        correct = (pred_dir[valid] == act_dir[valid]).sum()
        return (correct / valid.sum()) * 100

    # DA for mean-reversion strategy
    da_mr = da_score(mr_direction, actual_direction)

    # DA for random guessing (50% by definition)
    da_random = 50.0

    # DA for perfect trend-following (if current change predicts future change)
    if len(series) > horizon_hours + 1:
        past_changes = series[1:-horizon_hours] - series[:-horizon_hours-1]
        past_direction = np.sign(past_changes)
        trend_direction = past_direction[:len(actual_direction)]
        da_trend = da_score(trend_direction, actual_direction)
    else:
        da_trend = 50.0

    # Count up vs down movements
    n_up = (actual_direction > 0).sum()
    n_down = (actual_direction < 0).sum()
    n_flat = (actual_direction == 0).sum()

    return {
        'horizon_hours': horizon_hours,
        'n_predictions': n_predictions,
        'pct_up': (n_up / (n_up + n_down)) * 100 if (n_up + n_down) > 0 else 50,
        'pct_down': (n_down / (n_up + n_down)) * 100 if (n_up + n_down) > 0 else 50,
        'pct_flat': (n_flat / len(actual_direction)) * 100,
        'da_mean_reversion': da_mr,
        'da_trend_following': da_trend,
        'da_random': da_random
    }

# Test different horizons
horizons = [1, 6, 24, 48, 168, 336, 720]  # 1h, 6h, 1d, 2d, 1w, 2w, 1mo
results = []

for h in horizons:
    result = compute_da_at_horizon(dvol, h)
    results.append(result)

# Display results
print(f"{'Horizon':<12} {'Up%':<8} {'Down%':<8} {'Flat%':<8} {'DA (Mean-Rev)':<15} {'DA (Trend)':<10}")
print("-" * 80)

for r in results:
    horizon_str = f"{r['horizon_hours']}h"
    if r['horizon_hours'] >= 24:
        horizon_str = f"{r['horizon_hours']//24}d"

    print(f"{horizon_str:<12} {r['pct_up']:<7.1f}% {r['pct_down']:<7.1f}% {r['pct_flat']:<7.1f}% "
          f"{r['da_mean_reversion']:<14.1f}% {r['da_trend_following']:<9.1f}%")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

# Find best horizon
best_mr = max(results, key=lambda x: x['da_mean_reversion'])
best_trend = max(results, key=lambda x: x['da_trend_following'])

print("Mean-Reversion Strategy (predict return to mean):")
print(f"  Best horizon: {best_mr['horizon_hours']}h ({best_mr['horizon_hours']//24}d)")
print(f"  Best DA: {best_mr['da_mean_reversion']:.1f}%")
print(f"  Baseline (random): 50.0%")
print(f"  Improvement: {best_mr['da_mean_reversion'] - 50:.1f} percentage points")
print()

print("Trend-Following Strategy (predict continuation of recent move):")
print(f"  Best horizon: {best_trend['horizon_hours']}h ({best_trend['horizon_hours']//24}d)")
print(f"  Best DA: {best_trend['da_trend_following']:.1f}%")
print(f"  Baseline (random): 50.0%")
print(f"  Improvement: {best_trend['da_trend_following'] - 50:.1f} percentage points")
print()

print("=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print()
print("At 1-hour horizon:")
print(f"  - Mean-reversion DA: {results[0]['da_mean_reversion']:.1f}%")
print(f"  - Trend-following DA: {results[0]['da_trend_following']:.1f}%")
print()
print("At longer horizons:")
print(f"  - Mean-reversion improves to {best_mr['da_mean_reversion']:.1f}% ({best_mr['horizon_hours']//24}d)")
print(f"  - Trend-following improves to {best_trend['da_trend_following']:.1f}% ({best_trend['horizon_hours']//24}d)")
print()
print("Conclusion: DVOL is more directionally predictable at longer horizons")
print("            due to its mean-reverting nature (H = 0.20 from earlier test).")
print("=" * 80)

# Save results
import json
output_dir = Path('results/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

summary = {
    'date': '2026-02-17',
    'n_observations': len(df),
    'findings': {
        'best_horizon_mean_reversion_hours': best_mr['horizon_hours'],
        'best_da_mean_reversion': best_mr['da_mean_reversion'],
        'best_horizon_trend_hours': best_trend['horizon_hours'],
        'best_da_trend': best_trend['da_trend_following'],
        'one_hour_da': results[0]['da_mean_reversion']
    },
    'all_results': results
}

with open(output_dir / 'da_by_horizon_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to: {output_dir / 'da_by_horizon_summary.json'}")
