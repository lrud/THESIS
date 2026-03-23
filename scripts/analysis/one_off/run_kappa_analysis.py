#!/usr/bin/env python
"""
Run Kappa Analysis - Directional Accuracy Investigation

Analyzes the R² vs Directional Accuracy mismatch using the quadratic relationship:
    E[R²_OOS] = kappa * (2p - 1)²
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

def main():
    print("=" * 90)
    print("KAPPA PARAMETER ANALYSIS")
    print("=" * 90)
    print()
    print("Theoretical Framework (Zhang, 2026):")
    print("  E[R²_OOS] = kappa * (2p - 1)²")
    print()

    # Unified framework results (13 linear + tree models)
    unified_results = [
        ('OLS_WithLags_Jumps', 0.9490, 51.2, 1.65, 'Linear'),
        ('RF_Lags_Jumps', 0.9492, 51.0, 1.65, 'Tree'),
        ('RF_Lags', 0.9485, 51.4, 1.66, 'Tree'),
        ('OLS_WithLags', 0.9480, 51.2, 1.67, 'Linear'),
        ('HAR_RV', 0.9454, 50.6, 1.71, 'Linear'),
        ('XGB_Lags', 0.9429, 51.1, 1.75, 'Tree'),
        ('XGB_Lags_Jumps', 0.9384, 50.4, 1.82, 'Tree'),
        ('RF_NoLag_Jumps', 0.7564, 51.0, 3.61, 'Tree'),
        ('OLS_NoLags_Jumps', 0.7393, 51.3, 3.74, 'Linear'),
        ('OLS_NoLags', 0.7363, 51.3, 3.76, 'Linear'),
        ('XGB_NoLag_Jumps', 0.7304, 51.1, 3.80, 'Tree'),
        ('XGB_NoLag', 0.6989, 50.2, 4.02, 'Tree'),
        ('RF_NoLag', 0.6914, 50.8, 4.06, 'Tree')
    ]

    # LSTM results from CLI training (fixed evaluation)
    lstm_results = [
        ('LSTM_market_lags_512x7', 0.9287, 50.1, 1.71, 'LSTM'),
        ('LSTM_jump_aware_512x7', 0.7986, 49.6, 2.87, 'LSTM'),
        ('LSTM_market_jumps_512x7', 0.6100, 49.6, 4.00, 'LSTM'),
        ('LSTM_market_512x7', 0.6135, 49.4, 3.98, 'LSTM'),
        ('LSTM_market_256x3', 0.6145, 50.8, 3.97, 'LSTM')
    ]

    # Combine all results
    all_results = unified_results + lstm_results

    # Create DataFrame
    df = pd.DataFrame(all_results,
                      columns=['Model', 'R2', 'DA', 'RMSE', 'type'])
    df['p'] = df['DA'] / 100.0
    df['kappa'] = df['R2'] / ((2 * df['p'] - 1) ** 2)
    df['kappa_vs_gaussian'] = df['kappa'] / 0.64
    df = df.sort_values('R2', ascending=False)

    # Print summary
    print(f"Total models: {len(df)}")
    print(f"  Linear: {len(df[df['type']=='Linear'])}")
    print(f"  Tree: {len(df[df['type']=='Tree'])}")
    print(f"  LSTM: {len(df[df['type']=='LSTM'])}")
    print()

    # Print table
    print("-" * 90)
    print(f"{'Model':<30} {'R2':>8} {'DA%':>7} {'kappa':>10} {'kappa/0.64':>10}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['Model']:<30} {row['R2']:>8.4f} {row['DA']:>6.1f}% {row['kappa']:>10.2f} {row['kappa_vs_gaussian']:>10.2f}")

    # Statistics
    print("-" * 90)
    print(f"{'Mean:':<40} {df['kappa'].mean():>10.2f} {df['kappa_vs_gaussian'].mean():>10.2f}")
    print(f"{'Median:':<40} {df['kappa'].median():>10.2f} {df['kappa_vs_gaussian'].median():>10.2f}")
    print(f"{'Std:':<40} {df['kappa'].std():>10.2f}")
    print()

    # Key findings
    print("=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    print()
    print(f"1. Mean kappa = {df['kappa'].mean():.2f}")
    print(f"   This is {df['kappa'].mean()/0.64:.1f}x the Gaussian baseline (0.64)")
    print()
    print("2. Interpretation:")
    print("   - For p = 0.50 (50% DA): (2p - 1)² = 0")
    print("   - For p = 0.51 (51% DA): (2p - 1)² = 0.0004")
    print(f"   - With kappa = {df['kappa'].mean():.0f}: R² ≈ {df['kappa'].mean() * 0.0004:.4f}")
    print()
    print("3. Conclusion: High kappa explains high R² with ~50% DA")
    print("   - Heavy-tailed DVOL distribution allows high level accuracy")
    print("   - Directional prediction remains ~50% (effectively random)")
    print("=" * 90)

    # Save results
    output_dir = Path('results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'kappa_analysis_results.csv', index=False)

    summary = {
        'date': '2026-02-17',
        'n_models': int(len(df)),
        'mean_kappa': float(df['kappa'].mean()),
        'median_kappa': float(df['kappa'].median()),
        'std_kappa': float(df['kappa'].std()),
        'mean_r2': float(df['R2'].mean()),
        'mean_da': float(df['DA'].mean()),
        'kappa_vs_gaussian': float(df['kappa_vs_gaussian'].mean())
    }

    with open(output_dir / 'kappa_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  CSV: {output_dir / 'kappa_analysis_results.csv'}")
    print(f"  JSON: {output_dir / 'kappa_analysis_summary.json'}")

if __name__ == '__main__':
    main()
