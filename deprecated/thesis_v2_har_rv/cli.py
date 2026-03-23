"""
Command-line interface for HAR-RV Analysis Module.

Usage:
    python -m scripts.thesis_v2.har_rv.cli --analysis comprehensive
    python -m scripts.thesis_v2.har_rv.cli --analysis diagnostics --data-version v1.1
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='HAR-RV Model Analysis - Refactored Modular Version'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/bitcoin_lstm_features.csv',
        help='Path to features dataset'
    )
    parser.add_argument(
        '--data-version',
        type=str,
        default='v1.1',
        help='Data version identifier for output files'
    )
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['baseline', 'decay', 'comprehensive', 'diagnostics', 'all'],
        default='all',
        help='Which analysis to run: baseline (OLS only), decay (HAR-RV vs naive), comprehensive (all models), diagnostics (statistical tests), or all'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/thesis_v2',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick run with fewer estimators for RF/XGBoost (for testing)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"HAR-RV ANALYSIS (Modular v2.0) - {args.data_version.upper()}")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Analysis: {args.analysis}")
    print()

    # Import analysis functions
    from . import (
        run_phase1_baseline_analysis,
        run_comprehensive_baseline_comparison,
        run_phase1_baseline_with_diagnostics
    )

    # Adjust hyperparameters for quick run
    n_est = 20 if args.quick else 100
    max_depth = 5 if args.quick else 10

    # Run selected analyses
    if args.analysis in ['baseline', 'all']:
        run_phase1_baseline_analysis(args.data_path, args.data_version, args.output_dir)
        print()

    if args.analysis in ['comprehensive', 'all']:
        run_comprehensive_baseline_comparison(args.data_path, args.data_version, args.output_dir)
        print()

    if args.analysis in ['diagnostics', 'all']:
        run_phase1_baseline_with_diagnostics(args.data_path, args.data_version, args.output_dir)
        print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
