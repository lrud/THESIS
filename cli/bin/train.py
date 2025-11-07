#!/usr/bin/env python3
"""
Simple Training CLI
==================

Minimal command-line interface for training LSTM models with custom parameters.

Usage:
    python cli/bin/train.py jump_aware --epochs 100 --lr 0.001
    python cli/bin/train.py rolling --batch-size 64 --hidden-size 256 --use-multi-gpu
    python cli/bin/train.py differenced --epochs 150 --dropout 0.4
    python cli/bin/train.py jump_aware --config configs/experiment.json

Author: Claude Code Assistant
"""

import argparse
import sys
import json
from pathlib import Path

# Add paths for imports
sys.path.append('cli/config')
sys.path.append('cli/scripts')
from config import get_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM models with custom parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train jump-aware model with default parameters
  python cli/bin/train.py jump_aware

  # Train with custom parameters and multi-GPU support
  python cli/bin/train.py rolling --epochs 100 --hidden-size 256 --lr 0.0005 --use-multi-gpu

  # Load configuration from file
  python cli/bin/train.py jump_aware --config configs/my_experiment.json
        """
    )

    # Model type (required)
    parser.add_argument('model_type',
                       choices=['jump_aware', 'rolling', 'differenced'],
                       help='Model type to train')

    # Configuration file option
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')

    # Training parameters
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', '--learning-rate', type=float,
                       help='Learning rate')
    parser.add_argument('--hidden-size', type=int,
                       help='LSTM hidden layer size')
    parser.add_argument('--num-layers', type=int,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int,
                       help='Early stopping patience')
    parser.add_argument('--sequence-length', type=int,
                       help='Input sequence length')

    # Model-specific parameters
    parser.add_argument('--window-size', type=int,
                       help='Rolling window size (for jump_aware, rolling)')
    parser.add_argument('--rolling-window', type=int,
                       help='Rolling window for rolling normalization')
    parser.add_argument('--forecast-horizon', type=int,
                       help='Forecast horizon (for differenced)')
    parser.add_argument('--weight-jump-periods', type=float,
                       help='Weight for jump periods (for jump_aware)')

    # Hardware options
    parser.add_argument('--use-multi-gpu', action='store_true',
                       help='Enable multi-GPU training with DataParallel (AMD ROCm 7 supported)')

    # Output options
    parser.add_argument('--save-prefix', type=str, default='cli',
                       help='Prefix for saved model files')
    parser.add_argument('--results-dir', type=str, default='results/cli_training',
                       help='Directory to save results')

    return parser.parse_args()


def load_config_from_file(config_path: str):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def build_config_from_args(args):
    """Build configuration from command line arguments."""
    # Start with config file if provided
    if args.config:
        config_overrides = load_config_from_file(args.config)
    else:
        config_overrides = {}

    # Override with command line arguments
    arg_mapping = {
        'epochs': 'epochs',
        'batch_size': 'batch_size',
        'lr': 'learning_rate',
        'learning_rate': 'learning_rate',
        'hidden_size': 'hidden_size',
        'num_layers': 'num_layers',
        'dropout': 'dropout',
        'patience': 'patience',
        'sequence_length': 'sequence_length',
        'window_size': 'window_size',
        'rolling_window': 'rolling_window',
        'forecast_horizon': 'forecast_horizon',
        'weight_jump_periods': 'weight_jump_periods',
        'use_multi_gpu': 'use_multi_gpu'
    }

    for arg_name, config_key in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config_overrides[config_key] = arg_value

    return config_overrides


def main():
    """Main training function."""
    args = parse_arguments()

    # Build configuration
    config_overrides = build_config_from_args(args)
    config = get_config(args.model_type, **config_overrides)

    # Import and run appropriate trainer
    if args.model_type == 'jump_aware':
        from trainers.jump_aware_trainer import train_jump_aware
        train_jump_aware(config.to_dict(), args.save_prefix, args.results_dir)
    elif args.model_type == 'rolling':
        from trainers.rolling_trainer import train_rolling
        train_rolling(config.to_dict(), args.save_prefix, args.results_dir)
    elif args.model_type == 'differenced':
        from trainers.differenced_trainer import train_differenced
        train_differenced(config.to_dict(), args.save_prefix, args.results_dir)


if __name__ == '__main__':
    main()