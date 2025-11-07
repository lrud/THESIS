# Bitcoin DVOL LSTM Forecasting - Claude AI Assistant Guide

## Project Overview

This repository implements a graduate-level research project for forecasting Bitcoin implied volatility (DVOL) using LSTM neural networks with jump-aware modeling capabilities. The project has achieved state-of-the-art performance with R² = 0.9076 using ultra-large models (5.41M parameters) and incorporates sophisticated statistical validation methodologies.

**Research Objective**: Develop and validate LSTM models for next-day Bitcoin DVOL (Deribit 30-day implied volatility index) forecasting using on-chain metrics and historical volatility patterns.

**Current Implementation Status**: Production-ready CLI training system with multi-GPU support, real-time monitoring, and comprehensive statistical validation framework.

## Core Training Commands

### Primary Model Training Interface
```bash
# Ultra-large model (current best performance: R² = 0.9076)
.venv/bin/python cli/bin/train.py jump_aware \
  --hidden-size 512 --num-layers 3 --dropout 0.4 \
  --batch-size 32 --lr 0.0001 --epochs 100 \
  --use-multi-gpu --save-prefix ultra_large

# Standard model configurations
.venv/bin/python cli/bin/train.py jump_aware --epochs 50
.venv/bin/python cli/bin/train.py rolling --epochs 50
.venv/bin/python cli/bin/train.py differenced --epochs 50
```

### Environment Configuration
```bash
# PyTorch with ROCm 7.0 for AMD GPU support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Install project dependencies
pip3 install -r requirements-pytorch.txt
```

### Real-time Training Monitoring
```bash
# Monitor training progress and convergence metrics
tail -f results/logs/current_training.log
```

## Repository Architecture

The repository has undergone comprehensive code consolidation (November 2025) to eliminate redundancy and improve maintainability.

### Current Directory Structure
```
cli/                          # Primary training interface
├── bin/train.py             # Main CLI entry point
├── config/config.py         # Configuration management system
└── scripts/trainers/        # Modular trainer implementations

scripts/utils/               # Consolidated shared utilities
├── metrics.py               # Unified evaluation metrics
├── har_rv.py                # Unified HAR-RV implementation
└── __init__.py

src/core/                    # Centralized core utilities
scripts/modeling/            # LSTM neural network components
scripts/analysis/            # Statistical validation frameworks
scripts/benchmarking/        # Benchmarking utilities
deprecated/                  # Archived superseded implementations
```

### Deprecated Components (Do Not Use)
- `scripts/modeling/main_*.py` - Replaced by CLI trainers
- `scripts/benchmarking/models/har_rv_*.py` - Replaced by `scripts/utils/har_rv.py`

## Implementation Patterns

### Standard Training Procedure
```bash
python cli/bin/train.py <model_type> [configuration_parameters]

# Available model types: jump_aware, rolling, differenced
# Key parameters: --hidden-size, --num-layers, --dropout, --lr, --epochs, --use-multi-gpu
```

### Multi-GPU Training Protocol
```bash
# Learning rate automatically scaled for DataParallel stability
python cli/bin/train.py jump_aware --use-multi-gpu --lr 0.0001
# Internal scaling: 0.0001 → 0.00005 for dual GPU configuration
```

### Configuration Management
All models utilize `cli/config/config.py` for baseline configurations with CLI parameter override support. Default configurations prevent gradient instability in large-scale models.

### Consolidated Utility Usage
```python
# Evaluation metrics (use consolidated implementation)
from scripts.utils.metrics import calculate_metrics
performance_metrics = calculate_metrics(y_true, y_pred)

# HAR-RV models (use consolidated implementation)
from scripts.utils.har_rv import create_har_rv_model, create_har_rv_differenced
har_model = create_har_rv_model(daily_lag=1, weekly_lag=5, monthly_lag=22)
```

## Key Files and Functionality

### Training Infrastructure
- `cli/bin/train.py` - Principal training interface with parameter validation
- `cli/scripts/trainers/jump_aware_trainer.py` - Highest performing model implementation
- `cli/config/config.py` - Hierarchical configuration management system

### Data Resources
- `data/processed/bitcoin_lstm_features.csv` - Primary dataset (37,949 hourly observations)
- Feature set includes: DVOL, transaction volume, active addresses, NVRV, and volatility risk premium

### Consolidated Utilities
- `scripts/utils/metrics.py` - Single-source evaluation metrics (RMSE, MAE, MAPE, R², directional accuracy)
- `scripts/utils/har_rv.py` - Unified HAR-RV implementation supporting standard and differenced targets

### Results and Artifacts
- `results/cli_training/` - Training results with comprehensive JSON metadata
- `results/logs/current_training.log` - Real-time training convergence monitoring
- `models/` - Trained model checkpoints with configuration preservation

## Performance Benchmarks

### Model Performance Comparison (Tested Results)
| Model Architecture | Parameters | R² | RMSE | MAE | Validation Status |
|-------------------|------------|----|-----|----|------------------|
| Ultra-Large Jump-Aware | 5.41M | 0.9076 | 2.57 | 1.88 | State-of-the-art |
| Large Jump-Aware | 1.36M | 0.9000 | 2.67 | 1.99 | Excellent |
| Rolling Window | 210K | 0.8804 | 3.04 | 2.39 | Validated |
| Jump-Aware | 210K | 0.8624 | 3.14 | 2.48 | Established |

### Critical Implementation Insights
- Ultra-large models (512 hidden units, 3 layers) achieve optimal performance
- Multi-GPU training requires conservative learning rate scaling for numerical stability
- Real-time logging provides training progress updates at 2-epoch intervals
- Early stopping typically converges at approximately 25 epochs for large architectures

## Technical Implementation Details

### LSTM Model Architecture
- Input: Sequential 24-hour windows of engineered features
- Architecture: Multi-layer LSTM with configurable hidden units and dropout
- Regularization: Dropout regularization and L2 penalty
- Hardware Support: AMD ROCm 7.0 with DataParallel multi-GPU training
- Training: Early stopping with patience-based convergence criteria

### Jump-Aware Modeling
- Jump detection using Lee-Mykland statistical test
- Weighted loss function (2x weight for jump periods)
- Regime-aware performance validation across normal and crisis periods
- Statistical validation against major cryptocurrency market events

### Data Engineering
- Rolling window normalization for regime adaptation
- Feature engineering from on-chain metrics and market data
- Temporal split validation (train: April 2021 - December 2023)
- Statistical validation framework with residual analysis

## Common Implementation Issues and Solutions

### Gradient Instability in Large Models
**Problem**: NaN values during training of large models with aggressive learning rates
**Solution**: Implement conservative learning rate scaling for multi-GPU training
```bash
# Recommended configuration
--lr 0.0001 --use-multi-gpu  # Automatic scaling to 0.00005

# Problematic configuration
--lr 0.001 --use-multi-gpu   # Results in gradient explosion
```

### Post-Consolidation Import Dependencies
**Problem**: Import errors following utility consolidation
**Solution**: Utilize consolidated utility modules
```python
# Deprecated imports
from scripts.modeling.evaluator import calculate_metrics
from scripts.benchmarking.models.har_rv_model import create_har_rv_model

# Current consolidated imports
from scripts.utils.metrics import calculate_metrics
from scripts.utils.har_rv import create_har_rv_model
```

### Multi-GPU Training Verification
**Problem**: Single GPU utilization despite multi-GPU flag
**Solution**: Verify ROCm 7.0 installation and GPU detection
```bash
# Hardware verification
python -c "import torch; print(torch.cuda.device_count())"
# Expected output: 2 for AMD RX 7900 XT configuration
```

### Real-time Logging System
**Problem**: Absence of training progress logs
**Solution**: Ensure log directory structure exists
```bash
mkdir -p results/logs
```

## Development Workflow

### Model Development Protocol
```bash
# Phase 1: Baseline validation
python cli/bin/train.py jump_aware --epochs 10

# Phase 2: Architectural scaling
python cli/bin/train.py jump_aware --hidden-size 256 --epochs 10

# Phase 3: Ultra-large model training
python cli/bin/train.py jump_aware --hidden-size 512 --num-layers 3 --lr 0.0001 --use-multi-gpu
```

### Model Evaluation Framework
```bash
# Results automatically archived with comprehensive metadata
# Analysis available in results/cli_training/*.json files

# Real-time convergence monitoring
tail -f results/logs/current_training.log
```

### Benchmarking Protocol
```python
# HAR-RV baseline comparisons
from scripts.utils.har_rv import create_har_rv_model
benchmark_model = create_har_rv_model(daily_lag=1, weekly_lag=5, monthly_lag=22)
benchmark_model.fit(rv_series)
benchmark_predictions = benchmark_model.predict(rv_series)
```

## Recent Developments (November 2025)

### Code Consolidation Implementation
- Achieved 50% reduction in duplicate code through systematic consolidation
- Established `scripts/utils/` centralized utility module
- Implemented backward compatibility preservation through `deprecated/` archival
- Maintained all original functionality with improved maintainability

### CLI System Enhancement
- Integrated multi-GPU support with automatic learning rate scaling
- Implemented real-time training monitoring with detailed convergence metrics
- Established hierarchical configuration management with parameter override capability
- Validated ultra-large model training capability (5.41M parameters)

### Performance Advancement
- Achieved 90.76% R² with ultra-large jump-aware model architecture
- Implemented conservative training protocols to ensure numerical stability
- Demonstrated multi-GPU efficiency for large-scale model training
- Established real-time progress tracking for training convergence monitoring

## Validation and Testing Procedures

### Model Validation Protocol
```bash
# Rapid validation (single epoch)
python cli/bin/train.py jump_aware --epochs 1 --hidden-size 64

# Comprehensive training with monitoring
python cli/bin/train.py jump_aware --hidden-size 512 --num-layers 3 --use-multi-gpu
tail -f results/logs/current_training.log
```

### Utility Validation
```python
# Consolidated metrics validation
from scripts.utils.metrics import calculate_metrics
test_metrics = calculate_metrics([1,2,3], [1.1,2.1,2.9])
assert 'RMSE' in test_metrics and 'R²' in test_metrics

# HAR-RV consolidation validation
from scripts.utils.har_rv import create_har_rv_model
test_model = create_har_rv_model()
assert hasattr(test_model, 'fit') and hasattr(test_model, 'predict')
```

## Documentation References

### Essential Documentation
- `docs/QUICK_REFERENCE.md` - Comprehensive performance summary and statistical validation
- `docs/ultra_large_model_results.md` - Latest ultra-large model experimental results
- `docs/final_code_consolidation_summary.md` - Detailed consolidation methodology and impact analysis
- `scripts/utils/README.md` - Consolidated utilities implementation guide

### Technical Specifications
- LSTM temporal sequence modeling with 24-hour input windows
- Jump-aware training with statistically weighted loss functions
- Rolling window normalization for regime adaptation in non-stationary environments
- Multi-GPU DataParallel training with automatic learning rate scaling

## Computational Requirements

### Minimum Configuration
- **CPU**: Single-threaded training feasible but computationally intensive
- **Memory**: 8GB RAM minimum for data loading and model training
- **Storage**: 5GB for model checkpoints and training artifacts

### Recommended Configuration
- **GPU**: Dual AMD Radeon RX 7900 XT (20GB VRAM per GPU)
- **Software Stack**: ROCm 7.0 with PyTorch integration
- **Memory**: 16GB+ RAM for efficient data pipeline operations
- **Storage**: 10GB+ for multiple model checkpoint preservation

## Emergency Recovery Procedures

### Consolidation Rollback Protocol
```bash
# Restore deprecated implementations if required
cp deprecated/modeling/main_*.py scripts/modeling/
cp deprecated/modeling/evaluator.py scripts/modeling/

# All original functionality preserved in deprecated/ directory structure
```

---

**Documentation Version**: November 7, 2025
**Implementation Status**: Production Ready with Ultra-Large Model Support
**Validation Results**: R² = 0.9076 (5.41M parameter jump-aware architecture)