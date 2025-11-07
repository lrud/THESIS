# CLI Training System

A modular, academic command-line interface for training LSTM models with custom parameters.

## 🏗️ **Architecture**

```
scripts/
├── train.py                    # Main CLI entry point
├── config.py                   # Configuration management
└── trainers/                   # Modular trainers
    ├── __init__.py
    ├── jump_aware_trainer.py   # Jump-aware LSTM training
    ├── rolling_trainer.py      # Rolling window LSTM training
    └── differenced_trainer.py  # Differenced LSTM training
```

## 🚀 **Usage Examples**

### **Basic Training**
```bash
# Train jump-aware model with default parameters
python scripts/train.py jump_aware

# Train rolling model with default parameters
python scripts/train.py rolling

# Train differenced model with default parameters
python scripts/train.py differenced
```

### **Custom Parameters**
```bash
# Train jump-aware model with custom parameters
python scripts/train.py jump_aware --epochs 100 --hidden-size 256 --lr 0.0005

# Train rolling model with larger batch size
python scripts/train.py rolling --batch-size 64 --dropout 0.4

# Train differenced model with more layers
python scripts/train.py differenced --num-layers 3 --patience 20
```

### **Configuration Files**
```bash
# Create configuration file
cat > configs/experiment.json << EOF
{
  "epochs": 100,
  "hidden_size": 256,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "dropout": 0.2
}
EOF

# Train using configuration file
python scripts/train.py jump_aware --config configs/experiment.json
```

## 📊 **Available Parameters**

### **Common Parameters (All Models)**
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`, `--learning-rate`: Learning rate
- `--hidden-size`: LSTM hidden layer size
- `--num-layers`: Number of LSTM layers
- `--dropout`: Dropout rate
- `--patience`: Early stopping patience
- `--sequence-length`: Input sequence length

### **Model-Specific Parameters**

#### **Jump-Aware Model**
- `--window-size`: Rolling window size for normalization
- `--weight-jump-periods`: Weight for jump periods (default: 2.0)

#### **Rolling Model**
- `--rolling-window`: Rolling window for normalization

#### **Differenced Model**
- `--forecast-horizon`: Forecast horizon

### **Output Options**
- `--save-prefix`: Prefix for saved model files (default: 'cli')
- `--results-dir`: Directory to save results (default: 'results/cli_training')

## 🎯 **Default Configurations**

### **Jump-Aware Model**
```json
{
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.3,
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 50,
  "patience": 10,
  "sequence_length": 24,
  "window_size": 720,
  "weight_jump_periods": 2.0
}
```

### **Rolling Model**
```json
{
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.3,
  "learning_rate": 0.0001,
  "batch_size": 32,
  "epochs": 200,
  "patience": 15,
  "sequence_length": 24,
  "rolling_window": 720
}
```

### **Differenced Model**
```json
{
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.3,
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 100,
  "patience": 15,
  "sequence_length": 24,
  "forecast_horizon": 24
}
```

## 📁 **Output Files**

### **Models**
- `models/cli_jump_aware_best.pth`
- `models/cli_rolling_best.pth`
- `models/cli_differenced_best.pth`

### **Results**
- `results/cli_training/cli_jump_aware_YYYYMMDD_HHMMSS.json`
- `results/cli_training/cli_rolling_YYYYMMDD_HHMMSS.json`
- `results/cli_training/cli_differenced_YYYYMMDD_HHMMSS.json`

Each result file contains:
- Configuration used
- Training time and metrics
- Evaluation results (overall, normal periods, jump periods)
- Training history
- Model path and parameter count

## 📖 **Examples**

### **Quick Test Run**
```bash
# Quick test with fewer epochs
python scripts/train.py jump_aware --epochs 10 --save-prefix test
```

### **High-Performance Model**
```bash
# Large model for better performance
python scripts/train.py jump_aware \
  --hidden-size 256 \
  --num-layers 3 \
  --dropout 0.2 \
  --batch-size 64 \
  --epochs 100 \
  --save-prefix large_model
```

### **Research Experiment**
```bash
# Research configuration with custom settings
python scripts/train.py rolling \
  --epochs 300 \
  --lr 0.00005 \
  --patience 30 \
  --rolling-window 1440 \
  --save-prefix research_exp
```

### **Batch Training**
```bash
# Train all models with consistent parameters
python scripts/train.py jump_aware --epochs 100 --hidden-size 256 --lr 0.0005
python scripts/train.py rolling --epochs 100 --hidden-size 256 --lr 0.0005
python scripts/train.py differenced --epochs 100 --hidden-size 256 --lr 0.0005
```

## 🔧 **Modular Design**

The system is designed to be modular and academic:

1. **Separate Trainers**: Each model type has its own trainer script
2. **Configuration Management**: Centralized configuration with defaults
3. **Result Tracking**: Automatic saving of experiments with timestamps
4. **Parameter Overrides**: Easy parameter customization without code changes
5. **Reproducible**: Configuration files enable reproducible experiments

This design makes it easy to:
- Run systematic experiments
- Compare different configurations
- Reproduce results
- Extend with new model types