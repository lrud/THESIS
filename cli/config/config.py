#!/usr/bin/env python3
"""
Configuration Management for LSTM Training
=========================================

Centralized configuration definitions for different model types.
Academic and modular - each model type has its own configuration.

"""

import json
from typing import Dict, Any
from pathlib import Path


class ModelConfig:
    """Base configuration class for LSTM models."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def update(self, **kwargs):
        """Update configuration with new values."""
        self.config.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)


class JumpAwareConfig(ModelConfig):
    """Configuration for jump-aware LSTM models."""

    DEFAULT = {
        "model_type": "jump_aware",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "weight_jump_periods": 2.0,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class RollingConfig(ModelConfig):
    """Configuration for rolling window LSTM models."""

    DEFAULT = {
        "model_type": "rolling",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 200,
        "patience": 15,
        "sequence_length": 24,
        "rolling_window": 72,
        "early_stop_patience": 15,
        "use_multi_gpu": False,  # Enable multi-GPU training with DataParallel
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class DifferencedConfig(ModelConfig):
    """Configuration for differenced LSTM models."""

    DEFAULT = {
        "model_type": "differenced",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100,
        "patience": 15,
        "sequence_length": 24,
        "forecast_horizon": 24,
        "early_stop_patience": 15,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class MarketConfig(ModelConfig):
    """Configuration for market LSTM models."""

    DEFAULT = {
        "model_type": "market",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class MarketJumpsConfig(ModelConfig):
    """Configuration for market + jumps LSTM models."""

    DEFAULT = {
        "model_type": "market_jumps",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "weight_jump_periods": 2.0,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class MarketLagsConfig(ModelConfig):
    """Configuration for market + lags LSTM models."""

    DEFAULT = {
        "model_type": "market_lags",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class NoLagsConfig(ModelConfig):
    """Configuration for no-lags LSTM models (market features only)."""

    DEFAULT = {
        "model_type": "no_lags",
        "hidden_size": 512,
        "num_layers": 3,
        "dropout": 0.4,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class NoLagsJumpsConfig(ModelConfig):
    """Configuration for no-lags + jumps LSTM models."""

    DEFAULT = {
        "model_type": "no_lags_jumps",
        "hidden_size": 512,
        "num_layers": 3,
        "dropout": 0.4,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "weight_jump_periods": 2.0,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class LagsConfig(ModelConfig):
    """Configuration for lags LSTM models (market + lag features)."""

    DEFAULT = {
        "model_type": "lags",
        "hidden_size": 512,
        "num_layers": 3,
        "dropout": 0.4,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


class LagsJumpsConfig(ModelConfig):
    """Configuration for lags + jumps LSTM models (all features)."""

    DEFAULT = {
        "model_type": "lags_jumps",
        "hidden_size": 512,
        "num_layers": 3,
        "dropout": 0.4,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "sequence_length": 24,
        "window_size": 72,
        "weight_jump_periods": 2.0,
        "early_stop_patience": 10,
        "use_multi_gpu": False,
    }

    def __init__(self, **overrides):
        config = self.DEFAULT.copy()
        config.update(overrides)
        super().__init__(config)


def get_config(model_type: str, **overrides) -> ModelConfig:
    """Factory function to get configuration for a model type."""
    config_classes = {
        "jump_aware": JumpAwareConfig,
        "rolling": RollingConfig,
        "differenced": DifferencedConfig,
        "market": MarketConfig,
        "market_jumps": MarketJumpsConfig,
        "market_lags": MarketLagsConfig,
        "no_lags": NoLagsConfig,
        "no_lags_jumps": NoLagsJumpsConfig,
        "lags": LagsConfig,
        "lags_jumps": LagsJumpsConfig,
    }

    if model_type not in config_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(config_classes.keys())}"
        )

    return config_classes[model_type](**overrides)
