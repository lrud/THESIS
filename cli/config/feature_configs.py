"""Feature set configurations for unified LSTM training."""

FEATURE_SETS = {
    "no_lags": {"input_size": 4, "use_sample_weighting": False},
    "no_lags_jumps": {"input_size": 8, "use_sample_weighting": True},
    "lags": {"input_size": 7, "use_sample_weighting": False},
    "lags_jumps": {"input_size": 11, "use_sample_weighting": True},
    "market": {"input_size": 4, "use_sample_weighting": False},
    "market_jumps": {"input_size": 8, "use_sample_weighting": True},
    "market_lags": {"input_size": 7, "use_sample_weighting": False},
    "jump_aware": {"input_size": 11, "use_sample_weighting": True},
}


def get_feature_config(model_type: str) -> dict:
    """Get feature configuration for model type."""
    return FEATURE_SETS[model_type]
