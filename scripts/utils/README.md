# Shared Utilities Module

This directory contains consolidated utilities used across different model types and training approaches to eliminate code duplication and improve maintainability.

## Modules

### `metrics.py`
Unified evaluation metrics calculation for all model types (LSTM, HAR-RV, baselines).

**Functions:**
- `calculate_metrics(y_true, y_pred)` - Calculate RMSE, MAE, MAPE, R², Directional Accuracy
- `print_metrics_comparison(metrics_dict, title)` - Format model comparison tables
- `calculate_model_performance_summary(y_true, y_pred, model_name)` - Complete performance summary

**Replaces:**
- `scripts/modeling/evaluator.py` (duplicate metrics function)
- `scripts/benchmarking/utils/evaluator_har.py` (duplicate metrics function)

### `har_rv.py`
Consolidated HAR-RV model implementation supporting both standard and differenced targets.

**Classes:**
- `HARRV` - Unified HAR-RV model
- `HARRVConfig` - Configuration class

**Functions:**
- `create_har_rv_model(...)` - Create HAR-RV model (standard)
- `create_har_rv_differenced(...)` - Create HAR-RV model (differenced)

**Replaces:**
- `scripts/benchmarking/models/har_rv_model.py`
- `scripts/benchmarking/models/har_rv_differenced.py`

**Key Features:**
- Single class supporting both standard and differenced targets via `difference_target` parameter
- Backward compatibility functions maintained
- Unified coefficient and R² calculation methods

## Usage Examples

### Metrics Calculation
```python
from scripts.utils import calculate_metrics, print_metrics_comparison

# Calculate metrics for a single model
metrics = calculate_metrics(y_true, y_pred)

# Compare multiple models
all_metrics = {
    'LSTM': calculate_metrics(y_true, lstm_pred),
    'HAR-RV': calculate_metrics(y_true, har_pred),
    'Naive': calculate_metrics(y_true, naive_pred)
}
print_metrics_comparison(all_metrics, "Model Comparison")
```

### HAR-RV Models
```python
from scripts.utils import create_har_rv_model, create_har_rv_differenced

# Standard HAR-RV
har_model = create_har_rv_model(daily_lag=1, weekly_lag=5, monthly_lag=22)
har_model.fit(rv_series)
predictions = har_model.predict(rv_series)

# Differenced HAR-RV
har_diff = create_har_rv_differenced(daily_lag=1, weekly_lag=5, monthly_lag=22)
har_diff.fit(rv_series)
diff_pred, recon_pred = har_diff.predict(rv_series, return_reconstruction=True)
```

## Benefits

1. **Single Source of Truth**: All metrics and HAR-RV functionality in one place
2. **Reduced Duplication**: ~30% reduction in duplicate code
3. **Improved Maintainability**: Changes only need to be made in one location
4. **Backward Compatibility**: Existing code continues to work with minimal changes
5. **Type Safety**: Better type hints and documentation
6. **Testing**: Easier to test and validate consolidated functions

## Migration Notes

### For Existing Code Using Old Metrics Functions
```python
# Old way (still works but deprecated)
from scripts.modeling.evaluator import calculate_metrics

# New way (recommended)
from scripts.utils import calculate_metrics
```

### For Existing Code Using Old HAR-RV Models
```python
# Old way (still works but deprecated)
from scripts.benchmarking.models.har_rv_model import create_har_rv_model

# New way (recommended)
from scripts.utils import create_har_rv_model
```

## Future Extensions

Potential additional utilities to consolidate:
- Data loading utilities (from multiple LSTM data loaders)
- Visualization utilities (scatter plots, time series plots)
- Statistical validation utilities
- Configuration management utilities