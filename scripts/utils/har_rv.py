"""
Backward-compatible wrapper for HAR-RV module.

This wrapper maintains the original import path for existing code while
delegating to the new modular implementation.

Deprecation Notice: This module redirects to scripts.thesis_v2.har_rv
Please update imports to use the new location.
"""

import warnings
warnings.warn(
    "scripts.utils.har_rv is deprecated and redirects to scripts.thesis_v2.har_rv. "
    "Please update your imports to use: from scripts.thesis_v2.har_rv import ...",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new modular location
from scripts.thesis_v2.har_rv import (
    # Models
    HARRVConfig,
    HARRV,
    create_har_rv_model,
    create_har_rv_differenced,
    create_comprehensive_har_rv_model,
    evaluate_comprehensive_har_rv,

    # Diagnostics
    calculate_statistical_diagnostics,

    # Baseline runners
    run_phase1_baseline_analysis,
    run_phase1_baseline_with_diagnostics,
    run_random_forest_baseline,
    run_xgboost_baseline,
    run_har_rv_volatility_focused,
    run_har_rv_comprehensive,
    run_comprehensive_baseline_comparison,

    # Visualization
    create_baseline_comparison_table,
    create_statistical_diagnostics_summary,
)

# Re-export for backward compatibility
__all__ = [
    'HARRVConfig',
    'HARRV',
    'create_har_rv_model',
    'create_har_rv_differenced',
    'create_comprehensive_har_rv_model',
    'evaluate_comprehensive_har_rv',
    'calculate_statistical_diagnostics',
    'run_phase1_baseline_analysis',
    'run_phase1_baseline_with_diagnostics',
    'run_random_forest_baseline',
    'run_xgboost_baseline',
    'run_har_rv_volatility_focused',
    'run_har_rv_comprehensive',
    'run_comprehensive_baseline_comparison',
    'create_baseline_comparison_table',
    'create_statistical_diagnostics_summary',
]
