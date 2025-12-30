"""
Refactored HAR-RV Analysis Module.

This modular package provides HAR-RV model implementation, statistical diagnostics,
baseline model runners, and visualization functions for Bitcoin DVOL volatility forecasting.

Usage:
    from scripts.thesis_v2.har_rv import (
        HARRV, create_har_rv_model,
        run_comprehensive_baseline_comparison,
        run_phase1_baseline_with_diagnostics
    )

    # Run comprehensive baseline comparison
    results = run_comprehensive_baseline_comparison(
        'data/processed/bitcoin_lstm_features.csv',
        data_version='v1.1'
    )

Main Components:
    - models: HARRVConfig, HARRV class, factory functions
    - diagnostics: Statistical testing beyond R²
    - baseline: OLS, HAR-RV, Random Forest, XGBoost runners
    - visualization: Plotting and table generation
"""

from .models import (
    HARRVConfig,
    HARRV,
    create_har_rv_model,
    create_har_rv_differenced,
    create_comprehensive_har_rv_model,
    evaluate_comprehensive_har_rv
)

from .diagnostics import calculate_statistical_diagnostics

from .baseline import (
    run_phase1_baseline_analysis,
    run_phase1_baseline_with_diagnostics,
    run_random_forest_baseline,
    run_xgboost_baseline,
    run_har_rv_volatility_focused,
    run_har_rv_comprehensive,
    run_comprehensive_baseline_comparison
)

from .visualization import (
    create_baseline_comparison_table,
    create_statistical_diagnostics_summary
)

__all__ = [
    # Models
    'HARRVConfig',
    'HARRV',
    'create_har_rv_model',
    'create_har_rv_differenced',
    'create_comprehensive_har_rv_model',
    'evaluate_comprehensive_har_rv',

    # Diagnostics
    'calculate_statistical_diagnostics',

    # Baseline runners
    'run_phase1_baseline_analysis',
    'run_phase1_baseline_with_diagnostics',
    'run_random_forest_baseline',
    'run_xgboost_baseline',
    'run_har_rv_volatility_focused',
    'run_har_rv_comprehensive',
    'run_comprehensive_baseline_comparison',

    # Visualization
    'create_baseline_comparison_table',
    'create_statistical_diagnostics_summary',
]

__version__ = '2.0.0'
