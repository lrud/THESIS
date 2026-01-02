"""
HAR-RV Module - DEPRECATED

This module has been deprecated. The HAR-RV functionality has been moved to
deprecated/thesis_v2_har_rv/ as the analysis is now conducted through Jupyter
notebooks in the notebooks/ directory.

For HAR-RV analysis, please use:
  notebooks/benchmarking.ipynb - Comprehensive baseline analysis including HAR-RV

Migration Guide:
  - Old: from scripts.utils.har_rv import create_har_rv_model
  - New: Use notebooks/benchmarking.ipynb or implement HAR-RV directly using sklearn

Archived: 2025-12-30
Reason: Jupyter-based analysis provides better reproducibility and documentation.
"""

import warnings


class DeprecationWrapper:
    """Wrapper that raises deprecation warning for all attribute access."""

    def __init__(self, module_name):
        self.module_name = module_name

    def __getattr__(self, name):
        warnings.warn(
            f"HAR-RV module is deprecated. The functionality has been moved to "
            f"deprecated/thesis_v2_har_rv/. For analysis, use notebooks/benchmarking.ipynb",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(
            f"Cannot import '{name}' from deprecated HAR-RV module. "
            f"Use notebooks/benchmarking.ipynb for HAR-RV analysis."
        )


# Create a fake module that raises warnings on any access
import sys
sys.modules[__name__] = DeprecationWrapper(__name__)
