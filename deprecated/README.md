# Deprecated Code Archive

This directory contains legacy code that has been superseded by newer implementations.

---

## Deprecated 2026-03-23: src/core/ Module

The entire `src/core/` directory was removed as it contained duplicate code that was never actively used.

| Deprecated File | Replacement | Notes |
|-----------------|-------------|-------|
| `modeling/src_core_model.py` | `scripts/modeling/model.py` | Identical copy - was never imported |
| `modeling/src_core_evaluator.py` | `scripts/utils/metrics.py` | Replaced by consolidated metrics module |
| `modeling/src_core_README.md` | N/A | Documentation for unused module |

**Reason for deprecation:** The `src/core/` path was added to `sys.path` in trainers but was shadowed by `scripts/modeling/` which was added first. No code ever imported from `src.core.*`.

---

## Deprecated 2025-12-29: thesis_v2_har_rv/

Refactored HAR-RV analysis module with modular structure. Contains:
- `models.py` - HARRVConfig, HARRV class, factory functions
- `diagnostics.py` - Statistical testing beyond R²
- `baseline.py` - OLS, HAR-RV, Random Forest, XGBoost runners
- `visualization.py` - Plotting and table generation
- `cli.py` - Command-line interface

**Replacement:** `scripts/utils/har_rv.py` (consolidated implementation)

---

## Deprecated 2025-11-07: Original Training Scripts

Moved from `scripts/modeling/` to `deprecated/modeling/`.

| Deprecated File | Replacement |
|-----------------|-------------|
| `modeling/main.py` | `cli/bin/train.py` |
| `modeling/main_jump_aware.py` | `cli/scripts/trainers/jump_aware_trainer.py` |
| `modeling/main_rolling.py` | `cli/scripts/trainers/rolling_trainer.py` |
| `modeling/main_differenced.py` | `cli/scripts/trainers/differenced_trainer.py` |
| `modeling/data_loader.py` | `scripts/modeling/data_loader_*.py` |
| `modeling/trainer.py` | `cli/scripts/trainers/` |
| `modeling/evaluator_har.py` | `scripts/utils/metrics.py` |
| `modeling/har_rv_model.py` | `scripts/utils/har_rv.py` |
| `modeling/har_rv_differenced.py` | `scripts/utils/har_rv.py` |
| `har_rv_v1.0.py` | `scripts/utils/har_rv.py` |

---

## Current Active System

- `cli/bin/train.py` - Main training interface
- `cli/config/config.py` - Configuration management
- `cli/scripts/trainers/` - Modular trainer implementations
- `scripts/modeling/` - Model definitions and data loaders
- `scripts/utils/` - Consolidated utilities (metrics, HAR-RV)


