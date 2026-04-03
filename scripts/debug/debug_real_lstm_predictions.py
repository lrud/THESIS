#!/usr/bin/env python3
"""
Debug with Real LSTM Predictions

Test jump signal generation using actual LSTM predictions vs current DVOL.

Date: November 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_real_lstm_predictions():
    """
    Test jump signal generation with real LSTM predictions
    """
    print("🔍 DEBUGGING WITH REAL LSTM PREDICTIONS")
    print("=" * 80)
    print("Testing jump signals using actual LSTM predictions vs current DVOL")
    print("=" * 80)

    # 1. Load jump-enhanced data
    jump_data_path = "data/processed/bitcoin_lstm_features_with_jumps.csv"

    if not os.path.exists(jump_data_path):
        print(f"❌ Jump-enhanced data not found: {jump_data_path}")
        return False

    print(f"✅ Jump-enhanced data found: {jump_data_path}")

    # 2. Load and analyze data
    try:
        df = pd.read_csv(jump_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"\n📊 Dataset Analysis:")
        print(f"  • Total records: {len(df):,}")

        # Check if LSTM predictions exist
        if 'predicted_dvol' not in df.columns:
            print("❌ No 'predicted_dvol' column found!")
            return False

        # Calculate prediction differences
        df['vol_change'] = df['predicted_dvol'] - df['dvol']
        df['vol_change_pct'] = (df['vol_change'] / df['dvol']) * 100

        print(f"  • LSTM predictions column: ✅ found")
        print(f"  • Vol change analysis:")
        print(f"    - Mean change: {df['vol_change'].mean():.3f}")
        print(f"    - Std change: {df['vol_change'].std():.3f}")
        print(f"    - Max change: {df['vol_change'].max():.3f}")
        print(f"    - Min change: {df['vol_change'].min():.3f}")
        print(f"    - Mean % change: {df['vol_change_pct'].mean():.2f}%")
        print(f"    - Max % change: {df['vol_change_pct'].max():.2f}%")
        print(f"    - Min % change: {df['vol_change_pct'].min():.2f}%")

        # Find periods with significant prediction differences
        significant_changes = df[abs(df['vol_change_pct']) >= 2.0]  # 2%+ DVOL change
        print(f"  • Significant prediction changes (2%+): {len(significant_changes):,}")

        # Find jump events with significant prediction changes
        jump_events = df[df['jump_indicator'] > 0]
        jump_with_predictions = jump_events[abs(jump_events['vol_change_pct']) >= 1.0]
        print(f"  • Jump events with significant predictions: {len(jump_with_predictions):,}")

        if len(jump_with_predictions) == 0:
            print("⚠️ No jump events with significant prediction differences found")
            # Use most extreme prediction changes instead
            extreme_changes = df[abs(df['vol_change_pct']) >= 5.0].nlargest(5, 'vol_change_pct')
            print(f"  • Using {len(extreme_changes)} extreme prediction changes instead")
            test_events = extreme_changes
        else:
            test_events = jump_with_predictions.head(5)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

    # 3. Test clean jump signal generator
    try:
        from scripts.backtesting.clean_jump_signal_generator import CleanJumpSignalGenerator

        # Use realistic thresholds
        config = {
            'jump_trading': {
                'jump_probability_threshold': 0.0,      # NO threshold
                'min_jump_magnitude': 0.0,             # NO threshold
                'hours_before_jump': 3,
                'hours_after_jump': 24,
                'max_position_size': 1.0,
                'volatility_expansion_threshold': 1.0,  # 1% DVOL threshold
                'use_lstm_predictions': True
            }
        }
        generator = CleanJumpSignalGenerator(config)

        print(f"\n🔧 Jump Signal Generator Configuration:")
        print(f"  ✅ Volatility expansion threshold: {config['jump_trading']['volatility_expansion_threshold']}%")
        print(f"  ✅ Max position size: {config['jump_trading']['max_position_size']}")

    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return False

    # 4. Test signal generation with real predictions
    try:
        print(f"\n🧪 Testing Signal Generation with Real LSTM Predictions:")

        trades_generated = 0
        total_tested = 0

        for i, (_, row) in enumerate(test_events.iterrows()):
            timestamp = row['timestamp']
            vol_change = row['vol_change']
            vol_change_pct = row['vol_change_pct']
            jump_indicator = row.get('jump_indicator', 0)
            jump_magnitude = row.get('jump_magnitude', 0)

            print(f"\n  📍 Test Event {i+1}: {timestamp}")
            print(f"     Current DVOL: {row['dvol']:.2f}")
            print(f"     Predicted DVOL: {row['predicted_dvol']:.2f}")
            print(f"     Vol Change: {vol_change:+.3f} ({vol_change_pct:+.2f}%)")
            print(f"     Jump Indicator: {jump_indicator}")
            if jump_indicator > 0:
                print(f"     Jump Magnitude: {jump_magnitude:.4f}")

            # Prepare current data
            current_data = {
                'dvol': row['dvol'],
                'predicted_dvol': row['predicted_dvol'],
                'jump_indicator': jump_indicator,
                'jump_magnitude': jump_magnitude,
                'jump_any': row.get('jump_any', 0),
                'jump_cluster_7d': row.get('jump_cluster_7d', 0),
                'hours_since_jump': row.get('hours_since_jump', 999)
            }

            # Prepare historical data
            historical_idx = df[df['timestamp'] < timestamp].index
            if len(historical_idx) > 0:
                historical_data = {
                    'price': df.loc[historical_idx, 'dvol'],
                    'dvol': df.loc[historical_idx, 'dvol']
                }
            else:
                continue

            # Generate jump signals
            signals = generator.generate_signals(timestamp, current_data, historical_data)
            combined = generator.combine_signals(signals)

            total_tested += 1

            print(f"     Signal: {combined['position_size']:+.3f}")
            print(f"     Action: {combined['action']}")
            print(f"     Reason: {combined['reason']}")
            print(f"     Jump type: {combined['jump_type']}")
            print(f"     Confidence: {combined['confidence']:.3f}")

            # Check for actual trades
            if abs(combined['position_size']) > 0.01:
                trades_generated += 1
                print(f"     🎯 TRADE GENERATED! Position: {combined['position_size']:+.3f}")
            else:
                print(f"     ⚠️ No trade - Position: {combined['position_size']:.3f}")

        print(f"\n📊 Signal Generation Summary:")
        print(f"  • Total events tested: {total_tested}")
        print(f"  • Trades generated: {trades_generated}")
        print(f"  • Trade generation rate: {(trades_generated/total_tested*100):.1f}%")

        if trades_generated > 0:
            print(f"  🎯 SUCCESS: Real LSTM predictions are generating jump trades!")
            return True
        else:
            print(f"  ❌ ISSUE: Even with real LSTM predictions, no trades generated")
            print(f"  🔧 May need to adjust signal generation logic or thresholds")
            return False

    except Exception as e:
        print(f"❌ Error testing signal generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_real_lstm_predictions()

    if success:
        print(f"\n🎯 CONCLUSION:")
        print("✅ Clean jump signal generation works with real LSTM predictions")
        print("✅ LSTM prediction differences are driving trade generation")
        print("✅ Jump-focused strategy is ready for backtesting")
    else:
        print(f"\n⚠️ CONCLUSION:")
        print("❌ Signal generation still needs adjustment")
        print("🔧 Consider more aggressive parameters or logic changes")
        print("📊 LSTM predictions may be too similar to current DVOL")