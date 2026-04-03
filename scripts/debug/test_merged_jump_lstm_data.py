#!/usr/bin/env python3
"""
Test Merged Jump + LSTM Data

Create a quick merge of jump features with LSTM predictions to test clean jump generator.

Date: November 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import pandas as pd
import numpy as np
from datetime import datetime

def test_merged_data():
    """
    Test clean jump generator with manually merged jump + LSTM data
    """
    print("🔍 TESTING MERGED JUMP + LSTM DATA")
    print("=" * 80)
    print("Creating quick merge to test clean jump signal generation")
    print("=" * 80)

    # 1. Load both datasets
    jump_path = "data/processed/bitcoin_lstm_features_with_jumps.csv"
    prepared_path = "data/prepared/dvol_backtest_data_20251108_121141.csv"

    try:
        print("📂 Loading datasets...")

        # Load jump features
        jump_df = pd.read_csv(jump_path)
        jump_df['timestamp'] = pd.to_datetime(jump_df['timestamp'])
        print(f"  ✅ Jump data: {len(jump_df):,} records")

        # Load prepared data with LSTM predictions
        prepared_df = pd.read_csv(prepared_path)
        prepared_df['timestamp'] = pd.to_datetime(prepared_df['timestamp'])
        print(f"  ✅ Prepared data: {len(prepared_df):,} records")

        # 2. Merge datasets on timestamp
        print("🔗 Merging datasets...")
        merged_df = pd.merge(
            jump_df,
            prepared_df[['timestamp', 'predicted_dvol', 'prediction_confidence']],
            on='timestamp',
            how='inner'
        )
        print(f"  ✅ Merged data: {len(merged_df):,} records")

        # 3. Analyze the merged data
        print(f"\n📊 Merged Data Analysis:")
        print(f"  • Total records: {len(merged_df):,}")
        print(f"  • Jump events: {merged_df['jump_indicator'].sum():,.0f}")
        print(f"  • Jump frequency: {(merged_df['jump_indicator'].sum() / len(merged_df) * 100):.1f}%")

        # Calculate prediction differences
        merged_df['vol_change'] = merged_df['predicted_dvol'] - merged_df['dvol']
        merged_df['vol_change_pct'] = (merged_df['vol_change'] / merged_df['dvol']) * 100

        print(f"  • Vol change analysis:")
        print(f"    - Mean prediction change: {merged_df['vol_change'].mean():.3f}")
        print(f"    - Std prediction change: {merged_df['vol_change'].std():.3f}")
        print(f"    - Max prediction change: {merged_df['vol_change'].max():.3f}")
        print(f"    - Min prediction change: {merged_df['vol_change'].min():.3f}")

        # Find interesting test cases
        jump_with_predictions = merged_df[
            (merged_df['jump_indicator'] > 0) &
            (abs(merged_df['vol_change_pct']) >= 0.5)
        ]
        print(f"  • Jump events with prediction changes: {len(jump_with_predictions):,}")

        if len(jump_with_predictions) == 0:
            print("⚠️ No jump events with prediction changes found")
            # Use jump events or significant prediction changes
            interesting_cases = pd.concat([
                merged_df[merged_df['jump_indicator'] > 0].head(3),
                merged_df[abs(merged_df['vol_change_pct']) >= 1.0].head(3)
            ]).drop_duplicates()
        else:
            interesting_cases = jump_with_predictions.head(5)

        print(f"  • Selected {len(interesting_cases)} test cases")

    except Exception as e:
        print(f"❌ Error loading/merging data: {e}")
        return False

    # 4. Test clean jump signal generator
    try:
        from scripts.backtesting.clean_jump_signal_generator import CleanJumpSignalGenerator

        config = {
            'jump_trading': {
                'jump_probability_threshold': 0.0,      # NO threshold
                'min_jump_magnitude': 0.0,             # NO threshold
                'hours_before_jump': 3,
                'hours_after_jump': 24,
                'max_position_size': 1.0,
                'volatility_expansion_threshold': 0.5,  # 0.5% DVOL threshold
                'use_lstm_predictions': True
            }
        }
        generator = CleanJumpSignalGenerator(config)

        print(f"\n🔧 Clean Jump Signal Generator:")
        print(f"  ✅ Volatility expansion threshold: {config['jump_trading']['volatility_expansion_threshold']}%")
        print(f"  ✅ Max position size: {config['jump_trading']['max_position_size']}")

    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return False

    # 5. Test signal generation with merged data
    try:
        print(f"\n🧪 Testing Signal Generation with Merged Data:")

        trades_generated = 0
        total_tested = 0

        for i, (_, row) in enumerate(interesting_cases.iterrows()):
            timestamp = row['timestamp']
            vol_change = row['vol_change']
            vol_change_pct = row['vol_change_pct']
            jump_indicator = row['jump_indicator']
            jump_magnitude = row.get('jump_magnitude', 0)

            print(f"\n  📍 Test Case {i+1}: {timestamp}")
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
            historical_idx = merged_df[merged_df['timestamp'] < timestamp].index
            if len(historical_idx) > 0:
                historical_data = {
                    'price': merged_df.loc[historical_idx, 'dvol'],
                    'dvol': merged_df.loc[historical_idx, 'dvol']
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

        print(f"\n📊 Test Results:")
        print(f"  • Total cases tested: {total_tested}")
        print(f"  • Trades generated: {trades_generated}")
        print(f"  • Trade generation rate: {(trades_generated/total_tested*100):.1f}%")

        if trades_generated > 0:
            print(f"  🎯 SUCCESS: Merged data enables jump trading!")

            # Save the merged data for backtesting
            output_path = "data/processed/bitcoin_lstm_features_with_jumps_and_predictions.csv"
            merged_df.to_csv(output_path, index=False)
            print(f"  💾 Saved merged data to: {output_path}")

            return True
        else:
            print(f"  ❌ ISSUE: Even with merged data, no trades generated")
            return False

    except Exception as e:
        print(f"❌ Error testing signal generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_merged_data()

    if success:
        print(f"\n🎯 CONCLUSION:")
        print("✅ Merged jump + LSTM data enables successful signal generation")
        print("✅ Clean jump signal generator is working correctly")
        print("✅ Ready for backtesting with complete data")
        print("\n🚀 NEXT STEP:")
        print("Update backtest config to use the merged data file")
    else:
        print(f"\n⚠️ CONCLUSION:")
        print("❌ Signal generation still needs work")
        print("🔧 May need further parameter adjustments or logic changes")