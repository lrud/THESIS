import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
import json
from datetime import datetime

class JumpDetectionAnalysis:
    """
    Detect and analyze jump events in Bitcoin DVOL time series.

    Uses multiple methods:
    1. Lee-Mykland (2008): Statistical jump test using bipower variation
    2. Standard deviation threshold: Simple >3σ outlier detection
    3. Z-score on returns: Identifies abnormal changes

    Fat-tail events in data period (2021-2025):
    - May 2021: China mining ban
    - May-July 2022: Luna/UST collapse, 3AC
    - Nov 2022: FTX collapse
    - March 2023: Banking crisis (SVB)
    - 2024-2025: ETF approval, institutional adoption
    """

    def __init__(self, data_path='data/processed/bitcoin_lstm_features.csv', data_version='v1.0'):
        self.data_path = data_path
        self.data_version = data_version
        self.df = None
        self.jumps = None
        
    def load_data(self):
        """Load DVOL data."""
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        print("=" * 80)
        print("JUMP DETECTION ANALYSIS")
        print("=" * 80)
        print(f"\nData Version: {self.data_version}")
        print(f"Dataset: {len(self.df):,} hourly observations")
        print(f"Period: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"DVOL range: {self.df['dvol'].min():.2f} - {self.df['dvol'].max():.2f}")
        print(f"DVOL mean: {self.df['dvol'].mean():.2f}, std: {self.df['dvol'].std():.2f}")
        print()
        
    def method1_lee_mykland(self, window=22, alpha=0.999):
        """
        Lee-Mykland (2008) jump test.
        
        Uses bipower variation (robust to jumps) to detect significant price jumps.
        Standard in academic literature for high-frequency data.
        
        Args:
            window: Rolling window for variance estimation (22 hours ≈ 1 day)
            alpha: Significance level (0.999 = 99.9% confidence)
        """
        print("=" * 80)
        print("METHOD 1: LEE-MYKLAND (2008) JUMP TEST")
        print("=" * 80)
        print("Academic standard for detecting jumps in high-frequency data")
        print()
        
        returns = self.df['dvol'].pct_change()
        n = len(returns)
        
        abs_returns = np.abs(returns)
        bipower_var = (np.pi / 2) * (abs_returns.rolling(window).mean() ** 2)
        
        L = (returns ** 2) / bipower_var
        
        c = np.sqrt(2 / np.pi)
        S_n = (1 / c) * np.sqrt(window) * (L - 1)
        
        beta = np.sqrt(2 * np.log(n))
        c_alpha = beta - (np.log(np.pi) + np.log(np.log(n))) / (2 * beta)
        
        threshold = c_alpha
        
        jumps_lm = S_n > threshold
        jumps_lm = jumps_lm.fillna(False)
        
        self.df['lee_mykland_stat'] = S_n
        self.df['lee_mykland_jump'] = jumps_lm
        
        n_jumps = jumps_lm.sum()
        pct_jumps = (n_jumps / len(self.df)) * 100
        
        print(f"Parameters:")
        print(f"  Window size: {window} hours")
        print(f"  Significance: {alpha*100}%")
        print(f"  Critical value: {threshold:.4f}")
        print()
        print(f"Results:")
        print(f"  Jumps detected: {n_jumps:,} ({pct_jumps:.2f}% of observations)")
        
        if n_jumps > 0:
            jump_dates = self.df[jumps_lm]['timestamp']
            jump_sizes = returns[jumps_lm]
            print(f"  Mean jump size: {jump_sizes.mean()*100:.2f}%")
            print(f"  Max jump size: {jump_sizes.max()*100:.2f}%")
            print(f"  Min jump size: {jump_sizes.min()*100:.2f}%")
        print()
        
        return jumps_lm
    
    def method2_sigma_threshold(self, n_sigma=3):
        """
        Simple threshold method: DVOL > mean + n*std.
        
        Easier to interpret but less statistically rigorous.
        Good for identifying extreme outliers.
        """
        print("=" * 80)
        print(f"METHOD 2: {n_sigma}-SIGMA THRESHOLD")
        print("=" * 80)
        print("Simple outlier detection: DVOL exceeds mean + 3*std")
        print()
        
        mean_dvol = self.df['dvol'].mean()
        std_dvol = self.df['dvol'].std()
        threshold = mean_dvol + n_sigma * std_dvol
        
        jumps_sigma = self.df['dvol'] > threshold
        
        self.df['sigma_threshold'] = threshold
        self.df['sigma_jump'] = jumps_sigma
        
        n_jumps = jumps_sigma.sum()
        pct_jumps = (n_jumps / len(self.df)) * 100
        
        print(f"Parameters:")
        print(f"  Mean DVOL: {mean_dvol:.2f}")
        print(f"  Std DVOL: {std_dvol:.2f}")
        print(f"  Threshold: {threshold:.2f}")
        print()
        print(f"Results:")
        print(f"  Outliers detected: {n_jumps:,} ({pct_jumps:.2f}% of observations)")
        
        if n_jumps > 0:
            outlier_values = self.df[jumps_sigma]['dvol']
            print(f"  Mean outlier DVOL: {outlier_values.mean():.2f}")
            print(f"  Max outlier DVOL: {outlier_values.max():.2f}")
        print()
        
        return jumps_sigma
    
    def method3_return_zscore(self, threshold=3.5):
        """
        Z-score on returns: Identifies abnormal changes.
        
        Detects sudden spikes/drops rather than absolute levels.
        Complementary to sigma threshold.
        """
        print("=" * 80)
        print("METHOD 3: RETURN Z-SCORE")
        print("=" * 80)
        print("Detects abnormal DVOL changes (not levels)")
        print()
        
        returns = self.df['dvol'].diff()
        mean_return = returns.mean()
        std_return = returns.std()
        
        z_scores = (returns - mean_return) / std_return
        jumps_zscore = np.abs(z_scores) > threshold
        jumps_zscore = jumps_zscore.fillna(False)
        
        self.df['return_zscore'] = z_scores
        self.df['zscore_jump'] = jumps_zscore
        
        n_jumps = jumps_zscore.sum()
        pct_jumps = (n_jumps / len(self.df)) * 100
        
        print(f"Parameters:")
        print(f"  Mean return: {mean_return:.4f}")
        print(f"  Std return: {std_return:.4f}")
        print(f"  Z-score threshold: {threshold}")
        print()
        print(f"Results:")
        print(f"  Abnormal changes: {n_jumps:,} ({pct_jumps:.2f}% of observations)")
        
        if n_jumps > 0:
            jump_returns = returns[jumps_zscore]
            print(f"  Mean jump magnitude: {jump_returns.abs().mean():.2f}")
            print(f"  Max positive jump: {jump_returns.max():.2f}")
            print(f"  Max negative jump: {jump_returns.min():.2f}")
        print()
        
        return jumps_zscore
    
    def create_composite_jump_indicator(self):
        """
        Create composite jump indicator using all methods.
        
        Jump = TRUE if any method detects it (union)
        Conservative: Flags potential jumps for model to handle
        """
        print("=" * 80)
        print("COMPOSITE JUMP INDICATOR")
        print("=" * 80)
        print("Combined: Jump if ANY method detects it")
        print()
        
        self.df['jump_any'] = (
            self.df['lee_mykland_jump'] | 
            self.df['sigma_jump'] | 
            self.df['zscore_jump']
        )
        
        self.df['jump_all'] = (
            self.df['lee_mykland_jump'] & 
            self.df['sigma_jump'] & 
            self.df['zscore_jump']
        )
        
        n_any = self.df['jump_any'].sum()
        n_all = self.df['jump_all'].sum()
        
        print(f"Union (any method): {n_any:,} jumps ({n_any/len(self.df)*100:.2f}%)")
        print(f"Intersection (all methods): {n_all:,} jumps ({n_all/len(self.df)*100:.2f}%)")
        print()
        
        print("Method agreement:")
        print(f"  Lee-Mykland only: {self.df['lee_mykland_jump'].sum():,}")
        print(f"  Sigma only: {self.df['sigma_jump'].sum():,}")
        print(f"  Z-score only: {self.df['zscore_jump'].sum():,}")
        print()
        
    def identify_major_events(self):
        """
        Link detected jumps to known crypto events.
        Validates that jump detection captures real-world crises.
        """
        print("=" * 80)
        print("MAJOR EVENTS VALIDATION")
        print("=" * 80)
        print("Checking if detected jumps align with known crypto crises")
        print()
        
        known_events = [
            ('2021-05-19', 'China mining ban'),
            ('2022-05-09', 'Luna/UST collapse'),
            ('2022-06-13', '3AC liquidity crisis'),
            ('2022-11-08', 'FTX collapse'),
            ('2023-03-10', 'SVB banking crisis'),
            ('2024-01-10', 'Bitcoin ETF approval'),
        ]
        
        for date_str, event_name in known_events:
            try:
                event_date = pd.to_datetime(date_str)
                
                window_start = event_date - pd.Timedelta(days=3)
                window_end = event_date + pd.Timedelta(days=3)
                
                window_df = self.df[
                    (self.df['timestamp'] >= window_start) & 
                    (self.df['timestamp'] <= window_end)
                ]
                
                if len(window_df) > 0:
                    jumps_in_window = window_df['jump_any'].sum()
                    max_dvol = window_df['dvol'].max()
                    
                    if jumps_in_window > 0:
                        print(f"✓ {date_str} - {event_name}")
                        print(f"  {jumps_in_window} jumps detected in ±3 day window")
                        print(f"  Max DVOL: {max_dvol:.2f}")
                    else:
                        print(f"✗ {date_str} - {event_name}")
                        print(f"  No jumps detected (Max DVOL: {max_dvol:.2f})")
                else:
                    print(f"? {date_str} - {event_name}")
                    print(f"  No data available for this period")
                print()
            except Exception as e:
                print(f"! {date_str} - {event_name}: Error - {e}")
                print()
    
    def create_jump_features(self):
        """
        Engineer features for LSTM model:
        1. jump_indicator: Binary flag (0/1)
        2. jump_magnitude: Size of jump
        3. days_since_jump: Time since last jump
        4. jump_cluster: Number of jumps in past 7 days
        """
        print("=" * 80)
        print("FEATURE ENGINEERING FOR LSTM")
        print("=" * 80)
        print("Creating jump-related features for model training")
        print()
        
        self.df['jump_indicator'] = self.df['jump_any'].astype(int)
        
        returns = self.df['dvol'].pct_change()
        self.df['jump_magnitude'] = np.where(
            self.df['jump_any'],
            returns.abs(),
            0
        )
        
        hours_since_jump = []
        last_jump_idx = -999
        for i in range(len(self.df)):
            if self.df['jump_any'].iloc[i]:
                last_jump_idx = i
            hours_since_jump.append(i - last_jump_idx)
        
        self.df['hours_since_jump'] = hours_since_jump
        self.df['days_since_jump'] = self.df['hours_since_jump'] / 24.0
        
        self.df['jump_cluster_7d'] = self.df['jump_indicator'].rolling(
            window=24*7, min_periods=1
        ).sum()
        
        print("Features created:")
        print("  1. jump_indicator: Binary (0/1)")
        print("  2. jump_magnitude: Absolute return size during jumps")
        print("  3. days_since_jump: Days since last jump event")
        print("  4. jump_cluster_7d: Count of jumps in past 7 days")
        print()
        
        print("Feature statistics:")
        print(f"  Mean jump magnitude: {self.df[self.df['jump_any']]['jump_magnitude'].mean():.4f}")
        print(f"  Mean days since jump: {self.df['days_since_jump'].mean():.2f}")
        print(f"  Max jump cluster: {self.df['jump_cluster_7d'].max():.0f} jumps/week")
        print()
    
    def analyze_jump_impact_on_forecast(self):
        """
        Analyze how jumps affect forecast difficulty.
        Shows why jump-aware modeling matters.
        """
        print("=" * 80)
        print("JUMP IMPACT ON FORECASTING")
        print("=" * 80)
        print("Comparing jump vs non-jump periods")
        print()
        
        jump_periods = self.df[self.df['jump_any']]
        normal_periods = self.df[~self.df['jump_any']]
        
        print("DVOL Statistics:")
        print(f"  Normal periods: mean={normal_periods['dvol'].mean():.2f}, std={normal_periods['dvol'].std():.2f}")
        print(f"  Jump periods:   mean={jump_periods['dvol'].mean():.2f}, std={jump_periods['dvol'].std():.2f}")
        print()
        
        normal_volatility = normal_periods['dvol'].diff().std()
        jump_volatility = jump_periods['dvol'].diff().std()
        
        print("Return volatility (change in DVOL):")
        print(f"  Normal periods: {normal_volatility:.4f}")
        print(f"  Jump periods:   {jump_volatility:.4f}")
        print(f"  Ratio:          {jump_volatility/normal_volatility:.2f}x more volatile")
        print()
        
        print("Implication for LSTM:")
        print("  Jump periods are inherently harder to predict")
        print("  Should weight these periods differently in loss function")
        print("  Or: Report separate metrics for normal vs. jump periods")
        print()
    
    def visualize_jumps(self):
        """Create visualizations of detected jumps."""
        print("=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        axes[0].plot(self.df['timestamp'], self.df['dvol'], 
                     linewidth=0.8, alpha=0.7, label='DVOL')
        
        jump_points = self.df[self.df['jump_any']]
        axes[0].scatter(jump_points['timestamp'], jump_points['dvol'],
                       color='red', s=20, alpha=0.6, label='Detected Jumps', zorder=5)
        
        axes[0].set_title('Bitcoin DVOL with Detected Jump Events', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('DVOL')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        method_comparison = pd.DataFrame({
            'Lee-Mykland': self.df['lee_mykland_jump'].astype(int),
            'Sigma Threshold': self.df['sigma_jump'].astype(int),
            'Return Z-score': self.df['zscore_jump'].astype(int)
        })
        
        axes[1].plot(self.df['timestamp'], method_comparison['Lee-Mykland'].cumsum(),
                    label='Lee-Mykland', linewidth=2)
        axes[1].plot(self.df['timestamp'], method_comparison['Sigma Threshold'].cumsum(),
                    label='Sigma Threshold', linewidth=2)
        axes[1].plot(self.df['timestamp'], method_comparison['Return Z-score'].cumsum(),
                    label='Return Z-score', linewidth=2)
        
        axes[1].set_title('Cumulative Jump Detection by Method', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Cumulative Jumps')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        recent_df = self.df[self.df['timestamp'] >= '2023-01-01']
        axes[2].plot(recent_df['timestamp'], recent_df['dvol'],
                    linewidth=1, alpha=0.7)
        
        recent_jumps = recent_df[recent_df['jump_any']]
        axes[2].scatter(recent_jumps['timestamp'], recent_jumps['dvol'],
                       color='red', s=30, alpha=0.8, zorder=5)
        
        for _, row in recent_jumps.iterrows():
            axes[2].axvline(x=row['timestamp'], color='red', 
                          alpha=0.2, linestyle='--', linewidth=1)
        
        axes[2].set_title('Recent Period Detail (2023-2025)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('DVOL')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path('results/visualizations/jumps')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'jump_detection_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved: {output_path}")
        plt.close()
        
        self._create_distribution_plots()
    
    def _create_distribution_plots(self):
        """Create distribution comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        normal_dvol = self.df[~self.df['jump_any']]['dvol']
        jump_dvol = self.df[self.df['jump_any']]['dvol']
        
        axes[0, 0].hist(normal_dvol, bins=50, alpha=0.7, label='Normal', density=True)
        axes[0, 0].hist(jump_dvol, bins=30, alpha=0.7, label='Jump', density=True)
        axes[0, 0].set_title('DVOL Distribution: Normal vs Jump Periods')
        axes[0, 0].set_xlabel('DVOL')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        returns = self.df['dvol'].pct_change()
        normal_returns = returns[~self.df['jump_any']]
        jump_returns = returns[self.df['jump_any']]
        
        axes[0, 1].hist(normal_returns.dropna(), bins=50, alpha=0.7, 
                       label='Normal', density=True, range=(-0.5, 0.5))
        axes[0, 1].hist(jump_returns.dropna(), bins=30, alpha=0.7,
                       label='Jump', density=True, range=(-0.5, 0.5))
        axes[0, 1].set_title('Return Distribution: Normal vs Jump Periods')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        jump_counts_daily = self.df.set_index('timestamp')['jump_any'].resample('D').sum()
        axes[1, 0].plot(jump_counts_daily.index, jump_counts_daily.values, linewidth=1)
        axes[1, 0].set_title('Daily Jump Frequency Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Jumps per Day')
        axes[1, 0].grid(True, alpha=0.3)
        
        stats_df = pd.DataFrame({
            'Method': ['Lee-Mykland', 'Sigma (3σ)', 'Z-score', 'Composite'],
            'Count': [
                self.df['lee_mykland_jump'].sum(),
                self.df['sigma_jump'].sum(),
                self.df['zscore_jump'].sum(),
                self.df['jump_any'].sum()
            ]
        })
        
        axes[1, 1].bar(stats_df['Method'], stats_df['Count'], alpha=0.7)
        axes[1, 1].set_title('Jump Detection by Method')
        axes[1, 1].set_ylabel('Number of Jumps')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = Path('results/visualizations/jumps/jump_distributions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plots saved: {output_path}")
        plt.close()
    
    def save_jump_data(self):
        """Save dataset with jump indicators."""
        # Include data version in output path
        output_path = f'data/processed/bitcoin_lstm_features_{self.data_version}_with_jumps.csv'
        self.df.to_csv(output_path, index=False)
        print(f"\n" + "=" * 80)
        print(f"SAVED: {output_path}")
        print("=" * 80)
        print("Dataset now includes jump detection features:")
        print("  - jump_indicator: Binary (0/1)")
        print("  - jump_magnitude: Size of jump")
        print("  - days_since_jump: Time since last jump")
        print("  - jump_cluster_7d: Jump count in 7-day window")
        print()

    def analyze_jump_persistence_and_export_masks(self, output_dir='results/thesis_v2'):
        """
        Analyze jump persistence patterns and export standardized jump period masks.

        This method addresses the research question: How long do jump effects persist?
        Based on empirical analysis and literature standards for jump window definition.
        """
        print("=" * 80)
        print("JUMP PERSISTENCE ANALYSIS & MASK EXPORT")
        print("=" * 80)
        print("Analyzing jump duration patterns for thesis evaluation framework")
        print()

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Analyze persistence patterns for different jump magnitudes
        jumps_df = self.df[self.df['jump_any']].copy()
        print(f"Found {len(jumps_df):,} jump periods ({len(jumps_df)/len(self.df)*100:.1f}% of data)")
        print()

        # Categorize jumps by magnitude for persistence analysis
        jumps_df['jump_category'] = pd.cut(
            jumps_df['jump_magnitude'],
            bins=[0, 1, 2, np.inf],
            labels=['Small', 'Medium', 'Large']
        )

        print("Jump magnitude distribution:")
        for category in ['Small', 'Medium', 'Large']:
            count = len(jumps_df[jumps_df['jump_category'] == category])
            print(f"  {category}: {count:,} jumps ({count/len(jumps_df)*100:.1f}%)")
        print()

        # Analyze persistence: How long until DVOL returns to near-normal levels?
        print("Analyzing jump persistence patterns:")
        print("-" * 50)

        persistence_results = {}
        windows_to_test = [24, 48, 72]  # 1, 2, 3 days

        for window_hours in windows_to_test:
            print(f"\n{window_hours}h window analysis:")

            # Calculate DVOL volatility in post-jump windows
            post_jump_volatilities = []

            for _, jump_row in jumps_df.iterrows():
                jump_time = jump_row['timestamp']
                jump_dvol = jump_row['dvol']

                # Get DVOL values in post-jump window
                window_end = jump_time + pd.Timedelta(hours=window_hours)
                window_data = self.df[
                    (self.df['timestamp'] > jump_time) &
                    (self.df['timestamp'] <= window_end)
                ]

                if len(window_data) > 0:
                    # Calculate volatility in this window
                    window_vol = window_data['dvol'].std()
                    post_jump_volatilities.append(window_vol)

            # Compare to baseline volatility (non-jump periods)
            baseline_vol = self.df[~self.df['jump_any']]['dvol'].std()
            avg_post_jump_vol = np.mean(post_jump_volatilities) if post_jump_volatilities else baseline_vol

            volatility_ratio = avg_post_jump_vol / baseline_vol
            print(f"  Baseline volatility: {baseline_vol:.4f}")
            print(f"  Post-jump volatility: {avg_post_jump_vol:.4f}")
            print(f"  Volatility ratio: {volatility_ratio:.2f}x")

            # Persistence criterion: volatility still >1.5x baseline
            persistent = volatility_ratio > 1.5
            print(f"  Significant effect: {'Yes' if persistent else 'No'}")

            persistence_results[window_hours] = {
                'volatility_ratio': volatility_ratio,
                'persistent': persistent,
                'avg_volatility': avg_post_jump_vol
            }

        # Export standardized jump masks based on persistence analysis
        print(f"\n" + "=" * 60)
        print("EXPORTING STANDARDIZED JUMP MASKS")
        print("=" * 60)

        # Create output dataframe with timestamps and jump indicators
        jump_masks = self.df[['timestamp']].copy()
        jump_masks['jump_indicator'] = self.df['jump_any'].astype(int)
        jump_masks['jump_magnitude'] = self.df['jump_magnitude']

        # Add jump period masks for different window definitions
        for window_hours in windows_to_test:
            mask_col = f'jump_period_{window_hours}h'
            jump_masks[mask_col] = 0

            # Mark jump periods (jump day + specified hours)
            for _, jump_row in jumps_df.iterrows():
                jump_time = jump_row['timestamp']
                window_end = jump_time + pd.Timedelta(hours=window_hours)

                mask_condition = (
                    (jump_masks['timestamp'] >= jump_time) &
                    (jump_masks['timestamp'] <= window_end)
                )
                jump_masks.loc[mask_condition, mask_col] = 1

        # Export jump masks
        masks_file = f"{output_dir}/jump_period_masks_{self.data_version}.csv"
        jump_masks.to_csv(masks_file, index=False)
        print(f"Jump masks exported: {masks_file}")

        # Create summary statistics
        summary_stats = {
            'data_version': self.data_version,
            'total_observations': len(self.df),
            'total_jumps': len(jumps_df),
            'jump_percentage': len(jumps_df) / len(self.df) * 100,
            'persistence_analysis': persistence_results,
            'recommended_windows': []
        }

        # Recommend windows based on persistence analysis
        for window_hours in windows_to_test:
            if persistence_results[window_hours]['persistent']:
                summary_stats['recommended_windows'].append(window_hours)

        # Export summary
        summary_file = f"{output_dir}/jump_period_summary_{self.data_version}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        print(f"Jump analysis summary: {summary_file}")

        # Print key findings
        print(f"\n" + "=" * 60)
        print("KEY FINDINGS FOR THESIS METHODOLOGY")
        print("=" * 60)

        print(f"\nJump Characteristics:")
        print(f"• Total jumps detected: {summary_stats['total_jumps']:,}")
        print(f"• Jump frequency: {summary_stats['jump_percentage']:.1f}% of observations")
        print(f"• Recommended evaluation windows: {summary_stats['recommended_windows']} hours")

        print(f"\nPersistence Analysis Results:")
        for window_hours, results in persistence_results.items():
            status = "Significant" if results['persistent'] else "Diminished"
            print(f"• {window_hours}h post-jump: {status} effect ({results['volatility_ratio']:.2f}x volatility)")

        print(f"\nMethodological Implications:")
        if len(summary_stats['recommended_windows']) > 0:
            print(f"• Use {summary_stats['recommended_windows']}h windows for jump-period evaluation")
            print(f"• Jump effects persist beyond immediate jump detection")
        else:
            print(f"• Jump effects diminish within 24h")
            print(f"• Use 24h windows for conservative evaluation")

        return jump_masks, summary_stats

    def run_full_analysis(self, export_masks=True):
        """Execute complete jump detection pipeline."""
        self.load_data()

        self.method1_lee_mykland()
        self.method2_sigma_threshold()
        self.method3_return_zscore()

        self.create_composite_jump_indicator()
        self.identify_major_events()

        self.create_jump_features()
        self.analyze_jump_impact_on_forecast()

        self.visualize_jumps()
        self.save_jump_data()

        if export_masks:
            self.analyze_jump_persistence_and_export_masks()

        print("\n" + "=" * 80)
        print("JUMP DETECTION ANALYSIS COMPLETE")
        print("=" * 80)
        print("Next step: Train LSTM with jump-aware features")
        print("File: scripts/modeling/main_rolling_with_jumps.py")
        print("=" * 80)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Jump Detection Analysis for Bitcoin DVOL'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/bitcoin_lstm_features.csv',
        help='Path to the features dataset (default: data/processed/bitcoin_lstm_features.csv)'
    )
    parser.add_argument(
        '--data-version',
        type=str,
        default='v1.0',
        help='Data version identifier for output files (default: v1.0)'
    )
    parser.add_argument(
        '--v1-1',
        dest='use_v1_1',
        action='store_true',
        help='Use v1.1 dataset (shorthand for --data-path data/archive/v1.1_2025-12-29/bitcoin_lstm_features.csv --data-version v1.1)'
    )

    args = parser.parse_args()

    # Handle shorthand --v1-1 flag
    if args.use_v1_1:
        args.data_path = 'data/archive/v1.1_2025-12-29/bitcoin_lstm_features.csv'
        args.data_version = 'v1.1'

    # Run analysis
    analyzer = JumpDetectionAnalysis(data_path=args.data_path, data_version=args.data_version)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
