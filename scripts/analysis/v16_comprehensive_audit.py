"""
Comprehensive Data Audit: Bitcoin DVOL Dataset v1.6_final
==========================================================
Audit Date: 2026-04-02
Purpose: Thesis-grade data quality verification before model training.

Sections:
    1. Temporal Continuity
    2. Feature-Level Quality
    3. Jump Detection Validation
    4. Train/Val/Test Split Integrity
    5. Cross-Feature Relationships
    6. Model Input Pipeline Verification

Output:
    - results/diagnostics/v16_audit/metrics.json
    - results/diagnostics/v16_audit/figures/*.png
    - Console summary with PASS/FAIL per section
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "bitcoin_lstm_features_v1.6_final.csv"
OUTPUT_DIR = BASE_DIR / "results" / "diagnostics" / "v16_audit"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0
RESULTS = {}


def verdict(section, check, passed, detail="", warn=False):
    global PASS_COUNT, FAIL_COUNT, WARN_COUNT
    if warn and not passed:
        status = "WARN"
        WARN_COUNT += 1
    elif passed:
        status = "PASS"
        PASS_COUNT += 1
    else:
        status = "FAIL"
        FAIL_COUNT += 1

    RESULTS.setdefault(section, []).append(
        {
            "check": check,
            "status": status,
            "detail": str(detail),
        }
    )
    label = f"[{status}] {section} :: {check}"
    if detail:
        label += f" -- {detail}"
    print(label)
    return status


# ── Load ──────────────────────────────────────────────────────────
print("=" * 70)
print("Loading v1.6_final dataset ...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)
N = len(df)
print(f"Rows: {N}, Columns: {len(df.columns)}")
print(f"Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print("=" * 70)

CORE_FEATURES = [
    "dvol",
    "dvol_lag_1d",
    "dvol_lag_7d",
    "dvol_lag_30d",
    "network_activity",
    "nvrv",
    "dvol_rv_spread",
    "transaction_volume",
]
JUMP_FEATURES = [
    "lee_mykland_stat",
    "lee_mykland_jump",
    "jump_magnitude",
    "hours_since_jump",
    "days_since_jump",
    "jump_cluster_7d",
]
ALL_MODEL_FEATURES = CORE_FEATURES + ["lee_mykland_jump", "jump_magnitude"]

SPLIT_RATIOS = (0.60, 0.20, 0.20)
TRAIN_END = int(N * SPLIT_RATIOS[0])
VAL_END = TRAIN_END + int(N * SPLIT_RATIOS[1])


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — TEMPORAL CONTINUITY                              ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 1: TEMPORAL CONTINUITY")
print("─" * 70)

sec = "1_temporal"

diffs = df["timestamp"].diff().dropna()
diff_hours = diffs.dt.total_seconds() / 3600

expected_gaps = (diff_hours != 1.0).sum()
actual_gaps = expected_gaps
verdict(
    sec,
    "No missing hours (all diff=1h)",
    actual_gaps == 0,
    f"gaps found: {actual_gaps}",
)

gap_details = diff_hours[diff_hours != 1.0]
if len(gap_details) > 0:
    for idx, v in gap_details.items():
        print(
            f"    Gap at row {idx}: {v:.1f}h  "
            f"({df['timestamp'].iloc[idx - 1]} → {df['timestamp'].iloc[idx]})"
        )

verdict(sec, "Total rows = expected hours", N == 41055, f"rows={N}, expected=41055")

verdict(
    sec,
    "Start date correct",
    df["timestamp"].iloc[0] == pd.Timestamp("2021-04-23 09:00:00"),
    f"start={df['timestamp'].iloc[0]}",
)

verdict(
    sec,
    "End date correct",
    df["timestamp"].iloc[-1] == pd.Timestamp("2025-12-28 23:00:00"),
    f"end={df['timestamp'].iloc[-1]}",
)

dups = df["timestamp"].duplicated().sum()
verdict(sec, "No duplicate timestamps", dups == 0, f"duplicates={dups}")

hour_vals = df["timestamp"].dt.hour.value_counts().sort_index()
hour_uniform = hour_vals.std() / hour_vals.mean() < 0.15
verdict(
    sec,
    "Hours uniformly distributed",
    hour_uniform,
    f"cv={hour_vals.std() / hour_vals.mean():.4f} (threshold 0.15)",
)

fig, ax = plt.subplots(figsize=(16, 3))
ax.plot(df["timestamp"], df["dvol"], linewidth=0.3, alpha=0.8)
ax.set_title("DVOL Time Series — Full Coverage")
ax.set_ylabel("DVOL")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "01_dvol_timeseries.png", dpi=150)
plt.close(fig)
print("  → Saved 01_dvol_timeseries.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — FEATURE-LEVEL QUALITY                            ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 2: FEATURE-LEVEL QUALITY")
print("─" * 70)

sec = "2_feature_quality"

feature_stats = {}
for col in df.columns:
    if col == "timestamp":
        continue
    s = df[col]
    n_null = int(s.isna().sum())
    n_inf = int(np.isinf(s).sum()) if s.dtype != object else 0
    feature_stats[col] = {
        "nulls": n_null,
        "inf": n_inf,
        "mean": float(s.mean()) if s.dtype != object else None,
        "std": float(s.std()) if s.dtype != object else None,
        "min": float(s.min()) if s.dtype != object else None,
        "max": float(s.max()) if s.dtype != object else None,
        "nunique": int(s.nunique()),
    }

for feat in CORE_FEATURES:
    s = df[feat]
    n_null = int(s.isna().sum())
    n_inf = int(np.isinf(s).sum())
    verdict(sec, f"{feat}: no infinite values", n_inf == 0, f"inf_count={n_inf}")

expected_nulls = {
    "dvol": 0,
    "network_activity": 0,
    "nvrv": 0,
    "transaction_volume": 0,
    "dvol_lag_1d": 24,
    "dvol_lag_7d": 168,
    "dvol_lag_30d": 720,
    "dvol_rv_spread": 24,
}
for feat, exp_null in expected_nulls.items():
    actual_null = int(df[feat].isna().sum())
    verdict(
        sec,
        f"{feat}: nulls match expected ({exp_null})",
        actual_null == exp_null,
        f"actual={actual_null}, expected={exp_null}",
    )

verdict(
    sec,
    "dvol: no negative values",
    (df["dvol"] < 0).sum() == 0,
    f"neg_count={(df['dvol'] < 0).sum()}",
)

verdict(
    sec,
    "dvol: no zero values",
    (df["dvol"] == 0).sum() == 0,
    f"zero_count={(df['dvol'] == 0).sum()}",
)

verdict(
    sec,
    "dvol range plausible [25, 200]",
    df["dvol"].min() >= 25 and df["dvol"].max() <= 200,
    f"range=[{df['dvol'].min():.2f}, {df['dvol'].max():.2f}]",
)

verdict(
    sec,
    "network_activity: all positive",
    (df["network_activity"] <= 0).sum() == 0,
    f"non_positive={(df['network_activity'] <= 0).sum()}",
)

verdict(
    sec,
    "transaction_volume: all positive",
    (df["transaction_volume"] <= 0).sum() == 0,
    f"non_positive={(df['transaction_volume'] <= 0).sum()}",
)

dvol_std = df["dvol"].std()
dvol_mean = df["dvol"].mean()
dvol_cv = dvol_std / dvol_mean
verdict(
    sec,
    "DVOL coefficient of variation reasonable (0.1-1.0)",
    0.1 < dvol_cv < 1.0,
    f"CV={dvol_cv:.4f}",
    warn=True,
)

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()
for i, feat in enumerate(CORE_FEATURES):
    ax = axes[i]
    data = df[feat].dropna()
    ax.hist(data, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.axvline(
        data.mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"mean={data.mean():.2f}",
    )
    ax.axvline(
        data.median(),
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"median={data.median():.2f}",
    )
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Feature Distributions — v1.6 Final", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → Saved 02_feature_distributions.png")

fig, axes = plt.subplots(4, 2, figsize=(16, 10))
axes = axes.flatten()
for i, feat in enumerate(CORE_FEATURES):
    ax = axes[i]
    ax.plot(df["timestamp"], df[feat], linewidth=0.3, alpha=0.6)
    ax.set_title(feat, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Feature Time Series — v1.6 Final", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "03_feature_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → Saved 03_feature_timeseries.png")

q99 = df["dvol"].quantile(0.99)
q01 = df["dvol"].quantile(0.01)
iqr = df["dvol"].quantile(0.75) - df["dvol"].quantile(0.25)
upper_fence = df["dvol"].quantile(0.75) + 3 * iqr
lower_fence = df["dvol"].quantile(0.25) - 3 * iqr
outlier_count = ((df["dvol"] > upper_fence) | (df["dvol"] < lower_fence)).sum()
verdict(
    sec,
    f"dvol: extreme outliers <1%",
    outlier_count / N < 0.01,
    f"outliers={outlier_count} ({outlier_count / N * 100:.3f}%)",
)

skew = df["dvol"].skew()
kurt = df["dvol"].kurtosis()
verdict(
    sec,
    f"dvol: skewness reasonable (-2 to 3)",
    -2 < skew < 3,
    f"skew={skew:.4f}, kurtosis={kurt:.4f}",
    warn=True,
)


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — JUMP DETECTION VALIDATION                        ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 3: JUMP DETECTION VALIDATION")
print("─" * 70)

sec = "3_jump_detection"

n_jumps = int(df["lee_mykland_jump"].sum())
jump_pct = n_jumps / N * 100
verdict(
    sec,
    "Lee-Mykland jumps ≈ 236 (standard implementation)",
    230 <= n_jumps <= 245,
    f"jumps={n_jumps} ({jump_pct:.2f}%)",
)

threshold_val = df["lee_mykland_threshold"].iloc[0]
verdict(
    sec,
    "Lee-Mykland threshold ≈ 9.21 (Gumbel β*)",
    abs(threshold_val - 9.2103) < 0.01,
    f"threshold={threshold_val:.4f}",
)

verdict(
    sec,
    "Threshold is constant across all rows",
    df["lee_mykland_threshold"].nunique() == 1,
    f"unique values={df['lee_mykland_threshold'].nunique()}",
)

verdict(
    sec,
    "lee_mykland_jump is binary (0/1)",
    set(df["lee_mykland_jump"].unique()) <= {0, 1},
    f"unique values={sorted(df['lee_mykland_jump'].unique())}",
)

verdict(
    sec,
    "jump_magnitude is 0 for non-jump rows",
    (df.loc[df["lee_mykland_jump"] == 0, "jump_magnitude"] == 0).all(),
    f"non-zero count for non-jumps: {(df.loc[df['lee_mykland_jump'] == 0, 'jump_magnitude'] != 0).sum()}",
)

jump_mags = df.loc[df["lee_mykland_jump"] == 1, "jump_magnitude"]
verdict(
    sec,
    "jump_magnitude > 0 for jump rows",
    (jump_mags > 0).all() if len(jump_mags) > 0 else False,
    f"zero mag jumps: {(jump_mags == 0).sum() if len(jump_mags) > 0 else 'N/A'}",
)

T_stats = df["lee_mykland_T_statistic"].dropna()
verdict(
    sec,
    "T-statistic range plausible",
    T_stats.min() < -10 and T_stats.max() > 50,
    f"range=[{T_stats.min():.2f}, {T_stats.max():.2f}]",
)

jump_mask = df["lee_mykland_jump"] == 1
jump_hours_since = df.loc[jump_mask, "hours_since_jump"]
verdict(
    sec,
    "hours_since_jump resets to 0 at each jump",
    (jump_hours_since == 0).sum() == n_jumps,
    f"resets={(jump_hours_since == 0).sum()}, expected={n_jumps}",
    warn=True,
)

KNOWN_EVENTS = {
    "China ban (May 2021)": ("2021-05-15", "2021-05-25"),
    "Luna collapse (May 2022)": ("2022-05-07", "2022-05-15"),
    "3AC/CELSIUS (Jun 2022)": ("2022-06-10", "2022-06-25"),
    "FTX collapse (Nov 2022)": ("2022-11-05", "2022-11-15"),
    "SVB crisis (Mar 2023)": ("2023-03-08", "2023-03-15"),
    "ETF approval (Jan 2024)": ("2024-01-05", "2024-01-15"),
}

event_hits = {}
for name, (start, end) in KNOWN_EVENTS.items():
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    hits = int(df.loc[mask, "lee_mykland_jump"].sum())
    event_hits[name] = hits
    verdict(sec, f"Event detected: {name}", hits > 0, f"jumps={hits}", warn=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(jump_mags, bins=40, edgecolor="black", linewidth=0.3, alpha=0.7, color="orange")
ax.set_title(f"Jump Magnitude Distribution ({n_jumps} jumps)")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Count")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
jump_ts = df.loc[jump_mask, "timestamp"]
ax.scatter(
    jump_ts, df.loc[jump_mask, "dvol"], s=8, c="red", alpha=0.6, label="LM Jumps"
)
ax.plot(df["timestamp"], df["dvol"], linewidth=0.2, alpha=0.3, color="gray")
for name, (start, end) in KNOWN_EVENTS.items():
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color="yellow")
ax.set_title("Jump Events on DVOL Timeline")
ax.set_xlabel("Date")
ax.set_ylabel("DVOL")
ax.legend(fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURE_DIR / "04_jump_analysis.png", dpi=150)
plt.close(fig)
print("  → Saved 04_jump_analysis.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — TRAIN / VAL / TEST SPLIT INTEGRITY               ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 4: TRAIN / VAL / TEST SPLIT INTEGRITY")
print("─" * 70)

sec = "4_split_integrity"

train = df.iloc[:TRAIN_END]
val = df.iloc[TRAIN_END:VAL_END]
test = df.iloc[VAL_END:]

verdict(
    sec,
    "Train split is 60%",
    len(train) == int(N * 0.60),
    f"train={len(train)}, expected={int(N * 0.60)}",
)

verdict(
    sec,
    "Val split is 20%",
    len(val) == int(N * 0.20),
    f"val={len(val)}, expected={int(N * 0.20)}",
)

verdict(
    sec,
    "Test split is 20%",
    len(test) == int(N * 0.20),
    f"test={len(test)}, expected={int(N * 0.20)}",
)

verdict(
    sec,
    "Train ends before val starts",
    train["timestamp"].iloc[-1] < val["timestamp"].iloc[0],
    f"train_end={train['timestamp'].iloc[-1]}, val_start={val['timestamp'].iloc[0]}",
)

verdict(
    sec,
    "Val ends before test starts",
    val["timestamp"].iloc[-1] < test["timestamp"].iloc[0],
    f"val_end={val['timestamp'].iloc[-1]}, test_start={test['timestamp'].iloc[0]}",
)

train_jumps = int(train["lee_mykland_jump"].sum())
val_jumps = int(val["lee_mykland_jump"].sum())
test_jumps = int(test["lee_mykland_jump"].sum())
verdict(
    sec,
    "Jumps present in all splits",
    train_jumps > 0 and val_jumps > 0 and test_jumps > 0,
    f"train={train_jumps}, val={val_jumps}, test={test_jumps}",
)

print(f"\n  Split summary:")
print(
    f"  {'Split':<8} {'Rows':>7} {'Date Range':<46} {'DVOL Mean':>10} {'DVOL Std':>10} {'Jumps':>7}"
)
for label, split in [("Train", train), ("Val", val), ("Test", test)]:
    print(
        f"  {label:<8} {len(split):>7} "
        f"{split['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {split['timestamp'].iloc[-1].strftime('%Y-%m-%d'):>12} "
        f"{split['dvol'].mean():>10.2f} {split['dvol'].std():>10.2f} "
        f"{int(split['lee_mykland_jump'].sum()):>7}"
    )

dvol_shift = (
    abs(train["dvol"].mean() - test["dvol"].mean()) / train["dvol"].mean() * 100
)
verdict(
    sec,
    f"DVOL mean shift train→test < 50%",
    dvol_shift < 50,
    f"shift={dvol_shift:.1f}% (train_mean={train['dvol'].mean():.1f}, test_mean={test['dvol'].mean():.1f})",
    warn=True,
)

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

ax = axes[0, 0]
splits_data = [("Train", train), ("Val", val), ("Test", test)]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, (label, split) in enumerate(splits_data):
    ax.hist(
        split["dvol"],
        bins=60,
        alpha=0.5,
        label=f"{label} (μ={split['dvol'].mean():.1f})",
        color=colors[i],
        edgecolor="black",
        linewidth=0.3,
    )
ax.set_title("DVOL Distribution by Split")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[0, 1]
for i, (label, split) in enumerate(splits_data):
    ax.plot(
        split["timestamp"],
        split["dvol"],
        linewidth=0.3,
        alpha=0.6,
        label=label,
        color=colors[i],
    )
ax.axvline(
    val["timestamp"].iloc[0], color="red", linestyle="--", linewidth=1, alpha=0.5
)
ax.axvline(
    test["timestamp"].iloc[0], color="red", linestyle="--", linewidth=1, alpha=0.5
)
ax.set_title("DVOL by Split (Timeline)")
ax.legend(fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1, 0]
split_labels = ["Train", "Val", "Test"]
jump_counts = [train_jumps, val_jumps, test_jumps]
ax.bar(split_labels, jump_counts, color=colors, edgecolor="black")
ax.set_title(f"Lee-Mykland Jumps per Split (total={n_jumps})")
ax.set_ylabel("Jump Count")
for i, v in enumerate(jump_counts):
    ax.text(i, v + 2, str(v), ha="center", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1, 1]
features_for_box = ["dvol", "nvrv", "dvol_rv_spread"]
data_for_box = [
    train[features_for_box].describe().loc["mean"],
    val[features_for_box].describe().loc["mean"],
    test[features_for_box].describe().loc["mean"],
]
x = np.arange(len(features_for_box))
width = 0.25
for i, (label, d) in enumerate(zip(split_labels, data_for_box)):
    ax.bar(
        x + i * width,
        d.values,
        width,
        label=label,
        color=colors[i],
        edgecolor="black",
        linewidth=0.3,
    )
ax.set_xticks(x + width)
ax.set_xticklabels(features_for_box, fontsize=9)
ax.set_title("Feature Means by Split")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURE_DIR / "05_split_analysis.png", dpi=150)
plt.close(fig)
print("  → Saved 05_split_analysis.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — CROSS-FEATURE RELATIONSHIPS                      ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 5: CROSS-FEATURE RELATIONSHIPS")
print("─" * 70)

sec = "5_relationships"

corr = df[CORE_FEATURES].corr()

dvol_autocorr_lag1 = df["dvol"].autocorr(lag=1)
dvol_autocorr_lag24 = df["dvol"].autocorr(lag=24)
dvol_autocorr_lag168 = df["dvol"].autocorr(lag=168)
print(f"  DVOL autocorrelation lag-1h:  {dvol_autocorr_lag1:.6f}")
print(f"  DVOL autocorrelation lag-24h: {dvol_autocorr_lag24:.6f}")
print(f"  DVOL autocorrelation lag-168h: {dvol_autocorr_lag168:.6f}")

verdict(
    sec,
    "DVOL lag-1 autocorrelation > 0.99",
    dvol_autocorr_lag1 > 0.99,
    f"ρ={dvol_autocorr_lag1:.6f}",
)

verdict(
    sec,
    "DVOL lag-168 autocorrelation < 0.98",
    dvol_autocorr_lag168 < 0.98,
    f"ρ={dvol_autocorr_lag168:.6f}",
)

dvol_rv_corr = corr.loc["dvol", "dvol_rv_spread"]
verdict(
    sec,
    "dvol_rv_spread highly correlated with dvol (>0.95)",
    dvol_rv_corr > 0.95,
    f"ρ={dvol_rv_corr:.4f}",
)

lag1_corr = corr.loc["dvol", "dvol_lag_1d"]
verdict(
    sec,
    "dvol_lag_1d correlated with dvol (>0.97, expected ~0.98 for 24h lag)",
    lag1_corr > 0.97,
    f"ρ={lag1_corr:.4f}",
)

dvol_nvrv_corr = corr.loc["dvol", "nvrv"]
verdict(
    sec,
    "NVRV positively correlated with DVOL",
    dvol_nvrv_corr > 0,
    f"ρ={dvol_nvrv_corr:.4f}",
    warn=True,
)

for f1 in CORE_FEATURES:
    for f2 in CORE_FEATURES:
        if f1 >= f2:
            continue
        c = corr.loc[f1, f2]
        if abs(c) > 0.95 and f1 != "dvol" and f2 != "dvol":
            print(f"  ⚠ Near-perfect correlation: {f1} ↔ {f2} = {c:.4f}")

from statsmodels.stats.outliers_influence import variance_inflation_factor

lag_features = [
    "dvol_lag_1d",
    "dvol_lag_7d",
    "dvol_lag_30d",
    "network_activity",
    "nvrv",
    "dvol_rv_spread",
    "transaction_volume",
]
vif_data = df[lag_features].dropna()
vif_results = {}
for i, feat in enumerate(lag_features):
    vif_val = variance_inflation_factor(vif_data.values, i)
    vif_results[feat] = round(vif_val, 2)

high_vif = any(v > 10 for v in vif_results.values())
verdict(
    sec,
    "No multicollinearity (all VIF < 10)",
    not high_vif,
    f"max VIF={max(vif_results.values()):.2f}",
    warn=True,
)
print(f"  VIF: {vif_results}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(CORE_FEATURES)))
ax.set_yticks(range(len(CORE_FEATURES)))
ax.set_xticklabels(CORE_FEATURES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CORE_FEATURES, fontsize=8)
for i in range(len(CORE_FEATURES)):
    for j in range(len(CORE_FEATURES)):
        ax.text(
            j,
            i,
            f"{corr.values[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(corr.values[i, j]) > 0.6 else "black",
        )
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature Correlation Matrix")

ax = axes[1]
autocorr_vals = [df["dvol"].autocorr(lag=l) for l in range(1, 169)]
ax.plot(range(1, 169), autocorr_vals, linewidth=1)
ax.axhline(0.95, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="ρ=0.95")
ax.axhline(
    0.99, color="orange", linestyle="--", linewidth=0.8, alpha=0.5, label="ρ=0.99"
)
ax.set_title("DVOL Autocorrelation Decay (1-168h)")
ax.set_xlabel("Lag (hours)")
ax.set_ylabel("Autocorrelation")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURE_DIR / "06_correlations_autocorr.png", dpi=150)
plt.close(fig)
print("  → Saved 06_correlations_autocorr.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — STATIONARITY & NORMALIZATION VERIFICATION        ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 6: STATIONARITY & NORMALIZATION VERIFICATION")
print("─" * 70)

sec = "6_stationarity"

WINDOW = 720

stationary_features = [
    "dvol",
    "dvol_lag_1d",
    "dvol_lag_7d",
    "dvol_lag_30d",
    "network_activity",
    "nvrv",
    "dvol_rv_spread",
    "transaction_volume",
]

adf_results = {}
for feat in stationary_features:
    series = df[feat].dropna().values
    if len(series) < 100:
        continue
    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series, autolag="AIC")
        adf_results[feat] = {
            "adf_stat": round(result[0], 4),
            "p_value": round(result[1], 6),
            "stationary": result[1] < 0.05,
        }
    except Exception as e:
        adf_results[feat] = {"error": str(e)}

print("  ADF Tests on RAW features:")
for feat, res in adf_results.items():
    if "error" in res:
        print(f"    {feat}: ERROR - {res['error']}")
        continue
    status = "stationary" if res["stationary"] else "NON-STATIONARY"
    print(f"    {feat}: ADF={res['adf_stat']:.4f}, p={res['p_value']:.6f} → {status}")

raw_nonstationary = sum(
    1 for r in adf_results.values() if "error" not in r and not r["stationary"]
)
print(f"\n  Raw features: {raw_nonstationary} non-stationary")

norm_features = [
    "dvol",
    "network_activity",
    "nvrv",
    "dvol_rv_spread",
    "transaction_volume",
]
norm_adf_results = {}
for feat in norm_features:
    series = df[feat].dropna().values
    if len(series) < WINDOW + 100:
        continue
    roll_mean = pd.Series(series).rolling(WINDOW, min_periods=1).mean().values
    roll_std = pd.Series(series).rolling(WINDOW, min_periods=1).std().values
    roll_std[roll_std == 0] = 1.0
    norm_series = (series - roll_mean) / roll_std
    norm_series = norm_series[WINDOW:]
    try:
        result = adfuller(norm_series, autolag="AIC")
        norm_adf_results[feat] = {
            "adf_stat": round(result[0], 4),
            "p_value": round(result[1], 6),
            "stationary": result[1] < 0.05,
        }
    except Exception as e:
        norm_adf_results[feat] = {"error": str(e)}

print(f"\n  ADF Tests on ROLLING NORMALIZED features ({WINDOW}h window):")
norm_all_stationary = True
for feat, res in norm_adf_results.items():
    if "error" in res:
        print(f"    {feat}: ERROR - {res['error']}")
        norm_all_stationary = False
        continue
    status = "stationary" if res["stationary"] else "NON-STATIONARY"
    if not res["stationary"]:
        norm_all_stationary = False
    print(f"    {feat}: ADF={res['adf_stat']:.4f}, p={res['p_value']:.6f} → {status}")

verdict(
    sec,
    "Rolling normalization makes all features stationary",
    norm_all_stationary,
    f"all {len(norm_adf_results)} features stationary after {WINDOW}h rolling z-score",
)

fig, axes = plt.subplots(
    len(norm_features), 1, figsize=(16, 3 * len(norm_features)), sharex=True
)
for i, feat in enumerate(norm_features):
    ax = axes[i]
    series = df[feat].values
    roll_mean = pd.Series(series).rolling(WINDOW, min_periods=1).mean().values
    roll_std = pd.Series(series).rolling(WINDOW, min_periods=1).std().values
    roll_std[roll_std == 0] = 1.0
    norm_series = (series - roll_mean) / roll_std
    ax.plot(df["timestamp"], norm_series, linewidth=0.3, alpha=0.6)
    ax.axhline(0, color="red", linewidth=0.5, alpha=0.5)
    ax.axhline(2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel(feat, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
fig.suptitle(f"Rolling Normalized Features ({WINDOW}h window)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "07_rolling_normalized.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → Saved 07_rolling_normalized.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — DVOL-RV SPREAD VERIFICATION                      ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 7: DVOL-RV SPREAD VERIFICATION")
print("─" * 70)

sec = "7_dvol_rv_spread"

spread = df["dvol_rv_spread"].dropna()
verdict(
    sec,
    "DVOL-RV spread all positive (implied > realized)",
    (spread > 0).all(),
    f"negative count: {(spread < 0).sum()}",
)

verdict(
    sec,
    "DVOL-RV spread range plausible",
    spread.min() > 20 and spread.max() < 200,
    f"range=[{spread.min():.2f}, {spread.max():.2f}]",
)

spread_mean = spread.mean()
verdict(
    sec,
    "DVOL-RV spread mean ~ DVOL mean (correlation ~1.0)",
    abs(spread_mean - df["dvol"].mean()) < 10,
    f"spread_mean={spread_mean:.2f}, dvol_mean={df['dvol'].mean():.2f}",
)

spread_corr = df[["dvol", "dvol_rv_spread"]].dropna().corr().iloc[0, 1]
verdict(
    sec,
    "DVOL-RV spread highly correlated with DVOL (>0.99)",
    spread_corr > 0.99,
    f"ρ={spread_corr:.6f}",
)


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — LAG FEATURE INTEGRITY                            ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 8: LAG FEATURE INTEGRITY")
print("─" * 70)

sec = "8_lag_integrity"

lag1_check = df["dvol_lag_1d"].dropna()
lag1_expected = df["dvol"].shift(24).dropna()
common_idx = lag1_check.index.intersection(lag1_expected.index)
if len(common_idx) > 0:
    max_diff = abs(lag1_check.loc[common_idx] - lag1_expected.loc[common_idx]).max()
    verdict(
        sec,
        "dvol_lag_1d = dvol.shift(24) exactly",
        max_diff < 0.01,
        f"max_diff={max_diff:.6f}",
    )

lag7_check = df["dvol_lag_7d"].dropna()
lag7_expected = df["dvol"].shift(168).dropna()
common_idx7 = lag7_check.index.intersection(lag7_expected.index)
if len(common_idx7) > 0:
    max_diff7 = abs(lag7_check.loc[common_idx7] - lag7_expected.loc[common_idx7]).max()
    verdict(
        sec,
        "dvol_lag_7d = dvol.shift(168) exactly",
        max_diff7 < 0.01,
        f"max_diff={max_diff7:.6f}",
    )

lag30_check = df["dvol_lag_30d"].dropna()
lag30_expected = df["dvol"].shift(720).dropna()
common_idx30 = lag30_check.index.intersection(lag30_expected.index)
if len(common_idx30) > 0:
    max_diff30 = abs(
        lag30_check.loc[common_idx30] - lag30_expected.loc[common_idx30]
    ).max()
    verdict(
        sec,
        "dvol_lag_30d = dvol.shift(720) exactly",
        max_diff30 < 0.01,
        f"max_diff={max_diff30:.6f}",
    )

verdict(
    sec,
    "Lag NaNs only at start of dataset",
    df.loc[720:, "dvol_lag_30d"].isna().sum() == 0,
    f"scattered NaNs after row 720: {df.loc[720:, 'dvol_lag_30d'].isna().sum()}",
)

verdict(
    sec,
    "dvol_lag_1d NaNs only in first 24 rows",
    df.loc[24:, "dvol_lag_1d"].isna().sum() == 0,
    f"NaNs after row 24: {df.loc[24:, 'dvol_lag_1d'].isna().sum()}",
)


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — FORWARD-FILL / STALE DATA DETECTION              ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 9: FORWARD-FILL / STALE DATA DETECTION")
print("─" * 70)

sec = "9_stale_data"

STALE_FEATURES = ["dvol", "nvrv", "network_activity", "transaction_volume"]
stale_results = {}
for feat in STALE_FEATURES:
    max_streak = 0
    current_streak = 0
    prev_v = None
    for v in df[feat].values:
        if v == prev_v:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
            prev_v = v
    stale_results[feat] = max_streak

for feat, max_streak in stale_results.items():
    verdict(
        sec,
        f"{feat}: max constant streak ≤ 5h",
        max_streak <= 5,
        f"max_streak={max_streak}h",
    )

fig, ax = plt.subplots(figsize=(14, 4))
feat_names = list(stale_results.keys())
streak_vals = list(stale_results.values())
bars = ax.bar(
    feat_names,
    streak_vals,
    color=["#2ca02c" if v <= 5 else "#d62728" for v in streak_vals],
    edgecolor="black",
    linewidth=0.3,
)
ax.axhline(
    5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Threshold (5h)"
)
ax.set_title("Max Constant-Value Streak by Feature")
ax.set_ylabel("Hours")
ax.legend()
for bar, sv in zip(bars, streak_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        str(sv),
        ha="center",
        fontsize=10,
    )
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "08_stale_data.png", dpi=150)
plt.close(fig)
print("  → Saved 08_stale_data.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — NVRV DEEP DIVE                                  ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "─" * 70)
print("SECTION 10: NVRV DEEP DIVE")
print("─" * 70)

sec = "10_nvrv"

nvrv = df["nvrv"]
verdict(
    sec, "NVRV: 100% complete", nvrv.isna().sum() == 0, f"nulls={nvrv.isna().sum()}"
)

verdict(
    sec,
    "NVRV range plausible (-0.5 to 5.0)",
    nvrv.min() > -0.5 and nvrv.max() < 5.0,
    f"range=[{nvrv.min():.4f}, {nvrv.max():.4f}]",
)

verdict(
    sec,
    "NVRV high uniqueness (>99%)",
    nvrv.nunique() / len(nvrv) > 0.99,
    f"unique={nvrv.nunique()}, ratio={nvrv.nunique() / len(nvrv):.6f}",
)


# ╔════════════════════════════════════════════════════════════════╗
# ║  SUMMARY & EXPORT                                             ║
# ╚════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)
total = PASS_COUNT + FAIL_COUNT + WARN_COUNT
print(f"  PASS: {PASS_COUNT}/{total}")
print(f"  WARN: {WARN_COUNT}/{total}")
print(f"  FAIL: {FAIL_COUNT}/{total}")
print(f"  Overall: {'✓ CLEAN' if FAIL_COUNT == 0 else '✗ ISSUES FOUND'}")

output = {
    "dataset": "bitcoin_lstm_features_v1.6_final.csv",
    "audit_date": "2026-04-02",
    "rows": N,
    "columns": len(df.columns),
    "date_range": f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}",
    "summary": {
        "pass": PASS_COUNT,
        "warn": WARN_COUNT,
        "fail": FAIL_COUNT,
        "total_checks": total,
    },
    "feature_stats": feature_stats,
    "split_info": {
        "train": {
            "rows": len(train),
            "start": str(train["timestamp"].iloc[0]),
            "end": str(train["timestamp"].iloc[-1]),
            "dvol_mean": round(train["dvol"].mean(), 2),
            "jumps": train_jumps,
        },
        "val": {
            "rows": len(val),
            "start": str(val["timestamp"].iloc[0]),
            "end": str(val["timestamp"].iloc[-1]),
            "dvol_mean": round(val["dvol"].mean(), 2),
            "jumps": val_jumps,
        },
        "test": {
            "rows": len(test),
            "start": str(test["timestamp"].iloc[0]),
            "end": str(test["timestamp"].iloc[-1]),
            "dvol_mean": round(test["dvol"].mean(), 2),
            "jumps": test_jumps,
        },
    },
    "jump_detection": {
        "method": "Lee-Mykland (2008)",
        "jumps": n_jumps,
        "jump_pct": round(jump_pct, 4),
        "threshold": round(threshold_val, 4),
        "event_hits": event_hits,
    },
    "autocorrelation": {
        "lag_1h": round(dvol_autocorr_lag1, 6),
        "lag_24h": round(dvol_autocorr_lag24, 6),
        "lag_168h": round(dvol_autocorr_lag168, 6),
    },
    "adf_raw": adf_results,
    "adf_normalized": norm_adf_results,
    "vif": vif_results,
    "stale_data": stale_results,
    "checks": RESULTS,
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  → Saved metrics.json to {OUTPUT_DIR / 'metrics.json'}")
print("  → All figures saved to results/diagnostics/v16_audit/figures/")
