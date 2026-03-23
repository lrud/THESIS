#!/usr/bin/env python3
"""
Generate a comprehensive comparison table PNG for regression vs classification results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

# Set up figure - more height for title space
fig, ax = plt.subplots(figsize=(18, 14))
ax.axis('off')

# Title - positioned at top with more padding
fig.text(0.5, 0.97, 'Comprehensive Model Comparison: Regression vs Classification',
         ha='center', va='top', fontsize=18, weight='bold')
fig.text(0.5, 0.945, 'Bitcoin DVOL Forecasting (v1.6_final: 41,055 samples, 2021-04-23 to 2025-12-28)',
         ha='center', va='top', fontsize=10, style='italic')
fig.text(0.5, 0.92, 'Regression: Level Forecast vs Classification: Binary Direction',
         ha='center', va='top', fontsize=9, style='italic')

# Table data
table_data = [
    # Regression - Level Forecast
    ['REGRESSION - Level Forecast', '', '', '', '', ''],
    ['Model', 'Features', 'R2', 'RMSE', 'Dir%', 'vs Random'],
    ['XGB_NoLag_Jumps', '8', '0.9940', '0.53', '49.3%', '-0.7pp'],
    ['XGB_Lags_Jumps', '11', '0.9936', '0.55', '49.1%', '-0.9pp'],
    ['RF_NoLag_Jumps', '8', '0.9935', '0.55', '49.5%', '-0.5pp'],
    ['RF_Lags_Jumps', '11', '0.9935', '0.55', '49.3%', '-0.7pp'],
    ['RF_NoLag', '4', '0.9927', '0.58', '49.7%', '-0.3pp'],
    ['XGB_Lags', '7', '0.9927', '0.58', '49.5%', '-0.5pp'],
    ['XGB_NoLag', '4', '0.9927', '0.58', '49.5%', '-0.5pp'],
    ['RF_Lags', '7', '0.9927', '0.59', '49.4%', '-0.6pp'],
    ['', '', '', '', '', ''],
    # Regression - Best Directional
    ['REGRESSION - Best Directional', '', '', '', '', ''],
    ['Model', 'Window', 'R2', 'RMSE', 'Dir%', 'vs Random'],
    ['HAR_RV', '168h', '0.9511', '1.51', '50.8%', '+0.8pp'],
    ['HAR_RV', '336h', '0.9441', '1.62', '50.7%', '+0.7pp'],
    ['HAR_RV', '72h', '0.9592', '1.38', '50.3%', '+0.3pp'],
    ['HAR_RV', '720h', '0.9389', '1.69', '50.3%', '+0.3pp'],
    ['', '', '', '', '', ''],
    # Classification - Binary Direction
    ['CLASSIFICATION - Binary Direction', '', '', '', '', ''],
    ['Model', 'Type', 'Accuracy', 'F1', 'PT-stat', 'Sig'],
    ['LDA_HAR', 'Linear', '54.29%', '0.0000', '-0.04', ''],
    ['LDA_NoLags', 'Linear', '53.97%', '0.0857', '0.11', ''],
    ['LDA_WithLags_Jumps', 'Linear', '53.84%', '0.1359', '0.35', ''],
    ['LDA_NoLags_Jumps', 'Linear', '53.79%', '0.1009', '-0.06', ''],
    ['XGB_NoLag', 'Tree', '52.96%', '0.3557', '1.82', '*'],
    ['XGB_NoLag_Jumps', 'Tree', '52.70%', '0.3621', '1.53', ''],
    ['XGB_Lags', 'Tree', '52.19%', '0.3562', '0.63', ''],
    ['RF_NoLag', 'Tree', '51.39%', '0.4610', '1.66', '*'],
    ['Logistic_NoLags', 'Linear', '50.29%', '0.4878', '0.74', ''],
    ['', '', '', '', '', ''],
    # Multi-Window Classification
    ['MULTI-WINDOW CLASSIFICATION', '', '', '', '', ''],
    ['Window', 'Best Model', 'Accuracy', 'F1', 'Avg Acc', 'vs Random'],
    ['72h (3d)', 'LDA_NoLags_Jumps', '54.62%', '0.0886', '53.87%', '+3.87pp'],
    ['168h (7d)', 'Logistic_NoLags', '54.43%', '0.0756', '53.71%', '+3.71pp'],
    ['336h (14d)', 'Logistic_NoLags', '54.35%', '0.0841', '53.75%', '+3.75pp'],
    ['720h (30d)', 'LDA_NoLags', '54.43%', '0.0783', '53.62%', '+3.62pp'],
]

# Track sections and significant rows
current_section = None
cell_colors = []
significant_rows = []  # Track which rows to highlight
best_rows = []  # Track best performers in each section

# Section row indices (for reference)
section_indices = {
    'REGRESSION - Level': None,
    'REGRESSION - Best Directional': None,
    'CLASSIFICATION': None,
    'MULTI-WINDOW': None
}

for i, row in enumerate(table_data):
    # Identify section headers
    if 'REGRESSION - Level' in row[0]:
        current_section = 'REGRESSION - Level'
        section_indices['REGRESSION - Level'] = i
        cell_colors.append(['#1f77b4'] * 6)  # Blue header
    elif 'REGRESSION - Best Directional' in row[0]:
        current_section = 'REGRESSION - Best Directional'
        section_indices['REGRESSION - Best Directional'] = i
        cell_colors.append(['#2ca02c'] * 6)  # Green header
    elif 'CLASSIFICATION - Binary Direction' in row[0]:
        current_section = 'CLASSIFICATION'
        section_indices['CLASSIFICATION'] = i
        cell_colors.append(['#ff7f0e'] * 6)  # Orange header
    elif 'MULTI-WINDOW CLASSIFICATION' in row[0]:
        current_section = 'MULTI-WINDOW'
        section_indices['MULTI-WINDOW'] = i
        cell_colors.append(['#9467bd'] * 6)  # Purple header
    elif row[0] == 'Model' or row[0] == 'Window':
        cell_colors.append(['#d0d0d0'] * 6)  # Gray column header
    elif row[0] == '':
        cell_colors.append(['#f0f0f0'] * 6)  # Light gray separator
    else:
        # Data rows - apply light tint based on current section
        if current_section == 'REGRESSION - Level':
            cell_colors.append(['#e6f2ff'] * 6)  # Light blue
        elif current_section == 'REGRESSION - Best Directional':
            cell_colors.append(['#e6ffe6'] * 6)  # Light green
        elif current_section == 'CLASSIFICATION':
            cell_colors.append(['#fff0e6'] * 6)  # Light orange
        elif current_section == 'MULTI-WINDOW':
            cell_colors.append(['#f0e6ff'] * 6)  # Light purple
        else:
            cell_colors.append(['#ffffff'] * 6)  # White default

    # Track significant rows for highlighting (rows with '*' in last column)
    if row[-1] == '*':
        significant_rows.append(i)

# Create table - positioned lower to leave room for title
table = plt.table(cellText=table_data, cellLoc='left', colWidths=[0.20, 0.12, 0.11, 0.11, 0.11, 0.12],
                  bbox=[0.02, 0.30, 0.98, 0.73], cellColours=cell_colors)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# Style section headers
section_rows = [i for i, row in enumerate(table_data) if any(x in row[0] for x in
                ['REGRESSION', 'CLASSIFICATION', 'MULTI-WINDOW'])]

for i in section_rows:
    for j in range(6):
        cell = table[(i, j)]
        cell.set_facecolor('#bbbbbb')
        cell.set_text_props(weight='bold')

# Style column headers
for i, row in enumerate(table_data):
    if row[0] in ['Model', 'Window']:
        for j in range(6):
            cell = table[(i, j)]
            cell.set_facecolor('#555555')
            cell.set_text_props(weight='bold', color='white')

# Highlight significant rows (yellow background, bold red text)
for row_idx in significant_rows:
    for j in range(6):
        cell = table[(row_idx, j)]
        cell.set_facecolor('#ffeb99')
        cell.set_text_props(weight='bold')
    # Make the Sig column bold red
    cell = table[(row_idx, 5)]
    cell.set_text_props(weight='bold', color='#d62728')

# Key insights section
insight_y = 0.26
fig.text(0.02, insight_y, 'KEY INSIGHTS:', fontsize=11, weight='bold')
insight_y -= 0.014

insights = [
    '1. REGRESSION: High R2 (99%+) from DVOL autocorrelation (~0.999)',
    '   Best: XGB_NoLag_Jumps R2=0.9940, but Dir%=49.3% (worse than random!)',
    '',
    '2. REGRESSION: HAR_RV wins for directional accuracy',
    '   Best: HAR_RV@168h with Dir%=50.8% (+0.8pp vs random)',
    '',
    '3. CLASSIFICATION: No statistical significance at 5% level',
    '   Only 2 models marginally significant (p<0.10): RF_NoLag, XGB_NoLag',
    '',
    '4. CLASSIFICATION: LDA models are degenerate',
    '   High accuracy (54%) but F1~0 (predicts majority class only)',
    '',
    '5. CRITICAL: Hourly DVOL direction is fundamentally unpredictable',
    '   Best models cannot statistically beat random guessing',
    '',
    '6. WINDOW: 72h optimal for R2; minimal impact on classification',
]

for line in insights:
    fig.text(0.04, insight_y, line, fontsize=9)
    insight_y -= 0.011

# Dataset specs at bottom
fig.text(0.02, 0.010,
         'Dataset: v1.6_final | Samples: 41,055 (hourly) | Period: 2021-04-23 to 2025-12-28 | Split: 60/20/20 | Baseline: 50%',
         fontsize=8, style='italic')

# Legend
legend_elements = [
    mpatches.Patch(color='#1f77b4', label='Regression (Level)'),
    mpatches.Patch(color='#2ca02c', label='Regression (Best Dir%)'),
    mpatches.Patch(color='#ff7f0e', label='Classification'),
    mpatches.Patch(color='#9467bd', label='Multi-Window'),
    mpatches.Patch(color='#ffeb99', label='Marginal Sig (p<0.10)'),
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
           bbox_to_anchor=(0.98, 0.01), ncol=5)

plt.tight_layout()
plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)

# Save PNG
output_png = 'results/visualizations/comprehensive_model_comparison.png'
import os
os.makedirs(os.path.dirname(output_png), exist_ok=True)
plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
print(f"PNG saved: {output_png}")
plt.close()

# Export CSV for Excel
csv_path = 'results/visualizations/comprehensive_model_comparison.csv'
import csv

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)

    # Write main data
    writer.writerow(['Section', 'Model', 'Type/Features/Window', 'Metric1', 'Metric2', 'Metric3', 'Metric4', 'Metric5'])

    # Regression - Level
    writer.writerow(['REGRESSION - Level Forecast', '', '', '', '', '', '', ''])
    writer.writerow(['Model', 'Features', 'R2', 'RMSE', 'Dir%', 'vs Random', ''])
    for row in table_data[2:10]:
        writer.writerow([''] + row)

    writer.writerow([])

    # Regression - Best Directional
    writer.writerow(['REGRESSION - Best Directional', '', '', '', '', '', ''])
    writer.writerow(['Model', 'Window', 'R2', 'RMSE', 'Dir%', 'vs Random', ''])
    for row in table_data[12:16]:
        writer.writerow([''] + row)

    writer.writerow([])

    # Classification
    writer.writerow(['CLASSIFICATION - Binary Direction', '', '', '', '', '', ''])
    writer.writerow(['Model', 'Type', 'Accuracy', 'F1', 'PT-stat', 'Sig', ''])
    for row in table_data[18:27]:
        writer.writerow([''] + row)

    writer.writerow([])

    # Multi-Window
    writer.writerow(['MULTI-WINDOW CLASSIFICATION', '', '', '', '', '', ''])
    writer.writerow(['Window', 'Best Model', 'Accuracy', 'F1', 'Avg Acc', 'vs Random', ''])
    for row in table_data[29:33]:
        writer.writerow([''] + row)

    writer.writerow([])
    writer.writerow(['Key Insights', '', '', '', '', '', ''])
    for insight in insights:
        writer.writerow([insight, '', '', '', '', '', ''])

print(f"CSV saved: {csv_path}")
