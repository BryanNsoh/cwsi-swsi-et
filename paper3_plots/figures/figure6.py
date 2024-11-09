import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd
import numpy as np

def generate_figure6(conn, pdf):
    """
    Figure 6: Combined Index Analysis and Weighting System
    Analysis of combined CWSI and SWSI indices with weighting and decision thresholds.
    """
    label = "Figure 6: Combined Index Analysis and Weighting System"
    print(f"Generating {label}")

    # Extract CWSI data
    cwsi_query = """
    SELECT d.timestamp, d.plot_id, d.value as cwsi
    FROM data d
    WHERE d.variable_name = 'cwsi'
    """
    cwsi_df = pd.read_sql_query(cwsi_query, conn)
    cwsi_df = filter_valid_dates(cwsi_df, 'timestamp')
    cwsi_df['timestamp'] = pd.to_datetime(cwsi_df['timestamp'])
    cwsi_df['date'] = cwsi_df['timestamp'].dt.date

    # Extract SWSI data
    swsi_query = """
    SELECT d.timestamp, d.plot_id, d.value as swsi
    FROM data d
    WHERE d.variable_name = 'swsi'
    """
    swsi_df = pd.read_sql_query(swsi_query, conn)
    swsi_df = filter_valid_dates(swsi_df, 'timestamp')
    swsi_df['timestamp'] = pd.to_datetime(swsi_df['timestamp'])
    swsi_df['date'] = swsi_df['timestamp'].dt.date

    # Aggregate SWSI by date
    swsi_daily = swsi_df.groupby('date')['swsi'].mean().reset_index()

    # Merge CWSI and SWSI
    combined_df = pd.merge(cwsi_df, swsi_daily, on='date', how='left')

    # Calculate weighted combined index (60% SWSI, 40% CWSI)
    combined_df['combined_index'] = 0.6 * combined_df['swsi'] + 0.4 * combined_df['cwsi']

    # Remove rows with NaN values in combined_index
    combined_df = combined_df.dropna(subset=['combined_index'])

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)

    # Panel (a): CWSI vs SWSI correlation analysis
    sns.scatterplot(x='swsi', y='cwsi', data=combined_df, alpha=0.5, ax=axes[0], color=CUSTOM_COLORS['accent_purple'])
    # Calculate and plot regression line
    sns.regplot(x='swsi', y='cwsi', data=combined_df, scatter=False, ax=axes[0], color=CUSTOM_COLORS['red'], line_kws={'linewidth':2})
    corr = combined_df[['swsi', 'cwsi']].corr().iloc[0,1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=axes[0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
    style_axis(axes[0], 
              title='Correlation between SWSI and CWSI',
              xlabel='Soil Water Stress Index (SWSI)',
              ylabel='Crop Water Stress Index (CWSI)')
    style_legend(axes[0], loc='upper left')

    # Panel (b): Time series of both indices with weighted average overlay
    combined_daily = combined_df.groupby('date').agg({'swsi': 'mean', 'cwsi': 'mean', 'combined_index': 'mean'}).reset_index()
    sns.lineplot(x='date', y='swsi', data=combined_daily, label='SWSI', color=CUSTOM_COLORS['blue'], linewidth=2.5, ax=axes[1])
    sns.lineplot(x='date', y='cwsi', data=combined_daily, label='CWSI', color=CUSTOM_COLORS['green'], linewidth=2.5, ax=axes[1])
    sns.lineplot(x='date', y='combined_index', data=combined_daily, label='Combined Index (60% SWSI, 40% CWSI)', color=CUSTOM_COLORS['accent_purple'], linewidth=2.5, ax=axes[1])
    style_axis(axes[1], 
              title='Time Series of SWSI, CWSI, and Combined Index',
              ylabel='Index Value')
    style_legend(axes[1], loc='upper right')

    # Panel (c): Decision threshold analysis
    threshold = 0.7
    sns.lineplot(x='date', y='combined_index', data=combined_daily, label='Combined Index', color=CUSTOM_COLORS['accent_purple'], linewidth=2.5, ax=axes[2])
    axes[2].axhline(y=threshold, color=CUSTOM_COLORS['red'], linestyle='--', label=f'Irrigation Threshold ({threshold})')
    style_axis(axes[2], 
              title='Irrigation Decision Threshold Based on Combined Index',
              xlabel='Date',
              ylabel='Combined Index Value')
    style_legend(axes[2], loc='upper right')

    # Adjust layout to make room for the caption
    plt.subplots_adjust(top=0.95)
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 6: Combined Index Analysis and Weighting System added to PDF.")
