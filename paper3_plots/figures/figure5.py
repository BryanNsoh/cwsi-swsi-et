import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd
import numpy as np

def generate_figure5(conn, pdf):
    """
    Figure 5: Soil Moisture Monitoring and SWSI Calculation
    Time series plots of soil moisture at different depths and Soil Water Stress Index (SWSI).
    """
    label = "Figure 5: Soil Moisture Monitoring and SWSI Calculation"
    print(f"Generating {label}")

    # Extract TDR soil moisture data
    tdr_query = """
    SELECT d.timestamp, d.plot_id, d.variable_name, d.value
    FROM data d
    WHERE d.variable_name LIKE 'TDR%'
    """
    tdr_df = pd.read_sql_query(tdr_query, conn)
    tdr_df = filter_valid_dates(tdr_df, 'timestamp')
    tdr_df['timestamp'] = pd.to_datetime(tdr_df['timestamp'])
    tdr_df['date'] = tdr_df['timestamp'].dt.date

    # Extract depth information from variable_name
    # Example variable_name: TDR5001A20624
    tdr_df['Depth_cm'] = tdr_df['variable_name'].str.extract(r'TDR\d{4}[A-C]\d(\d{2})\d{2}$')[0]

    # Handle cases where Depth_cm is NaN (no match)
    tdr_df['Depth_cm'] = tdr_df['Depth_cm'].fillna('00')  # Assign '00' for non-applicable sensors

    # Convert to numeric, handle '00' as NaN
    tdr_df['Depth_cm'] = pd.to_numeric(tdr_df['Depth_cm'], errors='coerce')
    tdr_df = tdr_df.dropna(subset=['Depth_cm'])

    # Convert Depth_cm to integer
    tdr_df['Depth_cm'] = tdr_df['Depth_cm'].astype(int)

    # Aggregate soil moisture by depth and date
    soil_moisture = tdr_df.groupby(['date', 'Depth_cm'])['value'].mean().reset_index()

    # Plot soil moisture over time at different depths
    fig, ax = plt.subplots(figsize=(15, 8))
    for depth in sorted(soil_moisture['Depth_cm'].unique()):
        depth_data = soil_moisture[soil_moisture['Depth_cm'] == depth]
        sns.lineplot(x='date', y='value', data=depth_data, label=f'Depth {depth} cm', linewidth=2.5, ax=ax)
    style_axis(ax, 
              title='Soil Moisture at Different Depths Over Time',
              xlabel='Date',
              ylabel='Soil Moisture (%)')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 5a: Soil Moisture at Different Depths Over Time added to PDF.")

    # Calculate SWSI (Assuming SWSI = normalized soil moisture)
    # Normalize soil moisture across all depths and dates
    min_moisture = soil_moisture['value'].min()
    max_moisture = soil_moisture['value'].max()
    soil_moisture['SWSI'] = (soil_moisture['value'] - min_moisture) / (max_moisture - min_moisture)

    # Aggregate SWSI by date
    swsi_daily = soil_moisture.groupby('date')['SWSI'].mean().reset_index()

    # Plot SWSI over time with uncertainty bands (e.g., ±0.05)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(x='date', y='SWSI', data=swsi_daily, label='SWSI', color=CUSTOM_COLORS['brown'], linewidth=2.5, ax=ax)
    ax.fill_between(swsi_daily['date'], swsi_daily['SWSI'] - 0.05, swsi_daily['SWSI'] + 0.05, color=CUSTOM_COLORS['brown'], alpha=0.2)
    style_axis(ax, 
              title='Soil Water Stress Index Over Time',
              xlabel='Date',
              ylabel='Soil Water Stress Index (SWSI)')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 5b: Soil Water Stress Index Over Time added to PDF.")
