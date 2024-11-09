import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd

def generate_figure3(conn, pdf):
    """
    Figure 3: Seasonal Weather Patterns and Environmental Variables
    Multi-panel time series plots for precipitation, temperature, solar radiation, wind speed, and VPD.
    Enhanced with growth stage overlays and irrigation event markers.
    """
    label = "Figure 3: Seasonal Weather Patterns and Environmental Variables"
    print(f"Generating {label}")

    # Extract weather data
    weather_query = """
    SELECT date(timestamp) as date, variable_name, value
    FROM data
    WHERE variable_name IN ('Rain_1m_Tot', 'TaMax_2m', 'TaMin_2m', 'Ta_2m_Avg', 'RH_2m_Avg', 'Solar_2m_Avg', 'WndAveSpd_3m')
    """
    weather_df = pd.read_sql_query(weather_query, conn)
    weather_df = filter_valid_dates(weather_df, 'date')
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Pivot data to wide format
    weather_pivot = weather_df.pivot_table(index='date', columns='variable_name', values='value', aggfunc='mean').reset_index()

    # Extract irrigation events
    irrigation_query = """
    SELECT date(date) as date, SUM(amount_mm) as irrigation_mm
    FROM irrigation_events
    GROUP BY date(date)
    """
    irrigation_df = pd.read_sql_query(irrigation_query, conn)
    irrigation_df = filter_valid_dates(irrigation_df, 'date')
    irrigation_df['date'] = pd.to_datetime(irrigation_df['date'])

    # Merge datasets
    data_all = pd.merge(weather_pivot, irrigation_df, on='date', how='left')
    data_all['irrigation_mm'] = data_all['irrigation_mm'].fillna(0)

    # Calculate VPD
    data_all['VPD'] = calc_vpd(data_all['Ta_2m_Avg'], data_all['RH_2m_Avg'])

    # Define growth stages
    growth_stages = [
        ('V12', '2024-07-01'),  # When CWSI monitoring started
        ('VT', '2024-07-15'),   # Before reproductive stage
        ('R1', '2024-07-20'),
        ('R2', '2024-07-24'),
        ('R3', '2024-08-01'),
        ('R4', '2024-08-07'),
        ('R5', '2024-08-16')
    ]

    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(15, 25), sharex=True)

    # Subplot 1: Precipitation and Irrigation
    ax = axes[0]
    ax.bar(data_all['date'], data_all['Rain_1m_Tot'], 
           label='Rainfall (mm)', color=CUSTOM_COLORS['primary_blue'], alpha=0.7, edgecolor='black')
    ax.bar(data_all['date'], data_all['irrigation_mm'], 
           bottom=data_all['Rain_1m_Tot'], label='Irrigation (mm)', 
           color=CUSTOM_COLORS['accent_green'], alpha=0.7, edgecolor='black')
    style_axis(ax, 
              title='Daily Precipitation and Irrigation Events',
              ylabel='Amount (mm)')
    style_legend(ax, loc='upper right')

    # Subplot 2: Temperature
    ax = axes[1]
    ax.plot(data_all['date'], data_all['TaMax_2m'], 
            label='Maximum Temp (°C)', color=CUSTOM_COLORS['accent_orange'], linewidth=2.5)
    ax.plot(data_all['date'], data_all['TaMin_2m'], 
            label='Minimum Temp (°C)', color=CUSTOM_COLORS['primary_blue'], linewidth=2.5)
    ax.fill_between(data_all['date'], data_all['TaMin_2m'], data_all['TaMax_2m'],
                   color=CUSTOM_COLORS['accent_orange'], alpha=0.1)
    style_axis(ax, 
              title='Daily Air Temperature Range',
              ylabel='Temperature (°C)')
    style_legend(ax, title='Temperature', loc='upper right')

    # Subplot 3: Solar Radiation
    ax = axes[2]
    ax.plot(data_all['date'], data_all['Solar_2m_Avg'], 
            label='Solar Radiation (W/m²)', color=CUSTOM_COLORS['accent_purple'], linewidth=2.5)
    style_axis(ax, 
              title='Daily Solar Radiation',
              ylabel='Solar Radiation (W/m²)')
    style_legend(ax, loc='upper right')

    # Subplot 4: Wind Speed
    ax = axes[3]
    ax.plot(data_all['date'], data_all['WndAveSpd_3m'], 
            label='Wind Speed (m/s)', color=CUSTOM_COLORS['secondary_blue'], linewidth=2.5)
    style_axis(ax, 
              title='Daily Wind Speed',
              ylabel='Wind Speed (m/s)')
    style_legend(ax, loc='upper right')

    # Subplot 5: VPD
    ax = axes[4]
    ax.plot(data_all['date'], data_all['VPD'], 
            label='Vapor Pressure Deficit (VPD)', color=CUSTOM_COLORS['neutral_gray'], linewidth=2.5)
    style_axis(ax, 
              title='Daily Vapor Pressure Deficit (VPD)',
              ylabel='VPD (kPa)')
    style_legend(ax, loc='upper right')

    # Add growth stage overlays and irrigation event markers
    for stage, date in growth_stages:
        stage_date = pd.to_datetime(date)
        for ax in axes:
            ax.axvline(x=stage_date, color='gray', linestyle='--', alpha=0.5)
            ax.text(stage_date, ax.get_ylim()[1], stage, rotation=90, verticalalignment='bottom', fontsize=12, color='gray')

    # Highlight key irrigation events (e.g., July 26 fertigation)
    fertigation_date = pd.to_datetime('2024-07-26')
    for ax in axes:
        ax.axvline(x=fertigation_date, color='blue', linestyle='-.', alpha=0.7)
        ax.text(fertigation_date, ax.get_ylim()[1], 'Fertigation', rotation=90, verticalalignment='bottom', fontsize=12, color='blue')

    # Adjust layout to make room for the caption
    plt.subplots_adjust(top=0.95)
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"{label} added to PDF.")
