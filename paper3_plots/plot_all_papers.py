# irrigation_graphs_generator.py

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages for PDF generation
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ================================
# Enhanced Plotting Configuration
# ================================

# Define a comprehensive color palette
CUSTOM_COLORS = {
    'primary_blue': '#2274A5',
    'secondary_blue': '#1B3B6F',
    'accent_green': '#32936F',
    'accent_orange': '#E83151',
    'accent_purple': '#6B4E71',
    'neutral_gray': '#4A4E69',
    'brown': '#A0522D',
    'red': '#E74C3C',
    'blue': '#3498DB',
    'green': '#2ECC71',
    'orange': '#E67E22',
    'purple': '#9B59B6',
    'light_gray': '#F8F9FA'
}

# Define a custom color palette for sequential plots
custom_palette = sns.color_palette([
    CUSTOM_COLORS['primary_blue'],
    CUSTOM_COLORS['accent_green'],
    CUSTOM_COLORS['accent_orange'],
    CUSTOM_COLORS['accent_purple'],
    CUSTOM_COLORS['secondary_blue']
])

# Set the custom palette as default
sns.set_palette(custom_palette)

# Enhanced plot styling
plt.rcParams.update({
    # Font sizes
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,

    # Font families
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],

    # Figure size and DPI
    'figure.figsize': (15, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,

    # Line widths
    'axes.linewidth': 2.5,
    'grid.linewidth': 1.0,
    'lines.linewidth': 2.5,

    # Grid styling
    'grid.alpha': 0.2,
    'grid.color': CUSTOM_COLORS['neutral_gray'],
    'axes.grid': True,  # Enable grid by default
    'grid.linestyle': '--',  # Dashed grid lines

    # Legend styling
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',

    # Background color
    'axes.facecolor': CUSTOM_COLORS['light_gray'],
    'figure.facecolor': CUSTOM_COLORS['light_gray'],

    # Spacing
    'figure.constrained_layout.use': True,  # Use constrained_layout for all figures
    'figure.autolayout': False,  # Disable when using constrained_layout
})

# Set the default seaborn style
sns.set_style("whitegrid")  # Use seaborn's whitegrid style instead of the deprecated style name

# Custom plotting functions for consistent styling
def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to axis."""
    if title:
        ax.set_title(title, pad=20, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)

    # Beef up the spines
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # Add subtle grid
    ax.grid(True, linestyle='--', alpha=0.2)

    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=2.5)

    return ax

def style_legend(ax, title=None, loc='upper right'):
    """Apply consistent legend styling."""
    if ax.get_legend():
        legend = ax.legend(title=title, loc=loc, frameon=True, fontsize=14, title_fontsize=16)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('black')
        if title:
            legend.get_title().set_fontsize(16)
    return ax

# ================================
# Database and Output Configuration
# ================================

# Define the path to the SQLite database
DATABASE_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\experiment_data_20241024.sqlite"

# Define the output directory for the PDF
OUTPUT_DIR = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\paper3_plots"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define the PDF output path
PDF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "all_figures.pdf")

# ================================
# Utility Functions
# ================================

def connect_db(db_path):
    """Establish a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database at {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def filter_valid_dates(df, date_column='date'):
    """Filter out records with invalid or out-of-range dates."""
    # Convert to datetime, coerce errors to NaT
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    initial_count = len(df)
    # Keep only dates from 2024 onwards
    df = df[df[date_column].dt.year >= 2024]
    filtered_count = len(df)
    print(f"Filtered dates: {initial_count - filtered_count} records removed due to invalid dates.")
    return df

# ================================
# Plotting Functions
# ================================

def generate_figure3(conn, pdf):
    """
    Figure 3: Seasonal Weather Patterns and Environmental Variables
    Multi-panel time series plots for precipitation, temperature, solar radiation, wind speed, and VPD.
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

    # Debug: Print date range
    print(f"Figure 3 - Weather Data Date Range: {weather_df['date'].min()} to {weather_df['date'].max()}")

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

    # Debug: Print irrigation data date range
    print(f"Figure 3 - Irrigation Data Date Range: {irrigation_df['date'].min()} to {irrigation_df['date'].max()}")

    # Merge datasets
    data_all = pd.merge(weather_pivot, irrigation_df, on='date', how='left')
    data_all['irrigation_mm'] = data_all['irrigation_mm'].fillna(0)

    # Calculate VPD
    def calc_vpd(temp, rh):
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        vpd = es * (1 - rh / 100)
        return vpd

    data_all['VPD'] = calc_vpd(data_all['Ta_2m_Avg'], data_all['RH_2m_Avg'])

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

    # Adjust layout to make room for the caption
    plt.subplots_adjust(top=0.95)
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"{label} added to PDF.")

def generate_figure4(conn, pdf):
    """
    Figure 4: CWSI Implementation Analysis
    Four-panel figure analyzing Crop Water Stress Index (CWSI) for corn and soybeans.
    """
    label = "Figure 4: CWSI Implementation Analysis"
    print(f"Generating {label}")

    # Extract CWSI data and plot information
    cwsi_query = """
    SELECT d.timestamp, d.plot_id, d.value as cwsi, p.field
    FROM data d
    JOIN plots p ON d.plot_id = p.plot_id
    WHERE d.variable_name = 'cwsi'
    """
    cwsi_df = pd.read_sql_query(cwsi_query, conn)
    cwsi_df = filter_valid_dates(cwsi_df, 'timestamp')
    cwsi_df['timestamp'] = pd.to_datetime(cwsi_df['timestamp'])
    cwsi_df['date'] = cwsi_df['timestamp'].dt.date

    # Debug: Print CWSI data date range
    print(f"Figure 4 - CWSI Data Date Range: {cwsi_df['timestamp'].min()} to {cwsi_df['timestamp'].max()}")

    # Separate corn and soybean data
    corn_cwsi = cwsi_df[cwsi_df['field'] == 'LINEAR_CORN'].copy()
    soybean_cwsi = cwsi_df[cwsi_df['field'] == 'LINEAR_SOYBEAN'].copy()

    # Define growth stages for corn based on provided data
    corn_growth_stages = [
        ('Emergence', '2024-04-24', '2024-05-12'),
        ('VE', '2024-05-12', '2024-05-16'),
        ('V1', '2024-05-16', '2024-05-20'),
        ('V2', '2024-05-20', '2024-05-29'),
        ('V3', '2024-05-29', '2024-06-02'),
        ('V4', '2024-06-02', '2024-06-07'),
        ('V5', '2024-06-07', '2024-06-11'),
        ('V6', '2024-06-11', '2024-06-13'),
        ('V7', '2024-06-13', '2024-06-17'),
        ('V8', '2024-06-17', '2024-06-20'),
        ('V9', '2024-06-20', '2024-06-24'),
        ('V10', '2024-06-24', '2024-06-27'),
        ('V11', '2024-06-27', '2024-06-29'),
        ('V12', '2024-06-29', '2024-07-05'),
        ('V13', '2024-07-05', '2024-07-08'),
        ('V14', '2024-07-08', '2024-07-11'),
        ('VT/R1', '2024-07-11', '2024-07-20'),
        ('R2', '2024-07-20', '2024-07-24'),
        ('R3', '2024-07-24', '2024-08-01'),
        ('R4', '2024-08-01', '2024-08-07'),
        ('R5', '2024-08-07', '2024-08-16'),
        ('R5.25', '2024-08-16', '2024-08-24'),
        ('R5.5', '2024-08-24', '2024-08-28')
    ]

    def assign_corn_stage(row):
        for stage, start, end in corn_growth_stages:
            if pd.to_datetime(start) <= row['timestamp'] <= pd.to_datetime(end):
                return stage
        return 'Unknown'

    # Assign growth stages
    corn_cwsi = corn_cwsi.copy()
    corn_cwsi['Growth Stage'] = corn_cwsi.apply(assign_corn_stage, axis=1)

    # Panel (a): Daily CWSI for corn with theoretical bounds
    corn_daily = corn_cwsi.groupby('date')['cwsi'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(corn_daily['date'], corn_daily['cwsi'], label='Corn CWSI', color=CUSTOM_COLORS['accent_green'], linewidth=2.5)
    ax.axhline(y=0, color=CUSTOM_COLORS['red'], linestyle='--', label='Theoretical Lower Bound')
    ax.axhline(y=1, color=CUSTOM_COLORS['red'], linestyle='--', label='Theoretical Upper Bound')
    style_axis(ax, 
              title='Daily CWSI Values for Corn',
              ylabel='CWSI')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 4a: Daily CWSI for Corn added to PDF.")

    # Panel (b): Daily CWSI for soybeans with transition point
    label_b = "Figure 4b: Daily CWSI for Soybeans with Transition Point"
    print(f"Generating {label_b}")

    soybean_daily = soybean_cwsi.groupby('date')['cwsi'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(soybean_daily['date'], soybean_daily['cwsi'], label='Soybean CWSI', color=CUSTOM_COLORS['primary_blue'], linewidth=2.5)
    ax.axhline(y=0, color=CUSTOM_COLORS['red'], linestyle='--', label='Theoretical Lower Bound')
    ax.axhline(y=1, color=CUSTOM_COLORS['red'], linestyle='--', label='Theoretical Upper Bound')
    transition_date = pd.to_datetime('2024-08-01')
    ax.axvline(transition_date, color=CUSTOM_COLORS['accent_purple'], linestyle='-.', label='Transition to Empirical Method')
    style_axis(ax, 
              title='Daily CWSI Values for Soybeans',
              ylabel='CWSI')
    style_legend(ax, loc='upper right')
    fig.suptitle(label_b, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 4b: Daily CWSI for Soybeans added to PDF.")

    # Panel (c): Boxplots of CWSI distribution by growth stage (Corn)
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='Growth Stage', y='cwsi', data=corn_cwsi, 
               order=[stage for stage, _, _ in corn_growth_stages], 
               palette='viridis', ax=ax)
    
    style_axis(ax, 
              title='CWSI Distribution by Growth Stage (Corn)',
              xlabel='Growth Stage',
              ylabel='CWSI')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add the main title
    fig.suptitle(label, fontsize=24, fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 4c: CWSI Distribution by Growth Stage (Corn) added to PDF.")

    # Panel (d): Scatterplot of theoretical vs empirical CWSI
    # Extract temperature data for theoretical CWSI calculation
    temp_query = """
    SELECT d.timestamp, d.plot_id, d.value as temperature
    FROM data d
    WHERE d.variable_name = 'Ta_2m_Avg'
    """
    temp_df = pd.read_sql_query(temp_query, conn)
    temp_df = filter_valid_dates(temp_df, 'timestamp')
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    
    # Merge with CWSI data
    cwsi_temp_df = pd.merge(cwsi_df, temp_df, on=['timestamp', 'plot_id'], how='left')
    
    # Calculate theoretical CWSI
    Tmin, Tmax = 15, 35  # Example Tmin and Tmax values
    cwsi_temp_df['cwsi_theoretical'] = (cwsi_temp_df['temperature'] - Tmin) / (Tmax - Tmin)
    cwsi_temp_df['cwsi_theoretical'] = cwsi_temp_df['cwsi_theoretical'].clip(0, 1)
    
    # Remove rows where theoretical CWSI is NaN (due to missing temperature)
    cwsi_temp_df = cwsi_temp_df.dropna(subset=['cwsi_theoretical', 'cwsi'])
    
    # Debug: Print theoretical vs empirical CWSI data range
    print(f"Figure 4 - Theoretical CWSI Data Range: {cwsi_temp_df['cwsi_theoretical'].min()} to {cwsi_temp_df['cwsi_theoretical'].max()}")
    print(f"Figure 4 - Empirical CWSI Data Range: {cwsi_temp_df['cwsi'].min()} to {cwsi_temp_df['cwsi'].max()}")
    
    # Scatterplot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='cwsi_theoretical', y='cwsi', data=cwsi_temp_df, alpha=0.5, ax=ax, color=CUSTOM_COLORS['accent_purple'])
    ax.plot([0, 1], [0, 1], color=CUSTOM_COLORS['red'], linestyle='--', label='1:1 Line')
    style_axis(ax, 
              title='Comparison of Theoretical vs Empirical CWSI',
              xlabel='Theoretical CWSI',
              ylabel='Empirical CWSI')
    style_legend(ax, loc='upper left')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 4d: Comparison of Theoretical vs Empirical CWSI added to PDF.")

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
    # Breakdown:
    # - TDR
    # - 5001 (plot number)
    # - A (node)
    # - 2 (treatment)
    # - 06 (depth)
    # - 24 (year)
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

    # Debug: Print soil moisture date range and depth levels
    print(f"Figure 5 - Soil Moisture Data Date Range: {soil_moisture['date'].min()} to {soil_moisture['date'].max()}")
    print(f"Figure 5 - Depth Levels: {sorted(soil_moisture['Depth_cm'].unique())} cm")

    # Plot soil moisture over time at different depths
    fig, ax = plt.subplots(figsize=(15, 8))
    for depth in sorted(soil_moisture['Depth_cm'].unique()):
        depth_data = soil_moisture[soil_moisture['Depth_cm'] == depth]
        ax.plot(depth_data['date'], depth_data['value'], label=f'Depth {depth} cm', linewidth=2.5)
    style_axis(ax, 
              title='Soil Moisture at Different Depths Over Time',
              xlabel='Date',
              ylabel='Soil Moisture (%)')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 5a: Soil Moisture at Different Depths Over Time added to PDF.")

    # Calculate SWSI (Assuming SWSI = normalized soil moisture)
    # Here, we normalize soil moisture across all depths and dates
    min_moisture = soil_moisture['value'].min()
    max_moisture = soil_moisture['value'].max()
    soil_moisture['SWSI'] = (soil_moisture['value'] - min_moisture) / (max_moisture - min_moisture)

    # Aggregate SWSI by date
    swsi_daily = soil_moisture.groupby('date')['SWSI'].mean().reset_index()

    # Debug: Print SWSI data date range
    print(f"Figure 5 - SWSI Data Date Range: {swsi_daily['date'].min()} to {swsi_daily['date'].max()}")

    # Plot SWSI over time with uncertainty bands (e.g., ±0.05)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(swsi_daily['date'], swsi_daily['SWSI'], label='SWSI', color=CUSTOM_COLORS['brown'], linewidth=2.5)
    ax.fill_between(swsi_daily['date'], swsi_daily['SWSI'] - 0.05, swsi_daily['SWSI'] + 0.05, color=CUSTOM_COLORS['brown'], alpha=0.2)
    style_axis(ax, 
              title='Soil Water Stress Index Over Time',
              xlabel='Date',
              ylabel='Soil Water Stress Index (SWSI)')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 5b: Soil Water Stress Index Over Time added to PDF.")

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

    # Extract SWSI data (assuming it's stored as 'swsi' in data table)
    swsi_query = """
    SELECT d.timestamp, d.plot_id, d.value as swsi
    FROM data d
    WHERE d.variable_name = 'swsi'
    """
    swsi_df = pd.read_sql_query(swsi_query, conn)
    swsi_df = filter_valid_dates(swsi_df, 'timestamp')
    swsi_df['timestamp'] = pd.to_datetime(swsi_df['timestamp'])
    swsi_df['date'] = swsi_df['timestamp'].dt.date

    # Debug: Print CWSI and SWSI date ranges
    print(f"Figure 6 - CWSI Data Date Range: {cwsi_df['timestamp'].min()} to {cwsi_df['timestamp'].max()}")
    print(f"Figure 6 - SWSI Data Date Range: {swsi_df['timestamp'].min()} to {swsi_df['timestamp'].max()}")

    # Aggregate SWSI by date
    swsi_daily = swsi_df.groupby('date')['swsi'].mean().reset_index()

    # Merge CWSI and SWSI
    combined_df = pd.merge(cwsi_df, swsi_daily, on='date', how='left')

    # Calculate weighted combined index (60% SWSI, 40% CWSI)
    combined_df['combined_index'] = 0.6 * combined_df['swsi'] + 0.4 * combined_df['cwsi']

    # Remove rows with NaN values in combined_index
    combined_df = combined_df.dropna(subset=['combined_index'])

    # Debug: Print combined index date range
    print(f"Figure 6 - Combined Index Data Date Range: {combined_df['date'].min()} to {combined_df['date'].max()}")

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
    # Aggregate by date
    combined_daily = combined_df.groupby('date').agg({'swsi': 'mean', 'cwsi': 'mean', 'combined_index': 'mean'}).reset_index()

    # Debug: Print combined_daily date range
    print(f"Figure 6 - Combined Daily Data Date Range: {combined_daily['date'].min()} to {combined_daily['date'].max()}")

    axes[1].plot(combined_daily['date'], combined_daily['swsi'], label='SWSI', color=CUSTOM_COLORS['blue'], linewidth=2.5)
    axes[1].plot(combined_daily['date'], combined_daily['cwsi'], label='CWSI', color=CUSTOM_COLORS['green'], linewidth=2.5)
    axes[1].plot(combined_daily['date'], combined_daily['combined_index'], label='Combined Index (60% SWSI, 40% CWSI)', color=CUSTOM_COLORS['accent_purple'], linewidth=2.5)
    style_axis(axes[1], 
              title='Time Series of SWSI, CWSI, and Combined Index',
              ylabel='Index Value')
    style_legend(axes[1], loc='upper right')

    # Panel (c): Decision threshold analysis
    # Define irrigation trigger threshold, e.g., combined_index > 0.7
    threshold = 0.7
    axes[2].plot(combined_daily['date'], combined_daily['combined_index'], label='Combined Index', color=CUSTOM_COLORS['accent_purple'], linewidth=2.5)
    axes[2].axhline(y=threshold, color=CUSTOM_COLORS['red'], linestyle='--', label=f'Irrigation Threshold ({threshold})')
    style_axis(axes[2], 
              title='Irrigation Decision Threshold Based on Combined Index',
              xlabel='Date',
              ylabel='Combined Index Value')
    style_legend(axes[2], loc='upper right')

    # Adjust layout to make room for the caption
    plt.subplots_adjust(top=0.95)
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 6: Combined Index Analysis and Weighting System added to PDF.")

def generate_figure7(conn, pdf):
    """
    Figure 7: System Response Analysis
    Multi-day analysis of canopy temperature, soil moisture, recovery patterns, and time lag post-irrigation.
    """
    label = "Figure 7: System Response Analysis"
    print(f"Generating {label}")

    # Extract canopy temperature data (variable_name starts with 'IRT')
    irt_query = """
    SELECT d.timestamp, d.plot_id, d.value as canopy_temp
    FROM data d
    WHERE d.variable_name LIKE 'IRT%'
    """
    irt_df = pd.read_sql_query(irt_query, conn)
    irt_df = filter_valid_dates(irt_df, 'timestamp')
    irt_df['timestamp'] = pd.to_datetime(irt_df['timestamp'])
    irt_df['date'] = irt_df['timestamp'].dt.date

    # Extract soil moisture data (assuming SWSI)
    swsi_query = """
    SELECT d.timestamp, d.plot_id, d.value as swsi
    FROM data d
    WHERE d.variable_name = 'swsi'
    """
    swsi_df = pd.read_sql_query(swsi_query, conn)
    swsi_df = filter_valid_dates(swsi_df, 'timestamp')
    swsi_df['timestamp'] = pd.to_datetime(swsi_df['timestamp'])
    swsi_df['date'] = swsi_df['timestamp'].dt.date

    # Extract irrigation events
    irrigation_query = """
    SELECT date, amount_mm
    FROM irrigation_events
    """
    irrigation_df = pd.read_sql_query(irrigation_query, conn)
    irrigation_df = filter_valid_dates(irrigation_df, 'date')
    irrigation_df['date'] = pd.to_datetime(irrigation_df['date'])

    # Debug: Print IRT, SWSI, and Irrigation data date ranges
    print(f"Figure 7 - IRT Data Date Range: {irt_df['timestamp'].min()} to {irt_df['timestamp'].max()}")
    print(f"Figure 7 - SWSI Data Date Range: {swsi_df['timestamp'].min()} to {swsi_df['timestamp'].max()}")
    print(f"Figure 7 - Irrigation Events Date Range: {irrigation_df['date'].min()} to {irrigation_df['date'].max()}")

    # Aggregate canopy temperature by date
    if irt_df['canopy_temp'].notna().any():
        irt_daily = irt_df.groupby('date')['canopy_temp'].mean().reset_index()
    else:
        print("Figure 7 - No valid IRT data found. Skipping IRT-related plots.")
        irt_daily = pd.DataFrame(columns=['date', 'canopy_temp'])

    # Aggregate SWSI by date
    swsi_daily = swsi_df.groupby('date')['swsi'].mean().reset_index()

    # Merge datasets
    if not irt_daily.empty:
        response_df = pd.merge(irt_daily, swsi_daily, on='date', how='outer')
    else:
        response_df = swsi_daily.copy()

    # Convert 'date' column to datetime in response_df
    response_df['date'] = pd.to_datetime(response_df['date'])

    # Merge with irrigation events
    response_df = pd.merge(response_df, irrigation_df, on='date', how='left')
    response_df['amount_mm'] = response_df['amount_mm'].fillna(0)

    # Check if response_df has valid data
    if response_df['date'].isna().all():
        print("Figure 7 - Merged Response Data has no valid dates. Skipping Figure 7.")
        return

    # Define a 7-day rolling window for analysis
    if 'canopy_temp' in response_df.columns:
        response_df['canopy_temp_7d_avg'] = response_df['canopy_temp'].rolling(window=7, min_periods=1).mean()
    else:
        response_df['canopy_temp_7d_avg'] = np.nan

    response_df['swsi_7d_avg'] = response_df['swsi'].rolling(window=7, min_periods=1).mean()

    # Debug: Print merged response_df date range
    print(f"Figure 7 - Merged Response Data Date Range: {response_df['date'].min()} to {response_df['date'].max()}")

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(15, 22), sharex=True)

    # Panel (a): Canopy temperature response
    if 'canopy_temp' in response_df.columns and response_df['canopy_temp'].notna().any():
        axes[0].plot(response_df['date'], response_df['canopy_temp'], label='Canopy Temperature (IRT)', color=CUSTOM_COLORS['orange'], linewidth=2.5)
        axes[0].set_ylabel('Canopy Temperature (°C)')
        axes[0].set_title('Canopy Temperature Response to Irrigation/Rainfall')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No Canopy Temperature Data Available', horizontalalignment='center', verticalalignment='center', fontsize=16)
        axes[0].set_title('Canopy Temperature Response to Irrigation/Rainfall')
        axes[0].set_ylabel('Canopy Temperature (°C)')

    style_axis(axes[0])
    style_legend(axes[0], loc='upper right')

    # Panel (b): Soil moisture changes
    axes[1].plot(response_df['date'], response_df['swsi'], label='Soil Water Stress Index (SWSI)', color=CUSTOM_COLORS['brown'], linewidth=2.5)
    style_axis(axes[1], 
              title='Soil Moisture Changes',
              ylabel='SWSI')
    style_legend(axes[1], loc='upper right')

    # Panel (c): Recovery patterns across treatments (Assuming treatment data is available)
    # For simplicity, plot SWSI recovery after irrigation
    axes[2].plot(response_df['date'], response_df['swsi_7d_avg'], label='7-Day Avg SWSI', color=CUSTOM_COLORS['green'], linewidth=2.5)
    style_axis(axes[2], 
              title='Recovery Patterns of SWSI After Irrigation Events',
              ylabel='Average SWSI')
    style_legend(axes[2], loc='upper right')

    # Panel (d): Time lag analysis between irrigation and plant response
    axes[3].bar(response_df['date'], response_df['amount_mm'], label='Irrigation Amount (mm)', color=CUSTOM_COLORS['blue'], width=1.0)
    style_axis(axes[3], 
              title='Irrigation Events Over Time',
              xlabel='Date',
              ylabel='Irrigation Amount (mm)')
    style_legend(axes[3], loc='upper right')

    # Adjust layout to make room for the caption
    plt.subplots_adjust(top=0.95)
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 7: System Response Analysis added to PDF.")

def generate_figure8(conn, pdf):
    """
    Figure 8: Monthly Stress Index Distributions
    Boxplots showing CWSI, SWSI, and combined index distributions across treatments monthly.
    """
    label = "Figure 8: Monthly Stress Index Distributions"
    print(f"Generating {label}")

    # Extract CWSI and SWSI data
    stress_query = """
    SELECT d.timestamp, p.treatment, d.variable_name, d.value
    FROM data d
    JOIN plots p ON d.plot_id = p.plot_id
    WHERE d.variable_name IN ('cwsi', 'swsi')
    """
    stress_df = pd.read_sql_query(stress_query, conn)
    stress_df = filter_valid_dates(stress_df, 'timestamp')
    stress_df['timestamp'] = pd.to_datetime(stress_df['timestamp'])
    stress_df['month'] = stress_df['timestamp'].dt.to_period('M').astype(str)

    # Debug: Print stress data date range
    print(f"Figure 8 - Stress Data Date Range: {stress_df['timestamp'].min()} to {stress_df['timestamp'].max()}")

    # Pivot data to have cwsi and swsi in separate columns
    stress_pivot = stress_df.pivot_table(index=['timestamp', 'treatment', 'month'], columns='variable_name', values='value').reset_index()

    # Calculate combined index
    stress_pivot['combined_index'] = 0.6 * stress_pivot['swsi'] + 0.4 * stress_pivot['cwsi']

    # Melt for boxplot
    boxplot_df = stress_pivot.melt(id_vars=['treatment', 'month'], value_vars=['cwsi', 'swsi', 'combined_index'], var_name='Index', value_name='Value')

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='month', y='Value', hue='Index', data=boxplot_df, palette='Set2', ax=ax)
    style_axis(ax, 
              title='Monthly Stress Index Distributions Across Treatments',
              xlabel='Month',
              ylabel='Index Value')
    style_legend(ax, title='Index', loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 8: Monthly Stress Index Distributions added to PDF.")

def generate_figure9(conn, pdf):
    """
    Figure 9: Irrigation Application Patterns
    Stacked bar graphs showing daily and cumulative irrigation amounts by treatment.
    """
    label = "Figure 9: Irrigation Application Patterns"
    print(f"Generating {label}")

    # Extract irrigation events with treatment information
    irrigation_query = """
    SELECT ie.date, ie.amount_mm, p.treatment
    FROM irrigation_events ie
    JOIN plots p ON ie.plot_id = p.plot_id
    """
    irrigation_df = pd.read_sql_query(irrigation_query, conn)
    irrigation_df = filter_valid_dates(irrigation_df, 'date')
    irrigation_df['date'] = pd.to_datetime(irrigation_df['date'])

    # Debug: Print irrigation data date range
    print(f"Figure 9 - Irrigation Data Date Range: {irrigation_df['date'].min()} to {irrigation_df['date'].max()}")

    # Define treatment names
    treatment_names = {
        1: 'IoT-Fuzzy',
        2: 'CWSI + SWSI',
        3: 'CWSI only',
        4: 'SWSI',
        5: 'ET-Model',
        6: "Grower's Practice"
    }

    # Map treatment numbers to names
    irrigation_df['treatment_name'] = irrigation_df['treatment'].map(treatment_names)

    # Aggregate daily irrigation by treatment
    daily_irrigation = irrigation_df.groupby(['date', 'treatment_name'])['amount_mm'].sum().reset_index()

    # Pivot for stacked bar
    pivot_daily = daily_irrigation.pivot(index='date', columns='treatment_name', values='amount_mm').fillna(0)

    # Plot stacked bar for daily irrigation amounts
    fig, ax = plt.subplots(figsize=(15, 6))
    pivot_daily.plot(kind='bar', stacked=True, ax=ax, color=sns.color_palette("Set2", n_colors=pivot_daily.shape[1]))
    style_axis(ax, 
              title='Daily Irrigation Amounts by Treatment',
              xlabel='Date',
              ylabel='Irrigation Amount (mm)')
    style_legend(ax, title='Treatment', loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 9a: Daily Irrigation Amounts by Treatment added to PDF.")

    # Calculate cumulative irrigation
    pivot_daily_cumulative = pivot_daily.cumsum()

    # Plot stacked line for cumulative irrigation amounts
    fig, ax = plt.subplots(figsize=(15, 6))
    pivot_daily_cumulative.plot(kind='line', ax=ax, marker='o', linewidth=2.5)
    style_axis(ax, 
              title='Cumulative Irrigation Amounts by Treatment',
              xlabel='Date',
              ylabel='Cumulative Irrigation Amount (mm)')
    style_legend(ax, title='Treatment', loc='upper left')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 9b: Cumulative Irrigation Amounts by Treatment added to PDF.")

def generate_figure10(conn, pdf):
    """
    Figure 10: Yield Distribution Analysis
    Box plots of yield distribution by treatment, scatter plots of individual plot yields.
    """
    try:
        label = "Figure 10: Yield Distribution Analysis"
        print(f"Generating {label}")

        # Modified SQL query to avoid duplicate columns
        yield_query = """
        SELECT y.id, y.plot_id, y.trt_name, y.crop_type, 
               y.avg_yield_bu_ac, y.yield_kg_ha, 
               y.irrigation_applied_inches, y.irrigation_applied_mm,
               p.treatment, p.field
        FROM yields y
        JOIN plots p ON y.plot_id = p.plot_id
        """
        yield_df = pd.read_sql_query(yield_query, conn)

        # Debug: Print yield data info
        print(f"Figure 10 - Yield Data Columns: {yield_df.columns.tolist()}")
        print(f"Figure 10 - Number of rows: {len(yield_df)}")
        print(f"Figure 10 - Unique treatments: {yield_df['treatment'].tolist()}")
        print(f"Figure 10 - Unique crop types: {yield_df['crop_type'].unique().tolist()}")
        print(f"Figure 10 - Data types:\n{yield_df.dtypes}")

        # First plot: Box plots of yield by treatment
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='treatment', y='yield_kg_ha', data=yield_df, hue='treatment', legend=False, ax=ax)
            style_axis(ax, 
                      title='Yield Distribution by Treatment',
                      xlabel='Treatment',
                      ylabel='Yield (kg/ha)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print("Figure 10a: Yield Distribution by Treatment added to PDF.")
        except Exception as e:
            print(f"Error in first plot of Figure 10: {str(e)}")
            plt.close('all')

        # Second plot: Scatter plot of individual plot yields
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convert treatment to numeric if it isn't already
            yield_df['treatment'] = pd.to_numeric(yield_df['treatment'], errors='coerce')
            
            # Add jitter to treatment values
            jitter = np.random.normal(0, 0.1, len(yield_df))
            
            # Create scatter plot with different markers for each crop type
            for crop_type in yield_df['crop_type'].unique():
                mask = yield_df['crop_type'] == crop_type
                ax.scatter(yield_df.loc[mask, 'treatment'] + jitter[mask],
                          yield_df.loc[mask, 'yield_kg_ha'],
                          label=crop_type,
                          alpha=0.7)
            
            ax.set_xticks(sorted(yield_df['treatment'].unique()))
            style_axis(ax, 
                      title='Individual Plot Yields by Treatment and Crop Type',
                      xlabel='Treatment',
                      ylabel='Yield (kg/ha)')
            ax.legend(title='Crop Type', loc='upper right')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print("Figure 10b: Individual Plot Yields by Treatment and Crop Type added to PDF.")
        except Exception as e:
            print(f"Error in second plot of Figure 10: {str(e)}")
            plt.close('all')

    except Exception as e:
        print(f"Error in generate_figure10: {str(e)}")
        plt.close('all')
        raise

def generate_figure11(conn, pdf):
    """
    Figure 11: Water Use Efficiency Metrics
    Comparison of irrigation water use efficiency metrics across treatments.
    """
    label = "Figure 11: Water Use Efficiency Metrics"
    print(f"Generating {label}")

    # Extract yield and irrigation data with treatment information
    efficiency_query = """
    SELECT y.plot_id, p.treatment, y.yield_kg_ha, y.irrigation_applied_mm
    FROM yields y
    JOIN plots p ON y.plot_id = p.plot_id
    """
    efficiency_df = pd.read_sql_query(efficiency_query, conn)

    # Debug: Print efficiency data columns and rows
    print(f"Figure 11 - Efficiency Data Columns: {efficiency_df.columns.tolist()}")
    print(f"Figure 11 - Efficiency Data Rows: {len(efficiency_df)}")
    print(f"Figure 11 - Missing Irrigation Applied Entries: {efficiency_df['irrigation_applied_mm'].isna().sum()}")

    # Calculate IWUE and CWUE
    # Assuming IWUE = Yield / Irrigation Applied
    efficiency_df['IWUE'] = efficiency_df['yield_kg_ha'] / efficiency_df['irrigation_applied_mm']

    # Handle cases where irrigation_applied_mm is zero to avoid division by zero
    efficiency_df['IWUE'] = efficiency_df['IWUE'].replace([np.inf, -np.inf], np.nan)

    # Calculate CWUE (Yield per total evapotranspiration)
    # Assuming 'eto' is available as a daily value; sum 'eto' per plot
    eto_query = """
    SELECT plot_id, SUM(value) as total_eto
    FROM data
    WHERE variable_name = 'eto'
    GROUP BY plot_id
    """
    eto_df = pd.read_sql_query(eto_query, conn)
    efficiency_df = pd.merge(efficiency_df, eto_df, on='plot_id', how='left')
    efficiency_df['CWUE'] = efficiency_df['yield_kg_ha'] / efficiency_df['total_eto']

    # Handle cases where total_eto is zero or NaN
    efficiency_df['CWUE'] = efficiency_df['CWUE'].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in IWUE or CWUE
    initial_count = len(efficiency_df)
    efficiency_df = efficiency_df.dropna(subset=['IWUE', 'CWUE'])
    final_count = len(efficiency_df)
    print(f"Figure 11 - Dropped {initial_count - final_count} rows due to NaN in IWUE or CWUE")

    # Debug: Print IWUE and CWUE statistics
    print(f"Figure 11 - IWUE Stats:\n{efficiency_df['IWUE'].describe()}")
    print(f"Figure 11 - CWUE Stats:\n{efficiency_df['CWUE'].describe()}")

    # Plot IWUE by treatment
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='treatment', y='IWUE', data=efficiency_df, palette='Set3', ax=ax)
    style_axis(ax, 
              title='IWUE by Treatment',
              xlabel='Treatment',
              ylabel='Irrigation Water Use Efficiency (kg/ha per mm)')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11a: IWUE by Treatment added to PDF.")

    # Plot CWUE by treatment
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='treatment', y='CWUE', data=efficiency_df, palette='Set3', ax=ax)
    style_axis(ax, 
              title='CWUE by Treatment',
              xlabel='Treatment',
              ylabel='Crop Water Use Efficiency (kg/ha per mm ETo)')
    style_legend(ax, loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11b: CWUE by Treatment added to PDF.")

    # Scatter plot: Applied water vs Yield
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='irrigation_applied_mm', y='yield_kg_ha', hue='treatment', data=efficiency_df, palette='deep', ax=ax)
    style_axis(ax, 
              title='Relationship between Applied Water and Yield by Treatment',
              xlabel='Irrigation Applied (mm)',
              ylabel='Yield (kg/ha)')
    style_legend(ax, title='Treatment', loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')

    # Save the figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11c: Applied Water vs Yield added to PDF.")

def main():
    """Main function to generate all figures and compile them into a single PDF."""
    conn = None
    try:
        # Connect to the database
        conn = connect_db(DATABASE_PATH)
        if conn is None:
            print("Failed to connect to the database. Exiting.")
            return

        # Initialize PdfPages
        with PdfPages(PDF_OUTPUT_PATH) as pdf:
            # Generate Figures
            generate_figure3(conn, pdf)
            generate_figure4(conn, pdf)
            generate_figure5(conn, pdf)
            generate_figure6(conn, pdf)
            generate_figure7(conn, pdf)
            generate_figure8(conn, pdf)
            generate_figure9(conn, pdf)
            generate_figure10(conn, pdf)
            generate_figure11(conn, pdf)
            # generate_figure12(conn, pdf)  # Uncomment and implement if needed

            # Optional: Add PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Irrigation and Stress Index Analysis Figures'
            d['Author'] = 'Your Name'
            d['Subject'] = 'Generated by irrigation_graphs_generator.py'
            d['Keywords'] = 'Irrigation, CWSI, SWSI, Water Use Efficiency, Agriculture'
            d['CreationDate'] = datetime.now()
            d['ModDate'] = datetime.now()

        print(f"All figures have been compiled into {PDF_OUTPUT_PATH}")

    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()