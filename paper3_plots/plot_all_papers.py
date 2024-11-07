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
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def add_statistical_annotations(ax, df, x, y):
    """
    Add ANOVA and Tukey's HSD annotations to the plot.
    """
    # Perform one-way ANOVA
    groups = df.groupby(x)[y].apply(list)
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA results: F-statistic={f_stat:.2f}, p-value={p_val:.4f}")
    
    # If significant, perform Tukey's HSD
    if p_val < 0.05:
        tukey = pairwise_tukeyhsd(endog=df[y], groups=df[x], alpha=0.05)
        print(tukey.summary())
        # You can add more sophisticated annotations here if needed
        ax.text(0.95, 0.95, f'ANOVA p={p_val:.3f}', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    else:
        ax.text(0.95, 0.95, f'ANOVA p={p_val:.3f}', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    return ax

# ================================
# Treatment Names Mapping
# ================================

TREATMENT_NAMES = {
    1: 'IoT-Fuzzy',
    2: 'CWSI + SWSI',
    3: 'CWSI only',
    4: 'SWSI',
    5: 'ET-Model',
    6: "Grower's Practice"
}

# ================================
# Plotting Functions
# ================================

def generate_representative_days_analysis(conn, pdf):
    """
    Create three-panel comparison for wet, dry, and normal days.
    Each panel shows:
    - Canopy temperature (from IRT data)
    - Soil moisture at 6cm, 18cm, 30cm, 42cm (from TDR data)
    - Rainfall/irrigation events
    Focus on response patterns rather than the entire season.
    """
    label = "Figure 12: Representative Days Analysis (Wet, Dry, Normal)"
    print(f"Generating {label}")
    
    # Identify representative days
    # 1. Wet: Highest rainfall day
    rainfall_query = """
    SELECT date(timestamp) as date, SUM(value) as total_rain
    FROM data
    WHERE variable_name = 'Rain_1m_Tot'
    GROUP BY date
    ORDER BY total_rain DESC
    LIMIT 1
    """
    wet_day_df = pd.read_sql_query(rainfall_query, conn)
    wet_day = wet_day_df['date'].iloc[0] if not wet_day_df.empty else None
    print(f"Wet Day: {wet_day}")

    # 2. Dry: Longest period without rain
    # Find days with zero rain
    dry_days_query = """
    SELECT date(timestamp) as date
    FROM data
    WHERE variable_name = 'Rain_1m_Tot'
    GROUP BY date
    HAVING SUM(value) = 0
    ORDER BY date
    """
    dry_days_df = pd.read_sql_query(dry_days_query, conn)
    # Find the longest consecutive dry days (assuming daily data)
    dry_days = dry_days_df['date'].tolist()
    if dry_days:
        # Convert to datetime
        dry_days_dt = pd.to_datetime(dry_days)
        # Find the longest consecutive sequence
        dry_days_sorted = sorted(dry_days_dt)
        longest_streak = []
        current_streak = [dry_days_sorted[0]]
        for current in dry_days_sorted[1:]:
            if current - current_streak[-1] == pd.Timedelta(days=1):
                current_streak.append(current)
            else:
                if len(current_streak) > len(longest_streak):
                    longest_streak = current_streak
                current_streak = [current]
        if len(current_streak) > len(longest_streak):
            longest_streak = current_streak
        dry_day = longest_streak[-1]  # Last day of the longest dry streak
        print(f"Dry Day: {dry_day.date()}")
    else:
        dry_day = None
        print("No Dry Day identified.")

    # 3. Normal: Day with average VPD
    # Calculate daily average VPD
    vpd_query = """
    SELECT date(timestamp) as date, AVG(value) as avg_vpd
    FROM data
    WHERE variable_name = 'VPD'
    GROUP BY date
    ORDER BY ABS(avg_vpd - (SELECT AVG(value) FROM data WHERE variable_name='VPD'))
    LIMIT 1
    """
    normal_day_df = pd.read_sql_query(vpd_query, conn)
    normal_day = normal_day_df['date'].iloc[0] if not normal_day_df.empty else None
    print(f"Normal Day: {normal_day}")

    # Define the representative days
    representative_days = {
        'Wet': wet_day,
        'Dry': dry_day,
        'Normal': normal_day
    }

    # Iterate over each representative day and plot
    for condition, day in representative_days.items():
        if day is None:
            print(f"No {condition} day found. Skipping.")
            continue

        # Define a 3-day window around the representative day
        day_dt = pd.to_datetime(day)
        start_day = day_dt - pd.Timedelta(days=1)
        end_day = day_dt + pd.Timedelta(days=1)

        # Fetch canopy temperature (IRT data)
        irt_query = f"""
        SELECT timestamp, value as canopy_temp
        FROM data
        WHERE variable_name LIKE 'IRT%' 
        AND date(timestamp) BETWEEN '{start_day.date()}' AND '{end_day.date()}'
        """
        irt_df = pd.read_sql_query(irt_query, conn)
        irt_df['timestamp'] = pd.to_datetime(irt_df['timestamp'])

        # Fetch TDR soil moisture data at different depths
        tdr_query = f"""
        SELECT timestamp, variable_name, value
        FROM data
        WHERE variable_name LIKE 'TDR%'
        AND date(timestamp) BETWEEN '{start_day.date()}' AND '{end_day.date()}'
        """
        tdr_df = pd.read_sql_query(tdr_query, conn)
        tdr_df['timestamp'] = pd.to_datetime(tdr_df['timestamp'])
        # Extract depth from variable_name
        tdr_df['Depth_cm'] = tdr_df['variable_name'].str.extract(r'TDR\d{4}[A-C]\d(\d{2})\d{2}$')[0]
        tdr_df['Depth_cm'] = pd.to_numeric(tdr_df['Depth_cm'], errors='coerce')
        tdr_df = tdr_df.dropna(subset=['Depth_cm'])

        # Fetch irrigation events
        irrigation_query = f"""
        SELECT date, SUM(amount_mm) as total_irrigation
        FROM irrigation_events
        WHERE date BETWEEN '{start_day.date()}' AND '{end_day.date()}'
        GROUP BY date
        """
        irrigation_df = pd.read_sql_query(irrigation_query, conn)
        irrigation_df['date'] = pd.to_datetime(irrigation_df['date'])

        # Calculate VPD
        weather_query = f"""
        SELECT date(timestamp) as date, AVG(value) as avg_temp, AVG(RH_2m_Avg) as avg_rh
        FROM data
        WHERE variable_name IN ('Ta_2m_Avg', 'RH_2m_Avg')
        AND date(timestamp) BETWEEN '{start_day.date()}' AND '{end_day.date()}'
        GROUP BY date
        """
        weather_df = pd.read_sql_query(weather_query, conn)
        weather_df['VPD'] = weather_df.apply(lambda row: calc_vpd(row['avg_temp'], row['avg_rh']), axis=1)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Plot canopy temperature
        if not irt_df.empty:
            sns.lineplot(x='timestamp', y='canopy_temp', data=irt_df, label='Canopy Temp (IRT)', ax=ax1, color=CUSTOM_COLORS['accent_orange'])
        
        # Plot soil moisture at different depths
        for depth in [6, 18, 30, 42]:
            depth_data = tdr_df[tdr_df['Depth_cm'] == depth]
            if not depth_data.empty:
                sns.lineplot(x='timestamp', y='value', data=depth_data, label=f'Soil Moisture {depth} cm', ax=ax1)

        ax1.set_title(f'{condition} Day Analysis: {day.date()}')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Canopy Temp (°C) / Soil Moisture (%)')
        ax1.legend(loc='upper left')
        style_axis(ax1)

        # Create a second y-axis for VPD
        ax2 = ax1.twinx()
        if not weather_df.empty:
            sns.lineplot(x='date', y='VPD', data=weather_df, label='VPD', ax=ax2, color=CUSTOM_COLORS['red'], linestyle='--')
            ax2.set_ylabel('VPD (kPa)')
            ax2.legend(loc='upper right')
            style_axis(ax2)

        # Highlight irrigation events
        if not irrigation_df.empty:
            for _, row in irrigation_df.iterrows():
                ax1.axvline(x=row['date'], color=CUSTOM_COLORS['blue'], linestyle=':', alpha=0.7)
                ax1.text(row['date'], ax1.get_ylim()[1], 'Irrigation', rotation=90, verticalalignment='top', fontsize=12, color=CUSTOM_COLORS['blue'])

        # Save the figure to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"{label} for {condition} day added to PDF.")

def generate_tdr_neutron_comparison(conn, pdf):
    """
    Placeholder function for TDR vs Neutron Probe comparison.
    Will be implemented when neutron probe data is available.
    """
    print("Skipping TDR vs Neutron Probe comparison - data not yet available")
    return

def update_treatment_labels(ax):
    """
    Update treatment labels to actual treatment names.
    """
    ax.set_xticks(range(len(TREATMENT_NAMES)))
    ax.set_xticklabels([TREATMENT_NAMES[i] for i in sorted(TREATMENT_NAMES.keys())], rotation=45, ha='right')
    return ax

def add_error_bars(ax, df, treatment_col, yield_col):
    """Add standard error bars to yield plots."""
    stats_df = df.groupby(treatment_col).agg({
        yield_col: ['mean', 'std', 'count']
    }).reset_index()
    stats_df.columns = ['treatment', 'mean', 'std', 'count']
    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
    treatments = sorted(stats_df['treatment'].unique())
    means = stats_df['mean']
    ses = stats_df['se']
    ax.errorbar(x=range(len(treatments)), y=means, yerr=ses, fmt='none', capsize=5, color='black')
    return ax

def add_statistical_annotations_anova_tukey(ax, df, treatment_col, yield_col):
    """
    Perform ANOVA and Tukey's HSD test and annotate the plot.
    """
    # Perform one-way ANOVA
    groups = df.groupby(treatment_col)[yield_col].apply(list)
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}")
    
    # Add ANOVA p-value to the plot
    ax.text(0.95, 0.95, f'ANOVA p={p_val:.3f}', transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # If significant, perform Tukey's HSD
    if p_val < 0.05:
        tukey = pairwise_tukeyhsd(endog=df[yield_col], groups=df[treatment_col], alpha=0.05)
        print(tukey.summary())
        # You can add more detailed annotations based on Tukey's results if desired
    
    return ax

def calc_vpd(temp, rh):
    """
    Calculate Vapor Pressure Deficit (VPD) given temperature and relative humidity.
    VPD (kPa) = saturation vapor pressure - actual vapor pressure
    """
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    vpd = es * (1 - rh / 100)
    return vpd

# ================================
# Figure Generation Functions
# ================================

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

def generate_figure4(conn, pdf):
    """
    Figure 4: CWSI Implementation Analysis
    Daily CWSI values for corn and soybeans, with growth stage analysis.
    """
    label = "Figure 4: CWSI Implementation Analysis"
    print(f"Generating {label}")

    try:
        # Corn CWSI Analysis (Figure 4a)
        cwsi_query = """
        SELECT d.timestamp, d.value as cwsi, p.treatment, p.field,
               CASE 
                   WHEN date(d.timestamp) <= '2024-07-20' THEN 'VT/R1'
                   WHEN date(d.timestamp) <= '2024-07-30' THEN 'R2'
                   WHEN date(d.timestamp) <= '2024-08-10' THEN 'R3'
                   WHEN date(d.timestamp) <= '2024-08-20' THEN 'R4'
                   WHEN date(d.timestamp) <= '2024-08-25' THEN 'R5'
                   WHEN date(d.timestamp) <= '2024-08-27' THEN 'R5.25'
                   ELSE 'R5.5'
               END as growth_stage
        FROM data d
        JOIN plots p ON d.plot_id = p.plot_id
        WHERE d.variable_name = 'cwsi'
        AND p.field = 'LINEAR_CORN'
        """
        corn_cwsi = pd.read_sql_query(cwsi_query, conn)
        
        # Rename the column to match what the plotting code expects
        corn_cwsi = corn_cwsi.rename(columns={'growth_stage': 'Growth Stage'})

        # Create boxplot for corn
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Growth Stage', y='cwsi', data=corn_cwsi, 
                   hue='Growth Stage', palette='Set3', legend=False, ax=ax)
        style_axis(ax, 
                  title='CWSI Distribution by Growth Stage (Corn)',
                  xlabel='Growth Stage',
                  ylabel='CWSI')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print("Figure 4c: CWSI Distribution by Growth Stage (Corn) added to PDF.")

        # Soybean CWSI Analysis (Figure 4b)
        soybean_query = """
        SELECT d.timestamp, d.value as cwsi, p.treatment, p.field,
               CASE 
                   WHEN date(d.timestamp) <= '2024-07-20' THEN 'V6/R1'
                   WHEN date(d.timestamp) <= '2024-07-30' THEN 'R2'
                   WHEN date(d.timestamp) <= '2024-08-10' THEN 'R3'
                   WHEN date(d.timestamp) <= '2024-08-20' THEN 'R4'
                   WHEN date(d.timestamp) <= '2024-08-25' THEN 'R5'
                   WHEN date(d.timestamp) <= '2024-08-27' THEN 'R5.5'
                   ELSE 'R6'
               END as growth_stage
        FROM data d
        JOIN plots p ON d.plot_id = p.plot_id
        WHERE d.variable_name = 'cwsi'
        AND p.field = 'LINEAR_SOYBEAN'
        """
        soybean_cwsi = pd.read_sql_query(soybean_query, conn)

        # Rename the column to match what the plotting code expects
        soybean_cwsi = soybean_cwsi.rename(columns={'growth_stage': 'Growth Stage'})

        # Create boxplot for soybeans
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Growth Stage', y='cwsi', data=soybean_cwsi,
                   hue='Growth Stage', palette='Set3', legend=False, ax=ax)
        style_axis(ax, 
                  title='CWSI Distribution by Growth Stage (Soybeans)',
                  xlabel='Growth Stage',
                  ylabel='CWSI')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print("Figure 4d: CWSI Distribution by Growth Stage (Soybeans) added to PDF.")

    except Exception as e:
        print(f"Error in generate_figure4: {str(e)}")
        plt.close('all')
        raise

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

def generate_figure7(conn, pdf):
    """
    Figure 7: System Response Analysis
    Analyze how canopy temperature and soil moisture respond to irrigation events.
    """
    label = "Figure 7: System Response Analysis"
    print(f"Generating {label}")

    try:
        # Get IRT (canopy temperature) data
        irt_query = """
        SELECT date(timestamp) as date, AVG(value) as canopy_temp
        FROM data
        WHERE variable_name LIKE 'IRT%'
        GROUP BY date(timestamp)
        """
        irt_df = pd.read_sql_query(irt_query, conn)
        irt_df['date'] = pd.to_datetime(irt_df['date']).dt.date  # Convert to date

        # Get SWSI data
        swsi_query = """
        SELECT date(timestamp) as date, AVG(value) as swsi
        FROM data
        WHERE variable_name = 'swsi'
        GROUP BY date(timestamp)
        """
        swsi_df = pd.read_sql_query(swsi_query, conn)
        swsi_df['date'] = pd.to_datetime(swsi_df['date']).dt.date  # Convert to date

        # Get irrigation events
        irrigation_query = """
        SELECT date(date) as date, SUM(amount_mm) as irrigation_mm
        FROM irrigation_events
        GROUP BY date(date)
        """
        irrigation_df = pd.read_sql_query(irrigation_query, conn)
        irrigation_df['date'] = pd.to_datetime(irrigation_df['date']).dt.date  # Convert to date

        # Convert all dates to datetime.date objects for consistent merging
        response_df = pd.merge(irt_df, swsi_df, on='date', how='outer')
        response_df = pd.merge(response_df, irrigation_df, on='date', how='left')
        response_df['irrigation_mm'] = response_df['irrigation_mm'].fillna(0)

        # Sort by date
        response_df = response_df.sort_values('date')

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot canopy temperature
        ax1.plot(response_df['date'], response_df['canopy_temp'], 
                label='Canopy Temperature', color='red', linewidth=2)
        ax1.set_ylabel('Canopy Temperature (°C)')
        ax1.legend(loc='upper left')

        # Plot SWSI
        ax2.plot(response_df['date'], response_df['swsi'], 
                label='SWSI', color='blue', linewidth=2)
        ax2.set_ylabel('SWSI')
        ax2.legend(loc='upper left')

        # Add irrigation events to both plots
        for ax in [ax1, ax2]:
            irrigation_events = response_df[response_df['irrigation_mm'] > 0]
            for _, event in irrigation_events.iterrows():
                ax.axvline(x=event['date'], color='green', alpha=0.3, linestyle='--')
                ax.text(event['date'], ax.get_ylim()[1], f"{event['irrigation_mm']:.1f}mm", 
                       rotation=90, va='top')

        # Formatting
        plt.xlabel('Date')
        fig.suptitle(label, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"{label} added to PDF.")

    except Exception as e:
        print(f"Error in generate_figure7: {str(e)}")
        plt.close('all')
        raise

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

    # Map treatment numbers to names
    irrigation_df['treatment_name'] = irrigation_df['treatment'].map(TREATMENT_NAMES)

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
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 9b: Cumulative Irrigation Amounts by Treatment added to PDF.")

def generate_figure10(conn, pdf):
    """
    Figure 10: Yield Distribution Analysis
    Box plots of yield distribution by treatment, scatter plots of individual plot yields.
    """
    label = "Figure 10: Yield Distribution Analysis"
    print(f"Generating {label}")

    # Extract yield data
    yield_query = """
    SELECT y.id, y.plot_id, y.trt_name, y.crop_type, 
           y.avg_yield_bu_ac, y.yield_kg_ha, 
           y.irrigation_applied_inches, y.irrigation_applied_mm,
           p.treatment, p.field
    FROM yields y
    JOIN plots p ON y.plot_id = p.plot_id
    """
    yield_df = pd.read_sql_query(yield_query, conn)

    # Map treatment numbers to names
    yield_df['treatment_name'] = yield_df['treatment'].map(TREATMENT_NAMES)

    # First plot: Box plots of yield by treatment
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='treatment_name', y='yield_kg_ha', data=yield_df, 
                hue='treatment_name', palette='Set3', legend=False, ax=ax)
    add_error_bars(ax, yield_df, 'treatment_name', 'yield_kg_ha')
    style_axis(ax, 
              title='Yield Distribution by Treatment',
              xlabel='Treatment',
              ylabel='Yield (kg/ha)')
    add_statistical_annotations_anova_tukey(ax, yield_df, 'treatment_name', 'yield_kg_ha')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 10a: Yield Distribution by Treatment added to PDF.")

    # Second plot: Scatter plot of individual plot yields
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Add jitter to treatment positions for better visibility
    jitter = 0.1
    unique_treatments = sorted(yield_df['treatment'].unique())
    treatment_positions = {t: i for i, t in enumerate(unique_treatments)}
    yield_df['treatment_pos'] = yield_df['treatment'].map(treatment_positions)
    yield_df['jittered_pos'] = yield_df['treatment_pos'] + np.random.uniform(-jitter, jitter, size=len(yield_df))
    
    # Create scatter plot
    sns.scatterplot(x='jittered_pos', y='yield_kg_ha', hue='crop_type', data=yield_df, palette='deep', ax=ax, alpha=0.7)
    
    # Set x-ticks to treatment names
    ax.set_xticks(range(len(unique_treatments)))
    ax.set_xticklabels([TREATMENT_NAMES[t] for t in unique_treatments], rotation=45, ha='right')
    
    style_axis(ax, 
              title='Individual Plot Yields by Treatment and Crop Type',
              xlabel='Treatment',
              ylabel='Yield (kg/ha)')
    style_legend(ax, title='Crop Type', loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 10b: Individual Plot Yields by Treatment and Crop Type added to PDF.")

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

    # Map treatment numbers to names
    efficiency_df['treatment_name'] = efficiency_df['treatment'].map(TREATMENT_NAMES)

    # Calculate IWUE and CWUE
    # IWUE = Yield / Irrigation Applied
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

    # Plot IWUE by treatment
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='treatment_name', y='IWUE', data=efficiency_df, 
                hue='treatment_name', palette='Set3', legend=False, ax=ax)
    add_error_bars(ax, efficiency_df, 'treatment_name', 'IWUE')
    style_axis(ax, 
              title='IWUE by Treatment',
              xlabel='Treatment',
              ylabel='Irrigation Water Use Efficiency (kg/ha per mm)')
    add_statistical_annotations_anova_tukey(ax, efficiency_df, 'treatment_name', 'IWUE')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11a: IWUE by Treatment added to PDF.")

    # Plot CWUE by treatment
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='treatment_name', y='CWUE', data=efficiency_df, 
                hue='treatment_name', palette='Set3', legend=False, ax=ax)
    add_error_bars(ax, efficiency_df, 'treatment_name', 'CWUE')
    style_axis(ax, 
              title='CWUE by Treatment',
              xlabel='Treatment',
              ylabel='Crop Water Use Efficiency (kg/ha per mm ETo)')
    add_statistical_annotations_anova_tukey(ax, efficiency_df, 'treatment_name', 'CWUE')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11b: CWUE by Treatment added to PDF.")

    # Scatter plot: Applied water vs Yield
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='irrigation_applied_mm', y='yield_kg_ha', hue='treatment_name', data=efficiency_df, palette='deep', ax=ax)
    style_axis(ax, 
              title='Relationship between Applied Water and Yield by Treatment',
              xlabel='Irrigation Applied (mm)',
              ylabel='Yield (kg/ha)')
    style_legend(ax, title='Treatment', loc='upper right')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11c: Applied Water vs Yield added to PDF.")

def generate_figure12(conn, pdf):
    """
    Figure 12: Representative Days Analysis (Wet, Dry, Normal)
    Shows canopy temperature, soil moisture at various depths, and irrigation events for selected days.
    """
    label = "Figure 12: Representative Days Analysis (Wet, Dry, Normal)"
    print(f"Generating {label}")

    try:
        # Get weather data with proper CASE statements
        weather_query = """
        SELECT 
            date(timestamp) as date,
            AVG(CASE WHEN variable_name = 'Ta_2m_Avg' THEN value END) as avg_temp,
            AVG(CASE WHEN variable_name = 'RH_2m_Avg' THEN value END) as avg_rh,
            SUM(CASE WHEN variable_name = 'Rain_1m_Tot' THEN value ELSE 0 END) as daily_rain
        FROM data
        WHERE variable_name IN ('Ta_2m_Avg', 'RH_2m_Avg', 'Rain_1m_Tot')
        AND date(timestamp) BETWEEN '2024-08-21' AND '2024-08-23'
        GROUP BY date(timestamp)
        """
        
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Get hourly data for the representative days
        hourly_query = """
        SELECT 
            datetime(timestamp) as timestamp,
            date(timestamp) as date,
            strftime('%H', timestamp) as hour,
            CASE 
                WHEN variable_name = 'Ta_2m_Avg' THEN value 
                ELSE NULL 
            END as temperature,
            CASE 
                WHEN variable_name = 'RH_2m_Avg' THEN value 
                ELSE NULL 
            END as humidity
        FROM data
        WHERE variable_name IN ('Ta_2m_Avg', 'RH_2m_Avg')
        AND date(timestamp) IN (
            '2024-08-22',  -- wet day
            '2024-08-06'   -- dry day
        )
        ORDER BY timestamp
        """
        
        hourly_df = pd.read_sql_query(hourly_query, conn)
        
        # Process the hourly data
        hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
        hourly_df['hour'] = hourly_df['timestamp'].dt.hour
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot temperature
        for date in ['2024-08-22', '2024-08-06']:
            day_data = hourly_df[hourly_df['date'] == date]
            label = 'Wet Day' if date == '2024-08-22' else 'Dry Day'
            ax1.plot(day_data['hour'], day_data['temperature'], 
                    label=label, marker='o')
        
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot humidity
        for date in ['2024-08-22', '2024-08-06']:
            day_data = hourly_df[hourly_df['date'] == date]
            label = 'Wet Day' if date == '2024-08-22' else 'Dry Day'
            ax2.plot(day_data['hour'], day_data['humidity'], 
                    label=label, marker='o')
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Relative Humidity (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle(label)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"{label} added to PDF.")

    except Exception as e:
        print(f"Error in generate_figure12: {str(e)}")
        plt.close('all')
        raise

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
            generate_figure12(conn, pdf)
            # Skip figure 13 for now
            # generate_figure13(conn, pdf)  # Commented out until neutron probe data is available

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
