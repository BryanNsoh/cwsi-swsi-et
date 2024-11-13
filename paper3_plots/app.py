# app.py

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io

# ================================
# Streamlit App Configuration
# ================================

st.set_page_config(
    page_title="Irrigation and Stress Index Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Define the output directory for the PDF (not used in Streamlit)
# OUTPUT_DIR = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\paper3_plots"

# Define the PDF output path (use in-memory buffer)
# PDF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "all_figures.pdf")

# ================================
# Utility Functions
# ================================

def connect_db(db_path):
    """Establish a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

@st.cache_resource
def get_connection():
    """Cached database connection."""
    return connect_db(DATABASE_PATH)

def filter_valid_dates(df, date_column='date'):
    """Filter out records with invalid or out-of-range dates."""
    # Convert to datetime, coerce errors to NaT
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    initial_count = len(df)
    # Keep only dates from 2024 onwards
    df = df[df[date_column].dt.year >= 2024]
    filtered_count = len(df)
    return df

def add_statistical_annotations_anova_tukey(ax, df, treatment_col, yield_col):
    """
    Perform ANOVA and Tukey's HSD test and annotate the plot.
    """
    try:
        # Perform one-way ANOVA
        groups = df.groupby(treatment_col)[yield_col].apply(list)
        f_stat, p_val = stats.f_oneway(*groups)
        
        # Add ANOVA p-value to the plot
        ax.text(0.95, 0.95, f'ANOVA p={p_val:.3f}', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # If significant, perform Tukey's HSD
        if p_val < 0.05:
            tukey = pairwise_tukeyhsd(endog=df[yield_col], groups=df[treatment_col], alpha=0.05)
            # You can add more sophisticated annotations here if needed
            # For simplicity, we just note that Tukey's test was performed
            ax.text(0.95, 0.90, 'Tukey HSD performed', transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    except Exception as e:
        st.warning(f"Error in statistical annotations: {e}")
    return ax

def add_error_bars(ax, df, treatment_col, yield_col):
    """Add standard error bars to yield plots."""
    try:
        stats_df = df.groupby(treatment_col).agg({
            yield_col: ['mean', 'std', 'count']
        }).reset_index()
        stats_df.columns = ['treatment', 'mean', 'std', 'count']
        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        treatments = sorted(stats_df['treatment'].unique())
        means = stats_df['mean']
        ses = stats_df['se']
        ax.errorbar(x=range(len(treatments)), y=means, yerr=ses, fmt='none', capsize=5, color='black')
    except Exception as e:
        st.warning(f"Error adding error bars: {e}")
    return ax

def calc_vpd(temp, rh):
    """
    Calculate Vapor Pressure Deficit (VPD) given temperature and relative humidity.
    VPD (kPa) = saturation vapor pressure - actual vapor pressure
    """
    try:
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        vpd = es * (1 - rh / 100)
        return vpd
    except Exception as e:
        st.warning(f"Error calculating VPD for temp={temp}, rh={rh}: {e}")
        return np.nan

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

def update_treatment_labels(ax):
    """
    Update treatment labels to actual treatment names.
    """
    ax.set_xticks(range(len(TREATMENT_NAMES)))
    ax.set_xticklabels([TREATMENT_NAMES[i] for i in sorted(TREATMENT_NAMES.keys())], rotation=45, ha='right')
    return ax

# ================================
# Plotting Functions
# ================================

def generate_representative_days_analysis(conn):
    """
    Generate Figure 12: Representative Days Analysis (Wet, Dry, Normal)
    Returns the Matplotlib figure.
    """
    label = "Figure 12: Representative Days Analysis (Wet, Dry, Normal)"
    try:
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

        # 2. Dry: Longest period without rain
        dry_days_query = """
        SELECT date(timestamp) as date
        FROM data
        WHERE variable_name = 'Rain_1m_Tot'
        GROUP BY date
        HAVING SUM(value) = 0
        ORDER BY date
        """
        dry_days_df = pd.read_sql_query(dry_days_query, conn)
        dry_day = None
        if not dry_days_df.empty:
            # Convert to datetime
            dry_days_dt = pd.to_datetime(dry_days_df['date'])
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
            dry_day = longest_streak[-1] if longest_streak else None

        # 3. Normal: Day with average VPD
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

        # Define the representative days
        representative_days = {
            'Wet': wet_day,
            'Dry': dry_day,
            'Normal': normal_day
        }

        # Initialize PDF buffer
        buffer = io.BytesIO()
        pdf = PdfPages(buffer)

        # Iterate over each representative day and plot
        for condition, day in representative_days.items():
            if day is None:
                st.warning(f"No {condition} day found. Skipping.")
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
            SELECT date(timestamp) as date, AVG(CASE WHEN variable_name = 'Ta_2m_Avg' THEN value END) as avg_temp, 
                   AVG(CASE WHEN variable_name = 'RH_2m_Avg' THEN value END) as avg_rh
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
            else:
                st.warning(f"No Canopy Temperature data available for {condition} day.")

            # Plot soil moisture at different depths
            for depth in [6, 18, 30, 42]:
                depth_data = tdr_df[tdr_df['Depth_cm'] == depth]
                if not depth_data.empty:
                    sns.lineplot(x='timestamp', y='value', data=depth_data, label=f'Soil Moisture {depth} cm', ax=ax1)
                else:
                    st.warning(f"No Soil Moisture data available for {depth} cm on {condition} day.")

            ax1.set_title(f'{condition} Day Analysis: {day}', fontsize=20)
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
            else:
                st.warning(f"No VPD data available for {condition} day.")

            # Highlight irrigation events
            if not irrigation_df.empty:
                for _, row in irrigation_df.iterrows():
                    ax1.axvline(x=row['date'], color=CUSTOM_COLORS['blue'], linestyle=':', alpha=0.7)
                    ax1.text(row['date'], ax1.get_ylim()[1], 'Irrigation', rotation=90, verticalalignment='top', fontsize=12, color=CUSTOM_COLORS['blue'])
            else:
                st.warning(f"No Irrigation Events to highlight for {condition} day.")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        pdf.close()
        buffer.seek(0)
        return buffer

def generate_figure3(conn):
    """
    Generate Figure 3: Seasonal Weather Patterns and Environmental Variables
    Returns the Matplotlib figure.
    """
    label = "Figure 3: Seasonal Weather Patterns and Environmental Variables"
    try:
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

        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

def generate_figure4(conn):
    """
    Generate Figure 4: CWSI Implementation Analysis
    Returns the Matplotlib figure.
    """
    label = "Figure 4: CWSI Implementation Analysis"
    try:
        # Corn CWSI Analysis
        corn_query = """
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
        corn_cwsi = pd.read_sql_query(corn_query, conn)
        corn_cwsi = corn_cwsi.rename(columns={'growth_stage': 'Growth Stage'})

        # Create boxplot for corn
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Growth Stage', y='cwsi', data=corn_cwsi, 
                   hue='Growth Stage', palette='Set3', legend=False, ax=ax)
        style_axis(ax, 
                  title='CWSI Distribution by Growth Stage (Corn)',
                  xlabel='Growth Stage',
                  ylabel='CWSI')
        plt.tight_layout()
        fig.suptitle(label, fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.9)

        return fig

    except Exception as e:
        st.error(f"Error generating {label}: {e}")
        st.text(traceback.format_exc())
        plt.close('all')
        return None

def generate_figure5(conn):
    """
    Generate Figure 5: Soil Moisture Monitoring and SWSI Calculation
    Returns the Matplotlib figure.
    """
    label = "Figure 5: Soil Moisture Monitoring and SWSI Calculation"
    try:
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
        tdr_df['Depth_cm'] = tdr_df['variable_name'].str.extract(r'TDR\d{4}[A-C]\d(\d{2})\d{2}$')[0]
        tdr_df['Depth_cm'] = pd.to_numeric(tdr_df['Depth_cm'], errors='coerce')
        tdr_df = tdr_df.dropna(subset=['Depth_cm'])
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Calculate SWSI (Assuming SWSI = normalized soil moisture)
        min_moisture = soil_moisture['value'].min()
        max_moisture = soil_moisture['value'].max()
        soil_moisture['SWSI'] = (soil_moisture['value'] - min_moisture) / (max_moisture - min_moisture)

        # Aggregate SWSI by date
        swsi_daily = soil_moisture.groupby('date')['SWSI'].mean().reset_index()

        # Plot SWSI over time with uncertainty bands (e.g., ±0.05)
        fig_sws, ax_sws = plt.subplots(figsize=(15, 6))
        sns.lineplot(x='date', y='SWSI', data=swsi_daily, label='SWSI', color=CUSTOM_COLORS['brown'], linewidth=2.5, ax=ax_sws)
        ax_sws.fill_between(swsi_daily['date'], swsi_daily['SWSI'] - 0.05, swsi_daily['SWSI'] + 0.05, color=CUSTOM_COLORS['brown'], alpha=0.2)
        style_axis(ax_sws, 
                  title='Soil Water Stress Index Over Time',
                  xlabel='Date',
                  ylabel='Soil Water Stress Index (SWSI)')
        style_legend(ax_sws, loc='upper right')
        fig_sws.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Combine both figures vertically
        combined_fig = plt.figure(figsize=(15, 20))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = combined_fig.add_subplot(gs[0])
        for line in fig.axes[0].lines:
            ax1.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), linewidth=2.5)
        ax1.fill_between(soil_moisture['date'], 0, 0, color='none')  # Dummy fill
        style_axis(ax1, 
                  title='Soil Moisture at Different Depths Over Time',
                  xlabel='Date',
                  ylabel='Soil Moisture (%)')
        ax1.legend(loc='upper right')

        ax2 = combined_fig.add_subplot(gs[1])
        for line in fig_sws.axes[0].lines:
            ax2.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), linewidth=2.5)
        ax2.fill_between(swsi_daily['date'], swsi_daily['SWSI'] - 0.05, swsi_daily['SWSI'] + 0.05, color=CUSTOM_COLORS['brown'], alpha=0.2)
        style_axis(ax2, 
                  title='Soil Water Stress Index Over Time',
                  xlabel='Date',
                  ylabel='Soil Water Stress Index (SWSI)')
        ax2.legend(loc='upper right')

        combined_fig.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return combined_fig

def generate_figure6(conn):
    """
    Generate Figure 6: CWSI and SWSI Analysis
    Returns the Matplotlib figure.
    """
    label = "Figure 6: CWSI and SWSI Analysis"
    try:
        # Fetch CWSI data
        cwsi_query = """
        SELECT 
            date(timestamp) as date,
            AVG(value) as cwsi
        FROM data 
        WHERE variable_name = 'cwsi'
        GROUP BY date(timestamp)
        """
        cwsi_df = pd.read_sql_query(cwsi_query, conn)
        if cwsi_df.empty:
            st.warning("No CWSI data found.")
            return None

        # Fetch SWSI data
        swsi_query = """
        SELECT 
            date(timestamp) as date,
            AVG(value) as swsi
        FROM data 
        WHERE variable_name = 'swsi'
        GROUP BY date(timestamp)
        """
        swsi_df = pd.read_sql_query(swsi_query, conn)
        if swsi_df.empty:
            st.warning("No SWSI data found.")
            return None

        # Fetch irrigation events
        irrigation_query = """
        SELECT 
            date(date) as date, 
            SUM(amount_mm) as amount_mm
        FROM irrigation_events
        GROUP BY date(date)
        """
        irrigation_df = pd.read_sql_query(irrigation_query, conn)
        if irrigation_df.empty:
            st.warning("No irrigation events found.")
            return None

        # Process dates
        for df in [cwsi_df, swsi_df, irrigation_df]:
            df['date'] = pd.to_datetime(df['date'])
            
        # Merge CWSI and SWSI data
        indices_df = pd.merge(cwsi_df, swsi_df, on='date', how='inner')

        # Merge with irrigation events
        indices_df = pd.merge(indices_df, irrigation_df, on='date', how='left')
        indices_df['amount_mm'] = indices_df['amount_mm'].fillna(0)

        # Create figure
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.5, 1])

        # Panel 1: Correlation
        ax1 = fig.add_subplot(gs[0])
        sns.scatterplot(
            data=indices_df,
            x='swsi',
            y='cwsi',
            alpha=0.5,
            color=CUSTOM_COLORS['accent_purple'],
            ax=ax1
        )

        # Add regression line
        sns.regplot(
            data=indices_df,
            x='swsi',
            y='cwsi',
            scatter=False,
            color=CUSTOM_COLORS['red'],
            ax=ax1
        )

        # Add correlation coefficient
        corr = indices_df['swsi'].corr(indices_df['cwsi'])
        ax1.text(
            0.05, 0.95, 
            f'Correlation: {corr:.2f}',
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )
        style_axis(ax1, 
                  title='Correlation between SWSI and CWSI',
                  xlabel='SWSI',
                  ylabel='CWSI')

        # Panel 2: Time series
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(
            indices_df['date'],
            indices_df['cwsi'],
            label='CWSI',
            color=CUSTOM_COLORS['blue'],
            linewidth=2
        )
        ax2.plot(
            indices_df['date'],
            indices_df['swsi'],
            label='SWSI',
            color=CUSTOM_COLORS['green'],
            linewidth=2
        )

        # Add threshold line
        ax2.axhline(
            y=0.5,
            color='red',
            linestyle='--',
            label='Threshold (0.5)'
        )

        # Mark irrigation events
        for _, event in irrigation_df.iterrows():
            ax2.axvline(
                x=event['date'],
                color='gray',
                alpha=0.3,
                linestyle=':'
            )
            ax2.text(
                event['date'],
                ax2.get_ylim()[1],
                f"{event['amount_mm']:.1f}mm",
                rotation=90,
                verticalalignment='bottom'
            )

        style_axis(ax2,
                  title='CWSI and SWSI Over Time',
                  xlabel='Date',
                  ylabel='Index Value')
        style_legend(ax2)

        # Panel 3: Irrigation day analysis
        ax3 = fig.add_subplot(gs[2])

        # Get index values only on irrigation days
        irrigation_dates = irrigation_df['date'].unique()
        irrigation_indices = indices_df[indices_df['date'].isin(irrigation_dates)].copy()

        # Plot values on irrigation days
        ax3.scatter(
            irrigation_indices['date'],
            irrigation_indices['cwsi'],
            label='CWSI',
            color=CUSTOM_COLORS['blue'],
            marker='o',
            s=100
        )
        ax3.scatter(
            irrigation_indices['date'],
            irrigation_indices['swsi'],
            label='SWSI',
            color=CUSTOM_COLORS['green'],
            marker='s',
            s=100
        )

        # Add threshold
        ax3.axhline(
            y=0.5,
            color='red',
            linestyle='--',
            label='Threshold'
        )

        style_axis(ax3,
                  title='Index Values on Irrigation Days',
                  xlabel='Date',
                  ylabel='Index Value')
        style_legend(ax3)

        # Format dates on all x-axes
        for ax in [ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=7))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save figure
        return fig

    except Exception as e:
        st.error(f"Error generating {label}: {e}")
        st.text(traceback.format_exc())
        plt.close('all')
        return None

def generate_figure7(conn):
    """
    Generate Figure 7: System Response Analysis
    Returns the Matplotlib figure.
    """
    label = "Figure 7: System Response Analysis"
    try:
        # Get IRT (canopy temperature) data
        irt_query = """
        SELECT date(timestamp) as date, AVG(value) as canopy_temp
        FROM data
        WHERE variable_name LIKE 'IRT%'
        GROUP BY date(timestamp)
        """
        irt_df = pd.read_sql_query(irt_query, conn)
        irt_df['date'] = pd.to_datetime(irt_df['date']).dt.date

        # Get SWSI data
        swsi_query = """
        SELECT date(timestamp) as date, AVG(value) as swsi
        FROM data
        WHERE variable_name = 'swsi'
        GROUP BY date(timestamp)
        """
        swsi_df = pd.read_sql_query(swsi_query, conn)
        swsi_df['date'] = pd.to_datetime(swsi_df['date']).dt.date

        # Get irrigation events
        irrigation_query = """
        SELECT date(date) as date, SUM(amount_mm) as irrigation_mm
        FROM irrigation_events
        GROUP BY date(date)
        """
        irrigation_df = pd.read_sql_query(irrigation_query, conn)
        irrigation_df['date'] = pd.to_datetime(irrigation_df['date']).dt.date

        # Merge data
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
                       rotation=90, va='top', fontsize=12, color='green')

        # Formatting
        ax2.set_xlabel('Date')
        fig.suptitle(label, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

def generate_figure8(conn):
    """
    Generate Figure 8: Monthly Stress Index Distributions
    Returns the Matplotlib figure.
    """
    label = "Figure 8: Monthly Stress Index Distributions"
    try:
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    except Exception as e:
        st.error(f"Error generating {label}: {e}")
        st.text(traceback.format_exc())
        plt.close('all')
        return None

def generate_figure9(conn):
    """
    Generate Figure 9: Irrigation Application Patterns
    Returns the Matplotlib figure.
    """
    label = "Figure 9: Irrigation Application Patterns"
    try:
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Calculate cumulative irrigation
        pivot_daily_cumulative = pivot_daily.cumsum()

        # Plot stacked line for cumulative irrigation amounts
        fig_cum, ax_cum = plt.subplots(figsize=(15, 6))
        pivot_daily_cumulative.plot(kind='line', ax=ax_cum, marker='o', linewidth=2.5)
        style_axis(ax_cum, 
                  title='Cumulative Irrigation Amounts by Treatment',
                  xlabel='Date',
                  ylabel='Cumulative Irrigation Amount (mm)')
        style_legend(ax_cum, title='Treatment', loc='upper left')
        fig_cum.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Combine both figures vertically
        combined_fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        ax1 = combined_fig.add_subplot(gs[0])
        for col in pivot_daily.columns:
            ax1.bar(pivot_daily.index, pivot_daily[col], label=col)
        style_axis(ax1, 
                  title='Daily Irrigation Amounts by Treatment',
                  xlabel='Date',
                  ylabel='Irrigation Amount (mm)')
        style_legend(ax1, title='Treatment', loc='upper right')

        ax2 = combined_fig.add_subplot(gs[1])
        for col in pivot_daily_cumulative.columns:
            ax2.plot(pivot_daily_cumulative.index, pivot_daily_cumulative[col], label=col, marker='o')
        style_axis(ax2, 
                  title='Cumulative Irrigation Amounts by Treatment',
                  xlabel='Date',
                  ylabel='Cumulative Irrigation Amount (mm)')
        style_legend(ax2, title='Treatment', loc='upper left')

        combined_fig.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return combined_fig

    except Exception as e:
        st.error(f"Error generating {label}: {e}")
        st.text(traceback.format_exc())
        plt.close('all')
        return None

def generate_figure10(conn):
    """
    Generate Figure 10: Yield Distribution Analysis
    Returns the Matplotlib figure.
    """
    label = "Figure 10: Yield Distribution Analysis"
    try:
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
        yield_df['treatment_name'] = yield_df['treatment'].map(TREATMENT_NAMES)

        # First plot: Box plots of yield by treatment
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='treatment_name', y='yield_kg_ha', data=yield_df, 
                    hue='treatment_name', palette='Set3', legend=False, ax=ax)
        add_error_bars(ax, yield_df, 'treatment_name', 'yield_kg_ha')
        add_statistical_annotations_anova_tukey(ax, yield_df, 'treatment_name', 'yield_kg_ha')
        style_axis(ax, 
                  title='Yield Distribution by Treatment',
                  xlabel='Treatment',
                  ylabel='Yield (kg/ha)')
        fig.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Second plot: Scatter plot of individual plot yields
        fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
        
        # Add jitter to treatment positions for better visibility
        jitter = 0.1
        unique_treatments = sorted(yield_df['treatment'].unique())
        treatment_positions = {t: i for i, t in enumerate(unique_treatments)}
        yield_df['treatment_pos'] = yield_df['treatment'].map(treatment_positions)
        yield_df['jittered_pos'] = yield_df['treatment_pos'] + np.random.uniform(-jitter, jitter, size=len(yield_df))
        
        # Create scatter plot
        sns.scatterplot(x='jittered_pos', y='yield_kg_ha', hue='crop_type', data=yield_df, palette='deep', ax=ax_scatter, alpha=0.7)
        
        # Set x-ticks to treatment names
        ax_scatter.set_xticks(range(len(unique_treatments)))
        ax_scatter.set_xticklabels([TREATMENT_NAMES[t] for t in unique_treatments], rotation=45, ha='right')
        
        style_axis(ax_scatter, 
                  title='Individual Plot Yields by Treatment and Crop Type',
                  xlabel='Treatment',
                  ylabel='Yield (kg/ha)')
        style_legend(ax_scatter, title='Crop Type', loc='upper right')
        fig_scatter.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Combine both figures vertically
        combined_fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        ax1 = combined_fig.add_subplot(gs[0])
        for col in yield_df['treatment_name'].unique():
            subset = yield_df[yield_df['treatment_name'] == col]
            sns.boxplot(x='treatment_name', y='yield_kg_ha', data=subset, palette='Set3', ax=ax1, showfliers=False)
        add_error_bars(ax1, yield_df, 'treatment_name', 'yield_kg_ha')
        add_statistical_annotations_anova_tukey(ax1, yield_df, 'treatment_name', 'yield_kg_ha')
        style_axis(ax1, 
                  title='Yield Distribution by Treatment',
                  xlabel='Treatment',
                  ylabel='Yield (kg/ha)')
        fig_scatter = sns.scatterplot(x='jittered_pos', y='yield_kg_ha', hue='crop_type', data=yield_df, palette='deep', ax=ax1, alpha=0.7)
        style_legend(ax1, title='Crop Type', loc='upper right')

        ax2 = combined_fig.add_subplot(gs[1])
        for col in yield_df['crop_type'].unique():
            subset = yield_df[yield_df['crop_type'] == col]
            sns.scatterplot(x='jittered_pos', y='yield_kg_ha', data=subset, label=col, alpha=0.7, ax=ax2)
        style_axis(ax2, 
                  title='Individual Plot Yields by Treatment and Crop Type',
                  xlabel='Treatment',
                  ylabel='Yield (kg/ha)')
        style_legend(ax2, title='Crop Type', loc='upper right')

        combined_fig.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return combined_fig

def generate_figure11(conn):
    """
    Generate Figure 11: Water Use Efficiency Metrics
    Returns the Matplotlib figure.
    """
    label = "Figure 11: Water Use Efficiency Metrics"
    try:
        # Extract yield and irrigation data with treatment information
        efficiency_query = """
        SELECT y.plot_id, p.treatment, y.yield_kg_ha, y.irrigation_applied_mm
        FROM yields y
        JOIN plots p ON y.plot_id = p.plot_id
        """
        efficiency_df = pd.read_sql_query(efficiency_query, conn)
        efficiency_df['treatment_name'] = efficiency_df['treatment'].map(TREATMENT_NAMES)

        # Calculate IWUE and CWUE
        efficiency_df['IWUE'] = efficiency_df['yield_kg_ha'] / efficiency_df['irrigation_applied_mm']
        efficiency_df['IWUE'] = efficiency_df['IWUE'].replace([np.inf, -np.inf], np.nan)

        # Fetch Evapotranspiration (ETO) Data
        eto_query = """
        SELECT plot_id, SUM(value) as total_eto
        FROM data
        WHERE variable_name = 'eto'
        GROUP BY plot_id
        """
        eto_df = pd.read_sql_query(eto_query, conn)
        efficiency_df = pd.merge(efficiency_df, eto_df, on='plot_id', how='left')
        efficiency_df['CWUE'] = efficiency_df['yield_kg_ha'] / efficiency_df['total_eto']
        efficiency_df['CWUE'] = efficiency_df['CWUE'].replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN in IWUE or CWUE
        efficiency_df = efficiency_df.dropna(subset=['IWUE', 'CWUE'])

        # Plot IWUE by treatment
        fig_iwue, ax_iwue = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='treatment_name', y='IWUE', data=efficiency_df, 
                    hue='treatment_name', palette='Set3', legend=False, ax=ax_iwue)
        add_error_bars(ax_iwue, efficiency_df, 'treatment_name', 'IWUE')
        add_statistical_annotations_anova_tukey(ax_iwue, efficiency_df, 'treatment_name', 'IWUE')
        style_axis(ax_iwue, 
                  title='IWUE by Treatment',
                  xlabel='Treatment',
                  ylabel='Irrigation Water Use Efficiency (kg/ha per mm)')
        fig_iwue.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Plot CWUE by treatment
        fig_cwue, ax_cwue = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='treatment_name', y='CWUE', data=efficiency_df, 
                    hue='treatment_name', palette='Set3', legend=False, ax=ax_cwue)
        add_error_bars(ax_cwue, efficiency_df, 'treatment_name', 'CWUE')
        add_statistical_annotations_anova_tukey(ax_cwue, efficiency_df, 'treatment_name', 'CWUE')
        style_axis(ax_cwue, 
                  title='CWUE by Treatment',
                  xlabel='Treatment',
                  ylabel='Crop Water Use Efficiency (kg/ha per mm ETo)')
        fig_cwue.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Scatter plot: Applied water vs Yield
        fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='irrigation_applied_mm', y='yield_kg_ha', hue='treatment_name', data=efficiency_df, palette='deep', ax=ax_scatter)
        style_axis(ax_scatter, 
                  title='Relationship between Applied Water and Yield by Treatment',
                  xlabel='Irrigation Applied (mm)',
                  ylabel='Yield (kg/ha)')
        style_legend(ax_scatter, title='Treatment', loc='upper right')
        fig_scatter.suptitle(label, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Combine all three figures vertically
        combined_fig = plt.figure(figsize=(15, 18))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

        ax1 = combined_fig.add_subplot(gs[0])
        sns.boxplot(x='treatment_name', y='IWUE', data=efficiency_df, 
                    hue='treatment_name', palette='Set3', legend=False, ax=ax1)
        add_error_bars(ax1, efficiency_df, 'treatment_name', 'IWUE')
        add_statistical_annotations_anova_tukey(ax1, efficiency_df, 'treatment_name', 'IWUE')
        style_axis(ax1, 
                  title='IWUE by Treatment',
                  xlabel='Treatment',
                  ylabel='IWUE (kg/ha per mm)')
        combined_fig.suptitle(label, fontsize=24, fontweight='bold')

        ax2 = combined_fig.add_subplot(gs[1])
        sns.boxplot(x='treatment_name', y='CWUE', data=efficiency_df, 
                    hue='treatment_name', palette='Set3', legend=False, ax=ax2)
        add_error_bars(ax2, efficiency_df, 'treatment_name', 'CWUE')
        add_statistical_annotations_anova_tukey(ax2, efficiency_df, 'treatment_name', 'CWUE')
        style_axis(ax2, 
                  title='CWUE by Treatment',
                  xlabel='Treatment',
                  ylabel='CWUE (kg/ha per mm ETo)')

        ax3 = combined_fig.add_subplot(gs[2])
        sns.scatterplot(x='irrigation_applied_mm', y='yield_kg_ha', hue='treatment_name', data=efficiency_df, palette='deep', ax=ax3)
        style_axis(ax3, 
                  title='Relationship between Applied Water and Yield by Treatment',
                  xlabel='Irrigation Applied (mm)',
                  ylabel='Yield (kg/ha)')
        style_legend(ax3, title='Treatment', loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return combined_fig

# ================================
# Main Streamlit App
# ================================

def main():
    st.title("Irrigation and Stress Index Analysis Dashboard")

    st.sidebar.title("Navigation")
    pages = [
        "Figure 3: Seasonal Weather Patterns",
        "Figure 4: CWSI Implementation Analysis",
        "Figure 5: Soil Moisture Monitoring and SWSI",
        "Figure 6: CWSI and SWSI Analysis",
        "Figure 7: System Response Analysis",
        "Figure 8: Monthly Stress Index Distributions",
        "Figure 9: Irrigation Application Patterns",
        "Figure 10: Yield Distribution Analysis",
        "Figure 11: Water Use Efficiency Metrics",
        "Figure 12: Representative Days Analysis",
        "Download All Figures as PDF"
    ]
    selection = st.sidebar.radio("Go to", pages)

    conn = get_connection()
    if conn is None:
        st.stop()

    if selection == "Figure 3: Seasonal Weather Patterns":
        st.header("Figure 3: Seasonal Weather Patterns and Environmental Variables")
        fig = generate_figure3(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 4: CWSI Implementation Analysis":
        st.header("Figure 4: CWSI Implementation Analysis")
        fig = generate_figure4(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 5: Soil Moisture Monitoring and SWSI":
        st.header("Figure 5: Soil Moisture Monitoring and SWSI Calculation")
        fig = generate_figure5(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 6: CWSI and SWSI Analysis":
        st.header("Figure 6: CWSI and SWSI Analysis")
        fig = generate_figure6(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 7: System Response Analysis":
        st.header("Figure 7: System Response Analysis")
        fig = generate_figure7(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 8: Monthly Stress Index Distributions":
        st.header("Figure 8: Monthly Stress Index Distributions")
        fig = generate_figure8(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 9: Irrigation Application Patterns":
        st.header("Figure 9: Irrigation Application Patterns")
        fig = generate_figure9(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 10: Yield Distribution Analysis":
        st.header("Figure 10: Yield Distribution Analysis")
        fig = generate_figure10(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 11: Water Use Efficiency Metrics":
        st.header("Figure 11: Water Use Efficiency Metrics")
        fig = generate_figure11(conn)
        if fig:
            st.pyplot(fig)

    elif selection == "Figure 12: Representative Days Analysis":
        st.header("Figure 12: Representative Days Analysis (Wet, Dry, Normal)")
        fig_buffer = generate_representative_days_analysis(conn)
        st.download_button(
            label="Download Figure 12 as PDF",
            data=fig_buffer,
            file_name="Figure12_Representative_Days_Analysis.pdf",
            mime="application/pdf"
        )
        st.write("### Representative Days Figures")
        st.write("This PDF contains the plots for Wet, Dry, and Normal representative days.")

    elif selection == "Download All Figures as PDF":
        st.header("Download All Figures as PDF")
        if st.button("Generate and Download PDF"):
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                # Generate each figure and add to PDF
                st.write("Generating Figure 3...")
                fig3 = generate_figure3(conn)
                if fig3:
                    pdf.savefig(fig3, bbox_inches='tight')
                    plt.close(fig3)

                st.write("Generating Figure 4...")
                fig4 = generate_figure4(conn)
                if fig4:
                    pdf.savefig(fig4, bbox_inches='tight')
                    plt.close(fig4)

                st.write("Generating Figure 5...")
                fig5 = generate_figure5(conn)
                if fig5:
                    pdf.savefig(fig5, bbox_inches='tight')
                    plt.close(fig5)

                st.write("Generating Figure 6...")
                fig6 = generate_figure6(conn)
                if fig6:
                    pdf.savefig(fig6, bbox_inches='tight')
                    plt.close(fig6)

                st.write("Generating Figure 7...")
                fig7 = generate_figure7(conn)
                if fig7:
                    pdf.savefig(fig7, bbox_inches='tight')
                    plt.close(fig7)

                st.write("Generating Figure 8...")
                fig8 = generate_figure8(conn)
                if fig8:
                    pdf.savefig(fig8, bbox_inches='tight')
                    plt.close(fig8)

                st.write("Generating Figure 9...")
                fig9 = generate_figure9(conn)
                if fig9:
                    pdf.savefig(fig9, bbox_inches='tight')
                    plt.close(fig9)

                st.write("Generating Figure 10...")
                fig10 = generate_figure10(conn)
                if fig10:
                    pdf.savefig(fig10, bbox_inches='tight')
                    plt.close(fig10)

                st.write("Generating Figure 11...")
                fig11 = generate_figure11(conn)
                if fig11:
                    pdf.savefig(fig11, bbox_inches='tight')
                    plt.close(fig11)

                st.write("Generating Figure 12...")
                fig12_buffer = generate_representative_days_analysis(conn)
                pdf.attach_note("Figure 12: Representative Days Analysis")
                pdf.infodict()['Title'] = 'Irrigation and Stress Index Analysis Figures'
                pdf.infodict()['Author'] = 'Your Name'
                pdf.infodict()['Subject'] = 'Generated by Streamlit App'
                pdf.infodict()['Keywords'] = 'Irrigation, CWSI, SWSI, Water Use Efficiency, Agriculture'
                pdf.infodict()['CreationDate'] = datetime.now()
                pdf.infodict()['ModDate'] = datetime.now()
                # Add Figure 12
                # Assuming Figure 12 has already been saved in fig12_buffer
                pdf.attach_note("Figure 12: Representative Days Analysis")
                fig12 = plt.figure()
                pdf.savefig(fig12_buffer, bbox_inches='tight')
                plt.close(fig12_buffer)

            buffer.seek(0)
            st.download_button(
                label="Download All Figures as PDF",
                data=buffer,
                file_name="All_Figures_Irrigation_Analysis.pdf",
                mime="application/pdf"
            )
            st.success("PDF generated successfully!")

    # Add any additional pages or functionality as needed

if __name__ == "__main__":
    main()
