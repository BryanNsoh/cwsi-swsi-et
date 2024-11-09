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
