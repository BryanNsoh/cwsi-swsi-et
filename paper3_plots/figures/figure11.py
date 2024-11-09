import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS, TREATMENT_NAMES
import pandas as pd
import numpy as np

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
    add_statistical_annotations(ax, efficiency_df, 'treatment_name', 'IWUE')
    style_axis(ax, 
              title='IWUE by Treatment',
              xlabel='Treatment',
              ylabel='Irrigation Water Use Efficiency (kg/ha per mm)')
    fig.suptitle(label, fontsize=24, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Figure 11a: IWUE by Treatment added to PDF.")

    # Plot CWUE by treatment
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='treatment_name', y='CWUE', data=efficiency_df, 
                hue='treatment_name', palette='Set3', legend=False, ax=ax)
    add_statistical_annotations(ax, efficiency_df, 'treatment_name', 'CWUE')
    style_axis(ax, 
              title='CWUE by Treatment',
              xlabel='Treatment',
              ylabel='Crop Water Use Efficiency (kg/ha per mm ETo)')
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
