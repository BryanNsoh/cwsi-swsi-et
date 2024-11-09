import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS, TREATMENT_NAMES
import pandas as pd
import numpy as np

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
    add_statistical_annotations(ax, yield_df, 'treatment_name', 'yield_kg_ha')
    style_axis(ax, 
              title='Yield Distribution by Treatment',
              xlabel='Treatment',
              ylabel='Yield (kg/ha)')
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
