import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd
import numpy as np

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
