import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd

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
