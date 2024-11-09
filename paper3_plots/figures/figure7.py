import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd
import numpy as np

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
