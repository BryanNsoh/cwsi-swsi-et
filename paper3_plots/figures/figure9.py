import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS, TREATMENT_NAMES
import pandas as pd
import numpy as np

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
