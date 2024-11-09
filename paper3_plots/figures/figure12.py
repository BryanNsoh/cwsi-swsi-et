import matplotlib.pyplot as plt
import seaborn as sns
from paper3_plots.utilities import connect_db, filter_valid_dates, calc_vpd, style_axis, style_legend, CUSTOM_COLORS
import pandas as pd
import numpy as np

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
