# src/other_scheduling.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_recent_values(series, n_days=3):
    end_date = pd.Timestamp.now().floor('D')
    start_date = end_date - pd.Timedelta(days=n_days)
    
    # Ensure the series index is timezone-naive
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    
    recent_data = series.loc[start_date:end_date]
    return recent_data[recent_data.notnull() & (recent_data != 0)], start_date, end_date

def process_treatment_two(df, plot, file_path):
    if 'cwsi' not in df.columns or 'swsi' not in df.columns:
        logger.warning(f"Missing CWSI or SWSI data for treatment two, plot {plot}. Skipping.")
        return None
    
    recommendations = []
    for date, row in df.iterrows():
        cwsi_value = row['cwsi']
        swsi_value = row['swsi']
        
        if pd.notna(cwsi_value) and pd.notna(swsi_value):
            final_value = cwsi_value * 0.4 + swsi_value * 0.6
        elif pd.notna(cwsi_value):
            final_value = cwsi_value
        elif pd.notna(swsi_value):
            final_value = swsi_value
        else:
            final_value = None
        
        recommendation = 'Irrigate' if final_value is not None and final_value > 0.5 else 'Do not irrigate'
        recommendations.append(recommendation)
    
    df['recommendation'] = recommendations
    df.to_csv(file_path)  # Overwrite the original CSV with the new column
    
    # For the summary, we'll use today's values or the last available if today's not present
    today = pd.Timestamp.now().floor('D')
    if today in df.index:
        summary_row = df.loc[today]
    else:
        summary_row = df.iloc[-1]
    
    return {
        'plot': plot,
        'treatment': 2,
        'cwsi_avg': summary_row['cwsi'],
        'swsi_avg': summary_row['swsi'],
        'final_value': final_value,
        'recommendation': summary_row['recommendation'],
        'date': summary_row.name.strftime('%Y-%m-%d')
    }

def process_treatment_three(df, plot, file_path):
    if 'cwsi' not in df.columns:
        logger.warning(f"Missing CWSI data for treatment three, plot {plot}. Skipping.")
        return None
    
    recommendations = []
    for date, row in df.iterrows():
        cwsi_value = row['cwsi']
        
        if pd.notna(cwsi_value):
            recommendation = 'Irrigate' if cwsi_value > 0.5 else 'Do not irrigate'
        else:
            recommendation = "CWSI value is null"
        recommendations.append(recommendation)
    
    df['recommendation'] = recommendations
    df.to_csv(file_path)  # Overwrite the original CSV with the new column
    
    # For the summary, we'll use today's values or the last available if today's not present
    today = pd.Timestamp.now().floor('D')
    if today in df.index:
        summary_row = df.loc[today]
    else:
        summary_row = df.iloc[-1]
    
    return {
        'plot': plot,
        'treatment': 3,
        'cwsi_avg': summary_row['cwsi'],
        'recommendation': summary_row['recommendation'],
        'date': summary_row.name.strftime('%Y-%m-%d')
    }

def process_treatment_four(df, plot, file_path):
    if 'swsi' not in df.columns:
        logger.warning(f"Missing SWSI data for treatment four, plot {plot}. Skipping.")
        return None
    
    recommendations = []
    for date, row in df.iterrows():
        swsi_value = row['swsi']
        
        if pd.notna(swsi_value):
            recommendation = 'Irrigate' if swsi_value > 0.5 else 'Do not irrigate'
        else:
            recommendation = "SWSI value is null"
        recommendations.append(recommendation)
    
    df['recommendation'] = recommendations
    df.to_csv(file_path)  # Overwrite the original CSV with the new column
    
    # For the summary, we'll use today's values or the last available if today's not present
    today = pd.Timestamp.now().floor('D')
    if today in df.index:
        summary_row = df.loc[today]
    else:
        summary_row = df.iloc[-1]
    
    return {
        'plot': plot,
        'treatment': 4,
        'swsi_avg': summary_row['swsi'],
        'recommendation': summary_row['recommendation'],
        'date': summary_row.name.strftime('%Y-%m-%d')
    }

def process_csv_file(file_path):
    logger.info(f"Processing file for irrigation recommendation: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    df.set_index('TIMESTAMP', inplace=True)
    
    plot_number = file_path.split('_')[-2]  # Assuming the plot number is the second-to-last part of the filename
    treatment = int(file_path.split('_')[2][3])  # Assuming the treatment number is in the format 'trtX'

    result = None
    if treatment == 2:
        result = process_treatment_two(df, plot_number, file_path)
    elif treatment == 3:
        result = process_treatment_three(df, plot_number, file_path)
    elif treatment == 4:
        result = process_treatment_four(df, plot_number, file_path)
    else:
        logger.warning(f"Unsupported treatment {treatment} for plot {plot_number}. Skipping.")
    
    return result

def main(input_folder, output_folder):
    recommendations = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                result = process_csv_file(file_path)
                if result:
                    recommendations.append(result)

    # Create recommendations DataFrame and save to CSV
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'recommendations.csv')
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(output_file, index=False)
    logger.info(f"Recommendations saved to {output_file}")

if __name__ == "__main__":
    input_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    output_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\recommendations"
    main(input_folder, output_folder)