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

def process_treatment_two(df, plot):
    if 'cwsi' not in df.columns or 'swsi' not in df.columns:
        logger.warning(f"Missing CWSI or SWSI data for treatment two, plot {plot}. Skipping.")
        return None
    
    cwsi_data, _, _ = get_recent_values(df['cwsi'], 3)
    swsi_data, _, _ = get_recent_values(df['swsi'], 3)
    
    if cwsi_data.empty or swsi_data.empty:
        logger.warning(f"Missing non-zero CWSI or SWSI data for treatment two, plot {plot}. Skipping.")
        return None

    cwsi_avg = cwsi_data.mean()
    swsi_avg = swsi_data.mean()
    
    final_value = cwsi_avg * 0.4 + swsi_avg * 0.6
    recommendation = 'Irrigate' if final_value > 0.5 else 'Do not irrigate'
    
    logger.info(f"Processing treatment two for plot {plot}:")
    logger.info(f"CWSI Avg: {cwsi_avg:.2f}")
    logger.info(f"SWSI Avg: {swsi_avg:.2f}")
    logger.info(f"Final Value: {final_value:.2f}")
    logger.info(f"Recommendation: {recommendation}")
    
    return {
        'plot': plot,
        'treatment': 2,
        'cwsi_avg': cwsi_avg,
        'swsi_avg': swsi_avg,
        'final_value': final_value,
        'recommendation': recommendation
    }

def process_treatment_three(df, plot):
    if 'cwsi' not in df.columns:
        logger.warning(f"Missing CWSI data for treatment three, plot {plot}. Skipping.")
        return None
    
    cwsi_data, _, _ = get_recent_values(df['cwsi'], 4)
    
    if cwsi_data.empty:
        logger.warning(f"Missing non-zero CWSI data for treatment three, plot {plot}. Skipping.")
        return None

    cwsi_avg = cwsi_data.mean()
    
    recommendation = 'Irrigate' if cwsi_avg > 0.5 else 'Do not irrigate'
    
    logger.info(f"Processing treatment three for plot {plot}:")
    logger.info(f"CWSI Avg: {cwsi_avg:.2f}")
    logger.info(f"Recommendation: {recommendation}")
    
    return {
        'plot': plot,
        'treatment': 3,
        'cwsi_avg': cwsi_avg,
        'recommendation': recommendation
    }

def process_treatment_four(df, plot):
    if 'swsi' not in df.columns:
        logger.warning(f"Missing SWSI data for treatment four, plot {plot}. Skipping.")
        return None
    
    swsi_data, _, _ = get_recent_values(df['swsi'], 3)
    
    if swsi_data.empty:
        logger.warning(f"Missing non-zero SWSI data for treatment four, plot {plot}. Skipping.")
        return None

    swsi_avg = swsi_data.mean()
    
    recommendation = 'Irrigate' if swsi_avg > 0.5 else 'Do not irrigate'
    
    logger.info(f"Processing treatment four for plot {plot}:")
    logger.info(f"SWSI Avg: {swsi_avg:.2f}")
    logger.info(f"Recommendation: {recommendation}")
    
    return {
        'plot': plot,
        'treatment': 4,
        'swsi_avg': swsi_avg,
        'recommendation': recommendation
    }

def process_csv_file(file_path):
    logger.info(f"Processing file for irrigation recommendation: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    df.set_index('TIMESTAMP', inplace=True)
    
    plot_number = file_path.split('_')[-2]  # Assuming the plot number is the second-to-last part of the filename
    treatment = int(file_path.split('_')[2][3])  # Assuming the treatment number is in the format 'trtX'

    if treatment == 2:
        return process_treatment_two(df, plot_number)
    elif treatment == 3:
        return process_treatment_three(df, plot_number)
    elif treatment == 4:
        return process_treatment_four(df, plot_number)
    else:
        logger.warning(f"Unsupported treatment {treatment} for plot {plot_number}. Skipping.")
        return None

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
