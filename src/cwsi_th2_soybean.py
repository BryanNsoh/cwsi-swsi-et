# src/cwsi_th2_soybean.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import time, timedelta
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_cwsi_soybean(row, reference_temp):
    """
    Calculate CWSI for soybeans using the reference temperature method.
    """
    Tc = row['canopy_temp']
    Ta = row['Ta_2m_Avg']
    Tc_ref = reference_temp

    if Tc_ref - Ta == 0:
        return None

    cwsi = (Tc - Tc_ref) / (Tc_ref - Ta)
    return max(0, min(cwsi, 1))  # Clip CWSI between 0 and 1

def process_csv_file(file_path, reference_df):
    logger.info(f"Processing file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    # Find the CanopyTemp_1368 column
    canopy_temp_column = 'CanopyTemp_1368'
    if canopy_temp_column not in df.columns:
        logger.warning(f"No {canopy_temp_column} column found in {file_path}")
        return None
    
    # Filter data between 12 PM and 5 PM
    df['time'] = df['TIMESTAMP'].dt.time
    mask = (df['time'] >= time(12, 0)) & (df['time'] <= time(17, 0))
    
    # Rename CanopyTemp_1368 column to 'canopy_temp' for CWSI calculation
    df.loc[mask, 'canopy_temp'] = df.loc[mask, canopy_temp_column]
    
    # Remove existing CWSI column if it exists
    if 'cwsi' in df.columns:
        df = df.drop(columns=['cwsi'])
    
    # Merge with reference temperature data
    df = pd.merge_asof(df, reference_df, on='TIMESTAMP', direction='nearest')
    
    # Calculate CWSI only for the filtered rows
    df.loc[mask, 'cwsi'] = df.loc[mask].apply(lambda row: calculate_cwsi_soybean(row, row['CanopyTemp_1368']), axis=1)
    
    # Remove temporary columns
    df = df.drop(columns=['time', 'canopy_temp', 'CanopyTemp_1368'])
    
    # Log the range of computed CWSI values
    cwsi_values = df['cwsi'].dropna()
    if not cwsi_values.empty:
        logger.info(f"CWSI value ranges for {file_path}: Min={cwsi_values.min():.4f}, Max={cwsi_values.max():.4f}, Mean={cwsi_values.mean():.4f}")
    else:
        logger.info(f"No CWSI values computed for {file_path}")
    
    # Save the updated DataFrame back to the same CSV file
    df.to_csv(file_path, index=False)
    logger.info(f"Updated CWSI values in file: {file_path}")
    
    return df

def main(input_folder):
    # Find the most recent CanopyTemp_1368 CSV file
    reference_csv_files = glob.glob(os.path.join(input_folder, 'CanopyTemp_1368_*.csv'))
    if not reference_csv_files:
        logger.error("No CanopyTemp_1368 CSV files found in the input folder.")
        return
    
    latest_reference_file = max(reference_csv_files, key=os.path.getctime)
    logger.info(f"Using reference temperature file: {latest_reference_file}")

    # Load reference temperature data
    reference_df = pd.read_csv(latest_reference_file, parse_dates=['TIMESTAMP'])
    reference_df = reference_df[['TIMESTAMP', 'CanopyTemp_1368']]

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv') and 'SOYBEAN' in file.upper():
                file_path = os.path.join(root, file)
                process_csv_file(file_path, reference_df)

if __name__ == "__main__":
    input_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    main(input_folder)