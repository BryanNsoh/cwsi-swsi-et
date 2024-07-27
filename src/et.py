# src/et.py

import os
import pandas as pd
import numpy as np
import pyet
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# User-defined current crop stage
CURRENT_CROP_STAGE = 'mid-season'  # Options: 'initial', 'development', 'mid-season', 'late-season'

# Metadata
ELEVATION = 876  # meters
LATITUDE = 41.15  # degrees
LONGITUDE = -100.77  # degrees
WIND_HEIGHT = 3  # meters
STEFAN_BOLTZMANN = 5.67e-8
CP = 1005
GRAVITY = 9.81
K = 0.41
CROP_HEIGHT = 1.6
SURFACE_ALBEDO = 0.23

EMERGENCE_DATE = '2023-05-15'  # Format: 'YYYY-MM-DD'

# Crop coefficients and growth stages (days)
CROP_DATA = {
    'corn': {
        'stages': [30, 40, 50, 30],  # [initial, development, mid-season, late-season]
        'kc': [0.3, 1.2, 0.35]  # [kc_ini, kc_mid, kc_end]
    },
    'soybean': {
        'stages': [15, 30, 55, 25],
        'kc': [0.4, 1.15, 0.5]
    }
}

def get_crop_coefficient(crop_type):
    kc_values = CROP_DATA[crop_type]['kc']
    
    if CURRENT_CROP_STAGE == 'initial':
        return kc_values[0]
    elif CURRENT_CROP_STAGE == 'development':
        return np.mean([kc_values[0], kc_values[1]])
    elif CURRENT_CROP_STAGE == 'mid-season':
        return kc_values[1]
    elif CURRENT_CROP_STAGE == 'late-season':
        return np.mean([kc_values[1], kc_values[2]])
    else:
        logger.warning(f"Invalid crop stage: {CURRENT_CROP_STAGE}. Using mid-season Kc.")
        return kc_values[1]

def get_crop_type_from_filename(file_path):
    file_name = os.path.basename(file_path)
    if 'CORN' in file_name.upper():
        return 'corn'
    elif 'SOYBEAN' in file_name.upper():
        return 'soybean'
    else:
        logger.warning(f"Unable to determine crop type from filename: {file_name}")
        return None

def calculate_et(df, crop_type):
    # Resample to daily, handling missing data
    df_daily = df.set_index('TIMESTAMP').resample('D').mean()
    
    required_columns = ['Ta_2m_Avg', 'TaMax_2m', 'TaMin_2m', 'RHMax_2m', 'RHMin_2m', 'WndAveSpd_3m', 'Solar_2m_Avg']
    df_daily = df_daily.dropna(subset=required_columns)
    
    if df_daily.empty:
        logger.warning(f"Dataframe is empty after resampling and dropping NaN values for {crop_type}.")
        return pd.DataFrame(columns=['TIMESTAMP', 'eto', 'etc'])
    
    # Convert solar radiation to MJ/m^2/day
    df_daily['Solar_2m_Avg_MJ'] = df_daily['Solar_2m_Avg'] * 0.0864
    
    lat_rad = LATITUDE * np.pi / 180
    
    # Prepare input data
    inputs = {
        'tmean': df_daily['Ta_2m_Avg'],
        'wind': df_daily['WndAveSpd_3m'],
        'rs': df_daily['Solar_2m_Avg_MJ'],
        'tmax': df_daily['TaMax_2m'],
        'tmin': df_daily['TaMin_2m'],
        'rh': (df_daily['RHMax_2m'] + df_daily['RHMin_2m']) / 2,
        'elevation': ELEVATION,
        'lat': lat_rad
    }
    
    # Calculate ETo
    df_daily['eto'] = pyet.combination.pm_asce(**inputs)
    
    # Get Kc based on current crop stage
    kc = get_crop_coefficient(crop_type)
    
    # Calculate ETc
    df_daily['etc'] = df_daily['eto'] * kc
    df_daily['kc'] = kc
    
    # Reset index to get TIMESTAMP as a column and return required columns
    return df_daily.reset_index()[['TIMESTAMP', 'eto', 'etc', 'kc']]

def process_csv_file(file_path):
    logger.info(f"Processing file for ET calculation: {file_path}")
    
    # Determine crop type from file name
    crop_type = get_crop_type_from_filename(file_path)
    if crop_type is None:
        logger.error(f"Skipping file due to unknown crop type: {file_path}")
        return None
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    et_df = calculate_et(df, crop_type)
    
    if not et_df.empty:
        # Merge ET data back into original dataframe
        df = pd.merge(df, et_df, on='TIMESTAMP', how='left')
        
        # Log ET statistics
        for et_type in ['eto', 'etc']:
            valid_et = df[et_type].dropna()
            if len(valid_et) > 0:
                logger.info(f"{et_type.upper()} stats for {crop_type} - Count: {len(valid_et)}, Mean: {valid_et.mean():.4f}, Min: {valid_et.min():.4f}, Max: {valid_et.max():.4f}")
            else:
                logger.info(f"No valid {et_type.upper()} values calculated for {crop_type}")
        
        # Save updated dataframe
        df.to_csv(file_path, index=False)
        logger.info(f"Updated ET values in file: {file_path}")
    else:
        logger.warning(f"No ET values calculated for file: {file_path}")
    
    return df

def main(input_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_csv_file(file_path)

if __name__ == "__main__":
    input_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    main(input_folder)