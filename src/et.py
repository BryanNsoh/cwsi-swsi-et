import os
import pandas as pd
import numpy as np
import pyet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def calculate_et(df):
    # Resample to daily, handling missing data
    df_daily = df.set_index('TIMESTAMP').resample('D').mean()
    
    required_columns = ['Ta_2m_Avg', 'TaMax_2m', 'TaMin_2m', 'RHMax_2m', 'RHMin_2m', 'WndAveSpd_3m', 'Solar_2m_Avg']
    df_daily = df_daily.dropna(subset=required_columns)
    
    if df_daily.empty:
        logger.warning("Dataframe is empty after resampling and dropping NaN values.")
        return pd.DataFrame(columns=['TIMESTAMP', 'et'])
    
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
    
    # Calculate ET
    df_daily['et'] = pyet.combination.pm_asce(**inputs)
    
    # Reset index to get TIMESTAMP as a column and return only TIMESTAMP and et
    return df_daily.reset_index()[['TIMESTAMP', 'et']]

def process_csv_file(file_path):
    logger.info(f"Processing file for ET calculation: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    et_df = calculate_et(df)
    
    if not et_df.empty:
        # Merge ET data back into original dataframe
        df = pd.merge(df, et_df, on='TIMESTAMP', how='left')
        
        # Log ET statistics
        valid_et = df['et'].dropna()
        if len(valid_et) > 0:
            logger.info(f"ET stats - Count: {len(valid_et)}, Mean: {valid_et.mean():.4f}, Min: {valid_et.min():.4f}, Max: {valid_et.max():.4f}")
        else:
            logger.info("No valid ET values calculated")
        
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