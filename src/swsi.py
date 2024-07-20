import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOIL_DATA = {
    '8815': {'fc': 0.269, 'pwp': 0.115},
    '8816': {'fc': 0.279, 'pwp': 0.126},
    '8869': {'fc': 0.291, 'pwp': 0.143}
}
AVG_FC = np.mean([data['fc'] for data in SOIL_DATA.values()])
AVG_PWP = np.mean([data['pwp'] for data in SOIL_DATA.values()])
MAD = 0.45

def calculate_soil_properties():
    fc = AVG_FC
    pwp = AVG_PWP
    awc = fc - pwp
    vwct = fc - MAD * awc
    return fc, pwp, awc, vwct

def calculate_swsi(vwc_values):
    valid_vwc = [vwc/100 for vwc in vwc_values if pd.notna(vwc) and vwc != 0]
    
    if len(valid_vwc) < 1:
        return None

    avg_vwc = np.mean(valid_vwc)
    fc, pwp, awc, vwct = calculate_soil_properties()
    
    if avg_vwc < vwct:
        return (vwct - avg_vwc) / (vwct - pwp)
    else:
        return 0

def process_csv_file(file_path):
    logger.info(f"Processing file: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    tdr_columns = [col for col in df.columns if 'TDR' in col]
    if not tdr_columns:
        logger.warning(f"No TDR columns found in {file_path}")
        return None
    
    df['all_tdr_invalid'] = (df[tdr_columns] == 0).all(axis=1) | df[tdr_columns].isnull().all(axis=1)
    df_filtered = df[~df['all_tdr_invalid']].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    df_filtered['swsi'] = df_filtered[tdr_columns].apply(calculate_swsi, axis=1)
    df.loc[df_filtered.index, 'swsi'] = df_filtered['swsi']  # Use .loc to avoid SettingWithCopyWarning
    df = df.drop(columns=['all_tdr_invalid'])
    
    non_zero_tdr = df_filtered[tdr_columns].values.flatten()
    non_zero_tdr = non_zero_tdr[~np.isnan(non_zero_tdr) & (non_zero_tdr != 0)]  # Remove NaN and zero values
    
    valid_swsi = df_filtered['swsi'].dropna()
    
    if len(non_zero_tdr) > 0:
        logger.info(f"TDR stats - Count: {len(non_zero_tdr)}, Mean: {np.mean(non_zero_tdr):.4f}, Min: {np.min(non_zero_tdr):.4f}, Max: {np.max(non_zero_tdr):.4f}")
    else:
        logger.info("No valid TDR values found")
    
    if len(valid_swsi) > 0:
        logger.info(f"SWSI stats - Count: {len(valid_swsi)}, Mean: {valid_swsi.mean():.4f}, Min: {valid_swsi.min():.4f}, Max: {valid_swsi.max():.4f}")
    else:
        logger.info("No valid SWSI values calculated")
    
    df.to_csv(file_path, index=False)
    logger.info(f"Updated SWSI values in file: {file_path}")
    
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