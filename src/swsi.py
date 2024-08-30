import os
import pandas as pd
import numpy as np
import logging
import re
from datetime import timedelta
from collections import defaultdict

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

def calculate_swsi(row, df, tdr_columns, tdr_usage_summary):
    valid_vwc = {}
    current_timestamp = row['TIMESTAMP']
    three_days_ago = current_timestamp - timedelta(days=3)
    
    for col in tdr_columns:
        if pd.notna(row[col]) and row[col] != 0:
            valid_vwc[col] = row[col] / 100
            tdr_usage_summary[col]['current'] += 1
        else:
            # Look for the last valid value within the past 3 days
            mask = (df['TIMESTAMP'] <= current_timestamp) & \
                   (df['TIMESTAMP'] > three_days_ago) & \
                   (df[col].notna()) & (df[col] != 0)
            last_valid = df.loc[mask, col].iloc[-1] if mask.any() else None
            if pd.notna(last_valid):
                valid_vwc[col] = last_valid / 100
                tdr_usage_summary[col]['past'] += 1
    
    if len(valid_vwc) < 1:
        tdr_usage_summary['missing_data'] += 1
        return None

    avg_vwc = np.mean(list(valid_vwc.values()))
    fc, pwp, awc, vwct = calculate_soil_properties()
    
    if avg_vwc < vwct:
        return (vwct - avg_vwc) / (vwct - pwp)
    else:
        return 0

def select_tdr_columns(df):
    pattern = r'^TDR\d{4}[A-E][1256](06|18|30)24$'
    tdr_columns = [col for col in df.columns if re.match(pattern, col)]
    logger.info(f"Selected TDR columns: {tdr_columns}")
    return tdr_columns

def process_csv_file(file_path):
    logger.info(f"Processing file: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    tdr_columns = select_tdr_columns(df)
    if not tdr_columns:
        logger.warning(f"No valid TDR columns found in {file_path}")
        return None
    
    tdr_usage_summary = {col: {'current': 0, 'past': 0} for col in tdr_columns}
    tdr_usage_summary['missing_data'] = 0

    df['all_tdr_invalid'] = (df[tdr_columns] == 0).all(axis=1) | df[tdr_columns].isnull().all(axis=1)
    df_filtered = df[~df['all_tdr_invalid']].copy()
    
    df_filtered['swsi'] = df_filtered.apply(lambda row: calculate_swsi(row, df_filtered, tdr_columns, tdr_usage_summary), axis=1)
    df.loc[df_filtered.index, 'swsi'] = df_filtered['swsi']
    df = df.drop(columns=['all_tdr_invalid'])
    
    non_zero_tdr = df_filtered[tdr_columns].values.flatten()
    non_zero_tdr = non_zero_tdr[~np.isnan(non_zero_tdr) & (non_zero_tdr != 0)]
    
    valid_swsi = df_filtered['swsi'].dropna()
    
    logger.info("\nTDR Usage Summary:")
    for col in tdr_columns:
        current = tdr_usage_summary[col]['current']
        past = tdr_usage_summary[col]['past']
        total = current + past
        if total > 0:
            logger.info(f"{col}: Used {total} times ({current} current, {past} past 3 days)")
    logger.info(f"Missing data points: {tdr_usage_summary['missing_data']}")

    if len(non_zero_tdr) > 0:
        logger.info(f"\nTDR stats - Count: {len(non_zero_tdr)}, Mean: {np.mean(non_zero_tdr):.4f}, Min: {np.min(non_zero_tdr):.4f}, Max: {np.max(non_zero_tdr):.4f}")
    else:
        logger.info("\nNo valid TDR values found")
    
    if len(valid_swsi) > 0:
        logger.info(f"\nSWSI stats - Count: {len(valid_swsi)}, Mean: {valid_swsi.mean():.4f}, Min: {valid_swsi.min():.4f}, Max: {valid_swsi.max():.4f}")
    else:
        logger.info("\nNo valid SWSI values calculated")
    
    df.to_csv(file_path, index=False)
    logger.info(f"\nUpdated SWSI values in file: {file_path}")
    
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