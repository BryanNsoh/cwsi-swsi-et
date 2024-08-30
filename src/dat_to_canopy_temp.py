# src/dat_to_canopy_temp.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_irt_temp(sb_temp_c, targ_mv, mC2, mC1, mC0, bC2, bC1, bC0):
    m = mC2 * sb_temp_c**2 + mC1 * sb_temp_c + mC0
    b = bC2 * sb_temp_c**2 + bC1 * sb_temp_c + bC0
    sb_temp_k = sb_temp_c + 273.15
    targ_temp_k = ((sb_temp_k**4) + m * targ_mv + b)**0.25
    return targ_temp_k - 273.15

def parse_dat_file_to_csv_with_temps(file_name, output_csv_1368_path, encoding='ISO-8859-1'):
    with open(file_name, "r", encoding=encoding) as file:
        lines = file.readlines()
    headers = lines[1].strip().split(",")
    data_lines = lines[4:]
    data = pd.DataFrame([line.strip().split(",") for line in data_lines], columns=headers)
    
    data.columns = data.columns.str.replace('"', "").str.replace("RECORD", "RecNbr")
    data.columns = data.columns.str.replace("_Avg", "")
    data = data.replace({"NAN": np.nan, '"NAN"': np.nan})
    data["TIMESTAMP"] = data["TIMESTAMP"].str.replace('"', "")
    
    for col in data.columns:
        if col != "TIMESTAMP":
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
    data = data[~data["TIMESTAMP"].isna()]
    
    # Define the coefficients for each sensor
    sensor_coefficients = {
        '1371': {'mC2': 77988.3, 'mC1': 8788030, 'mC0': 1620750000, 'bC2': 4298, 'bC1': 75731.6, 'bC0': -1569380},
        '1373': {'mC2': 84144.1, 'mC1': 8699300, 'mC0': 1583000000, 'bC2': 7825.17, 'bC1': 167578, 'bC0': 2947160},
        '1368': {'mC2': 119752, 'mC1': 10139500, 'mC0': 1787020000, 'bC2': 3884.19, 'bC1': 133998, 'bC0': -5140420},
        '1369': {'mC2': 118054, 'mC1': 10108500, 'mC0': 1790280000, 'bC2': 4019.44, 'bC1': 135393, 'bC0': -3958680},
        '1378': {'mC2': 110897, 'mC1': 9229850, 'mC0': 1632750000, 'bC2': 5598.07, 'bC1': 58091.5, 'bC0': 779020},
        '1374': {'mC2': 106949, 'mC1': 9387600, 'mC0': 1655170000, 'bC2': 4204.25, 'bC1': 102781, 'bC0': 450621},
        '1379': {'mC2': 80669, 'mC1': 8626800, 'mC0': 1580720000, 'bC2': 4165.68, 'bC1': 87436.7, 'bC0': 1813590},
        '1377': {'mC2': 105966, 'mC1': 9068660, 'mC0': 1634120000, 'bC2': 3816.74, 'bC1': 120624, 'bC0': 1025890}
    }
    
    # Compute canopy temperatures for each sensor
    for sensor, coeffs in sensor_coefficients.items():
        bod_col = f'BodC_{sensor}'
        tarm_col = f'TarmV_{sensor}'
        canopy_temp_col = f'CanopyTemp_{sensor}'
        if bod_col in data.columns and tarm_col in data.columns:
            data[canopy_temp_col] = compute_irt_temp(
                sb_temp_c=data[bod_col],
                targ_mv=data[tarm_col],
                mC2=coeffs['mC2'],
                mC1=coeffs['mC1'],
                mC0=coeffs['mC0'],
                bC2=coeffs['bC2'],
                bC1=coeffs['bC1'],
                bC0=coeffs['bC0']
            )
    
    # Resample to hourly data
    data_hourly = data.set_index("TIMESTAMP").resample('h').mean().reset_index()
    
    # Save the processed data for sensor 1368 to a CSV file
    canopy_temp_1368_df = data_hourly[["TIMESTAMP", "CanopyTemp_1368"]]
    canopy_temp_1368_df.to_csv(output_csv_1368_path, index=False)
    
    return data_hourly

def plot_canopy_temperatures(data, start_date, end_date, sensors, output_path):
    plt.figure(figsize=(14, 7))

    for sensor in sensors:
        canopy_temp_col = f'CanopyTemp_{sensor}'
        if canopy_temp_col in data.columns:
            plt.plot(data["TIMESTAMP"], data[canopy_temp_col], label=f"CanopyTemp {sensor}")

    plt.xlabel("Timestamp")
    plt.ylabel("Canopy Temperature (°C)")
    plt.title(f"Canopy Temperatures from {start_date} to {end_date}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_dat_files(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.dat'):
                dat_file_path = os.path.join(root, file)
                output_csv_1368_path = os.path.join(output_folder, f"CanopyTemp_1368_{os.path.splitext(file)[0]}.csv")
                
                logger.info(f"Processing file: {dat_file_path}")
                processed_data_df = parse_dat_file_to_csv_with_temps(dat_file_path, output_csv_1368_path)
                
                # Plot canopy temperatures for the last 10 days for all sensors
                latest_timestamp = processed_data_df["TIMESTAMP"].max()
                start_date_last_10_days = latest_timestamp - pd.Timedelta(days=10)
                filtered_data_last_10_days = processed_data_df[
                    (processed_data_df["TIMESTAMP"] >= start_date_last_10_days) &
                    (processed_data_df["TIMESTAMP"] <= latest_timestamp)
                ]
                plot_output_path = os.path.join(output_folder, f"CanopyTemp_plot_{os.path.splitext(file)[0]}.png")
                plot_canopy_temperatures(filtered_data_last_10_days, start_date_last_10_days.date(), latest_timestamp.date(), sensors=['1371', '1368', '1377'], output_path=plot_output_path)
                
                logger.info(f"Processed {file} and saved results in {output_folder}")
                
def process_single_dat_file(dat_file_path, output_folder):
    file_name = os.path.basename(dat_file_path)
    output_csv_1368_path = os.path.join(output_folder, f"CanopyTemp_1368_{os.path.splitext(file_name)[0]}.csv")
    
    logger.info(f"Processing file: {dat_file_path}")
    processed_data_df = parse_dat_file_to_csv_with_temps(dat_file_path, output_csv_1368_path)
    
    # Plot canopy temperatures for the last 10 days for all sensors
    latest_timestamp = processed_data_df["TIMESTAMP"].max()
    start_date_last_10_days = latest_timestamp - pd.Timedelta(days=10)
    filtered_data_last_10_days = processed_data_df[
        (processed_data_df["TIMESTAMP"] >= start_date_last_10_days) &
        (processed_data_df["TIMESTAMP"] <= latest_timestamp)
    ]
    plot_output_path = os.path.join(output_folder, f"CanopyTemp_plot_{os.path.splitext(file_name)[0]}.png")
    plot_canopy_temperatures(filtered_data_last_10_days, start_date_last_10_days.date(), latest_timestamp.date(), sensors=['1371', '1368', '1377'], output_path=plot_output_path)
    
    logger.info(f"Processed {file_name} and saved results in {output_folder}")

def main(input_folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for folder in input_folders:
        process_dat_files(folder, output_folder)

if __name__ == "__main__":
    corn_base_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr"
    soybean_base_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr"
    output_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    main([corn_base_folder, soybean_base_folder], output_folder)