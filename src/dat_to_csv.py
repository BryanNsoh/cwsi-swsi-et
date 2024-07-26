import os
import re
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta

def load_sensor_mapping(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_dat_file(file_name):
    with open(file_name, "r") as file:
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
    
    # Correct column names if necessary
    if 'TDR5006B11724' in data.columns:
        data['TDR5006B11824'] = data['TDR5006B11724']
        data.drop('TDR5006B11724', axis=1, inplace=True)
    
    if 'TDR5026A23824' in data.columns:
        data['TDR5026A23024'] = data['TDR5026A23824']
        data.drop('TDR5026A23824', axis=1, inplace=True)
    
    # Resample to hourly data
    data_hourly = data.set_index("TIMESTAMP").resample('h').mean().reset_index()
    
    return data_hourly.sort_values("TIMESTAMP")  # Sort by timestamp

def get_dat_files(folder_path, crop_type):
    if crop_type == 'corn':
        patterns = [r'nodeA.*\.dat', r'nodeB.*\.dat', r'nodeC.*\.dat']
    elif crop_type == 'soybean':
        patterns = [r'SoyNodeA.*_NodeA\.dat', r'SoyNodeB.*_NodeB\.dat', r'SoyNodeC.*_NodeC\.dat']
    
    dat_files = []
    for file in os.listdir(folder_path):
        for pattern in patterns:
            if re.match(pattern, file, re.IGNORECASE):
                dat_files.append(os.path.join(folder_path, file))
                break
    return dat_files

def parse_weather_csv(filename):
    df = pd.read_csv(
        filename,
        header=1,
        skiprows=[2, 3],
        parse_dates=["TIMESTAMP"],
        date_format="%Y-%m-%d %H:%M:%S",
        low_memory=False
    )
    
    df = df.rename(columns=lambda x: x.strip())
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"])
    df = df.apply(pd.to_numeric, errors="coerce")

    # Select only the required weather columns
    weather_columns = ['TIMESTAMP', 'Ta_2m_Avg', 'RH_2m_Avg', 'WndAveSpd_3m', 'WndAveDir_3m', 'PresAvg_1pnt5m', 'Solar_2m_Avg', 'Rain_1m_Tot', 'TaMax_2m', 'TaMin_2m', 'RHMax_2m', 'RHMin_2m']
    df = df[weather_columns]

    # Ensure TIMESTAMP is recognized as datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # Shift timestamps back by 5 hours
    df['TIMESTAMP'] = df['TIMESTAMP'] - timedelta(hours=5)

    # Resample to hourly data
    df_hourly = df.set_index("TIMESTAMP").resample('h').mean().reset_index()
    
    return df_hourly.sort_values("TIMESTAMP")  # Sort by timestamp

def process_folder(folder_path, sensor_mapping, crop_type, output_folder, weather_data):
    dat_files = get_dat_files(folder_path, crop_type)
    
    for dat_file in dat_files:
        if os.path.exists(dat_file):
            print(f"Processing file: {dat_file}")
            df = parse_dat_file(dat_file)
            crop_specific_mapping = [sensor for sensor in sensor_mapping if sensor['field'] == f'LINEAR_{crop_type.upper()}']
            process_and_save_data(df, crop_specific_mapping, crop_type, output_folder, weather_data)
        else:
            print(f"File not found: {dat_file}")

def process_and_save_data(df, sensor_mapping, crop_type, output_folder, weather_data):
    sensor_groups = {}
    for sensor in sensor_mapping:
        key = (sensor['treatment'], sensor['plot_number'], sensor['field'])
        if key not in sensor_groups:
            sensor_groups[key] = []
        sensor_groups[key].append(sensor['sensor_id'])

    for (treatment, plot_number, field), sensors in sensor_groups.items():
        columns_to_save = ['TIMESTAMP'] + [s for s in sensors if s in df.columns]
        df_to_save = df[columns_to_save].dropna(subset=columns_to_save[1:], how='all')
        
        if not df_to_save.empty:
            # Get the date range where at least one sensor has data
            start_date = df_to_save['TIMESTAMP'].min()
            end_date = df_to_save['TIMESTAMP'].max()

            # Filter weather data to match the sensor data range
            weather_data_filtered = weather_data[
                (weather_data['TIMESTAMP'] >= start_date) &
                (weather_data['TIMESTAMP'] <= end_date)
            ]

            # Merge with filtered weather data
            merged_df = pd.merge(df_to_save, weather_data_filtered, on='TIMESTAMP', how='outer')
            merged_df = merged_df.sort_values("TIMESTAMP")

            file_name = f"{field}_trt{treatment}_plot_{plot_number}_{datetime.now().strftime('%Y%m%d')}.csv"
            output_path = os.path.join(output_folder, file_name)
            merged_df.to_csv(output_path, index=False)
            print(f"Saved data to {output_path}")
        else:
            print(f"No data to save for {field} plot {plot_number}")

def create_dated_folder(base_path):
    current_date = datetime.now().strftime("%Y-%m-%d")
    dated_folder = os.path.join(base_path, f"data-{current_date}")
    os.makedirs(dated_folder, exist_ok=True)
    return dated_folder

def main(corn_folders, soybean_folders, sensor_mapping_path, output_folder, weather_csv_path):
    sensor_mapping = load_sensor_mapping(sensor_mapping_path)
    
    dated_output_folder = create_dated_folder(output_folder)
    
    # Load and process weather data
    weather_data = parse_weather_csv(weather_csv_path)
    
    print("Processing corn data")
    for folder in corn_folders:
        print(f"Processing corn folder: {folder}")
        process_folder(folder, sensor_mapping, crop_type='corn', output_folder=dated_output_folder, weather_data=weather_data)
    
    print("Processing soybean data")
    for folder in soybean_folders:
        print(f"Processing soybean folder: {folder}")
        process_folder(folder, sensor_mapping, crop_type='soybean', output_folder=dated_output_folder, weather_data=weather_data)

if __name__ == "__main__":
    corn_folders = [
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-03-2024",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-08-2024-discontinuous",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-14-2024-discont-nodeC only",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-15-2024-discont-unsure",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-19-2024"
    ]
    soybean_folders = [
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-15-24",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-19-2024"
    ]
    sensor_mapping_path = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\sensor_mapping.yaml"
    output_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    weather_csv_path = r"C:\Users\bnsoh2\Downloads\North_Platte_3SW_Beta_1min (8).csv"
    
    main(corn_folders, soybean_folders, sensor_mapping_path, output_folder, weather_csv_path)