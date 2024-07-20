import os
import pandas as pd
import numpy as np
import logging
from datetime import time, timedelta

# Configuration
STEFAN_BOLTZMANN = 5.67e-8
CP = 1005
K = 0.41
CROP_HEIGHT = 2.7
SURFACE_ALBEDO = 0.23

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def celsius_to_kelvin(temp_celsius):
    return temp_celsius + 273.15

def saturated_vapor_pressure(temperature_celsius):
    return 0.6108 * np.exp(17.27 * temperature_celsius / (temperature_celsius + 237.3))

def vapor_pressure_deficit(temperature_celsius, relative_humidity):
    es = saturated_vapor_pressure(temperature_celsius)
    ea = es * (relative_humidity / 100)
    return es - ea

def net_radiation(solar_radiation, air_temp_celsius, canopy_temp_celsius, surface_albedo=0.23, emissivity_a=0.85, emissivity_c=0.98):
    air_temp_kelvin = celsius_to_kelvin(air_temp_celsius)
    canopy_temp_kelvin = celsius_to_kelvin(canopy_temp_celsius)
    Rns = (1 - surface_albedo) * solar_radiation
    Rnl = emissivity_c * STEFAN_BOLTZMANN * canopy_temp_kelvin**4 - emissivity_a * STEFAN_BOLTZMANN * air_temp_kelvin**4
    return Rns - Rnl

def soil_heat_flux(net_radiation):
    return net_radiation * 0.1

def aerodynamic_resistance(wind_speed, measurement_height, zero_plane_displacement, roughness_length):
    return (np.log((measurement_height - zero_plane_displacement) / roughness_length) * 
            np.log((measurement_height - zero_plane_displacement) / (roughness_length * 0.1))) / (K**2 * wind_speed)

def psychrometric_constant(atmospheric_pressure_pa):
    return (CP * atmospheric_pressure_pa) / (0.622 * 2.45e6)

def slope_saturation_vapor_pressure(temperature_celsius):
    return 4098 * saturated_vapor_pressure(temperature_celsius) / (temperature_celsius + 237.3)**2

def convert_wind_speed(u3, crop_height):
    z0 = 0.1 * crop_height
    return u3 * (np.log(2/z0) / np.log(3/z0))

def calculate_cwsi_th1(row, crop_height, surface_albedo=0.23):
    Ta = row['Ta_2m_Avg']
    RH = row['RH_2m_Avg']
    Rs = row['Solar_2m_Avg']
    u3 = row['WndAveSpd_3m']
    P = row['PresAvg_1pnt5m'] * 100
    Tc = row['canopy_temp']
    
    u2 = convert_wind_speed(u3, crop_height)
    
    VPD = vapor_pressure_deficit(Ta, RH)
    Rn = net_radiation(Rs, Ta, Tc, surface_albedo)
    G = soil_heat_flux(Rn)
    
    zero_plane_displacement = 0.67 * crop_height
    roughness_length = 0.123 * crop_height
    
    ra = aerodynamic_resistance(u2, 2, zero_plane_displacement, roughness_length)
    γ = psychrometric_constant(P)
    Δ = slope_saturation_vapor_pressure(Ta)
    
    ρ = P / (287.05 * celsius_to_kelvin(Ta))
    
    numerator = (Tc - Ta) - ((ra * (Rn - G)) / (ρ * CP)) + (VPD / γ)
    denominator = ((Δ + γ) * ra * (Rn - G)) / (ρ * CP * γ) + (VPD / γ)
    
    if denominator == 0:
        return None
    
    cwsi = numerator / denominator
    return cwsi if 0 <= cwsi <= 2 else None

def process_csv_file(file_path):
    logger.info(f"Processing file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    # Find the IRT column
    irt_column = next((col for col in df.columns if 'irt' in col.lower()), None)
    if not irt_column:
        logger.warning(f"No IRT column found in {file_path}")
        return None
    
    # Filter data between 12 PM and 5 PM
    df['time'] = df['TIMESTAMP'].dt.time
    mask = (df['time'] >= time(12, 0)) & (df['time'] <= time(17, 0))
    
    # Rename IRT column to 'canopy_temp' for CWSI calculation
    df.loc[mask, 'canopy_temp'] = df.loc[mask, irt_column]
    
    # Remove existing CWSI column if it exists
    if 'cwsi' in df.columns:
        df = df.drop(columns=['cwsi'])
    
    # Calculate CWSI only for the filtered rows
    df.loc[mask, 'cwsi'] = df.loc[mask].apply(lambda row: calculate_cwsi_th1(row, CROP_HEIGHT, SURFACE_ALBEDO), axis=1)
    
    # Remove temporary columns
    df = df.drop(columns=['time', 'canopy_temp'])
    
    # Log the range of computed CWSI values
    cwsi_values = df['cwsi'].dropna()
    if not cwsi_values.empty:
        logger.info(f"CWSI value ranges for {file_path}: Min={cwsi_values.min()}, Max={cwsi_values.max()}, Mean={cwsi_values.mean()}")
    else:
        logger.info(f"No CWSI values computed for {file_path}")
    
    # Save the updated DataFrame back to the same CSV file
    df.to_csv(file_path, index=False)
    logger.info(f"Updated CWSI values in file: {file_path}")
    
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
