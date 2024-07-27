# src/fuzz.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FuzzyIrrigationController:
    def __init__(self):
        self.setup_fuzzy_system()

    def setup_fuzzy_system(self):
        # Define input ranges
        x_et = np.arange(0, 20, 0.1)
        x_swsi = np.arange(0, 1, 0.01)
        x_cwsi = np.arange(0, 1, 0.01)
        x_irrigation = np.arange(0, 1, 0.01)
    
        # Create fuzzy variables
        self.etc = ctrl.Antecedent(x_et, 'etc')
        self.swsi = ctrl.Antecedent(x_swsi, 'swsi')
        self.cwsi = ctrl.Antecedent(x_cwsi, 'cwsi')
        self.irrigation = ctrl.Consequent(x_irrigation, 'irrigation')
    
        # Define membership functions for ET
        self.etc['very_low'] = fuzz.trimf(x_et, [0, 0, 2])
        self.etc['low'] = fuzz.trimf(x_et, [1, 2.5, 4])
        self.etc['medium'] = fuzz.trimf(x_et, [3, 4.5, 6])
        self.etc['high'] = fuzz.trimf(x_et, [5, 6.5, 8])
        self.etc['very_high'] = fuzz.trimf(x_et, [7, 10, 10])
    
        self.swsi['very_wet'] = fuzz.trimf(x_swsi, [0, 0, 0.25])
        self.swsi['wet'] = fuzz.trimf(x_swsi, [0, 0.25, 0.5])
        self.swsi['normal'] = fuzz.trimf(x_swsi, [0.25, 0.5, 0.75])
        self.swsi['dry'] = fuzz.trimf(x_swsi, [0.5, 0.75, 1])
        self.swsi['very_dry'] = fuzz.trimf(x_swsi, [0.75, 1, 1])
    
        self.cwsi['no_stress'] = fuzz.trimf(x_cwsi, [0, 0, 0.25])
        self.cwsi['low_stress'] = fuzz.trimf(x_cwsi, [0, 0.25, 0.5])
        self.cwsi['moderate_stress'] = fuzz.trimf(x_cwsi, [0.25, 0.5, 0.75])
        self.cwsi['high_stress'] = fuzz.trimf(x_cwsi, [0.5, 0.75, 1])
        self.cwsi['severe_stress'] = fuzz.trimf(x_cwsi, [0.75, 1, 1])
    
        self.irrigation['none'] = fuzz.trimf(x_irrigation, [0, 0, 0.2])
        self.irrigation['very_low'] = fuzz.trimf(x_irrigation, [0, 0.2, 0.4])
        self.irrigation['low'] = fuzz.trimf(x_irrigation, [0.2, 0.4, 0.6])
        self.irrigation['medium'] = fuzz.trimf(x_irrigation, [0.4, 0.6, 0.8])
        self.irrigation['high'] = fuzz.trimf(x_irrigation, [0.6, 0.8, 1])
        self.irrigation['very_high'] = fuzz.trimf(x_irrigation, [0.8, 1, 1])
    
        # Define fuzzy rules
        rules = [
            ctrl.Rule(self.cwsi['severe_stress'], self.irrigation['very_high']),
            ctrl.Rule(self.cwsi['high_stress'], self.irrigation['high']),
            ctrl.Rule(self.cwsi['moderate_stress'], self.irrigation['medium']),
            ctrl.Rule(self.cwsi['low_stress'], self.irrigation['low']),
            ctrl.Rule(self.cwsi['no_stress'], self.irrigation['very_low']),
            
            ctrl.Rule(self.swsi['very_dry'], self.irrigation['very_high']),
            ctrl.Rule(self.swsi['dry'], self.irrigation['high']),
            ctrl.Rule(self.swsi['normal'], self.irrigation['medium']),
            ctrl.Rule(self.swsi['wet'], self.irrigation['low']),
            ctrl.Rule(self.swsi['very_wet'], self.irrigation['none']),
            
            ctrl.Rule(self.cwsi['severe_stress'] & self.swsi['very_dry'], self.irrigation['very_high']),
            ctrl.Rule(self.cwsi['high_stress'] & self.swsi['dry'], self.irrigation['high']),
            ctrl.Rule(self.cwsi['moderate_stress'] & self.swsi['normal'], self.irrigation['medium']),
            ctrl.Rule(self.cwsi['low_stress'] & self.swsi['wet'], self.irrigation['low']),
            ctrl.Rule(self.cwsi['no_stress'] & self.swsi['very_wet'], self.irrigation['none']),
            
            ctrl.Rule(self.etc['very_high'] & self.cwsi['high_stress'], self.irrigation['high']),
            ctrl.Rule(self.etc['high'] & self.swsi['dry'], self.irrigation['medium']),
            ctrl.Rule(self.etc['medium'] & (self.cwsi['moderate_stress'] | self.swsi['normal']), self.irrigation['medium']),
            ctrl.Rule(self.etc['low'] & (self.cwsi['low_stress'] | self.swsi['wet']), self.irrigation['low']),
            ctrl.Rule(self.etc['very_low'] & (self.cwsi['no_stress'] | self.swsi['very_wet']), self.irrigation['none']),
        ]
    
        # Create and simulate control system
        self.irrigation_ctrl = ctrl.ControlSystem(rules)
        self.irrigation_sim = ctrl.ControlSystemSimulation(self.irrigation_ctrl)

    def get_recent_swsi(self, series):
        # Get the most recent non-NaN SWSI value
        return series.dropna().iloc[-1] if not series.dropna().empty else None

    def get_recent_cwsi(self, series, n_days=3):
        end_date = pd.Timestamp.now().floor('D')
        start_date = end_date - pd.Timedelta(days=n_days)
        
        # Ensure the series index is timezone-naive
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        
        recent_data = series.loc[start_date:end_date]
        valid_data = recent_data[(recent_data >= 0) & (recent_data <= 1.5)]
        
        return valid_data.iloc[-1] if not valid_data.empty else None

    def compute_irrigation(self, df, plot):
        # Compute inputs for fuzzy system
        et_data, et_start, et_end = self.get_recent_values(df['etc'], 6)
        et_avg = et_data.mean()

        swsi_value = self.get_recent_swsi(df['swsi'])
        cwsi_value = self.get_recent_cwsi(df['cwsi'])

        # Log the values used for decision making
        logger.info(f"Values used for irrigation decision for plot {plot}:")
        logger.info(f"ET Average: {et_avg:.2f} (from {et_start} to {et_end})")
        logger.info(f"Most recent SWSI: {swsi_value:.2f}" if swsi_value is not None else "No valid SWSI value found")
        logger.info(f"Most recent valid CWSI: {cwsi_value:.2f}" if cwsi_value is not None else "No valid CWSI value found in the last 3 days")

        # Set inputs for fuzzy system
        self.irrigation_sim.input['etc'] = et_avg
        self.irrigation_sim.input['swsi'] = swsi_value if swsi_value is not None else 0.5  # Default value if no valid SWSI
        self.irrigation_sim.input['cwsi'] = min(cwsi_value, 1) if cwsi_value is not None else 0.5  # Default value if no valid CWSI, cap at 1

        # Compute output
        self.irrigation_sim.compute()

        irrigation_amount = self.irrigation_sim.output['irrigation']

        logger.info(f"Recommended Irrigation Amount for plot {plot}: {irrigation_amount:.2f} inches")

        return irrigation_amount, et_avg, swsi_value, cwsi_value

    def get_recent_values(self, series, n_days=3):
        end_date = pd.Timestamp.now().floor('D')
        start_date = end_date - pd.Timedelta(days=n_days)
        
        # Ensure the series index is timezone-naive
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        
        recent_data = series.loc[start_date:end_date]
        return recent_data, start_date, end_date

def process_csv_file(file_path, controller):
    logger.info(f"Processing file for irrigation recommendation: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    df.set_index('TIMESTAMP', inplace=True)
    
    plot_number = file_path.split('_')[-2]  # Assuming the plot number is the second-to-last part of the filename
    crop_type = 'corn' if 'CORN' in file_path else 'soybean'

    irrigation_amount, et_avg, swsi_value, cwsi_value = controller.compute_irrigation(df, plot_number)
    
    return {
        'plot': plot_number,
        'crop': crop_type,
        'irrigation': irrigation_amount,
        'et_avg': et_avg,
        'swsi': swsi_value,
        'cwsi': cwsi_value
    }

def main(input_folder, output_file):
    controller = FuzzyIrrigationController()
    recommendations = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv') and ('CORN' in file or 'SOYBEAN' in file) and 'trt1' in file:
                file_path = os.path.join(root, file)
                result = process_csv_file(file_path, controller)
                recommendations.append(result)

    # Create recommendations DataFrame and save to CSV
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(output_file, index=False)
    logger.info(f"Recommendations saved to {output_file}")

if __name__ == "__main__":
    input_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    output_file = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\recommendations\fuzzy-trt-1.csv"
    main(input_folder, output_file)