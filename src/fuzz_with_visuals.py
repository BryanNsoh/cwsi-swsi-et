# src/fuzz_with_visuals.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from datetime import datetime, timedelta
import logging
from matplotlib import gridspec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FuzzyIrrigationController:
    def __init__(self, days_back=3):
        self.days_back = days_back
        self.setup_fuzzy_system()
        self.image_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(self.image_dir, exist_ok=True)

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
    
        rules = [
            # Strong rule to disregard CWSI when soil is very wet
            ctrl.Rule(self.swsi['very_wet'], self.irrigation['none']),
            
            # Existing CWSI rules, now only apply when soil is not very wet
            ctrl.Rule(self.cwsi['severe_stress'] & ~self.swsi['very_wet'], self.irrigation['very_high']),
            ctrl.Rule(self.cwsi['high_stress'] & ~self.swsi['very_wet'], self.irrigation['high']),
            ctrl.Rule(self.cwsi['moderate_stress'] & ~self.swsi['very_wet'], self.irrigation['medium']),
            ctrl.Rule(self.cwsi['low_stress'] & ~self.swsi['very_wet'], self.irrigation['low']),
            ctrl.Rule(self.cwsi['no_stress'] & ~self.swsi['very_wet'], self.irrigation['very_low']),
            
            # Existing SWSI rules (unchanged)
            ctrl.Rule(self.swsi['very_dry'], self.irrigation['very_high']),
            ctrl.Rule(self.swsi['dry'], self.irrigation['high']),
            ctrl.Rule(self.swsi['normal'], self.irrigation['medium']),
            ctrl.Rule(self.swsi['wet'], self.irrigation['low']),
            
            # Existing combined CWSI and SWSI rules, now only apply when soil is not very wet
            ctrl.Rule(self.cwsi['severe_stress'] & self.swsi['very_dry'] & ~self.swsi['very_wet'], self.irrigation['very_high']),
            ctrl.Rule(self.cwsi['high_stress'] & self.swsi['dry'] & ~self.swsi['very_wet'], self.irrigation['high']),
            ctrl.Rule(self.cwsi['moderate_stress'] & self.swsi['normal'] & ~self.swsi['very_wet'], self.irrigation['medium']),
            ctrl.Rule(self.cwsi['low_stress'] & self.swsi['wet'] & ~self.swsi['very_wet'], self.irrigation['low']),
            
            # Existing ET rules, modified to not apply CWSI when soil is very wet
            ctrl.Rule(self.etc['very_high'] & self.cwsi['high_stress'] & ~self.swsi['very_wet'], self.irrigation['high']),
            ctrl.Rule(self.etc['high'] & self.swsi['dry'], self.irrigation['medium']),
            ctrl.Rule(self.etc['medium'] & ((self.cwsi['moderate_stress'] & ~self.swsi['very_wet']) | self.swsi['normal']), self.irrigation['medium']),
            ctrl.Rule(self.etc['low'] & ((self.cwsi['low_stress'] & ~self.swsi['very_wet']) | self.swsi['wet']), self.irrigation['low']),
            ctrl.Rule(self.etc['very_low'] & (self.cwsi['no_stress'] | self.swsi['very_wet']), self.irrigation['none']),
        ]
    
        # Create and simulate control system
        self.irrigation_ctrl = ctrl.ControlSystem(rules)
        self.irrigation_sim = ctrl.ControlSystemSimulation(self.irrigation_ctrl)

    def plot_membership_functions(self):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(10, 20))
        
        self.etc.view(ax=ax0)
        ax0.set_title('Evapotranspiration')
        self.swsi.view(ax=ax1)
        ax1.set_title('Surface Water Supply Index')
        self.cwsi.view(ax=ax2)
        ax2.set_title('Crop Water Stress Index')
        self.irrigation.view(ax=ax3)
        ax3.set_title('Irrigation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, 'membership_functions.png'))
        plt.close()

    def plot_recent_data(self, et_data, swsi_data, cwsi_data, plot):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 15))
        
        et_data.plot(ax=ax1)
        ax1.set_title('Recent Evapotranspiration')
        ax1.set_ylabel('ET')
        
        swsi_data.plot(ax=ax2)
        ax2.set_title('Recent Surface Water Supply Index')
        ax2.set_ylabel('SWSI')
        
        cwsi_data.plot(ax=ax3)
        ax3.set_title('Recent Crop Water Stress Index')
        ax3.set_ylabel('CWSI')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, f'recent_data_plot_{plot}.png'))
        plt.close()

    def plot_fuzzy_output(self, et_avg, swsi_avg, cwsi_max, irrigation_amount, plot):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        self.irrigation.view(sim=self.irrigation_sim, ax=ax)
        ax.plot([irrigation_amount, irrigation_amount], [0, 1], 'r--', linewidth=1.5, label='Output')
        
        ax.set_title(f'Fuzzy Irrigation Output (Plot {plot})')
        ax.set_ylabel('Membership')
        ax.set_xlabel('Irrigation Amount')
        ax.legend()
        
        plt.text(0.05, 0.95, f'Inputs:\nET: {et_avg:.2f}\nSWSI: {swsi_avg:.2f}\nCWSI: {cwsi_max:.2f}', 
                 transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, f'fuzzy_output_plot_{plot}.png'))
        plt.close()

    def get_recent_values(self, series):
        end_date = pd.Timestamp.now().floor('D')
        start_date = end_date - pd.Timedelta(days=self.days_back)
        
        # Ensure the series index is timezone-naive
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        
        recent_data = series.loc[start_date:end_date]
        return recent_data, start_date, end_date

    def get_recent_swsi(self, series):
        # Get the most recent non-NaN SWSI value
        return series.dropna().iloc[-1] if not series.dropna().empty else None

    def get_recent_cwsi(self, series):
        end_date = pd.Timestamp.now().floor('D')
        start_date = end_date - pd.Timedelta(days=self.days_back)
        
        # Ensure the series index is timezone-naive
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        
        # Get data for the last self.days_back days
        recent_data = series.loc[start_date:end_date]
        
        # Group by date and get the maximum CWSI for each day
        daily_max = recent_data.groupby(recent_data.index.date).max()
        
        # Filter valid CWSI values (between 0 and 1.5)
        valid_data = daily_max[(daily_max >= 0) & (daily_max <= 1.5)]
        
        if len(valid_data) > 0:
            # Sort values in descending order and take the average of up to 3 highest values
            top_3_avg = valid_data.sort_values(ascending=False).head(3).mean()
            # Cap the average at 1  
            return min(top_3_avg, 1)
        else:
            return None

    def compute_irrigation(self, df, plot):
        recommendations = []
        
        for _, row in df.iterrows():
            et_value = row['etc']
            swsi_value = row['swsi']
            cwsi_value = row['cwsi']
            
            # Set inputs for fuzzy system
            self.irrigation_sim.input['etc'] = et_value if not pd.isna(et_value) else 0
            self.irrigation_sim.input['swsi'] = swsi_value if not pd.isna(swsi_value) else 0.5
            self.irrigation_sim.input['cwsi'] = min(cwsi_value, 1) if not pd.isna(cwsi_value) else 0.5
            
            # Compute output
            self.irrigation_sim.compute()
            
            irrigation_amount = self.irrigation_sim.output['irrigation']
            recommendations.append(irrigation_amount)
        
        return recommendations

    def process_csv_file(self, file_path):
        logger.info(f"Processing file for irrigation recommendation: {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
        df.set_index('TIMESTAMP', inplace=True)
        
        plot_number = file_path.split('_')[-2]  # Assuming the plot number is the second-to-last part of the filename
        crop_type = 'corn' if 'CORN' in file_path else 'soybean'

        recommendations = self.compute_irrigation(df, plot_number)
        
        # Add recommendations to the main data CSV
        df['recommendation'] = recommendations
        df.to_csv(file_path)  # Overwrite the original CSV with the new column
        
        # For the summary, we'll use today's values or the last available if today's not present
        today = pd.Timestamp.now().floor('D')
        if today in df.index:
            summary_row = df.loc[today]
        else:
            summary_row = df.iloc[-1]
        
        return {
            'plot': plot_number,
            'crop': crop_type,
            'irrigation': summary_row['recommendation'],
            'et_avg': summary_row['etc'],
            'swsi': summary_row['swsi'],
            'cwsi': summary_row['cwsi'],
            'date': summary_row.name.strftime('%Y-%m-%d')
        }

def main(input_folder, output_file, days_back=4):
    controller = FuzzyIrrigationController(days_back)
    
    # Plot membership functions
    controller.plot_membership_functions()
    
    recommendations = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv') and ('CORN' in file or 'SOYBEAN' in file) and 'trt1' in file:
                file_path = os.path.join(root, file)
                result = controller.process_csv_file(file_path)
                recommendations.append(result)

    # Create recommendations DataFrame and save to CSV
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(output_file, index=False)
    logger.info(f"Recommendations saved to {output_file}")

    # Visualize the fuzzy control system and rule base
    controller.irrigation_ctrl.view()
    plt.savefig(os.path.join(controller.image_dir, 'fuzzy_control_system.png'))
    plt.close()

if __name__ == "__main__":
    input_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    output_file = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\recommendations\fuzzy-trt-1.csv"
    main(input_folder, output_file, days_back=6)