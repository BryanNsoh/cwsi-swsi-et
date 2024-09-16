# src/run_analysis.py

import os
from datetime import datetime
import logging
from dat_to_csv import main as dat_to_csv_main
from cwsi_th1 import main as cwsi_corn_main
from cwsi_th2_soybean import main as cwsi_soybean_main
from swsi import main as swsi_main
from et import main as et_main
from fuzz_with_visuals import main as fuzz_main
from other_scheduling import main as other_scheduling_main
from update_bigquery_tables import main as update_bigquery_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dated_folder(base_path):
    current_date = datetime.now().strftime("%Y-%m-%d")
    dated_folder = os.path.join(base_path, f"analysis-{current_date}")
    os.makedirs(dated_folder, exist_ok=True)
    return dated_folder

def get_all_subfolders(base_folder):
    return [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

def run_pipeline():
    # Define base paths
    base_output_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et"
    
    # Create dated folders
    dated_output_folder = create_dated_folder(base_output_folder)
    dated_data_folder = os.path.join(dated_output_folder, "data")
    dated_recommendations_folder = os.path.join(dated_output_folder, "recommendations")
    
    os.makedirs(dated_data_folder, exist_ok=True)
    os.makedirs(dated_recommendations_folder, exist_ok=True)
    
    # Define base folders for corn and soybean data
    corn_base_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr"
    soybean_base_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr"
    
    # Get all subfolders
    corn_folders = get_all_subfolders(corn_base_folder)
    soybean_folders = get_all_subfolders(soybean_base_folder)
    
    sensor_mapping_path = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\sensor_mapping.yaml"
    weather_csv_path = r"C:\Users\bnsoh2\Downloads\North_Platte_3SW_Beta_1minx.csv"
    
    logger.info("Running dat_to_csv...")
    dat_to_csv_main(corn_folders, soybean_folders, sensor_mapping_path, dated_data_folder, weather_csv_path)
    
    # Run CWSI calculation for corn
    logger.info("Calculating CWSI for corn...")
    cwsi_corn_main(dated_data_folder)
    
    # Run CWSI calculation for soybean
    logger.info("Calculating CWSI for soybean...")
    cwsi_soybean_main(dated_data_folder)
    
    # Run SWSI calculation
    logger.info("Calculating SWSI...")
    swsi_main(dated_data_folder)
    
    # Run ET calculation
    logger.info("Calculating ET...")
    et_main(dated_data_folder)
    
    # Run fuzzy logic irrigation scheduling
    logger.info("Running fuzzy logic irrigation scheduling...")
    fuzz_output_file = os.path.join(dated_recommendations_folder, "fuzzy-trt-1.csv")
    fuzz_main(dated_data_folder, fuzz_output_file)
    
    # Run other scheduling methods
    logger.info("Running other scheduling methods...")
    other_scheduling_main(dated_data_folder, dated_recommendations_folder)
    
    # Update BigQuery tables
    logger.info("Updating BigQuery tables...")
    update_bigquery_main(dated_data_folder)
    
    logger.info("Analysis pipeline complete. Results saved in: " + dated_output_folder)

if __name__ == "__main__":
    run_pipeline()