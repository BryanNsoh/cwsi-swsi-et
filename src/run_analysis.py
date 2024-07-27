# src/run_analysis.py

import os
from datetime import datetime
import logging
from dat_to_csv import main as dat_to_csv_main
from cwsi_th1 import main as cwsi_main
from swsi import main as swsi_main
from et import main as et_main
from fuzz_with_visuals import main as fuzz_main
from other_scheduling import main as other_scheduling_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dated_folder(base_path):
    current_date = datetime.now().strftime("%Y-%m-%d")
    dated_folder = os.path.join(base_path, f"analysis-{current_date}")
    os.makedirs(dated_folder, exist_ok=True)
    return dated_folder

def run_analysis():
    # Define base paths
    base_output_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et"
    
    # Create dated folders
    dated_output_folder = create_dated_folder(base_output_folder)
    dated_data_folder = os.path.join(dated_output_folder, "data")
    dated_recommendations_folder = os.path.join(dated_output_folder, "recommendations")
    
    os.makedirs(dated_data_folder, exist_ok=True)
    os.makedirs(dated_recommendations_folder, exist_ok=True)
    
    # Run dat_to_csv
    corn_folders = [
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-03-2024",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-08-2024-discontinuous",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-14-2024-discont-nodeC only",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-15-2024-discont-unsure",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-19-2024",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-26-2024"
    ]
    soybean_folders = [
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-15-24",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-19-2024",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-26-2024"
    ]
    sensor_mapping_path = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\sensor_mapping.yaml"
    weather_csv_path = r"C:\Users\bnsoh2\Downloads\North_Platte_3SW_Beta_1min (9).csv"
    
    logger.info("Running dat_to_csv...")
    dat_to_csv_main(corn_folders, soybean_folders, sensor_mapping_path, dated_data_folder, weather_csv_path)
    
    # Run CWSI calculation
    logger.info("Calculating CWSI...")
    cwsi_main(dated_data_folder)
    
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
    
    logger.info("Analysis complete. Results saved in: " + dated_output_folder)

if __name__ == "__main__":
    run_analysis()