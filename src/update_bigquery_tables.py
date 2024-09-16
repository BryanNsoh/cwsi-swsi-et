# src/update_bigquery_tables.py

import os
import pandas as pd
import re
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_bigquery_client():
    load_dotenv()
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not credentials_path:
        logger.error("Error: GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        logger.error(f"Error: Error creating BigQuery client: {e}")
        raise

def extract_dataset_and_table(file_name):
    match = re.match(r'(LINEAR_\w+_trt\d+)_plot_(\d+)_\d+\.csv', file_name)
    if match:
        dataset_name, table_name = match.groups()
        return dataset_name, f"plot_{table_name}"
    else:
        raise ValueError(f"Unable to extract dataset and table name from filename: {file_name}")

def upload_to_bigquery(df, client, dataset_id, table_id):
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition="WRITE_TRUNCATE",
    )

    try:
        job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"Loaded {len(df)} rows into {full_table_id}")
    except Exception as e:
        logger.error(f"Failed to upload data to {full_table_id}: {str(e)}")

def process_and_upload_file(file_path, client):
    file_name = os.path.basename(file_path)
    dataset_id, table_id = extract_dataset_and_table(file_name)
    
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    
    upload_to_bigquery(df, client, dataset_id, table_id)

def main(input_folder):
    client = create_bigquery_client()
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                logger.info(f"Processing file: {file_path}")
                process_and_upload_file(file_path, client)

if __name__ == "__main__":
    input_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\data"
    main(input_folder)