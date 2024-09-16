import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import yaml
from google.api_core import exceptions  # Correct import for NotFound exception

def load_sensor_mapping(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Sensor mapping file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Error parsing YAML file: {e}")
        raise

def create_bigquery_client():
    load_dotenv()
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not credentials_path:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        print(f"Error: Error creating BigQuery client: {e}")
        raise

def ensure_dataset_exists(client, dataset_id, delete_existing=False):
    dataset_ref = client.dataset(dataset_id)
    try:
        if delete_existing:
            client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
            print(f"Deleted existing dataset {dataset_id}")
        dataset = client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists")
    except exceptions.NotFound:
        # If dataset is not found, create it
        dataset = bigquery.Dataset(dataset_ref)
        dataset = client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}")

def create_or_update_schemaless_table(client, table_id, delete_existing=False):
    try:
        if delete_existing:
            client.delete_table(table_id, not_found_ok=True)
            print(f"Deleted existing table {table_id}")
        
        try:
            table = client.get_table(table_id)
            print(f"Table {table_id} already exists. No schema changes needed.")
        except exceptions.NotFound:
            # If table is not found, create a new table without schema
            table = bigquery.Table(table_id)
            table = client.create_table(table)
            print(f"Created schemaless table {table_id}")
    
    except Exception as e:
        print(f"Error: Error creating or updating table {table_id}: {e}")
        raise

def process_sensor_mapping(client, sensor_mapping, delete_existing=False):
    field_treatment_plot_sensor_map = {}
    for sensor in sensor_mapping:
        field = sensor['field']
        treatment = sensor['treatment']
        plot_number = sensor['plot_number']
        sensor_id = sensor['sensor_id']
        
        if field not in field_treatment_plot_sensor_map:
            field_treatment_plot_sensor_map[field] = {}
        if treatment not in field_treatment_plot_sensor_map[field]:
            field_treatment_plot_sensor_map[field][treatment] = {}
        if plot_number not in field_treatment_plot_sensor_map[field][treatment]:
            field_treatment_plot_sensor_map[field][treatment][plot_number] = set()
        
        field_treatment_plot_sensor_map[field][treatment][plot_number].add(sensor_id)
    
    for field, treatments in field_treatment_plot_sensor_map.items():
        for treatment, plots in treatments.items():
            dataset_id = f"{field}_trt{treatment}"
            ensure_dataset_exists(client, dataset_id, delete_existing)
            
            for plot_number, sensor_ids in plots.items():
                table_id = f"{client.project}.{dataset_id}.plot_{plot_number}"
                create_or_update_schemaless_table(client, table_id, delete_existing)

def main():
    try:
        yaml_path = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\sensor_mapping.yaml"
        sensor_mapping = load_sensor_mapping(yaml_path)
        client = create_bigquery_client()

        delete_flag = input("Do you want to delete existing tables/datasets before creating new ones? (y/n): ").lower() == 'y'

        if delete_flag:
            confirmation1 = input("Are you sure you want to delete existing tables/datasets? (y/n): ").lower()
            if confirmation1 == 'y':
                confirmation2 = input("This action is irreversible. Type 'y' again to confirm deletion: ").lower()
                if confirmation2 == 'y':
                    print("Deletion confirmed. Proceeding with table creation (including deletion of existing ones).")
                    process_sensor_mapping(client, sensor_mapping, delete_existing=True)
                else:
                    print("Deletion cancelled. Proceeding with table creation without deleting existing ones.")
                    process_sensor_mapping(client, sensor_mapping, delete_existing=False)
            else:
                print("Deletion cancelled. Proceeding with table creation without deleting existing ones.")
                process_sensor_mapping(client, sensor_mapping, delete_existing=False)
        else:
            print("Proceeding with table creation without deleting existing ones.")
            process_sensor_mapping(client, sensor_mapping, delete_existing=False)

        print("Process completed successfully")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
