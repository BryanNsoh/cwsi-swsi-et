import os
import pandas as pd
import re

def extract_dataset_and_table(file_name):
    match = re.match(r'(LINEAR_\w+_trt\d+)_plot_(\d+)_\d+\.csv', file_name)
    if match:
        dataset_name, table_name = match.groups()
        return dataset_name, f"plot_{table_name}"
    else:
        raise ValueError(f"Unable to extract dataset and table name from filename: {file_name}")

def generate_dbml_schema(data_folder):
    dbml_schema = "// Project: crop2cloud24\n\n"
    datasets = {}

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    dataset_name, table_name = extract_dataset_and_table(file)
                    df = pd.read_csv(file_path)
                    
                    if dataset_name not in datasets:
                        datasets[dataset_name] = {}
                    
                    if table_name not in datasets[dataset_name]:
                        datasets[dataset_name][table_name] = df.dtypes.to_dict()
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")

    for dataset_name, tables in datasets.items():
        dbml_schema += f"// Dataset: {dataset_name}\n"
        for table_name, columns in tables.items():
            dbml_schema += f"Table {dataset_name}.{table_name} {{\n"
            for column_name, dtype in columns.items():
                dbml_type = "varchar"
                if pd.api.types.is_integer_dtype(dtype):
                    dbml_type = "integer"
                elif pd.api.types.is_float_dtype(dtype):
                    dbml_type = "float"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    dbml_type = "timestamp"
                dbml_schema += f"  {column_name} {dbml_type}\n"
            dbml_schema += "}\n\n"

    return dbml_schema

# Usage
data_folder = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\analysis-2024-09-15\data\data-2024-09-15"
dbml_output = generate_dbml_schema(data_folder)
print(dbml_output)

# Optionally, save to a file
with open("bigquery_schema.dbml", "w") as f:
    f.write(dbml_output)