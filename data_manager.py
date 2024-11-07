import yaml

def setup_plot_metadata(cursor, sensor_mapping_path):
    """Extract unique plot metadata from sensor mapping and populate plots table"""
    with open(sensor_mapping_path, 'r') as f:
        sensor_data = yaml.safe_load(f)
    
    # Extract unique plot information
    plot_metadata = set()
    for sensor in sensor_data:
        if isinstance(sensor, dict):  # Skip any non-dict entries
            plot_metadata.add((
                sensor['plot_number'],
                sensor['treatment'],
                sensor['field']
            ))
    
    # Insert plot metadata
    cursor.executemany(
        "INSERT OR IGNORE INTO plots (plot_number, treatment, field) VALUES (?, ?, ?)",
        list(plot_metadata)
    )

# Call this before inserting yield or irrigation data
setup_plot_metadata(cursor, 'path/to/sensor_mapping.yaml')
