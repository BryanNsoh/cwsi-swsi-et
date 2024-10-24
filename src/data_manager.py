# src/data_manager.py

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Treatment name mappings
TREATMENT_NAMES = {
    1: "IoT-Fuzzy",
    2: "CWSI + SWSI",
    3: "CWSI only",
    4: "SWSI",
    5: "ET-Model",
    6: "Grower's Practice"
}

# Yield data for Soybean (average per treatment)
SOYBEAN_YIELDS = {
    1: {"trt_name": "IoT-Fuzzy", "avg_yield_bu_ac": 66.0, "yield_kg_ha": 4441.0},
    2: {"trt_name": "CWSI + SWSI", "avg_yield_bu_ac": 71.8, "yield_kg_ha": 4831.7},
    3: {"trt_name": "CWSI only", "avg_yield_bu_ac": 63.8, "yield_kg_ha": 4290.6},
    4: {"trt_name": "SWSI", "avg_yield_bu_ac": 63.8, "yield_kg_ha": 4290.6},
    5: {"trt_name": "ET-Model", "avg_yield_bu_ac": 85.0, "yield_kg_ha": 5713.2},
    6: {"trt_name": "Grower's Practice", "avg_yield_bu_ac": 56.9, "yield_kg_ha": 3823.9}
}

# Yield data for Corn (average per treatment)
CORN_YIELDS = {
    1: {"trt_name": "IoT-Fuzzy", "avg_yield_bu_ac": 263.2, "yield_kg_ha": 17701.9, "irrigation_applied_inches": 3.9, "irrigation_applied_mm": 99},
    2: {"trt_name": "CWSI + SWSI", "avg_yield_bu_ac": 279.9, "yield_kg_ha": 18826.6, "irrigation_applied_inches": 3.7, "irrigation_applied_mm": 94},
    3: {"trt_name": "CWSI only", "avg_yield_bu_ac": 269.3, "yield_kg_ha": 18110.7, "irrigation_applied_inches": 6.7, "irrigation_applied_mm": 170},
    4: {"trt_name": "SWSI", "avg_yield_bu_ac": 246.3, "yield_kg_ha": 16560.7, "irrigation_applied_inches": 3.7, "irrigation_applied_mm": 94},
    5: {"trt_name": "ET-Model", "avg_yield_bu_ac": 273.3, "yield_kg_ha": 18380.1, "irrigation_applied_inches": 8.5, "irrigation_applied_mm": 216},
    6: {"trt_name": "Grower's Practice", "avg_yield_bu_ac": 272.0, "yield_kg_ha": 18289.2, "irrigation_applied_inches": 9.0, "irrigation_applied_mm": 229}
}

# Irrigation events for Span 2 and Span 5
IRRIGATION_EVENTS = {
    "Span2": {
        1: {
            "averaged": [0, 0.45, 0.54, 0.46, 0.50, 0, 0.53],
            "plot_specific": {
                "2006": [0.84, 0.61],
                "2015": [0.70, 0.41],
                "2023": [0.59, 0.37]
            }
        },
        2: {
            "averaged": [0.2, 1, 1, 0, 0, 0.5, 0.75],
            "plot_specific": {
                "all": [1, 1]
            }
        },
        3: {
            "averaged": [0.2, 1, 1, 1, 1, 0, 0.75],
            "plot_specific": {
                "all": [0, 0]
            }
        },
        4: {
            "averaged": [0.2, 1, 1, 0, 0, 0.5, 0.75],
            "plot_specific": {
                "all": [1, 1]
            }
        },
        5: {
            "averaged": [0.5, 1, 1, 1, 0, 1, 0.83],
            "plot_specific": {
                "all": [1, 1]
            }
        },
        6: {
            "averaged": [1, 1, 1, 1, 1, 1, 1],
            "plot_specific": {
                "all": [1, 1]
            }
        }
    },
    "Span5": {
        1: {
            "averaged": [0, 0.45, 0.54, 0.6, 0.5, 0, 0.62],
            "plot_specific": {
                "5006": [0.70, 0.61],
                "5010": [0.42, 0.62],
                "5023": [0.55, 0.61]
            }
        },
        2: {
            "averaged": [0.2, 1, 1, 0, 0, 0.5, 0.75],
            "plot_specific": {
                "5003": [0, 0],
                "5012": [0, 0],
                "5026": [0, 0]
            }
        },
        3: {
            "averaged": [0.2, 1, 1, 1, 0, 0.5, 0.75],
            "plot_specific": {
                "5001": [1, 1],
                "5018": [1, 1],
                "5020": [1, 1]
            }
        },
        4: {
            "averaged": [0.2, 1, 1, 0, 0, 0.5, 0.75],
            "plot_specific": {
                "5007": [0, 0],
                "5009": [0, 0],
                "5027": [0, 0]
            }
        },
        5: {
            "averaged": [0.5, 1, 1, 1, 1, 1, 0.92],
            "plot_specific": {
                "all": [1, 1]  # Same for all plots
            }
        },
        6: {
            "averaged": [1, 1, 1, 1, 1, 1, 1],
            "plot_specific": {
                "all": [1, 1]  # Same for all plots
            }
        }
    }
}

# Plot to treatment mappings for both crops
PLOT_MAPPINGS = {
    # Corn plots (LINEAR_CORN)
    "5001": {"treatment": 3, "field": "LINEAR_CORN", "node": "C"},
    "5003": {"treatment": 2, "field": "LINEAR_CORN", "node": "C"},
    "5006": {"treatment": 1, "field": "LINEAR_CORN", "node": "B"},
    "5007": {"treatment": 4, "field": "LINEAR_CORN", "node": "B"},
    "5009": {"treatment": 4, "field": "LINEAR_CORN", "node": "C"},
    "5010": {"treatment": 1, "field": "LINEAR_CORN", "node": "C"},
    "5012": {"treatment": 2, "field": "LINEAR_CORN", "node": "B"},
    "5018": {"treatment": 3, "field": "LINEAR_CORN", "node": "A"},
    "5020": {"treatment": 3, "field": "LINEAR_CORN", "node": "A"},
    "5023": {"treatment": 1, "field": "LINEAR_CORN", "node": "A"},
    "5026": {"treatment": 2, "field": "LINEAR_CORN", "node": "A"},
    "5027": {"treatment": 4, "field": "LINEAR_CORN", "node": "A"},
    
    # Soybean plots (LINEAR_SOYBEAN)
    "2001": {"treatment": 2, "field": "LINEAR_SOYBEAN", "node": "A"},
    "2003": {"treatment": 3, "field": "LINEAR_SOYBEAN", "node": "A"},
    "2006": {"treatment": 1, "field": "LINEAR_SOYBEAN", "node": "B"},
    "2009": {"treatment": 4, "field": "LINEAR_SOYBEAN", "node": "A"},
    "2011": {"treatment": 2, "field": "LINEAR_SOYBEAN", "node": "B"},
    "2012": {"treatment": 4, "field": "LINEAR_SOYBEAN", "node": "B"},
    "2015": {"treatment": 1, "field": "LINEAR_SOYBEAN", "node": "C"},
    "2018": {"treatment": 3, "field": "LINEAR_SOYBEAN", "node": "C"},
    "2020": {"treatment": 3, "field": "LINEAR_SOYBEAN", "node": "C"},
    "2023": {"treatment": 1, "field": "LINEAR_SOYBEAN", "node": "C"},
    "2026": {"treatment": 2, "field": "LINEAR_SOYBEAN", "node": "C"}
}

class ExperimentDataStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('PRAGMA foreign_keys = ON;')  # Enable foreign key support
        self._initialize_database()
        logger.info(f"Connected to SQLite database at {self.db_path}")

    def _initialize_database(self):
        """Initialize the SQLite database with necessary tables."""
        cursor = self.conn.cursor()

        # Create plots table first
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plots (
                plot_id TEXT PRIMARY KEY,
                treatment INTEGER NOT NULL,
                field TEXT NOT NULL,
                node TEXT NOT NULL
            )
        """)

        # Create data table with foreign key to plots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                variable_name TEXT NOT NULL,
                value REAL,
                FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
            )
        """)

        # Create yields table with foreign key to plots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS yields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id TEXT NOT NULL,
                trt_name TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                avg_yield_bu_ac REAL,
                yield_kg_ha REAL,
                irrigation_applied_inches REAL,
                irrigation_applied_mm REAL,
                FOREIGN KEY (plot_id) REFERENCES plots(plot_id),
                UNIQUE(plot_id)
            )
        """)

        # Create irrigation_events table with foreign key to plots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS irrigation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id TEXT NOT NULL,
                treatment INTEGER NOT NULL,
                trt_name TEXT NOT NULL,
                date TEXT NOT NULL,
                amount_inches REAL,
                amount_mm REAL,
                notes TEXT,
                FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
            )
        """)

        # Insert plot mappings
        for plot_id, info in PLOT_MAPPINGS.items():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO plots (plot_id, treatment, field, node)
                    VALUES (?, ?, ?, ?)
                """, (plot_id, info['treatment'], info['field'], info['node']))
                logger.debug(f"Inserted/Existing plot: {plot_id}")
            except Exception as e:
                logger.error(f"Failed to insert plot {plot_id}. Error: {e}")

        self.conn.commit()
        logger.info("Initialized database and inserted plot mappings.")

    def insert_yields(self):
        """Insert yield data into the yields table."""
        cursor = self.conn.cursor()

        # For each plot, insert its corresponding yield data
        for plot_id, info in PLOT_MAPPINGS.items():
            field = info['field']
            treatment = info['treatment']
            crop_type = 'corn' if 'CORN' in field else 'soybean'

            if crop_type == 'corn':
                # Get yield data based on treatment
                yield_data = CORN_YIELDS.get(treatment)
                if yield_data:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO yields 
                            (plot_id, trt_name, crop_type, avg_yield_bu_ac, yield_kg_ha, irrigation_applied_inches, irrigation_applied_mm)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            plot_id,
                            yield_data['trt_name'],
                            crop_type,
                            yield_data['avg_yield_bu_ac'],
                            yield_data['yield_kg_ha'],
                            yield_data.get('irrigation_applied_inches'),
                            yield_data.get('irrigation_applied_mm')
                        ))
                        logger.debug(f"Inserted/Existing yield for corn plot {plot_id}")
                    except Exception as e:
                        logger.error(f"Failed to insert yield for corn plot {plot_id}. Error: {e}")
            else:
                # For soybean, compute total irrigation_applied from irrigation_events
                try:
                    cursor.execute("""
                        SELECT SUM(amount_inches), SUM(amount_mm)
                        FROM irrigation_events
                        WHERE plot_id = ?
                    """, (plot_id,))
                    irrigation_totals = cursor.fetchone()
                    total_inches = irrigation_totals[0] if irrigation_totals[0] is not None else 0
                    total_mm = irrigation_totals[1] if irrigation_totals[1] is not None else 0

                    # Get yield data based on treatment
                    yield_data = SOYBEAN_YIELDS.get(treatment)
                    if yield_data:
                        cursor.execute("""
                            INSERT OR IGNORE INTO yields 
                            (plot_id, trt_name, crop_type, avg_yield_bu_ac, yield_kg_ha, irrigation_applied_inches, irrigation_applied_mm)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            plot_id,
                            yield_data['trt_name'],
                            crop_type,
                            yield_data['avg_yield_bu_ac'],
                            yield_data['yield_kg_ha'],
                            total_inches,
                            total_mm
                        ))
                        logger.debug(f"Inserted/Existing yield for soybean plot {plot_id} with computed irrigation")
                except Exception as e:
                    logger.error(f"Failed to insert yield for soybean plot {plot_id}. Error: {e}")

        self.conn.commit()
        logger.info("Inserted all yield data.")

    def insert_irrigation_events(self):
        """Insert irrigation events into the irrigation_events table."""
        cursor = self.conn.cursor()

        for span, treatments in IRRIGATION_EVENTS.items():
            for trt_num, trt_data in treatments.items():
                trt_name = TREATMENT_NAMES.get(trt_num, f"Treatment {trt_num}")

                # Insert averaged irrigation events (dates before specific dates)
                averaged_amounts = trt_data.get("averaged", [])
                # Base date: assuming first date is "2024-07-05"
                base_date = datetime.strptime("2024-07-05", "%Y-%m-%d")
                for idx, amount in enumerate(averaged_amounts):
                    date = base_date + timedelta(days=idx * 5)  # Example increment, adjust as needed
                    date_str = date.strftime("%Y-%m-%d")
                    
                    # Find all plots with the current treatment
                    cursor.execute("""
                        SELECT plot_id FROM plots WHERE treatment = ?
                    """, (trt_num,))
                    plots = cursor.fetchall()
                    
                    for plot in plots:
                        plot_id = plot[0]
                        try:
                            cursor.execute("""
                                INSERT INTO irrigation_events (plot_id, treatment, trt_name, date, amount_inches, amount_mm, notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                plot_id,
                                trt_num,
                                trt_name,
                                date_str,
                                amount,
                                amount * 25.4,
                                "Averaged irrigation across treatment plots"
                            ))
                            logger.debug(f"Inserted averaged irrigation event for Plot {plot_id}, Treatment {trt_num} on {date_str}")
                        except Exception as e:
                            logger.error(f"Failed to insert averaged irrigation event for Plot {plot_id}, Treatment {trt_num} on {date_str}. Error: {e}")

                # Insert plot-specific irrigation events (dates after specific dates)
                plot_specific = trt_data.get("plot_specific", {})
                specific_dates = ["2024-08-19", "2024-08-30"]  # Example dates; adjust as needed
                for plot_num, amounts in plot_specific.items():
                    for idx, amount in enumerate(amounts):
                        if plot_num == "all":
                            # Averaged events already handled
                            continue
                        if idx < len(specific_dates):
                            date_str = specific_dates[idx]
                        else:
                            # If more amounts than specific_dates, increment days from the last specific_date
                            last_date = datetime.strptime(specific_dates[-1], "%Y-%m-%d")
                            date = last_date + timedelta(days=11 * (idx - len(specific_dates) + 1))  # Example increment
                            date_str = date.strftime("%Y-%m-%d")
                        try:
                            cursor.execute("""
                                INSERT INTO irrigation_events (plot_id, treatment, trt_name, date, amount_inches, amount_mm, notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                plot_num,
                                trt_num,
                                trt_name,
                                date_str,
                                amount,
                                amount * 25.4,
                                "Plot-specific irrigation amount"
                            ))
                            logger.debug(f"Inserted plot-specific irrigation event for Plot {plot_num}, Treatment {trt_num} on {date_str}")
                        except Exception as e:
                            logger.error(f"Failed to insert plot-specific irrigation event for Plot {plot_num}, Treatment {trt_num} on {date_str}. Error: {e}")

        self.conn.commit()
        logger.info("Inserted all irrigation events.")

    def process_analysis_folder(self, data_folder: str):
        """Process a data folder containing CSVs."""
        data_path = Path(data_folder)

        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")

        # Insert irrigation events before yields to allow yield insertion to compute irrigation for soybeans
        self.insert_irrigation_events()
        self.insert_yields()

        # Process each CSV file
        logger.info(f"Processing files from: {data_path}")
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {data_path}")
            return

        for csv_file in csv_files:
            logger.info(f"Processing file: {csv_file.name}")
            self._process_csv_file(csv_file)

    def _process_csv_file(self, csv_path: Path):
        """Process a single CSV file from the analysis folder."""
        # Parse filename components
        try:
            parts = csv_path.stem.split('_')
            if len(parts) < 5:
                raise ValueError("Filename does not have enough parts.")
            field = parts[0]  # 'LINEAR'
            crop_type = parts[1].lower()  # 'CORN' or 'SOYBEAN'
            treatment_str = parts[2]  # e.g., 'trt4'
            treatment = int(treatment_str.replace('trt', ''))
            plot_label = parts[3]  # 'plot'
            plot_number = parts[4]  # e.g., '2009'
        except (IndexError, ValueError) as e:
            logger.error(f"Filename {csv_path.name} is not in the expected format. Error: {e}")
            return

        # Validate treatment for crop type
        if crop_type == 'soybean' and treatment not in SOYBEAN_YIELDS:
            logger.error(f"Invalid treatment {treatment} for Soybean in {csv_path.name}. Skipping file.")
            return
        elif crop_type == 'corn' and treatment not in CORN_YIELDS:
            logger.error(f"Invalid treatment {treatment} for Corn in {csv_path.name}. Skipping file.")
            return

        # Ensure plot exists in plots table
        if plot_number not in PLOT_MAPPINGS:
            logger.error(f"Plot number {plot_number} not found in PLOT_MAPPINGS. Skipping file.")
            return

        # Read CSV data
        try:
            df = pd.read_csv(csv_path, parse_dates=['TIMESTAMP'])
        except Exception as e:
            logger.error(f"Failed to read CSV file {csv_path.name}. Error: {e}")
            return

        # Check if 'TIMESTAMP' column exists
        if 'TIMESTAMP' not in df.columns:
            logger.error(f"'TIMESTAMP' column missing in {csv_path.name}. Available columns: {df.columns.tolist()}")
            return

        # Extract relevant columns
        columns = df.columns.tolist()
        timestamp_col = 'TIMESTAMP'
        variable_cols = [col for col in columns if col != timestamp_col]

        # Log summary of columns
        logger.debug(f"Columns in {csv_path.name}: {columns}")
        logger.info(f"Found {len(variable_cols)} variables in {csv_path.name}")

        # Iterate through each row and insert data
        records = []
        for index, row in df.iterrows():
            timestamp = row[timestamp_col]
            for var in variable_cols:
                value = row[var]
                if pd.isna(value):
                    continue  # Skip missing values
                record = (
                    plot_number,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, pd.Timestamp) else str(timestamp),
                    var,
                    value
                )
                records.append(record)

        # Insert records into the database in batches
        if records:
            try:
                cursor = self.conn.cursor()
                cursor.executemany("""
                    INSERT INTO data (plot_id, timestamp, variable_name, value)
                    VALUES (?, ?, ?, ?)
                """, records)
                self.conn.commit()
                logger.info(f"Inserted {len(records)} records from {csv_path.name}")
            except Exception as e:
                logger.error(f"Failed to insert records from {csv_path.name}. Error: {e}")
        else:
            logger.warning(f"No valid data found in {csv_path.name}")

    def close(self):
        """Close the SQLite connection."""
        self.conn.close()
        logger.info("Closed SQLite connection.")

def create_experiment_store(base_path: str, analysis_folder: str) -> ExperimentDataStore:
    """Create a new experiment data store and process the analysis folder."""
    db_path = Path(base_path) / f'experiment_data_{datetime.now().strftime("%Y%m%d")}.sqlite'
    store = ExperimentDataStore(db_path)
    store.process_analysis_folder(analysis_folder)
    return store

if __name__ == "__main__":
    from pathlib import Path  # Ensure Path is imported in the main block

    # Direct path to where the data is
    base_path = Path(r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et")
    data_folder = Path(r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\analysis-2024-10-10\data\data-2024-10-10")

    # Create and populate the experiment data store using SQLite
    store = create_experiment_store(str(base_path), str(data_folder))
    logger.info(f"Created and populated experiment data store at {store.db_path}")

    # Close the store connection when done
    store.close()
