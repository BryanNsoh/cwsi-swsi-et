# Agricultural Irrigation Experiment Database Documentation

## Overview
This database contains data from an agricultural irrigation experiment conducted in 2024, comparing different irrigation strategies for corn and soybean crops. The experiment includes various sensor measurements, irrigation events, and yield data across multiple treatment plots.

## Database Schema

### 1. plots Table
Primary table establishing the base experimental units.

Fields:
- `plot_id` (TEXT, PRIMARY KEY): Unique identifier for each plot (e.g., "5001", "2001")
- `treatment` (INTEGER): Treatment number (1-6)
- `field` (TEXT): Field identifier ("LINEAR_CORN" or "LINEAR_SOYBEAN")
- `node` (TEXT): Sensor node location ("A", "B", or "C")

### 2. data Table
Contains all sensor readings and calculated indices.

Fields:
- `id` (INTEGER, PRIMARY KEY): Auto-incrementing unique identifier
- `plot_id` (TEXT): References plots.plot_id
- `timestamp` (TEXT): Timestamp in "YYYY-MM-DD HH:MM:SS" format
- `variable_name` (TEXT): Name of the measured variable
- `value` (REAL): Measured or calculated value

Variable Names and Units:
1. Weather Variables:
   - `BatV`: Battery Voltage (V)
   - `Ta_2m_Avg`: Air Temperature at 2m height (°C)
   - `RH_2m_Avg`: Relative Humidity at 2m height (%)
   - `WndAveSpd_3m`: Wind Speed at 3m height (m/s)
   - `WndAveDir_3m`: Wind Direction at 3m height (degrees)
   - `PresAvg_1pnt5m`: Atmospheric Pressure at 1.5m height (kPa)
   - `Solar_2m_Avg`: Solar Radiation at 2m height (W/m²)
   - `Rain_1m_Tot`: Rainfall at 1m height (mm)
   - `TaMax_2m`: Maximum Air Temperature at 2m height (°C)
   - `TaMin_2m`: Minimum Air Temperature at 2m height (°C)
   - `RHMax_2m`: Maximum Relative Humidity at 2m height (%)
   - `RHMin_2m`: Minimum Relative Humidity at 2m height (%)

2. Calculated Indices:
   - `eto`: Reference Evapotranspiration (mm/day)
   - `etc`: Crop Evapotranspiration (mm/day)
   - `kc`: Crop Coefficient (dimensionless)
   - `cwsi`: Crop Water Stress Index (dimensionless)
   - `swsi`: Soil Water Stress Index (dimensionless)

3. Sensor Readings:
Various sensor readings follow this nomenclature:
`[SensorType][PlotNumber][Node][Treatment][Depth][Year]`

Example: `TDR5001A20624`
- `SensorType`: Type of sensor
  - `TDR`: Time Domain Reflectometry (measures soil moisture)
  - `IRT`: Infrared Thermometer (measures canopy temperature)
  - `WAM`: Watermark (measures soil water potential)
  - `SAP`: Sapflow
  - `DEN`: Dendrometer
- `PlotNumber`: Four-digit plot identifier
- `Node`: Single letter (A, B, or C)
- `Treatment`: Single digit (1-6)
- `Depth`: Two-digit number (cm) or 'xx' for non-applicable sensors
  - Common depths: 06, 18, 30, 42 cm
- `Year`: Two-digit year ('24')

### 3. yields Table
Contains yield data for each plot.

Fields:
- `id` (INTEGER, PRIMARY KEY): Auto-incrementing unique identifier
- `plot_id` (TEXT): References plots.plot_id
- `trt_name` (TEXT): Treatment name
- `crop_type` (TEXT): "corn" or "soybean"
- `avg_yield_bu_ac` (REAL): Average yield in bushels per acre
- `yield_kg_ha` (REAL): Average yield in kilograms per hectare
- `irrigation_applied_inches` (REAL): Total irrigation applied in inches
- `irrigation_applied_mm` (REAL): Total irrigation applied in millimeters

### 4. irrigation_events Table
Records individual irrigation events.

Fields:
- `id` (INTEGER, PRIMARY KEY): Auto-incrementing unique identifier
- `plot_id` (TEXT): References plots.plot_id
- `treatment` (INTEGER): Treatment number (1-6)
- `trt_name` (TEXT): Treatment name
- `date` (TEXT): Date of irrigation event (YYYY-MM-DD)
- `amount_inches` (REAL): Irrigation amount in inches
- `amount_mm` (REAL): Irrigation amount in millimeters
- `notes` (TEXT): Additional information about the irrigation event

## Treatments
1. IoT-Fuzzy: Internet of Things-based Fuzzy Logic Control
2. CWSI + SWSI: Combined Crop and Soil Water Stress Index
3. CWSI only: Crop Water Stress Index
4. SWSI: Soil Water Stress Index
5. ET-Model: Evapotranspiration Model
6. Grower's Practice: Traditional irrigation management

## Plot Organization
- Corn Plots: 5xxx series (e.g., 5001, 5006)
- Soybean Plots: 2xxx series (e.g., 2001, 2006)

## Important Notes
1. Irrigation events before August 5, 2024, used averaged amounts across plots within the same treatment.
2. After August 5, 2024, irrigation amounts were plot-specific.
3. For soybeans, total irrigation amounts are computed by summing all irrigation events for each plot.
4. Some sensor readings may contain NULL values, indicated by missing data in the CSV files.
5. A special irrigation event occurred on July 26 for fertigation (fertilizer application with irrigation) where 0.3 inches were deducted from the application amount.

## Example Queries

1. Get total irrigation by plot:
```sql
SELECT plot_id, SUM(amount_inches) as total_inches, SUM(amount_mm) as total_mm
FROM irrigation_events
GROUP BY plot_id;
```

2. Get all sensor readings for a specific plot on a given day:
```sql
SELECT timestamp, variable_name, value
FROM data
WHERE plot_id = '5001' 
AND date(timestamp) = '2024-07-15'
ORDER BY timestamp;
```

3. Compare yields across treatments:
```sql
SELECT t.treatment, t.trt_name, 
       AVG(y.avg_yield_bu_ac) as avg_yield,
       AVG(y.irrigation_applied_inches) as avg_irrigation
FROM plots p
JOIN yields y ON p.plot_id = y.plot_id
GROUP BY t.treatment, t.trt_name;
```

## Data Quality Considerations
1. Sensor data may contain gaps due to maintenance or equipment issues
2. Weather data is collected at plot level and may show slight variations between plots
3. Irrigation amounts are precise to 0.01 inches
4. Yield measurements are averaged across multiple samples within each plot

## Units Summary
- Temperature: Celsius (°C)
- Humidity: Percentage (%)
- Wind Speed: meters per second (m/s)
- Wind Direction: degrees
- Pressure: kiloPascals (kPa)
- Solar Radiation: Watts per square meter (W/m²)
- Rainfall: millimeters (mm)
- Irrigation: inches and millimeters (dual reporting)
- Yield: bushels per acre (bu/ac) and kilograms per hectare (kg/ha)