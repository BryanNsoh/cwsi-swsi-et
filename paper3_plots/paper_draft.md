# Assessing Data-Driven Irrigation Scheduling Methods: Evaluating the Integration of Plant and Soil Water Stress Indices in the Crop2Cloud Platform

## 1. Introduction
- Global context of water scarcity in agriculture and critical need for efficient irrigation
- Evolution from traditional irrigation scheduling to data-driven approaches
- Challenges in current practices: labor-intensive data collection, delayed decision-making, integration issues
- Overview of scheduling methods evaluated:
  - CWSI-only scheduling
  - SWSI-only scheduling
  - Combined weighted indices (60% SWSI, 40% CWSI)
  - ET-based scheduling (baseline)
- Novel aspects of combining plant and soil water stress indicators
- Brief reference to Crop2Cloud platform implementation
- Research objectives:
  - Primary: Evaluate efficacy of combining plant and soil water stress indices
  - Secondary: Compare performance metrics, analyze index relationships, assess integration benefits

## 2. Methods

### 2.1 Study Site and Experimental Design

[Figure 1: Experimental Field Layout and Treatment Distribution]
*Detailed aerial view of the experimental field at WCREEC showing: (a) Complete field layout with span positions marked, (b) Individual plot boundaries with dimensions (5.3m × 33.2m), (c) Treatment distribution across plots with original boundaries overlaid with implemented treatment boundaries, (d) Sensor locations for each treatment, (e) Legend indicating treatment types and replication patterns. Scale bar and north arrow included.*

### 2.2 Data Collection Systems and Processing

[Figure 2: Sensor Network Architecture and Data Processing Workflow]
*Technical diagram showing: (a) Field sensor deployment with TDR depths and IRT positioning, (b) Communication pathways, (c) Power management setup, (d) Data processing workflow from collection through SMS Advanced software to GIS analysis, (e) Outlier identification and handling protocols including statistical thresholds and cross-validation methods.*

### 2.3 Environmental Conditions and Field Measurements

[Figure 3: Seasonal Weather Patterns and Environmental Variables]
*Multi-panel time series showing: (a) Daily precipitation and irrigation events, (b) Maximum/minimum air temperatures, (c) Solar radiation, (d) Wind speed, (e) Calculated vapor pressure deficit (VPD). All panels aligned on same time axis covering entire growing season with key growth stages marked.*

### 2.4 Irrigation Scheduling Methods Implementation

#### 2.4.1 CWSI Calculation and Validation

[Figure 4: CWSI Implementation Analysis]
*Four-panel figure showing: (a) Daily CWSI values for corn across growing season with theoretical bounds, (b) Same for soybeans showing transition point to empirical method, (c) Boxplots of CWSI distribution by growth stage, (d) Scatterplot comparing theoretical vs empirical CWSI values with 1:1 line.*

#### 2.4.2 SWSI Computation and Soil Moisture Monitoring

[Figure 5: Soil Moisture Monitoring and SWSI Calculation]
*Time series showing: (a) TDR-measured soil moisture at three depths with neutron probe validation points, (b) Calculated SWSI values with uncertainty bands, (c) Rainfall/irrigation events marked, (d) Cross-sectional soil moisture profile evolution.*

#### 2.4.3 Integration of Indices

[Figure 6: Combined Index Analysis and Weighting System]
*Three-panel comparison showing: (a) CWSI vs SWSI correlation analysis, (b) Time series of both indices with 60/40 weighted average overlay, (c) Decision threshold analysis showing irrigation trigger points based on combined index.*

### 2.5 Statistical Analysis Methods
- ANOVA design for yield comparisons
- Post-hoc tests for treatment differences
- Correlation analysis for index relationships
- Confidence interval calculations for efficiency metrics
- Spatial analysis of yield patterns using GIS

## 3. Results and Discussion

### 3.1 Response to Irrigation Events

[Figure 7: System Response Analysis]
*Multi-day analysis showing: (a) Canopy temperature response to irrigation/rainfall over 7-day periods, (b) Corresponding soil moisture changes at different depths, (c) Recovery patterns across treatments, (d) Time lag analysis between irrigation and plant response.*

### 3.2 Treatment Comparisons

[Figure 8: Monthly Stress Index Distributions]
*Monthly boxplots showing: (a) CWSI distribution across treatments, (b) SWSI distribution across treatments, (c) Combined index distribution, (d) Statistical comparison of distributions with confidence intervals.*

[Figure 9: Irrigation Application Patterns]
*Stacked bar graph showing: (a) Daily irrigation amounts by treatment, color-coded by application date, (b) Cumulative irrigation totals, (c) Timing of applications relative to stress indices, (d) Statistical analysis of treatment differences.*

### 3.3 Yield Analysis and Water Use Efficiency

[Figure 10: Yield Distribution Analysis]
*Comprehensive yield analysis showing: (a) Box plots of yield distribution by treatment with statistical significance indicators, (b) Individual plot yields with outliers identified, (c) Spatial yield patterns from GIS analysis, (d) Treatment effect size analysis.*

[Figure 11: Water Use Efficiency Metrics]
*Comparison of efficiency metrics: (a) IWUE by treatment [(Irrigated yield - Rainfed yield) / Irrigation water applied], (b) CWUE by treatment [Yield / Total evapotranspiration], (c) Relationship between applied water and yield, (d) Economic analysis of water use efficiency.*

## 4. Conclusions
- Synthesis of findings across methods
- Statistical significance of treatment differences
- Practical implications for irrigation scheduling
- Recommendations for implementation
- Future research priorities and technology development needs