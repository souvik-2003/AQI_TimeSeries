# AQI Factor Analysis and Forecasting

This project implements an end-to-end data analysis pipeline to investigate the drivers of the Air Quality Index (AQI) in Indian cities and produce short-term forecasts. It employs a two-phase hybrid study: a regression phase to quantify meteorological and pollutant factors explaining AQI variation, and a time-series phase to model the residual signal and generate 30-day probabilisitic forecasts.

## Project Architecture

The analysis is divided into five sequential phases:
- **Phase 1: Data Preparation:** Data cleaning, missing value imputation, and categorization (Good, Moderate, Hazardous).
- **Phase 2: Multiple Linear Regression:** OLS modeling with significance testing, multicollinearity (VIF) checks, and residual diagnostics.
- **Phase 3: Advanced Regression:** Multinomial Logistic Regression for AQI status classification, and Ridge/Lasso regularization to isolate key predictors.
- **Phase 4: Stationarity & ARIMA/SARIMA:** Modeling the autocorrelation in regression residuals using Box-Jenkins methodology. 
- **Phase 5: ARCH/GARCH & Forecasting:** Modeling volatility clustering an generating probabilistic 30-day AQI forecasts.

## Installation

Ensure you have Python 3.8+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/souvik-2003/AQI_TimeSeries.git
   cd AQI_TimeSeries
   ```

2. Setup a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   
   # For PDF generation
   pip install fpdf2
   ```

## Usage

### Run the Pipeline
To execute the pipeline and generate all analytics and plots in the console:
```bash
python main.py
```

### Generate the PDF Report
To run the full pipeline and bind all textual outputs and visualizations into a formatted PDF required for the project submission:
```bash
python report_generator.py
```

This will output the final `AQI_Project_Report.pdf` file in the project's root folder.

## Outputs

All visualizations are stored in the `plots/` directory when the scripts are run. These include:
- `aqi_distribution.png`: Histogram categorizing the AQI statuses.
- `correlation_heatmap.png`: Matrix showing statistical relationships between predictors.
- `ols_coefficients.png`: Predictor influence graph.
- `residual_diagnostics.png`: QQ-plots and heteroscedasticity checks.
- `acf_pacf.png`: Analysis of temporal correlation.
- `ridge_path.png` / `lasso_path.png`: Penalty parameter impact on predictors.
- `forecast_garch.png`: The calculated 30-day forecast paired with confidence margins. 

## Dataset Notes
The codebase requires the historical Indian City AQI csv from the Central Pollution Control Board (CPCB) placed within the `data/` directory (named `aqi_data.csv`).
