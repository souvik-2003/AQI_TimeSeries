import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data Path
DATA_PATH = os.path.join(DATA_DIR, "aqi_data.csv")

# Reproducibility
RANDOM_SEED = 42

# Validation & Forecasting Constants
CV_FOLDS = 5
VIF_THRESHOLD = 10
FORECAST_HORIZON = 30

# Model bounds
ARIMA_MAX_P = 5
ARIMA_MAX_Q = 5
GARCH_P = 1
GARCH_Q = 1

# AQI Categories Mapping definition based on breakpoints (0-100, 101-200, 201+)
# Used for discretization
AQI_THRESHOLDS = {
    "Good": (0, 100),
    "Moderate": (101, 200),
    "Hazardous": (201, 2000)  # arbitrary high bound for 201+
}
