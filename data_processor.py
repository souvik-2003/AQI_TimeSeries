import os
import pandas as pd
import numpy as np
from config import DATA_PATH, AQI_THRESHOLDS, RANDOM_SEED

def map_aqi_status(aqi):
    """Map AQI numeric to a status string based on config breakpoints."""
    if pd.isna(aqi):
        return np.nan
    for status, (low, high) in AQI_THRESHOLDS.items():
        if low <= aqi <= high:
            return status
    return "Hazardous" # Fallback for extreme values

def load_and_clean():
    """
    Load data, handle missing values, and engineer features.
    If DATA_PATH doesn't exist, invokes generate_synthetic_data() first.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Generating synthetic data...")
        generate_synthetic_data()
        
    df = pd.read_csv(DATA_PATH)
    
    # Standardise dates and index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # If multiple cities exist, filter for Kolkata
    if 'City' in df.columns:
        if 'Kolkata' in df['City'].values:
            df = df[df['City'] == 'Kolkata']
            df = df.drop(columns=['City'])

    # Numeric columns to process
    numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'Temperature', 'Humidity', 'WindSpeed', 'AQI']
    # Ensure columns exist, creating NaN ones if not
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Missing value handling
    for col in numeric_cols:
        # Forward-fill for small gaps (<= 3 consecutive days)
        df[col] = df[col].ffill(limit=3)
        # Linear interpolation for remaining gaps
        df[col] = df[col].interpolate(method='linear')
        
    # Drop rows that still have NaNs (usually exactly at the start)
    df = df.dropna(subset=numeric_cols).copy()

    # Create categorical label
    df['AQI_Status'] = df['AQI'].apply(map_aqi_status)
    # Define as ordered categorical
    df['AQI_Status'] = pd.Categorical(
        df['AQI_Status'], 
        categories=['Good', 'Moderate', 'Hazardous'], 
        ordered=True
    )
    
    # Data Quality Report
    print("=== Data Quality Report ===")
    print(f"Total rows: {len(df)}")
    missing_rate = df.isna().mean() * 100
    print("\nMissing values remaining (%):")
    print(missing_rate[missing_rate > 0].to_string() if missing_rate.sum() > 0 else "None")
    
    print("\nOutliers detected (> 3 SD):")
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        outliers = (df[col] < mean - 3*std) | (df[col] > mean + 3*std)
        count = outliers.sum()
        if count > 0:
            print(f"  {col}: {count} outliers ({(count/len(df))*100:.2f}%)")
            
    print("\nAQI_Status Distribution:")
    print(df['AQI_Status'].value_counts())
    print("===========================\n")
    
    return df

def generate_synthetic_data():
    """Generates a synthetic Kolkata dataset for the years 2015-2023."""
    np.random.seed(RANDOM_SEED)
    dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Seasonality effects: high in winter, low in monsoon
    time_arr = np.arange(n)
    seasonality = np.cos(2 * np.pi * time_arr / 365.25)
    
    # Generate basic vars
    df = pd.DataFrame({'Date': dates})
    
    # Temperature: higher in summer
    df['Temperature'] = 28 + 6 * (-seasonality) + np.random.normal(0, 2, n)
    df['Humidity'] = 65 + 15 * (-seasonality) + np.random.normal(0, 5, n)
    df['WindSpeed'] = 3 + 1.5 * (-seasonality) + np.random.normal(0, 0.5, n)
    # Wind speed can't be negative
    df['WindSpeed'] = np.clip(df['WindSpeed'], 0.1, None)
    
    # Pollutants (higher in winter when seasonality > 0)
    df['PM2.5'] = 80 + 50 * seasonality + np.random.normal(0, 15, n)
    df['PM10'] = 120 + 70 * seasonality + np.random.normal(0, 20, n)
    df['NO2'] = 40 + 20 * seasonality + np.random.normal(0, 8, n)
    df['SO2'] = 15 + 5 * seasonality + np.random.normal(0, 3, n)
    
    # Clip pollutants to be > 0
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2']
    for p in pollutants:
        df[p] = np.clip(df[p], 5, None)
    
    # Composite AQI (simple proxy: dominated by PM2.5 + some PM10 + noise)
    df['AQI'] = (df['PM2.5'] * 1.5) + (df['PM10'] * 0.3) + df['NO2'] * 0.1 + np.random.normal(0, 10, n)
    df['AQI'] = np.clip(df['AQI'], 10, 500)
    
    # City
    df['City'] = 'Kolkata'
    
    # Add some random missingness
    for col in pollutants + ['Temperature', 'Humidity', 'WindSpeed', 'AQI']:
        mask = np.random.rand(n) < 0.02
        df.loc[mask, col] = np.nan
        
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved synthetic dataset to {DATA_PATH}")

if __name__ == "__main__":
    load_and_clean()
