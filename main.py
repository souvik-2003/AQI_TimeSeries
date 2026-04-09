import os
from data_processor import load_and_clean
from regression_analysis import run_phase_2, run_phase_3
from time_series_analysis import run_phase_4, run_phase_5

def main():
    print("=" * 60)
    print("  AQI FACTOR ANALYSIS AND FORECASTING - UNIFIED REPORT  ")
    print("=" * 60)
    
    # Phase 1: Load and prepare data
    df = load_and_clean()
    print(f"\nSuccessfully loaded {len(df)} days of AQI data.\n")
    
    # Phase 2: Multiple Linear Regression (OLS)
    ols_model, residuals = run_phase_2(df)
    
    # Phase 3: Advanced Regression
    log_reg, ridge_cv, lasso_cv = run_phase_3(df)
    
    # Phase 4: Stationarity and ARIMA
    arima_model = run_phase_4(residuals)
    
    # Phase 5: ARCH/GARCH and Forecasting
    final_arima, garch_model, forecast_mean, forecast_95 = run_phase_5(arima_model, df)
    
    # Unified Summary
    print("\n" + "=" * 60)
    print("  FINAL CONCLUSIONS  ")
    print("=" * 60)
    
    # Top predictors from OLS
    pvals = ols_model.pvalues.drop('const')
    sig_predictors = pvals[pvals < 0.05].index.tolist()
    print("1. Key Drivers of AQI (OLS Significance p < 0.05):")
    print(f"   {', '.join(sig_predictors) if sig_predictors else 'None'}")
    
    # Features removed by Lasso
    lasso_coefs = lasso_cv.coef_
    zero_f = [col for col, coef in zip(['PM2.5', 'PM10', 'NO2', 'SO2', 'Temperature', 'Humidity', 'WindSpeed'], lasso_coefs) if coef == 0]
    print(f"\n2. Predictors eliminated by Lasso Regularization:")
    print(f"   {', '.join(zero_f) if zero_f else 'None'}")
    
    # Time Series Best Model
    print(f"\n3. Time Series Dynamics:")
    print(f"   Best ARIMA Model for Residuals: {arima_model.order}")
    print(f"   Volatility Model Evaluated: GARCH(1,1)")
    
    # Forecasting Output
    print("\n4. Next 5 Days Forecasting Summary (Total AQI):")
    for date, mean_val, lower, upper in zip(forecast_mean.index[:5], forecast_mean.values[:5], forecast_95['Lower'].values[:5], forecast_95['Upper'].values[:5]):
        print(f"   {date.date()}: {mean_val:.1f} (95% CI: {lower:.1f} to {upper:.1f})")
        
    print("\nAll plots have been successfully saved to the 'plots/' directory.")
    print("=" * 60)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
