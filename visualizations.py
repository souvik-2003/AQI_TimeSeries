import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOTS_DIR

sns.set_theme(style="whitegrid")

def save_fig(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path

def plot_aqi_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='AQI_Status', order=['Good', 'Moderate', 'Hazardous'], palette='vlag', ax=ax)
    ax.set_title('AQI Status Distribution')
    return save_fig(fig, 'aqi_distribution.png')

def plot_correlation_heatmap(df, cols):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax)
    ax.set_title('Predictor Correlation Heatmap')
    return save_fig(fig, 'correlation_heatmap.png')

def plot_coefs_ols(coefs, pvalues, title="OLS Coefficients"):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['green' if p < 0.05 else 'gray' for p in pvalues]
    coefs.plot(kind='barh', color=colors, ax=ax)
    ax.set_title(f"{title} (Green if p < 0.05)")
    ax.axvline(0, color='black', linewidth=1)
    return save_fig(fig, 'ols_coefficients.png')

def plot_residuals_diagnostic(fitted, residuals):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter vs Fitted
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5, ax=ax1)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title('Residuals vs Fitted (Heteroscedasticity)')
    ax1.set_xlabel('Fitted AQI')
    ax1.set_ylabel('Residuals')
    
    # Histogram distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title('Residuals Distribution (Normality)')
    
    return save_fig(fig, 'residual_diagnostics.png')

def plot_regularisation_path(alphas, coefs, model_name="Ridge"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    ax.set_xlabel('Alpha (Log Scale)')
    ax.set_ylabel('Coefficients')
    ax.set_title(f'{model_name} Regularisation Path')
    ax.axis('tight')
    return save_fig(fig, f'{model_name.lower()}_path.png')

def plot_acf_pacf(series, title="ACF & PACF"):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series.dropna(), lags=40, ax=ax1, title=f"ACF: {title}")
    plot_pacf(series.dropna(), lags=40, ax=ax2, title=f"PACF: {title}")
    plt.tight_layout()
    return save_fig(fig, 'acf_pacf.png')

def plot_forecast_garch(mean_forecast, conf_int_80, conf_int_95, historical_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot last 60 days of historical
    hist_subset = historical_data.iloc[-60:]
    ax.plot(hist_subset.index, hist_subset.values, label='Historical AQI', color='black')
    
    # Forecast
    forecast_idx = mean_forecast.index
    ax.plot(forecast_idx, mean_forecast.values, label='Forecast Mean', color='blue')
    
    # Confidence Intervals
    ax.fill_between(forecast_idx, conf_int_95.iloc[:, 0], conf_int_95.iloc[:, 1], color='blue', alpha=0.1, label='95% CI')
    ax.fill_between(forecast_idx, conf_int_80.iloc[:, 0], conf_int_80.iloc[:, 1], color='blue', alpha=0.3, label='80% CI')
    
    ax.set_title('30-Day AQI Forecast with GARCH Uncertainty Bounds')
    ax.set_ylabel('AQI')
    ax.legend(loc='upper left')
    
    return save_fig(fig, 'forecast_garch.png')
