import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
from config import VIF_THRESHOLD, CV_FOLDS, RANDOM_SEED
import visualizations as viz

def run_phase_2(df):
    """
    Phase 2: Multiple Linear Regression
    Returns the OLS summary, diagnostics, and the residual series.
    """
    print("--- Phase 2: OLS Regression ---")
    predictors = ['PM2.5', 'PM10', 'NO2', 'SO2', 'Temperature', 'Humidity', 'WindSpeed']
    
    # viz: correlation heatmap
    viz.plot_correlation_heatmap(df, predictors)
    
    X = df[predictors].copy()
    y = df['AQI']
    
    # Add constant for intercept
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()
    
    print(ols_model.summary().tables[0])
    
    # VIF Multicollinearity Check
    print("\n[VIF Check]")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    print(vif_data)
    flagged = vif_data[vif_data["VIF"] > VIF_THRESHOLD]
    if not flagged.empty:
        print(f"WARNING: Variables with VIF > {VIF_THRESHOLD}: {list(flagged['Variable'])}")
        
    # Residual Diagnostics
    print("\n[Residual Diagnostics]")
    # 1. Normality (Jarque-Bera)
    jb_test = sms.jarque_bera(ols_model.resid)
    print(f"Jarque-Bera (Normality) p-value: {jb_test[1]:.4f}")
    
    # 2. Heteroscedasticity (Breusch-Pagan)
    bp_test = sms.het_breuschpagan(ols_model.resid, ols_model.model.exog)
    print(f"Breusch-Pagan (Heteroscedasticity) p-value: {bp_test[1]:.4f}")
    
    # 3. Autocorrelation (Durbin-Watson) -> 2.0 means no autocorrelation
    dw_stat = sm.stats.stattools.durbin_watson(ols_model.resid)
    print(f"Durbin-Watson (Autocorrelation) stat: {dw_stat:.4f}")
    
    # Visualizations
    viz.plot_coefs_ols(ols_model.params.drop('const'), ols_model.pvalues.drop('const'))
    viz.plot_residuals_diagnostic(ols_model.fittedvalues, ols_model.resid)
    
    # Extract Residuals
    residuals = ols_model.resid
    return ols_model, residuals

def run_phase_3(df):
    """
    Phase 3: Advanced Regression (Logistic, Ridge, Lasso)
    """
    print("\n--- Phase 3: Advanced Regression ---")
    predictors = ['PM2.5', 'PM10', 'NO2', 'SO2', 'Temperature', 'Humidity', 'WindSpeed']
    
    X = df[predictors]
    # For Logistic regression, use categorical outcome
    y_cat = df['AQI_Status']
    
    # Train-test split
    X_train, X_test, y_cat_train, y_cat_test, y_train, y_test = train_test_split(
        X, y_cat, df['AQI'], test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Standardize predictors for all models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled_all = scaler.fit_transform(X)
    
    # 1. Logistic Regression
    print("\n[Multinomial Logistic Regression]")
    log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_reg.fit(X_train_scaled, y_cat_train)
    preds = log_reg.predict(X_test_scaled)
    probs = log_reg.predict_proba(X_test_scaled)
    acc = (preds == y_cat_test).mean()
    print(f"Classification Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_cat_test, preds))
    
    # AUC per class
    try:
        auc = roc_auc_score(y_cat_test, probs, multi_class='ovr')
        print(f"ROC-AUC (OVR): {auc:.4f}")
    except ValueError:
        print("ROC-AUC not computed (perhaps only one class in test set)")
        
    # 2. Ridge & Lasso with CV
    print("\n[Ridge & Lasso Regression]")
    alphas = np.logspace(-3, 3, 100)
    
    ridge_cv = RidgeCV(alphas=alphas, cv=CV_FOLDS)
    ridge_cv.fit(X_train_scaled, y_train)
    
    lasso_cv = LassoCV(alphas=alphas, cv=CV_FOLDS, random_state=RANDOM_SEED, max_iter=5000)
    lasso_cv.fit(X_train_scaled, y_train)
    
    print(f"Ridge optimal alpha: {ridge_cv.alpha_:.4f}")
    print(f"Lasso optimal alpha: {lasso_cv.alpha_:.4f}")
    
    r_preds = ridge_cv.predict(X_test_scaled)
    l_preds = lasso_cv.predict(X_test_scaled)
    ols_preds = sm.add_constant(X_test).dot(sm.OLS(y_train, sm.add_constant(X_train)).fit().params)
    
    print("\nRMSE Comparison (Test Set):")
    print(f"OLS:   {mean_squared_error(y_test, ols_preds)**0.5:.4f}")
    print(f"Ridge: {mean_squared_error(y_test, r_preds)**0.5:.4f}")
    print(f"Lasso: {mean_squared_error(y_test, l_preds)**0.5:.4f}")
    
    lasso_coefs = pd.Series(lasso_cv.coef_, index=predictors)
    zero_coefs = lasso_coefs[lasso_coefs == 0]
    if not zero_coefs.empty:
        print(f"\nLasso drove these predictors to zero (feature selection): {list(zero_coefs.index)}")
    else:
        print("\nLasso kept all predictors.")
        
    # Generate Regularization Path info for plotting
    ridge_coefs = []
    lasso_coef_paths = []
    
    for a in alphas:
        r = Ridge(alpha=a).fit(X_scaled_all, df['AQI'])
        ridge_coefs.append(r.coef_)
        
        l = Lasso(alpha=a, max_iter=5000).fit(X_scaled_all, df['AQI'])
        lasso_coef_paths.append(l.coef_)
        
    viz.plot_regularisation_path(alphas, ridge_coefs, "Ridge")
    viz.plot_regularisation_path(alphas, lasso_coef_paths, "Lasso")
    
    return log_reg, ridge_cv, lasso_cv
