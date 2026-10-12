import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_ltv_model(cleaned_file_path: str):
    """
    Splits data into an observation window and prediction window,
    engineers features, trains an LTV regression model, and evaluates it.
    """
    print(f"Loading cleaned dataset from {cleaned_file_path}...")
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_file_path}")
        
    df = pd.read_csv(cleaned_file_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    
    print(f"Dataset date range: {min_date.date()} to {max_date.date()}")
    
    # Define split: First 9 months observation, remaining 3 months prediction target
    observation_cutoff = min_date + pd.DateOffset(months=9)
    print(f"Observation Window: {min_date.date()} to {observation_cutoff.date()}")
    print(f"Prediction Window: {observation_cutoff.date()} to {max_date.date()}")
    
    df_obs = df[df['InvoiceDate'] <= observation_cutoff]
    df_pred = df[df['InvoiceDate'] > observation_cutoff]
    
    print("Engineering features from observation window...")
    snapshot_date = observation_cutoff + pd.Timedelta(days=1)
    
    X_features = df_obs.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                                  # Frequency
        'TotalAmount': ['sum', 'mean', 'max']                    # Spend metrics
    })
    
    X_features.columns = ['Recency', 'Frequency', 'Monetary_Total', 'Monetary_Mean', 'Monetary_Max']
    X_features.reset_index(inplace=True)
    
    print("Calculating future spend (Target Variable) from prediction window...")
    y_target = df_pred.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    y_target.columns = ['CustomerID', 'Future_Spend']
    
    model_df = pd.merge(X_features, y_target, on='CustomerID', how='left')
    model_df['Future_Spend'] = model_df['Future_Spend'].fillna(0)
    
    feature_cols = ['Recency', 'Frequency', 'Monetary_Total', 'Monetary_Mean', 'Monetary_Max']
    X = model_df[feature_cols]
    y = model_df['Future_Spend']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): £{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): £{rmse:.2f}")
    print(f"R-Squared (R²): {r2:.4f}")
    
    model_df['Predicted_Future_Spend'] = rf_model.predict(X)
    
    return model_df, rf_model

if __name__ == "__main__":
    input_path = "data/cleaned_online_retail.csv"
    output_path = "data/customer_ltv_predictions.csv"
    
    result_df, model = build_ltv_model(input_path)
    result_df.to_csv(output_path, index=False)
    print(f"\nLTV predictions successfully saved to '{output_path}'.")