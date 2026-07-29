"""
Brazilian Used Car Price Prediction (FIPE Dataset)
Author: Matheus Paixão
Domain: Automotive Market Analytics & Machine Learning

This module performs data preprocessing, categorical feature encoding, 
train-test splitting, model training (Random Forest & XGBoost), and 
evaluation (MSE, MAE, R² Score) for predicting vehicle prices in Brazil.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "precos_carros_brasil.csv")


def load_and_clean_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads CSV dataset and performs initial data cleaning (null & duplicate removal)."""
    print(f"[+] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Remove rows where all columns are null
    initial_rows = len(df)
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    cleaned_rows = len(df)
    
    print(f"[+] Data Cleaning: Removed {initial_rows - cleaned_rows} invalid/duplicate rows.")
    print(f"[+] Cleaned Dataset Shape: {df.shape}")
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical variables into numerical values for ML models."""
    df_proc = df.copy()
    
    # Label encode brand
    le_brand = LabelEncoder()
    df_proc["brand"] = le_brand.fit_transform(df_proc["brand"])
    
    # Ensure engine_size is float
    if df_proc["engine_size"].dtype == "object":
        df_proc["engine_size"] = (
            df_proc["engine_size"]
            .astype(str)
            .str.replace(",", ".")
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
        )
    
    # Map fuel and gear to integers
    df_proc["fuel"] = df_proc["fuel"].replace({"Alcool": 0, "Gasolina": 1, "Diesel": 2})
    df_proc["gear"] = df_proc["gear"].replace({"manual": 0, "automatic": 1})
    
    return df_proc


def train_and_evaluate(df: pd.DataFrame):
    """Splits data, trains ML models, and evaluates prediction performance."""
    # Define features and target
    feature_cols = ["brand", "year_model", "engine_size", "fuel", "gear"]
    target_col = "avg_price_brl"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 75% Train, 25% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"[+] Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    models = {
        "Random Forest (Default)": RandomForestRegressor(random_state=42),
        "Random Forest (Tuned)": RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(random_state=42),
    }
    
    results = []
    
    print("\n" + "=" * 60)
    print(f"{'Model Performance Evaluation':^60}")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "MSE": mse,
            "MAE (R$)": mae,
            "R² Score": r2
        })
        
        print(f"-> {name}:")
        print(f"   - MAE: R$ {mae:,.2f}")
        print(f"   - MSE: {mse:,.2f}")
        print(f"   - R² Score: {r2:.4f} ({r2 * 100:.2f}% Variance Explained)")
        print("-" * 60)
        
    return results


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[!] Dataset file not found at {DATA_PATH}")
        return
        
    df_raw = load_and_clean_data(DATA_PATH)
    df_processed = preprocess_features(df_raw)
    train_and_evaluate(df_processed)


if __name__ == "__main__":
    main()
