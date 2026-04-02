import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_multi_models():
    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    
    # 1. Define our 4 Targets (6 hours into the future)
    targets = {
        'hwy_16_flow': 'target_hwy16_6h',
        'hwy_59_flow_est': 'target_hwy59_6h', # Our new estimated flow
        'lake_francis_height': 'target_lake_6h',
        'watts_ok_height': 'target_watts_6h'
    }

    # 2. Create the future "truth" columns for the AI to learn from
    for col, target_name in targets.items():
        if col in df.columns:
            df[target_name] = df[col].shift(-6)

    # 3. Features (The inputs the AI uses)
    features = [
        'savoy_height', 
        'osage_creek_flow', 
        'hwy_59_height',
        'precip_fayetteville', 
        'precip_springdale', 
        'precip_bentonville', 
        'precip_siloam',
        'precip_fayetteville_saturation', # New saturation features
        'seasonal_cycle',
        'lake_headroom'
    ]

    # 4. Train a specific model for each location
    models = {}
    for col, target_name in targets.items():
        if col in df.columns:
            print(f"Training model for {col}...")
            df_clean = df.dropna(subset=[target_name] + features)
            
            X = df_clean[features]
            y = df_clean[target_name]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            models[col] = model
            
            # Save each model individually
            joblib.dump(model, f'model_{col}.pkl')

    print("Success: All 4 models trained and saved.")

if __name__ == "__main__":
    train_multi_models()