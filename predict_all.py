import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_multi_models():
    if not os.path.exists('master_training_data.csv'):
        print("❌ Error: master_training_data.csv not found. Run merge_data.py first.")
        return

    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    # FILL GAPS: Copy the last known value forward to fill holes
    df = df.ffill().bfill()


    # 1. Define 4 Targets (6 hours into the future)
    # CHANGED: Swapped watts_ok_height to watts_ok_flow
    targets = {
        'hwy_16_flow': 'target_hwy16_6h',
        'hwy_59_flow_est': 'target_hwy59_6h', 
        'lake_francis_height': 'target_lake_6h', # Keep as height per your request
        'watts_ok_flow': 'target_watts_6h'     # Changed to Flow
    }

    # 2. Create the future "truth" columns
    for col, target_name in targets.items():
        if col in df.columns:
            df[target_name] = df[col].shift(-6)
        else:
            print(f"⚠️ Warning: {col} not found in CSV. Skipping target.")

    # 3. Features (Inputs)
    # MUST match exactly what merge_data.py outputs
    features = [
        'savoy_height', 
        'osage_creek_flow', 
        'hwy_59_height',
        'precip_fayetteville', 
        'precip_springdale', 
        'precip_bentonville', 
        'precip_siloam',
        'precip_fayetteville_saturation', 
        'seasonal_cycle',
        'lake_headroom'
    ]

    # 4. Train and Save
    for col, target_name in targets.items():
        if target_name in df.columns:
            print(f"Training model for {col}...")
            
            # Drop rows where either features or the target are NaN
            df_clean = df.dropna(subset=[target_name] + features)
            
            if df_clean.empty:
                print(f"❌ Not enough data to train {col}. Check for NaNs.")
                continue

            X = df_clean[features]
            y = df_clean[target_name]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save using the original column name so the UI knows which is which
            joblib.dump(model, f'model_{col}.pkl')

    print("✅ Success: All models trained and saved as .pkl files.")


    models = {}
    for col, target_name in targets.items():
        if col in df.columns:
            # THIS IS THE LINE THAT DELETES DATA:
            df_clean = df.dropna(subset=[target_name] + features)
            
            # --- ADD THIS CHECK HERE ---
            print(f"📊 DATA CHECK for {col}:")
            print(f"   Original rows: {len(df)}")
            print(f"   Rows after dropping NaNs: {len(df_clean)}")
            # ---------------------------

            if len(df_clean) < 10:
                print(f"   ❌ Skipping {col}: Not enough clean rows.")
                continue

if __name__ == "__main__":
    train_multi_models()