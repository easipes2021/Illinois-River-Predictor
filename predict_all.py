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
    
    # 1. FILL GAPS: Ensures missing sensor readings don't delete entire rows
    df = df.ffill().bfill()

    # 2. Define 4 Targets (6 hours into the future)
    targets = {
        'hwy_16_flow': 'target_hwy16_6h',
        'hwy_59_flow_est': 'target_hwy59_6h', 
        'lake_francis_height': 'target_lake_6h',
        'watts_ok_flow': 'target_watts_6h'
    }

    # 3. Create the future "truth" columns (Shift -6 hours)
    for col, target_name in targets.items():
        if col in df.columns:
            df[target_name] = df[col].shift(-6)
        else:
            print(f"⚠️ Warning: {col} not found in CSV. Skipping target.")

    # 4. Features: The AI's "Eyes"
    # MUST match exactly what merge_data.py outputs
    features = [
        # --- Current Levels ---
        'savoy_height', 
        'osage_creek_flow', 
        'hwy_59_height',
        
        # --- Lagged Features (The "Wave Detector") ---
        'savoy_height_3h_ago', 
        'savoy_height_6h_ago', 
        'osage_creek_flow_3h_ago', 
        'osage_creek_flow_6h_ago',
        
        # --- Rainfall Data ---
        'precip_fayetteville', 
        'precip_springdale', 
        'precip_bentonville', 
        'precip_siloam',
        
        # --- Soil & Seasonal Logic ---
        'precip_fayetteville_saturation', 
        'seasonal_cycle',
        'lake_headroom'
    ]

    # 5. Train and Save
    for col, target_name in targets.items():
        if target_name in df.columns:
            # Drop rows where either features or the target are NaN
            df_clean = df.dropna(subset=[target_name] + features)
            
            print(f"📊 DATA CHECK for {col}: {len(df_clean)} clean rows.")

            if len(df_clean) < 10:
                print(f"❌ Skipping {col}: Not enough data.")
                continue

            X = df_clean[features]
            y = df_clean[target_name]
            
            # Training the model
            # 150 trees helps the AI better understand these complex lag patterns
            model = RandomForestRegressor(n_estimators=150, min_samples_leaf=1, max_features='sqrt', random_state=42)
            model.fit(X, y)
            
            # Save the model
            joblib.dump(model, f'model_{col}.pkl')
            print(f"✅ Model saved: model_{col}.pkl")

    print("🚀 Success: All AI models updated with lagged trend features.")
    # Add this right after model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=features)
    print(f"Top 3 Drivers for {col}:")
    print(importances.sort_values(ascending=False).head(3))

if __name__ == "__main__":
    train_multi_models()