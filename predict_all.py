import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

def train_multi_models():
    if not os.path.exists('master_training_data.csv'):
        print("❌ Error: master_training_data.csv not found. Run merge_data.py first.")
        return

    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    
    # 1. FILL GAPS: Ensures missing sensor readings don't delete entire rows
    df = df.ffill().bfill()

    # 2. Define prediction targets and horizons
    base_targets = [
        'hwy_16_flow',
        'hwy_59_flow_est', 
        'watts_ok_flow'
    ]
    horizons = [6, 12, 24]

    # 3. Create the future "truth" columns
    targets = {}
    for col in base_targets:
        if col in df.columns:
            for h in horizons:
                target_name = f'target_{col}_{h}h'
                targets[f'{col}_{h}h'] = target_name
                df[target_name] = df[col].shift(-h)
        else:
            print(f"⚠️ Warning: {col} not found in CSV. Skipping targets.")

    # 4. Features: The AI's "Eyes"
    # MUST match exactly what merge_data.py outputs
    features = [
        # --- Current Levels ---
        'savoy_height', 
        'osage_creek_flow', 
        'hwy_59_height',
        
        # --- Original Lagged Features (The "Wave Detector") ---
        'savoy_height_3h_ago', 
        'savoy_height_6h_ago', 
        'osage_creek_flow_3h_ago', 
        'osage_creek_flow_6h_ago',
        
        # --- 🆕 PHASE 1: Extended Lag Features ---
        'savoy_height_12h_ago',
        'savoy_height_24h_ago',
        'osage_creek_flow_12h_ago',
        'osage_creek_flow_24h_ago',
        
        # --- 🆕 PHASE 1: Trend Indicators ---
        'savoy_height_trend_6h',
        'savoy_height_trend_24h',
        'osage_creek_flow_trend_6h',
        'osage_creek_flow_trend_24h',
        
        # --- Rainfall Data (Original) ---
        'precip_fayetteville', 
        'precip_springdale', 
        'precip_bentonville', 
        'precip_siloam',
        
        # --- Original Saturation (72-hour) ---
        'precip_fayetteville_saturation',
        
        # --- 🆕 PHASE 1: Multiple Precipitation Windows ---
        'precip_fayetteville_24h',
        'precip_fayetteville_48h',
        'precip_fayetteville_168h',
        
        # --- Soil & Seasonal Logic ---
        'seasonal_cycle',
        
        # --- 🆕 PHASE 1: Hour-of-Day Features ---
        'hour_sin',
        'hour_cos'
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
            
            # Transitioned to XGBoost for better handling of non-linear trends and lags
            # n_estimators=500 with learning_rate=0.05 provides better convergence than RF
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                objective='reg:squarederror'
            )
            model.fit(X, y)
            
            # Save the model
            joblib.dump(model, f'model_{col}.pkl')
            print(f"✅ XGBoost Model saved: model_{col}.pkl")

            # Add this right after model.fit(X, y)
            importances = pd.Series(model.feature_importances_, index=features)
            print(f"Top 3 Drivers for {col}:")
            print(importances.sort_values(ascending=False).head(3))

    print("🚀 Success: All AI models updated with lagged trend features.")

if __name__ == "__main__":
    train_multi_models()