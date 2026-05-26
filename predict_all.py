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
    base_features = [
        # --- Current Levels (Upstream) ---
        'savoy_height', 
        'osage_creek_flow', 
        
        # --- Original Lagged Features ---
        'savoy_height_3h_ago', 'savoy_height_6h_ago', 
        'osage_creek_flow_3h_ago', 'osage_creek_flow_6h_ago',
        
        # --- Extended Lag Features ---
        'savoy_height_12h_ago', 'savoy_height_24h_ago',
        'osage_creek_flow_12h_ago', 'osage_creek_flow_24h_ago',
        
        # --- Trend Indicators ---
        'savoy_height_trend_6h', 'savoy_height_trend_24h',
        'osage_creek_flow_trend_6h', 'osage_creek_flow_trend_24h',
        
        # --- Rainfall Data ---
        'precip_fayetteville', 'precip_springdale', 'precip_bentonville', 'precip_siloam',
        
        # --- Precipitation Windows & Saturation ---
        'precip_fayetteville_saturation',
        'precip_fayetteville_24h', 'precip_fayetteville_48h', 'precip_fayetteville_168h',
        'precip_fayetteville_720h', # 🆕 PHASE 2: 30-day soil moisture
        
        # --- 🆕 PHASE 2: NWS QPF Forward Windows ---
        'qpf_next_6h', 'qpf_next_12h', 'qpf_next_24h',
        
        # --- Soil & Seasonal Logic ---
        'seasonal_cycle', 'hour_sin', 'hour_cos'
    ]

    # 🆕 PHASE 2: Cascade Inputs Architecture
    hwy_16_features = base_features.copy()
    
    hwy_59_features = base_features + [
        'hwy_16_flow', 'hwy_16_height',
        'hwy_16_flow_3h_ago', 'hwy_16_flow_6h_ago', 'hwy_16_flow_12h_ago', 'hwy_16_flow_trend_6h',
        'hwy_16_height_3h_ago', 'hwy_16_height_6h_ago', 'hwy_16_height_12h_ago', 'hwy_16_height_trend_6h'
    ]
    
    watts_features = hwy_59_features + [
        'hwy_59_height', 'hwy_59_flow_est',
        'hwy_59_height_3h_ago', 'hwy_59_height_6h_ago', 'hwy_59_height_trend_6h',
        'hwy_59_flow_est_3h_ago', 'hwy_59_flow_est_6h_ago', 'hwy_59_flow_est_trend_6h'
    ]
    
    feature_sets = {
        'hwy_16_flow': hwy_16_features,
        'hwy_59_flow_est': hwy_59_features,
        'watts_ok_flow': watts_features
    }

    # 5. Train and Save
    for col, target_name in targets.items():
        base_target_col = '_'.join(col.split('_')[:-1]) if 'hwy_59' not in col else 'hwy_59_flow_est'
        if col.startswith('hwy_16'): base_target_col = 'hwy_16_flow'
        elif col.startswith('hwy_59'): base_target_col = 'hwy_59_flow_est'
        elif col.startswith('watts'): base_target_col = 'watts_ok_flow'
            
        features = feature_sets.get(base_target_col, base_features)
        
        if target_name in df.columns:
            # Drop rows where either features or the target are NaN
            df_clean = df.dropna(subset=[target_name] + features)
            
            print(f"📊 DATA CHECK for {col}: {len(df_clean)} clean rows.")

            if len(df_clean) < 10:
                print(f"❌ Skipping {col}: Not enough data.")
                continue

            X = df_clean[features]
            y = df_clean[target_name]
            
            # 🆕 PHASE 2: Sample weighting for flood events
            # If the current flow is high, weight the sample 3x higher so the model learns from floods
            sample_weights = np.ones(len(y))
            if base_target_col in df_clean.columns:
                threshold = df_clean[base_target_col].quantile(0.95) # Top 5% of flow events
                sample_weights = np.where(df_clean[base_target_col] > threshold, 3.0, 1.0)
            
            # Transitioned to XGBoost for better handling of non-linear trends and lags
            # Configured with regularization (max_depth=3, reg_alpha=1.0, reg_lambda=10.0, etc.)
            # to prevent long-term trailing rainfall features from dominating predictions.
            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=10.0,
                random_state=42,
                objective='reg:squarederror'
            )
            model.fit(X, y, sample_weight=sample_weights)
            
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