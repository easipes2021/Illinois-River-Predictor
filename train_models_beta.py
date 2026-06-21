import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

def retrain_beta_models():
    print("🚀 Training High-Resolution Beta (Watershed) XGBoost Models...")
    df = pd.read_csv('master_training_data_beta.csv', index_col=0, parse_dates=True)

    locations = {
        'hwy_16_flow': 'Hwy 16',
        'hwy_59_flow_est': 'Hwy 59',
        'watts_ok_flow': 'Watts Bridge'
    }

    horizons = {'6h': 24, '12h': 48, '24h': 96}  # Updated shifts for 15-min intervals

    # Base features: gauges and QPF
    gauge_features = [
        'savoy_height', 'osage_creek_flow',
        'savoy_height_3h_ago', 'savoy_height_6h_ago', 
        'osage_creek_flow_3h_ago', 'osage_creek_flow_6h_ago',
        'savoy_height_12h_ago', 'savoy_height_24h_ago',
        'osage_creek_flow_12h_ago', 'osage_creek_flow_24h_ago',
        'savoy_height_trend_6h', 'savoy_height_trend_24h',
        'osage_creek_flow_trend_6h', 'osage_creek_flow_trend_24h',
        'qpf_next_6h', 'qpf_next_12h', 'qpf_next_24h'
    ]

    hwy16_gauges = [
        'hwy_16_flow', 'hwy_16_height',
        'hwy_16_flow_3h_ago', 'hwy_16_flow_6h_ago', 'hwy_16_flow_12h_ago', 'hwy_16_flow_trend_6h',
        'hwy_16_height_3h_ago', 'hwy_16_height_6h_ago', 'hwy_16_height_12h_ago', 'hwy_16_height_trend_6h'
    ]

    # Dynamically grab all 26 HUC-12 precipitation features
    precip_features = [c for c in df.columns if c.startswith('precip_11110103')]

    base_features = gauge_features + precip_features

    for target in locations.keys():
        print(f"\n--- Training High-Res Beta XGBoost models for {target} ---")
        
        features = base_features.copy()
        if target in ['hwy_59_flow_est', 'watts_ok_flow']:
            features.extend(hwy16_gauges)
            
        for horizon, shift_steps in horizons.items():
            model_df = df.copy()
            model_df['target_future'] = model_df[target].shift(-shift_steps)
            model_df = model_df.dropna(subset=features + ['target_future'])
            
            # Subsample 0 flow values to prevent biasing the model
            zero_mask = model_df['target_future'] <= 5
            zeros_df = model_df[zero_mask]
            non_zeros_df = model_df[~zero_mask]
            if len(zeros_df) > 200:
                zeros_df = zeros_df.sample(200, random_state=42)
            model_df = pd.concat([zeros_df, non_zeros_df]).sort_index()

            X = model_df[features]
            y = model_df['target_future']
            
            # Use 'hist' tree method which is drastically faster and scales well with 200+ features
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                objective='reg:absoluteerror',
                random_state=42
            )
            
            model.fit(X, y)
            filename = f'model_beta_{target}_{horizon}.pkl'
            joblib.dump(model, filename)
            print(f"✅ Saved {filename} with {len(features)} features")

if __name__ == '__main__':
    retrain_beta_models()
