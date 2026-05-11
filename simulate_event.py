import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

def simulate_rain(added_inches):
    """
    Simulates the impact of additional rainfall on all gauges.
    """
    if not os.path.exists('master_training_data.csv'):
        return {"error": "Data missing"}

    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    current_row = df.tail(1).copy()
    
    # 🆕 Dynamically get features from an existing model to ensure perfect match
    features = []
    sample_model_path = 'model_watts_ok_flow_6h.pkl'
    if os.path.exists(sample_model_path):
        model = joblib.load(sample_model_path)
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            # Fallback to hardcoded list from predict_all.py
            features = [
                'savoy_height', 'osage_creek_flow', 'hwy_59_height',
                'savoy_height_3h_ago', 'savoy_height_6h_ago', 'osage_creek_flow_3h_ago', 'osage_creek_flow_6h_ago',
                'savoy_height_12h_ago', 'savoy_height_24h_ago', 'osage_creek_flow_12h_ago', 'osage_creek_flow_24h_ago',
                'savoy_height_trend_6h', 'savoy_height_trend_24h', 'osage_creek_flow_trend_6h', 'osage_creek_flow_trend_24h',
                'precip_fayetteville', 'precip_springdale', 'precip_bentonville', 'precip_siloam',
                'precip_fayetteville_saturation', 
                'precip_fayetteville_24h', 'precip_fayetteville_48h', 'precip_fayetteville_168h',
                'seasonal_cycle', 'lake_headroom', 'hour_sin', 'hour_cos'
            ]
    
    # Modify precipitation features for simulation
    added_mm = added_inches * 25.4
    
    # We apply the rain to the main precip columns AND the saturation/window columns
    precip_cols = [c for c in features if 'precip' in c]
    for col in precip_cols:
        if col in current_row.columns:
            current_row[col] += added_mm

    results = {}
    gauges = ['hwy_16_flow', 'hwy_59_flow_est', 'lake_francis_height', 'watts_ok_flow']
    horizons = [6, 12, 24]

    for gauge in gauges:
        results[gauge] = {}
        for h in horizons:
            model_path = f'model_{gauge}_{h}h.pkl'
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Ensure we use the exact features expected by THIS model
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                else:
                    model_features = features
                    
                pred = model.predict(current_row[model_features])[0]
                results[gauge][f'projected_{h}h'] = float(pred)

    return results

if __name__ == "__main__":
    import xgboost as xgb # Ensure xgb is available for joblib.load
    if len(sys.argv) > 1:
        try:
            inches = float(sys.argv[1])
            print(json.dumps(simulate_rain(inches)))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        print(json.dumps({"error": "No rain amount provided"}))
