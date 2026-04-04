import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import pytz
import numpy as np
import json

def generate_multi_forecast():
    if not os.path.exists('master_training_data.csv'):
        print("❌ Error: master_training_data.csv missing.")
        return

    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    current_row = df.tail(1).copy()
    
    # Timezone Handling
    utc_time = pd.to_datetime(current_row.index[0], utc=True)
    local_tz = pytz.timezone('US/Central')
    local_time = utc_time.astimezone(local_tz)
    forecast_time = local_time + timedelta(hours=6)
    
    # MUST MATCH predict_all.py FEATURES EXACTLY
    features = [
        # --- Current Levels ---
        'savoy_height', 
        'osage_creek_flow', 
        'hwy_59_height',
        
        # --- Lagged Features (The "Lookback" for Upstream Rise) ---
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

    print(f"\n{'='*55}")
    print(f"   ILLINOIS RIVER SYSTEM REPORT")
    print(f"   Local Time: {local_time.strftime('%I:%M %p')}")
    print(f"{'='*55}")

    # Locations map - Matching your new CFS preference
    locations = {
        'hwy_16_flow': ('Hwy 16 (Siloam)', 'CFS'),
        'hwy_59_flow_est': ('Hwy 59 (AR Bridge)', 'EST. CFS'),
        'lake_francis_height': ('Lake Francis Level', 'ft (MSL)'),
        'watts_ok_flow': ('Watts Bridge (OK)', 'CFS') # Changed to Flow/CFS
    }

    forecast_results = {}

    forecast_results = {
        "timestamp": local_time.strftime('%Y-%m-%d %I:%M %p')
    }

    for key, (label, unit) in locations.items():
        model_path = f'model_{key}.pkl'
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                
                # 1. Get the Prediction from the AI
                pred = model.predict(current_row[features])[0]
                
                # 2. Get the Current Value from the CSV row
                # We use .item() to ensure it's a standard Python float, not a numpy object
                current_val = current_row[key].iloc[0] if key in current_row.columns else 0.0
                
                # 3. STRUCTURE THE DATA FOR THE WEBSITE (This is the fix!)
                forecast_results[key] = {
                    "current": round(float(current_val), 2),
                    "projected": round(float(pred), 2)
                }
                
                # Console Logging
                print(f"{label}: {current_val:.2f} -> {pred:.2f} {unit}")

            except Exception as e:
                print(f"   [!] Error predicting {label}: {e}")
        else:
            print(f"   [!] Model for {label} (.pkl) not found.")

    # 4. Save the nested JSON
    with open('forecasts.json', 'w') as f:
        json.dump(forecast_results, f, indent=4)
    
    print("✅ Web Dashboard Updated with Nested Data.")

if __name__ == "__main__":
    generate_multi_forecast()