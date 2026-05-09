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
    
    # --- CLEANING STEP: Handle missing data ---
    # 1. Fill small gaps (under 2 hours) with the last known value
    # df = df.ffill(limit=2) 
    
    # 2. If data is still missing (older than 2 hours), fill with 0 
    # to prevent NaN errors, but keep track of it
    # df = df.fillna(0)
    # ------------------------------------------

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

        # Initialize a dictionary to store the 'last known good value' if not already done
    if 'last_known_values' not in globals():
        last_known_values = {}

    for key, (label, unit) in locations.items():
        # ... (keep your model_path logic) ...
        
        raw_val = current_row[key].iloc[0]
        
        # Define your physical limits
        LIMITS = {
            'savoy_flow':(0,40000),
            'osage_creek_flow':(0,50000),
            'hwy_16_flow': (0, 50000),
            'lake_francis_height': (900, 930),
            'watts_ok_flow': (0, 40000)
        }
        
        # Get the value, handle NaNs
        current_val = float(raw_val) if not pd.isna(raw_val) else last_known_values.get(key, 0.0)
        
        # Apply Sanity Limits
        if key in LIMITS:
            min_v, max_v = LIMITS[key]
            if not (min_v <= current_val <= max_v):
                print(f"⚠️ {label} value ({current_val}) out of bounds! Using last known: {last_known_values.get(key, 0.0)}")
                current_val = last_known_values.get(key, 0.0)
        
        # Update the 'last known' record
        last_known_values[key] = current_val

        # Now proceed with your model.predict() using 'current_val'...
        
        # DEBUG: Check if the key exists in our DataFrame columns
        if key not in current_row.columns:
            print(f" ⚠️  CRITICAL: '{key}' not found in master_training_data.csv columns!")
            continue

        raw_val = current_row[key].iloc[0]
        
        # Check if the data is actually a number
        if pd.isna(raw_val):
            print(f" ⚠️  CRITICAL: '{key}' is NaN (Empty) in the latest CSV row!")
            current_val = 0.0
        else:
            current_val = float(raw_val)

        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"DEBUG: Model input for {key}: {current_row[features].values}")
                # Create a DataFrame with the correct names
                input_df = pd.DataFrame([current_row[features].values[0]], columns=features)

                # Now predict using the DataFrame
                pred = model.predict(input_df)[0]
                
                forecast_results[key] = {
                    "current": round(current_val, 2),
                    "projected": round(float(pred), 2)
                }
                
                print(f"✅ {label}: {current_val:.2f} -> {pred:.2f} {unit}")

            except Exception as e:
                print(f"   [!] Error predicting {label}: {e}")
        else:
            print(f"   [!] Model for {label} (.pkl) not found.")

    # 4. Save the nested JSON
    with open('forecasts.json', 'w') as f:
        json.dump(forecast_results, f, indent=4)
    
    print("✅ Web Dashboard Updated with Nested Data.")
    print(f"Data status: {df.isnull().sum().sum()} missing values found.")

    # Force-feed the model a "Flood" scenario
    flood_scenario = current_row.copy()
    flood_scenario['precip_fayetteville'] = 5.0  # 5 inches of rain
    flood_scenario['savoy_height'] = 10.0         # 10ft river height

    # Use the same DataFrame-wrap fix we used before
    flood_df = pd.DataFrame([flood_scenario[features].values[0]], columns=features)
    flood_pred = model.predict(flood_df)[0]

    print(f"DEBUG: Normal Forecast: {pred}")
    print(f"DEBUG: Flood Forecast: {flood_pred}")

if __name__ == "__main__":
    generate_multi_forecast()