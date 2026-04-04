import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import pytz
import numpy as np

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

    for key, (label, unit) in locations.items():
        model_path = f'model_{key}.pkl'
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # Ensure we only pass the features the model expects
                pred = model.predict(current_row[features])[0]
                current_val = current_row[key].values[0] if key in current_row.columns else np.nan
                
                display_val = f"{current_val:.2f}" if not pd.isna(current_val) else "OFFLINE"
                
                print(f"{label}:")
                print(f"   Current:   {display_val:>8} {unit}")
                print(f"   Projected: {pred:>8.2f} {unit} (at {forecast_time.strftime('%I:%M %p')})")
                print("-" * 55)
                
                # Save for the website JSON
                forecast_results[key] = round(float(pred), 2)
            except Exception as e:
                print(f"   [!] Error predicting {label}: {e}")
        else:
            print(f"   [!] Model for {label} (.pkl) not found. Run predict_all.py.")

    # Save the JSON for your website to stop the "Loading" state
    import json
    with open('forecasts.json', 'w') as f:
        json.dump(forecast_results, f)

if __name__ == "__main__":
    generate_multi_forecast()