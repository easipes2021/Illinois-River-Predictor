import pandas as pd
import joblib
import os
import xgboost as xgb
from datetime import timedelta
import pytz
import json
from beta_accuracy_tracker import run_full_beta_accuracy_pipeline


def safe_float(value, fallback=None):
    if pd.isna(value):
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


def build_feature_row(raw_row, features):
    row = raw_row.reindex(features)
    row = row.astype(float, errors='ignore')
    return row.ffill().fillna(0)


def build_gauge_history(df, key, name, unit, window=24):
    if key not in df.columns:
        return {'name': name, 'unit': unit, 'history': []}

    recent = df[[key]].tail(window).copy()
    recent.index = pd.to_datetime(recent.index, utc=True)
    index_name = recent.index.name or 'timestamp'
    recent = recent.reset_index()

    return {
        'name': name,
        'unit': unit,
        'history': [
            {
                'timestamp': row[index_name].strftime('%Y-%m-%dT%H:%M:%SZ'),
                'value': safe_float(row[key], None)
            }
            for _, row in recent.iterrows()
            if not pd.isna(row[key])
        ]
    }


def generate_multi_forecast():
    if not os.path.exists('master_training_data_beta.csv'):
        print("❌ Error: master_training_data_beta.csv missing.")
        return

    df = pd.read_csv('master_training_data_beta.csv', index_col=0, parse_dates=True)
    if df.empty:
        print("❌ Error: master_training_data_beta.csv is empty.")
        return

    # Fill missing values with previous readings to ensure current readings show up
    df = df.ffill()
    
    current_row = df.tail(1).copy()
    utc_time = pd.to_datetime(current_row.index[0], utc=True)
    local_tz = pytz.timezone('US/Central')
    local_time = utc_time.astimezone(local_tz)


    locations = {
        'hwy_16_flow': ('Hwy 16 (Siloam)', 'CFS'),
        'hwy_59_flow_est': ('Hwy 59 (AR Bridge)', 'EST. CFS'),
        'lake_francis_height': ('Lake Francis Level', 'ft (MSL)'),
        'watts_ok_flow': ('Watts Bridge (OK)', 'CFS')
    }

    # 🆕 Use ISO format for robust front-end parsing
    timestamp_iso = utc_time.isoformat()
    
    forecast_results = {
        'timestamp': timestamp_iso,
        'forecast_time_6h': (utc_time + timedelta(hours=6)).isoformat(),
        'forecast_time_12h': (utc_time + timedelta(hours=12)).isoformat(),
        'forecast_time_24h': (utc_time + timedelta(hours=24)).isoformat(),
    }
    
    history_results = {
        'timestamp': timestamp_iso,
        'gauges': {},
        'comparison': {} # 🆕 For historical overlay
    }



    last_known_values = {}
    limits = {
        'hwy_16_flow': (0, 50000),
        'hwy_59_flow_est': (0, 50000),
        'lake_francis_height': (900, 930),
        'watts_ok_flow': (0, 40000)
    }

    for key, (label, unit) in locations.items():
        if key not in current_row.columns:
            print(f"⚠️  CRITICAL: '{key}' not found in master_training_data.csv columns!")
            forecast_results[key] = {'current': None, 'projected': None}
            continue

        raw_val = current_row[key].iloc[0]
        current_val = safe_float(raw_val, last_known_values.get(key))

        if current_val is None:
            print(f"⚠️  Missing latest reading for {label}; projection will still be generated.")
        elif key in limits:
            min_v, max_v = limits[key]
            if not (min_v <= current_val <= max_v):
                print(f"⚠️  {label} value ({current_val}) out of bounds! Keeping last known value.")
                current_val = last_known_values.get(key, current_val)

        if current_val is not None:
            last_known_values[key] = current_val

        forecast_results[key] = {'current': None if current_val is None else round(current_val, 2)}
        
        if key == 'lake_francis_height':
            for h in [6, 12, 24]:
                forecast_results[key][f'projected_{h}h'] = None
            continue
            
        
        
        for h in [6, 12, 24]:
            model_path = f'model_beta_{key}_{h}h.pkl'
            if not os.path.exists(model_path):
                print(f"   [!] Model for {label} ({h}h) not found.")
                forecast_results[key][f'projected_{h}h'] = None
                continue

            try:
                model = joblib.load(model_path)
                features = model.get_booster().feature_names
                
                for f in features:
                    if f not in current_row.columns:
                        current_row[f] = 0
                
                input_df = current_row[features].copy()
                input_df = input_df.fillna(0) # Build robust row
                
                pred = model.predict(input_df)[0]
                pred = max(0.0, float(pred)) # Negative flow clipping
                forecast_results[key][f'projected_{h}h'] = round(pred, 2)
                print(f"✅ {label} ({h}h): {current_val if current_val is not None else 'N/A'} -> {pred:.2f} {unit}")

            except Exception as e:
                print(f"   [!] Error predicting {label} {h}h: {e}")
                forecast_results[key][f'projected_{h}h'] = None

    # Just save forecasts_beta.json, don't overwrite history or weather
    with open('forecasts_beta.json', 'w') as f:
        json.dump(forecast_results, f, indent=4)

    print("✅ Beta Forecast Data Packaged.")
    
    # Run the beta accuracy tracker
    try:
        run_full_beta_accuracy_pipeline(forecast_results)
    except Exception as e:
        print(f"⚠️ Accuracy tracking failed (non-fatal): {e}")


if __name__ == '__main__':
    generate_multi_forecast()