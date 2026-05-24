import pandas as pd
import joblib
import os
import xgboost as xgb
from datetime import timedelta
import pytz
import json
from accuracy_tracker import run_full_accuracy_pipeline


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
    if not os.path.exists('master_training_data.csv'):
        print("❌ Error: master_training_data.csv missing.")
        return

    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    if df.empty:
        print("❌ Error: master_training_data.csv is empty.")
        return

    # Fill missing values with previous readings to ensure current readings show up
    df = df.ffill()
    
    current_row = df.tail(1).copy()
    utc_time = pd.to_datetime(current_row.index[0], utc=True)
    local_tz = pytz.timezone('US/Central')
    local_time = utc_time.astimezone(local_tz)

    features = [
        'savoy_height',
        'osage_creek_flow',
        'hwy_59_height',
        'savoy_height_3h_ago',
        'savoy_height_6h_ago',
        'osage_creek_flow_3h_ago',
        'osage_creek_flow_6h_ago',
        # 🆕 PHASE 1: Extended lags
        'savoy_height_12h_ago',
        'savoy_height_24h_ago',
        'osage_creek_flow_12h_ago',
        'osage_creek_flow_24h_ago',
        # 🆕 PHASE 1: Trend indicators
        'savoy_height_trend_6h',
        'savoy_height_trend_24h',
        'osage_creek_flow_trend_6h',
        'osage_creek_flow_trend_24h',
        # Rainfall
        'precip_fayetteville',
        'precip_springdale',
        'precip_bentonville',
        'precip_siloam',
        'precip_fayetteville_saturation',
        # 🆕 PHASE 1: Multiple precip windows
        'precip_fayetteville_24h',
        'precip_fayetteville_48h',
        'precip_fayetteville_168h',
        # Seasonal & storage
        'seasonal_cycle',
        # 🆕 PHASE 1: Hour-of-day features
        'hour_sin',
        'hour_cos'
    ]

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

    # 🆕 Generate Historical Comparison (Last Week)
    last_week_ts = utc_time - timedelta(days=7)
    # Convert to naive for comparison with index
    last_week_naive = last_week_ts.replace(tzinfo=None)
    
    # Get 24 hours of data starting from 7 days ago
    comp_range = df[df.index >= last_week_naive].head(24)
    if not comp_range.empty:
        for key in ['hwy_16_flow', 'hwy_59_flow_est', 'watts_ok_flow']:
            if key in comp_range.columns:
                history_results['comparison'][key] = [safe_float(x, None) for x in comp_range[key]]

    last_known_values = {}
    limits = {
        'hwy_16_flow': (0, 50000),
        'hwy_59_flow_est': (0, 50000),
        'lake_francis_height': (900, 930),
        'watts_ok_flow': (0, 40000)
    }

    feature_row = build_feature_row(current_row.reindex(columns=features).iloc[0], features)

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
            model_path = f'model_{key}_{h}h.pkl'
            if not os.path.exists(model_path):
                print(f"   [!] Model for {label} ({h}h) not found.")
                forecast_results[key][f'projected_{h}h'] = None
                continue

            try:
                model = joblib.load(model_path)
                input_df = pd.DataFrame([feature_row.values], columns=features)
                pred = model.predict(input_df)[0]

                forecast_results[key][f'projected_{h}h'] = round(float(pred), 2)
                print(f"✅ {label} ({h}h): {current_val if current_val is not None else 'N/A'} -> {pred:.2f} {unit}")

            except Exception as e:
                print(f"   [!] Error predicting {label} {h}h: {e}")
                forecast_results[key][f'projected_{h}h'] = None

    history_keys = {
        'hwy_16_flow': ('Hwy 16 (Siloam)', 'CFS'),
        'hwy_59_flow_est': ('Hwy 59 (AR Bridge)', 'EST. CFS'),
        'lake_francis_height': ('Lake Francis Level', 'ft (MSL)'),
        'watts_ok_flow': ('Watts Bridge (OK)', 'CFS')
    }

    for key, (label, unit) in history_keys.items():
        history_results['gauges'][key] = build_gauge_history(df, key, label, unit)

    # 🆕 Generate Weather and Saturation Data
    weather_results = {
        'timestamp': local_time.strftime('%Y-%m-%d %I:%M %p'),
        'saturation': {
            'fayetteville': safe_float(current_row['precip_fayetteville_saturation'].iloc[0]) if 'precip_fayetteville_saturation' in current_row.columns else 0,
            'recent_24h': safe_float(current_row['precip_fayetteville_24h'].iloc[0]) if 'precip_fayetteville_24h' in current_row.columns else 0
        },
        'forecast': []
    }

    if os.path.exists('weather_forecast.csv'):
        weather_df = pd.read_csv('weather_forecast.csv')
        # Get rows where timestamp is >= current time, just take the next 5 records
        if not weather_df.empty and 'precip_expected_mm' in weather_df.columns:
            # Drop na and filter to values > 0 to show only actual rain
            future_rain = weather_df[weather_df['precip_expected_mm'] > 0].head(5)
            for _, r in future_rain.iterrows():
                weather_results['forecast'].append({
                    'timestamp': str(r.get('timestamp', '')),
                    'precip_mm': safe_float(r['precip_expected_mm'])
                })

    # 🆕 Generate Event-Mode Simulations (0" to 3" in 0.5" steps)
    try:
        from simulate_event import simulate_rain
        sim_results = {}
        for amt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            key = f"{amt}in"
            sim_results[key] = simulate_rain(amt)
            
        with open('simulations.json', 'w') as f:
            json.dump(sim_results, f, indent=4)
        print("✅ Rain Simulations Generated (0.5\" to 3.0\").")
    except Exception as e:
        print(f"⚠️ Simulation generation failed: {e}")

    with open('forecasts.json', 'w') as f:
        json.dump(forecast_results, f, indent=4)

    with open('history.json', 'w') as f:
        json.dump(history_results, f, indent=4)
        
    with open('weather.json', 'w') as f:
        json.dump(weather_results, f, indent=4)

    print("✅ Dashboard Data Packaged (including Historical Comparison).")
    print(f"Data status: {df.isnull().sum().sum()} missing values found.")

    # 🆕 Run accuracy tracking pipeline
    try:
        run_full_accuracy_pipeline(forecast_results)
    except Exception as e:
        print(f"⚠️ Accuracy tracking failed (non-fatal): {e}")


if __name__ == '__main__':
    generate_multi_forecast()