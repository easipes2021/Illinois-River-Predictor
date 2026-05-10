import pandas as pd
import joblib
import os
from datetime import timedelta
import pytz
import json


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
        'lake_headroom',
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

    forecast_results = {
        'timestamp': local_time.strftime('%Y-%m-%d %I:%M %p'),
        'forecast_time': (local_time + timedelta(hours=6)).strftime('%Y-%m-%d %I:%M %p')
    }

    history_results = {
        'timestamp': local_time.strftime('%Y-%m-%d %I:%M %p'),
        'forecast_time': (local_time + timedelta(hours=6)).strftime('%Y-%m-%d %I:%M %p'),
        'gauges': {}
    }

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

        model_path = f'model_{key}.pkl'
        if not os.path.exists(model_path):
            print(f"   [!] Model for {label} (.pkl) not found.")
            forecast_results[key] = {'current': current_val, 'projected': None}
            continue

        try:
            model = joblib.load(model_path)
            print(f"DEBUG: Model input for {key}: {feature_row.values}")
            input_df = pd.DataFrame([feature_row.values], columns=features)
            pred = model.predict(input_df)[0]

            forecast_results[key] = {
                'current': None if current_val is None else round(current_val, 2),
                'projected': round(float(pred), 2)
            }
            print(f"✅ {label}: {current_val if current_val is not None else 'N/A'} -> {pred:.2f} {unit}")

        except Exception as e:
            print(f"   [!] Error predicting {label}: {e}")
            forecast_results[key] = {'current': current_val, 'projected': None}

    history_keys = {
        'hwy_16_flow': ('Hwy 16 (Siloam)', 'CFS'),
        'hwy_59_flow_est': ('Hwy 59 (AR Bridge)', 'EST. CFS'),
        'lake_francis_height': ('Lake Francis Level', 'ft (MSL)'),
        'watts_ok_flow': ('Watts Bridge (OK)', 'CFS')
    }

    for key, (label, unit) in history_keys.items():
        history_results['gauges'][key] = build_gauge_history(df, key, label, unit)

    with open('forecasts.json', 'w') as f:
        json.dump(forecast_results, f, indent=4)

    with open('history.json', 'w') as f:
        json.dump(history_results, f, indent=4)

    print("✅ Web Dashboard Updated with Nested Data.")
    print(f"Data status: {df.isnull().sum().sum()} missing values found.")


if __name__ == '__main__':
    generate_multi_forecast()

if __name__ == "__main__":
    generate_multi_forecast()