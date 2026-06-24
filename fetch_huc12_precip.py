import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

def fetch_huc12_precip(days_history=30):
    with open('huc12_centroids.json', 'r') as f:
        centroids = json.load(f)

    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=days_history)).strftime('%Y-%m-%d')
    
    # We will build a unified dataframe for the CSV, and a dictionary for the frontend JSON
    all_data = []
    frontend_data = {}
    
    print(f"Fetching precipitation for {len(centroids)} sub-watersheds (Past {days_history} days)...")
    
    for huc12, info in centroids.items():
        name = info['name']
        lat, lon = info['lat'], info['lon']
        
        # Open-Meteo Forecast API with past_days for real-time up-to-date 15-minute resolution
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days={days_history}&minutely_15=precipitation&timezone=UTC"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if 'minutely_15' in data:
                    df = pd.DataFrame({
                        'timestamp': data['minutely_15']['time'],
                        f'precip_{huc12}': data['minutely_15']['precipitation']
                    })
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.set_index('timestamp')
                    all_data.append(df)
                    
                    # For frontend: get last 1hr and 24hr sums
                    # Data is 15-minute, so 1hr = tail(4), 24hr = tail(96)
                    last_24h = df.tail(96).sum().iloc[0]
                    last_1h = df.tail(4).sum().iloc[0]
                    
                    # Convert mm to inches
                    frontend_data[huc12] = {
                        'name': name,
                        'qpe_1hr': round(last_1h / 25.4, 3),
                        'qpe_24hr': round(last_24h / 25.4, 3),
                        'qpf_24hr': 0.0 # Placeholder for forecast
                    }
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    # Now get the forecast (QPF) from the live API
    forecast_end = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
    for huc12, info in centroids.items():
        lat, lon = info['lat'], info['lon']
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&minutely_15=precipitation&timezone=UTC&start_date={end_date}&end_date={forecast_end}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if 'minutely_15' in data:
                    df = pd.DataFrame({
                        'timestamp': data['minutely_15']['time'],
                        'precip': data['minutely_15']['precipitation']
                    })
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    # Get next 24 hours from now
                    now = pd.Timestamp.utcnow()
                    next_24 = df[(df['timestamp'] >= now) & (df['timestamp'] <= now + timedelta(hours=24))]
                    qpf_24_mm = next_24['precip'].sum()
                    
                    if huc12 in frontend_data:
                        frontend_data[huc12]['qpf_24hr'] = round(qpf_24_mm / 25.4, 2)
        except Exception as e:
            print(f"Error fetching forecast for {huc12}: {e}")

    # Output frontend JSON
    with open('watershed_rainfall.json', 'w') as f:
        json.dump(frontend_data, f, indent=4)
    print("✅ Saved watershed_rainfall.json")

    # Output backend CSV for training
    if all_data:
        merged_df = pd.concat(all_data, axis=1)
        merged_df.to_csv('huc12_training_data_beta.csv')
        print(f"✅ Saved huc12_training_data_beta.csv ({len(merged_df)} hourly records backfilled)")

if __name__ == "__main__":
    fetch_huc12_precip()
