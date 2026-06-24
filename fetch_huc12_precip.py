import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

def fetch_huc12_precip(days_history=730):
    with open('huc12_centroids.json', 'r') as f:
        centroids = json.load(f)

    # Archive API start and end dates (up to 5 days ago to avoid API gaps)
    archive_end = (datetime.utcnow() - timedelta(days=5)).strftime('%Y-%m-%d')
    archive_start = (datetime.utcnow() - timedelta(days=days_history)).strftime('%Y-%m-%d')
    
    all_data = []
    frontend_data = {}
    
    print(f"Fetching precipitation for {len(centroids)} sub-watersheds (Past {days_history} days history + live forecast)...")
    
    for huc12, info in centroids.items():
        name = info['name']
        lat, lon = info['lat'], info['lon']
        
        # 1. Fetch from Archive API (historical backfill in hourly resolution)
        archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={archive_start}&end_date={archive_end}&hourly=precipitation&timezone=UTC"
        
        # 2. Fetch from Forecast API (recent past + live forecast in 15-minute resolution)
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=7&minutely_15=precipitation&timezone=UTC"
        
        df_archive = pd.DataFrame()
        df_forecast = pd.DataFrame()
        
        try:
            # Get Archive Data
            resp = requests.get(archive_url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if 'hourly' in data:
                    df_raw_archive = pd.DataFrame({
                        'timestamp': data['hourly']['time'],
                        f'precip_{huc12}': data['hourly']['precipitation']
                    })
                    df_raw_archive['timestamp'] = pd.to_datetime(df_raw_archive['timestamp'], utc=True)
                    df_raw_archive = df_raw_archive.set_index('timestamp')
                    
                    # Upsample from hourly to 15-minute by repeating and dividing by 4 to preserve total volume
                    df_archive = df_raw_archive.resample('15min').ffill() / 4
            else:
                print(f"⚠️ Archive status {resp.status_code} for {name}")
        except Exception as e:
            print(f"Error fetching archive for {name}: {e}")
            
        try:
            # Get Forecast Data (with past_days=7 to overlap archive and fetch latest real-time)
            resp = requests.get(forecast_url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if 'minutely_15' in data:
                    df_forecast = pd.DataFrame({
                        'timestamp': data['minutely_15']['time'],
                        f'precip_{huc12}': data['minutely_15']['precipitation']
                    })
                    df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'], utc=True)
                    df_forecast = df_forecast.set_index('timestamp')
            else:
                print(f"⚠️ Forecast status {resp.status_code} for {name}")
        except Exception as e:
            print(f"Error fetching forecast for {name}: {e}")

        # Combine datasets using combine_first (retains df_forecast for any overlaps)
        if not df_forecast.empty and not df_archive.empty:
            df_combined = df_forecast.combine_first(df_archive)
        elif not df_forecast.empty:
            df_combined = df_forecast
        elif not df_archive.empty:
            df_combined = df_archive
        else:
            print(f"❌ No precipitation data fetched for centroid {name}")
            continue

        all_data.append(df_combined)
        
        # For frontend: get last 1hr and 24hr sums from the combined dataframe
        # Data is 15-minute, so 1hr = tail(4), 24hr = tail(96)
        last_24h = df_combined.tail(96).sum().iloc[0] if len(df_combined) >= 96 else df_combined.sum().iloc[0]
        last_1h = df_combined.tail(4).sum().iloc[0] if len(df_combined) >= 4 else df_combined.sum().iloc[0]
        
        # Convert mm to inches
        frontend_data[huc12] = {
            'name': name,
            'qpe_1hr': round(last_1h / 25.4, 3),
            'qpe_24hr': round(last_24h / 25.4, 3),
            'qpf_24hr': 0.0 # Placeholder for forecast
        }

    # Now get the forecast (QPF) from the live API for the frontend
    forecast_end = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
    for huc12, info in centroids.items():
        lat, lon = info['lat'], info['lon']
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&minutely_15=precipitation&timezone=UTC&start_date=2026-06-24&end_date={forecast_end}"
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
        print(f"✅ Saved huc12_training_data_beta.csv ({len(merged_df)} records backfilled to {days_history} days)")

if __name__ == "__main__":
    fetch_huc12_precip()
