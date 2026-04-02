import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_regional_precip():
    # Key locations in the watershed (Lat, Lon)
    locations = {
        'precip_fayetteville': (36.06, -94.17),
        'precip_springdale': (36.18, -94.12),
        'precip_bentonville': (36.37, -94.20),
        'precip_siloam': (36.18, -94.54)
    }
    
    # Time window for the pull
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    precip_data = []

    for name, coords in locations.items():
        print(f"Fetching rain data for {name}...")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords[0]}&longitude={coords[1]}&start_date={start_date}&end_date={end_date}&hourly=precipitation&timezone=auto"
        
        response = requests.get(url).json()
        if 'hourly' in response:
            df = pd.DataFrame({
                'timestamp': response['hourly']['time'],
                name: response['hourly']['precipitation']
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            precip_data.append(df.set_index('timestamp'))

    # Merge all locations into one CSV
    final_df = pd.concat(precip_data, axis=1)
    final_df.to_csv('regional_precip_actual.csv')
    print("✅ Regional precipitation data saved.")

if __name__ == "__main__":
    fetch_regional_precip()