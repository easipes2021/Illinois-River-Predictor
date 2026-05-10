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
    timeout_seconds = 10  # 🆕 Add timeout to prevent hanging

    for name, coords in locations.items():
        print(f"Fetching rain data for {name}...")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords[0]}&longitude={coords[1]}&start_date={start_date}&end_date={end_date}&hourly=precipitation&timezone=auto"
        
        try:
            # 🆕 Add timeout parameter and error handling
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            if 'hourly' in data:
                df = pd.DataFrame({
                    'timestamp': data['hourly']['time'],
                    name: data['hourly']['precipitation']
                })
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                precip_data.append(df.set_index('timestamp'))
                print(f"   ✅ Success: {len(df)} hourly records for {name}")
            else:
                print(f"   ⚠️ Warning: No hourly data in response for {name}")
        
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout: Request took longer than {timeout_seconds}s for {name}")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection Error: Could not reach API for {name}")
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP Error: {e} for {name}")
        except Exception as e:
            print(f"   ❌ Error fetching {name}: {e}")

    if precip_data:
        # Merge all locations into one CSV
        final_df = pd.concat(precip_data, axis=1)
        final_df.to_csv('regional_precip_actual.csv')
        print(f"✅ Regional precipitation data saved ({len(final_df)} rows).")
    else:
        print("❌ No precipitation data was successfully fetched!")

if __name__ == "__main__":
    fetch_regional_precip()