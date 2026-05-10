import requests
import pandas as pd
from datetime import datetime

# Coordinates for the Illinois River headwaters (near Fayetteville/Savoy)
LAT = 36.06
LON = -94.17

def get_weather_forecast():
    print(f"Fetching weather forecast for {LAT}, {LON}...")
    
    timeout_seconds = 10  # 🆕 Add timeout to prevent hanging
    header = {'User-Agent': 'IllinoisRiverPredictor/1.0 (ethan@example.com)'}
    
    try:
        # Step 1: Get the 'grid endpoint' for these coordinates
        point_url = f"https://api.weather.gov/points/{LAT},{LON}"
        print(f"   Fetching grid data from {point_url}...")
        
        res = requests.get(point_url, headers=header, timeout=timeout_seconds)
        res.raise_for_status()  # 🆕 Check for HTTP errors
        
        try:
            grid_url = res.json()['properties']['forecastGridData']
        except KeyError:
            print("❌ Error: Invalid response format from weather.gov points endpoint")
            return pd.DataFrame()
        
        # Step 2: Get the actual grid data
        print(f"   Fetching forecast grid data...")
        forecast_res = requests.get(grid_url, headers=header, timeout=timeout_seconds)
        forecast_res.raise_for_status()  # 🆕 Check for HTTP errors
        
        try:
            data = forecast_res.json()['properties']
        except KeyError:
            print("❌ Error: Invalid response format from forecast grid endpoint")
            return pd.DataFrame()
        
        # Step 3: Extract Quantitative Precipitation Forecast (QPF) - Amount of rain
        if 'quantitativePrecipitation' not in data:
            print("⚠️ Warning: quantitativePrecipitation not in forecast data")
            return pd.DataFrame()
        
        precip_data = data['quantitativePrecipitation']['values']
        
        if not precip_data:
            print("⚠️ Warning: No precipitation forecast data available")
            return pd.DataFrame()
        
        df = pd.DataFrame(precip_data)
        df['validTime'] = pd.to_datetime(df['validTime'].str.split('/').str[0])
        df.rename(columns={'value': 'precip_expected_mm', 'validTime': 'timestamp'}, inplace=True)
        
        print(f"✅ Success: Fetched {len(df)} forecast records")
        return df
    
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: Request took longer than {timeout_seconds}s")
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not reach weather.gov API")
        return pd.DataFrame()
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching weather forecast: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    weather_df = get_weather_forecast()
    if not weather_df.empty:
        weather_df.to_csv('weather_forecast.csv', index=False)
        print("✅ Forecast saved to weather_forecast.csv")
    else:
        print("⚠️ No forecast data to save")