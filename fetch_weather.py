import requests
import pandas as pd
from datetime import datetime

# Coordinates for the Illinois River headwaters (near Fayetteville/Savoy)
LAT = 36.06
LON = -94.17

def get_weather_forecast():
    print(f"Fetching weather forecast for {LAT}, {LON}...")
    
    # Step 1: Get the 'grid endpoint' for these coordinates
    point_url = f"https://api.weather.gov/points/{LAT},{LON}"
    header = {'User-Agent': 'IllinoisRiverPredictor/1.0 (ethan@example.com)'}
    
    res = requests.get(point_url, headers=header)
    grid_url = res.json()['properties']['forecastGridData']
    
    # Step 2: Get the actual grid data
    forecast_res = requests.get(grid_url, headers=header)
    data = forecast_res.json()['properties']
    
    # Step 3: Extract Quantitative Precipitation Forecast (QPF) - Amount of rain
    precip_data = data['quantitativePrecipitation']['values']
    
    df = pd.DataFrame(precip_data)
    df['validTime'] = pd.to_datetime(df['validTime'].str.split('/').str[0])
    df.rename(columns={'value': 'precip_expected_mm', 'validTime': 'timestamp'}, inplace=True)
    
    return df

if __name__ == "__main__":
    weather_df = get_weather_forecast()
    weather_df.to_csv('weather_forecast.csv', index=False)
    print("Success! Forecast saved to weather_forecast.csv")