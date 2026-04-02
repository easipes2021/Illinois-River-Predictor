import pandas as pd

def merge_datasets():
    print("Merging river and weather data with soil saturation logic...")
    
    # 1. Load Data
    river_df = pd.read_csv('illinois_river_network.csv', index_col=0, parse_dates=True)
    weather_df = pd.read_csv('weather_forecast.csv', index_col='timestamp', parse_dates=True)
    
    # 2. Resample to 1-hour intervals (Average for levels, Sum for rain)
    river_hourly = river_df.resample('1H').mean()
    weather_hourly = weather_df.resample('1H').sum() # Summing rain per hour
    
    # 3. Combine
    master_df = river_hourly.join(weather_hourly, how='left')
    master_df['precip_expected_mm'] = master_df['precip_expected_mm'].fillna(0)
    
    # 4. ADD SOIL SATURATION INDEX (48-hour rolling rainfall)
    # This looks back at the last 48 rows (hours) and sums the rain
    master_df['soil_saturation_index'] = master_df['precip_expected_mm'].rolling(window=48, min_periods=1).sum()
    
    # 5. ADD A "TREND" COLUMN
    # Is the river currently rising or falling at Savoy?
    master_df['savoy_trend'] = master_df['savoy_height'].diff()
    
    # 6. Save
    master_df.to_csv('master_training_data.csv')
    print("Success! Master dataset updated with Soil Saturation Index.")
    print(master_df[['precip_expected_mm', 'soil_saturation_index']].tail(10))

if __name__ == "__main__":
    merge_datasets()