import pandas as pd
import numpy as np

def merge_datasets():
    print("Loading data...")
    river_df = pd.read_csv('illinois_river_network.csv', index_col=0, parse_dates=True)
    weather_df = pd.read_csv('weather_forecast.csv', index_col='timestamp', parse_dates=True)
    
    # --- DIAGNOSTIC PRINT ---
    print("Columns found in river data:", river_df.columns.tolist())
    # ------------------------

    river_hourly = river_df.resample('1H').mean()
    weather_hourly = weather_df.resample('1H').sum()
    
    master_df = river_hourly.join(weather_hourly, how='left')
    master_df['precip_expected_mm'] = master_df['precip_expected_mm'].fillna(0)
    
    # Features
    master_df['soil_saturation_index'] = master_df['precip_expected_mm'].rolling(window=48, min_periods=1).sum()
    
    # SAFETY CHECK: Only calculate if the column exists
    if 'savoy_height' in master_df.columns:
        master_df['savoy_trend'] = master_df['savoy_height'].diff()
    
    master_df['whitewater_cfs_proxy'] = 0
    master_df['day_of_year'] = master_df.index.dayofyear
    master_df['seasonal_cycle'] = np.sin(2 * np.pi * master_df['day_of_year'] / 365)

    if 'hwy_59_height' in master_df.columns:
        master_df['is_overbank'] = (master_df['hwy_59_height'] > 10.0).astype(int)

    # FIXED LAKE LOGIC: Check if column exists first
    if 'lake_francis_height' in master_df.columns:
        master_df['lake_headroom'] = (1.5 - master_df['lake_francis_height']).clip(lower=0)
    else:
        print("Warning: 'lake_francis_height' not found. Creating empty column.")
        master_df['lake_headroom'] = 0

    if 'osage_creek_flow' in master_df.columns:
        master_df['tributary_proxy_pulse'] = master_df['osage_creek_flow'].diff().fillna(0)

    master_df.to_csv('master_training_data.csv')
    print("Success! Master dataset updated.")

if __name__ == "__main__":
    merge_datasets()