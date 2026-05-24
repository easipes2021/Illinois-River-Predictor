import sys
import os

# Import the existing pipeline functions
from fetch_all_gauges import fetch_river_network
from fetch_nwa_precip import fetch_regional_precip
from merge_data import merge_datasets
from predict_all import train_multi_models

def main():
    days_to_fetch = 730  # 2 years of history
    print(f"============================================================")
    print(f"🚀 BOOTSTRAPPING MODEL HISTORY: Fetching {days_to_fetch} days of data")
    print(f"============================================================")
    
    # 1. Fetch historical river data
    print("\n--- 1. Fetching Historical USGS Gauge Data ---")
    river_data = fetch_river_network(days=days_to_fetch)
    if river_data.empty:
        print("❌ Failed to fetch river history. Aborting.")
        sys.exit(1)
    river_data.to_csv('illinois_river_network.csv')
    
    # 2. Fetch historical precipitation data
    print("\n--- 2. Fetching Historical Open-Meteo Precipitation Data ---")
    # Our modified fetch_regional_precip now accepts a days argument and writes to regional_precip_actual.csv directly
    fetch_regional_precip(days=days_to_fetch)
    
    # 3. Merge data
    # (Note: we don't fetch_weather.py here because historical weather forecast is irrelevant;
    # merge_data.py will handle missing weather_forecast.csv gracefully by filling expected precip with 0, 
    # and relying purely on actual historical precip from regional_precip_actual.csv)
    print("\n--- 3. Merging Historical Datasets ---")
    
    # If weather_forecast.csv doesn't exist or is small, merge_data.py might complain.
    # We will ensure an empty dummy file exists if it doesn't.
    if not os.path.exists('weather_forecast.csv'):
        with open('weather_forecast.csv', 'w') as f:
            f.write("timestamp,precip_expected_mm\n")
            
    master_df = merge_datasets()
    if master_df is None or master_df.empty:
        print("❌ Failed to merge data. Aborting.")
        sys.exit(1)
        
    print(f"✅ Generated master_training_data.csv with {len(master_df)} hourly records.")
    
    # 4. Train Models
    print("\n--- 4. Training Models on Massive Historical Dataset ---")
    train_multi_models()
    
    print("\n🎉 BOOTSTRAPPING COMPLETE. The `.pkl` models are now incredibly smart.")
    
if __name__ == "__main__":
    main()
