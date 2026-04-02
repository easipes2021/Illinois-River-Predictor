import pandas as pd
import numpy as np
import json
import os

def apply_sskp_rating(H, meta):
    """
    Implements the Piecewise Power Law with Continuity Adjustment.
    H: Gage Height (ft)
    meta: Dictionary containing A, B coefficients and breakpoint
    """
    if pd.isna(H) or H <= 0:
        return 0
    
    bp = meta['piecewise_breakpoint']
    
    # Low Branch: Q = A * H^B
    low_q = meta['low_flow']['A'] * (np.power(H, meta['low_flow']['B']))
    
    if H <= bp:
        return low_q
    
    # High Branch: Q = A * H^B
    high_q = meta['high_flow']['A'] * (np.power(H, meta['high_flow']['B']))
    
    # Continuity Adjustment (Ensures no 'jump' at the breakpoint)
    low_at_break = meta['low_flow']['A'] * (np.power(bp, meta['low_flow']['B']))
    high_at_break = meta['high_flow']['A'] * (np.power(bp, meta['high_flow']['B']))
    
    if high_at_break == 0: return high_q
    
    scale_factor = low_at_break / high_at_break
    return high_q * scale_factor

def merge_datasets():
    print("🔄 Starting Data Merge with Regional Precipitation...")
    
    # Load Main Data
    river_df = pd.read_csv('illinois_river_network.csv', index_col=0, parse_dates=True)
    weather_df = pd.read_csv('weather_forecast.csv', index_col='timestamp', parse_dates=True)
    
    # NEW: Load Regional Actuals
    if os.path.exists('regional_precip_actual.csv'):
        regional_df = pd.read_csv('regional_precip_actual.csv', index_col=0, parse_dates=True)
    else:
        regional_df = pd.DataFrame()

    # Resample and Join
    master_df = river_df.resample('1h').mean().join(weather_df.resample('1h').sum(), how='left')
    
    if not regional_df.empty:
        master_df = master_df.join(regional_df.resample('1h').sum(), how='left')

    # Fill all precip columns with 0
    precip_cols = [c for c in master_df.columns if 'precip_' in c]
    master_df[precip_cols] = master_df[precip_cols].fillna(0)

    # ... (Keep your Rating Curve and Feature logic here) ...
    # Add a unique Soil Saturation index for EACH station if you want to be extra precise:
    for col in precip_cols:
        master_df[f'{col}_saturation'] = master_df[col].rolling(window=72, min_periods=1).sum()

    master_df.to_csv('master_training_data.csv')
    print(f"🚀 Success! Master dataset updated with {len(precip_cols)} rain stations.")

    # 2. Standardize Time & Resample
    river_df.index = pd.to_datetime(river_df.index)
    weather_df.index = pd.to_datetime(weather_df.index)
    
    river_hourly = river_df.resample('1h').mean()
    weather_hourly = weather_df.resample('1h').sum()
    
    # 3. Join into Master DataFrame
    master_df = river_hourly.join(weather_hourly, how='left')
    
    # 4. FIX: Add Seasonal Cycle (Required by predict_all.py)
    master_df['day_of_year'] = master_df.index.dayofyear
    master_df['seasonal_cycle'] = np.sin(2 * np.pi * master_df['day_of_year'] / 365)
    
    # 5. Weather & Soil Saturation
    master_df['precip_expected_mm'] = master_df.get('precip_expected_mm', 0).fillna(0)
    master_df['soil_saturation_index'] = master_df['precip_expected_mm'].rolling(window=48, min_periods=1).sum()
    
    # 6. River Trends
    if 'savoy_height' in master_df.columns:
        master_df['savoy_trend'] = master_df['savoy_height'].diff().fillna(0)
    else:
        master_df['savoy_trend'] = 0

    # 7. Apply SSKP Rating Curve for Hwy 59
    if os.path.exists('rating_curve_metadata.json'):
        with open('rating_curve_metadata.json', 'r') as f:
            meta = json.load(f)
        print("✅ Using custom SSKP Rating Curve metadata.")
    else:
        print("⚠️ Metadata missing. Using calibrated defaults for Hwy 59.")
        meta = {
            "piecewise_breakpoint": 2.5,
            "low_flow": {"A": 38.5, "B": 1.85},
            "high_flow": {"A": 42.0, "B": 2.1}
        }

    if 'hwy_59_height' in master_df.columns:
        master_df['hwy_59_flow_est'] = master_df['hwy_59_height'].apply(lambda x: apply_sskp_rating(x, meta))
        
        # Diagnostic Print
        valid_heights = master_df['hwy_59_height'].dropna()
        if not valid_heights.empty:
            last_h = valid_heights.iloc[-1]
            last_f = master_df['hwy_59_flow_est'].loc[valid_heights.index[-1]]
            print(f"📍 RATING CHECK: {last_h:.2f} ft -> {last_f:.2f} CFS")
    
    # 8. FIX: Add Lake Headroom (Required by predict_all.py)
    # Using 911.0 ft as the standard spillway elevation
    if 'lake_francis_height' in master_df.columns:
        master_df['lake_headroom'] = (911.0 - master_df['lake_francis_height']).clip(lower=0)
    else:
        print("⚠️ 'lake_francis_height' missing. Headroom set to 0.")
        master_df['lake_headroom'] = 0

    # Load the new regional data
    regional_precip = pd.read_csv('regional_precip_actual.csv', index_col=0, parse_dates=True)
    
    # Join it to the master dataframe
    master_df = master_df.join(regional_precip.resample('1h').sum(), how='left')
    
    # Fill missing values with 0
    precip_cols = [c for c in master_df.columns if 'precip_' in c]
    master_df[precip_cols] = master_df[precip_cols].fillna(0)

    # 9. Final Cleanup & Save
    # Ensure we don't have empty trailing rows from the join
    master_df = master_df.dropna(subset=['hwy_59_height', 'watts_ok_height'], how='all')
    
    master_df.to_csv('master_training_data.csv')
    print(f"🚀 Success! Master dataset saved with {len(master_df)} records.")

if __name__ == "__main__":
    merge_datasets()