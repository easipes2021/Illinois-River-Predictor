import pandas as pd
import numpy as np
import json
import os


def apply_sskp_rating(H, meta):
    """
    Implements the piecewise rating relationship used for Hwy 59.
    """
    if pd.isna(H) or H <= 0:
        return 0.0

    bp = meta.get('piecewise_breakpoint', 2.5)
    low_q = meta['low_flow']['A'] * (np.power(H, meta['low_flow']['B']))

    if H <= bp:
        return low_q

    high_q = meta['high_flow']['A'] * (np.power(H, meta['high_flow']['B']))
    low_at_break = meta['low_flow']['A'] * (np.power(bp, meta['low_flow']['B']))
    high_at_break = meta['high_flow']['A'] * (np.power(bp, meta['high_flow']['B']))

    if high_at_break == 0:
        return high_q

    scale_factor = low_at_break / high_at_break
    return high_q * scale_factor


def merge_datasets():
    print("🔄 Starting Data Merge with Lagged Trends...")

    if not os.path.exists('illinois_river_network.csv') or not os.path.exists('weather_forecast.csv'):
        print("❌ Critical data files missing. Check fetch scripts.")
        return None

    river_df = pd.read_csv('illinois_river_network.csv', index_col=0, parse_dates=True)
    weather_df = pd.read_csv('weather_forecast.csv', index_col=0, parse_dates=True)

    precip_file = 'regional_precip_actual.csv'
    if os.path.exists(precip_file):
        regional_df = pd.read_csv(precip_file, index_col=0, parse_dates=True)
        print(f"✅ Found {precip_file}")
    else:
        regional_df = pd.DataFrame()
        print(f"⚠️ {precip_file} not found. Creating empty DataFrame.")

    for df in [river_df, weather_df, regional_df]:
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    river_hourly = river_df.resample('1h').mean()
    weather_hourly = weather_df.resample('1h').sum()
    master_df = river_hourly.join(weather_hourly, how='left')

    if not regional_df.empty:
        regional_hourly = regional_df.resample('1h').sum()
        master_df = master_df.join(regional_hourly, how='left')

    precip_cols = [c for c in master_df.columns if 'precip_' in c]
    master_df[precip_cols] = master_df[precip_cols].fillna(0)

    master_df['day_of_year'] = master_df.index.dayofyear
    master_df['seasonal_cycle'] = np.sin(2 * np.pi * master_df['day_of_year'] / 365)

    for col in precip_cols:
        master_df[f'{col}_saturation'] = master_df[col].rolling(window=72, min_periods=1).sum()

    upstream_cols = ['savoy_height', 'osage_creek_flow']
    for col in upstream_cols:
        if col in master_df.columns:
            master_df[f'{col}_3h_ago'] = master_df[col].shift(3)
            master_df[f'{col}_6h_ago'] = master_df[col].shift(6)

    if 'savoy_height' in master_df.columns:
        master_df['savoy_trend'] = master_df['savoy_height'].diff().fillna(0)

    if 'lake_francis_height' in master_df.columns:
        master_df['lake_headroom'] = (911.0 - master_df['lake_francis_height']).clip(lower=0)
    else:
        master_df['lake_headroom'] = 0

    metadata_path = 'rating_curve_metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = {
            'piecewise_breakpoint': 2.5,
            'low_flow': {'A': 38.5, 'B': 1.85},
            'high_flow': {'A': 42.0, 'B': 2.1}
        }

    if 'hwy_59_height' in master_df.columns:
        master_df['hwy_59_flow_est'] = master_df['hwy_59_height'].apply(lambda x: apply_sskp_rating(x, meta))

    master_df = master_df.dropna(subset=['hwy_59_height', 'watts_ok_height'], how='all')
    master_df.to_csv('master_training_data.csv')

    print(f"🚀 Success! Master dataset saved with {len(master_df)} rows.")
    return master_df


if __name__ == '__main__':
    merge_datasets()
