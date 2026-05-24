import pandas as pd
import numpy as np
import json
import os


def apply_hwy59_interpolation(H, paired_data):
    """
    Uses historical paired data for Hwy 59 to interpolate flow from height.
    This is much more accurate than a simple piecewise power law.
    """
    if pd.isna(H) or H <= 0:
        return 0.0
    
    # Linear interpolation using historical height/flow pairs
    return np.interp(H, paired_data['value_H'], paired_data['value_Q'])


def apply_sskp_rating(H, meta):
    """
    Implements the piecewise rating relationship used for Hwy 59. (Fallback)
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

    # Load historical paired data for Hwy 59 flow estimation
    paired_path = 'paired_stage_flow.csv'
    paired_data = None
    if os.path.exists(paired_path):
        pdf = pd.read_csv(paired_path)
        # Average Q for each unique H and sort for interpolation
        paired_data = pdf.groupby('value_H')['value_Q'].mean().reset_index().sort_values('value_H')
        print(f"✅ Loaded {len(paired_data)} historical rating points for Hwy 59.")

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

    # 🆕 PHASE 1: Hour-of-day feature (sin/cos cyclic encoding)
    master_df['hour_of_day'] = master_df.index.hour
    master_df['hour_sin'] = np.sin(2 * np.pi * master_df['hour_of_day'] / 24)
    master_df['hour_cos'] = np.cos(2 * np.pi * master_df['hour_of_day'] / 24)

    for col in precip_cols:
        master_df[f'{col}_saturation'] = master_df[col].rolling(window=72, min_periods=1).sum()
        # PHASE 1: Multiple precipitation windows (24h, 48h, 168h)
        master_df[f'{col}_24h'] = master_df[col].rolling(window=24, min_periods=1).sum()
        master_df[f'{col}_48h'] = master_df[col].rolling(window=48, min_periods=1).sum()
        master_df[f'{col}_168h'] = master_df[col].rolling(window=168, min_periods=1).sum()
        # 🆕 PHASE 2 (E): 30-day soil moisture index
        master_df[f'{col}_720h'] = master_df[col].rolling(window=720, min_periods=1).sum()

    upstream_cols = ['savoy_height', 'osage_creek_flow']
    for col in upstream_cols:
        if col in master_df.columns:
            master_df[f'{col}_3h_ago'] = master_df[col].shift(3)
            master_df[f'{col}_6h_ago'] = master_df[col].shift(6)
            # PHASE 1: Extended lag features (12h, 24h)
            master_df[f'{col}_12h_ago'] = master_df[col].shift(12)
            master_df[f'{col}_24h_ago'] = master_df[col].shift(24)
            # PHASE 1: Trend indicators
            master_df[f'{col}_trend_6h'] = master_df[col].diff(6).fillna(0)
            master_df[f'{col}_trend_24h'] = master_df[col].diff(24).fillna(0)

    if 'savoy_height' in master_df.columns:
        master_df['savoy_trend'] = master_df['savoy_height'].diff().fillna(0)

    # 🆕 PHASE 2 (B): Cascade lag features — Hwy 16 lags for Hwy 59 model
    for col in ['hwy_16_flow', 'hwy_16_height']:
        if col in master_df.columns:
            master_df[f'{col}_3h_ago'] = master_df[col].shift(3)
            master_df[f'{col}_6h_ago'] = master_df[col].shift(6)
            master_df[f'{col}_12h_ago'] = master_df[col].shift(12)
            master_df[f'{col}_trend_6h'] = master_df[col].diff(6).fillna(0)

    # 🆕 PHASE 2 (A): NWS QPF forward precipitation windows
    # The weather_forecast.csv uses 'precip_expected_mm' indexed by forecast time.
    # We construct lead-time sum features: how much rain is forecast to fall
    # in the next 6h, 12h, and 24h from each row's timestamp.
    if 'precip_expected_mm' in master_df.columns:
        master_df['precip_expected_mm'] = master_df['precip_expected_mm'].fillna(0)
        # Forward-looking rolling sums (shift by -N to look N steps ahead)
        master_df['qpf_next_6h'] = master_df['precip_expected_mm'].rolling(window=6, min_periods=1).sum().shift(-6).fillna(0)
        master_df['qpf_next_12h'] = master_df['precip_expected_mm'].rolling(window=12, min_periods=1).sum().shift(-12).fillna(0)
        master_df['qpf_next_24h'] = master_df['precip_expected_mm'].rolling(window=24, min_periods=1).sum().shift(-24).fillna(0)
        print("✅ NWS QPF forward features computed (qpf_next_6h, 12h, 24h).")
    else:
        master_df['qpf_next_6h'] = 0
        master_df['qpf_next_12h'] = 0
        master_df['qpf_next_24h'] = 0
        print("⚠️  precip_expected_mm not in master_df — QPF features set to 0.")

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
        if paired_data is not None:
            master_df['hwy_59_flow_est'] = master_df['hwy_59_height'].apply(lambda x: apply_hwy59_interpolation(x, paired_data))
        else:
            master_df['hwy_59_flow_est'] = master_df['hwy_59_height'].apply(lambda x: apply_sskp_rating(x, meta))

    # 🆕 PHASE 2 (B): Create Hwy 59 lags for Watts Bridge model (must happen after hwy_59_flow_est is calculated)
    for col in ['hwy_59_height', 'hwy_59_flow_est']:
        if col in master_df.columns:
            master_df[f'{col}_3h_ago'] = master_df[col].shift(3)
            master_df[f'{col}_6h_ago'] = master_df[col].shift(6)
            master_df[f'{col}_trend_6h'] = master_df[col].diff(6).fillna(0)

    master_df = master_df.dropna(subset=['hwy_59_height', 'watts_ok_height'], how='all')
    master_df.to_csv('master_training_data.csv')

    print(f"🚀 Success! Master dataset saved with {len(master_df)} rows.")
    return master_df


if __name__ == '__main__':
    merge_datasets()
