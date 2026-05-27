import pandas as pd
import os

def merge_beta_data():
    if not os.path.exists('master_training_data.csv'):
        print("❌ Error: master_training_data.csv missing.")
        return
        
    if not os.path.exists('huc12_training_data_beta.csv'):
        print("❌ Error: huc12_training_data_beta.csv missing.")
        return

    # Load gauge and time data (ignore the old precip columns)
    master_df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    
    # Columns to keep from master (gauges and their lags/trends + time features + saturation)
    # Exclude all 'precip_' except the saturation and qpf
    drop_cols = [c for c in master_df.columns if ('precip_' in c and 'saturation' not in c and 'qpf' not in c)]
    # Also drop hwy_59_flow if it exists since we use hwy_59_flow_est
    if 'hwy_59_flow' in master_df.columns:
        drop_cols.append('hwy_59_flow')
    master_df = master_df.drop(columns=drop_cols, errors='ignore')

    # Load new HUC12 data
    huc12_df = pd.read_csv('huc12_training_data_beta.csv', index_col=0, parse_dates=True)
    
    # Merge them. Since both are hourly and timezone-aware, we use an inner join to ensure timestamps match exactly
    # master_training_data is naive in UTC, huc12 is timezone aware UTC. Let's align them.
    master_df.index = pd.to_datetime(master_df.index, utc=True)
    huc12_df.index = pd.to_datetime(huc12_df.index, utc=True)
    
    merged_df = master_df.join(huc12_df, how='inner')
    
    # Aggregate into the 4 Routing Zones
    # Zone 1: Upper Mainstem (Prefixes 01, 02, 04)
    upper_cols = [c for c in merged_df.columns if c.startswith(('precip_1111010301', 'precip_1111010302', 'precip_1111010304'))]
    merged_df['precip_upper_zone'] = merged_df[upper_cols].mean(axis=1)
    
    # Zone 2: Osage Creek (Prefix 03)
    osage_cols = [c for c in merged_df.columns if c.startswith('precip_1111010303')]
    merged_df['precip_osage_zone'] = merged_df[osage_cols].mean(axis=1)
    
    # Zone 3: Flint Creek (Prefix 05)
    flint_cols = [c for c in merged_df.columns if c.startswith('precip_1111010305')]
    merged_df['precip_flint_zone'] = merged_df[flint_cols].mean(axis=1)
    
    # Zone 4: Lower Mainstem (Prefix 06)
    lower_cols = [c for c in merged_df.columns if c.startswith('precip_1111010306')]
    merged_df['precip_lower_zone'] = merged_df[lower_cols].mean(axis=1)

    # Drop the individual 26 creek columns to prevent the "curse of dimensionality"
    all_huc_cols = [c for c in merged_df.columns if c.startswith('precip_11110103')]
    merged_df = merged_df.drop(columns=all_huc_cols)

    # Generate Time-Lagged Features for the 4 zones
    zones = ['precip_upper_zone', 'precip_osage_zone', 'precip_flint_zone', 'precip_lower_zone']
    for zone in zones:
        merged_df[f'{zone}_3h_ago'] = merged_df[zone].shift(3)
        merged_df[f'{zone}_6h_ago'] = merged_df[zone].shift(6)
        merged_df[f'{zone}_12h_ago'] = merged_df[zone].shift(12)
        merged_df[f'{zone}_24h_sum'] = merged_df[zone].rolling(window=24, min_periods=1).sum()
        merged_df[f'{zone}_48h_sum'] = merged_df[zone].rolling(window=48, min_periods=1).sum()

    # Drop NaNs created by lagging
    merged_df = merged_df.dropna()

    merged_df.to_csv('master_training_data_beta.csv')
    print(f"✅ Beta master dataset created: {len(merged_df)} rows.")
    print("Features included:", merged_df.columns.tolist())

if __name__ == '__main__':
    merge_beta_data()
