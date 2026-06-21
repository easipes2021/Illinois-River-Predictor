import pandas as pd
import os

def merge_beta_data():
    if not os.path.exists('master_training_data.csv'):
        print("❌ Error: master_training_data.csv missing.")
        return
        
    if not os.path.exists('huc12_training_data_beta.csv'):
        print("❌ Error: huc12_training_data_beta.csv missing.")
        return

    # Load gauge and time data (already at 15-min resolution from merge_data.py)
    master_df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    
    drop_cols = [c for c in master_df.columns if ('precip_' in c and 'saturation' not in c and 'qpf' not in c)]
    if 'hwy_59_flow' in master_df.columns:
        drop_cols.append('hwy_59_flow')
    master_df = master_df.drop(columns=drop_cols, errors='ignore')

    # Load new HUC12 data (already at 15-min resolution from fetch_huc12_precip.py)
    huc12_df = pd.read_csv('huc12_training_data_beta.csv', index_col=0, parse_dates=True)
    
    master_df.index = pd.to_datetime(master_df.index, utc=True)
    huc12_df.index = pd.to_datetime(huc12_df.index, utc=True)
    
    merged_df = master_df.join(huc12_df, how='inner')
    
    # 🆕 SPATIAL RESOLUTION UPGRADE: Keep all 26 HUC-12 grids
    # We no longer average them into 4 zones!
    
    zones = [c for c in merged_df.columns if c.startswith('precip_11110103')]
    
    # 15-minute intervals: 1 hour = 4 steps
    for zone in zones:
        # Lags
        merged_df[f'{zone}_3h_ago'] = merged_df[zone].shift(12)
        merged_df[f'{zone}_6h_ago'] = merged_df[zone].shift(24)
        merged_df[f'{zone}_12h_ago'] = merged_df[zone].shift(48)
        
        # Short-term intensity
        merged_df[f'{zone}_3h_sum'] = merged_df[zone].rolling(window=12, min_periods=1).sum()
        merged_df[f'{zone}_6h_sum'] = merged_df[zone].rolling(window=24, min_periods=1).sum()
        
        # Standard sums
        merged_df[f'{zone}_24h_sum'] = merged_df[zone].rolling(window=96, min_periods=1).sum()
        merged_df[f'{zone}_48h_sum'] = merged_df[zone].rolling(window=192, min_periods=1).sum()
        
        # Long-term saturation (Antecedent Moisture)
        merged_df[f'{zone}_168h_sum'] = merged_df[zone].rolling(window=672, min_periods=1).sum()
        merged_df[f'{zone}_720h_sum'] = merged_df[zone].rolling(window=2880, min_periods=1).sum()
        
        # Non-linear interaction: Intensity * Saturation
        merged_df[f'{zone}_runoff_risk'] = merged_df[f'{zone}_6h_sum'] * merged_df[f'{zone}_168h_sum']

    # Drop NaNs created by lagging
    merged_df = merged_df.dropna()

    merged_df.to_csv('master_training_data_beta.csv')
    print(f"✅ Beta master dataset created: {len(merged_df)} rows at 15-min resolution.")
    print(f"Features included: {len(merged_df.columns)} columns")

if __name__ == '__main__':
    merge_beta_data()
