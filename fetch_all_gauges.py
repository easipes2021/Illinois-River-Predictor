import dataretrieval.nwis as nwis
import pandas as pd

STATIONS = {
    'savoy': '07194800',
    'osage_creek': '07195000',
    'hwy_16': '07195400',
    'hwy_59': '07195430',
    'lake_francis': '07195450',
    'watts_ok': '07195500'
}

def fetch_river_network(days=7):
    all_data = []
    
    for name, site_id in STATIONS.items():
        print(f"Fetching {name} ({site_id})...")
        try:
            # We fetch all available data for the site
            df, _ = nwis.get_iv(sites=site_id, period=f'P{days}D')
            
            if df.empty:
                print(f"  ! No data found for {name} in the last {days} days.")
                continue

            # Standard USGS Codes: 00060 (Discharge), 00065 (Gage Height)
            # We map them to friendly names only if they exist in the data
            rename_dict = {}
            if '00060' in df.columns: rename_dict['00060'] = f'{name}_flow'
            if '00065' in df.columns: rename_dict['00065'] = f'{name}_height'
            
            df = df.rename(columns=rename_dict)
            
            # Keep only our renamed columns
            keep_cols = [c for c in df.columns if c in rename_dict.values()]
            all_data.append(df[keep_cols])
            print(f"  ✓ Success: Found {list(rename_dict.values())}")

        except Exception as e:
            print(f"  × Error fetching {name}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Join everything together by time
    final_df = pd.concat(all_data, axis=1)
    return final_df

if __name__ == "__main__":
    river_data = fetch_river_network(days=5)
    if not river_data.empty:
        river_data.to_csv('illinois_river_network.csv')
        print("\nSUCCESS: Data saved to illinois_river_network.csv")
    else:
        print("\nFAILURE: No data was collected.")