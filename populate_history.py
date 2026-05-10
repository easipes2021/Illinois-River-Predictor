import json
from datetime import datetime, timedelta
import pytz
import pandas as pd
import dataretrieval.nwis as nwis
import socket

# 🆕 Set socket timeout to prevent indefinite hangs
socket.setdefaulttimeout(15)

# Define gauge mappings: key -> (site_id, parameter_code, name, unit)
GAUGE_CONFIG = {
    'hwy_16_flow': ('07195400', '00060', 'Hwy 16 (Siloam)', 'CFS'),
    'hwy_59_flow_est': ('07195430', '00060', 'Hwy 59 (AR Bridge)', 'EST. CFS'),
    'lake_francis_height': ('07195495', '00065', 'Lake Francis Level', 'ft (MSL)'),
    'watts_ok_flow': ('07195500', '00060', 'Watts Bridge (OK)', 'CFS')
}

def fetch_historical_gauge(site_id, param_code, days=30):
    """Fetch historical data for a gauge."""
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days)
    
    try:
        # 🆕 Use get_iv instead of get_record for instantaneous values (faster, more reliable)
        df, meta = nwis.get_iv(
            sites=site_id, 
            start=start_date, 
            end=end_date,
            parameterCd=param_code
        )
        
        if df.empty:
            print(f"  ⚠️ No data found for site {site_id}")
            return []
        
        # 🆕 NWIS returns columns like 'USGS.site_id.parameterCd.statCode'
        # Get the first column that matches our parameter code
        matching_cols = [col for col in df.columns if param_code in col]
        
        if not matching_cols:
            print(f"  ⚠️ Parameter code {param_code} not found in data for {site_id}")
            return []
        
        data_col = matching_cols[0]  # Use first matching column
        
        # Convert to list of dicts
        history = []
        for idx, row in df.iterrows():
            value = row[data_col]
            if pd.notna(value):
                try:
                    history.append({
                        'timestamp': pd.to_datetime(idx, utc=True).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'value': float(value)
                    })
                except (ValueError, TypeError):
                    print(f"  ⚠️ Skipping invalid value: {value}")
                    continue
        
        print(f"  ✅ Success: {len(history)} data points for {site_id}")
        return history
    
    except socket.timeout:
        print(f"  ❌ Timeout: Request took too long for site {site_id}")
        return []
    except Exception as e:
        print(f"  ❌ Error fetching data for site {site_id}: {type(e).__name__}: {e}")
        return []

def main():
    print("Fetching historical gauge data (last 30 days)...\n")
    
    local_tz = pytz.timezone('US/Central')
    local_time = datetime.now(local_tz)
    
    history_results = {
        'timestamp': local_time.strftime('%Y-%m-%d %I:%M %p'),
        'forecast_time': (local_time + timedelta(hours=6)).strftime('%Y-%m-%d %I:%M %p'),
        'gauges': {}
    }
    
    success_count = 0
    total_count = len(GAUGE_CONFIG)
    
    for key, (site_id, param_code, name, unit) in GAUGE_CONFIG.items():
        print(f"Fetching historical data for {name} ({site_id})...")
        history = fetch_historical_gauge(site_id, param_code, days=30)
        
        history_results['gauges'][key] = {
            'name': name,
            'unit': unit,
            'history': history
        }
        
        if history:
            success_count += 1
    
    # Save results
    try:
        with open('history.json', 'w') as f:
            json.dump(history_results, f, indent=4)
        print(f"\n✅ history.json saved ({success_count}/{total_count} gauges successful)")
    except Exception as e:
        print(f"\n❌ Error saving history.json: {e}")

if __name__ == "__main__":
    main()