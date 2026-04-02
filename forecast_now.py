import pandas as pd
import joblib
from datetime import datetime, timedelta
import pytz

def generate_multi_forecast():
    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    
    # Timezone Fix
    current_row = df.tail(1).copy()
    utc_time = pd.to_datetime(current_row.index[0], utc=True)
    local_tz = pytz.timezone('US/Central')
    local_time = utc_time.astimezone(local_tz)
    forecast_time = local_time + timedelta(hours=6)
    
    features = ['savoy_height', 'osage_creek_flow', 'hwy_59_height', 
                'precip_expected_mm', 'soil_saturation_index', 
                'savoy_trend', 'seasonal_cycle', 'lake_headroom']

    print(f"\n{'='*55}")
    print(f"   ILLINOIS RIVER SYSTEM REPORT")
    print(f"   Local Time: {local_time.strftime('%I:%M %p')}")
    print(f"{'='*55}")

    # Locations map - make sure the 'key' matches the model filename exactly!
    locations = {
        'hwy_16_flow': ('Hwy 16 (Siloam)', 'CFS'),
        'hwy_59_flow_est': ('Hwy 59 (AR Bridge)', 'EST. CFS'), # Updated key
        'lake_francis_height': ('Lake Francis Level', 'ft (MSL)'),
        'watts_ok_height': ('Watts Bridge (OK)', 'ft')
    }

    for key, (label, unit) in locations.items():
        try:
            model = joblib.load(f'model_{key}.pkl')
            pred = model.predict(current_row[features])[0]
            current_val = current_row[key].values[0]
            
            # Formatting the Sea Level height so it's not confusing
            if 'lake' in key and current_val > 800:
                display_val = f"{current_val:.2f}"
            else:
                display_val = f"{current_val:.2f}" if not pd.isna(current_val) else "OFFLINE"
            
            print(f"{label}:")
            print(f"  Current:   {display_val:>8} {unit}")
            print(f"  Projected: {pred:>8.2f} {unit} (at {forecast_time.strftime('%I:%M %p')})")
            print("-" * 55)
        except:
            print(f"  [!] Model for {label} needs more training data...")

if __name__ == "__main__":
    generate_multi_forecast()

import pandas as pd
from datetime import datetime

def generate_forecast_report():
    # Load the latest master data to get current readings
    df = pd.read_csv('master_training_data.csv')
    latest_row = df.iloc[-1]
    
    # Define the current value that was causing the error
    current_59 = latest_row['hwy_59_height'] 
    
    # Mocking predictions for this example - ensure your model fills these
    predictions = {
        'savoy': 2.45,
        'hwy_59': 3.10, 
        'lake_francis': 2.15,
        'watts': 3.40
    }

    with open('forecast_report.txt', 'w') as f:
        f.write("--- 6-HOUR AI FORECAST ---\n")
        f.write(f"Updated: {datetime.now().strftime('%m/%d %H:%M')} CST\n")
        f.write("--------------------------\n")
        
        for loc, val in predictions.items():
            f.write(f"{loc.upper()}: {val:.2f} FT\n")
        
        # Now current_59 is defined and will work:
        if predictions['hwy_59'] > (current_59 + 0.05):
            f.write("\nTREND: 📈 RISING")
        elif predictions['hwy_59'] < (current_59 - 0.05):
            f.write("\nTREND: 📉 FALLING")
        else:
            f.write("\nTREND: ↔️ STABLE")

if __name__ == "__main__":
    generate_forecast_report()