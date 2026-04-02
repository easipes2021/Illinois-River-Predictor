#!/bin/zsh

echo "--- STEP 1: Fetching Current River Levels ---"
python3 fetch_all_gauges.py

echo "--- STEP 1.5: Fetching Regional Precip (NWA Network) ---"
python3 fetch_nwa_precip.py

echo "--- STEP 2: Fetching Weather Forecast ---"
python3 fetch_weather.py

echo "--- STEP 3: Merging & Applying SSKP Rating Table ---"
python3 merge_data.py

echo "--- STEP 4: Training & Refining AI Models ---"
python3 predict_all.py

echo "--- STEP 5: Generating System Dashboard ---"
python3 forecast_now.py