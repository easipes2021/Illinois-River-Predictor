# Illinois River Predictor

An AI-powered flow prediction system for the Illinois River that uses machine learning to forecast water levels and discharge rates 6 hours into the future. The system ingests real-time USGS gauge data, weather forecasts, and regional precipitation data to make accurate predictions at multiple monitoring stations.

## 📊 Features

- **Real-time Gauge Monitoring**: Fetches live water level and flow data from 6 USGS monitoring stations
- **Weather Integration**: Incorporates current weather forecasts and regional precipitation patterns
- **Machine Learning Forecasts**: Uses Random Forest models to predict stream flow 6 hours ahead
- **Multi-location Tracking**: Monitors four key prediction points across the Illinois River network:
  - Hwy 16 (Siloam Springs)
  - Hwy 59 (SSKP rating curve)
  - Lake Francis (height forecast)
  - Watts Bridge, OK (flow forecast)
- **Interactive Dashboard**: Web-based UI displaying current conditions and 6-hour forecasts
- **Rating Curves**: Applies piecewise power-law rating relationships to convert stage (height) to discharge

## 🏗️ System Architecture

### Data Pipeline
```
Fetch USGS Gauges → Fetch Weather → Fetch Regional Precip
        ↓                 ↓                      ↓
    [river_df]     [weather_df]         [regional_df]
        └────────────────┬───────────────────┘
                         ↓
                   Merge Data (merge_data.py)
                         ↓
              [master_training_data.csv]
                         ↓
              Train Models (predict_all.py)
                         ↓
           [model_hwy_16_flow.pkl, etc.]
                         ↓
           Generate Forecast (forecast_now.py)
                         ↓
              [forecasts.json] → [index.html]
```

### Core Components

| File | Purpose |
|------|---------|
| `fetch_all_gauges.py` | Fetches USGS real-time gauge data (levels & discharge) |
| `fetch_weather.py` | Retrieves weather forecasts |
| `fetch_nwa_precip.py` | Gets regional precipitation from NWA network |
| `merge_data.py` | Combines all data sources, creates lagged features, applies rating curves |
| `predict_all.py` | Trains Random Forest models for each prediction location |
| `forecast_now.py` | Generates 6-hour forecasts using trained models |
| `build_rating_curve.py` | Develops piecewise rating relationships (stage → discharge) |
| `index.html` | Interactive web dashboard for viewing forecasts |
| `update_model.sh` | Orchestration script running the full pipeline |

## 🌊 Monitoring Stations

The system tracks data from these USGS monitoring locations:

- **Savoy (07194800)**: Upper watershed, height & flow monitoring
- **Osage Creek (07195000)**: Tributary flow measurement
- **Hwy 16 (07195400)**: Stage height monitoring
- **Hwy 59 (07195430)**: Key reference point with detailed rating curve
- **Lake Francis (07195495)**: Reservoir level tracking
- **Watts Bridge, OK (07195500)**: Lower boundary station

## 🤖 Machine Learning Model

### Model Type
Random Forest Regressor with 150 trees
- **Input Features**: 13 engineered features (current gauges + lagged trends + precipitation + seasonality)
- **Output**: 4 parallel predictions (flow/height at different locations)
- **Forecast Horizon**: 6 hours ahead

### Key Features
The models use:
- **Current Observations**: Savoy height, Osage Creek flow, Hwy 59 height
- **Lagged Trends**: 3-hour and 6-hour historical values (detect rising/falling patterns)
- **Regional Precipitation**: Fayetteville, Springdale, Bentonville, Siloam Springs rainfall
- **Saturation Index**: Tracks soil moisture effects on runoff
- **Seasonal Cycle**: Month-based seasonal patterns
- **Lake Headroom**: Available storage in Lake Francis

## 📋 Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

### Setup
```bash
# Clone repository
git clone https://github.com/easipes2021/Illinois-River-Predictor.git
cd Illinois-River-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Full Pipeline (Recommended)
Run the orchestration script to execute all steps:
```bash
./update_model.sh
```

This:
1. Fetches current river levels from USGS
2. Fetches regional precipitation data
3. Fetches weather forecasts
4. Merges all data with lagged features
5. Retrains AI models
6. Generates 6-hour forecasts
7. Updates the web dashboard

### Individual Components

**Fetch current gauge data:**
```bash
python fetch_all_gauges.py
```

**Retrain models with latest data:**
```bash
python predict_all.py
```

**Generate new forecast:**
```bash
python forecast_now.py
```

**Build/update rating curve (one-time setup):**
```bash
python build_rating_curve.py
```

**View the dashboard:**
Open `index.html` in a web browser to see the interactive dashboard with current predictions.

## 📊 Data Files

### Inputs
- `illinois_river_network.csv`: Real-time USGS gauge readings
- `weather_forecast.csv`: Meteorological forecast data
- `regional_precip_actual.csv`: Historical & current precipitation

### Training Data
- `master_training_data.csv`: Merged dataset with all features used for model training

### Outputs
- `model_hwy_16_flow.pkl`: Trained model for Hwy 16 flow prediction
- `model_hwy_59_flow_est.pkl`: Trained model for Hwy 59 flow estimation
- `model_lake_francis_height.pkl`: Trained model for Lake Francis level
- `model_watts_ok_flow.pkl`: Trained model for Watts Bridge flow
- `forecasts.json`: Latest 6-hour forecast in JSON format
- `rating_curve_metadata.json`: Piecewise rating curve parameters

## 🎯 Forecast Interpretation

The system generates forecasts for 4 prediction locations:

```json
{
  "hwy_16_flow": {
    "current": 1250.5,
    "forecast_6h": 1380.2,
    "trend": "↑ Rising",
    "unit": "CFS"
  },
  ...
}
```

- **Current**: Latest observed value
- **Forecast_6h**: Predicted value 6 hours from now
- **Trend**: Direction of movement (rising/falling/stable)
- **Unit**: CFS (cubic feet per second) for flow, feet for height

## 📈 Model Performance

The Random Forest approach was chosen for:
- Ability to capture non-linear relationships between rainfall and streamflow
- Robustness to missing data and outliers
- Fast inference for real-time predictions
- Feature importance interpretability

Models are retrained with each update cycle to adapt to seasonal patterns and long-term trends.

## 🔧 Advanced Configuration

### Lagged Features
Edit `merge_data.py` to adjust which previous time steps are included as features (currently 3h and 6h lag).

### Model Hyperparameters
In `predict_all.py`, adjust:
- `n_estimators`: Number of trees (default: 150)
- `min_samples_leaf`: Minimum samples per leaf node
- `max_features`: Features per split

### Rating Curve Fitting
`build_rating_curve.py` uses piecewise power-law regression. Adjust:
- Breakpoint selection algorithm
- Low-flow vs high-flow segment fitting

## 📝 Data Notes

- Data is fetched in UTC and converted to US/Central time for display
- Missing gauge readings are forward-filled then back-filled
- Forecasts require at least 10 clean data points per model
- Rating curve derived from historical paired stage-flow USGS measurements

## 🐛 Troubleshooting

**"master_training_data.csv not found"**
- Run `merge_data.py` first to combine all data sources

**"Not enough data for model training"**
- Ensure at least 7 days of historical USGS data is available
- Check network connectivity to USGS servers

**Stale forecasts**
- Run `forecast_now.py` again to refresh predictions
- Verify `master_training_data.csv` is up-to-date by running `merge_data.py`

## 📄 License

This project is maintained for water resource monitoring and forecasting in the Illinois River watershed.

## 👤 Author

Created for watershed management and flood forecasting applications.


